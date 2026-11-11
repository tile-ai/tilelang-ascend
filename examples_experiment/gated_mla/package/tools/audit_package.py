"""Verify checkout file/AST integrity without opening historical source paths."""

import argparse
import ast
import hashlib
import json
from pathlib import Path
from pathlib import PurePosixPath


def checked_path(root, relative):
    parts = PurePosixPath(relative)
    if parts.is_absolute() or ".." in parts.parts or "\\" in relative or ":" in relative:
        raise AssertionError(f"Expected a portable package-relative path: {relative}")
    path = root / Path(*parts.parts)
    if not path.resolve().is_relative_to(root.resolve()):
        raise AssertionError(f"Path escapes package: {relative}")
    return path


def audit_filelist(root):
    listed = set()
    for line in (root / "FILELIST.sha256").read_text(encoding="utf-8").splitlines():
        digest, separator, relative = line.partition("  ")
        if not separator or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise AssertionError("Malformed FILELIST.sha256 entry")
        if relative in listed or relative == "FILELIST.sha256":
            raise AssertionError(f"Duplicate/self-referential FILELIST entry: {relative}")
        path = checked_path(root, relative)
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise AssertionError(f"FILELIST digest mismatch: {relative}")
        listed.add(relative)
    required = {"README.md", "pyproject.toml", "source_manifest.json"}
    required.update(path.relative_to(root).as_posix() for folder in ("gmla", "tests", "tools") for path in (root / folder).rglob("*.py"))
    if missing := required - listed:
        raise AssertionError(f"Source files absent from FILELIST: {sorted(missing)}")
    return len(listed)


def audit(root):
    filelist_count = audit_filelist(root)
    manifest = json.loads((root / "source_manifest.json").read_text(encoding="utf-8"))
    definitions = 0
    for entry in manifest["selected_sources"]:
        path = checked_path(root, entry["destination"])
        if hashlib.sha256(path.read_bytes()).hexdigest() != entry["destination_sha256"]:
            raise AssertionError(f"Packaged source changed: {path}")
        nodes = {n.name: n for n in ast.parse(path.read_text(encoding="utf-8")).body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
        for name, digest in entry["definition_ast_sha256"].items():
            if hashlib.sha256(ast.dump(nodes[name], include_attributes=False).encode()).hexdigest() != digest:
                raise AssertionError(f"Retained definition changed: {path}:{name}")
            definitions += 1
    for entry in manifest["new_package_files"]:
        if hashlib.sha256(checked_path(root, entry["path"]).read_bytes()).hexdigest() != entry["sha256"]:
            raise AssertionError(f"Package adapter/test changed: {entry['path']}")
    for path in root.rglob("*.py"):
        compile(path.read_text(encoding="utf-8"), str(path), "exec")
    return {
        "status": "source_and_AST_verified",
        "selected_source_records": len(manifest["selected_sources"]),
        "retained_definitions": definitions,
        "new_files": len(manifest["new_package_files"]),
        "selection": manifest["selection"],
        "NPU_run": False,
        "filelist_entries": filelist_count,
        "historical_origin_files_opened": False,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = audit(args.package_root)
    if args.output:
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
