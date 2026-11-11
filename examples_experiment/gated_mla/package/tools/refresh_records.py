"""Refresh reviewed nonproduction changes; refuse production/reference drift."""

import argparse
import hashlib
import json
from pathlib import Path

from audit_package import checked_path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def refresh(root):
    manifest_path = root / "source_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    registered = set()
    for entry in manifest["selected_sources"]:
        name = entry["destination"]
        registered.add(name)
        if digest(checked_path(root, name)) != entry["destination_sha256"]:
            raise ValueError(f"Retained production/reference source changed; explicit provenance review required: {name}")
    changes = []
    for entry in manifest["new_package_files"]:
        name = entry["path"]
        registered.add(name)
        actual = digest(checked_path(root, name))
        if name.startswith("gmla/") and actual != entry["sha256"]:
            raise ValueError(f"Production source changed; this tool must not bless it: {name}")
        if actual != entry["sha256"]:
            changes.append({"path": name, "previous_sha256": entry["sha256"], "sha256": actual})
            entry["sha256"] = actual
    for path in sorted((root / "gmla").rglob("*.py")):
        if path.relative_to(root).as_posix() not in registered:
            raise ValueError(f"Unregistered production Python source: {path}")
    for folder in ("tests", "tools"):
        for path in sorted((root / folder).rglob("*.py")):
            name = path.relative_to(root).as_posix()
            if name not in registered:
                manifest["new_package_files"].append({"path": name, "sha256": digest(path)})
                changes.append({"path": name, "previous_sha256": None, "sha256": digest(path)})
    if changes:
        manifest.setdefault("reviewed_nonproduction_updates", []).append(changes)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    ignored = {"__pycache__", ".pytest_cache", ".ruff_cache", "build", "dist", ".git"}
    files = [
        path
        for path in root.rglob("*")
        if path.is_file()
        and path.name != "FILELIST.sha256"
        and not any(part in ignored or part.endswith(".egg-info") for part in path.relative_to(root).parts)
    ]
    lines = [f"{digest(path)}  {path.relative_to(root).as_posix()}" for path in sorted(files)]
    (root / "FILELIST.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"updated_nonproduction_files": changes, "filelist_entries": len(files), "production_changes_permitted": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    print(json.dumps(refresh(args.package_root.resolve()), indent=2))


if __name__ == "__main__":
    main()
