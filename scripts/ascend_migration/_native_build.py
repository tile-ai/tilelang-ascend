"""Record and verify the source and library identity of an explicit local build."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
SOURCE_PATHS = ("CMakeLists.txt", "config.cmake", "src", "include", "cmake", "3rdparty/patches")
TVM_PATHS = ("CMakeLists.txt", "config.cmake", "src", "include", "cmake", "3rdparty", "python")
TVM_COMMIT = "c2921fdaf795b1103d21abc962e83a209c7258d7"
TVM_PATCHES = (
    "3rdparty/patches/tvm_backtrace_symbol_name_length.patch",
    "3rdparty/patches/tvm_slice_step_fix.patch",
)
LIBRARY_PATHS = (
    "build/libtilelang_module.so",
    "build/tvm/libtvm.so",
    "build/tvm/libtvm_runtime.so",
)


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tvm_provenance(tvm_source: Path) -> dict[str, Any]:
    try:
        commit = subprocess.check_output(["git", "-C", str(tvm_source), "rev-parse", "HEAD"], text=True).strip()
        for patch in TVM_PATCHES:
            subprocess.run(
                ["git", "-C", str(tvm_source), "apply", "--reverse", "--check", str(ROOT / patch)],
                check=True,
                capture_output=True,
                text=True,
            )
    except subprocess.CalledProcessError as error:
        raise ValueError("Actual TVM source does not have the recorded base and patches") from error
    if commit != TVM_COMMIT:
        raise ValueError(f"TVM reference checkout changed: {commit}")
    return {
        "reference_checkout": str(tvm_source),
        "reference_commit": commit,
        "applied_patches": [{"path": name, "sha256": file_digest(ROOT / name)} for name in TVM_PATCHES],
    }


def source_identity(root: Path, paths: tuple[str, ...]) -> dict[str, Any]:
    """Hash source contents and logical names, including linked dependency headers."""
    files: list[Path] = []
    for name in paths:
        start = root / name
        if start.is_file():
            files.append(start)
        elif start.is_dir():
            visited: set[Path] = set()
            for current, directories, names in os.walk(start, followlinks=True):
                resolved = Path(current).resolve()
                if resolved in visited:
                    directories[:] = []
                    continue
                visited.add(resolved)
                directories[:] = sorted(item for item in directories if item not in {".git", "__pycache__"})
                files.extend(Path(current) / item for item in names if item != ".git")
    if not files:
        raise ValueError(f"No source inputs found under {root}")
    digest = hashlib.sha256()
    for path in sorted(files):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(b"\0")
        digest.update(file_digest(path).encode())
        digest.update(b"\n")
    return {"algorithm": "sha256", "digest": digest.hexdigest(), "files": len(files)}


def verify_cmake_source(cache: Path, tvm_source: Path, *, build_source_root: Path | None = None) -> None:
    entries = {}
    for line in cache.read_text().splitlines():
        if not line.startswith(("#", "//")) and ":" in line and "=" in line:
            name, value = line.split("=", 1)
            entries[name.split(":", 1)[0]] = value
    project = ROOT if build_source_root is None else build_source_root
    for key, expected in (("TVM_SOURCE_DIR", tvm_source), ("CMAKE_HOME_DIRECTORY", project)):
        if key not in entries or Path(entries[key]).resolve() != expected.resolve():
            raise ValueError(f"CMakeCache {key} does not identify the recorded build source")


def write_manifest(
    output: str | Path,
    *,
    tvm_source: str | Path,
    build_command: list[str],
    build_log: str | Path,
) -> dict[str, Any]:
    """Call only after the recorded build command has completed successfully."""
    tvm_source = Path(tvm_source).resolve()
    build_log = Path(build_log).resolve()
    libraries = []
    for name in LIBRARY_PATHS:
        path = (ROOT / name).resolve()
        libraries.append({"path": str(path), "sha256": file_digest(path)})
    cache = ROOT / "build/CMakeCache.txt"
    verify_cmake_source(cache, tvm_source)
    report = {
        "schema_version": 1,
        "repo_root": str(ROOT),
        "project_source": source_identity(ROOT, SOURCE_PATHS),
        "tvm_source_root": str(tvm_source),
        "tvm_source": source_identity(tvm_source, TVM_PATHS),
        "tvm_provenance": tvm_provenance(tvm_source),
        "cmake_cache": {"path": str(cache), "sha256": file_digest(cache)},
        "libraries": libraries,
        "build": {
            "command": build_command,
            "log": str(build_log),
            "log_sha256": file_digest(build_log),
        },
    }
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + "\n")
    return report


def write_relocated_manifest(original_manifest: str | Path, snapshot_root: str | Path, output: str | Path) -> dict[str, Any]:
    """Snapshot an already verified build, preserving its original build record.

    The caller prepares a source checkout with identical relative source
    contents. This copies artifacts and evidence; it does not claim a build
    happened in the destination or rewrite the original CMakeCache.
    """
    original_path = Path(original_manifest).resolve()
    original = verify_manifest(original_path)
    if original["schema_version"] != 1:
        raise ValueError("Relocation requires an original, non-relocated build manifest")
    snapshot = Path(snapshot_root).resolve()
    if snapshot == ROOT.resolve():
        raise ValueError("Build snapshot must be separate from the build source checkout")
    if source_identity(snapshot, SOURCE_PATHS) != original["project_source"]:
        raise ValueError("Snapshot source contents differ from the verified original build")

    def copy(source: str | Path, relative: str) -> Path:
        destination = (snapshot / relative).resolve()
        if not destination.is_relative_to(snapshot):
            raise ValueError("Snapshot artifacts must not link back to mutable build files")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        return destination

    origin_copy = copy(original_path, "build/ascend_migration_original_manifest.json")
    cache = copy(original["cmake_cache"]["path"], "build/CMakeCache.txt")
    log = copy(original["build"]["log"], "build/ascend_migration_original_build.log")
    by_name = {Path(item["path"]).name: item for item in original["libraries"]}
    libraries = []
    for relative in LIBRARY_PATHS:
        item = by_name[Path(relative).name]
        destination = copy(item["path"], relative)
        digest = file_digest(destination)
        if digest != item["sha256"]:
            raise ValueError("Copied snapshot library does not match the original build")
        libraries.append({"path": str(destination), "sha256": digest})
    report = {key: value for key, value in original.items() if key != "path"}
    report.update(
        schema_version=2,
        artifact_kind="relocated_verified_build",
        repo_root=str(snapshot),
        libraries=libraries,
        cmake_cache={"path": str(cache), "sha256": file_digest(cache)},
        build={**original["build"], "log": str(log)},
        relocated_from={
            "manifest": str(origin_copy),
            "manifest_sha256": file_digest(origin_copy),
            "build_source_root": original["repo_root"],
        },
    )
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + "\n")
    return report


def _relocated_build_source(report: dict[str, Any]) -> Path:
    if report.get("artifact_kind") != "relocated_verified_build":
        raise ValueError("Relocated manifest must identify an existing verified build")
    relocation = report.get("relocated_from")
    if not isinstance(relocation, dict):
        raise ValueError("Relocated manifest is missing its original build evidence")
    origin_path = Path(relocation["manifest"])
    if not origin_path.resolve().is_relative_to(ROOT.resolve()):
        raise ValueError("Original build evidence must be an immutable snapshot copy")
    if file_digest(origin_path) != relocation["manifest_sha256"]:
        raise ValueError("Original build manifest copy changed")
    origin = json.loads(origin_path.read_text())
    build_source = Path(relocation["build_source_root"])
    if not build_source.is_absolute() or origin.get("schema_version") != 1 or origin.get("repo_root") != str(build_source):
        raise ValueError("Relocated manifest does not identify the original build checkout")
    for field in ("project_source", "tvm_source_root", "tvm_source", "tvm_provenance"):
        if report[field] != origin[field]:
            raise ValueError(f"Relocated build {field} differs from the original evidence")
    for field in ("cmake_cache", "build"):
        copied = Path(report[field]["path" if field == "cmake_cache" else "log"]).resolve()
        if not copied.is_relative_to(ROOT.resolve()):
            raise ValueError("Relocated build evidence must not reference mutable build files")
    if (
        report["cmake_cache"]["sha256"] != origin["cmake_cache"]["sha256"]
        or report["build"]["command"] != origin["build"]["command"]
        or report["build"]["log_sha256"] != origin["build"]["log_sha256"]
    ):
        raise ValueError("Relocated build cache, command or log provenance changed")
    original_libraries = {Path(item["path"]).name: item["sha256"] for item in origin["libraries"]}
    relocated_libraries = {}
    for item in report["libraries"]:
        path = Path(item["path"]).resolve()
        if not path.is_relative_to(ROOT.resolve()):
            raise ValueError("Relocated libraries must be independent snapshot copies")
        relocated_libraries[path.name] = item["sha256"]
    if relocated_libraries != original_libraries:
        raise ValueError("Relocated libraries differ from the original build evidence")
    return build_source


def verify_manifest(path: str | Path) -> dict[str, Any]:
    """Reject stale source, library, configure, or build-log identities."""
    path = Path(path).resolve()
    report = json.loads(path.read_text())
    if not isinstance(report, dict):
        raise ValueError("Native build manifest must be a JSON object")
    schema = report.get("schema_version")
    if type(schema) is not int or schema not in (1, 2) or report.get("repo_root") != str(ROOT):
        raise ValueError("Native build manifest schema or repository does not match")
    build_source_root = ROOT if schema == 1 else _relocated_build_source(report)
    if report["project_source"] != source_identity(ROOT, SOURCE_PATHS):
        raise ValueError("Native build manifest does not match the current project sources")
    tvm_source = Path(report["tvm_source_root"])
    verify_cmake_source(Path(report["cmake_cache"]["path"]), tvm_source, build_source_root=build_source_root)
    if report["tvm_provenance"] != tvm_provenance(tvm_source):
        raise ValueError("Native build manifest does not match the TVM commit and patches")
    if report["tvm_source"] != source_identity(tvm_source, TVM_PATHS):
        raise ValueError("Native build manifest does not match the actual TVM build sources")
    expected = {str((ROOT / name).resolve()) for name in LIBRARY_PATHS}
    if {item["path"] for item in report["libraries"]} != expected:
        raise ValueError("Native build manifest does not identify all required local libraries")
    for item in [*report["libraries"], report["cmake_cache"]]:
        if file_digest(Path(item["path"])) != item["sha256"]:
            raise ValueError(f"Native build input or library changed: {item['path']}")
    build = report["build"]
    if not build.get("command") or file_digest(Path(build["log"])) != build["log_sha256"]:
        raise ValueError("Native build command or log identity is missing or changed")
    return {"path": str(path), **report}
