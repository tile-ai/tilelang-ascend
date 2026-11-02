# Copyright (c) Tile-AI Organization.
# Licensed under the MIT License.
"""Configure packaged native dependencies without importing the TileLang runtime."""

from __future__ import annotations

import re
import shutil
import ctypes
import json
import subprocess
from importlib import metadata
from collections.abc import Iterable
from pathlib import Path


NATIVE_RUNTIME_MANIFEST = "ascend_sync_z3_runtime_files.txt"
PYPI_RUNTIME_CONFIG = "ascend_sync_z3_runtime.json"
NATIVE_LIBRARIES = ("libtvm.so", "libtvm_runtime.so", "libtilelang.so", "libtilelang_module.so")
_Z3_LIBRARY_NAME = re.compile(r"libz3\.so(?:\.[0-9]+)*\Z")
_Z3_HANDLES: dict[str, ctypes.CDLL] = {}


def _pypi_runtime_enabled(directory: Path) -> bool | None:
    config = directory / PYPI_RUNTIME_CONFIG
    if not config.exists():
        return None  # Direct CMake builds retain their ordinary ELF dependencies.
    try:
        value = json.loads(config.read_text(encoding="utf-8"))
        if not isinstance(value, dict) or set(value) != {"enabled"}:
            raise ValueError("expected only the enabled field")
        if type(value["enabled"]) is not bool:
            raise ValueError("enabled must be a boolean")
        return value["enabled"]
    except (OSError, ValueError) as error:
        raise RuntimeError(f"Invalid native runtime configuration {config}: {error}") from error


def configure_pypi_runtime(search_dirs: Iterable[str | Path], destination_dirs: Iterable[str | Path]) -> bool:
    """Use the current CMake manifest, including OFF, to configure Python packages."""
    directories = [Path(directory) for directory in search_dirs]
    manifest = next(
        (directory / NATIVE_RUNTIME_MANIFEST for directory in directories if (directory / NATIVE_RUNTIME_MANIFEST).exists()),
        None,
    )
    if manifest is None:
        raise FileNotFoundError("Native runtime manifest not found in the configured CMake build")
    enabled = _validate_native_runtime_manifest(manifest)
    protected = {directory.resolve() for directory in directories}
    protected.add(manifest.parent.resolve())
    destinations = {Path(directory).resolve() for directory in destination_dirs}
    if destinations & protected:
        raise ValueError("Python package runtime configuration must not overwrite CMake sources")
    for directory in destinations:
        directory.mkdir(parents=True, exist_ok=True)
        for candidate in directory.iterdir():
            if _Z3_LIBRARY_NAME.fullmatch(candidate.name) and (candidate.is_file() or candidate.is_symlink()):
                candidate.unlink()
        (directory / PYPI_RUNTIME_CONFIG).write_text(json.dumps({"enabled": enabled}) + "\n", encoding="utf-8")
    return enabled


def patch_packaged_library(library: str | Path) -> None:
    """Patch a package COPY; CMake outputs retain their native Z3 dependency."""
    library = Path(library)
    enabled = _pypi_runtime_enabled(library.parent)
    if enabled is None:
        raise RuntimeError(f"Missing Python package runtime configuration beside {library}")
    patchelf = shutil.which("patchelf")
    if not patchelf:
        raise RuntimeError("patchelf is required; install the Linux build dependencies")
    needed = subprocess.check_output([patchelf, "--print-needed", str(library)], text=True)
    z3_dependencies = [name for name in needed.splitlines() if _Z3_LIBRARY_NAME.fullmatch(name)]
    if z3_dependencies and not enabled:
        raise RuntimeError(f"OFF runtime configuration disagrees with Z3 dependency in {library}")
    command = [patchelf, "--set-rpath", "$ORIGIN"]
    for dependency in z3_dependencies:
        command.extend(["--remove-needed", dependency])
    subprocess.run([*command, str(library)], check=True)


def remove_legacy_package_outputs(directory: str | Path, protected_dirs: Iterable[str | Path]) -> None:
    """Remove old top-level wheel payloads, never the current raw CMake outputs."""
    directory = Path(directory).resolve()
    if directory in {Path(value).resolve() for value in protected_dirs}:
        raise ValueError(f"Refusing to clean a source/CMake directory: {directory}")
    if directory.is_dir():
        for candidate in directory.iterdir():
            if (candidate.name in NATIVE_LIBRARIES or _Z3_LIBRARY_NAME.fullmatch(candidate.name)) and (
                candidate.is_file() or candidate.is_symlink()
            ):
                candidate.unlink()


def stage_packaged_libraries(search_dirs: Iterable[str | Path], destination_dirs: Iterable[str | Path]) -> None:
    """Copy all required compiler/runtime libraries before patching their package copies."""
    directories = [Path(directory) for directory in search_dirs]
    sources = {}
    for name in NATIVE_LIBRARIES:
        source = next((directory / name for directory in directories if (directory / name).is_file()), None)
        if source is None:
            raise FileNotFoundError(f"Missing current CMake output {name}")
        sources[name] = source
    for destination in {Path(directory).resolve() for directory in destination_dirs}:
        for name, source in sources.items():
            target = destination / name
            if target.resolve() == source.resolve():
                raise ValueError(f"Refusing to patch original CMake output {source}")
            if target.is_symlink():
                target.unlink()
            shutil.copy2(source, target)
            patch_packaged_library(target)


def preload_pypi_z3_runtime(library: str | Path) -> ctypes.CDLL | None:
    """Resolve Z3 from the RUNNING environment before loading a packaged compiler."""
    if not _pypi_runtime_enabled(Path(library).parent):
        return None
    try:
        distribution = metadata.distribution("z3-solver")
    except metadata.PackageNotFoundError as error:
        raise RuntimeError("Install z3-solver>=4.13.0,<4.15.5 to load this TileLang package") from error
    match = re.match(r"^(\d+)\.(\d+)\.(\d+)(?:[.+-]|$)", distribution.version)
    if match is None or not (4, 13, 0) <= tuple(map(int, match.groups())) < (4, 15, 5):
        raise RuntimeError(f"Requires z3-solver>=4.13.0,<4.15.5; found {distribution.version}")
    candidates = [distribution.locate_file(f"z3/{subdir}/libz3.so") for subdir in ("lib", "lib64")]
    path = next((Path(candidate).resolve() for candidate in candidates if Path(candidate).is_file()), None)
    if path is None:
        raise RuntimeError("The installed z3-solver has no native libz3.so; reinstall z3-solver")
    key = str(path)
    if key not in _Z3_HANDLES:
        try:
            handle = ctypes.CDLL(key, mode=ctypes.RTLD_GLOBAL)
            version = [ctypes.c_uint() for _ in range(4)]
            handle.Z3_get_version.argtypes = [ctypes.POINTER(ctypes.c_uint)] * 4
            handle.Z3_get_version.restype = None
            handle.Z3_get_version(*(ctypes.byref(value) for value in version))
            actual = tuple(value.value for value in version)
            if not (4, 13, 0) <= actual[:3] < (4, 15, 5):
                raise RuntimeError(f"Native Z3 version {actual} is outside >=4.13.0,<4.15.5")
            _Z3_HANDLES[key] = handle
        except (OSError, AttributeError) as error:
            raise RuntimeError(f"Cannot load native Z3 from {path}: {error}") from error
    return _Z3_HANDLES[key]


def copy_native_notices(source_root: str | Path, package_dirs: Iterable[str | Path]) -> list[Path]:
    """Include the notices for the compiler and bundled native dependencies."""
    source_root = Path(source_root)
    notices = (
        ("3rdparty/tvm/LICENSE", "licenses/tvm/LICENSE"),
        ("3rdparty/tvm/NOTICE", "licenses/tvm/NOTICE"),
        ("src/transform/ascend_sync/NOTICE", "licenses/ascend_sync/NOTICE"),
        ("src/transform/ascend_sync/licenses/Z3-LICENSE.txt", "licenses/z3/LICENSE.txt"),
    )
    for original, _destination in notices:
        if not (source_root / original).is_file():
            raise FileNotFoundError(f"Missing native dependency notice: {source_root / original}")
    copied = []
    seen = set()
    for package_dir in package_dirs:
        package_dir = Path(package_dir)
        for original, relative_target in notices:
            target = package_dir / relative_target
            if target.resolve() in seen:
                continue
            seen.add(target.resolve())
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_root / original, target)
            copied.append(target)
    return copied


def _validate_native_runtime_manifest(manifest: Path) -> bool:
    """Validate the selected build manifest and report whether Z3 is enabled."""
    if not manifest.is_file():
        raise ValueError(f"Native runtime manifest is not a file: {manifest}")

    basenames = set()
    for line_number, line in enumerate(manifest.read_text(encoding="utf-8").splitlines(), 1):
        entry = line.strip()
        if not entry:
            continue
        source = Path(entry)
        if not source.is_absolute() or ".." in source.parts or _Z3_LIBRARY_NAME.fullmatch(source.name) is None:
            raise ValueError(f"Invalid native runtime entry at {manifest}:{line_number}: {entry}")
        if not source.is_file():
            raise FileNotFoundError(f"Native runtime file listed at {manifest}:{line_number} is missing: {source}")
        if source.name in basenames:
            raise ValueError(f"Duplicate native runtime basename in {manifest}: {source.name}")
        basenames.add(source.name)
    return bool(basenames)
