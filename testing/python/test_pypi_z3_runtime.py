# Copyright (c) Tile-AI Organization.
# Licensed under the MIT License.
"""Python-package Z3 resolution and real ELF handoff, without importing TileLang."""

import ctypes
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


_HELPER = Path(__file__).resolve().parents[2] / "tilelang" / "_native_runtime.py"
_SPEC = importlib.util.spec_from_file_location("pypi_z3_runtime_helper", _HELPER)
runtime = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(runtime)


def _manifest(directory, entries):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / runtime.NATIVE_RUNTIME_MANIFEST).write_text("".join(f"{entry}\n" for entry in entries))


def _marker(directory, enabled):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / runtime.PYPI_RUNTIME_CONFIG).write_text(json.dumps({"enabled": enabled}))


@pytest.mark.parametrize("enabled", [False, True])
def test_current_manifest_controls_package_without_environment_override(tmp_path, enabled):
    build, package = tmp_path / "build", tmp_path / "package"
    build.mkdir()
    library = build / "libz3.so"
    library.write_bytes(b"current-build-provider")
    _manifest(build, [library] if enabled else [])
    package.mkdir()
    (package / "libz3.so").write_bytes(b"stale")
    (package / "libz3.so.4.15").symlink_to(tmp_path / "missing")
    (package / "libunrelated.so").write_bytes(b"preserve")
    assert runtime.configure_pypi_runtime([build], [package]) is enabled
    assert runtime._pypi_runtime_enabled(package) is enabled
    assert not list(package.glob("libz3.so*"))
    assert library.read_bytes() == b"current-build-provider"
    assert (package / "libunrelated.so").read_bytes() == b"preserve"


def test_runtime_configuration_requires_the_configured_build_and_protects_sources(tmp_path):
    with pytest.raises(FileNotFoundError, match="manifest"):
        runtime.configure_pypi_runtime([tmp_path], [tmp_path / "package"])
    _manifest(tmp_path, [])
    with pytest.raises(ValueError, match="overwrite CMake"):
        runtime.configure_pypi_runtime([tmp_path], [tmp_path])


def test_configure_and_stage_multiple_packages_preserves_native_sources(tmp_path, monkeypatch):
    build, output = tmp_path / "build", tmp_path / "native-output"
    output.mkdir()
    providers = [output / name for name in ("libz3.so", "libz3.so.4.15")]
    for provider in providers:
        provider.write_bytes(b"native-z3")
    _manifest(build, providers)
    for name in runtime.NATIVE_LIBRARIES:
        (build / name).write_bytes(name.encode())
    destinations = [tmp_path / "wheel/tilelang/lib", tmp_path / "editable/lib"]
    patched = []
    monkeypatch.setattr(runtime, "patch_packaged_library", patched.append)

    assert runtime.configure_pypi_runtime([tmp_path / "no-manifest", build], destinations) is True
    runtime.stage_packaged_libraries([build], [*destinations, destinations[0]])

    assert set(patched) == {destination / name for destination in destinations for name in runtime.NATIVE_LIBRARIES}
    assert len(patched) == len(destinations) * len(runtime.NATIVE_LIBRARIES)
    for provider in providers:
        assert provider.read_bytes() == b"native-z3"
    for name in runtime.NATIVE_LIBRARIES:
        assert (build / name).read_bytes() == name.encode()
        for destination in destinations:
            assert (destination / name).read_bytes() == name.encode()
            assert not list(destination.glob("libz3.so*"))


def test_off_first_manifest_stops_search_despite_stale_libraries(tmp_path):
    current, stale = tmp_path / "current", tmp_path / "stale"
    _manifest(current, [])
    stale.mkdir()
    source = stale / "libz3.so"
    source.write_bytes(b"stale-z3")
    _manifest(stale, [source])
    (current / "libz3.so").write_bytes(b"leftover-from-on-build")
    destination = tmp_path / "package/lib"

    assert runtime.configure_pypi_runtime([current, stale], [destination]) is False
    assert runtime._pypi_runtime_enabled(destination) is False
    assert not list(destination.glob("libz3.so*"))
    assert source.read_bytes() == b"stale-z3"
    assert (current / "libz3.so").read_bytes() == b"leftover-from-on-build"


def test_first_manifest_selects_its_own_absolute_files(tmp_path):
    first, second, output = tmp_path / "first", tmp_path / "second", tmp_path / "native-output"
    output.mkdir()
    source = output / "libz3.so"
    source.write_bytes(b"first-provider")
    _manifest(first, [source])
    _manifest(second, [second / "missing/libz3.so"])
    destination = tmp_path / "package/lib"

    assert runtime.configure_pypi_runtime([first, second], [destination]) is True
    assert source.read_bytes() == b"first-provider"
    assert not list(destination.glob("libz3.so*"))


def test_missing_native_file_fails_before_package_mutation(tmp_path):
    build, destination = tmp_path / "build", tmp_path / "package/lib"
    build.mkdir()
    source = build / "libz3.so"
    source.write_bytes(b"z3")
    _manifest(build, [source, build / "libz3.so.4.15"])

    with pytest.raises(FileNotFoundError, match="Native runtime file.*is missing"):
        runtime.configure_pypi_runtime([build], [destination])
    assert not destination.exists()
    assert source.read_bytes() == b"z3"


@pytest.mark.parametrize("entry", ["relative", "relative_parent", "wrong_name", "absolute_parent"])
def test_invalid_native_path_entry_is_rejected(tmp_path, entry):
    entries = {
        "relative": "libz3.so",
        "relative_parent": "../libz3.so",
        "wrong_name": str(tmp_path / "not-z3.so"),
        "absolute_parent": str(tmp_path / ".." / "libz3.so"),
    }
    _manifest(tmp_path, [entries[entry]])

    with pytest.raises(ValueError, match="Invalid native runtime entry"):
        runtime.configure_pypi_runtime([tmp_path], [tmp_path / "package"])


def test_duplicate_native_basenames_are_rejected_before_package_mutation(tmp_path):
    sources = [tmp_path / directory / "libz3.so" for directory in ("first", "second")]
    for source in sources:
        source.parent.mkdir()
        source.write_bytes(b"provider")
    _manifest(tmp_path, sources)
    destination = tmp_path / "package"

    with pytest.raises(ValueError, match="Duplicate native runtime basename"):
        runtime.configure_pypi_runtime([tmp_path], [destination])
    assert not destination.exists()


def test_manifest_directory_is_rejected(tmp_path):
    (tmp_path / runtime.NATIVE_RUNTIME_MANIFEST).mkdir()
    with pytest.raises(ValueError, match="manifest is not a file"):
        runtime.configure_pypi_runtime([tmp_path], [tmp_path / "package"])


def test_on_then_off_removes_staged_z3_and_preserves_sources(tmp_path):
    build = tmp_path / "build"
    build.mkdir()
    sources = [build / name for name in ("libz3.so", "libz3.so.4.15")]
    for source in sources:
        source.write_bytes(b"native-z3")
    _manifest(build, sources)
    destinations = [tmp_path / "wheel/tilelang/lib", tmp_path / "editable/lib"]
    assert runtime.configure_pypi_runtime([build], destinations) is True
    for destination in destinations:
        for source in sources:
            (destination / source.name).write_bytes(b"stale-bundled-provider")

    _manifest(build, [])
    assert runtime.configure_pypi_runtime([build], destinations) is False
    for source in sources:
        assert source.read_bytes() == b"native-z3"
        for destination in destinations:
            assert not (destination / source.name).exists()
            assert runtime._pypi_runtime_enabled(destination) is False


def test_off_cleanup_only_removes_matching_direct_files(tmp_path):
    build, destination = tmp_path / "build", tmp_path / "package/lib"
    _manifest(build, [])
    destination.mkdir(parents=True)
    preserved = ["libtilelang.so", "libz3.so.debug", "libz3.so.4.15.backup", "notes.txt"]
    for name in preserved:
        (destination / name).write_bytes(b"keep")
    nested = destination / "libz3.so.999"
    nested.mkdir()
    (nested / "libz3.so").write_bytes(b"nested-keep")
    (destination / "libz3.so").write_bytes(b"stale")
    (destination / "libz3.so.4.15").symlink_to("missing-native-file")

    assert runtime.configure_pypi_runtime([build], [destination]) is False
    for name in preserved:
        assert (destination / name).read_bytes() == b"keep"
    assert (nested / "libz3.so").read_bytes() == b"nested-keep"
    assert not (destination / "libz3.so").exists()
    assert not (destination / "libz3.so.4.15").is_symlink()


@pytest.mark.parametrize("target", ["current", "other_source", "source_alias"])
def test_configuration_protects_search_directories_and_aliases(tmp_path, target):
    current, other_source, source_alias = tmp_path / "current", tmp_path / "other-source", tmp_path / "source-alias"
    _manifest(current, [])
    other_source.mkdir()
    for directory in (current, other_source):
        (directory / "libz3.so").write_bytes(b"source-library")
    if target == "source_alias":
        source_alias.symlink_to(current, target_is_directory=True)
    destinations = {"current": current, "other_source": other_source, "source_alias": source_alias}

    with pytest.raises(ValueError, match="overwrite CMake"):
        runtime.configure_pypi_runtime([current, other_source], [destinations[target]])
    for directory in (current, other_source):
        assert (directory / "libz3.so").read_bytes() == b"source-library"


def test_staging_refuses_to_patch_colocated_native_sources(tmp_path, monkeypatch):
    for name in runtime.NATIVE_LIBRARIES:
        (tmp_path / name).write_bytes(b"native-library")
    monkeypatch.setattr(runtime, "patch_packaged_library", lambda path: pytest.fail("patched original CMake output"))

    with pytest.raises(ValueError, match="Refusing to patch original CMake output"):
        runtime.stage_packaged_libraries([tmp_path], [tmp_path])
    assert all((tmp_path / name).read_bytes() == b"native-library" for name in runtime.NATIVE_LIBRARIES)


@pytest.mark.parametrize("contents", ['{"enabled": 1}', "{}", "null", "not-json"])
def test_malformed_marker_does_not_silently_become_native_or_off(tmp_path, contents):
    (tmp_path / runtime.PYPI_RUNTIME_CONFIG).write_text(contents)
    with pytest.raises(RuntimeError, match="Invalid native runtime configuration"):
        runtime.preload_pypi_z3_runtime(tmp_path / "libtilelang_module.so")


@pytest.mark.parametrize("marker", [None, False])
def test_off_and_native_builds_do_not_query_or_load_pypi_z3(tmp_path, monkeypatch, marker):
    if marker is not None:
        _marker(tmp_path, marker)

    def forbidden(*args, **kwargs):
        pytest.fail("OFF/native builds must not query or load Python-package Z3")

    monkeypatch.setattr(runtime.metadata, "distribution", forbidden)
    monkeypatch.setattr(runtime.ctypes, "CDLL", forbidden)
    assert runtime.preload_pypi_z3_runtime(tmp_path / "libtilelang_module.so") is None


def test_missing_pypi_dependency_is_actionable(tmp_path, monkeypatch):
    _marker(tmp_path, True)

    def missing(name):
        raise runtime.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(runtime.metadata, "distribution", missing)
    with pytest.raises(RuntimeError, match="Install z3-solver>=4.13.0,<4.15.5"):
        runtime.preload_pypi_z3_runtime(tmp_path / "libtilelang_module.so")


@pytest.mark.parametrize("version", ["4.12.6.0", "4.15.5.0", "4.16.0", "broken"])
def test_outside_main_version_range_is_rejected_before_loading(tmp_path, monkeypatch, version):
    _marker(tmp_path, True)
    distribution = type("Distribution", (), {"version": version})()
    monkeypatch.setattr(runtime.metadata, "distribution", lambda name: distribution)
    monkeypatch.setattr(runtime.ctypes, "CDLL", lambda *args, **kwargs: pytest.fail("loaded"))
    with pytest.raises(RuntimeError, match="Requires z3-solver"):
        runtime.preload_pypi_z3_runtime(tmp_path / "libtilelang_module.so")


@pytest.mark.parametrize("version", [(4, 13, 0, 0), (4, 15, 4, 0), (4, 15, 5, 0)])
def test_native_version_checked_and_provider_handle_retained(tmp_path, monkeypatch, version):
    _marker(tmp_path / "package", True)
    binary = tmp_path / "z3/lib/libz3.so"
    binary.parent.mkdir(parents=True)
    binary.write_bytes(b"provider")
    distribution = type(
        "Distribution",
        (),
        {"version": "4.15.4.0", "locate_file": lambda self, name: tmp_path / name},
    )()
    monkeypatch.setattr(runtime.metadata, "distribution", lambda name: distribution)

    class GetVersion:
        def __call__(self, *pointers):
            for pointer, value in zip(pointers, version):
                ctypes.cast(pointer, ctypes.POINTER(ctypes.c_uint))[0] = value

    handle = type("Provider", (), {"Z3_get_version": GetVersion()})()
    calls = []

    def load(path, *, mode):
        calls.append((path, mode))
        return handle

    monkeypatch.setattr(runtime.ctypes, "CDLL", load)
    if version[:3] >= (4, 15, 5):
        with pytest.raises(RuntimeError, match="Native Z3 version"):
            runtime.preload_pypi_z3_runtime(tmp_path / "package/libtilelang_module.so")
    else:
        assert runtime.preload_pypi_z3_runtime(tmp_path / "package/libtilelang_module.so") is handle
        assert runtime.preload_pypi_z3_runtime(tmp_path / "package/libtilelang_module.so") is handle
        assert calls == [(str(binary.resolve()), ctypes.RTLD_GLOBAL)]


@pytest.fixture
def native_build(tmp_path):
    compiler = shutil.which("cc")
    if not sys.platform.startswith("linux") or not compiler or not shutil.which("patchelf"):
        pytest.skip("real ELF check requires Linux, a C compiler and patchelf")
    build = tmp_path / "native-build"
    build.mkdir()
    provider = build / "libz3.so"
    body = """
void Z3_get_version(unsigned *a, unsigned *b, unsigned *c, unsigned *d) {
  *a = 4; *b = 15; *c = 4; *d = 0;
}
"""
    subprocess.run(
        [
            compiler,
            "-shared",
            "-fPIC",
            "-x",
            "c",
            "-",
            "-Wl,-soname,libz3.so.4.15",
            "-o",
            str(provider),
        ],
        input=body,
        text=True,
        check=True,
        capture_output=True,
    )
    body = """
extern void Z3_get_version(unsigned *, unsigned *, unsigned *, unsigned *);
int probe(void) {
  unsigned a, b, c, d;
  Z3_get_version(&a, &b, &c, &d);
  return a * 10000 + b * 100 + c;
}
"""
    consumer = build / "consumer.so"
    subprocess.run(
        [
            compiler,
            "-shared",
            "-fPIC",
            "-x",
            "c",
            "-",
            "-x",
            "none",
            str(provider),
            "-o",
            str(consumer),
        ],
        input=body,
        text=True,
        check=True,
        capture_output=True,
    )
    for name in runtime.NATIVE_LIBRARIES:
        shutil.copy2(consumer, build / name)
    _manifest(build, [provider])
    return build


def test_real_elf_copy_and_cross_soname_runtime_resolution(native_build, tmp_path):
    before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in native_build.glob("*.so")}
    package = tmp_path / "package"
    runtime.configure_pypi_runtime([native_build], [package])
    runtime.stage_packaged_libraries([native_build], [package])
    for name in runtime.NATIVE_LIBRARIES:
        needed = subprocess.check_output(["patchelf", "--print-needed", str(package / name)], text=True)
        assert "libz3" not in needed
        assert subprocess.check_output(["patchelf", "--print-rpath", str(package / name)], text=True).strip() == "$ORIGIN"
    assert before == {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in native_build.glob("*.so")}
    assert not list(package.glob("libz3.so*"))

    # An independently installed runtime provider has a different SONAME.
    installed = tmp_path / "installed"
    provider = installed / "z3/lib/libz3.so"
    provider.parent.mkdir(parents=True)
    subprocess.run(
        [
            shutil.which("cc"),
            "-shared",
            "-fPIC",
            "-x",
            "c",
            "-",
            "-Wl,-soname,libz3.so.4.13",
            "-o",
            str(provider),
        ],
        input="void Z3_get_version(unsigned*a,unsigned*b,unsigned*c,unsigned*d){*a=4;*b=13;*c=0;*d=0;}",
        text=True,
        check=True,
        capture_output=True,
    )
    info = installed / "z3_solver-4.13.0.0.dist-info"
    info.mkdir()
    (info / "METADATA").write_text("Metadata-Version: 2.1\nName: z3-solver\nVersion: 4.13.0.0\n")
    script = """
import ctypes, importlib.util, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
spec = importlib.util.spec_from_file_location('helper', sys.argv[2])
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)
library = sys.argv[3]
try:
    ctypes.CDLL(library)
except OSError as error:
    assert 'Z3_get_version' in str(error), error
else:
    raise AssertionError('unprovided symbol was unexpectedly resolved')
provider = helper.preload_pypi_z3_runtime(library)
assert provider is not None
consumer = ctypes.CDLL(library)
consumer.probe.restype = ctypes.c_int
assert consumer.probe() == 41300
assert str(Path(sys.argv[1]) / 'z3/lib/libz3.so') in Path('/proc/self/maps').read_text()
"""
    subprocess.run(
        [
            sys.executable,
            "-I",
            "-B",
            "-c",
            script,
            str(installed),
            str(_HELPER),
            str(package / "libtilelang_module.so"),
        ],
        check=True,
        capture_output=True,
        text=True,
    )


def test_off_marker_cannot_publish_an_on_binary(native_build, tmp_path):
    _manifest(native_build, [])
    package = tmp_path / "package"
    runtime.configure_pypi_runtime([native_build], [package])
    with pytest.raises(RuntimeError, match="OFF runtime configuration disagrees"):
        runtime.stage_packaged_libraries([native_build], [package])


def test_missing_tool_is_reported_without_modifying_library(tmp_path, monkeypatch):
    _marker(tmp_path, True)
    library = tmp_path / "libtilelang_module.so"
    library.write_bytes(b"preserve")
    monkeypatch.setattr(runtime.shutil, "which", lambda name: None)
    with pytest.raises(RuntimeError, match="patchelf is required"):
        runtime.patch_packaged_library(library)
    assert library.read_bytes() == b"preserve"


def test_old_wheel_payload_cleanup_preserves_raw_outputs_and_unrelated_files(tmp_path):
    package_root, raw = tmp_path / "build-lib", tmp_path / "build-temp/native"
    package_root.mkdir()
    raw.mkdir(parents=True)
    for name in [*runtime.NATIVE_LIBRARIES, "libz3.so", "libz3.so.4.15"]:
        (package_root / name).write_bytes(b"stale")
        (raw / name).write_bytes(b"raw")
    (package_root / "unrelated.so").write_bytes(b"keep")
    runtime.remove_legacy_package_outputs(package_root, [raw])
    assert sorted(p.name for p in package_root.iterdir()) == ["unrelated.so"]
    assert all(p.read_bytes() == b"raw" for p in raw.iterdir())
    with pytest.raises(ValueError, match="Refusing to clean"):
        runtime.remove_legacy_package_outputs(raw, [raw])
