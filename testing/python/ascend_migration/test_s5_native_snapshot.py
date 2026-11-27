"""Host-only provenance tests using synthetic bytes, never claimed as builds."""

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture
def build_fixture(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("s5_snapshot_build", ROOT / "scripts/ascend_migration/_native_build.py")
    build = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(build)
    original = tmp_path / "original"
    snapshot = tmp_path / "snapshot"
    monkeypatch.setattr(build, "ROOT", original)
    monkeypatch.setattr(build, "tvm_provenance", lambda _: {"fixture": "fixed provenance"})
    tvm_source = tmp_path / "fixed-tvm"
    cache = original / "build/CMakeCache.txt"
    log = original / "build.log"
    paths = [original / "src/example.cc", tvm_source / "python/parser.py", cache, log]
    paths.extend(original / name for name in build.LIBRARY_PATHS)
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("synthetic original bytes\n")
    cache.write_text(f"TVM_SOURCE_DIR:PATH={tvm_source}\nCMAKE_HOME_DIRECTORY:INTERNAL={original}\n")
    original_manifest = original / "native.json"
    build.write_manifest(
        original_manifest,
        tvm_source=tvm_source,
        build_command=["synthetic-build-fixture"],
        build_log=log,
    )
    (snapshot / "src").mkdir(parents=True)
    (snapshot / "src/example.cc").write_bytes((original / "src/example.cc").read_bytes())
    relocated = snapshot / "native.json"
    return build, original, snapshot, original_manifest, relocated


def test_snapshot_preserves_real_original_cache_and_survives_original_changes(build_fixture, monkeypatch):
    build, original, snapshot, manifest, relocated = build_fixture
    report = build.write_relocated_manifest(manifest, snapshot, relocated)
    assert report["artifact_kind"] == "relocated_verified_build"
    assert report["build"]["command"] == ["synthetic-build-fixture"]
    assert (snapshot / "build/CMakeCache.txt").read_bytes() == (original / "build/CMakeCache.txt").read_bytes()
    for relative in ("src/example.cc", "build/CMakeCache.txt", *build.LIBRARY_PATHS):
        (original / relative).write_text("new original workspace bytes\n")
    monkeypatch.setattr(build, "ROOT", snapshot)
    assert build.verify_manifest(relocated)["schema_version"] == 2


def test_only_a_source_identical_to_the_verified_build_can_be_relocated(build_fixture):
    build, _, snapshot, manifest, relocated = build_fixture
    (snapshot / "src/example.cc").write_text("different source\n")
    with pytest.raises(ValueError, match="source contents differ"):
        build.write_relocated_manifest(manifest, snapshot, relocated)


@pytest.mark.parametrize(
    "changed",
    [
        "src/example.cc",
        "build/libtilelang_module.so",
        "build/CMakeCache.txt",
        "build/ascend_migration_original_build.log",
        "build/ascend_migration_original_manifest.json",
    ],
)
def test_snapshot_mutation_cannot_be_hidden_by_changes_in_the_original(build_fixture, monkeypatch, changed):
    build, _, snapshot, manifest, relocated = build_fixture
    build.write_relocated_manifest(manifest, snapshot, relocated)
    monkeypatch.setattr(build, "ROOT", snapshot)
    (snapshot / changed).write_text("changed snapshot bytes\n")
    with pytest.raises(ValueError):
        build.verify_manifest(relocated)


@pytest.mark.parametrize("field", ["source", "libraries", "command", "cache_origin"])
def test_relocated_attestation_must_match_its_preserved_original(build_fixture, monkeypatch, field):
    build, original, snapshot, manifest, relocated = build_fixture
    report = build.write_relocated_manifest(manifest, snapshot, relocated)
    if field == "source":
        report["project_source"]["digest"] = "f" * 64
    elif field == "libraries":
        report["libraries"][0]["path"] = str(original / build.LIBRARY_PATHS[0])
    elif field == "command":
        report["build"]["command"] = ["not-the-original-command"]
    else:
        report["relocated_from"]["build_source_root"] = str(snapshot)
    relocated.write_text(json.dumps(report))
    monkeypatch.setattr(build, "ROOT", snapshot)
    with pytest.raises(ValueError):
        build.verify_manifest(relocated)


def test_symlinked_artifacts_cannot_be_overwritten_by_an_original_rebuild(build_fixture):
    build, original, snapshot, manifest, relocated = build_fixture
    (snapshot / "build").symlink_to(original / "build", target_is_directory=True)
    with pytest.raises(ValueError, match="must not link back"):
        build.write_relocated_manifest(manifest, snapshot, relocated)


def test_nested_relocation_requires_an_original_build_attestation(build_fixture, monkeypatch):
    build, _, snapshot, manifest, relocated = build_fixture
    build.write_relocated_manifest(manifest, snapshot, relocated)
    monkeypatch.setattr(build, "ROOT", snapshot)
    with pytest.raises(ValueError, match="non-relocated"):
        build.write_relocated_manifest(relocated, snapshot.parent / "second", snapshot / "new.json")
