"""Host-only checks that migration runners cannot silently report false passes."""

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[3]
RUNNERS = ROOT / "scripts/ascend_migration"
spec = importlib.util.spec_from_file_location("migration_common", RUNNERS / "_common.py")
common = importlib.util.module_from_spec(spec)
spec.loader.exec_module(common)


@pytest.mark.parametrize(
    "body,status,passed,failed,skipped",
    [
        ("def test_ok(): assert 2 + 2 == 4", "PASS", 1, 0, 0),
        ("def test_bad(): assert False, 'injected failure'", "FAIL", 0, 1, 0),
        (
            "import pytest\n@pytest.mark.skip(reason='missing device')\ndef test_skip(): pass",
            "SKIP",
            0,
            0,
            1,
        ),
        (
            "import pytest\n@pytest.mark.xfail(reason='unimplemented')\ndef test_xfail(): assert False",
            "SKIP",
            0,
            0,
            1,
        ),
        ("def helper_only(): pass", "FAIL", 0, 0, 0),
        (
            "import pytest\ndef test_ok(): pass\n@pytest.mark.skip(reason='missing')\ndef test_skip(): pass",
            "BLOCKED",
            1,
            0,
            1,
        ),
    ],
)
def test_junit_requires_all_mandatory_cases(tmp_path, body, status, passed, failed, skipped):
    test_file = tmp_path / "test_fixture.py"
    test_file.write_text(body + "\n")
    env = dict(os.environ, PYTEST_DISABLE_PLUGIN_AUTOLOAD="1")
    result = common.run_pytest([str(test_file)], tmp_path / "artifacts", env=env, timeout=30)
    assert result["status"] == status, result
    assert result["counts"] == {
        "passed": passed,
        "failed": failed,
        "skipped": skipped,
        "total": passed + failed + skipped,
    }


def test_process_timeout_kills_descendants(tmp_path):
    pid_file = tmp_path / "child.pid"
    program = (
        "import pathlib,subprocess,sys,time; "
        "child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)']); "
        f"pathlib.Path({str(pid_file)!r}).write_text(str(child.pid)); "
        "time.sleep(60)"
    )
    result = common.run_process([sys.executable, "-c", program], tmp_path, "timeout", timeout=1)
    assert result["status"] == "TIMEOUT"
    assert result["exit_code"] == 124
    child_pid = int(pid_file.read_text())
    proc_stat = Path(f"/proc/{child_pid}/stat")
    # A dead, unreaped orphan can briefly remain as a zombie in containers.
    assert not proc_stat.exists() or proc_stat.read_text().split()[2] == "Z"


def test_deselected_mandatory_node_cannot_pass(tmp_path):
    test_file = tmp_path / "test_required.py"
    test_file.write_text("def test_ok(): pass\ndef test_required(): pass\n")
    (tmp_path / "conftest.py").write_text(
        "def pytest_collection_modifyitems(items):\n    items[:] = [item for item in items if item.name != 'test_required']\n"
    )
    nodes = [f"{test_file}::test_ok", f"{test_file}::test_required"]
    result = common.run_pytest(nodes, tmp_path / "artifacts", env=dict(os.environ, PYTEST_DISABLE_PLUGIN_AUTOLOAD="1"))
    assert result["counts"]["passed"] == 1
    assert result["status"] == "FAIL"
    assert result["missing_required_nodes"] == [nodes[1]]


def guarded_environment(tmp_path):
    (tmp_path / "sitecustomize.py").write_text(
        "import sys\n"
        "class Guard:\n"
        "    def find_spec(self, fullname, path=None, target=None):\n"
        "        if fullname.split('.')[0] in {'tilelang','tvm','torch','torch_npu','acl'}:\n"
        "            raise ImportError('device/runtime import forbidden: ' + fullname)\n"
        "sys.meta_path.insert(0, Guard())\n"
    )
    return dict(os.environ, PYTHONPATH=str(tmp_path))


@pytest.mark.parametrize("entry", ["s1_check_baseline", "s1_check_legacy_lowering", "s1_run_baseline_npu"])
def test_help_does_not_import_a_runtime(tmp_path, entry):
    result = subprocess.run(
        [sys.executable, str(RUNNERS / f"{entry}.py"), "--help"],
        env=guarded_environment(tmp_path),
        text=True,
        capture_output=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr
    assert "--output" in result.stdout


def test_host_audit_does_not_import_a_runtime(tmp_path):
    output = tmp_path / "host.json"
    result = subprocess.run(
        [sys.executable, str(RUNNERS / "s1_check_baseline.py"), "--output", str(output)],
        env=guarded_environment(tmp_path),
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    report = json.loads(output.read_text())
    assert report["status"] == "PASS"
    assert report["counts"]["passed"] > 0


def test_optimized_python_cannot_skip_source_assertions(tmp_path):
    output = tmp_path / "optimized.json"
    result = subprocess.run(
        [sys.executable, "-O", str(RUNNERS / "s1_check_baseline.py"), "--output", str(output)],
        env=guarded_environment(tmp_path),
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert result.returncode == 2, result.stderr
    report = json.loads(output.read_text())
    assert report["status"] == "BLOCKED"
    assert report["counts"]["passed"] == 0


@pytest.mark.parametrize("entry", ["s1_check_legacy_lowering", "s1_run_baseline_npu"])
def test_missing_build_evidence_is_blocked_before_runtime_import(tmp_path, entry):
    output = tmp_path / "blocked.json"
    env = guarded_environment(tmp_path)
    env["TL_ASCEND_MIGRATION_BUILD_MANIFEST"] = str(tmp_path / "does-not-exist.json")
    result = subprocess.run(
        [sys.executable, str(RUNNERS / f"{entry}.py"), "--platform", "a3", "--output", str(output)],
        env=env,
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert result.returncode == 2, result.stderr
    report = json.loads(output.read_text())
    assert report["status"] == "BLOCKED"
    assert report["counts"]["passed"] == 0


@pytest.mark.parametrize("changed", ["project", "tvm_python", "library", "cmake", "build_log"])
def test_native_identity_rejects_stale_inputs(tmp_path, monkeypatch, changed):
    build_spec = importlib.util.spec_from_file_location("migration_build", RUNNERS / "_native_build.py")
    build = importlib.util.module_from_spec(build_spec)
    build_spec.loader.exec_module(build)
    monkeypatch.setattr(build, "ROOT", tmp_path)
    # The fixture has no Git history; content identities remain real, not mocked.
    monkeypatch.setattr(build, "tvm_provenance", lambda _: {"fixture": "fixed provenance"})
    project = tmp_path / "src/example.cc"
    tvm_source = tmp_path / "tvm"
    tvm_python = tvm_source / "python/parser.py"
    cmake = tmp_path / "build/CMakeCache.txt"
    log = tmp_path / "build.log"
    libraries = [tmp_path / name for name in build.LIBRARY_PATHS]
    for path in (project, tvm_python, cmake, log, *libraries):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("original input\n")
    cmake.write_text(f"TVM_SOURCE_DIR:PATH={tvm_source}\nCMAKE_HOME_DIRECTORY:INTERNAL={tmp_path}\n")
    manifest = tmp_path / "manifest.json"
    build.write_manifest(
        manifest,
        tvm_source=tvm_source,
        build_command=["cmake", "--build", "build"],
        build_log=log,
    )
    assert build.verify_manifest(manifest)["project_source"]["files"] > 0
    mutated = {
        "project": project,
        "tvm_python": tvm_python,
        "library": libraries[0],
        "cmake": cmake,
        "build_log": log,
    }[changed]
    mutated.write_text("changed input\n")
    with pytest.raises(ValueError):
        build.verify_manifest(manifest)
