"""Exercise S2 CLI isolation and failure reporting without a device runtime."""

import json
import importlib.util
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[3]
RUNNERS = ROOT / "scripts/ascend_migration"
ENTRIES = ("s2_check_solver", "s2_check_task_ir", "s2_check_capabilities")


def test_empty_required_file_cannot_hide_behind_a_passing_file(tmp_path):
    spec = importlib.util.spec_from_file_location("s2_runner_common", RUNNERS / "_common.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    passing = tmp_path / "test_passing.py"
    empty = tmp_path / "test_empty.py"
    passing.write_text("def test_ok(): pass\n")
    empty.write_text("# no collected tests\n")
    result = module.run_pytest(
        [str(passing), str(empty)],
        tmp_path / "artifacts",
        env=dict(os.environ, PYTEST_DISABLE_PLUGIN_AUTOLOAD="1"),
    )
    assert result["counts"]["passed"] == 1
    assert result["status"] == "FAIL"
    assert result["missing_required_nodes"] == [str(empty)]


def isolated_environment(tmp_path, *, block_z3=False):
    blocked = {"tilelang", "tvm", "torch", "torch_npu", "acl"}
    if block_z3:
        blocked.add("z3")
    (tmp_path / "sitecustomize.py").write_text(
        "import sys\n"
        "class Guard:\n"
        "    def find_spec(self, fullname, path=None, target=None):\n"
        f"        if fullname.split('.')[0] in {blocked!r}:\n"
        "            raise ModuleNotFoundError('runtime import forbidden: ' + fullname, name=fullname)\n"
        "sys.meta_path.insert(0, Guard())\n"
    )
    return dict(os.environ, PYTHONPATH=str(tmp_path), PYTEST_DISABLE_PLUGIN_AUTOLOAD="1")


def arguments(entry, output):
    args = [sys.executable, str(RUNNERS / f"{entry}.py"), "--output", str(output)]
    if entry != "s2_check_solver":
        args += ["--platform", "a3"]
    return args


@pytest.mark.parametrize("entry", ENTRIES)
def test_s2_help_has_no_runtime_or_solver_import(tmp_path, entry):
    result = subprocess.run(
        [sys.executable, str(RUNNERS / f"{entry}.py"), "--help"],
        env=isolated_environment(tmp_path, block_z3=True),
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr
    assert "--output" in result.stdout


@pytest.mark.parametrize("entry", ["s2_check_solver", "s2_check_capabilities"])
def test_s2_host_entry_can_run_without_a_device_runtime(tmp_path, entry):
    output = tmp_path / "host.json"
    result = subprocess.run(
        arguments(entry, output),
        env=isolated_environment(tmp_path),
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(output.read_text())
    assert report["step"] == "S2"
    assert report["status"] == "PASS"
    assert report["counts"]["passed"] > 0
    assert report["counts"]["failed"] == report["counts"]["skipped"] == 0


def test_s2_missing_z3_is_blocked_not_a_partial_pass(tmp_path):
    output = tmp_path / "missing-z3.json"
    result = subprocess.run(
        arguments("s2_check_solver", output),
        env=isolated_environment(tmp_path, block_z3=True),
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 2, result.stdout + result.stderr
    report = json.loads(output.read_text())
    assert report["step"] == "S2"
    assert report["status"] == "BLOCKED"


def test_s2_native_requires_build_evidence_before_runtime_import(tmp_path):
    output = tmp_path / "native.json"
    env = isolated_environment(tmp_path)
    env["TL_ASCEND_MIGRATION_BUILD_MANIFEST"] = str(tmp_path / "absent.json")
    result = subprocess.run(
        arguments("s2_check_task_ir", output),
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 2, result.stdout + result.stderr
    report = json.loads(output.read_text())
    assert report["step"] == "S2"
    assert report["status"] == "BLOCKED"
    assert report["counts"]["passed"] == 0


@pytest.mark.parametrize("entry", ENTRIES)
def test_s2_optimized_python_cannot_skip_assertions(tmp_path, entry):
    output = tmp_path / "optimized.json"
    args = arguments(entry, output)
    args.insert(1, "-O")
    result = subprocess.run(
        args,
        env=isolated_environment(tmp_path),
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 2, result.stdout + result.stderr
    report = json.loads(output.read_text())
    assert report["status"] == "BLOCKED"
    assert report["counts"]["passed"] == 0
