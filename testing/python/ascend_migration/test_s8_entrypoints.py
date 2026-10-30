"""CLI isolation and failure reporting for the three independent S8 entries."""

import json
import subprocess
import sys

import pytest

from test_s2_entrypoints import RUNNERS, isolated_environment


ENTRIES = (
    "s8_check_integration_ir",
    "s8_run_workloads_npu",
    "s8_benchmark_npu",
)
NPU_ENTRIES = ENTRIES[1:]


def run_blocked(tmp_path, entry, extra=(), *, optimize=False):
    output = tmp_path / "report.json"
    command = [sys.executable]
    if optimize:
        command.append("-O")
    command.extend([str(RUNNERS / f"{entry}.py"), "--platform", "a3", "--output", str(output), *extra])
    env = isolated_environment(tmp_path, block_z3=True)
    env["TL_ASCEND_MIGRATION_BUILD_MANIFEST"] = str(tmp_path / "absent-build.json")
    env["TL_ASCEND_PIPELINE_COST_PROFILE"] = str(tmp_path / "absent-profile.json")
    result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode == 2, result.stdout + result.stderr
    report = json.loads(output.read_text())
    assert report["step"] == "S8" and report["status"] == "BLOCKED"
    assert report["counts"]["passed"] == 0
    assert "runtime import forbidden" not in result.stdout + result.stderr
    return report


@pytest.mark.parametrize("entry", ENTRIES)
def test_help_needs_neither_runtime_nor_solver(tmp_path, entry):
    result = subprocess.run(
        [sys.executable, str(RUNNERS / f"{entry}.py"), "--help"],
        env=isolated_environment(tmp_path, block_z3=True),
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "--output" in result.stdout and "--platform" in result.stdout
    assert "--cost-profile" not in result.stdout


@pytest.mark.parametrize("entry", ENTRIES)
def test_manifest_is_required_before_any_runtime_import(tmp_path, entry):
    report = run_blocked(tmp_path, entry)
    error = report["error"].lower()
    assert "manifest" in error or "build evidence" in error
    assert "profile" not in error


@pytest.mark.parametrize("entry", ENTRIES)
def test_explicit_missing_manifest_is_not_ignored(tmp_path, entry):
    report = run_blocked(tmp_path, entry, ["--build-manifest", str(tmp_path / "explicit.json")])
    assert "explicit.json" in report["error"]


@pytest.mark.parametrize("entry", ENTRIES)
def test_optimization_cannot_disable_checks(tmp_path, entry):
    error = run_blocked(tmp_path, entry, optimize=True)["error"].lower()
    assert "optimization" in error or "pythonoptimize" in error


@pytest.mark.parametrize("entry", ENTRIES)
@pytest.mark.parametrize("value", ["nan", "inf", "0", "-1"])
def test_timeout_is_checked_before_any_runtime_import(tmp_path, entry, value):
    report = run_blocked(tmp_path, entry, ["--timeout", value])
    assert "timeout" in report["error"].lower()


@pytest.mark.parametrize("entry", NPU_ENTRIES)
@pytest.mark.parametrize("flag,value", [("--repeat", "0"), ("--seed", "-1"), ("--device", "-1")])
def test_device_parameters_fail_before_any_runtime_import(tmp_path, entry, flag, value):
    report = run_blocked(tmp_path, entry, [flag, value])
    assert flag.removeprefix("--") in report["error"].lower()
