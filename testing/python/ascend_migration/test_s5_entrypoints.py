"""Host-only S5 CLI checks: missing evidence must never import a device runtime."""

import json
import subprocess
import sys

import pytest

from test_s2_entrypoints import RUNNERS, isolated_environment
from test_s5_cost_model import synthetic_profile

ENTRIES = (
    "s5_check_versions_ir",
    "s5_run_pipeline_npu",
    "s5_run_persistent_pipeline_npu",
)
NPU_ENTRIES = ENTRIES[1:]


def command(entry, output):
    return [
        sys.executable,
        str(RUNNERS / f"{entry}.py"),
        "--platform",
        "a3",
        "--output",
        str(output),
    ]


def run_blocked(tmp_path, entry, extra=(), *, optimized=False):
    output = tmp_path / "report.json"
    args = command(entry, output)
    if optimized:
        args.insert(1, "-O")
    env = isolated_environment(tmp_path, block_z3=True)
    env["TL_ASCEND_MIGRATION_BUILD_MANIFEST"] = str(tmp_path / "absent-manifest.json")
    env["TL_ASCEND_PIPELINE_COST_PROFILE"] = str(tmp_path / "absent-profile.json")
    result = subprocess.run([*args, *extra], env=env, text=True, capture_output=True, timeout=30)
    assert result.returncode == 2, result.stdout + result.stderr
    report = json.loads(output.read_text())
    assert report["step"] == "S5"
    assert report["status"] == "BLOCKED"
    assert report["counts"]["passed"] == 0
    assert "runtime import forbidden" not in result.stdout + result.stderr
    return report


@pytest.mark.parametrize("entry", ENTRIES)
def test_help_does_not_import_runtime_or_solver(tmp_path, entry):
    result = subprocess.run(
        [sys.executable, str(RUNNERS / f"{entry}.py"), "--help"],
        env=isolated_environment(tmp_path, block_z3=True),
        text=True,
        capture_output=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr
    assert "--output" in result.stdout


@pytest.mark.parametrize("entry", ENTRIES)
def test_missing_manifest_is_checked_before_runtime_import(tmp_path, entry):
    profile = tmp_path / "synthetic-profile.json"
    profile.write_text(json.dumps(synthetic_profile()))
    extra = ["--cost-profile", str(profile)] if entry in NPU_ENTRIES else []
    report = run_blocked(tmp_path, entry, extra)
    assert "manifest" in report["error"].lower()


@pytest.mark.parametrize("entry", ENTRIES)
def test_optimized_python_cannot_disable_checks(tmp_path, entry):
    report = run_blocked(tmp_path, entry, optimized=True)
    assert "optimization" in report["error"].lower() or "pythonoptimize" in report["error"].lower()


@pytest.mark.parametrize("entry", ENTRIES)
@pytest.mark.parametrize("value", ["nan", "inf", "0", "-1"])
def test_timeout_must_be_finite_and_positive(tmp_path, entry, value):
    run_blocked(tmp_path, entry, ["--timeout", value])


@pytest.mark.parametrize("entry", NPU_ENTRIES)
@pytest.mark.parametrize("flag,value", [("--repeat", "0"), ("--seed", "-1"), ("--device", "-1")])
def test_device_parameters_are_checked_before_runtime_import(tmp_path, entry, flag, value):
    run_blocked(tmp_path, entry, [flag, value])


@pytest.mark.parametrize("entry", NPU_ENTRIES)
@pytest.mark.parametrize("content", [None, "{}", "not json"])
def test_missing_or_invalid_cost_profile_is_blocked(tmp_path, entry, content):
    profile = tmp_path / "invalid-profile.json"
    if content is not None:
        profile.write_text(content)
    report = run_blocked(tmp_path, entry, ["--cost-profile", str(profile)])
    assert "cost profile" in report["error"].lower()


def test_calibration_needs_manifest_but_no_existing_cost_profile(tmp_path):
    output_profile = tmp_path / "new-profile.json"
    report = run_blocked(tmp_path, "s5_run_pipeline_npu", ["--calibrate-profile", str(output_profile)])
    assert "manifest" in report["error"].lower()
    assert not output_profile.exists()


@pytest.mark.parametrize("value", ["nan", "inf", "0", "-1"])
def test_calibration_timeout_must_be_finite_and_positive(tmp_path, value):
    report = run_blocked(
        tmp_path,
        "s5_run_pipeline_npu",
        ["--calibrate-profile", str(tmp_path / "model.json"), "--calibration-timeout", value],
    )
    assert "calibration timeout" in report["error"].lower()
