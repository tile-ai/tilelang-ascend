"""Host checks for S3 CLI dependency isolation and failure reports."""

import json
import subprocess
import sys

import pytest

from test_s2_entrypoints import RUNNERS, isolated_environment

ENTRIES = ("s3_check_loop_ir", "s3_check_persistent_domain", "s3_run_loop_npu")


def command(entry, output):
    return [
        sys.executable,
        str(RUNNERS / f"{entry}.py"),
        "--platform",
        "a3",
        "--output",
        str(output),
    ]


@pytest.mark.parametrize("entry", ENTRIES)
def test_s3_help_does_not_import_runtime_or_solver(tmp_path, entry):
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
def test_s3_missing_native_evidence_prevents_runtime_load(tmp_path, entry):
    output = tmp_path / "missing.json"
    env = isolated_environment(tmp_path, block_z3=True)
    env["TL_ASCEND_MIGRATION_BUILD_MANIFEST"] = str(tmp_path / "absent.json")
    result = subprocess.run(
        command(entry, output),
        env=env,
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert result.returncode == 2, result.stdout + result.stderr
    report = json.loads(output.read_text())
    assert report["step"] == "S3"
    assert report["status"] == "BLOCKED"
    assert report["counts"]["passed"] == 0


@pytest.mark.parametrize("entry", ENTRIES)
def test_s3_optimized_python_is_blocked(tmp_path, entry):
    output = tmp_path / "optimized.json"
    args = command(entry, output)
    args.insert(1, "-O")
    result = subprocess.run(
        args,
        env=isolated_environment(tmp_path, block_z3=True),
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert result.returncode == 2, result.stdout + result.stderr
    report = json.loads(output.read_text())
    assert report["status"] == "BLOCKED"
    assert report["counts"]["passed"] == 0


@pytest.mark.parametrize(
    "entry,value",
    [("s3_check_loop_ir", "nan"), ("s3_check_persistent_domain", "inf"), ("s3_run_loop_npu", "-1")],
)
def test_s3_timeout_must_be_finite_and_positive(tmp_path, entry, value):
    output = tmp_path / "timeout.json"
    result = subprocess.run(
        [*command(entry, output), "--timeout", value],
        env=isolated_environment(tmp_path, block_z3=True),
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert result.returncode == 2, result.stdout + result.stderr
    assert json.loads(output.read_text())["status"] == "BLOCKED"


def test_s3_npu_repeat_must_be_positive(tmp_path):
    output = tmp_path / "repeat.json"
    result = subprocess.run(
        [*command("s3_run_loop_npu", output), "--repeat", "0"],
        env=isolated_environment(tmp_path, block_z3=True),
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert result.returncode == 2, result.stdout + result.stderr
    assert json.loads(output.read_text())["status"] == "BLOCKED"
