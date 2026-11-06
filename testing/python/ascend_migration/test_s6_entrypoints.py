"""Host-only S6 entrypoint and per-launch paired-evidence contracts."""

import copy
import importlib.util
import json
import shutil
import subprocess
import sys

import pytest

from test_s2_entrypoints import RUNNERS, isolated_environment
from test_s5_cost_model import synthetic_profile

ENTRIES = ("s6_check_owner_ir", "s6_check_lifecycle_guards", "s6_run_reuse_npu")


def command(entry, output):
    return [
        sys.executable,
        str(RUNNERS / f"{entry}.py"),
        "--platform",
        "a3",
        "--output",
        str(output),
    ]


def blocked(tmp_path, entry, extra=(), *, optimize=False, environment=None):
    output = tmp_path / "report.json"
    args = command(entry, output)
    if optimize:
        args.insert(1, "-O")
    env = environment or isolated_environment(tmp_path, block_z3=True)
    env["TL_ASCEND_MIGRATION_BUILD_MANIFEST"] = str(tmp_path / "missing-manifest.json")
    env["TL_ASCEND_PIPELINE_COST_PROFILE"] = str(tmp_path / "missing-profile.json")
    result = subprocess.run([*args, *extra], env=env, text=True, capture_output=True, timeout=30)
    assert result.returncode == 2, result.stdout + result.stderr
    report = json.loads(output.read_text())
    assert report["step"] == "S6" and report["status"] == "BLOCKED"
    assert report["counts"]["passed"] == 0
    assert "runtime import forbidden" not in result.stdout + result.stderr
    return report


@pytest.mark.parametrize("entry", ENTRIES)
def test_help_is_stdlib_only(tmp_path, entry):
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
def test_missing_manifest_prevents_runtime_import(tmp_path, entry):
    profile = tmp_path / "synthetic-profile.json"
    profile.write_text(json.dumps(synthetic_profile()))
    extra = ["--cost-profile", str(profile)] if entry == ENTRIES[-1] else []
    report = blocked(tmp_path, entry, extra)
    assert "manifest" in report["error"].lower()


@pytest.mark.parametrize("entry", ENTRIES)
def test_optimized_python_cannot_skip_assertions(tmp_path, entry):
    report = blocked(tmp_path, entry, optimize=True)
    assert "optimization" in report["error"].lower() or "pythonoptimize" in report["error"].lower()


@pytest.mark.parametrize("entry", ENTRIES)
@pytest.mark.parametrize("value", ["nan", "inf", "0", "-1"])
def test_timeout_is_finite_positive(tmp_path, entry, value):
    blocked(tmp_path, entry, ["--timeout", value])


@pytest.mark.parametrize("flag,value", [("--repeat", "0"), ("--seed", "-1"), ("--device", "-1")])
def test_reuse_device_parameters_precede_runtime(tmp_path, flag, value):
    blocked(tmp_path, ENTRIES[-1], [flag, value])


@pytest.mark.parametrize("content", [None, "{}", "not JSON"])
def test_cost_profile_must_be_existing_and_valid(tmp_path, content):
    path = tmp_path / "invalid-profile.json"
    if content is not None:
        path.write_text(content)
    report = blocked(tmp_path, ENTRIES[-1], ["--cost-profile", str(path)])
    assert "cost profile" in report["error"].lower()


def test_reuse_validation_does_not_require_msprof(tmp_path):
    profile = tmp_path / "synthetic-profile.json"
    profile.write_text(json.dumps(synthetic_profile()))
    env = isolated_environment(tmp_path, block_z3=True)
    binary = tmp_path / "bin"
    binary.mkdir()
    (binary / "git").symlink_to(shutil.which("git"))
    env["PATH"] = str(binary)
    report = blocked(tmp_path, ENTRIES[-1], ["--cost-profile", str(profile)], environment=env)
    assert "manifest" in report["error"].lower()
    assert "msprof" not in report["error"].lower()


@pytest.fixture
def reuse(monkeypatch):
    monkeypatch.syspath_prepend(str(RUNNERS))
    spec = importlib.util.spec_from_file_location("s6_reuse_contract", RUNNERS / "_s6_npu.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def paired_report(module):
    return {
        "repeat": 3,
        "cases": [
            {
                "name": name,
                "evidence": {
                    "case_parameters": {
                        **spec,
                        "rounds": [
                            {
                                "index": index,
                                "golden_status": "PASS",
                                "input_sha256": {"A": f"a{index}", "Scratch": f"scratch{index}"},
                                "output_sha256": f"output{index}",
                                "golden_sha256": f"golden{index}",
                            }
                            for index in range(3)
                        ],
                    },
                    "kernels": [
                        {
                            "resolved_flags": ["-O2"],
                            "resolved_pass_configs": {"tl.ascend_pipeline_planner": spec["strategy"]},
                        }
                    ],
                },
            }
            for name, spec in module.PARAMETERS.items()
        ],
    }


def test_every_changed_input_launch_is_compared_between_strategies(reuse):
    report = paired_report(reuse)
    reuse.compare_pairs(report)
    assert len(report["strategy_comparisons"]) == len(reuse.FIXTURES)
    assert all(item["compared_launches"] == 3 for item in report["strategy_comparisons"])


@pytest.mark.parametrize(
    "corruption",
    [
        "inputs",
        "output",
        "golden",
        "missing_round",
        "same_scratch",
        "same_a",
        "index",
        "strategy",
        "flags",
        "parameters",
    ],
)
def test_paired_evidence_rejects_changed_or_incomplete_records(reuse, corruption):
    report = copy.deepcopy(paired_report(reuse))
    left = report["cases"][0]["evidence"]
    right = report["cases"][1]["evidence"]
    round_ = right["case_parameters"]["rounds"][1]
    if corruption == "inputs":
        round_["input_sha256"]["A"] = "changed"
    elif corruption in ("output", "golden"):
        round_[corruption + "_sha256"] = "changed"
    elif corruption == "missing_round":
        right["case_parameters"]["rounds"].pop()
    elif corruption in ("same_scratch", "same_a"):
        for side in (left, right):
            for value in side["case_parameters"]["rounds"]:
                value["input_sha256"]["Scratch" if corruption == "same_scratch" else "A"] = "unchanged"
    elif corruption == "index":
        for side in (left, right):
            side["case_parameters"]["rounds"][1]["index"] = 0
    elif corruption == "strategy":
        right["kernels"][0]["resolved_pass_configs"]["tl.ascend_pipeline_planner"] = "legacy"
    elif corruption == "flags":
        right["kernels"][0]["resolved_flags"] = ["-O0"]
    else:
        right["case_parameters"]["tiles"] = 99
    with pytest.raises(AssertionError):
        reuse.compare_pairs(report)
