"""Host-only checks for S4 CLI isolation and actual-output comparison failures."""

import copy
import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from test_s2_entrypoints import RUNNERS, isolated_environment

ENTRIES = ("s4_check_pipeline_ir", "s4_check_sync_coverage", "s4_run_vector_npu")


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
def test_s4_help_does_not_import_runtime_or_solver(tmp_path, entry):
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
def test_s4_missing_manifest_prevents_runtime_load(tmp_path, entry):
    output = tmp_path / "missing.json"
    env = isolated_environment(tmp_path, block_z3=True)
    env["TL_ASCEND_MIGRATION_BUILD_MANIFEST"] = str(tmp_path / "absent.json")
    result = subprocess.run(command(entry, output), env=env, text=True, capture_output=True, timeout=30)
    assert result.returncode == 2, result.stdout + result.stderr
    report = json.loads(output.read_text())
    assert report["step"] == "S4"
    assert report["status"] == "BLOCKED"
    assert report["counts"]["passed"] == 0


@pytest.mark.parametrize("entry", ENTRIES)
def test_s4_optimized_python_is_blocked(tmp_path, entry):
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
    assert json.loads(output.read_text())["status"] == "BLOCKED"


@pytest.mark.parametrize("entry", ENTRIES)
@pytest.mark.parametrize("value", ["nan", "inf", "-1"])
def test_s4_timeout_is_finite_and_positive(tmp_path, entry, value):
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


@pytest.mark.parametrize("flag,value", [("--repeat", "0"), ("--device", "-1"), ("--seed", "-1")])
def test_s4_device_parameters_are_validated_before_runtime_load(tmp_path, flag, value):
    output = tmp_path / "invalid.json"
    result = subprocess.run(
        [*command("s4_run_vector_npu", output), flag, value],
        env=isolated_environment(tmp_path, block_z3=True),
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert result.returncode == 2, result.stdout + result.stderr
    assert json.loads(output.read_text())["status"] == "BLOCKED"


def _npu_entry(monkeypatch):
    monkeypatch.syspath_prepend(str(RUNNERS))
    spec = importlib.util.spec_from_file_location("s4_npu_entry", RUNNERS / "s4_run_vector_npu.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _paired_results(module):
    return {
        "cases": [
            {
                "name": name,
                "evidence": {
                    "case_parameters": {
                        **module.parameters(name),
                        "input_sha256": {"A": "a" * 64, "B": "b" * 64},
                        "output_sha256": "c" * 64,
                    },
                    "kernels": [{"resolved_pass_configs": {"tl.ascend_pipeline_planner": strategy}}],
                },
            }
            for name, (_, _, strategy) in module.PARAMETERS.items()
        ]
    }


def test_s4_compares_all_four_actual_output_pairs(monkeypatch):
    entry = _npu_entry(monkeypatch)
    report = _paired_results(entry)
    entry.compare_outputs(report)
    assert len(report["strategy_comparisons"]) == 4
    assert all(pair["status"] == "PASS" for pair in report["strategy_comparisons"])


@pytest.mark.parametrize("corruption", ["input", "output", "strategy", "missing_case"])
def test_s4_comparison_rejects_different_or_mislabeled_results(monkeypatch, corruption):
    entry = _npu_entry(monkeypatch)
    report = copy.deepcopy(_paired_results(entry))
    constraint = report["cases"][1]["evidence"]
    if corruption == "input":
        constraint["case_parameters"]["input_sha256"]["A"] = "d" * 64
    elif corruption == "output":
        constraint["case_parameters"]["output_sha256"] = "d" * 64
    elif corruption == "strategy":
        constraint["kernels"][0]["resolved_pass_configs"]["tl.ascend_pipeline_planner"] = "legacy"
    else:
        report["cases"].pop()
    with pytest.raises((AssertionError, KeyError)):
        entry.compare_outputs(report)


def test_s4_final_comparison_failure_cannot_leave_a_pass_report(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(RUNNERS))
    import _npu_runner

    args = argparse.Namespace(
        output=tmp_path / "comparison.json",
        platform="a3",
        device=0,
        seed=0,
        repeat=1,
        timeout=1,
    )

    def successful_worker(command, *args, **kwargs):
        output = Path(command[command.index("--output") + 1])
        output.write_text(json.dumps({"status": "PASS"}))
        return {"command": command, "status": "PASS", "exit_code": 0}

    def reject_outputs(report):
        raise AssertionError("injected actual output mismatch")

    monkeypatch.setattr(_npu_runner, "run_process", successful_worker)
    report = {"counts": {"passed": 0, "failed": 0, "skipped": 0, "total": 0}}
    result = _npu_runner.run_cases(args, report, ("one",), Path(__file__), final_check=reject_outputs)
    final = json.loads(args.output.read_text())
    assert result == 1
    assert final["status"] == "FAIL"
    assert "injected actual output mismatch" in final["error"]


@pytest.mark.parametrize("entry", ENTRIES)
@pytest.mark.parametrize("failure", ["missing", "broken"])
def test_s4_solver_preflight_blocks_before_any_runtime_import(tmp_path, entry, failure):
    """Exercise real subprocess import failures after an isolated identity-check stub."""
    imports = tmp_path / "imports.txt"
    manifest = {
        "tvm_source_root": str(tmp_path / "tvm-source"),
        "libraries": [{"path": str(tmp_path / "build" / name)} for name in ("libtilelang_module.so", "libtvm.so", "libtvm_runtime.so")],
    }
    # The test never loads any library. Supplying a manifest stub reaches the
    # dependency preflight without borrowing unrelated native build artifacts.
    (tmp_path / "sitecustomize.py").write_text(
        "import sys, types\n"
        "from pathlib import Path\n"
        "native = types.ModuleType('_native_build')\n"
        f"native.verify_manifest = lambda path: {manifest!r}\n"
        "sys.modules['_native_build'] = native\n"
        "class Guard:\n"
        "    def find_spec(self, fullname, path=None, target=None):\n"
        "        root = fullname.split('.')[0]\n"
        "        if root in {'z3', 'tilelang', 'tvm', 'torch', 'torch_npu', 'acl'}:\n"
        f"            with Path({str(imports)!r}).open('a') as stream:\n"
        "                stream.write(fullname + '\\n')\n"
        "            if root != 'z3':\n"
        "                raise AssertionError('runtime import before dependency preflight')\n"
        f"            if {failure!r} == 'missing':\n"
        "                raise ModuleNotFoundError('No module named z3', name='z3')\n"
        "            raise ImportError('installed z3 import is broken')\n"
        "sys.meta_path.insert(0, Guard())\n"
    )
    output = tmp_path / "solver.json"
    result = subprocess.run(
        command(entry, output),
        env=dict(os.environ, PYTHONPATH=str(tmp_path), PYTEST_DISABLE_PLUGIN_AUTOLOAD="1"),
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert result.returncode == 2, result.stdout + result.stderr
    report = json.loads(output.read_text())
    assert report["status"] == "BLOCKED"
    assert report["counts"]["passed"] == 0
    requested = imports.read_text().splitlines()
    assert requested and all(name.split(".")[0] == "z3" for name in requested)
