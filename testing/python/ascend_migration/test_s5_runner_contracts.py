"""Host-only checks of reproducible collection and independent result comparisons."""

import copy
import csv
import importlib.util
import json
from argparse import Namespace
from pathlib import Path
import shlex
from types import SimpleNamespace

import pytest


RUNNERS = Path(__file__).resolve().parents[3] / "scripts/ascend_migration"


@pytest.fixture
def modules(monkeypatch):
    monkeypatch.syspath_prepend(str(RUNNERS))

    def load(name):
        spec = importlib.util.spec_from_file_location("test_" + name, RUNNERS / (name + ".py"))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    return {name: load(name) for name in ("_s5_calibration_matrix", "_s5_performance", "s5_run_pipeline_npu")}


def test_controller_keeps_every_failed_attempt_sample(modules):
    controller = modules["_s5_calibration_matrix"]
    batches = [[{"round": index // 125, "index": index} for index in range(375)] for _ in range(2)]
    before = copy.deepcopy(batches)
    result = controller.merge_attempts(batches)
    assert batches == before
    assert len(result) == 750
    assert {item["round"] for item in result} == set(range(6))
    assert [item["index"] for item in result[:375]] == list(range(375))
    assert {item["attempt"] for item in result} == {0, 1}


@pytest.mark.parametrize("batches", [[], [[{}]], [[{}]] * 3])
def test_controller_refuses_incomplete_or_excess_attempts(modules, batches):
    with pytest.raises(ValueError):
        modules["_s5_calibration_matrix"].merge_attempts(batches)


def test_controller_profiler_command_has_explicit_no_pmu_policy(modules, tmp_path):
    command = modules["_s5_calibration_matrix"].profiler_command(["python", "/source with spaces/worker.py"], tmp_path)
    assert command[0:2] == ["msprof", "op"]
    assert "--aic-metrics=BasicInfo" in command
    assert "--launch-count=375" in command
    assert "--warm-up=0" in command and "--replay-mode=kernel" in command
    assert command[-1] == "--application=python '/source with spaces/worker.py'"


@pytest.mark.parametrize("combined_pass", [True, False])
def test_reproduction_controller_uses_all_attempts_before_export(modules, monkeypatch, tmp_path, combined_pass):
    """Exercise the public controller's whole matrix using explicit synthetic measurements."""
    from test_s5_calibration_export import _cases, _samples
    import _s5_calibration

    controller = modules["_s5_calibration_matrix"]
    original_module = controller._module
    monkeypatch.setattr(
        controller,
        "_module",
        lambda name, path: (
            SimpleNamespace(verify_manifest=lambda path: {"synthetic_host_fixture": True})
            if name == "s5_calibration_native_identity"
            else original_module(name, path)
        ),
    )
    monkeypatch.setattr(controller.shutil, "which", lambda name: "/synthetic/msprof")
    cases = _cases(_s5_calibration)
    index = {tuple(case["worker"]["parameters"][key] for key in ("operation", "dtype", "tile", "pattern")): case for case in cases}
    accuracy_seen, attempts = set(), {}

    def worker_key(command):
        return tuple(
            int(command[command.index(flag) + 1]) if flag == "--tile" else command[command.index(flag) + 1]
            for flag in ("--operation", "--dtype", "--tile", "--pattern")
        )

    def collect(command, directory, name, **kwargs):
        profiler = command[0] == "msprof"
        worker_command = shlex.split(command[-1].removeprefix("--application=")) if profiler else command
        key = worker_key(worker_command)
        worker = copy.deepcopy(index[key]["worker"])
        worker["golden_checks"] = 15
        if profiler:
            assert key in accuracy_seen
            attempts[key] = attempts.get(key, 0) + 1
            assert attempts[key] <= 2
            worker["synthetic_attempt"] = attempts[key] - 1
        else:
            accuracy_seen.add(key)
        output = Path(worker_command[worker_command.index("--output") + 1])
        output.write_text(json.dumps(worker))
        return {"command": command, "exit_code": 0, "status": "PASS"}

    first_key = next(iter(index))

    def measured(directory, worker):
        key = tuple(worker["parameters"][field] for field in ("operation", "dtype", "tile", "pattern"))
        slopes = (16, 16, 16)
        if key == first_key and worker["synthetic_attempt"] == 0:
            slopes = (12, 16, 20) if combined_pass else (40, 80, 120)
        return _samples(_s5_calibration, slopes)

    monkeypatch.setattr(controller, "run_process", collect)
    monkeypatch.setattr(controller, "read_profiler_samples", measured)
    args = Namespace(
        output=tmp_path / "report.json",
        calibrate_profile=tmp_path / "model.json",
        calibration_source_root=None,
        build_manifest=tmp_path / "synthetic-manifest.json",
        platform="a3",
        device=0,
        seed=0,
        timeout=120,
        calibration_timeout=180,
    )
    report = {"counts": {"passed": 0, "failed": 0, "skipped": 0, "total": 0}}
    code = controller.run_calibration(args, report)
    saved = json.loads(args.output.read_text())
    assert attempts[first_key] == 2
    if combined_pass:
        assert code == 0 and saved["status"] == "PASS"
        profile = json.loads(args.calibrate_profile.read_text())
        assert len(saved["cases"]) == 32 and profile["evidence"]["application_launches"] == 12375
        assert profile["evidence"]["raw_samples"][0]["attempts"][0]["fit"]["status"] == "BLOCKED"
        assert len(profile["evidence"]["raw_samples"][0]["samples"]) == 750
    else:
        assert code == 2 and saved["status"] == "BLOCKED"
        assert len(attempts) == 1 and not args.calibrate_profile.exists()


def _paired(entry):
    report = {
        "cases": [
            {
                "name": name,
                "evidence": {
                    "case_parameters": {
                        **spec,
                        "input_sha256": {"A": "a", "B": "b"},
                        "output_sha256": "o",
                        "golden_sha256": "g",
                    },
                    "kernels": [{"resolved_pass_configs": {"tl.ascend_pipeline_planner": spec["strategy"]}}],
                },
            }
            for name, spec in entry.PARAMETERS.items()
        ]
    }
    return json.loads(json.dumps(report))


def test_all_actual_strategy_results_are_compared(modules):
    entry = modules["s5_run_pipeline_npu"]
    report = _paired(entry)
    entry.final_check(report)
    assert len(report["strategy_comparisons"]) == len(entry.FIXTURES)


@pytest.mark.parametrize("field", ["input_sha256", "output_sha256", "golden_sha256", "strategy"])
def test_strategy_comparison_cannot_accept_changed_data_or_strategy(modules, field):
    entry = modules["s5_run_pipeline_npu"]
    report = _paired(entry)
    if field == "strategy":
        report["cases"][1]["evidence"]["kernels"][0]["resolved_pass_configs"]["tl.ascend_pipeline_planner"] = "legacy"
    else:
        report["cases"][1]["evidence"]["case_parameters"][field] = "changed"
    with pytest.raises(AssertionError):
        entry.final_check(report)


def _csv_run(tmp_path):
    root = tmp_path / "OPPROFfixture" / "main_kernel"
    journal = []
    for round_index in range(3):
        for phase, repeat in (("warmup", 5), ("measured", 20)):
            for _ in range(repeat):
                index = len(journal)
                journal.append({"index": index, "round": round_index, "phase": phase})
                folder = root / str(index)
                folder.mkdir(parents=True)
                row = {
                    "Op Name": "main_kernel",
                    "Device Id": "0",
                    "Task Duration(us)": "2.125",
                    "Current Freq": "1650",
                    "Rated Freq": "1650",
                }
                with (folder / "OpBasicInfo.csv").open("w", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=list(row))
                    writer.writeheader()
                    writer.writerow(row)
    return {
        "status": "PASS",
        "launches": journal,
        "golden_checks": 3,
        "actual_device": {"logical_device": 0},
    }


def test_performance_attributes_every_real_invocation_once(modules, tmp_path):
    worker = _csv_run(tmp_path)
    module = modules["_s5_performance"]
    samples = module.read_execution_samples(tmp_path, worker)
    assert len(samples) == 75 and sum(item["phase"] == "measured" for item in samples) == 60
    assert {item["duration_ns"] for item in samples} == {2125}
    assert module._summary(samples)["median_ns"] == "2125"


def test_performance_io_failure_stays_on_atomic_final_report_path(modules, monkeypatch):
    performance = modules["_s5_performance"]

    def missing_artifact(*args):
        raise OSError("synthetic missing collector artifact")

    monkeypatch.setattr(performance, "_collect_performance", missing_artifact)
    with pytest.raises(ValueError, match="Cannot collect performance evidence"):
        performance.collect_performance({}, None, None, ())


@pytest.mark.parametrize("corruption", ["journal", "missing_csv", "extra_pmu", "clock"])
def test_performance_rejects_ambiguous_or_inconsistent_evidence(modules, tmp_path, corruption):
    worker = _csv_run(tmp_path)
    folder = tmp_path / "OPPROFfixture/main_kernel/0"
    if corruption == "journal":
        worker["launches"][0]["phase"] = "measured"
    elif corruption == "missing_csv":
        (folder / "OpBasicInfo.csv").unlink()
    elif corruption == "extra_pmu":
        (folder / "PipeUtilization.csv").write_text("unexpected\n")
    else:
        path = folder / "OpBasicInfo.csv"
        text = path.read_text().replace("1650,1650", "1500,1650")
        path.write_text(text)
    with pytest.raises(ValueError):
        modules["_s5_performance"].read_execution_samples(tmp_path, worker)
