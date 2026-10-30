"""Synthetic host-only tests for calibration attribution and rejection rules."""

import csv
import importlib.util
from pathlib import Path

import pytest

RUNNERS = Path(__file__).resolve().parents[3] / "scripts/ascend_migration"


@pytest.fixture
def calibration(monkeypatch):
    monkeypatch.syspath_prepend(str(RUNNERS))
    spec = importlib.util.spec_from_file_location("s5_calibration_test", RUNNERS / "_s5_calibration.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _samples(module, slopes):
    return [
        {
            "round": round_index,
            "count": count,
            "phase": "measured",
            "duration_ns": 5000 + count * slope + sample % 3,
        }
        for round_index, slope in enumerate(slopes)
        for count in module.COUNTS
        for sample in range(20)
    ]


def test_synthetic_count_slope_keeps_conservative_upper_bound(calibration):
    result = calibration.fit_effective_cost(_samples(calibration, [16, 16, 16]), bootstrap=400)
    assert result["status"] == "PASS", result
    assert result["conservative_effective_ns"] >= 16
    assert result["measured_sample_count"] == 300


def test_zero_signal_cannot_become_a_unit_cost(calibration):
    result = calibration.fit_effective_cost(_samples(calibration, [0, 0, 0]), bootstrap=400)
    assert result["status"] == "BLOCKED"
    assert "slope confidence interval includes zero" in result["reasons"]


def test_round_drift_is_part_of_confidence_not_hidden_by_identical_samples(calibration):
    result = calibration.fit_effective_cost(_samples(calibration, [1, 25, 50]), bootstrap=400)
    assert result["status"] == "BLOCKED"
    assert float(result["relative_confidence_half_width"]) > 0.20


def test_missing_operation_count_is_rejected(calibration):
    samples = _samples(calibration, [16, 16, 16])
    with pytest.raises(ValueError, match="five operation counts"):
        calibration.fit_effective_cost([sample for sample in samples if sample["count"] != 512])


def _reports(tmp_path, module):
    worker = {
        "status": "PASS",
        "parameters": {"mode": "profile", "rounds": 1, "warmup": 1, "samples": 2},
        "actual_device": {"logical_device": 0},
        "count_values": list(module.COUNTS),
        "launches": [],
    }
    for count in module.COUNTS:
        for phase, repeat in (("warmup", 1), ("measured", 2)):
            for _ in range(repeat):
                index = len(worker["launches"])
                worker["launches"].append({"index": index, "round": 0, "count": count, "phase": phase})
                directory = tmp_path / "OPPROF_synthetic/main_kernel" / str(index)
                directory.mkdir(parents=True)
                with (directory / "OpBasicInfo_fixture.csv").open("w", newline="") as stream:
                    writer = csv.writer(stream)
                    writer.writerow(["Op Name", "Task Duration(us)", "Device Id", "Current Freq", "Rated Freq"])
                    writer.writerow(["main_kernel", "6.000001", 0, 1650, 1650])
    return worker


def test_csv_invocation_ids_match_the_entire_declared_journal(tmp_path, calibration):
    worker = _reports(tmp_path, calibration)
    result = calibration.read_profiler_samples(tmp_path, worker)
    assert len(result) == 15
    assert result[0]["duration_ns"] == 6001
    assert result[-1]["count"] == 512


@pytest.mark.parametrize(
    "corruption",
    ["missing", "duplicate", "wrong_device", "wrong_clock", "wrong_policy", "extra_metrics"],
)
def test_corrupt_or_ambiguous_profiler_evidence_is_rejected(tmp_path, calibration, corruption):
    worker = _reports(tmp_path, calibration)
    root = tmp_path / "OPPROF_synthetic/main_kernel"
    first = root / "0/OpBasicInfo_fixture.csv"
    if corruption == "missing":
        (root / "14/OpBasicInfo_fixture.csv").unlink()
    elif corruption == "duplicate":
        (root / "0/OpBasicInfo_duplicate.csv").write_text(first.read_text())
    elif corruption == "wrong_device":
        first.write_text(first.read_text().replace("6.000001,0,", "6.000001,1,"))
    elif corruption == "wrong_clock":
        first.write_text(first.read_text().replace("1650,1650", "1500,1650"))
    elif corruption == "wrong_policy":
        worker["parameters"]["samples"] = 3
    else:
        (root / "0/PipeUtilization_unexpected.csv").write_text("unexpected PMU data\n")
    with pytest.raises(ValueError):
        calibration.read_profiler_samples(tmp_path, worker)
