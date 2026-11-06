"""Synthetic host-only coverage of bounded-attempt merging and canonical export."""

import importlib.util
from pathlib import Path

import pytest

RUNNERS = Path(__file__).resolve().parents[3] / "scripts/ascend_migration"


@pytest.fixture
def calibration(monkeypatch):
    monkeypatch.syspath_prepend(str(RUNNERS))
    spec = importlib.util.spec_from_file_location("s5_export_test", RUNNERS / "_s5_calibration.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _samples(module, slopes, attempt=0):
    result = []
    for local_round, slope in enumerate(slopes):
        for count in module.COUNTS:
            for phase, repeat in (("warmup", 5), ("measured", 20)):
                for _ in range(repeat):
                    ns = 5000 + slope * count
                    result.append(
                        {
                            "index": len(result),
                            "attempt": attempt,
                            "round_within_attempt": local_round,
                            "round": local_round + 3 * attempt,
                            "count": count,
                            "phase": phase,
                            "duration_ns": ns,
                            "duration_us_raw": f"{ns // 1000}.{ns % 1000:03d}",
                            "current_freq_mhz": 1650,
                            "rated_freq_mhz": 1650,
                        }
                    )
    return result


def _cases(module):
    cases = []
    for operation in ("gmub", "ubgm", "fill", "add"):
        for dtype in ("float16", "float32"):
            for tile in (32, 64):
                for pattern in ("dependent", "independent"):
                    samples = _samples(module, (16, 16, 16))
                    worker = {
                        "status": "PASS",
                        "parameters": {
                            "operation": operation,
                            "dtype": dtype,
                            "tile": tile,
                            "pattern": pattern,
                            "warmup": 5,
                            "samples": 20,
                            "rounds": 3,
                        },
                        "actual_device": {"soc": "SYNTHETIC_PROTOCOL_FIXTURE"},
                        "toolchain": {
                            "cann_version_info": "Version=synthetic-sdk\n",
                            "sha256": "a" * 64,
                        },
                        "kernel": {"resolved_flags": ["-O2"], "source": {"sha256": "b" * 64}},
                        "compilations": [{"source": {"sha256": "c" * 64}}],
                        "native_build": {
                            "project_source": {"digest": "d" * 64},
                            "libraries": [{"path": "/synthetic/lib.so", "sha256": "e" * 64}],
                        },
                        "count_values": list(module.COUNTS),
                        "independent_batch_tiles": 128,
                        "harness_source": {"sha256": "f" * 64},
                        "fixture_source": {"sha256": "a" * 64},
                    }
                    cases.append(
                        {
                            "worker": worker,
                            "samples": samples,
                            "fit": {
                                "status": "PASS",
                                "conservative_effective_ns": 16,
                                "method": "synthetic export protocol fixture",
                            },
                            "command": ["SYNTHETIC_DO_NOT_EXECUTE"],
                            "worker_report": {
                                "path": "/synthetic/profile.json",
                                "sha256": "b" * 64,
                            },
                            "accuracy_report": {
                                "path": "/synthetic/accuracy.json",
                                "sha256": "c" * 64,
                            },
                        }
                    )
    return cases


def _two_attempts(module, case, first_slopes):
    first = _samples(module, first_slopes)
    second = _samples(module, (16, 16, 16), attempt=1)
    first_fit = module.fit_effective_cost(first, bootstrap=400)
    second_fit = module.fit_effective_cost(second, bootstrap=400)
    case["samples"] = [*first, *second]
    case["fit"] = module.fit_effective_cost(case["samples"], bootstrap=400)
    case["attempts"] = [
        {"worker_report": case["worker_report"], "fit": first_fit},
        {
            "worker_report": {"path": "/synthetic/repeat-1.json", "sha256": "d" * 64},
            "fit": second_fit,
        },
    ]
    case["worker_reports"] = [attempt["worker_report"] for attempt in case["attempts"]]
    case["collector_commands"] = [case["command"], ["SYNTHETIC_REPEAT_DO_NOT_EXECUTE"]]
    return first_fit, second_fit


def test_second_attempt_pass_does_not_allow_failed_combined_export(calibration):
    cases = _cases(calibration)
    first, second = _two_attempts(calibration, cases[0], (40, 80, 120))
    assert first["status"] == "BLOCKED"
    assert second["status"] == "PASS"
    assert cases[0]["fit"]["status"] == "BLOCKED"
    with pytest.raises(ValueError, match="statistical gates"):
        calibration.assemble_profile(cases)


def test_all_attempt_samples_and_initial_failure_are_canonical_payload(calibration):
    cases = _cases(calibration)
    first, second = _two_attempts(calibration, cases[0], (12, 16, 20))
    assert first["status"] == "BLOCKED"
    assert second["status"] == cases[0]["fit"]["status"] == "PASS"
    profile = calibration.assemble_profile(cases)
    evidence = profile["evidence"]
    raw = evidence["raw_samples"][0]
    assert len(raw["samples"]) == 750 and raw["attempt_count"] == 2
    assert {sample["round"] for sample in raw["samples"]} == set(range(6))
    assert {sample["attempt"] for sample in raw["samples"]} == {0, 1}
    assert raw["attempts"][0]["fit"]["status"] == "BLOCKED"
    assert len(raw["worker_reports"]) == len(raw["collector_commands"]) == 2
    assert evidence["application_launches"] == 12375
    assert evidence["measured_samples"] == 9900
    assert len(profile["records"]) == 16
    before = profile["profile_sha256"]
    cases[0]["samples"][0]["duration_ns"] += 1
    assert calibration.assemble_profile(cases)["profile_sha256"] != before


def test_more_than_two_attempts_is_rejected(calibration):
    cases = _cases(calibration)
    cases[0]["attempts"] = [{}, {}, {}]
    cases[0]["samples"] *= 3
    with pytest.raises(ValueError, match="at most two"):
        calibration.assemble_profile(cases)
