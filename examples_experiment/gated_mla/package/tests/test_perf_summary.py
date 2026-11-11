"""CPU-only summary-contract tests; synthetic timings are not NPU measurements."""

import copy
import statistics

import pytest

from ._perf_tools import load_perf_tool


_perf_tool = load_perf_tool("run_paired_suite")
summarize = _perf_tool.summarize


METRICS = ("eager_event_us", "eager_wall_us", "graph_per_call_us")


def trial(index, candidate_samples, reference_samples):
    """Use complete independent-trial records, with distinct acquisition orders."""
    report = {
        "case": "decode17",
        "dtype": "float16",
        "target": "pto",
        "status": "balanced_benchmark_completed",
        "trial_id": index,
        "order_seed": 1700 + index,
        "shape": {"B": 1, "H": 96, "Tq": 1, "Tk": 4096},
        "visible": 17,
        "qualification_seeds": [0, 1, 17],
        "timed_seed": 0,
        "timed_inputs": {"q_content": {"shape": [1, 1, 96, 128], "dtype": "torch.float16", "sha256": "input-bytes"}},
        "timed_metadata": {"shape": [1, 8], "dtype": "torch.int32", "sha256": "metadata-bytes"},
        "budget": {"rtol": 0.01, "atol": 0.002, "max_normalized": 0.005},
        "environment": {"torch": "2.7.1", "tilelang_file": "/built/tilelang/__init__.py", "device": "Ascend910B3"},
        "package_sources": {"runtime.py": "candidate-hash"},
        "gate": {"mode": "isolated_original_kernel", "source_sha256": "gate-hash"},
        "adapter_sha256": "adapter-hash",
        "baseline_sources": {"sources": {"adapter_decode.py": "adapter-hash"}},
        "protocol_manifest_sha256": "protocol-hash",
        "paired_timing_sha256": "timing-hash",
        "harness_sha256": "driver-hash",
        "graph_batch": 50,
        "rounds": 6,
        "repeats_per_round": 10,
        "graph_repeats_per_round": 5,
        "measurement": {
            "config": {"warmup": 10, "graph_warmup_replays": 5, "time_unqualified": False, "order_seed": 1700 + index},
            "methods": {},
            "paired_ratios": {},
        },
    }
    for name, values in (("candidate", candidate_samples), ("source_decode", reference_samples)):
        report["measurement"]["methods"][name] = {
            "metrics": {
                metric: {"median": statistics.median(values), "p95": max(values), "count": len(values), "raw_samples": values[:]}
                for metric in METRICS
            }
        }
    for metric in METRICS:
        report["measurement"]["paired_ratios"][metric] = {
            "source_decode": {
                "qualified": True,
                "ratio_of_medians": statistics.median(reference_samples) / statistics.median(candidate_samples),
            }
        }
    return report


def trials():
    return [
        trial(0, [1, 10, 1000], [19, 20, 21]),
        trial(1, [99, 100, 101], [149, 150, 151]),
        trial(2, [29, 30, 31], [119, 120, 121]),
    ]


def row_for(rows, metric="eager_event_us", method="source_decode", **group):
    matches = [row for row in rows if row["metric"] == metric and row["method"] == method and all(row[k] == v for k, v in group.items())]
    assert len(matches) == 1
    return matches[0]


def test_trial_medians_ranges_and_ratios_are_not_pooled():
    reports = trials()
    before = copy.deepcopy(reports)
    rows = summarize(reports)
    candidate = row_for(rows, method="candidate")
    reference = row_for(rows)
    assert candidate["trial_medians_us"] == [10, 100, 30]
    assert candidate["median_of_trial_medians_us"] == 30
    assert candidate["range_of_trial_medians_us"] == [10, 100]
    assert candidate["trial_p95_us"] == [1000, 101, 31]
    pooled = [x for report in reports for x in report["measurement"]["methods"]["candidate"]["metrics"]["eager_event_us"]["raw_samples"]]
    assert statistics.median(pooled) == 31
    assert reference["trial_ratios"] == [2, 1.5, 4]
    assert reference["reference_over_candidate_median"] == 2
    assert reference["reference_over_candidate_median"] != reference["median_of_trial_medians_us"] / candidate["median_of_trial_medians_us"]
    assert reference["trials"] == reference["measured_trials"] == 3
    assert candidate["qualified_ratio"] is False
    assert candidate["reference_over_candidate_median"] is None
    assert reports == before


@pytest.mark.parametrize("bad_trial", [0, 1, 2])
@pytest.mark.parametrize("metric", METRICS)
def test_one_unqualified_trial_blocks_only_its_metric_ratio(bad_trial, metric):
    reports = trials()
    # Leave the numeric ratio deliberately populated: stale numbers must not grant qualification.
    reports[bad_trial]["measurement"]["paired_ratios"][metric]["source_decode"]["qualified"] = False
    rows = summarize(reports)
    row = row_for(rows, metric=metric)
    assert row["qualified_ratio"] is False
    assert row["reference_over_candidate_median"] is None
    assert row["trial_ratios"] == []
    assert row["measured_trials"] == 3
    for other_metric in set(METRICS) - {metric}:
        assert row_for(rows, metric=other_metric)["qualified_ratio"] is True


def test_failed_trial_never_contributes_a_qualified_ratio():
    reports = trials()
    reports[1]["status"] = "failed"
    for row in summarize(reports):
        assert row["qualified_ratio"] is False
        assert row["reference_over_candidate_median"] is None
        assert row["trial_ratios"] == []


def test_missing_ratio_record_does_not_promote_remaining_trials():
    reports = trials()
    del reports[1]["measurement"]["paired_ratios"]["graph_per_call_us"]["source_decode"]
    rows = summarize(reports)
    assert row_for(rows, metric="graph_per_call_us")["qualified_ratio"] is False
    assert row_for(rows)["qualified_ratio"] is True


@pytest.mark.parametrize(
    ("field", "changed"),
    [
        ("package_sources", {"runtime.py": "different-source"}),
        ("environment", {"torch": "different-runtime"}),
        ("baseline_sources", {"sources": {"adapter_decode.py": "different-baseline"}}),
        ("gate", {"mode": "torch_rounding"}),
        ("paired_timing_sha256", "different-timing"),
        ("harness_sha256", "different-driver"),
        ("protocol_manifest_sha256", "different-protocol"),
        ("qualification_seeds", [0, 1, 18]),
        ("timed_seed", 1),
        ("timed_inputs", {"q_content": {"shape": [1, 1, 96, 128], "dtype": "torch.float16", "sha256": "different-input"}}),
        ("timed_metadata", {"shape": [1, 8], "dtype": "torch.int32", "sha256": "different-metadata"}),
        ("graph_batch", 5),
        ("rounds", 12),
        ("repeats_per_round", 20),
        ("budget", {"rtol": 0.1, "atol": 0.002, "max_normalized": 0.005}),
    ],
)
def test_different_source_runtime_or_measurement_identity_is_rejected(field, changed):
    reports = trials()
    reports[1][field] = changed
    with pytest.raises(ValueError, match="identit"):
        summarize(reports)


@pytest.mark.parametrize(("key", "value"), [("warmup", 20), ("graph_warmup_replays", 10), ("time_unqualified", True)])
def test_different_engine_configuration_is_rejected(key, value):
    reports = trials()
    reports[1]["measurement"]["config"][key] = value
    with pytest.raises(ValueError, match="identit"):
        summarize(reports)


def test_distinct_backend_groups_are_not_accidentally_combined():
    reports = trials()
    ascend = copy.deepcopy(reports)
    for report in ascend:
        report["target"] = "ascendc"
        report["environment"]["tilelang_file"] = "/another/built/tilelang/__init__.py"
    rows = summarize(reports + ascend)
    assert len(rows) == 12
    for target in ("pto", "ascendc"):
        row = row_for(rows, target=target)
        assert row["trials"] == 3
        assert row["reference_over_candidate_median"] == 2


def test_missing_graph_measurement_stays_missing_not_zero():
    reports = trials()
    for report in reports:
        report["measurement"]["methods"]["source_decode"]["metrics"]["graph_per_call_us"] = {"median": None, "p95": None}
        report["measurement"]["paired_ratios"]["graph_per_call_us"]["source_decode"] = {"qualified": False, "ratio_of_medians": None}
    row = row_for(summarize(reports), metric="graph_per_call_us")
    assert row["measured_trials"] == 0
    assert row["trial_medians_us"] == []
    assert row["median_of_trial_medians_us"] is None
    assert row["range_of_trial_medians_us"] is None
    assert row["reference_over_candidate_median"] is None
