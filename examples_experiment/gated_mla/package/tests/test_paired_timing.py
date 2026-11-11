"""CPU-only mechanism tests; these are not NPU benchmark qualification."""

from collections import Counter
from contextlib import contextmanager
import copy
import json
from pathlib import Path
import statistics
import subprocess
import sys
import unittest

from ._perf_tools import load_perf_tool


_perf_tool = load_perf_tool("paired_timing")
MeasurementFailure = _perf_tool.MeasurementFailure
balanced_orders = _perf_tool.balanced_orders
run_measurements = _perf_tool.run_measurements
summarize = _perf_tool.summarize


NAMES = ("candidate", "historical_fp32", "source_decode")


class FakeEvent:
    def __init__(self, backend):
        self.backend = backend

    def record(self):
        self.at = self.backend.device_us

    def synchronize(self):
        self.backend.host_us += 5

    def elapsed_time(self, other):
        return (other.at - self.at) / 1000


class FakeGraph:
    def __init__(self, backend):
        self.backend, self.steps, self.replays = backend, [], 0

    def replay(self):
        self.replays += 1
        name = self.steps[0]
        if self.backend.replay_error == name and self.replays == 6:
            raise RuntimeError("graph replay unsupported")
        for step in self.steps:
            self.backend.work(step, replay=True)


class FakeBackend:
    def __init__(self, capture_error=None, replay_error=None):
        self.device_us = self.host_us = 0
        self.capturing = None
        self.capture_error, self.replay_error = capture_error or {}, replay_error
        self.calls, self.graphs = [], []
        self.event_cost = dict(zip(NAMES, (10, 20, 30)))
        self.host_cost = dict(zip(NAMES, (20, 30, 40)))

    def clock(self):
        return self.host_us / 1e6

    def Event(self, *, enable_timing):
        assert enable_timing
        return FakeEvent(self)

    def synchronize(self):
        self.host_us += 5

    def NPUGraph(self):
        graph = FakeGraph(self)
        self.graphs.append(graph)
        return graph

    @contextmanager
    def graph(self, graph):
        assert self.capturing is None
        self.capturing = graph
        try:
            yield
        finally:
            self.capturing = None

    def work(self, name, replay=False):
        if self.capturing is not None:
            if name in self.capture_error:
                raise self.capture_error[name]
            self.capturing.steps.append(name)
        self.calls.append((name, "capture" if self.capturing else "replay" if replay else "call"))
        self.device_us += self.event_cost[name]
        self.host_us += self.host_cost[name]
        return {"method": name}

    def methods(self):
        return {name: lambda name=name: self.work(name) for name in NAMES}


def known_graph_policy(error):
    return {"continue": "graph" in str(error) and "unsupported" in str(error), "category": "test_policy"}


class PairedTimingTests(unittest.TestCase):
    def run_fake(self, backend=None, *, qualification=None, check=None, **kwargs):
        backend = backend or FakeBackend()
        params = dict(warmup=1, rounds=6, repeats=2, graph_batch=4, graph_repeats=2, order_seed=17)
        params.update(kwargs)
        result = run_measurements(
            backend.methods(),
            qualified=qualification or dict.fromkeys(NAMES, True),
            check=check or (lambda name, output, boundary: output["method"] == name),
            backend=backend,
            clock=backend.clock,
            **params,
        )
        return result, backend

    def test_import_and_fake_run_do_not_import_torch(self):
        code = r"""
import importlib.util
from pathlib import Path
import sys
import types

actual_test = Path(sys.argv[1]).resolve()
original_path = list(sys.path)
# Existing unrelated modules must not be replaced by the GMLA test helper.
names = ("paired_timing", "paired_benchmark", "host_decode_diagnostic", "run_paired_suite")
sentinels = {name: types.ModuleType(name) for name in names}
sys.modules.update(sentinels)
namespace = "_gmla_perf_lazy_probe"
package = types.ModuleType(namespace)
package.__path__ = [str(actual_test.parent)]
sys.modules[namespace] = package
spec = importlib.util.spec_from_file_location(namespace + ".timing_test", actual_test)
test_module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = test_module
spec.loader.exec_module(test_module)
assert Path(test_module.__file__).resolve() == actual_test
for name in names:
    tool = test_module.load_perf_tool(name)
    assert Path(tool.__file__).resolve() == actual_test.parents[1] / "benchmark" / (name + ".py")
    assert sys.modules[name] is sentinels[name]
test_module.PairedTimingTests().run_fake()
assert sys.path == original_path
assert "torch" not in sys.modules
assert "torch_npu" not in sys.modules
"""
        result = subprocess.run([sys.executable, "-c", code, str(Path(__file__).resolve())], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_balanced_positions_and_pair_order(self):
        orders = balanced_orders(NAMES, 12, 17)
        self.assertEqual(orders, balanced_orders(NAMES, 12, 17))
        self.assertNotEqual(orders, balanced_orders(NAMES, 12, 18))
        for begin in (0, 6):
            cycle = orders[begin : begin + 6]
            self.assertEqual(len(set(cycle)), 6)
            for position in range(3):
                self.assertEqual(Counter(row[position] for row in cycle), dict.fromkeys(NAMES, 2))
            for first in NAMES:
                for second in NAMES:
                    if first != second:
                        self.assertEqual(sum(row.index(first) < row.index(second) for row in cycle), 3)
        with self.assertRaises(ValueError):
            balanced_orders(NAMES, 7, 1)

    def test_statistics_have_explicit_nearest_rank_and_sample_stdev(self):
        values = list(range(1, 21))
        record = summarize(values)
        self.assertEqual((record["median"], record["p90"], record["p95"]), (10.5, 18, 19))
        self.assertEqual(record["stdev"], statistics.stdev(values))
        self.assertEqual(record["raw_samples"], values)
        self.assertEqual(summarize([4])["stdev"], 0)
        self.assertIsNone(summarize([])["median"])
        with self.assertRaises(ValueError):
            summarize([float("nan")])

    def test_sample_counts_timing_boundaries_capture_once_and_ratios(self):
        backend = FakeBackend()
        checks = []

        def check(name, output, boundary):
            checks.append((name, boundary))
            # Deliberately huge checking cost must not enter any timing sample.
            backend.device_us += 1_000_000
            backend.host_us += 1_000_000
            return {"qualified": output["method"] == name, "read_only": True}

        result, _ = self.run_fake(backend, check=check)
        self.assertEqual(result["status"], "completed_with_explicit_qualification")
        self.assertEqual(len(backend.graphs), 3)
        self.assertEqual([len(g.steps) for g in backend.graphs], [4, 4, 4])
        self.assertEqual([g.replays for g in backend.graphs], [17, 17, 17])  # five warmups +12 measurements
        self.assertEqual(len(checks), 36)
        for name in NAMES:
            metrics = result["methods"][name]["metrics"]
            self.assertEqual([metrics[key]["count"] for key in metrics], [12, 12, 12])
            self.assertAlmostEqual(metrics["eager_event_us"]["median"], backend.event_cost[name])
            self.assertAlmostEqual(metrics["eager_wall_us"]["median"], backend.host_cost[name] + 5)
            self.assertAlmostEqual(metrics["graph_per_call_us"]["median"], backend.event_cost[name])
        ratio = result["paired_ratios"]["eager_event_us"]["historical_fp32"]
        self.assertTrue(ratio["qualified"])
        self.assertEqual(ratio["summary"]["median"], 2)
        self.assertEqual(len(ratio["paired_samples"]), 12)
        self.assertAlmostEqual(result["paired_ratios"]["eager_wall_us"]["historical_fp32"]["summary"]["median"], 35 / 25)
        json.dumps(result)  # report must contain no event/graph/tensor objects

    def test_unqualified_method_not_called_and_projected_order_balanced(self):
        qualification = dict.fromkeys(NAMES, True)
        qualification["source_decode"] = False
        result, backend = self.run_fake(qualification=qualification)
        self.assertFalse(any(name == "source_decode" for name, _ in backend.calls))
        actual = [tuple(row["order"]) for row in result["rounds"]["eager"]]
        self.assertEqual(Counter(actual), {("candidate", "historical_fp32"): 3, ("historical_fp32", "candidate"): 3})
        self.assertIsNone(result["paired_ratios"]["eager_event_us"]["source_decode"]["ratio_of_medians"])

    def test_diagnostic_method_never_earns_a_ratio(self):
        qualification = dict.fromkeys(NAMES, True)
        qualification["source_decode"] = False
        result, _ = self.run_fake(
            qualification=qualification,
            time_unqualified=True,
            check=lambda name, output, boundary: name != "source_decode",
        )
        self.assertEqual(result["methods"]["source_decode"]["metrics"]["eager_event_us"]["count"], 12)
        for metric in result["paired_ratios"]:
            self.assertFalse(result["paired_ratios"][metric]["source_decode"]["qualified"])
            self.assertIsNone(result["paired_ratios"][metric]["source_decode"]["ratio_of_medians"])

    def test_callback_cannot_promote_initially_unqualified_method(self):
        qualification = dict.fromkeys(NAMES, True)
        qualification["source_decode"] = False

        def check(name, output, boundary):
            qualification["source_decode"] = True
            return True

        result, _ = self.run_fake(qualification=qualification, check=check, time_unqualified=True)
        for metric in result["paired_ratios"]:
            self.assertFalse(result["paired_ratios"][metric]["source_decode"]["qualified"])

    def test_qualified_postcheck_failure_aborts_and_keeps_samples(self):
        backend = FakeBackend()
        with self.assertRaises(MeasurementFailure) as caught:
            self.run_fake(backend, check=lambda name, output, boundary: False)
        report = caught.exception.partial_report
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["methods"]["candidate"]["metrics"]["eager_event_us"]["count"], 2)
        self.assertEqual(backend.graphs, [])

    def test_readonly_failure_is_fatal_even_for_diagnostic_method(self):
        qualification = dict.fromkeys(NAMES, True)
        qualification["source_decode"] = False
        with self.assertRaises(MeasurementFailure):
            self.run_fake(
                qualification=qualification,
                time_unqualified=True,
                check=lambda name, output, boundary: {"qualified": name != "source_decode", "read_only": name != "source_decode"},
            )

    def test_known_graph_failure_keeps_eager_qualification(self):
        backend = FakeBackend(capture_error={"source_decode": RuntimeError("graph capture unsupported")})
        result, _ = self.run_fake(backend, graph_error_policy=known_graph_policy)
        self.assertEqual(result["methods"]["source_decode"]["graph_status"], "graph_unavailable")
        self.assertTrue(result["paired_ratios"]["eager_event_us"]["source_decode"]["qualified"])
        self.assertFalse(result["paired_ratios"]["graph_per_call_us"]["source_decode"]["qualified"])
        self.assertTrue(result["paired_ratios"]["graph_per_call_us"]["historical_fp32"]["qualified"])

    def test_unknown_capture_failure_retains_eager_and_is_not_silently_skipped(self):
        backend = FakeBackend(capture_error={"source_decode": RuntimeError("unclassified capture failure")})
        snapshots = []
        with self.assertRaises(MeasurementFailure) as caught:
            self.run_fake(backend, graph_error_policy=known_graph_policy, on_progress=lambda r: snapshots.append(copy.deepcopy(r)))
        report = caught.exception.partial_report
        self.assertEqual(report["status"], "failed")
        self.assertTrue(report["paired_ratios"]["eager_event_us"]["source_decode"]["qualified"])
        self.assertTrue(any(r["methods"]["candidate"]["eager_status"] == "measured" for r in snapshots))

    def test_known_replay_failure_never_scores_partial_graph(self):
        backend = FakeBackend(replay_error="source_decode")
        result, _ = self.run_fake(backend, graph_error_policy=known_graph_policy)
        self.assertEqual(result["methods"]["source_decode"]["graph_status"], "graph_unavailable")
        self.assertFalse(result["paired_ratios"]["graph_per_call_us"]["source_decode"]["qualified"])
        self.assertTrue(result["paired_ratios"]["graph_per_call_us"]["historical_fp32"]["qualified"])

    def test_ambiguous_or_missing_check_record_fails(self):
        for invalid in (None, {}, {"qualified": True, "passed": False}):
            with self.subTest(invalid=invalid), self.assertRaises(MeasurementFailure):
                self.run_fake(check=lambda name, output, boundary, invalid=invalid: invalid)


if __name__ == "__main__":
    unittest.main()
