"""CPU regressions for alignment provenance and independent-trial reporting."""

import copy
from pathlib import Path
import subprocess
import sys
import unittest

from ._perf_tools import load_perf_tool


class AlignmentReportingTests(unittest.TestCase):
    def test_timing_engine_cannot_be_shadowed_by_baseline_module(self):
        import tempfile
        import types
        from unittest.mock import patch

        tool = load_perf_tool("aligned_benchmark")
        original_path = list(sys.path)
        sentinel = types.ModuleType("paired_timing")
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "paired_timing.py").write_text("raise AssertionError('wrong timing engine')\n")
            with patch.dict(sys.modules, {"paired_timing": sentinel}):
                try:
                    sys.path.insert(0, directory)
                    actual = tool.load_local_tool("paired_timing")
                    self.assertEqual(Path(actual.__file__).resolve(), Path(tool.__file__).with_name("paired_timing.py"))
                    self.assertEqual(len(actual.balanced_orders(("candidate", "official_fia_core"), 6, 0)), 6)
                    self.assertIs(sys.modules["paired_timing"], sentinel)
                finally:
                    sys.path[:] = original_path

    def test_exact_output_comparison_rejects_signed_zero_and_nonfinite_reference(self):
        import torch

        tool = load_perf_tool("aligned_benchmark")
        positive, negative = torch.tensor([0.0]), torch.tensor([-0.0])
        with self.assertRaisesRegex(AssertionError, "payload changed"):
            tool.assert_same_bytes(torch, positive, negative, "output")
        tool.require_finite_core_reference(torch, positive, positive)
        for left, right in ((torch.tensor([float("inf")]), positive), (positive, torch.tensor([float("nan")]))):
            with self.assertRaisesRegex(AssertionError, "must be finite"):
                tool.require_finite_core_reference(torch, left, right)

    def report(self):
        stats = {"median": 10.0, "p95": 12.0, "count": 12}
        return {
            "case": "decode17",
            "dtype": "float16",
            "target": "pto",
            "alignment_scope": "shared-lp-core",
            "aligned_methods": {"production_path": True},
            "core_prepared_inputs": {"ql": {"sha256": "same"}},
            "status": "balanced_benchmark_completed",
            "trial_id": 0,
            "rounds": 6,
            "repeats_per_round": 2,
            "graph_repeats_per_round": 2,
            "measurement": {
                "config": {"rounds": 6, "order_seed": 0},
                "methods": {
                    name: {
                        "eager_status": "measured",
                        "graph_status": "measured",
                        "metrics": {key: dict(stats) for key in ("eager_event_us", "eager_wall_us", "graph_per_call_us")},
                    }
                    for name in ("candidate", "official_fia_core")
                },
                "paired_ratios": {
                    key: {"official_fia_core": {"qualified": True, "ratio_of_medians": 0.8}}
                    for key in ("eager_event_us", "eager_wall_us", "graph_per_call_us")
                },
            },
        }

    def test_scope_input_and_method_changes_cannot_be_pooled(self):
        tool = load_perf_tool("run_aligned_suite")
        first = self.report()
        for key, value in (
            ("alignment_scope", "prepared-full-chain"),
            ("aligned_methods", {"production_path": False}),
            ("core_prepared_inputs", {"ql": {"sha256": "different"}}),
            ("aligned_execution_sha256", "different"),
            ("candidate_kernels", {"attention": {"source_sha256": "different"}}),
        ):
            with self.subTest(key=key):
                second = copy.deepcopy(first)
                second["trial_id"] = 1
                second[key] = value
                with self.assertRaisesRegex(ValueError, "different source/runtime/input/timing"):
                    tool.summarize([first, second])

    def test_independent_orders_allowed_and_unqualified_ratio_suppressed(self):
        tool = load_perf_tool("run_aligned_suite")
        first, second = self.report(), self.report()
        second["trial_id"] = 1
        second["measurement"]["config"]["order_seed"] = 1
        rows = tool.summarize([first, second])
        self.assertTrue(all(r["alignment_scope"] == "shared-lp-core" for r in rows))
        self.assertTrue(all(r["reference_over_candidate_median"] == 0.8 for r in rows if r["method"] != "candidate"))
        second["measurement"]["paired_ratios"]["eager_event_us"]["official_fia_core"]["qualified"] = False
        rows = tool.summarize([first, second])
        row = next(r for r in rows if r["method"] == "official_fia_core" and r["metric"] == "eager_event_us")
        self.assertFalse(row["qualified_ratio"])
        self.assertIsNone(row["reference_over_candidate_median"])
        self.assertEqual(row["trial_ratios"], [])

    def test_new_tool_imports_and_suite_dry_run_do_not_load_torch(self):
        package = Path(__file__).resolve().parents[1]
        script = r"""
import importlib.util, json, pathlib, subprocess, sys, tempfile
root = pathlib.Path(sys.argv[1])
for name in ("aligned_execution", "aligned_benchmark", "run_aligned_suite"):
    spec = importlib.util.spec_from_file_location("_aligned_lazy_" + name, root / "benchmark" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
assert "torch" not in sys.modules and "torch_npu" not in sys.modules
with tempfile.TemporaryDirectory() as directory:
    out = pathlib.Path(directory) / "plan"
    command = [sys.executable, str(root / "benchmark/run_aligned_suite.py"), "--scope", "shared-lp-core",
               "--installed-root", "unused", "--inventory", "unused", "--benchmark-root", "unused",
               "--output", str(out), "--dry-run"]
    child = subprocess.run(command, capture_output=True, text=True)
    assert child.returncode == 0, child.stdout + child.stderr
    plan = json.loads((out / "suite.json").read_text())
    assert plan["planned_trials"] == 36 and plan["completed"] == []
    assert plan["alignment_scope"] == "shared-lp-core"
    assert all("aligned_benchmark.py" in item["command"][1] and "--scope" in item["command"] for item in plan["plan"])
"""
        child = subprocess.run([sys.executable, "-c", script, str(package)], capture_output=True, text=True)
        self.assertEqual(child.returncode, 0, child.stdout + child.stderr)

    def test_partial_graph_is_diagnostic_and_never_a_complete_trial(self):
        tool = load_perf_tool("run_aligned_suite")
        report = self.report()
        row = report["measurement"]["methods"]["official_fia_core"]
        row["graph_status"] = "graph_unavailable"
        row["metrics"]["graph_per_call_us"]["count"] = 3
        report["measurement"]["paired_ratios"]["graph_per_call_us"]["official_fia_core"]["qualified"] = False
        summary = tool.summarize([report])
        item = next(r for r in summary if r["metric"] == "graph_per_call_us" and r["method"] == "official_fia_core")
        self.assertEqual(item["measured_trials"], 0)
        self.assertIsNone(item["median_of_trial_medians_us"])
        self.assertEqual(item["diagnostic_partial_trial_medians_us"], [10.0])
        self.assertEqual(item["boundary_status_by_trial"], ["graph_unavailable"])
        self.assertFalse(item["qualified_ratio"])
        row["graph_status"] = "measured"
        with self.assertRaisesRegex(ValueError, "incomplete sample count"):
            tool.summarize([report])

    def test_same_trial_cannot_be_counted_as_independent_repetition(self):
        tool = load_perf_tool("run_aligned_suite")
        with self.assertRaisesRegex(ValueError, "distinct integer trial_id"):
            tool.summarize([self.report(), self.report()])
