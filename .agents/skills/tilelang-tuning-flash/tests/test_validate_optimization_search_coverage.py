"""Offline regression tests for the Flash search gate; no NPU imports are needed."""

from __future__ import annotations

import copy
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SKILL_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = SKILL_ROOT / "scripts" / "validate_optimization_search_coverage.py"
ASSETS = SKILL_ROOT / "assets"
SPEC = importlib.util.spec_from_file_location("flash_search_gate", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
GATE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GATE)


class SearchCoverageTest(unittest.TestCase):
    """Exercise plausible corrupted or incomplete search records, not document text."""

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.record_path = Path(self.temporary.name) / "coverage.json"

    def load_example(self, final: bool = True) -> dict:
        suffix = ".final" if final else ""
        path = ASSETS / f"optimization_search_coverage{suffix}.example.json"
        return json.loads(path.read_text(encoding="utf-8"))

    def simple_record(self) -> dict:
        """Keep the example evidence, isolating a single ordinary search obligation."""
        record = copy.deepcopy(self.load_example())
        case = record["cases"][0]
        case_id = case["case_id"]
        candidate_id = record["candidate_events"][0]["candidate_id"]
        case["source_window"]["access_pattern"] = "DIRECT_CONTIGUOUS"
        case["obligation_ids"] = ["O_CASE"]
        case["performance_target"] = {
            "logic": "ALL",
            "conditions": [{"metric": "kernel_time_us", "operator": "LE", "threshold": 10.0}],
        }
        record["obligations"] = [{
            "obligation_id": "O_CASE",
            "kind": "case_structure",
            "case_ids": [case_id],
            "status": "SATISFIED",
            "candidate_ids": [candidate_id],
            "case_dispositions": [{
                "case_id": case_id,
                "status": "TESTED_PATH",
                "candidate_ids": [candidate_id],
            }],
        }]
        return record

    def write_record(self, record: dict) -> Path:
        self.record_path.write_text(json.dumps(record), encoding="utf-8")
        return self.record_path

    def validate(self, record: dict, *, final: bool = False, allow_unmet: bool = False) -> str:
        return GATE.validate(self.write_record(record), final, allow_unmet)

    def assert_invalid(self, record: dict, *, final: bool = False) -> None:
        with self.assertRaises(GATE.ValidationError):
            self.validate(record, final=final)

    def add_convergence(self, record: dict, *, no_target: bool) -> None:
        case = record["cases"][0]
        key = "convergence_evidence" if no_target else "unmet_convergence_evidence"
        record[key] = {"case_bounds": [{
            "case_id": case["case_id"],
            "best_kernel_time_us": case["best_metrics"]["kernel_time_us"],
            "verified_lower_bound_us": 2.0,
            "evidence": ["fixtures/lower-bound-analysis.md"],
            "remaining_gap_explanation": "The remaining launch cost is measured separately.",
        }]}

    def test_initial_example_is_valid_but_cannot_finish(self) -> None:
        record = self.load_example(final=False)
        self.assertEqual(self.validate(record), "VALID_RECORD")
        with self.assertRaises(GATE.ValidationError):
            self.validate(record, final=True)

    def test_final_example_meets_numeric_target(self) -> None:
        self.assertEqual(self.validate(self.load_example(), final=True), "TARGET_MET")

    def test_missing_result_cannot_finish(self) -> None:
        record = self.simple_record()
        record["candidate_events"] = record["candidate_events"][:1]
        obligation = record["obligations"][0]
        obligation["status"] = "ACTIVE"
        obligation["case_dispositions"][0]["status"] = "PENDING"
        self.assertEqual(self.validate(record), "VALID_RECORD")
        self.assert_invalid(record, final=True)

    def test_candidate_identity_cannot_change_between_create_and_result(self) -> None:
        record = self.simple_record()
        record["candidate_events"][1]["identity"]["tail_strategy"] = "different tail handling"
        self.assert_invalid(record)

    def test_result_before_creation_is_rejected(self) -> None:
        record = self.simple_record()
        record["candidate_events"].reverse()
        self.assert_invalid(record)

    def test_duplicate_result_is_rejected(self) -> None:
        record = self.simple_record()
        record["candidate_events"].append(copy.deepcopy(record["candidate_events"][1]))
        self.assert_invalid(record)

    def test_unknown_applicable_case_is_rejected(self) -> None:
        record = self.simple_record()
        record["candidate_events"][0]["applicable_cases"].append("unrequested_shape")
        self.assert_invalid(record)

    def test_result_cannot_omit_an_applicable_case(self) -> None:
        record = self.simple_record()
        record["candidate_events"][1]["case_results"] = []
        self.assert_invalid(record)

    def test_duplicate_case_result_is_rejected(self) -> None:
        record = self.simple_record()
        results = record["candidate_events"][1]["case_results"]
        results.append(copy.deepcopy(results[0]))
        self.assert_invalid(record)

    def test_precision_only_does_not_count_as_performance_coverage(self) -> None:
        record = self.simple_record()
        result = record["candidate_events"][1]["case_results"][0]
        result["evidence"] = ["Only the correctness test passed."]
        del result["performance_measurement"]
        self.assert_invalid(record)

    def test_measured_time_requires_profiling_artifact(self) -> None:
        record = self.simple_record()
        result = record["candidate_events"][1]["case_results"][0]
        result["performance_measurement"]["artifact"] = ""
        self.assert_invalid(record)

    def test_obligation_cannot_claim_untested_case(self) -> None:
        record = self.simple_record()
        result = record["candidate_events"][1]
        result["result_status"] = "INAPPLICABLE_WITH_EVIDENCE"
        result["case_results"][0]["disposition"] = "INAPPLICABLE_WITH_EVIDENCE"
        self.assert_invalid(record)

    def test_promotion_requires_measurement_of_every_applicable_case(self) -> None:
        record = self.simple_record()
        result = record["candidate_events"][1]["case_results"][0]
        result["disposition"] = "NOT_RUN_WITH_EVIDENCE"
        result.pop("performance_measurement")
        obligation = record["obligations"][0]
        obligation["status"] = "ACTIVE"
        obligation["case_dispositions"][0]["status"] = "PENDING"
        self.assert_invalid(record)

    def prove_register_route(self, record: dict) -> None:
        source = record["cases"][0]["source_window"]
        source["register_window_feasible"] = True
        source["window_vregs"] = 2
        source.pop("register_infeasibility", None)
        source["register_feasibility_evidence"] = {
            "backend": "ascendc",
            "programming_mode": "Expert",
            "paths": ["fixtures/a3_route_api.py"],
            "symbols": ["fixture_verified_route"],
            "lowering_artifacts": ["fixtures/generated_route.cpp"],
            "finding": "Synthetic test evidence for a backend-supported direct route.",
        }

    def test_a3_register_feasibility_cannot_be_assumed_without_evidence(self) -> None:
        record = self.load_example()
        source = record["cases"][0]["source_window"]
        source["register_window_feasible"] = True
        source["window_vregs"] = 2
        source.pop("register_feasibility_evidence", None)
        self.assert_invalid(record)

    def test_a3_register_infeasibility_requires_backend_proof(self) -> None:
        record = self.load_example()
        source = record["cases"][0]["source_window"]
        source["register_window_feasible"] = False
        source.pop("register_infeasibility", None)
        self.assert_invalid(record)

    def test_direct_candidate_requires_proven_register_support(self) -> None:
        record = self.simple_record()
        for event in record["candidate_events"]:
            physical = event["identity"]["physical_dataflow"]
            physical.update({
                "route_kind": "CONTIGUOUS_LOAD_REGISTER_REORDER",
                "source_access": "CONTIGUOUS",
                "register_lane_reorder": True,
                "indexed_lane_access": False,
                "full_payload_materialization": False,
            })
        self.assert_invalid(record)

    def test_new_backend_evidence_does_not_invalidate_earlier_memory_route(self) -> None:
        record = self.load_example()
        source = record["cases"][0]["source_window"]
        source["register_window_feasible"] = None
        source["window_vregs"] = None
        self.assertEqual(self.validate(record), "VALID_RECORD")
        self.prove_register_route(record)
        for obligation in record["obligations"]:
            for item in obligation.get("route_plan", []):
                if item["route_kind"] == "CONTIGUOUS_LOAD_REGISTER_REORDER":
                    item["status"] = "PENDING"
                    item["candidate_ids"] = []
                    item.pop("hard_infeasibility", None)
        self.assertEqual(self.validate(record), "VALID_RECORD")
        record["cases"][0]["performance_target"] = None
        self.add_convergence(record, no_target=True)
        self.assert_invalid(record, final=True)

    def test_route_experiment_flag_cannot_hide_a_changed_physical_path(self) -> None:
        record = self.simple_record()
        for event in record["candidate_events"]:
            event["identity"]["physical_dataflow"]["route_experiment"] = False
        self.assert_invalid(record)

    def test_rejecting_unexecuted_candidate_requires_control_evidence(self) -> None:
        record = self.simple_record()
        result = record["candidate_events"][1]
        result["result_status"] = "REJECTED_WITH_EVIDENCE"
        result["case_results"][0]["disposition"] = "NOT_RUN_WITH_EVIDENCE"
        obligation = record["obligations"][0]
        obligation["status"] = "ACTIVE"
        obligation["case_dispositions"][0]["status"] = "PENDING"
        self.assert_invalid(record)
        result["rejection_evidence"] = {
            "failure_stage": "COMPILATION",
            "observed_error": "Candidate shape cannot lower to the checked AscendC API.",
            "control_check": "The unchanged B0 compiled on the same healthy device.",
            "candidate_finding": "Only this candidate's buffer shape is unsupported.",
        }
        self.assertEqual(self.validate(record), "VALID_RECORD")

    def blocked_record(self) -> dict:
        record = self.simple_record()
        result = record["candidate_events"][1]
        result["result_status"] = "BLOCKED_BY_ENVIRONMENT"
        result["case_results"][0]["disposition"] = "NOT_RUN_BLOCKED"
        result["case_results"][0].pop("performance_measurement", None)
        result["blocking_evidence"] = {
            "kind": "DEVICE_UNAVAILABLE",
            "failure_stage": "device initialization",
            "observed_error": "NPU reset before candidate launch.",
            "control_check": "B0 also cannot initialize the same device.",
            "resume_condition": "The device recovers and B0 health check succeeds.",
        }
        obligation = record["obligations"][0]
        obligation["status"] = "BLOCKED"
        obligation["blocking_reasons"] = ["Device unavailable before profiling."]
        disposition = obligation["case_dispositions"][0]
        disposition["status"] = "BLOCKED"
        disposition["evidence"] = ["logs/device-health.txt"]
        return record

    def test_environment_block_can_be_saved_but_cannot_finish_even_when_target_met(self) -> None:
        record = self.blocked_record()
        self.assertEqual(self.validate(record), "VALID_RECORD")
        self.assert_invalid(record, final=True)

    def recovered_record(self) -> dict:
        record = self.blocked_record()
        previous_id = record["candidate_events"][0]["candidate_id"]
        for event in copy.deepcopy(self.simple_record()["candidate_events"]):
            event["candidate_id"] = "C_RECOVERED"
            event["resumes_candidate_id"] = previous_id
            record["candidate_events"].append(event)
        obligation = record["obligations"][0]
        obligation["status"] = "SATISFIED"
        obligation["candidate_ids"] = [previous_id, "C_RECOVERED"]
        obligation.pop("blocking_reasons")
        disposition = obligation["case_dispositions"][0]
        disposition["status"] = "TESTED_PATH"
        disposition["candidate_ids"] = ["C_RECOVERED"]
        disposition["evidence"] = ["profiling/recovered/summary.txt"]
        return record

    def test_resumed_candidate_can_finish_without_deleting_blocked_history(self) -> None:
        record = self.recovered_record()
        self.assertEqual(self.validate(record, final=True), "TARGET_MET")
        self.assertEqual(record["candidate_events"][1]["result_status"], "BLOCKED_BY_ENVIRONMENT")

    def test_resumption_cannot_silently_change_candidate_identity(self) -> None:
        record = self.recovered_record()
        for event in record["candidate_events"][2:]:
            event["identity"]["tail_strategy"] = "a different algorithm after restart"
        self.assert_invalid(record)

    def test_successful_candidate_cannot_be_claimed_as_blocked_resume_source(self) -> None:
        record = self.recovered_record()
        for event in copy.deepcopy(record["candidate_events"][2:]):
            event["candidate_id"] = "C_BAD_RESUME"
            event["resumes_candidate_id"] = "C_RECOVERED"
            record["candidate_events"].append(event)
        self.assert_invalid(record)

    def test_nonfinite_timing_or_target_is_rejected(self) -> None:
        for field in ("measurement", "best_metric", "target"):
            for value in (float("inf"), float("nan")):
                with self.subTest(field=field, value=value):
                    record = self.simple_record()
                    if field == "measurement":
                        item = record["candidate_events"][1]["case_results"][0]
                        item["performance_measurement"]["kernel_time_us"] = value
                    elif field == "best_metric":
                        record["cases"][0]["best_metrics"]["kernel_time_us"] = value
                    else:
                        condition = record["cases"][0]["performance_target"]["conditions"][0]
                        condition["threshold"] = value
                    self.assert_invalid(record)

    def test_numeric_target_does_not_imply_other_conditions_met(self) -> None:
        record = self.simple_record()
        case = record["cases"][0]
        case["best_metrics"]["semantic_gm_util_pct"] = 50.0
        case["performance_target"]["conditions"].append({
            "metric": "semantic_gm_util_pct", "operator": "GE", "threshold": 80.0,
        })
        with self.assertRaises(GATE.SearchIncomplete):
            self.validate(record, final=True)
        case["performance_target"]["logic"] = "ANY"
        self.assertEqual(self.validate(record, final=True), "TARGET_MET")

    def test_unmet_target_requires_explicit_convergence_and_bound_evidence(self) -> None:
        record = self.simple_record()
        record["cases"][0]["performance_target"]["conditions"][0]["threshold"] = 1.0
        with self.assertRaises(GATE.SearchIncomplete):
            self.validate(record, final=True)
        with self.assertRaises(GATE.ValidationError):
            self.validate(record, final=True, allow_unmet=True)
        self.add_convergence(record, no_target=False)
        self.assertEqual(
            self.validate(record, final=True, allow_unmet=True), "UNMET_BUT_CONVERGED"
        )

    def test_no_target_is_not_misreported_as_target_met(self) -> None:
        record = self.simple_record()
        record["cases"][0]["performance_target"] = None
        with self.assertRaises(GATE.ValidationError):
            self.validate(record, final=True)
        self.add_convergence(record, no_target=True)
        self.assertEqual(
            self.validate(record, final=True), "CONVERGED_WITHOUT_NUMERIC_TARGET"
        )

    def test_convergence_cannot_use_another_versions_best_timing(self) -> None:
        record = self.simple_record()
        record["cases"][0]["performance_target"] = None
        self.add_convergence(record, no_target=True)
        record["convergence_evidence"]["case_bounds"][0]["best_kernel_time_us"] += 1.0
        self.assert_invalid(record, final=True)

    def test_lower_bound_cannot_exceed_measured_best(self) -> None:
        record = self.simple_record()
        record["cases"][0]["performance_target"] = None
        self.add_convergence(record, no_target=True)
        record["convergence_evidence"]["case_bounds"][0]["verified_lower_bound_us"] = 100.0
        self.assert_invalid(record, final=True)

    def test_cli_distinguishes_valid_invalid_and_incomplete(self) -> None:
        record = self.simple_record()
        path = self.write_record(record)
        good = subprocess.run(
            [sys.executable, str(SCRIPT), str(path), "--final"],
            text=True, capture_output=True, check=False,
        )
        self.assertEqual(good.returncode, 0, good.stderr)
        self.assertEqual(good.stdout.strip(), "TARGET_MET")
        record["cases"][0]["performance_target"]["conditions"][0]["threshold"] = 1.0
        self.write_record(record)
        incomplete = subprocess.run(
            [sys.executable, str(SCRIPT), str(path), "--final"],
            text=True, capture_output=True, check=False,
        )
        self.assertEqual(incomplete.returncode, 2, incomplete.stderr)
        path.write_text("{ invalid json", encoding="utf-8")
        invalid = subprocess.run(
            [sys.executable, str(SCRIPT), str(path)],
            text=True, capture_output=True, check=False,
        )
        self.assertEqual(invalid.returncode, 1, invalid.stderr)
        self.assertNotIn("Traceback", invalid.stderr)


if __name__ == "__main__":
    unittest.main()
