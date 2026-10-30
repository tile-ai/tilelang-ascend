"""Numeric S2 contracts; deliberately independent of TileLang and TVM imports."""

import copy
import importlib.util
from pathlib import Path
import random
import sys

import pytest

ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location("ascend_s2_numeric", ROOT / "tilelang/transform/ascend_z3_scheduler.py")
scheduler = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = scheduler
SPEC.loader.exec_module(scheduler)


def problem(n=3, mode="straight", **changes):
    result = {
        "schema_version": 1,
        "mode": mode,
        "latencies": [3] * n,
        "iis": [1] * n,
        "resource_flags": [1 << i for i in range(n)],
    }
    if mode == "modulo":
        result["ii_bounds"] = [1, 32]
    result.update(changes)
    return result


def solved(p):
    result = scheduler.solve_schedule(p)
    assert result["status"] == "SAT", result
    assert scheduler.verify_schedule(p, result) == {"valid": True, "errors": []}
    return result


@pytest.mark.parametrize("n", range(1, 17))
def test_straight_task_counts_and_dependency_families(n):
    p = problem(n, data_deps=[(i, i + 1, 0, 3) for i in range(n - 1)])
    result = solved(p)
    assert len(result["start_times"]) == n
    assert result["order"] == list(range(n))


@pytest.mark.parametrize("seed", range(32))
def test_seeded_dags(seed):
    rng = random.Random(seed)
    n = rng.randint(1, 16)
    latencies = [rng.randint(1, 9) for _ in range(n)]
    edges = [(u, v, 0, latencies[u]) for u in range(n) for v in range(u + 1, n) if rng.random() < 0.2]
    solved(
        problem(
            n,
            latencies=latencies,
            resource_flags=[1 << rng.randrange(3) for _ in range(n)],
            data_deps=edges,
        )
    )


@pytest.mark.parametrize("mode", ["straight", "modulo"])
def test_resource_masks_are_authoritative_even_without_pair_list(mode):
    p = problem(2, mode, resource_flags=[1, 1], iis=[4, 5])
    result = solved(p)
    assert abs(result["start_times"][0] - result["start_times"][1]) >= 4
    if mode == "modulo":
        assert result["ii"] >= 9


def test_unsat_is_distinct_from_fallback():
    p = problem(2, data_deps=[(0, 1, 0, 1), (1, 0, 0, 1)])
    result = scheduler.solve_schedule(p)
    assert result["status"] == "UNSAT"
    assert result["start_times"] == []
    assert result["order"] == []


def test_modulo_unsat_is_bounded_and_includes_upper_endpoint():
    p = problem(2, "modulo", resource_flags=[1, 1], iis=[4, 5], ii_bounds=[9, 9])
    assert solved(p)["ii"] == 9
    p["ii_bounds"] = [1, 8]
    result = scheduler.solve_schedule(p)
    assert result["status"] == "UNSAT" and "inclusive II domain" in result["reason"]


def test_modulo_feasibility_is_not_assumed_monotone_with_int64_bounds():
    p = problem(
        1,
        "modulo",
        latencies=[(1 << 63) - 3],
        enable_offset=True,
        manual_stages=[1],
        ii_bounds=[1, 16],
    )
    result = solved(p)
    assert result["ii"] == 1 and result["start_times"] == [1]


def test_buffer_versions_and_capacity():
    p = problem(
        2,
        "modulo",
        ii_bounds=[4, 4],
        enable_offset=True,
        manual_stages=[0, 1],
        num_stages=3,
        buffer_sizes=[32],
        memory_groups=[[64, 0]],
        data_deps=[(0, 1, 0, 5), (1, 0, -1, 2)],
    )
    result = solved(p)
    assert result["buffer_versions"] == [2]
    p["memory_groups"] = [[63, 0]]
    assert scheduler.solve_schedule(p)["status"] == "UNSAT"


def test_manual_stages_are_raw_and_register_generation_ties_are_ordered():
    p = problem(
        4,
        "modulo",
        enable_offset=True,
        manual_stages=[4, 4, 7, 7],
        register_phase_deps=[(0, 1, 0, 1), (1, 2, 0, 0), (2, 3, 0, 1), (3, 0, 1, 0)],
    )
    result = solved(p)
    assert [start // result["ii"] for start in result["start_times"]] == [4, 4, 7, 7]
    phases = [start % result["ii"] for start in result["start_times"]]
    assert phases[2] > phases[1]


def test_stage_order_and_pipe_order():
    p = problem(
        3,
        "modulo",
        enable_offset=True,
        manual_stages=[2, 1, 2],
        pipe_order_deps=[(0, 1)],
        stage_order_deps=[(1, 2)],
    )
    solved(p)
    p["stage_order_deps"] = [(0, 1)]
    assert scheduler.solve_schedule(p)["status"] == "UNSAT"


def test_offset_disabled_keeps_resource_within_one_window():
    p = problem(2, "modulo", resource_flags=[1, 1], data_deps=[(0, 1, 0, 3)])
    result = solved(p)
    assert max(result["start_times"]) - min(result["start_times"]) < result["ii"]


def test_owner_exclusion_wraparound():
    p = problem(2, "modulo", owner_exclusion_deps=[(0, 1, 6, 7)])
    assert solved(p)["ii"] >= 13


@pytest.mark.parametrize("field", ["start_times", "order", "ii", "buffer_versions"])
def test_corrupted_solutions_are_rejected(field):
    p = problem(
        2,
        "modulo",
        buffer_sizes=[8],
        memory_groups=[[16, 0]],
        num_stages=2,
        data_deps=[(0, 1, 0, 3)],
        resource_flags=[1, 1],
    )
    result = solved(p)
    if field in ("start_times", "order"):
        result[field] = [0, 0]
    elif field == "ii":
        result[field] = 0
    else:
        result[field] = [3]
    assert not scheduler.verify_schedule(p, result)["valid"]


def test_verifier_independently_reconstructs_missing_resource_pairs():
    p = problem(2, resource_flags=[1, 1])
    result = {
        "schema_version": 1,
        "status": "SAT",
        "start_times": [0, 0],
        "order": [0, 1],
        "ii": 0,
        "buffer_versions": [],
    }
    assert "Resource exclusion" in ";".join(scheduler.verify_schedule(p, result)["errors"])


def test_independent_verifier_blocks_a_broken_constraint_builder(monkeypatch):
    monkeypatch.setattr(scheduler, "_resource_pairs", lambda _: [])
    result = scheduler.solve_schedule(problem(2, resource_flags=[1, 1]))
    assert result["status"] == "ERROR"
    assert "Independent verifier" in result["reason"]


def test_verifier_checks_capacity_not_only_version_limits():
    p = problem(1, "modulo", buffer_sizes=[8], memory_groups=[[8, 0]], num_stages=2)
    result = solved(p)
    result["buffer_versions"] = [2]
    assert "capacity" in ";".join(scheduler.verify_schedule(p, result)["errors"])


def test_verifier_checks_single_window_and_owner_reverse_wrap():
    candidate = {
        "schema_version": 1,
        "status": "SAT",
        "start_times": [0, 7],
        "order": [0, 1],
        "ii": 5,
        "buffer_versions": [],
    }
    p = problem(2, "modulo", resource_flags=[1, 1])
    assert "single II window" in ";".join(scheduler.verify_schedule(p, candidate)["errors"])
    p = problem(2, "modulo", owner_exclusion_deps=[(0, 1, 4, 5)])
    candidate["start_times"] = [0, 4]
    candidate["ii"] = 6
    assert "Owner exclusion" in ";".join(scheduler.verify_schedule(p, candidate)["errors"])


@pytest.mark.parametrize("family,gap", [("RAW", 5), ("WAR", 1), ("WAW", 3)])
def test_access_dependency_families_are_not_resource_exclusions(family, gap):
    # The effect adapter supplies directed edges for each family; pure solving
    # must preserve their direction even when task ids are in the reverse order.
    p = problem(2, data_deps=[(1, 0, 0, gap)])
    result = solved(p)
    assert result["start_times"][0] >= result["start_times"][1] + gap, family


def test_verifier_rejects_phase_tie_even_when_numeric_inequality_holds():
    p = problem(
        2,
        "modulo",
        manual_stages=[0, 1],
        enable_offset=True,
        register_phase_deps=[(0, 1, 0, 0)],
    )
    result = {
        "schema_version": 1,
        "status": "SAT",
        "start_times": [0, 4],
        "order": [1, 0],
        "ii": 4,
        "buffer_versions": [],
    }
    assert "reversed at a tie" in ";".join(scheduler.verify_schedule(p, result)["errors"])


@pytest.mark.parametrize(
    "changes",
    [
        {"latencies": []},
        {"latencies": [True, 1, 1]},
        {"iis": [0, 1, 1]},
        {"data_deps": [(0, 9, 0, 1)]},
        {"data_deps": [(0, 1, -1, 1)]},
        {"resource_deps": [(0, 0)]},
        {"garbage": 1},
        {"timeout_ms": 0},
        {"timeout_ms": 1 << 31},
        {"latencies": [1 << 63, 1, 1]},
        {"register_phase_deps": [(0, 1, 2, 1)]},
    ],
)
def test_invalid_requests_are_not_unsat(changes):
    p = problem(**changes)
    with pytest.raises(scheduler.InvalidProblem):
        scheduler.solve_schedule(p)


def test_large_integer_arithmetic_is_exact():
    p = problem(2, data_deps=[(0, 1, 0, (1 << 53) + 7)])
    result = solved(p)
    assert result["start_times"][1] - result["start_times"][0] >= (1 << 53) + 7


def test_missing_dependency_does_not_affect_verifier(monkeypatch):
    p = problem(1)
    candidate = solved(p)

    def missing():
        raise ModuleNotFoundError("controlled missing z3", name="z3")

    monkeypatch.setattr(scheduler, "_load_z3", missing)
    assert scheduler.solve_schedule(p)["status"] == "MISSING_DEPENDENCY"
    assert scheduler.verify_schedule(p, candidate)["valid"]


def test_controlled_deadline_does_not_guess_a_hard_workload(monkeypatch):
    # Advancing a deterministic clock expires a real solve's overall budget.
    ticks = iter([0.0, 100.0])
    monkeypatch.setattr(scheduler, "_clock", lambda: next(ticks))
    result = scheduler.solve_schedule(problem(1))
    assert result["status"] == "TIMEOUT"
    assert result["start_times"] == []


@pytest.mark.parametrize("reason,status", [("timeout", "TIMEOUT"), ("max. resource limit exceeded", "UNKNOWN")])
def test_backend_unknown_is_never_unsat(reason, status):
    import z3

    class Backend:
        def set(self, **kwargs):
            pass

        def check(self):
            return z3.unknown

        def reason_unknown(self):
            return reason

    actual, detail = scheduler._check_solver(Backend(), z3, scheduler._clock() + 10)
    assert actual == status and detail == reason


def test_answer_after_global_deadline_is_timeout(monkeypatch):
    import z3

    class Backend:
        def set(self, **kwargs):
            pass

        def check(self):
            return z3.sat

    ticks = iter([0.0, 20.0])
    monkeypatch.setattr(scheduler, "_clock", lambda: next(ticks))
    status, _ = scheduler._check_solver(Backend(), z3, 10.0)
    assert status == "TIMEOUT"


def test_result_verification_is_inside_global_deadline(monkeypatch):
    clock = [0.0]
    verify = scheduler.verify_schedule
    monkeypatch.setattr(scheduler, "_clock", lambda: clock[0])

    def late_verification(p, candidate):
        checked = verify(p, candidate)
        clock[0] = 6.0
        return checked

    monkeypatch.setattr(scheduler, "verify_schedule", late_verification)
    result = scheduler.solve_schedule(problem(1, timeout_ms=5000))
    assert result["status"] == "TIMEOUT" and result["start_times"] == []


@pytest.mark.parametrize(
    "error",
    [
        OSError("controlled broken solver library"),
        ImportError("controlled broken solver installation"),
        ModuleNotFoundError("controlled missing internal dependency", name="z3_internal"),
    ],
)
def test_broken_backend_returns_error(monkeypatch, error):
    def broken():
        raise error

    monkeypatch.setattr(scheduler, "_load_z3", broken)
    result = scheduler.solve_schedule(problem(1))
    assert result["status"] == "ERROR" and str(error) in result["reason"]


def test_repeated_calls_use_private_contexts():
    p = problem(4, "modulo", resource_flags=[1, 1, 2, 2])
    results = [solved(copy.deepcopy(p)) for _ in range(3)]
    assert all(result["ii"] == results[0]["ii"] for result in results)
