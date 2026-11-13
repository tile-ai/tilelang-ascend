"""Small exhaustive integer oracles, independent of the Z3 constraint builder.

Complete manual stages bound the search to at most 4**4 phase assignments.
Resource conflicts are checked by occupied clock slots, rather than the
solver/verifier's disjunctions of interval inequalities.
"""

import importlib.util
import itertools
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location("s2_scheduler_oracle_subject", ROOT / "tilelang/transform/ascend_z3_scheduler.py")
scheduler = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = scheduler
SPEC.loader.exec_module(scheduler)
MAX_TIME = (1 << 63) - 1


def _problem(stages, **changes):
    count = len(stages)
    result = {
        "schema_version": 1,
        "mode": "modulo",
        "latencies": [1] * count,
        "iis": [1] * count,
        "resource_flags": [0] * count,
        "enable_offset": True,
        "manual_stages": list(stages),
        "ii_bounds": [1, 4],
        "timeout_ms": 1000,
    }
    result.update(changes)
    return result


CASES = [
    ("resource-wrap", _problem([0, 0], resource_flags=[1, 1], iis=[1, 2])),
    ("reverse-stage-pipe-tie", _problem([1, 0], pipe_order_deps=[(1, 0)])),
    ("register-generation-tie", _problem([0, 1], register_phase_deps=[(1, 0, 0, 0)])),
    (
        "opposite-tie-orders",
        _problem([0, 0], pipe_order_deps=[(0, 1)], register_phase_deps=[(1, 0, 0, 0)]),
    ),
    ("zero-reverse-owner-gap", _problem([0, 0], owner_exclusion_deps=[(0, 1, 1, 0)])),
    ("asymmetric-owner-wrap", _problem([0, 2], owner_exclusion_deps=[(0, 1, 2, 1)])),
    ("raw-distance", _problem([1, 0], data_deps=[(0, 1, 1, 1)])),
    ("reverse-zero-gap-data", _problem([0, 0, 1], data_deps=[(1, 0, 0, 0)])),
    ("inverted-stage-order", _problem([2, 0, 1], stage_order_deps=[(0, 1)])),
    ("multi-resource-mask", _problem([0, 2, 1], resource_flags=[1, 3, 2])),
    (
        "explicit-resource-pairs",
        _problem([0, 1, 0, 2], iis=[2, 2, 1, 1], resource_deps=[(0, 1), (2, 3)]),
    ),
    (
        "register-wrap-and-order",
        _problem(
            [2, 0, 1, 2],
            resource_flags=[1, 0, 2, 1],
            register_phase_deps=[(0, 1, 1, 2), (2, 3, 0, 0)],
            pipe_order_deps=[(1, 2)],
            stage_order_deps=[(1, 3)],
        ),
    ),
    (
        "int64-nonmonotone-domain",
        _problem([0, 2], latencies=[1, MAX_TIME - 2]),
    ),
]


def _oracle_valid(problem, period, phases):
    """Judge a complete assignment using integer clock occupancy and raw time."""
    if max(problem["iis"]) > period:
        return False
    stages = problem["manual_stages"]
    starts = [stage * period + phase for stage, phase in zip(stages, phases)]
    if any(start + latency > MAX_TIME for start, latency in zip(starts, problem["latencies"])):
        return False
    keys = [(phase, -stage, index) for index, (phase, stage) in enumerate(zip(phases, stages))]
    for producer, consumer, distance, gap in problem.get("data_deps", []):
        if starts[consumer] + distance * period - starts[producer] < gap:
            return False
        if producer != consumer and distance == gap == 0 and starts[producer] == starts[consumer] and keys[producer] >= keys[consumer]:
            return False
    occupied = [{(phases[task] + tick) % period for tick in range(issue)} for task, issue in enumerate(problem["iis"])]
    explicit = {frozenset(pair) for pair in problem.get("resource_deps", [])}
    for left, right in itertools.combinations(range(len(phases)), 2):
        conflict = problem["resource_flags"][left] & problem["resource_flags"][right]
        if (conflict or frozenset((left, right)) in explicit) and occupied[left] & occupied[right]:
            return False
    for left, right, forward_gap, backward_gap in problem.get("owner_exclusion_deps", []):
        clockwise = (phases[right] - phases[left]) % period
        if clockwise:
            if clockwise < forward_gap or period - clockwise < backward_gap:
                return False
        elif min(forward_gap, backward_gap) > 0 or max(forward_gap, backward_gap) > period:
            return False
    if any(keys[left] > keys[right] for left, right in problem.get("pipe_order_deps", [])):
        return False
    if any(stages[left] > stages[right] for left, right in problem.get("stage_order_deps", [])):
        return False
    for producer, consumer, wrap, gap in problem.get("register_phase_deps", []):
        if phases[consumer] + wrap * period - phases[producer] < gap:
            return False
        if producer != consumer and wrap == gap == 0 and phases[producer] == phases[consumer] and keys[producer] >= keys[consumer]:
            return False
    return True


def _candidate(problem, period, phases):
    stages = problem["manual_stages"]
    order = sorted(range(len(stages)), key=lambda task: (phases[task], -stages[task], task))
    return {
        "schema_version": 1,
        "status": "SAT",
        "start_times": [stage * period + phase for stage, phase in zip(stages, phases)],
        "order": order,
        "ii": period,
        "buffer_versions": [],
    }


@pytest.mark.parametrize("name,problem", CASES, ids=[case[0] for case in CASES])
def test_small_manual_stage_schedules_match_exhaustive_oracle(name, problem):
    feasible_periods = []
    for period in range(1, 5):
        fixed = dict(problem, ii_bounds=[period, period])
        witnesses = set()
        for phases in itertools.product(range(period), repeat=len(problem["manual_stages"])):
            feasible = _oracle_valid(problem, period, phases)
            # Cross-check the independent verifier over every assignment, not
            # merely the one candidate selected by the constraint builder.
            checked = scheduler.verify_schedule(fixed, _candidate(problem, period, phases))
            assert checked["valid"] is feasible, (name, period, phases, checked)
            if feasible:
                witnesses.add(phases)
        result = scheduler.solve_schedule(fixed)
        assert result["status"] == ("SAT" if witnesses else "UNSAT"), (name, period, result)
        if witnesses:
            feasible_periods.append(period)
            assert tuple(start % period for start in result["start_times"]) in witnesses
    result = scheduler.solve_schedule(problem)
    assert result["status"] == ("SAT" if feasible_periods else "UNSAT"), (name, result)
    if feasible_periods:
        assert result["ii"] == min(feasible_periods), (name, result, feasible_periods)
