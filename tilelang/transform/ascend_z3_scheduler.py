"""Pure integer scheduling protocol; import Z3 only for an explicit solve.

The equations are adapted from tile-ai/tilelang 1c0f402e's Ascend scheduler.
This module does not import TileLang/TVM, classify IR, emit annotations, or
assume hardware timing. A SAT result is always checked without using Z3.

Vector indices are stable source-order task ids. Dependencies are
``(u, v, iteration_distance, gap)``; a negative distance ``-b-1`` refers to
buffer ``b``'s version count. Buffer sizes and group capacities are physical
bytes. Modulo II bounds are inclusive; raw manual stages are never normalized.
Resource mask bits identify caller-defined core/PIPE resources, not device costs.
"""

from __future__ import annotations

from collections.abc import Mapping
import importlib
import threading
import time

INT64_MAX = (1 << 63) - 1
_lock = threading.Lock()
_clock = time.monotonic
_FIELDS = {
    "schema_version",
    "mode",
    "latencies",
    "iis",
    "resource_flags",
    "data_deps",
    "resource_deps",
    "owner_exclusion_deps",
    "pipe_order_deps",
    "stage_order_deps",
    "num_stages",
    "buffer_sizes",
    "memory_groups",
    "manual_stages",
    "enable_offset",
    "register_phase_deps",
    "ii_bounds",
    "timeout_ms",
    "seed",
}


class InvalidProblem(ValueError):
    """The request is malformed, rather than a well-formed infeasible problem."""


def _integer(value, name, minimum=0):
    if type(value) is not int or not minimum <= value <= INT64_MAX:
        raise InvalidProblem(f"{name} must be an integer in [{minimum}, {INT64_MAX}]")
    return value


def _vector(value, name, minimum=0):
    if not isinstance(value, (list, tuple)):
        raise InvalidProblem(f"{name} must be a list or tuple")
    return [_integer(item, f"{name}[{index}]", minimum) for index, item in enumerate(value)]


def _tuples(value, name, arity):
    if not isinstance(value, (list, tuple)):
        raise InvalidProblem(f"{name} must be a list or tuple")
    result = []
    for index, item in enumerate(value):
        if not isinstance(item, (list, tuple)) or len(item) != arity:
            raise InvalidProblem(f"{name}[{index}] must contain {arity} integers")
        result.append(tuple(_integer(x, name, -INT64_MAX) for x in item))
    return result


def validate_problem(problem: Mapping) -> dict:
    """Return normalized Python data, without importing a solver or runtime."""
    if not isinstance(problem, Mapping):
        raise InvalidProblem("problem must be a mapping")
    if any(not isinstance(key, str) for key in problem):
        raise InvalidProblem("Problem field names must be strings")
    extra = set(problem) - _FIELDS
    if extra:
        raise InvalidProblem(f"Unknown problem fields: {sorted(extra)}")
    if _integer(problem.get("schema_version", 1), "schema_version", 1) != 1:
        raise InvalidProblem("Unsupported scheduling schema_version")
    mode = problem.get("mode")
    if mode not in ("straight", "modulo"):
        raise InvalidProblem("mode must be straight or modulo")
    result = {"schema_version": 1, "mode": mode}
    for key, minimum in (("latencies", 1), ("iis", 1), ("resource_flags", 0)):
        result[key] = _vector(problem.get(key, []), key, minimum)
    n = len(result["latencies"])
    if not n or len(result["iis"]) != n or len(result["resource_flags"]) != n:
        raise InvalidProblem("latencies/iis/resource_flags must have the same positive length")
    for name, arity in (
        ("data_deps", 4),
        ("resource_deps", 2),
        ("owner_exclusion_deps", 4),
        ("pipe_order_deps", 2),
        ("stage_order_deps", 2),
        ("register_phase_deps", 4),
    ):
        result[name] = _tuples(problem.get(name, []), name, arity)
        for edge in result[name]:
            if not (0 <= edge[0] < n and 0 <= edge[1] < n):
                raise InvalidProblem(f"{name} contains an out-of-range task id: {edge}")
            if name in ("resource_deps", "owner_exclusion_deps") and edge[0] == edge[1]:
                raise InvalidProblem(f"{name} cannot exclude a task from itself")
    result["num_stages"] = _integer(problem.get("num_stages", 1), "num_stages", 1)
    result["buffer_sizes"] = _vector(problem.get("buffer_sizes", []), "buffer_sizes", 1)
    result["manual_stages"] = _vector(problem.get("manual_stages", []), "manual_stages")
    if result["manual_stages"] and len(result["manual_stages"]) != n:
        raise InvalidProblem("manual_stages must be empty or have one entry per task")
    result["enable_offset"] = problem.get("enable_offset", False)
    if type(result["enable_offset"]) is not bool:
        raise InvalidProblem("enable_offset must be boolean")
    if not result["enable_offset"] and any(result["manual_stages"]):
        raise InvalidProblem("Nonzero manual stages require enable_offset=True")
    groups = problem.get("memory_groups", [])
    if not isinstance(groups, (list, tuple)):
        raise InvalidProblem("memory_groups must be a list or tuple")
    result["memory_groups"] = []
    assigned = []
    for group in groups:
        group = _vector(group, "memory_groups entry")
        if len(group) < 2 or any(index >= len(result["buffer_sizes"]) for index in group[1:]):
            raise InvalidProblem("Each memory group needs a capacity and valid buffer ids")
        assigned.extend(group[1:])
        result["memory_groups"].append(group)
    if sorted(assigned) != list(range(len(result["buffer_sizes"]))):
        raise InvalidProblem("Each buffer must belong to exactly one capacity group")
    for u, v, distance, gap in result["data_deps"]:
        if gap < 0 or distance < -len(result["buffer_sizes"]):
            raise InvalidProblem(f"Invalid data dependency: {(u, v, distance, gap)}")
        if mode == "straight" and distance:
            raise InvalidProblem("Straight dependencies must have distance zero")
    if any(a < 0 or b < 0 for _, _, a, b in result["owner_exclusion_deps"]):
        raise InvalidProblem("Owner exclusion gaps must be nonnegative")
    for u, v, wrap, gap in result["register_phase_deps"]:
        if wrap not in (0, 1) or gap < 0:
            raise InvalidProblem(f"Invalid register phase dependency: {(u, v, wrap, gap)}")
    if result["register_phase_deps"] and (not result["manual_stages"] or not result["enable_offset"]):
        raise InvalidProblem("Register phase dependencies require manual stages and enable_offset")
    if mode == "straight":
        if (
            any(
                result[name]
                for name in (
                    "buffer_sizes",
                    "manual_stages",
                    "stage_order_deps",
                    "register_phase_deps",
                )
            )
            or result["enable_offset"]
        ):
            raise InvalidProblem("Modulo-only fields cannot be used in a straight problem")
        result["ii_bounds"] = []
    else:
        bounds = _vector(problem.get("ii_bounds", []), "ii_bounds", 1)
        if len(bounds) != 2 or bounds[0] > bounds[1]:
            raise InvalidProblem("modulo ii_bounds must be [lower, upper], inclusive")
        result["ii_bounds"] = bounds
    result["timeout_ms"] = _integer(problem.get("timeout_ms", 5000), "timeout_ms", 1)
    if result["timeout_ms"] > (1 << 31) - 1:
        raise InvalidProblem("timeout_ms must fit a positive signed int32")
    result["seed"] = _integer(problem.get("seed", 0), "seed")
    if result["seed"] > (1 << 31) - 1:
        raise InvalidProblem("seed must fit a nonnegative signed int32")
    return result


def _result(status, reason="", starts=None, order=None, ii=0, versions=None):
    return {
        "schema_version": 1,
        "status": status,
        "reason": reason,
        "start_times": [] if starts is None else starts,
        "order": [] if order is None else order,
        "ii": ii,
        "buffer_versions": [] if versions is None else versions,
    }


def verify_schedule(problem: Mapping, candidate: Mapping) -> dict:
    """Independently check a SAT candidate using only ordinary integer arithmetic."""
    p = validate_problem(problem)
    errors = []
    n = len(p["latencies"])
    if not isinstance(candidate, Mapping) or candidate.get("status") != "SAT":
        return {"valid": False, "errors": ["Expected a SAT candidate mapping"]}
    try:
        if _integer(candidate.get("schema_version"), "result schema_version", 1) != 1:
            raise InvalidProblem("Unsupported result schema_version")
        starts = _vector(candidate.get("start_times"), "start_times")
        order = _vector(candidate.get("order"), "order")
        versions = _vector(candidate.get("buffer_versions"), "buffer_versions", 1)
        ii = _integer(candidate.get("ii"), "result ii")
    except InvalidProblem as exc:
        return {"valid": False, "errors": [str(exc)]}
    if len(starts) != n or sorted(order) != list(range(n)):
        return {
            "valid": False,
            "errors": ["Result task coverage/order is not a complete permutation"],
        }
    modulo = p["mode"] == "modulo"
    if (modulo and not p["ii_bounds"][0] <= ii <= p["ii_bounds"][1]) or (not modulo and ii != 0):
        return {"valid": False, "errors": ["Result II is outside the requested domain"]}
    if modulo and (not ii or ii < max(p["iis"])):
        return {"valid": False, "errors": ["Modulo II cannot cover a task's issue interval"]}
    if len(versions) != len(p["buffer_sizes"]):
        return {"valid": False, "errors": ["Buffer version coverage is incomplete"]}
    phases = [value % ii for value in starts] if modulo else starts
    stages = [value // ii for value in starts] if modulo else [0] * n
    canonical = sorted(range(n), key=lambda i: (phases[i], -stages[i], i))
    if order != canonical:
        errors.append("Order differs from physical phase/stage/source ordering")
    rank = {task: index for index, task in enumerate(order)}
    for task, start in enumerate(starts):
        if start + p["latencies"][task] > INT64_MAX:
            errors.append(f"Task {task} completion overflows int64")
    for u, v, distance, gap in p["data_deps"]:
        actual_distance = versions[-distance - 1] if distance < 0 else distance
        if starts[v] + ii * actual_distance < starts[u] + gap:
            errors.append(f"Data dependency {u}->{v} is not covered")
        if u != v and distance == 0 and not gap and starts[u] == starts[v] and rank[u] >= rank[v]:
            errors.append(f"Zero-gap dependency {u}->{v} is reversed at a tie")
    # Recompute mask conflicts independently of the solver's pair constructor.
    pairs = {tuple(sorted(pair)) for pair in p["resource_deps"]}
    for left in range(n):
        for right in range(left + 1, n):
            if p["resource_flags"][left] & p["resource_flags"][right]:
                pairs.add((left, right))
    for u, v in sorted(pairs):
        forward = phases[v] - phases[u] >= p["iis"][u]
        backward = phases[u] - phases[v] >= p["iis"][v]
        if modulo:
            forward = forward and phases[u] - phases[v] + ii >= p["iis"][v]
            backward = backward and phases[v] - phases[u] + ii >= p["iis"][u]
        if not (forward or backward):
            errors.append(f"Resource exclusion {u},{v} overlaps")
    for u, v, forward_gap, backward_gap in p["owner_exclusion_deps"]:
        forward = phases[v] - phases[u] >= forward_gap
        backward = phases[u] - phases[v] >= backward_gap
        if modulo:
            forward = forward and phases[u] - phases[v] + ii >= backward_gap
            backward = backward and phases[v] - phases[u] + ii >= forward_gap
        if not (forward or backward):
            errors.append(f"Owner exclusion {u},{v} overlaps")
    for u, v in p["pipe_order_deps"]:
        if phases[u] > phases[v] or (phases[u] == phases[v] and rank[u] > rank[v]):
            errors.append(f"Pipe order {u}->{v} is reversed")
    for u, v in p["stage_order_deps"]:
        if stages[u] > stages[v]:
            errors.append(f"Stage order {u}->{v} is reversed")
    if p["manual_stages"] and stages != p["manual_stages"]:
        errors.append("Result does not preserve raw manual stages")
    if modulo and not p["enable_offset"]:
        for u in range(n):
            for v in range(u + 1, n):
                if p["resource_flags"][u] & p["resource_flags"][v] and abs(starts[u] - starts[v]) >= ii:
                    errors.append(f"Resource {u},{v} leaves its single II window")
    for u, v, wrap, gap in p["register_phase_deps"]:
        if phases[v] + wrap * ii < phases[u] + gap:
            errors.append(f"Register phase dependency {u}->{v} is not covered")
        if u != v and not wrap and not gap and phases[u] == phases[v] and rank[u] >= rank[v]:
            errors.append(f"Register phase dependency {u}->{v} is reversed at a tie")
    if any(version > p["num_stages"] for version in versions):
        errors.append("A buffer version count exceeds num_stages")
    for group in p["memory_groups"]:
        occupied = sum(p["buffer_sizes"][index] * versions[index] for index in group[1:])
        if occupied > group[0] or occupied > INT64_MAX:
            errors.append("Memory group capacity is exceeded")
    return {"valid": not errors, "errors": errors}


def _resource_pairs(p):
    pairs = {tuple(sorted(pair)) for pair in p["resource_deps"]}
    flags = p["resource_flags"]
    pairs.update((u, v) for u in range(len(flags)) for v in range(u + 1, len(flags)) if flags[u] & flags[v])
    return sorted(pairs)


def _remaining_ms(deadline):
    return max(0, int((deadline - _clock()) * 1000))


def _check_solver(solver, z3, deadline):
    remaining = _remaining_ms(deadline)
    if not remaining:
        return "TIMEOUT", "Scheduling deadline expired before solver.check()"
    solver.set(timeout=remaining)
    answer = solver.check()
    if not _remaining_ms(deadline):
        return "TIMEOUT", "Scheduling deadline expired during solver.check()"
    if answer == z3.sat:
        return "SAT", ""
    if answer == z3.unsat:
        return "UNSAT", ""
    reason = solver.reason_unknown()
    if "timeout" in reason.lower() or not _remaining_ms(deadline):
        return "TIMEOUT", reason
    return "UNKNOWN", reason


def _load_z3():
    return importlib.import_module("z3")


def _solve(p, z3, deadline):
    n = len(p["latencies"])
    ctx = z3.Context()
    modulo = p["mode"] == "modulo"
    low, high = p["ii_bounds"] if modulo else (0, 0)
    if modulo:
        low = max(low, max(p["iis"]))
    best = None
    while low <= high:
        if not _remaining_ms(deadline):
            return _result("TIMEOUT", "Scheduling deadline expired while constructing constraints")
        # Integer completion bounds and fixed manual stages can make feasibility
        # non-monotone in II. Only exhaustively rejected domain items prove UNSAT.
        ii = low if modulo else 0
        solver = z3.Solver(ctx=ctx) if modulo else z3.Optimize(ctx=ctx)
        if modulo:
            solver.set(random_seed=p["seed"])
            stages = [z3.Int(f"stage_{i}", ctx) for i in range(n)]
            phases = [z3.Int(f"phase_{i}", ctx) for i in range(n)]
            starts = [stages[i] * ii + phases[i] for i in range(n)]
            for i in range(n):
                solver.add(stages[i] >= 0, phases[i] >= 0, phases[i] < ii)
            for i, stage in enumerate(p["manual_stages"]):
                solver.add(stages[i] == stage)
        else:
            starts = [z3.Int(f"start_{i}", ctx) for i in range(n)]
            stages = [0] * n
            phases = starts
            solver.add(*[start >= 0 for start in starts])
        versions = [z3.Int(f"versions_{i}", ctx) for i in range(len(p["buffer_sizes"]))]
        for version in versions:
            solver.add(version >= 1, version <= p["num_stages"])
        for group in p["memory_groups"]:
            solver.add(z3.Sum([versions[index] * p["buffer_sizes"][index] for index in group[1:]]) <= group[0])
        for u, v, distance, gap in p["data_deps"]:
            distance_value = versions[-distance - 1] if distance < 0 else distance
            # At an equal issue time, source order breaks ties within a stage.
            strict_gap = max(gap, 1) if distance == 0 and not gap and u > v else gap
            solver.add(starts[v] - starts[u] >= strict_gap - ii * distance_value)
        for u, v in _resource_pairs(p):
            forward = phases[v] - phases[u] >= p["iis"][u]
            backward = phases[u] - phases[v] >= p["iis"][v]
            if modulo:
                forward = z3.And(forward, phases[u] - phases[v] + ii >= p["iis"][v])
                backward = z3.And(backward, phases[v] - phases[u] + ii >= p["iis"][u])
            solver.add(z3.Or(forward, backward))
        for u, v, gap_uv, gap_vu in p["owner_exclusion_deps"]:
            forward = phases[v] - phases[u] >= gap_uv
            backward = phases[u] - phases[v] >= gap_vu
            if modulo:
                forward = z3.And(forward, phases[u] - phases[v] + ii >= gap_vu)
                backward = z3.And(backward, phases[v] - phases[u] + ii >= gap_uv)
            solver.add(z3.Or(forward, backward))
        for u, v in p["pipe_order_deps"]:
            solver.add(phases[u] <= phases[v])
            if u != v:
                solver.add(
                    z3.Or(
                        phases[u] < phases[v],
                        stages[u] > stages[v],
                        z3.And(stages[u] == stages[v], u < v),
                    )
                )
        for u, v in p["stage_order_deps"]:
            solver.add(stages[u] <= stages[v])
        if modulo and not p["enable_offset"]:
            for u in range(n):
                for v in range(u + 1, n):
                    if p["resource_flags"][u] & p["resource_flags"][v]:
                        solver.add(starts[u] - starts[v] < ii, starts[v] - starts[u] < ii)
        for u, v, wrap, gap in p["register_phase_deps"]:
            solver.add(phases[v] + wrap * ii >= phases[u] + gap)
            if u != v and not wrap and not gap:
                solver.add(
                    z3.Or(
                        phases[u] < phases[v],
                        stages[u] > stages[v],
                        z3.And(stages[u] == stages[v], u < v),
                    )
                )
        if not modulo:
            makespan = z3.Int("makespan", ctx)
            solver.add(*[makespan >= starts[i] + p["latencies"][i] for i in range(n)])
            solver.minimize(makespan)
        for task in range(n):
            solver.add(starts[task] + p["latencies"][task] <= INT64_MAX)
        status, reason = _check_solver(solver, z3, deadline)
        if status not in ("SAT", "UNSAT"):
            return _result(status, reason)
        if status == "SAT":
            model = solver.model()
            values = [model.eval(start).as_long() for start in starts]
            buffer_versions = [model.eval(version).as_long() for version in versions]
            keys = [(values[i] % ii, -(values[i] // ii), i) if modulo else (values[i], 0, i) for i in range(n)]
            best = _result(
                "SAT",
                starts=values,
                order=sorted(range(n), key=keys.__getitem__),
                ii=ii,
                versions=buffer_versions,
            )
            break
        else:
            low += 1
    if best is None:
        return _result(
            "UNSAT",
            f"No feasible schedule in inclusive II domain {p['ii_bounds']}" if modulo else "Constraints are unsatisfiable",
        )
    checked = verify_schedule(p, best)
    if not _remaining_ms(deadline):
        return _result("TIMEOUT", "Scheduling deadline expired during result verification")
    if not checked["valid"]:
        return _result("ERROR", "Independent verifier rejected solver output: " + "; ".join(checked["errors"]))
    return best


def solve_schedule(problem: Mapping) -> dict:
    """Solve a numeric request; malformed requests raise :class:`InvalidProblem`."""
    p = validate_problem(problem)
    deadline = _clock() + p["timeout_ms"] / 1000
    try:
        z3 = _load_z3()
    except ModuleNotFoundError as exc:
        if exc.name == "z3":
            return _result(
                "MISSING_DEPENDENCY",
                "Install optional z3-solver>=4.13.0,<4.15.5 (ascend-scheduler extra)",
            )
        return _result("ERROR", f"Cannot load Z3: {type(exc).__name__}: {exc}")
    except Exception as exc:
        return _result("ERROR", f"Cannot load Z3: {type(exc).__name__}: {exc}")
    remaining = max(0.0, deadline - _clock())
    if not remaining or not _lock.acquire(timeout=remaining):
        return _result("TIMEOUT", "Scheduling deadline expired before acquiring the solver lock")
    try:
        return _solve(p, z3, deadline)
    except Exception as exc:
        return _result("ERROR", f"Solver error: {type(exc).__name__}: {exc}")
    finally:
        _lock.release()
