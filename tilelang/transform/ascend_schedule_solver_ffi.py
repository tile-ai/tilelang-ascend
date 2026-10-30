"""Explicit old-TVM FFI registration for the optional numeric scheduler."""

from __future__ import annotations


def _plain(value):
    """Accept numeric TVM containers without evaluating symbolic TIR."""
    from tilelang import tvm

    if isinstance(value, tvm.tir.IntImm):
        return bool(value.value) if value.dtype == "bool" else int(value.value)
    if isinstance(value, (tvm.ir.Array, list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, tvm.ir.DictAttrs):
        return {key: _plain(value[key]) for key in value.keys()}
    if isinstance(value, (tvm.ir.Map, dict)):
        result = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError("Solver FFI mapping field names must be strings")
            result[key] = _plain(item)
        return result
    if isinstance(value, (str, int, bool)):
        return value
    raise ValueError(f"Solver FFI requires numeric IntImm values, not {type(value).__name__}")


def register_solver():
    """Register a lazy thunk, preserving an existing user or prior registration."""
    from tilelang import tvm

    if tvm.get_global_func("tl.ascend.verify_schedule", allow_missing=True) is None:

        def verify(problem, candidate):
            from .ascend_z3_scheduler import verify_schedule

            return tvm.runtime.convert(verify_schedule(_plain(problem), _plain(candidate)))

        tvm.register_func("tl.ascend.verify_schedule", verify)

    existing = tvm.get_global_func("tl.ascend.solve_schedule", allow_missing=True)
    if existing is not None:
        return existing

    def solve(problem):
        from .ascend_z3_scheduler import solve_schedule

        return tvm.runtime.convert(solve_schedule(_plain(problem)))

    tvm.register_func("tl.ascend.solve_schedule", solve)
    return tvm.get_global_func("tl.ascend.solve_schedule")
