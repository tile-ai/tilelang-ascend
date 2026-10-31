# Copyright (c) Tile-AI Organization.
# Licensed under the MIT License.
"""Persistent trip-count equivalence and real upstream synchronization tests."""

from collections import Counter

import pytest

import tilelang
from tilelang import language as T
from tilelang import tvm

tir = tvm.tir
_WORKERS = 4
_WIDTH = 64
_PASS_CONFIGS = {
    "tl.ascend_sync_backend": "upstream",
    "tl.ascend_auto_sync": True,
    "tl.ascend_auto_sync_vs": True,
    "tl.ascend_auto_cv_combine": False,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: False,
    "tl.ascend_memory_planning": False,
}


def _persistent_abs(rows: int | None, *, nested: bool = False) -> tir.PrimFunc:
    # Follow the existing persistent language tests and single-V device
    # copy/abs fixtures. Each AIV owns a separate complete output plane.
    count = T.symbolic("rows") if rows is None else rows

    @T.prim_func
    def main(
        A: T.Tensor((2, count, _WIDTH), "float32"),
        O: T.Tensor((2, count, _WIDTH), "float32"),
    ):
        with T.Kernel(_WORKERS, is_npu=True) as (cid, vid):
            for row, _column in T.Persistent([count, 1], _WORKERS, cid, group_size=1):
                with T.Scope("V"):
                    local = T.alloc_ub((_WIDTH,), "float32")
                    for _repeat in T.serial(2 if nested else 1):
                        T.copy(A[vid, row, :], local)
                        T.tile.abs(local, local)
                        T.copy(local, O[vid, row, :])

    return main


def _nodes(function: tir.PrimFunc, kind: type) -> list:
    result = []
    tir.stmt_functor.post_order_visit(function.body, lambda node: result.append(node) if isinstance(node, kind) else None)
    return result


def _breaks(function: tir.PrimFunc) -> list[tir.Call]:
    return [call for call in _nodes(function, tir.Call) if call.op.name == "tl.loop_break"]


def _normalize(function: tir.PrimFunc) -> tir.PrimFunc:
    return tvm.get_global_func("tl.ascend_sync.NormalizeLegacyPersistent")(function)


def _canonical(
    domain: int = 9,
    wave: int = 4,
    *,
    divisor: int = 1,
    shell: str = "plain",
    nested: bool = False,
    mutation: str | None = None,
) -> tir.PrimFunc:
    worker = tir.Var("cid", "int32")
    loop = tir.Var("wave", "int32")
    output = tir.decl_buffer((max(domain * divisor, 1),), "int32", name="output")
    local = tir.decl_buffer((1,), "int32", name="local", scope="local")
    index = loop * (wave * divisor) + worker
    worker_base = worker if divisor == 1 else tir.floordiv(worker, divisor)
    condition = tir.const(domain, "int32") <= loop * wave + worker_base
    stop = tir.Evaluate(tir.call_intrin("handle", tvm.ir.Op.get("tl.loop_break")))
    payload = tir.BufferStore(output, index, [index])
    if nested:
        payload = tir.For(tir.Var("repeat", "int32"), 0, 2, tir.ForKind.SERIAL, payload)
    guard = tir.IfThenElse(condition, stop, None)
    extent = (domain + wave - 1) // wave if wave > 0 else 1
    minimum, kind, worker_extent = 0, tir.ForKind.SERIAL, wave * divisor
    if mutation == "minimum":
        minimum = 1
    elif mutation == "extent":
        extent += 1
    elif mutation == "parallel":
        kind = tir.ForKind.PARALLEL
    elif mutation == "worker_extent":
        worker_extent += 1
    elif mutation == "not_monotone":
        guard = tir.IfThenElse(tir.const(domain, "int32") <= worker - loop * wave, stop, None)
    elif mutation == "else":
        guard = tir.IfThenElse(condition, stop, tir.Evaluate(0))
    elif mutation == "mutable_guard":
        guard = tir.IfThenElse(tir.BufferLoad(output, [0]) <= index, stop, None)
    elif mutation == "equality":
        guard = tir.IfThenElse(loop * wave + worker_base == domain, stop, None)
    body = tir.SeqStmt([guard, payload])
    if mutation == "effect_before_guard":
        body = tir.SeqStmt([tir.BufferStore(output, 7, [0]), guard, payload])
    elif mutation == "second_break":
        body = tir.SeqStmt([guard, payload, stop])
    elif mutation == "mutable_let":
        body = tir.LetStmt(tir.Var("saved", "int32"), tir.BufferLoad(output, [0]), body)
    if shell == "allocate":
        body = tir.Allocate(local.data, local.dtype, [1], tir.const(True, "bool"), body)
    elif shell == "let":
        body = tir.LetStmt(tir.Var("pure", "int32"), loop + 1, body)
    elif shell == "declbuffer":
        body = tir.DeclBuffer(local, body)
    elif shell == "resource":
        body = tir.AttrStmt(tir.IntImm("int32", 0), "resource_scope", 1, body)
    body = tir.For(loop, minimum, extent, kind, body)
    body = tir.AttrStmt(tir.IntImm("int32", 0), "resource_scope", 1, body)
    body = tir.AttrStmt(
        tir.IterVar(None, worker, tir.IterVar.ThreadIndex, "blockIdx.x"),
        "thread_extent",
        worker_extent,
        body,
    )
    return tir.PrimFunc([output.data], body, buffer_map={output.data: output})


def _executed_indices(function: tir.PrimFunc, worker: int) -> list[int]:
    """Interpret real For/If/break semantics independently of the normalizer."""
    analyzer = tvm.arith.Analyzer()
    result = []

    class LoopBreak(Exception):
        pass

    def integer(expression: tir.PrimExpr, values: dict) -> int:
        value = analyzer.simplify(
            tir.stmt_functor.substitute(expression, {var: tir.const(number, var.dtype) for var, number in values.items()})
        )
        assert isinstance(value, tir.IntImm), value
        return int(value)

    def execute(statement: tir.Stmt, values: dict) -> None:
        if isinstance(statement, tir.SeqStmt):
            for child in statement.seq:
                execute(child, values)
        elif isinstance(statement, tir.AttrStmt):
            if statement.attr_key == "thread_extent":
                values = {**values, statement.node.var: worker}
            execute(statement.body, values)
        elif isinstance(statement, (tir.Allocate, tir.DeclBuffer)):
            execute(statement.body, values)
        elif isinstance(statement, tir.LetStmt):
            execute(statement.body, {**values, statement.var: integer(statement.value, values)})
        elif isinstance(statement, tir.For):
            minimum, extent = integer(statement.min, values), integer(statement.extent, values)
            assert extent >= 0
            for iteration in range(minimum, minimum + extent):
                try:
                    execute(statement.body, {**values, statement.loop_var: iteration})
                except LoopBreak:
                    break
        elif isinstance(statement, tir.IfThenElse):
            branch = statement.then_case if integer(statement.condition, values) else statement.else_case
            if branch is not None:
                execute(branch, values)
        elif isinstance(statement, tir.BufferStore):
            result.append(integer(statement.indices[0], values))
        else:
            assert isinstance(statement, tir.Evaluate)
            if isinstance(statement.value, tir.Call):
                assert statement.value.op.name == "tl.loop_break"
                raise LoopBreak

    execute(function.body, {})
    return result


@pytest.mark.parametrize("domain", [4, 5, 8, 9, 17])
@pytest.mark.parametrize("shell", ["plain", "allocate", "let", "declbuffer", "resource"])
def test_static_break_becomes_exact_per_worker_bound(domain: int, shell: str) -> None:
    original = _canonical(domain, shell=shell)
    normalized = _normalize(original)
    assert _breaks(original) and not _breaks(normalized)
    assert _normalize(normalized).same_as(normalized)
    coverage = Counter()
    for worker in range(_WORKERS):
        before = _executed_indices(original, worker)
        after = _executed_indices(normalized, worker)
        assert before == after
        coverage.update(after)
    assert coverage == Counter(range(domain))
    assert len(_nodes(original, tir.Allocate)) == len(_nodes(normalized, tir.Allocate))


def test_nested_ordinary_loop_keeps_iteration_order_and_scope() -> None:
    original = _canonical(nested=True, shell="allocate")
    normalized = _normalize(original)
    assert len(_nodes(normalized, tir.For)) == 2
    outer = _nodes(normalized, tir.For)[-1]
    assert isinstance(outer.body, tir.Allocate)
    for worker in range(_WORKERS):
        assert _executed_indices(original, worker) == _executed_indices(normalized, worker)


@pytest.mark.parametrize("domain,wave,divisor", [(3, 10, 2), (11, 10, 2), (3, 2, 2), (5, 4, 3)])
def test_simplified_divided_worker_guard_preserves_partial_wave_coverage(domain: int, wave: int, divisor: int) -> None:
    original = _canonical(domain, wave, divisor=divisor)
    normalized = _normalize(original)
    coverage = Counter()
    assert not _breaks(normalized)
    for worker in range(wave * divisor):
        before, after = _executed_indices(original, worker), _executed_indices(normalized, worker)
        assert before == after
        coverage.update(after)
    assert coverage == Counter(range(domain * divisor))


def test_divided_worker_requires_matching_actual_hardware_extent() -> None:
    with pytest.raises(tvm.error.TVMError, match="wave size must equal the worker extent"):
        _normalize(_canonical(11, 10, divisor=2, mutation="worker_extent"))


@pytest.mark.parametrize("domain,wave,divisor", [(3, 2, 2), (4, 4, 1)])
def test_equality_guard_requires_proven_equivalence_to_monotone_break(domain: int, wave: int, divisor: int) -> None:
    original = _canonical(domain, wave, divisor=divisor, mutation="equality")
    normalized = _normalize(original)
    assert not _breaks(normalized)
    coverage = Counter()
    for worker in range(wave * divisor):
        before, after = _executed_indices(original, worker), _executed_indices(normalized, worker)
        assert before == after
        coverage.update(after)
    assert coverage == Counter(range(domain * divisor))


@pytest.mark.parametrize(
    "mutation",
    [
        "minimum",
        "extent",
        "parallel",
        "worker_extent",
        "not_monotone",
        "else",
        "mutable_guard",
        "effect_before_guard",
        "second_break",
        "mutable_let",
        "equality",
    ],
)
def test_noncanonical_break_is_rejected(mutation: str) -> None:
    with pytest.raises(tvm.error.TVMError, match="canonical leading monotone break guard"):
        _normalize(_canonical(mutation=mutation))


def test_trip_count_arithmetic_must_not_overflow() -> None:
    with pytest.raises(tvm.error.TVMError, match="could overflow"):
        _normalize(_canonical(domain=2**31 - 1))


def test_body_rebinding_cannot_change_the_original_header_range_certificate() -> None:
    worker, loop, count = (tir.Var(name, "int32") for name in ("cid", "w", "count"))
    output = tir.decl_buffer((12,), "int32", name="output")
    stop = tir.Evaluate(tir.call_intrin("handle", tvm.ir.Op.get("tl.loop_break")))
    guard = tir.IfThenElse(loop * 2 + tir.floordiv(worker, 2) == 3, stop, None)
    payload = tir.BufferStore(output, loop * 4 + worker, [loop * 4 + worker])
    # Deliberately reuse a Var in two lexical scopes. The private normalizer
    # must reject the false certificate even when called outside the frontend
    # SSA pipeline: the original For runs three times, not the body's two.
    body = tir.For(loop, 0, count, tir.ForKind.SERIAL, tir.LetStmt(count, 2, tir.SeqStmt([guard, payload])))
    body = tir.LetStmt(count, 3, body)
    body = tir.AttrStmt(tir.IterVar(None, worker, tir.IterVar.ThreadIndex, "blockIdx.x"), "thread_extent", 4, body)
    function = tir.PrimFunc([output.data], body, buffer_map={output.data: output})
    assert _executed_indices(function, 0) == [0, 4, 8]
    with pytest.raises(tvm.error.TVMError, match="original trip count must equal"):
        _normalize(function)


@pytest.mark.parametrize("rows", [1, 4, None])
def test_partial_first_wave_and_dynamic_predicate_need_no_break_rewrite(rows: int | None) -> None:
    program = _persistent_abs(rows)
    assert not _breaks(program)
    assert _normalize(program).same_as(program)


@pytest.mark.parametrize("rows", [4, 11])
def test_existing_cube_only_persistent_reaches_the_upstream_engine(rows: int) -> None:
    from test_tilelang_ascend_language_persistent import _persistent_kernel

    if not tvm.get_global_func("tl.ascend_sync.Z3ProofServiceAvailable")():
        pytest.skip("requires USE_ASCEND_SYNC_Z3=ON for synchronization")
    with tvm.transform.PassContext(config=_PASS_CONFIGS):
        artifact = tilelang.lower(_persistent_kernel(rows), target="ascendc", platform="A3")
    function = artifact.device_mod.functions_items()[0][1]
    assert "tl.ascend_sync_proof" in function.attrs
    assert not _breaks(function)


@pytest.mark.parametrize("rows", [1, 4, 5, 6, 8, None])
@pytest.mark.parametrize("nested", [False, True])
def test_real_persistent_copy_abs_uses_upstream_engine(rows: int | None, nested: bool) -> None:
    if not tvm.get_global_func("tl.ascend_sync.Z3ProofServiceAvailable")():
        pytest.skip("requires USE_ASCEND_SYNC_Z3=ON for synchronization")
    program = _persistent_abs(rows, nested=nested)
    with tvm.transform.PassContext(config=_PASS_CONFIGS):
        artifact = tilelang.lower(program, target="ascendc", platform="A3")
    function = artifact.device_mod.functions_items()[0][1]
    assert "tl.ascend_sync_proof" in function.attrs
    assert not _breaks(function)
    events = [call.args[0].value for call in _nodes(function, tir.Call) if call.op.name == "tl.ascend_auto_set_flag"]
    assert "MTE2_V" in events and "V_MTE3" in events
