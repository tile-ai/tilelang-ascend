"""A2A3 raw SyncInsert configuration, platform and flat-IR contracts."""

import pytest

from tilelang import tvm
from test_a2a3_sync_insert_pass import _buffers, _copy, _events, _fill, _operations, _run, tir


def _raw(body, enabled=None, platform="A3", dialect="ascendc"):
    func = tir.PrimFunc([], body)
    target = tvm.target.Target({"kind": "ascend", "arch": "dav-2201", "keys": [dialect]})
    config = {} if enabled is None else {"tl.ascend_auto_sync": enabled}
    with tvm.transform.PassContext(config=config):
        return tvm.ffi.get_global_func("tl.transform.A2A3AscendSyncInsert")(target, platform)(tvm.IRModule.from_expr(func))["main"]


@pytest.mark.parametrize("enabled", [None, False])
def test_sync_default_and_disabled_keep_body(enabled):
    a = _buffers()["a"]
    body = tir.SeqStmt([_fill(a), _fill(a)])
    assert _raw(body, enabled=enabled).body.same_as(body)


@pytest.mark.parametrize("platform,has_barrier", [("A2", True), ("A3", True), ("A5", False), ("auto", True), ("custom", True)])
@pytest.mark.parametrize("dialect", ["ascendc", "pto"])
def test_sync_raw_platform_v_barrier_source_branch(platform, has_barrier, dialect):
    a = _buffers()["a"]
    after = _raw(tir.SeqStmt([_fill(a), _fill(a)]), enabled=True, platform=platform, dialect=dialect)
    barriers = [op for op in _operations(after) if isinstance(op, tir.Call) and op.op == tvm.ir.Op.get("tl.ascend_auto_barrier")]
    assert bool(barriers) == has_barrier
    if barriers:
        assert len(barriers[0].args) == 1
        assert barriers[0].args[0].value == "PIPE_V"


@pytest.mark.parametrize("sides,needs_event", [((1, 1), True), ((0, 0), True), ((0, 1), False)])
def test_sync_resource_scope_preserves_same_stream_and_separates_other_side(sides, needs_event):
    buffers = _buffers()
    first = tir.AttrStmt(0, "resource_scope", sides[0], _copy("copy_ub_to_gm", buffers["a"], buffers["output"]))
    second = tir.AttrStmt(0, "resource_scope", sides[1], _fill(buffers["b"]))
    # An enclosing same-side resource stream matches how CombineCV emits it.
    enclosing = tir.AttrStmt(0, "resource_scope", sides[0], tir.SeqStmt([first, second]))
    result = _run(buffers, [enclosing])
    assert ("MTE3_V" in _events(result)) == needs_event
    for operation in _operations(result):
        if isinstance(operation, tir.Call) and operation.op in (
            tvm.ir.Op.get("tl.ascend_auto_set_flag"),
            tvm.ir.Op.get("tl.ascend_auto_wait_flag"),
        ):
            assert len(operation.args) == 2


def test_sync_bind_buffer_load_keeps_source_all_barrier_before_value():
    buffers = _buffers()
    value = tir.Var("value", "float32")
    result = _run(buffers, [tir.SeqStmt([tir.Bind(value, tir.BufferLoad(buffers["a"], [0])), tir.Evaluate(value)])])
    operations = _operations(result)
    barriers = [op for op in operations if isinstance(op, tir.Call) and op.op == tvm.ir.Op.get("tl.ascend_auto_barrier")]
    assert len(barriers) == 1 and barriers[0].args[0].value == "PIPE_ALL"
    bindings = []
    tir.stmt_functor.post_order_visit(result.body, lambda n: bindings.append(n) if isinstance(n, tir.Bind) else None)
    assert len(bindings) == 1 and bindings[0].var.same_as(value)


def test_sync_loop_merges_flat_allocation_and_declaration_tails():
    buffers = _buffers()
    local = tir.decl_buffer((64,), "float32", name="loop_tmp", scope="shared.ub")
    body = tir.SeqStmt(
        [
            tir.AllocBuffer(local),
            tir.DeclBuffer(local),
            _copy("copy_gm_to_ub", buffers["input"], local),
            _copy("copy_ub_to_gm", local, buffers["output"]),
        ]
    )
    result = _run(buffers, [tir.For(tir.Var("i", "int32"), 0, 2, tir.ForKind.SERIAL, body)])
    assert "MTE2_MTE3" in _events(result)
    allocations = []
    tir.stmt_functor.post_order_visit(result.body, lambda n: allocations.append(n) if isinstance(n, tir.AllocBuffer) else None)
    assert sum(a.buffer.data.same_as(local.data) for a in allocations) == 1


def test_sync_preserves_shared_manual_two_operand_event_protocol():
    operations = [
        tir.Evaluate(tir.call_intrin("handle", tvm.ir.Op.get("tl.ascend_set_flag"), "MTE2_V", 3)),
        tir.Evaluate(tir.call_intrin("handle", tvm.ir.Op.get("tl.ascend_wait_flag"), "MTE2_V", 3)),
        tir.Evaluate(tir.call_intrin("handle", tvm.ir.Op.get("tl.ascend_pipe_barrier"), "PIPE_V")),
    ]
    after = _raw(tir.SeqStmt(operations), enabled=True)
    calls = _operations(after)
    assert len(calls) == 3
    assert len(calls[0].args) == len(calls[1].args) == 2
    assert calls[0].args[0].value == calls[1].args[0].value == "MTE2_V"
    assert calls[2].args[0].value == "PIPE_V"
    for actual, original in zip(calls, operations):
        assert actual.same_as(original.value)
