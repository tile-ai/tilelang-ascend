"""Focused contracts for the original TA safe pass."""

import pytest

from tilelang import tvm
from tilelang.ascend.a2a3 import transform

tirx = tvm.tirx


def _run(factory, body, buffers=(), attrs=None, config=None):
    func = tirx.PrimFunc([buffer.data for buffer in buffers], body, buffer_map={b.data: b for b in buffers})
    func = func.with_attr({"global_symbol": "main", **(attrs or {})})
    with tvm.transform.PassContext(config=config or {}):
        return factory()(tvm.IRModule.from_expr(func))["main"]


def _nodes(body, node_type):
    found = []
    tirx.stmt_functor.post_order_visit(body, lambda node: found.append(node) if isinstance(node, node_type) else None)
    return found


@pytest.mark.parametrize("scope", ["shared.ub", "local.fragment", "local"])
def test_safe_load_padding_preserves_ta_padding_map(scope):
    src = tirx.decl_buffer((8,), "float32", name="src")
    dst = tirx.decl_buffer((16,), "float32", name="dst", scope=scope)
    index = tirx.Var("i", "int32")
    loop = tirx.For(index, 0, 16, tirx.ForKind.SERIAL, tirx.BufferStore(dst, tirx.BufferLoad(src, [index]), [index]))
    block = tirx.SBlock(
        [], [], [], "root", loop, alloc_buffers=[dst], annotations={"padding_map": {dst.data: tirx.FloatImm("float32", -3)}}
    )
    after = _run(transform.LegalizeSafeMemoryAccess, tirx.SBlockRealize([], True, block), buffers=[src])
    store = _nodes(after.body, tirx.BufferStore)[0]
    assert isinstance(store.value, tirx.Call)
    assert store.value.op.same_as(tvm.ir.Op.get("tirx.if_then_else"))
    assert float(store.value.args[2]) == -3


def test_safe_global_store_and_extern_call_guards():
    src = tirx.decl_buffer((8,), "float32", name="src")
    index = tirx.Var("i", "int32")
    body = tirx.BufferStore(src, tirx.FloatImm("float32", 1), [index])
    loop = tirx.For(index, 0, 16, tirx.ForKind.SERIAL, body)
    after = _run(transform.LegalizeSafeMemoryAccess, loop, buffers=[src])
    assert _nodes(after.body, tirx.IfThenElse)
    call = tirx.call_extern("int32", "user_fn", tirx.BufferLoad(src, [index]))
    after = _run(transform.LegalizeSafeMemoryAccess, tirx.For(index, 0, 16, tirx.ForKind.SERIAL, tirx.Evaluate(call)), buffers=[src])
    assert _nodes(after.body, tirx.IfThenElse)


def test_safe_disabled_and_constant_index_follow_original_boundaries():
    src = tirx.decl_buffer((8,), "float32")
    index = tirx.Var("i", "int32")
    body = tirx.For(index, 0, 16, tirx.ForKind.SERIAL, tirx.BufferStore(src, tirx.FloatImm("float32", 1), [index]))
    disabled = _run(transform.LegalizeSafeMemoryAccess, body, buffers=[src], config={"tl.disable_safe_memory_legalize": True})
    tvm.ir.assert_structural_equal(disabled.body, body)
    # TA explicitly skips constant indices in its checker; do not silently
    # turn the migration into a different bounds validation policy.
    constant = tirx.For(index, 0, 16, tirx.ForKind.SERIAL, tirx.BufferStore(src, tirx.FloatImm("float32", 1), [9]))
    after = _run(transform.LegalizeSafeMemoryAccess, constant, buffers=[src])
    assert not _nodes(after.body, tirx.IfThenElse)
