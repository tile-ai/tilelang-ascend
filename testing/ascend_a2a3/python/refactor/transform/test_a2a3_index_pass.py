"""Focused contracts for the original TA index pass."""

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


@pytest.mark.parametrize("bits", [32, 64])
def test_index_config_rewrites_symbolic_indices_casts_and_immediates(bits):
    buffer = tirx.decl_buffer((32,), "int32")
    index = tirx.Var("i", "int32")
    offset = tirx.Cast("int32", index + 1)
    store = tirx.BufferStore(buffer, tirx.BufferLoad(buffer, [offset]), [index + 2])
    body = tirx.For(index, 0, 16, tirx.ForKind.SERIAL, store)
    after = _run(transform.ConfigIndexBitwidth, body, buffers=[buffer], config={"tl.config_index_bitwidth": bits})
    # TA promotes integer Vars to int64 independently of the requested
    # immediate/cast width. Binary type legalization follows the Var width.
    loop = _nodes(after.body, tirx.For)[0]
    assert str(loop.loop_var.dtype) == "int64"
    store = _nodes(after.body, tirx.BufferStore)[0]
    assert str(store.indices[0].dtype) == "int64"
    load = _nodes(after.body, tirx.BufferLoad)[0]
    assert str(load.indices[0].dtype) == f"int{bits}"


def test_index_without_config_is_unchanged():
    buffer = tirx.decl_buffer((4,), "int32")
    index = tirx.Var("i", "int32")
    body = tirx.For(index, 0, 4, tirx.ForKind.SERIAL, tirx.BufferStore(buffer, 1, [index]))
    after = _run(transform.ConfigIndexBitwidth, body, buffers=[buffer])
    tvm.ir.assert_structural_equal(after.body, body)
