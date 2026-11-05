"""Contracts for TA FlattenBuffer, including its flat TIRX allocation bridge."""

import pytest

from tilelang import tvm
from tilelang.ascend.a2a3 import transform

tirx = tvm.tirx


def _run(body, buffers=(), attrs=None):
    func = tirx.PrimFunc([b.data for b in buffers], body, buffer_map={b.data: b for b in buffers})
    func = func.with_attr({"global_symbol": "main", **(attrs or {})})
    return transform.FlattenBuffer()(tvm.IRModule.from_expr(func))["main"]


def _nodes(body, node_type):
    found = []
    tirx.stmt_functor.post_order_visit(body, lambda node: found.append(node) if isinstance(node, node_type) else None)
    return found


def test_flatten_strides_offsets_and_preserves_argument_buffer_contract():
    buffer = tirx.decl_buffer((2, 3), "float32", strides=[8, 1], elem_offset=2)
    body = tirx.BufferStore(buffer, tirx.BufferLoad(buffer, [1, 2]), [0, 1])
    after = _run(body, [buffer])
    assert list(after.buffer_map[buffer.data].shape) == list(buffer.shape)
    store = _nodes(after.body, tirx.BufferStore)[0]
    load = _nodes(after.body, tirx.BufferLoad)[0]
    assert [int(x) for x in store.indices] == [3]
    assert [int(x) for x in load.indices] == [12]
    assert len(store.buffer.shape) == 1
    # TA retains the flattened buffer elem_offset.  Do not silently import
    # the main fork's later elem_offset=0 change into this migration.
    assert int(store.buffer.elem_offset) == 2


def test_flatten_bool_allocations_casts_and_initializer():
    buffer = tirx.decl_buffer((2, 3), "bool", scope="local.var")
    body = tirx.SeqStmt(
        [
            tirx.AllocBuffer(buffer),
            tirx.BufferStore(buffer, tirx.BufferLoad(buffer, [1, 2]), [0, 1]),
        ]
    )
    after = _run(body, attrs={"tl.local_var_init": {buffer.data: tirx.IntImm("int32", 1)}})
    alloc = _nodes(after.body, tirx.AllocBuffer)[0]
    assert [int(x) for x in alloc.buffer.shape] == [6]
    assert str(alloc.buffer.dtype) == "int8"
    assert int(alloc.annotations["tl.local_var_init"]) == 1
    store = _nodes(after.body, tirx.BufferStore)[0]
    assert str(store.value.dtype) == "int8"
    casts = _nodes(after.body, tirx.Cast)
    assert {str(x.dtype) for x in casts} == {"bool", "int8"}


def test_flatten_matching_decl_preserves_axis_separators():
    buffer = tirx.decl_buffer((2, 3, 4), "float32", axis_separators=[1], scope="shared")
    body = tirx.SeqStmt(
        [
            tirx.AllocBuffer(buffer),
            tirx.DeclBuffer(buffer),
            tirx.BufferStore(buffer, tirx.FloatImm("float32", 0), [1, 2, 3]),
        ]
    )
    after = _run(body)
    assert [int(x) for x in _nodes(after.body, tirx.AllocBuffer)[0].buffer.shape] == [2, 12]
    assert [int(x) for x in _nodes(after.body, tirx.BufferStore)[0].indices] == [1, 11]
    assert [int(x) for x in _nodes(after.body, tirx.DeclBuffer)[0].buffer.shape] == [2, 12]


def test_flatten_mismatched_physical_allocation_uses_extent_product():
    view = tirx.decl_buffer((2, 3), "float32", strides=[8, 1], scope="shared")
    allocation = tirx.decl_buffer((2, 8), "float32", data=view.data, scope="shared")
    body = tirx.SeqStmt(
        [
            tirx.AllocBuffer(allocation),
            tirx.SeqStmt([tirx.DeclBuffer(view), tirx.BufferStore(view, tirx.FloatImm("float32", 0), [1, 2])]),
        ]
    )
    after = _run(body)
    assert [int(x) for x in _nodes(after.body, tirx.AllocBuffer)[0].buffer.shape] == [16]
    assert [int(x) for x in _nodes(after.body, tirx.BufferStore)[0].indices] == [10]


def test_flatten_mismatched_multiaxis_declaration_is_rejected():
    view = tirx.decl_buffer((2, 3), "float32", axis_separators=[1], scope="shared")
    allocation = tirx.decl_buffer((2, 4), "float32", data=view.data, scope="shared")
    body = tirx.SeqStmt([tirx.AllocBuffer(allocation), tirx.DeclBuffer(view), tirx.Evaluate(0)])
    with pytest.raises(tvm.error.InternalError, match="doesn't match Allocate extents"):
        _run(body)


def test_flatten_overflow_promotes_signed_indices():
    buffer = tirx.decl_buffer((65536, 65536), "float32")
    i, j = tirx.Var("i", "int32"), tirx.Var("j", "int32")
    body = tirx.For(
        i, 0, 65536, tirx.ForKind.SERIAL, tirx.For(j, 0, 65536, tirx.ForKind.SERIAL, tirx.Evaluate(tirx.BufferLoad(buffer, [i, j])))
    )
    after = _run(body, [buffer])
    load = _nodes(after.body, tirx.BufferLoad)[0]
    assert str(load.indices[0].dtype) == "int64"
    assert str(_nodes(after.body, tirx.For)[0].loop_var.dtype) == "int32"


def test_flatten_regions_and_legacy_schedule_gate():
    buffer = tirx.decl_buffer((2, 3), "float32")
    region = tirx.BufferRegion(buffer, [tvm.ir.Range(0, 2), tvm.ir.Range(1, 3)])
    block = tirx.SBlock([], [region], [region], "root", tirx.Evaluate(0), alloc_buffers=[buffer])
    body = tirx.SBlockRealize([], True, block)
    after = _run(body)
    flat = _nodes(after.body, tirx.SBlock)[0]
    assert [int(x) for x in flat.alloc_buffers[0].shape] == [6]
    assert int(flat.reads[0].region[0].min) == 1
    assert int(flat.reads[0].region[0].extent) == 5
    gated = _run(body, attrs={"from_legacy_te_schedule": True})
    tvm.ir.assert_structural_equal(gated.body, body)


def test_flatten_address_of_keeps_boolean_load_and_flattens_coordinates():
    buffer = tirx.decl_buffer((2, 3), "bool")
    ptr = tirx.call_intrin("handle", "tirx.address_of", tirx.BufferLoad(buffer, [1, 2]))
    after = _run(tirx.Evaluate(ptr), [buffer])
    load = _nodes(after.body, tirx.BufferLoad)[0]
    assert str(load.buffer.dtype) == "int8"
    assert str(load.dtype) == "bool"
    assert [int(x) for x in load.indices] == [5]
    assert not _nodes(after.body, tirx.Cast)


def test_flatten_dynamic_allocation_keeps_symbolic_extent():
    n = tirx.Var("n", "int32")
    buffer = tirx.decl_buffer((2, n), "float32", scope="shared")
    body = tirx.SeqStmt([tirx.AllocBuffer(buffer), tirx.Evaluate(tirx.BufferLoad(buffer, [1, 0]))])
    after = _run(body)
    allocation = _nodes(after.body, tirx.AllocBuffer)[0]
    analyzer = tvm.arith.Analyzer()
    assert analyzer.can_prove_equal(allocation.buffer.shape[0], n * 2)
    load = _nodes(after.body, tirx.BufferLoad)[0]
    assert analyzer.can_prove_equal(load.indices[0], n)
