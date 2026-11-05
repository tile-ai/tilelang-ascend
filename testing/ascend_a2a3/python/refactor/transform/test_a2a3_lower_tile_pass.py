"""TA LowerTileOp layout geometry and existing Ascend Op contracts."""

import pytest

from tilelang import tvm
from tilelang.ascend.a2a3.layout import Layout, make_zn_layout, make_nz_layout

tirx = tvm.tirx


def _run(body, buffers=(), inference=False):
    func = tirx.PrimFunc([b.data for b in buffers], body, buffer_map={b.data: b for b in buffers})
    func = func.with_attr({"global_symbol": "main", "target": tvm.target.Target({"kind": "ascend", "arch": "dav-2201"})})
    module = tvm.IRModule.from_expr(func)
    if inference:
        module = tvm.ffi.get_global_func("tl.transform.A2A3LayoutInference")()(module)
    return tvm.ffi.get_global_func("tl.transform.A2A3LowerTileOp")()(module)["main"]


def _nodes(body, typ):
    out = []
    tirx.stmt_functor.post_order_visit(body, lambda x: out.append(x) if isinstance(x, typ) else None)
    return out


def _root(body, buffers=(), annotations=None):
    block = tirx.SBlock([], [], [], "root", body, alloc_buffers=list(buffers), annotations=annotations or {})
    return tirx.SBlockRealize([], True, block)


@pytest.mark.parametrize("dtype", ["float16", "float32"])
@pytest.mark.parametrize("shape", [(7, 9), (200, 128), (2, 17, 33)])
@pytest.mark.parametrize("factory", [make_zn_layout, make_nz_layout])
def test_lower_tile_preserves_whole_static_fractal_tail_allocation(shape, factory, dtype):
    buffer = tirx.decl_buffer(shape, dtype, scope="shared.l1")
    layout = factory(buffer)
    indices = [n - 1 for n in shape]
    store = tirx.BufferStore(buffer, tirx.FloatImm(dtype, 1), indices)
    after = _run(_root(store, [buffer], {"layout_map": {buffer: layout}}))
    block = _nodes(after.body, tirx.SBlock)[0]
    rows, cols = shape[-2:]
    c0 = 256 // tvm.DataType(dtype).bits
    if factory is make_zn_layout:
        extent = ((cols + c0 - 1) // c0) * ((rows + 15) // 16 * 16) * c0
    else:
        extent = ((rows + c0 - 1) // c0) * ((cols + 15) // 16 * 16) * c0
    assert [int(x) for x in block.alloc_buffers[0].shape] == [*shape[:-2], extent]
    store = _nodes(after.body, tirx.BufferStore)[0]
    mapped = layout.map_forward_index(indices)
    assert len(store.indices) == len(mapped)
    assert all(tvm.arith.Analyzer().can_prove_equal(actual, expected) for actual, expected in zip(store.indices, mapped))


def test_lower_tile_column_major_and_padding_map_remap():
    buffer = tirx.decl_buffer((3, 5), "float16", scope="local.fragment")
    layout = Layout((3, 5), lambda i, j: [j, i], layout_tag=1)
    body = _root(
        tirx.BufferStore(buffer, tirx.FloatImm("float16", 1), [2, 4]),
        [buffer],
        {"layout_map": {buffer: layout}, "padding_map": {buffer.data: tirx.IntImm("int32", 0)}},
    )
    after = _run(body)
    block = _nodes(after.body, tirx.SBlock)[0]
    mapped = block.alloc_buffers[0]
    assert mapped.scope() == "local"
    assert [int(x) for x in mapped.shape] == [5, 3]
    assert [int(x) for x in _nodes(after.body, tirx.BufferStore)[0].indices] == [4, 2]
    assert not mapped.data.same_as(buffer.data)
    assert mapped.data in block.annotations["padding_map"]
    assert buffer.data not in block.annotations["padding_map"]


def test_lower_tile_inferred_fragment_uses_per_thread_shape_without_matrix_tag():
    source = tirx.decl_buffer((32,), "float32")
    fragment = tirx.decl_buffer((32,), "float32", scope="local.fragment")
    i = tirx.Var("i", "int32")
    body = _root(
        tirx.For(i, 0, 32, tirx.ForKind.PARALLEL, tirx.BufferStore(fragment, tirx.BufferLoad(source, [i]), [i])),
        [fragment],
        {"tilelang.is_npu_kernel_frame_dev_mode": True},
    )
    vid = tirx.Var("vid", "int32")
    binding = tirx.IterVar(tvm.ir.Range(0, 2), vid, tirx.IterVar.ThreadIndex, "cthread")
    after = _run(tirx.AttrStmt(binding, "thread_extent", 2, body), [source], inference=True)
    block = _nodes(after.body, tirx.SBlock)[0]
    assert block.alloc_buffers[0].scope() == "local"
    assert [int(x) for x in block.alloc_buffers[0].shape] == [16]
    assert _nodes(after.body, tirx.BufferStore)[0].buffer.same_as(block.alloc_buffers[0])


def test_lower_tile_flat_alloc_buffer_preserves_physical_shape_and_view_identity():
    buffer = tirx.decl_buffer((7, 9), "float16", scope="shared.l1")
    layout = make_zn_layout(buffer)
    body = tirx.SeqStmt([tirx.AllocBuffer(buffer), tirx.DeclBuffer(buffer), tirx.BufferStore(buffer, tirx.FloatImm("float16", 1), [6, 8])])
    after = _run(_root(body, annotations={"layout_map": {buffer: layout}}))
    allocation = _nodes(after.body, tirx.AllocBuffer)[0]
    assert [int(x) for x in allocation.buffer.shape] == [256]
    assert _nodes(after.body, tirx.BufferStore)[0].buffer.same_as(allocation.buffer)
    assert _nodes(after.body, tirx.DeclBuffer)[0].buffer.same_as(allocation.buffer)


def test_lower_tile_existing_ascend_copy_uses_original_lowering():
    import tilelang.ascend.a2a3.language as T

    source = tirx.decl_buffer((16, 16), "float16", scope="global")
    destination = tirx.decl_buffer((16, 16), "float16", scope="shared.ub")
    call = T.copy(source, destination)
    assert call.op.name == "tl.ascend_copy"
    after = _run(_root(tirx.Evaluate(call), [destination]), [source])
    assert not any(isinstance(call.op, tvm.ir.Op) and call.op.name == "tl.ascend_copy" for call in _nodes(after.body, tirx.Call))
    assert any(call.op.name == "tirx.call_extern" for call in _nodes(after.body, tirx.Call) if isinstance(call.op, tvm.ir.Op))


@pytest.mark.parametrize("tag,shape,physical", [(0, (3, 5), [3, 5]), (3, (17, 33), [1536])])
def test_lower_tile_row_major_and_zz_allocation_contract(tag, shape, physical):
    buffer = tirx.decl_buffer(shape, "float16", scope="shared.l1")
    layout = Layout(shape, (lambda i, j: [i, j]) if tag == 0 else (lambda i, j: [i * 33 + j]), layout_tag=tag)
    body = _root(
        tirx.BufferStore(buffer, tirx.FloatImm("float16", 1), [shape[0] - 1, shape[1] - 1]), [buffer], {"layout_map": {buffer: layout}}
    )
    after = _run(body)
    assert [int(x) for x in _nodes(after.body, tirx.SBlock)[0].alloc_buffers[0].shape] == physical
