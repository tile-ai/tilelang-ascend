"""Generic TA operators through the real A2/A3 LowerTileOp pass."""

import ctypes
import os

import pytest
from tilelang import tvm
from tilelang.language.copy_op import copy
from tilelang.language.fill_op import fill
from tilelang.language.gemm_op import gemm
from tilelang.layout import Fragment

# Migration-only native modules, before their sources are integrated in libtilelang.
_LIBRARIES = [
    ctypes.CDLL(os.environ[key], mode=ctypes.RTLD_GLOBAL)
    for key in ("A2A3_TEST_GENERIC_LIBRARY", "A2A3_TEST_LOWER_TILE_LIBRARY")
    if os.environ.get(key)
]


def _nodes(stmt, cls):
    found = []
    tvm.tirx.stmt_functor.post_order_visit(stmt, lambda node: found.append(node) if isinstance(node, cls) else None)
    return found


def _lower(call, buffers, layouts=None):
    t = tvm.tirx
    block = t.SBlock([], [], [], "root", t.Evaluate(call), alloc_buffers=list(buffers), annotations={"layout_map": layouts or {}})
    func = t.PrimFunc([], t.SBlockRealize([], True, block)).with_attr("target", tvm.target.Target({"kind": "ascend", "arch": "dav-2201"}))
    before = tvm.IRModule.from_expr(func)
    result = tvm.ffi.get_global_func("tl.transform.A2A3LowerTileOp")()(before)["main"]
    remaining = [
        node.op.name for node in _nodes(result.body, t.Call) if isinstance(node.op, tvm.ir.Op) and node.op.name.startswith("tl.tileop.")
    ]
    assert not remaining, f"A2/A3 operators were not lowered: {remaining}"
    return result


@pytest.mark.parametrize("width", [None, 4])
def test_copy_keeps_the_original_ascend_cpu_proxy_branch(width):
    t = tvm.tirx
    src = t.decl_buffer((16,), "float32", name="src", scope="global")
    dst = t.decl_buffer((16,), "float32", name="dst", scope="shared")
    result = _lower(copy(src, dst, coalesced_width=width), [src, dst])
    loops = _nodes(result.body, t.For)
    assert sorted((int(loop.extent), int(loop.kind)) for loop in loops) == [(4, 1), (4, 2)]
    # TA's proxy target directly vectorizes the logical loop without partitioning
    # it among hardware threads. The native target must retain that behavior.
    stores = _nodes(result.body, t.BufferStore)
    assert len(stores) == 1
    assert stores[0].buffer.data.same_as(dst.data)
    assert stores[0].value.buffer.data.same_as(src.data)
    if width is not None:
        assert any(int(loop.annotations.get("coalesced_width", 0)) == width for loop in loops)


def test_copy_preserves_source_bounds_zero_fill_and_dtype_conversion():
    t = tvm.tirx
    src = t.decl_buffer((8,), "float16", name="src", scope="global")
    dst = t.decl_buffer((8,), "float32", name="dst", scope="shared")
    src_region = t.BufferRegion(src, [tvm.ir.Range.from_min_extent(4, 8)])
    dst_region = t.BufferRegion(dst, [tvm.ir.Range.from_min_extent(0, 8)])
    result = _lower(copy(src_region, dst_region), [src, dst])
    stores = _nodes(result.body, t.BufferStore)
    assert len(stores) == 1
    expr = stores[0].value
    assert isinstance(expr, t.Call)
    assert expr.op.name == "tirx.if_then_else"
    assert isinstance(expr.args[1], t.Cast)
    assert expr.args[1].dtype == "float32"
    assert expr.args[2].value == 0.0


@pytest.mark.parametrize("scope", ["local", "local.fragment", "shared", "shared.dyn"])
def test_fill_keeps_scope_algorithms_and_region_adapter(scope):
    t = tvm.tirx
    dst = t.decl_buffer((8,), "float32", name="dst", scope=scope)
    layouts = {dst: Fragment((8,), forward_thread_fn=lambda i: 0, forward_index_fn=lambda i: i)} if scope == "local.fragment" else {}
    result = _lower(fill(dst, 3), [dst], layouts)
    stores = _nodes(result.body, t.BufferStore)
    assert len(stores) == 1
    assert isinstance(stores[0].value, t.Cast)
    assert stores[0].value.dtype == "float32"
    assert int(stores[0].value.value) == 3
    assert any(loop.kind == t.ForKind.VECTORIZED for loop in _nodes(result.body, t.For))


def test_fill_keeps_the_original_unsupported_ub_scope_boundary():
    dst = tvm.tirx.decl_buffer((8,), "float32", name="dst", scope="shared.ub")
    with pytest.raises(tvm.error.InternalError, match="Unsupported scope shared.ub"):
        _lower(fill(dst, 3), [dst])


def _reduction(kind, clear=True, cross_thread=False):
    t = tvm.tirx
    src = t.decl_buffer((2, 4), "float32", name="src", scope="local.fragment")
    dst = t.decl_buffer((2,), "float32", name="dst", scope="local.fragment")
    src_layout = Fragment(
        (2, 4),
        forward_thread_fn=(lambda i, j: j) if cross_thread else (lambda i, j: 0),
        forward_index_fn=(lambda i, j: i) if cross_thread else (lambda i, j: i * 4 + j),
    )
    dst_layout = Fragment(
        (2,),
        forward_thread_fn=(lambda i, rep: rep) if cross_thread else (lambda i: 0),
        forward_index_fn=lambda i: i,
        replicate=4 if cross_thread else 1,
    )
    call = t.Call(
        "handle",
        tvm.ir.Op.get("tl.tileop.reduce"),
        [src.access_ptr("r"), dst.access_ptr("w"), t.StringImm(kind), 1, t.const(clear, "bool")],
    )
    return call, [src, dst], {src: src_layout, dst: dst_layout}


@pytest.mark.parametrize("kind", ["sum", "abssum", "max", "min", "absmax"])
def test_reduce_keeps_all_original_thread_local_operators(kind):
    t = tvm.tirx
    result = _lower(*_reduction(kind))
    assert all(int(loop.extent) in (2, 4) for loop in _nodes(result.body, t.For))
    assert len(_nodes(result.body, t.BufferStore)) == 2
    assert not any(node.op.name == "tirx.call_extern" for node in _nodes(result.body, t.Call) if isinstance(node.op, tvm.ir.Op))


@pytest.mark.parametrize("kind", ["sum", "abssum"])
def test_reduce_clear_false_retains_temporary_and_accumulation(kind):
    t = tvm.tirx
    result = _lower(*_reduction(kind, clear=False))
    temps = [node for node in _nodes(result.body, t.AllocBuffer) if node.buffer.name.endswith("_clear")]
    assert len(temps) == 1
    stores = _nodes(result.body, t.BufferStore)
    assert len(stores) == 3
    final = stores[-1]
    assert isinstance(final.value, t.Add)
    assert final.value.a.buffer.data.same_as(final.buffer.data)
    assert final.value.b.buffer.name.endswith("_clear")


def test_reduce_region_adapter_rejects_partial_buffers():
    t = tvm.tirx
    call, buffers, layouts = _reduction("sum")
    src = buffers[0]
    partial = t.Call("handle", tvm.ir.Op.get("tl.region"), [t.BufferLoad(src, [0, 0]), 1, 1, 4])
    call = t.Call("handle", call.op, [partial, *call.args[1:]])
    with pytest.raises(tvm.error.InternalError, match="expects full buffers"):
        _lower(call, buffers, layouts)


def test_reduce_rejects_unconsumed_cuda_cross_thread_extern():
    with pytest.raises(tvm.error.InternalError, match="cannot reduce across threads"):
        _lower(*_reduction("sum", cross_thread=True))


def test_gemm_keeps_the_original_unsupported_ascend_boundary():
    t = tvm.tirx
    a, b = [t.decl_buffer((16, 16), "float16", name=name, scope="shared") for name in ("a", "b")]
    c = t.decl_buffer((16, 16), "float32", name="c", scope="local.fragment")
    with pytest.raises(tvm.error.InternalError, match="generic gemm does not support A2/A3"):
        _lower(gemm(a, b, c), [a, b, c])
    with pytest.raises(tvm.error.InternalError, match="kPack must be 1 or 2"):
        _lower(gemm(a, b, c, annotations={"k_pack": 3}), [a, b, c])


def test_cumsum_keeps_scope_errors_and_rejects_unconsumed_shared_extern():
    t = tvm.tirx
    for scope, diagnostic in [("shared", "no AscendC implementation"), ("local.fragment", "CumSum for fragment not implemented")]:
        src, dst = [t.decl_buffer((2, 4), "float32", name=name, scope=scope) for name in ("src", "dst")]
        call = t.Call("handle", tvm.ir.Op.get("tl.tileop.cumsum"), [src.access_ptr("r"), dst.access_ptr("w"), 1, t.const(False, "bool")])
        with pytest.raises(tvm.error.InternalError, match=diagnostic):
            _lower(call, [src, dst])
