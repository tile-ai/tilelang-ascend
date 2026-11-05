"""TA layout inference contracts; no complete A2/A3 pipeline is assembled."""

import pytest

from tilelang import tvm
from tilelang.ascend.a2a3.layout import Layout, layout_str
from tilelang.layout import Fragment

tirx = tvm.tirx


def _run(body, buffers=(), target="llvm"):
    func = tirx.PrimFunc([b.data for b in buffers], body, buffer_map={b.data: b for b in buffers})
    func = func.with_attr({"global_symbol": "main", "target": tvm.target.Target(target)})
    return tvm.ffi.get_global_func("tl.transform.A2A3LayoutInference")()(tvm.IRModule.from_expr(func))["main"]


def _nodes(body, typ):
    out = []
    tirx.stmt_functor.post_order_visit(body, lambda x: out.append(x) if isinstance(x, typ) else None)
    return out


def _thread(body, extent=8):
    var = tirx.Var("thread", "int32")
    iv = tirx.IterVar(tvm.ir.Range(0, extent), var, tirx.IterVar.ThreadIndex, "cthread")
    return tirx.AttrStmt(iv, "thread_extent", extent, body)


def _root(body, allocated=(), annotations=None):
    block = tirx.SBlock(
        [],
        [],
        [],
        "root",
        body,
        alloc_buffers=list(allocated),
        annotations={"tilelang.is_npu_kernel_frame_dev_mode": True, **(annotations or {})},
    )
    return tirx.SBlockRealize([], True, block)


def _layout_map(after):
    return _nodes(after.body, tirx.SBlock)[0].annotations["layout_map"]


@pytest.mark.parametrize("target", ["llvm", {"kind": "ascend", "arch": "dav-2201"}])
@pytest.mark.parametrize("threads", [None, 8])
def test_inference_plans_fragment_coordinates_with_or_without_thread_binding(threads, target):
    global_buffer = tirx.decl_buffer((32,), "float32", name="A")
    fragment = tirx.decl_buffer((32,), "float32", name="F", scope="local.fragment")
    i = tirx.Var("i", "int32")
    loop = tirx.For(i, 0, 32, tirx.ForKind.PARALLEL, tirx.BufferStore(fragment, tirx.BufferLoad(global_buffer, [i]), [i]))
    body = _root(loop, [fragment])
    after = _run(_thread(body, threads) if threads else body, [global_buffer], target=target)
    layout = _layout_map(after)[fragment]
    assert isinstance(layout, Fragment)
    assert int(layout.get_thread_size()) == (threads or 1)
    if threads:
        assert all(loop.kind != tirx.ForKind.PARALLEL for loop in _nodes(after.body, tirx.For))
    else:
        assert any(loop.kind == tirx.ForKind.PARALLEL for loop in _nodes(after.body, tirx.For))
    analyzer = tvm.arith.Analyzer()
    for index in range(32):
        local = int(analyzer.simplify(layout.map_forward_index([index])[0]))
        thread = int(analyzer.simplify(layout.map_forward_thread([index])[0]))
        assert local == (index % 4 if threads else index)
        assert thread == (index // 4 if threads else 0)


def test_inference_preserves_explicit_fragment_layout_and_address_map():
    fragment = tirx.decl_buffer((32,), "float32", scope="local.fragment")
    i = tirx.Var("i", "int32")
    layout = Fragment((32,), forward_thread_fn=lambda i: i % 8, forward_index_fn=lambda i: i // 8)
    address = tirx.Var("address", "handle")
    loop = tirx.For(i, 0, 32, tirx.ForKind.PARALLEL, tirx.BufferStore(fragment, tirx.FloatImm("float32", 1), [i]))
    after = _run(
        _thread(_root(loop, [fragment], {"layout_map": {fragment.data: layout}, "address_map": {address: tirx.IntImm("int32", 4)}}))
    )
    tvm.ir.assert_structural_equal(_layout_map(after)[fragment], layout)
    assert int(after.attrs["address_map"][address]) == 4
    assert _nodes(after.body, tirx.For)[0].kind != tirx.ForKind.PARALLEL


def test_inference_preserves_tagged_layout_without_reconstructing_it():
    buffer = tirx.decl_buffer((3, 5), "float16", scope="shared.l1")
    layout = Layout((3, 5), lambda i, j: [j, i], layout_tag=1)
    after = _run(_root(tirx.Evaluate(0), [buffer], {"layout_map": {buffer.data: layout}}))
    inferred = _layout_map(after)[buffer]
    assert layout_str(inferred) == "layout::ColumnMajor"
    tvm.ir.assert_structural_equal(inferred, layout)


def test_inference_fuses_power_of_two_parallel_nest_without_fragment_access():
    buffer = tirx.decl_buffer((4, 8), "float32")
    i, j = tirx.Var("i", "int32"), tirx.Var("j", "int32")
    body = tirx.For(
        i,
        0,
        4,
        tirx.ForKind.PARALLEL,
        tirx.For(j, 0, 8, tirx.ForKind.PARALLEL, tirx.BufferStore(buffer, tirx.FloatImm("float32", 1), [i, j])),
    )
    after = _run(_root(body), [buffer])
    assert any("fused" in str(loop.loop_var.name) for loop in _nodes(after.body, tirx.For))
    assert any(loop.kind == tirx.ForKind.PARALLEL for loop in _nodes(after.body, tirx.For))


def test_inference_rejects_missing_annotated_buffer_definition():
    missing = tirx.Var("missing", "handle")
    layout = Layout((4,), lambda i: [i])
    with pytest.raises(tvm.error.InternalError, match="has no matching definition"):
        _run(_root(tirx.Evaluate(0), annotations={"layout_map": {missing: layout}}))


def test_inference_preserves_fragment_partial_extent_predicate():
    fragment = tirx.decl_buffer((8,), "float32", scope="local.fragment")
    i = tirx.Var("i", "int32")
    loop = tirx.For(i, 0, 4, tirx.ForKind.PARALLEL, tirx.BufferStore(fragment, tirx.FloatImm("float32", 1), [i]))
    after = _run(_root(loop, [fragment]))
    assert [int(x) for x in _layout_map(after)[fragment].get_input_shape()] == [8]
    assert any(int(loop.extent) == 8 for loop in _nodes(after.body, tirx.For))
    predicates = _nodes(after.body, tirx.IfThenElse)
    assert len(predicates) == 1
    assert isinstance(predicates[0].condition, tirx.LT)
    assert int(predicates[0].condition.b) == 4


@pytest.mark.parametrize("threads", [None, 1, 2])
def test_inference_real_kernel_preserves_default_and_explicit_vid_partition(threads):
    import tilelang
    import tilelang.ascend.a2a3.language as T

    @T.prim_func
    def main(A: T.Tensor((32,), "float32")):
        if threads is None:
            with T.Kernel(1, is_npu=True) as (cid, vid):
                F = T.alloc_fragment((32,), "float32")
                for i in T.Parallel(32):
                    F[i] = A[i]
        else:
            with T.Kernel(1, threads=threads, is_npu=True) as _cid:
                F = T.alloc_fragment((32,), "float32")
                for i in T.Parallel(32):
                    F[i] = A[i]

    module = tirx.transform.BindTarget(tvm.target.Target({"kind": "ascend", "arch": "dav-2201"}))(tvm.IRModule.from_expr(main))
    module = tilelang.transform.MaterializeKernelLaunch(
        lower_grid_binding=True,
        lower_thread_binding=False,
        default_threads=None,
        unsupported_annotations=["cluster_dims"],
        launch_dim_tags=["cthread"],
    )(module)
    marker = "tilelang.is_npu_kernel_frame" if threads is None else "tilelang.is_npu_kernel_frame_dev_mode"
    assert any(marker in block.annotations for block in _nodes(module["main"].body, tirx.SBlock))
    after = tvm.ffi.get_global_func("tl.transform.A2A3LayoutInference")()(module)["main"]
    layouts = next(block.annotations["layout_map"] for block in _nodes(after.body, tirx.SBlock) if block.annotations.get("layout_map"))
    layout = next(iter(layouts.values()))
    # TA's default blockIdx.y vid was not a partition thread. The producer's
    # canonical cthread tag must not make that default form partition by vid.
    assert int(layout.get_thread_size()) == (threads or 1)
    analyzer = tvm.arith.Analyzer()
    for index in range(32):
        assert int(analyzer.simplify(layout.map_forward_thread([index])[0])) == (index // 4 % 2 if threads == 2 else 0)


def test_inference_raw_canonical_vid_without_launch_marker_retains_source_contract():
    source = tirx.decl_buffer((32,), "float32")
    fragment = tirx.decl_buffer((32,), "float32", scope="local.fragment")
    i = tirx.Var("i", "int32")
    loop = tirx.For(i, 0, 32, tirx.ForKind.PARALLEL, tirx.BufferStore(fragment, tirx.BufferLoad(source, [i]), [i]))
    block = tirx.SBlock([], [], [], "raw", loop, alloc_buffers=[fragment])
    body = _thread(tirx.SBlockRealize([], True, block), extent=2)
    after = _run(body, [source])
    assert int(_layout_map(after)[fragment].get_thread_size()) == 2
