"""Full TA cross-core staging and original source boundary contracts."""

import pytest

from tilelang import tvm
from tilelang.ascend.a2a3 import transform


tir = tvm.tirx


def _seq(*stmts):
    return tir.SeqStmt(list(stmts)) if len(stmts) > 1 else stmts[0]


def _binding(var, value, body):
    return _seq(tir.Bind(var, value), body)


def _root(body, buffers=(), params=()):
    block = tir.SBlock([], [], [], "tilelang_root", body, alloc_buffers=list(buffers))
    return tir.PrimFunc([b.data for b in params], tir.SBlockRealize([], True, block), buffer_map={b.data: b for b in params})


def _run(func):
    return transform.CrossCorePipeline()(tvm.IRModule({"main": func}))["main"]


def _nodes(body, cls):
    result = []
    tir.stmt_functor.post_order_visit(body, lambda node: result.append(node) if isinstance(node, cls) else None)
    return result


def _kernel(
    platform="A3",
    stages=2,
    interval=1,
    extent=12,
    direction="c2v",
    inner=None,
    shared=False,
    pipe=False,
    multi=False,
    unmarked=False,
    primitive=False,
):
    c = tir.decl_buffer((256,), "float32", name="cube", scope="wmma.accumulator")
    v = tir.decl_buffer((256,), "float32", name="vector", scope="shared.ub")
    w = tir.decl_buffer((16,), "float32", name="workspace_cv")
    k = tir.Var("k", "int32")

    def write(offset=None):
        offset = k * 16 if offset is None else offset
        return tir.Evaluate(
            tir.call_extern(
                "handle",
                "tl::ascend::copy_l0c_to_gm<float, float>",
                c.access_ptr("r", offset=offset, extent=16),
                w.access_ptr("w", extent=16),
            )
        )

    read = tir.Evaluate(
        tir.call_extern(
            "handle", "tl::ascend::copy_gm_to_ub<float>", w.access_ptr("r", extent=16), v.access_ptr("w", offset=k * 16, extent=16)
        )
    )
    work = write()
    if inner == "for":
        work = tir.For(tir.Var("j", "int32"), 0, 2, tir.ForKind.SERIAL, work)
    if inner == "if":
        work = tir.IfThenElse(k < 5, work, work)
    if inner == "noncall":
        work = tir.IfThenElse(k < 5, work, tir.Evaluate(0))
    if inner == "let":
        x = tir.Var("offset", "int32")
        body = _binding(x, k * 16, _seq(write(x), write(x + 16)))
        work = tir.For(tir.Var("j", "int32"), 0, 1, tir.ForKind.SERIAL, body)
    if inner == "bad_nested":
        work = _seq(
            tir.For(tir.Var("j", "int32"), 0, 1, tir.ForKind.SERIAL, work, annotations={"stage_loop": tir.const(True, "bool")}), write()
        )
    if shared:
        local = tir.Evaluate(tir.call_extern("handle", "consume", v.access_ptr("r", extent=16)))
        work = _seq(local, work)
    body = _seq(work, read, write()) if multi else _seq(read, work) if direction == "v2c" else _seq(work, read)
    stage = tir.SBlockRealize([], True, tir.SBlock([], [], [], "stage", body, alloc_buffers=[v] if shared else []))
    annotations = (
        {}
        if unmarked
        else {"num_stages": stages if primitive else tir.const(stages), "tl_cross_interval": interval if primitive else tir.const(interval)}
    )
    pipeline = tir.For(k, 0, extent, tir.ForKind.SERIAL, stage, annotations=annotations)
    out = _root(pipeline, [c] + ([] if shared else [v]), [w])
    if isinstance(extent, tir.Var):
        out = tir.PrimFunc([extent, w.data], out.body, buffer_map={w.data: w})
    if platform is not None:
        out = out.with_attr("npu_platform", platform)
    if pipe:
        out = out.with_attr(
            "pipe_infos", {tir.const(0): {"workspace_name": "workspace_cv", "slot_num": tir.const(3), "src_scope": "shared.ub"}}
        )
    return out


def _free_runtime_vars(func):
    """Lexical check permits the source's reused Var across sibling loops."""
    free = []

    def expr(value, bound):
        for var in _nodes(value, tir.Var):
            if not any(var.same_as(item) for item in bound) and not any(var.same_as(item) for item in free):
                free.append(var)

    def stmt(node, bound):
        if isinstance(node, tir.SeqStmt):
            local = list(bound)
            for child in node.seq:
                stmt(child, local)
                if isinstance(child, tir.Bind):
                    local.append(child.var)
        elif isinstance(node, tir.Bind):
            expr(node.value, bound)
        elif isinstance(node, tir.For):
            expr(node.min, bound)
            expr(node.extent, bound)
            stmt(node.body, [*bound, node.loop_var])
        elif isinstance(node, tir.SBlockRealize):
            expr(node.predicate, bound)
            stmt(node.block, bound)
        elif isinstance(node, tir.SBlock):
            stmt(node.body, [*bound, *(b.data for b in node.alloc_buffers)])
        elif isinstance(node, tir.Evaluate):
            expr(node.value, bound)
        elif isinstance(node, tir.BufferStore):
            expr(node.value, bound)
            for index in node.indices:
                expr(index, bound)
        elif isinstance(node, tir.IfThenElse):
            expr(node.condition, bound)
            stmt(node.then_case, bound)
            if node.else_case is not None:
                stmt(node.else_case, bound)
        elif isinstance(node, tir.AttrStmt):
            expr(node.value, bound)
            stmt(node.body, bound)
        else:
            raise AssertionError(type(node).__name__)

    stmt(func.body, list(func.params))
    return free


@pytest.mark.parametrize("platform", [None, "A2", "A3", "A5"])
@pytest.mark.parametrize("stages", [2, 3])
@pytest.mark.parametrize("interval", [1, 2])
def test_stage_binding_workspace_and_platform_source_branches(platform, stages, interval):
    result = _run(_kernel(platform, stages, interval))
    assert [int(x) for x in result.buffer_map[result.params[0]].shape] == [stages, 16]
    loops = _nodes(result.body, tir.For)
    assert len(loops) == 3
    assert int(loops[0].extent) == stages
    assert int(loops[1].extent) == stages
    assert int(loops[2].extent) == 12 // stages
    assert all(int(loop.annotations["tl_cross_interval"]) == interval for loop in loops[:2])
    assert len(_nodes(result.body, tir.Bind)) == 2
    assert not _free_runtime_vars(result)
    calls = [
        call for call in _nodes(result.body, tir.Call) if call.op.name == "tirx.tvm_access_ptr" and call.args[1].same_as(result.params[0])
    ]
    assert len(calls) == 2
    offsets = [call.args[2] for call in calls]
    if platform == "A5":
        assert all(isinstance(offset, tir.IntImm) and int(offset) == 0 for offset in offsets)
    else:
        stage_vars = [loop.loop_var for loop in loops[:2]]
        assert all(any(var.same_as(loops[0].loop_var) for var in _nodes(offset, tir.Var)) for offset in offsets)
        assert stage_vars[0].same_as(stage_vars[1])  # Source intentionally reuses this intermediate Var.


@pytest.mark.parametrize("inner", ["for", "if", "let"])
def test_nested_control_and_flat_bind_preserve_runtime_binding(inner):
    result = _run(_kernel(inner=inner))
    assert not _free_runtime_vars(result)
    if inner == "let":
        binds = [node for node in _nodes(result.body, tir.Bind) if node.var.name == "offset"]
        assert len(binds) == 1
        accesses = [call for call in _nodes(result.body, tir.Call) if call.op.name == "tirx.tvm_access_ptr" and call.args[1].name == "cube"]
        assert len(accesses) == 2
        assert all(any(var.same_as(binds[0].var) for var in _nodes(call.args[2], tir.Var)) for call in accesses)


@pytest.mark.parametrize("direction", ["c2v", "v2c"])
@pytest.mark.parametrize("shared", [False, True])
@pytest.mark.parametrize("multi", [False, True])
def test_repeated_resource_stages_and_exposed_shared_reads(direction, shared, multi):
    result = _run(_kernel(direction=direction, shared=shared, multi=multi))
    stage_loops = [loop for loop in _nodes(result.body, tir.For) if "stage_loop" in loop.annotations]
    expected = (4 if shared else 3) if multi else (3 if shared and direction == "c2v" else 2)
    assert len(stage_loops) == expected
    assert not _free_runtime_vars(result)
    if shared:
        buffers = [buffer for block in _nodes(result.body, tir.SBlock) for buffer in block.alloc_buffers]
        vector = next(buffer for buffer in buffers if buffer.name == "vector")
        if direction == "v2c" and not multi:
            # The read stage writes this buffer before consuming it: no exposed
            # read crosses a stage, so the original kill analysis avoids a copy.
            assert [int(x) for x in vector.shape] == [256]
            assert vector.data not in result.attrs["buffer_versions"]
        else:
            assert [int(x) for x in vector.shape] == [2, 256]
            assert int(result.attrs["buffer_versions"][vector.data]) == 2


def test_pipe_metadata_slot_multiplier_retains_all_other_fields():
    func = _kernel(pipe=True)
    result = _run(func)
    [(flag, fields)] = list(result.attrs["pipe_infos"].items())
    assert int(flag) == 0
    assert int(fields["slot_num"]) == 6
    assert fields["workspace_name"] == "workspace_cv"
    assert fields["src_scope"] == "shared.ub"


def test_symbolic_extent_keeps_original_multiply_branch():
    n = tir.Var("n", "int32")
    result = _run(_kernel(extent=n))
    outer = next(loop for loop in _nodes(result.body, tir.For) if "tl_original_extent" in loop.annotations)
    assert tvm.arith.Analyzer().can_prove_equal(outer.extent, n * 2)
    assert not _free_runtime_vars(result)


def test_unmarked_no_cross_and_absent_platform_preserve_noop_input():
    for func in [_kernel(unmarked=True), _root(tir.Evaluate(0))]:
        assert tvm.ir.structural_equal(_run(func), func)


def test_nested_stage_then_adjusted_access_rejects_original_free_var_output():
    with pytest.raises(tvm.error.InternalError, match="outside an active stage_loop"):
        _run(_kernel(inner="bad_nested"))


def test_multiple_cross_core_pipelines_preserve_source_rejection():
    first, second = _kernel(), _kernel()
    body = _seq(first.body.block.body, second.body.block.body)
    buffers = [*first.body.block.alloc_buffers, *second.body.block.alloc_buffers]
    params = [*first.buffer_map.values(), *second.buffer_map.values()]
    with pytest.raises(tvm.error.InternalError, match="only support one cross core pipeline"):
        _run(_root(body, buffers, params))


@pytest.mark.parametrize("interval", [1, 2])
def test_annotation_integer_representations_have_identical_results(interval):
    boxed = _run(_kernel(interval=interval))
    primitive = _run(_kernel(interval=interval, primitive=True))
    assert tvm.ir.structural_equal(boxed, primitive, map_free_vars=True)


@pytest.mark.parametrize("key", ["num_stages", "tl_cross_interval"])
@pytest.mark.parametrize("value", [2.0, "2"])
def test_noninteger_pipeline_annotations_are_rejected(key, value):
    func = _kernel()
    pipeline = func.body.block.body
    annotations = dict(pipeline.annotations.items())
    annotations[key] = value
    bad_loop = tir.For(pipeline.loop_var, pipeline.min, pipeline.extent, pipeline.kind, pipeline.body, annotations=annotations)
    with pytest.raises(tvm.error.InternalError, match="requires an integer annotation"):
        _run(_root(bad_loop, func.body.block.alloc_buffers, list(func.buffer_map.values())))


def test_noncall_inside_marked_pipeline_reports_cleanup_precondition():
    with pytest.raises(tvm.error.InternalError, match=r"requires Evaluate\(Call\).*after cleanup"):
        _run(_kernel(inner="noncall"))
