"""The TA planner keeps dependency, annotation, and empty-stage policies."""

import pytest
from tilelang import tvm


tir = tvm.tirx


def _planning_case(kind="buffer", stages=2, dtype="float32", extent=4, annotations=None):
    source = tir.decl_buffer((64,), dtype, name="source")
    output = tir.decl_buffer((64,), dtype, name="output")
    local = tir.decl_buffer((16,), dtype, name="local", scope="shared.ub")
    k = tir.Var("k", "int32")
    copy = tir.BufferStore(local, tir.BufferLoad(source, [k * 16]), [0])
    use = tir.BufferStore(output, tir.BufferLoad(local, [0]) + tir.const(1, dtype), [k * 16])
    if kind == "extern":
        copy = tir.Evaluate(
            tir.call_extern(
                "handle", f"tl::ascend::copy_gm_to_ub<{dtype}, 16>", source.access_ptr("r"), local.access_ptr("w"), 16, 1, 16, 0, 16
            )
        )
    elif kind == "intrinsic":
        use = tir.Evaluate(tir.call_intrin("handle", tvm.ir.Op.get("tl.ascend_exp"), output.access_ptr("w"), local.access_ptr("r"), 16))
    elif kind == "address":
        use = tir.Evaluate(tir.call_extern("handle", "unknown", tir.address_of(local[0])))
    elif kind == "conditional":
        copy = tir.IfThenElse(k < 3, copy, None)
    elif kind == "overlap":
        use = tir.BufferStore(local, tir.BufferLoad(source, [k * 16 + 1]), [0])
    elif kind == "disjoint":
        use = tir.BufferStore(local, tir.BufferLoad(source, [k * 16 + 1]), [1])
    elif kind == "empty":
        copy, use = tir.Evaluate(0), tir.Evaluate(0)
    body = copy if kind == "single" else tir.SeqStmt([copy, use])
    if kind == "block":
        body = tir.SBlockRealize([], True, tir.SBlock([], [], [], "body", body))
    elif kind == "block_if":
        body = tir.SBlockRealize([], True, tir.SBlock([], [], [], "body", tir.IfThenElse(k < 3, body, None)))
    elif kind == "block_else":
        body = tir.SBlockRealize([], True, tir.SBlock([], [], [], "body", tir.IfThenElse(k < 3, body, tir.Evaluate(0))))
    attrs = {"num_stages": stages, "user_marker": 7} if annotations is None else annotations
    loop = tir.For(k, 0, extent, tir.ForKind.PARALLEL if kind == "parallel" else tir.ForKind.SERIAL, body, annotations=attrs)
    block = tir.SBlock([], [], [], "root", loop, alloc_buffers=[local])
    return tir.PrimFunc(
        [source.data, output.data], tir.SBlockRealize([], True, block), buffer_map={source.data: source, output.data: output}
    )


def _run(func, target=None):
    if target is None:
        target = tvm.target.Target({"kind": "ascend", "arch": "dav-2201", "keys": ["ascend"], "model": "A3"})
    func = func.with_attr("target", target)
    factory = tvm.ffi.get_global_func("tl.transform.A2A3PipelinePlanning")
    return factory()(tvm.IRModule({"main": func}))["main"]


def _loops(func):
    result = []
    tir.stmt_functor.post_order_visit(func.body, lambda node: result.append(node) if isinstance(node, tir.For) else None)
    return result


@pytest.mark.parametrize("stages,order,stage", [(1, [1, 0], [0, 1]), (2, [0, 1], [0, 1]), (3, [0, 1], [0, 2])])
@pytest.mark.parametrize("dtype", ["float16", "float32", "int32"])
def test_copy_dependency_schedule(stages, order, stage, dtype):
    loop = _loops(_run(_planning_case(stages=stages, dtype=dtype)))[0]
    assert list(loop.annotations["software_pipeline_order"]) == order
    assert list(loop.annotations["software_pipeline_stage"]) == stage
    assert "num_stages" not in loop.annotations
    assert int(loop.annotations["user_marker"]) == 7
    assert "software_pipeline_async_stages" not in loop.annotations


@pytest.mark.parametrize("kind", ["extern", "intrinsic", "conditional", "block", "block_if"])
def test_collector_and_stage_containers(kind):
    loop = _loops(_run(_planning_case(kind)))[0]
    assert list(loop.annotations["software_pipeline_stage"]) == [0, 1]
    assert list(loop.annotations["software_pipeline_order"]) == [0, 1]


@pytest.mark.parametrize("extent", [0, 1, 4])
def test_empty_stage_preserves_loop_and_only_removes_scheduling(extent):
    attrs = {
        "num_stages": 3,
        "tl_pipeline_order": [0, 1],
        "tl_pipeline_stage": [0, 2],
        "software_pipeline_order": [0, 1],
        "software_pipeline_stage": [0, 2],
        "software_pipeline_async_stages": [0],
        "user_marker": 7,
    }
    loop = _loops(_run(_planning_case("empty", extent=extent, annotations=attrs)))[0]
    assert int(loop.extent) == extent
    assert dict(loop.annotations) == {"user_marker": 7}


def test_explicit_annotations_and_source_retained_keys():
    attrs = {"tl_pipeline_order": [0, 1], "tl_pipeline_stage": [0, 2], "user_marker": 7}
    loop = _loops(_run(_planning_case(annotations=attrs)))[0]
    assert list(loop.annotations["software_pipeline_order"]) == [0, 1]
    assert list(loop.annotations["software_pipeline_stage"]) == [0, 2]
    # The two source copy loops also retain the original tl_* keys.
    assert list(loop.annotations["tl_pipeline_order"]) == [0, 1]
    assert list(loop.annotations["tl_pipeline_stage"]) == [0, 2]


@pytest.mark.parametrize("key", ["tl_pipeline_order", "tl_pipeline_stage"])
def test_source_minus_one_annotation_is_left_for_other_planners(key):
    attrs = {"tl_pipeline_order": [0, 1], "tl_pipeline_stage": [0, 2]}
    attrs[key] = [-1, 1]
    loop = _loops(_run(_planning_case(annotations=attrs)))[0]
    assert "software_pipeline_stage" not in loop.annotations
    assert list(loop.annotations[key]) == [-1, 1]


@pytest.mark.parametrize(
    "kind,model,keys", [("ascend", "A2", ["ascend"]), ("ascend", "A3", ["ascend"]), ("ascend", "A3", ["ascend", "pto"])]
)
def test_ascend_targets_do_not_enable_cuda_async_annotations(kind, model, keys):
    target = tvm.target.Target({"kind": kind, "arch": "dav-2201", "model": model, "keys": keys})
    loop = _loops(_run(_planning_case(), target))[0]
    assert "software_pipeline_async_stages" not in loop.annotations
    assert list(loop.annotations["software_pipeline_stage"]) == [0, 1]


def test_no_request_is_unchanged():
    before = _planning_case(annotations={"user_marker": 7})
    after = _run(before)
    tvm.ir.assert_structural_equal(before.body, after.body)


@pytest.mark.parametrize(
    "kind,pattern", [("overlap", "Multiple writes"), ("single", "SeqStmt"), ("parallel", "Serial"), ("block_else", "Can't handle")]
)
def test_source_rejections(kind, pattern):
    with pytest.raises(Exception, match=pattern):
        _run(_planning_case(kind))


def test_num_stages_requires_positive_value():
    with pytest.raises(Exception, match="num_stages >= 1"):
        _run(_planning_case(stages=0))


def test_target_is_required():
    factory = tvm.ffi.get_global_func("tl.transform.A2A3PipelinePlanning")
    with pytest.raises(Exception, match="Require the target"):
        factory()(tvm.IRModule({"main": _planning_case()}))


@pytest.mark.parametrize("value", [2, tir.IntImm("int32", 2), tir.IntImm("int64", 2)])
def test_num_stages_any_primitive_and_intimm_abi(value):
    before = _planning_case(annotations={"num_stages": value})
    actual = _loops(before)[0].annotations["num_stages"]
    assert isinstance(actual, (int, tir.IntImm))
    loop = _loops(_run(before))[0]
    assert list(loop.annotations["software_pipeline_stage"]) == [0, 1]
    assert list(loop.annotations["software_pipeline_order"]) == [0, 1]


def test_real_language_pipelined_producer():
    from tilelang.ascend.a2a3 import language as T

    @T.prim_func
    def func(source: T.Tensor((64,), "float32"), output: T.Tensor((64,), "float32")):
        with T.Kernel(1, is_npu=True):
            local = T.alloc_ub((16,), "float32")
            for k in T.Pipelined(4, num_stages=2):
                local[0] = source[k]
                output[k] = local[0] + 1

    before = _loops(func)[0]
    assert isinstance(before.annotations["num_stages"], tir.IntImm)
    loop = _loops(_run(func))[0]
    assert list(loop.annotations["software_pipeline_stage"]) == [0, 1]


@pytest.mark.parametrize("empty_kind", ["evaluate", "pure_bind"])
def test_flat_pure_bind_counterpart_keeps_ir_and_only_disables_scheduling(empty_kind):
    loop_var = tir.Var("t", "int32")
    scalar = tir.Var("col0", "int32")
    statement = tir.Bind(scalar, loop_var * 128) if empty_kind == "pure_bind" else tir.Evaluate(0)
    body = tir.SBlockRealize([], True, tir.SBlock([], [], [], "stage", statement))
    loop = tir.For(loop_var, 0, 12, tir.ForKind.SERIAL, body, annotations={"num_stages": 2, "user_marker": 7})
    func = tir.PrimFunc([], loop)
    result = _loops(_run(func))[0]
    assert result.body.same_as(body)
    assert dict(result.annotations) == {"user_marker": 7}
    if empty_kind == "pure_bind":
        assert result.body.block.body.var.same_as(scalar)
        tvm.ir.assert_structural_equal(result.body.block.body.value, loop_var * 128)


@pytest.mark.parametrize("kind", ["read_state", "volatile", "opaque", "update_state", "store"])
def test_nonpure_or_volatile_stage_is_not_an_empty_counterpart(kind):
    source = tir.decl_buffer((16,), "int32", name="source")
    scalar = tir.Var("scalar", "int32")
    if kind == "opaque":
        statement = tir.Bind(scalar, tir.call_extern("int32", "side_effect"))
    elif kind == "update_state":
        statement = tir.Bind(scalar, tir.call_intrin("int32", "tirx.tvm_struct_set", source.data, 0, 0, 1))
    elif kind == "store":
        statement = tir.BufferStore(source, tir.const(1), [0])
    else:
        statement = tir.Bind(scalar, tir.BufferLoad(source, [0]))
        if kind == "volatile":
            statement = tir.AttrStmt(source.data, "volatile_scope", 1, statement)
    body = tir.SBlockRealize([], True, tir.SBlock([], [], [], "stage", statement))
    loop = tir.For(tir.Var("t", "int32"), 0, 4, tir.ForKind.SERIAL, body, annotations={"num_stages": 2})
    func = tir.PrimFunc([source.data], loop, buffer_map={source.data: source})
    # Keep the existing unsupported single-stage-container diagnostic. Merely
    # having a Bind must not silently disable an actual read/update stage.
    with pytest.raises(Exception, match="SeqStmt|Can't handle"):
        _run(func)
