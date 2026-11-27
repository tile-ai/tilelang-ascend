"""Fixed FIFO C/V handoff generation and independent joint fault injection."""

import pytest

import tilelang
from tilelang import tvm
from tilelang import language as T
from tilelang.engine.phase import LowerAndLegalize, OptimizeForTarget

from test_s7_kernels import make_cv_handoff_kernel

tir = tvm.tir
TARGET = tvm.target.Target("c -keys=ascend -model=ascendc")
SOURCE = "tl.ascend_handoff_source"
PLAN = "tl.ascend_handoff_plan"


def handoff_configs(strategy="a3"):
    return {
        "tl.ascend_pipeline_planner": "legacy",
        "tl.ascend_sync_strategy": strategy,
        "tl.ascend_auto_sync": True,
        "tl.ascend_auto_sync_vs": True,
        "tl.ascend_memory_planning": True,
        "tl.ascend_auto_cv_combine": True,
        "tl.ascend_auto_cross_core_sync": False,
    }


def _pass(name, function):
    with tvm.transform.PassContext(config=handoff_configs()):
        return getattr(tilelang.transform, name)(TARGET, "A3")(tvm.IRModule.from_expr(function))["main"]


def _synthetic(direction="v2c", trips=3, cores=1, *, guard=False):
    dtype = "float16" if direction == "v2c" else "float32"
    workspace = tir.decl_buffer((cores, 128, 64), dtype, name="workspace_0", scope="global")
    ub = tir.decl_buffer((64, 64), dtype, name="u", scope="shared.ub")
    local = tir.decl_buffer(
        (128, 64),
        "float16" if direction == "v2c" else "float32",
        name="cube_local",
        scope="shared.l1" if direction == "v2c" else "wmma.accumulator",
    )
    cid, vid, vr, cr = [tir.Var(name, "int32") for name in ("cid", "vid", "vr", "cr")]
    base = cid * 8192
    if direction == "v2c":
        vector = tir.Evaluate(
            tir.call_extern(
                "handle",
                "tl::ascend::copy_ub_to_gm<half,64,64>",
                ub.access_ptr("r"),
                workspace.access_ptr("w", offset=base + vid * 4096),
                64,
                64,
                64,
                64,
                64,
            )
        )
        cube = tir.Evaluate(
            tir.call_extern(
                "handle",
                "tl::ascend::copy_gm_to_l1<half,128,64>",
                workspace.access_ptr("r", offset=base),
                local.access_ptr("w"),
                64,
                128,
                64,
                128,
                64,
            )
        )
    else:
        cube = tir.Evaluate(
            tir.call_extern(
                "handle",
                "tl::ascend::copy_l0c_to_gm<float,float,layout::RowMajor,128,64,0>",
                local.access_ptr("r"),
                workspace.access_ptr("w", offset=base),
                64,
                128,
                64,
                128,
                64,
                False,
                0,
            )
        )
        vector = tir.Evaluate(
            tir.call_extern(
                "handle",
                "tl::ascend::copy_gm_to_ub<float,64,64>",
                workspace.access_ptr("r", offset=base + vid * 4096),
                ub.access_ptr("w"),
                64,
                64,
                64,
                tir.const(0, dtype),
                64,
                64,
            )
        )
    vector = tir.For(vr, 0, trips, tir.ForKind.SERIAL, vector)
    cube = tir.For(cr, 0, trips, tir.ForKind.SERIAL, cube)
    if guard:
        vector = tir.IfThenElse(cid < 1, vector, None)
        cube = tir.IfThenElse(cid < 1, cube, None)
    body = tir.SeqStmt(
        [
            tir.AttrStmt(tir.const(0), "resource_scope", 1, vector),
            tir.AttrStmt(tir.const(0), "resource_scope", 0, cube),
        ]
    )
    block = tir.Block([], [], [], "tilelang_root", body, alloc_buffers=[ub, local])
    body = tir.BlockRealize([], tir.const(True, "bool"), block)
    for variable, extent, tag in reversed([(cid, cores, "blockIdx.x"), (vid, 2, "threadIdx.x")]):
        axis = tir.IterVar(tvm.ir.Range.from_min_extent(0, extent), variable, tir.IterVar.ThreadIndex, tag)
        body = tir.AttrStmt(axis, "thread_extent", extent, body)
    handle = tir.Var("workspace_handle", "handle")
    return (
        tir.PrimFunc([handle], body, buffer_map={handle: workspace})
        .with_attr("global_symbol", "main")
        .with_attr("npu_cv_ratio", tir.StringImm("cv_1_2"))
        .with_attr("tir.noalias", True)
        .with_attr("target", TARGET)
        .with_attr("npu_platform", "A3")
    )


def _emitted(direction="v2c", trips=3, cores=1, **kwargs):
    return _pass(
        "AscendInsertSyncHandoff",
        _pass("AscendPrepareSyncHandoff", _synthetic(direction, trips, cores, **kwargs)),
    )


def _verify(function):
    before = tvm.ir.save_json(function)
    result = _pass("AscendVerifySyncHandoff", function)
    assert result.same_as(function)
    assert tvm.ir.save_json(function) == before


def _mutate_cross(function, transform):
    def rewrite(node):
        if not isinstance(node.value, tir.Call):
            return node
        name = getattr(node.value.op, "name", "")
        if name in {"tl.ascend_auto_set_cross_flag", "tl.ascend_auto_wait_cross_flag"}:
            value = transform(node.value, name)
            return tir.Evaluate(0) if value is None else tir.Evaluate(value)
        return node

    return function.with_body(tir.stmt_functor.ir_transform(function.body, None, rewrite, ["tir.Evaluate"]))


@pytest.mark.parametrize("direction", ["v2c", "c2v"])
@pytest.mark.parametrize("trips", [0, 1, 3, 17])
@pytest.mark.parametrize("cores", [1, 4])
def test_fixed_fifo_has_real_ready_release_and_exit_drain(direction, trips, cores):
    function = _emitted(direction, trips, cores)
    _verify(function)
    assert int(function.attrs[PLAN]["channel_count"]) == (0 if trips == 0 else 1)
    assert not bool(function.attrs[PLAN]["round_overlap"])


@pytest.mark.parametrize("direction", ["v2c", "c2v"])
def test_uniform_group_guard_preserves_complete_participation(direction):
    _verify(_emitted(direction, cores=4, guard=True))


@pytest.mark.parametrize("kind", ["ready", "release", "drain", "mode", "id", "pipe"])
def test_damaged_cross_protocol_is_rejected_independently(kind):
    function = _emitted("c2v")

    def mutate(call, name):
        arguments = list(call.args)
        if name.endswith("set_cross_flag"):
            flag = int(arguments[2])
            if kind == "mode":
                arguments[0] = tir.const(4)
            elif kind == "id":
                arguments[2] = tir.const(11)
            elif kind == "pipe" and flag == 1:
                arguments[1] = tir.StringImm("MTE3")
            elif kind == "release" and flag == 1:
                return None
        elif (kind == "ready" and int(arguments[0]) == 0) or (kind == "drain" and int(arguments[0]) == 1):
            return None
        return tir.Call(call.dtype, call.op, arguments)

    with pytest.raises(tvm.TVMError, match="handoff|Handoff|mode2|user pool|consumer|producer"):
        _verify(_mutate_cross(function, mutate))


def test_stale_pass_metadata_cannot_replace_actual_protocol_validation():
    function = _emitted()
    metadata = dict(function.attrs[PLAN].items())
    metadata["channel_count"] = 0
    with pytest.raises(tvm.TVMError, match="inconsistent handoff metadata"):
        _verify(function.with_attr(PLAN, metadata))


def test_explicit_cross_protocol_is_rejected_rather_than_silently_colliding():
    function = _synthetic()
    cross = tir.Evaluate(tir.Call("handle", tvm.ir.Op.get("tl.ascend_set_cross_flag"), [tir.StringImm("MTE3"), 0, 2]))
    function = function.with_body(tir.SeqStmt([function.body, cross]))
    with pytest.raises(tvm.TVMError, match="Explicit cross-core IDs/protocols"):
        _pass("AscendPrepareSyncHandoff", function)


def _with_extra_vector_fill(function, value):
    def append_read(block):
        ub = block.alloc_buffers[0]
        hidden = tir.Evaluate(
            tir.Call(
                "handle",
                tvm.ir.Op.get("tl.ascend_fill"),
                [
                    tir.StringImm("Fill<float>"),
                    ub.access_ptr("w"),
                    value,
                    4096,
                ],
            )
        )
        return tir.Block(
            block.iter_vars,
            block.reads,
            block.writes,
            block.name_hint,
            tir.SeqStmt([block.body, tir.AttrStmt(tir.const(0), "resource_scope", 1, hidden)]),
            block.init,
            block.alloc_buffers,
            block.match_buffers,
            block.annotations,
        )

    return function.with_body(tir.stmt_functor.ir_transform(function.body, None, append_read, ["tir.Block"]))


def test_hidden_gm_scalar_read_cannot_escape_the_release_lifetime():
    function = _synthetic("c2v")
    workspace = next(iter(function.buffer_map.values()))
    function = _with_extra_vector_fill(function, tir.BufferLoad(workspace, [0, 0, 0]))
    with pytest.raises(tvm.TVMError, match="hidden GM scalar read"):
        _pass("AscendPrepareSyncHandoff", function)


def test_nested_opaque_scalar_call_cannot_hide_a_gm_or_state_effect():
    function = _with_extra_vector_fill(_synthetic("c2v"), tir.call_extern("float32", "unknown_opaque_scalar"))
    with pytest.raises(tvm.TVMError, match="nested opaque call"):
        _pass("AscendPrepareSyncHandoff", function)


def test_internal_workspace_parameter_is_not_a_caller_entry_definition():
    function = _synthetic()
    scratch = tir.decl_buffer((1, 128, 64), "float16", name="hidden", scope="global")
    handle = tir.Var("hidden_handle", "handle")

    def append_read(node):
        if (
            not isinstance(node.value, tir.Call)
            or not isinstance(node.value.args[0], tir.StringImm)
            or not node.value.args[0].value.startswith("tl::ascend::copy_gm_to_l1")
        ):
            return node
        arguments = list(node.value.args)
        arguments[1] = scratch.access_ptr("r", offset=arguments[1].args[2])
        return tir.SeqStmt([node, tir.Evaluate(tir.Call(node.value.dtype, node.value.op, arguments))])

    body = tir.stmt_functor.ir_transform(function.body, None, append_read, ["tir.Evaluate"])
    buffers = dict(function.buffer_map.items())
    buffers[handle] = scratch
    function = tir.PrimFunc([*function.params, handle], body, function.ret_type, buffers, function.attrs)
    # A real external input is permitted by the fixed protocol envelope.
    _pass("AscendPrepareSyncHandoff", function)
    function = function.with_attr("auto_gm_indices", [tir.IntImm("uint32", 1)])
    with pytest.raises(tvm.TVMError, match="Internal GM workspace.*caller entry"):
        _pass("AscendPrepareSyncHandoff", function)


@pytest.mark.parametrize("direction", ["v2c", "c2v"])
def test_internal_channel_requires_and_keeps_its_real_producer_protocol(direction):
    function = _synthetic(direction).with_attr("auto_gm_indices", [tir.IntImm("uint32", 0)])
    function = _pass("AscendPrepareSyncHandoff", function)
    _verify(_pass("AscendInsertSyncHandoff", function))


def test_induction_endpoint_must_fit_the_original_integer_dtype():
    function = _synthetic(trips=1)

    def large_minimum(loop):
        if loop.loop_var.name != "vr":
            return loop
        return tir.For(
            loop.loop_var,
            tir.IntImm("int32", 2**31 - 1),
            loop.extent,
            loop.kind,
            loop.body,
            loop.thread_binding,
            loop.annotations,
        )

    function = function.with_body(tir.stmt_functor.ir_transform(function.body, None, large_minimum, ["tir.For"]))
    with pytest.raises(tvm.TVMError, match="endpoint/update.*actual integer dtype"):
        _pass("AscendPrepareSyncHandoff", function)


def _with_nonchannel_gm_effect(function, kind):
    axes, buffers = {}, []

    def collect(node):
        if isinstance(node, tir.AttrStmt) and node.attr_key == "thread_extent":
            axes[node.node.thread_tag] = node.node.var
        if isinstance(node, tir.Block):
            buffers.extend(node.alloc_buffers)

    tir.stmt_functor.post_order_visit(function.body, collect)
    cid, vid = axes["blockIdx.x"], axes["threadIdx.x"]
    ub = next(buffer for buffer in buffers if buffer.scope() == "shared.ub")
    extra = tir.decl_buffer((4, 128, 64), "float16", name="other_gm", scope="global")
    handle = tir.Var("other_handle", "handle")

    def inject(node):
        if (
            not isinstance(node.value, tir.Call)
            or not isinstance(node.value.args[0], tir.StringImm)
            or not node.value.args[0].value.startswith("tl::ascend::copy_ub_to_gm")
        ):
            return node
        statements = []
        if kind in {"shared_input", "cross_group_raw"}:
            read_offset = 0 if kind == "shared_input" else ((cid + 1) % 4) * 8192 + vid * 4096
            statements.append(
                tir.Evaluate(
                    tir.call_extern(
                        "handle",
                        "tl::ascend::copy_gm_to_ub<half,64,64>",
                        extra.access_ptr("r", offset=read_offset),
                        ub.access_ptr("w"),
                        64,
                        64,
                        64,
                        tir.const(0, "float16"),
                        64,
                        64,
                    )
                )
            )
        statements.append(node)
        if kind != "shared_input":
            write_offset = cid * 8192 + vid * 4096
            if kind == "output_without_cid":
                write_offset = vid * 4096
            elif kind == "output_without_vid":
                write_offset = cid * 8192
            arguments = list(node.value.args)
            arguments[2] = extra.access_ptr("w", offset=write_offset)
            statements.append(tir.Evaluate(tir.Call(node.value.dtype, node.value.op, arguments)))
        return tir.SeqStmt(statements)

    body = tir.stmt_functor.ir_transform(function.body, None, inject, ["tir.Evaluate"])
    mapping = dict(function.buffer_map.items())
    mapping[handle] = extra
    return tir.PrimFunc([*function.params, handle], body, function.ret_type, mapping, function.attrs)


@pytest.mark.parametrize("kind", ["output_without_cid", "cross_group_raw", "output_without_vid"])
def test_unprotected_other_gm_writes_are_rejected_across_groups_and_vector_actors(kind):
    function = _with_nonchannel_gm_effect(_synthetic(cores=4), kind)
    with pytest.raises(tvm.TVMError, match="non-channel or cross-group GM RAW/WAR/WAW"):
        _pass("AscendPrepareSyncHandoff", function)


def test_shared_global_input_read_read_is_not_a_false_cross_core_dependency():
    function = _with_nonchannel_gm_effect(_synthetic(cores=4), "shared_input")
    function = _pass("AscendPrepareSyncHandoff", function)
    _verify(_pass("AscendInsertSyncHandoff", function))


def _with_round_predicates(function, kind):
    def guard(loop):
        vector = loop.loop_var.name == "vr"
        if kind == "opposite_parity":
            predicate = loop.loop_var % 2 == (1 if vector else 0)
        elif kind == "shifted_window":
            predicate = loop.loop_var >= 2 if vector else loop.loop_var < 2
        else:
            predicate = loop.loop_var < 2
        return tir.For(
            loop.loop_var,
            loop.min,
            loop.extent,
            loop.kind,
            tir.IfThenElse(predicate, loop.body, None),
            loop.thread_binding,
            loop.annotations,
        )

    return function.with_body(tir.stmt_functor.ir_transform(function.body, None, guard, ["tir.For"]))


@pytest.mark.parametrize("kind", ["opposite_parity", "shifted_window"])
def test_equal_endpoint_counts_cannot_rematch_different_guard_epochs(kind):
    function = _with_round_predicates(_synthetic(trips=4), kind)
    with pytest.raises(tvm.TVMError, match="uniform guard/epoch coordinates differ"):
        _pass("AscendPrepareSyncHandoff", function)


def test_matching_guards_compare_epoch_values_instead_of_variable_names():
    function = _with_round_predicates(_synthetic(trips=4), "matching")
    function = _pass("AscendPrepareSyncHandoff", function)
    _verify(_pass("AscendInsertSyncHandoff", function))


def test_legacy_strategy_keeps_all_three_passes_identity():
    function = _synthetic()
    with tvm.transform.PassContext(config=handoff_configs("legacy")):
        for name in (
            "AscendPrepareSyncHandoff",
            "AscendInsertSyncHandoff",
            "AscendVerifySyncHandoff",
        ):
            result = getattr(tilelang.transform, name)(TARGET, "A3")(tvm.IRModule.from_expr(function))["main"]
            assert result.same_as(function)


@pytest.mark.parametrize("direction", ["v2c", "c2v"])
@pytest.mark.parametrize("trips", [1, 3, 17])
def test_real_copy_cv_gemm_fixture_passes_the_public_lowering_chain(direction, trips):
    function = make_cv_handoff_kernel(direction, trips=trips, cores=1)
    module = tvm.IRModule.from_expr(function.with_attr("npu_platform", "A3"))
    with tvm.transform.PassContext(config=handoff_configs()):
        lowered = OptimizeForTarget(LowerAndLegalize(module, TARGET), TARGET, "A3")
    assert SOURCE in lowered["main"].attrs and PLAN in lowered["main"].attrs


@pytest.mark.parametrize("direction", ["v2c", "c2v"])
def test_manual_legacy_reference_lowers_without_new_cross_ownership(direction):
    function = make_cv_handoff_kernel(direction, trips=17, cores=1, manual_legacy=True)
    module = tvm.IRModule.from_expr(function.with_attr("npu_platform", "A3"))
    with tvm.transform.PassContext(config=handoff_configs("legacy")):
        lowered = OptimizeForTarget(LowerAndLegalize(module, TARGET), TARGET, "A3")
    assert SOURCE not in lowered["main"].attrs and PLAN not in lowered["main"].attrs


def _large_vector_read(rows):
    source = """@T.prim_func
def main(Input: T.Tensor((rows, 16), "float16"),
         Workspace: T.Tensor((128, 64), "float16")):
    T.func_attr({"tir.noalias": True})
    with T.Kernel(1, is_npu=True) as (cid, vid):
        large = T.alloc_ub((rows, 16), "float16")
        small = T.alloc_ub((64, 64), "float16")
        cube = T.alloc_L1((128, 64), "float16")
        with T.Scope("V"):
            T.copy(Input, large)
            T.tile.fill(small, 1.0)
            T.copy(small, Workspace[vid * 64:(vid + 1) * 64, :])
        with T.Scope("C"):
            T.copy(Workspace, cube)
"""
    return tvm.script.from_source(source, extra_vars={"T": T, "rows": rows})


def test_datacopypad_row_limit_fits_real_ub_allocation_at_the_boundary():
    function = _large_vector_read(4095)
    module = tvm.IRModule.from_expr(function.with_attr("npu_platform", "A3"))
    with tvm.transform.PassContext(config=handoff_configs()):
        lowered = OptimizeForTarget(LowerAndLegalize(module, TARGET), TARGET, "A3")["main"]
    ub_sizes = [int(size) for variable, size in lowered.attrs["size_map"].items() if variable.type_annotation.storage_scope == "shared.ub"]
    allocated = []

    def allocation(node):
        if isinstance(node, tir.Allocate) and node.buffer_var.type_annotation.storage_scope == "shared.ub":
            byte_count = tvm.DataType(node.dtype).bits * tvm.DataType(node.dtype).lanes // 8
            for extent in node.extents:
                byte_count *= int(tvm.arith.Analyzer().simplify(extent))
            allocated.append(byte_count)

    tir.stmt_functor.post_order_visit(lowered.body, allocation)
    assert max(ub_sizes) >= 4095 * 16 * 2
    assert max(allocated) >= 4095 * 16 * 2
    assert 4095 * 16 * 2 + 64 * 64 * 2 <= 196352
    assert int(lowered.attrs[PLAN]["channel_count"]) == 1


def test_datacopypad_over_limit_is_rejected_even_with_a_legal_ub_budget():
    function = _large_vector_read(4096)
    assert 4096 * 16 * 2 + 64 * 64 * 2 <= 196352
    module = tvm.IRModule.from_expr(function.with_attr("npu_platform", "A3"))
    with (
        tvm.transform.PassContext(config=handoff_configs()),
        pytest.raises(tvm.TVMError, match="DataCopyPad.*4095"),
    ):
        OptimizeForTarget(LowerAndLegalize(module, TARGET), TARGET, "A3")
