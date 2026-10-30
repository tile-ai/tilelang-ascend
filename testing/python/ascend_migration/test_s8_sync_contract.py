"""Independent S8 source/value, physical-version and emitted-protocol faults.

These tests stop before device compilation. Malformed IR never reaches an NPU.
"""

import pytest

import tilelang
from tilelang import tvm
from tilelang.engine.phase import LowerAndLegalize, OptimizeForTarget
from test_s8_kernels import make_gemm_kernel, make_mixed_kernel, make_reduce_kernel

TIR = tvm.tir
TARGET = tvm.target.Target("c -keys=ascend -model=ascendc")
REMAP = "tl.ascend_auto_schedule_buffer_remap"


def configs():
    return {
        "tl.ascend_pipeline_planner": "auto",
        "tl.ascend_sync_strategy": "a3",
        "tl.ascend_auto_sync": True,
        "tl.ascend_memory_planning": True,
        "tl.ascend_auto_cv_combine": True,
        "tl.ascend_auto_cross_core_sync": False,
        "tl.ascend_auto_sync_vs": False,
    }


def lower(function):
    """Capture the actual final verifier input; also return legalized mask IR."""
    captured = []
    original = tilelang.transform.AscendVerifyAutoSchedule

    def observe(*args, **kwargs):
        verify_pass = original(*args, **kwargs)

        def run(module):
            captured.append(module["main"])
            return verify_pass(module)

        return run

    tilelang.transform.AscendVerifyAutoSchedule = observe
    try:
        module = tvm.IRModule.from_expr(function.with_attr("npu_platform", "A3"))
        with tvm.transform.PassContext(config=configs()):
            module = OptimizeForTarget(LowerAndLegalize(module, TARGET), TARGET, "A3")
    finally:
        tilelang.transform.AscendVerifyAutoSchedule = original
    assert len(captured) == 1
    return captured[0], module["main"]


def verify(function):
    before = tvm.ir.save_json(function)
    with tvm.transform.PassContext(config=configs()):
        checked = tilelang.transform.AscendVerifyAutoSchedule(TARGET, "A3")(tvm.IRModule.from_expr(function))["main"]
    assert checked.same_as(function)
    assert tvm.ir.save_json(function) == before


def rewrite_calls(function, callback):
    return function.with_body(TIR.stmt_functor.ir_transform(function.body, None, callback, ["tir.Call"]))


def call_name(call):
    return str(getattr(call.op, "name", ""))


@pytest.fixture(scope="module")
def gemm():
    return lower(make_gemm_kernel(trips=3))


@pytest.fixture(scope="module")
def reduction():
    return lower(make_reduce_kernel(trips=3))


@pytest.fixture(scope="module")
def mixed():
    return lower(make_mixed_kernel(trips=3, operation="reduce"))


@pytest.mark.parametrize("fixture", ["gemm", "reduction", "mixed"])
def test_final_source_physical_and_protocol_verifier_is_read_only(request, fixture):
    verify(request.getfixturevalue(fixture)[0])


def test_missing_all_boundary_is_rejected(gemm):
    changed = False

    def remove(node):
        nonlocal changed
        if not changed and isinstance(node.value, TIR.Call) and call_name(node.value) == "tl.ascend_auto_barrier":
            changed = True
            return TIR.Evaluate(0)
        return None

    function = gemm[0].with_body(TIR.stmt_functor.ir_transform(gemm[0].body, None, remove, ["tir.Evaluate"]))
    assert changed
    with pytest.raises(tvm.TVMError, match="ALL completion boundary"):
        verify(function)


@pytest.mark.parametrize("damage", ["ready", "release", "mode", "id", "pipe"])
def test_actual_cross_protocol_faults_are_not_hidden_by_stored_plan(mixed, damage):
    changed = False

    def mutate(node):
        nonlocal changed
        if changed or not isinstance(node.value, TIR.Call):
            return None
        call = node.value
        if call_name(call) != "tl.ascend_auto_set_cross_flag":
            return None
        args = list(call.args)
        flag = int(args[2])
        if damage in {"ready", "release"}:
            if flag != (0 if damage == "ready" else 1):
                return None
            changed = True
            return TIR.Evaluate(0)
        if damage == "mode":
            args[0] = TIR.IntImm("int32", 1)
        elif damage == "id":
            args[2] = TIR.IntImm("int32", 2)
        else:
            args[1] = TIR.StringImm("V")
        changed = True
        return TIR.Evaluate(TIR.Call(call.dtype, call.op, args))

    function = mixed[0].with_body(TIR.stmt_functor.ir_transform(mixed[0].body, None, mutate, ["tir.Evaluate"]))
    assert changed
    with pytest.raises(tvm.TVMError, match="Auto|auto|mode2|PIPE|producer|consumer"):
        verify(function)


@pytest.mark.parametrize("damage", ["swap_inputs", "clear", "n_actual"])
def test_gemm_ordered_operand_roles_and_payload_are_preserved(gemm, damage):
    changed = False

    def mutate(call):
        nonlocal changed
        if changed or call_name(call) != "tl.ascend_gemm_v0":
            return None
        args = list(call.args)
        if damage == "swap_inputs":
            args[1], args[2] = args[2], args[1]
        elif damage == "clear":
            args[4] = TIR.const(False, "bool")
        else:
            args[5] = args[5] - 16
        changed = True
        return TIR.Call(call.dtype, call.op, args)

    function = rewrite_calls(gemm[0], mutate)
    assert changed
    with pytest.raises(tvm.TVMError, match="Auto|auto|GEMM"):
        verify(function)


@pytest.mark.parametrize("damage", ["pitch", "slot"])
def test_materialized_version_metadata_must_match_source_and_actual_accesses(gemm, damage):
    rows = [dict(row.items()) for row in gemm[0].attrs[REMAP]]
    candidates = [row for row in rows if int(row["versions"]) == 2]
    assert candidates, "The full backend fixture must really materialize N2"
    row = candidates[0]
    if damage == "pitch":
        row["pitch"] = TIR.IntImm("int64", int(row["pitch"]) + 32)
    else:
        slots = [dict(slot.items()) for slot in row["slots"]]
        slots[0]["slot"] = TIR.IntImm("int64", 1 - int(slots[0]["slot"]))
        row["slots"] = slots
    with pytest.raises(tvm.TVMError, match="Auto|auto|selector|pitch|role"):
        verify(gemm[0].with_attr(REMAP, rows))


def test_actual_n2_pointer_selector_is_recomputed_independently(gemm):
    owners = [row for row in gemm[0].attrs[REMAP] if int(row["versions"]) == 2]
    assert owners
    data = owners[0]["physical_data"]
    changed = False

    def mutate(call):
        nonlocal changed
        if changed or call_name(call) != "tir.tvm_access_ptr" or not call.args[1].same_as(data):
            return None
        args = list(call.args)
        args[2] = args[2] + 16
        changed = True
        return TIR.Call(call.dtype, call.op, args)

    function = rewrite_calls(gemm[0], mutate)
    assert changed
    with pytest.raises(tvm.TVMError, match="Auto|auto|footprint|role"):
        verify(function)


def verify_mask(function):
    before = tvm.ir.save_json(function)
    checked = tvm.get_global_func("tl.transform._AscendVerifyAutoMaskContract")(function)
    assert checked.same_as(function) and tvm.ir.save_json(function) == before


def test_selected_reduction_has_an_independent_auto_mask_contract(reduction):
    verify_mask(reduction[1])
    with pytest.raises(tvm.TVMError, match="Unsupported constraint managed mask terminal"):
        tvm.get_global_func("tl.transform._AscendVerifyManagedMaskContract")(reduction[1])


@pytest.mark.parametrize("boundary", ["source", "final"])
def test_reduction_scratch_extent_cannot_replace_the_required_geometry(reduction, boundary):
    changed = False

    def shrink(call):
        nonlocal changed
        if changed or call_name(call) not in {"tl.ascend_reduce", "tl.ascend_reduce_fp32_v2"}:
            return None
        args = list(call.args)
        pointer = args[3]
        pointer_args = list(pointer.args)
        pointer_args[3] = TIR.IntImm(pointer_args[3].dtype, 1)
        args[3] = TIR.Call(pointer.dtype, pointer.op, pointer_args)
        changed = True
        return TIR.Call(call.dtype, call.op, args)

    original = reduction[0]
    if boundary == "source":
        original = original.attrs["tl.ascend_auto_schedule"]["source_function"]
    function = rewrite_calls(original, shrink)
    assert changed
    with pytest.raises(tvm.TVMError, match="scratch|Scratch|pointer.*interval|pointer.*extent"):
        if boundary == "source":
            tvm.get_global_func("tl.transform.AnalyzeAscendAutoSchedule")(function)
        else:
            verify(function)


def test_reduction_scratch_must_fit_its_real_source_allocation(reduction):
    source = reduction[0].attrs["tl.ascend_auto_schedule"]["source_function"]
    roots = []

    def find(node):
        if isinstance(node, TIR.Call) and call_name(node) in {
            "tl.ascend_reduce",
            "tl.ascend_reduce_fp32_v2",
        }:
            roots.append(node.args[3].args[1])

    TIR.stmt_functor.post_order_visit(source.body, find)
    assert roots and all(root.same_as(roots[0]) for root in roots)
    changed = False

    def shrink(node):
        nonlocal changed
        if isinstance(node, TIR.Allocate) and node.buffer_var.same_as(roots[0]):
            changed = True
            return TIR.Allocate(
                node.buffer_var,
                node.dtype,
                [TIR.IntImm("int64", 1)],
                node.condition,
                node.body,
                node.annotations,
            )
        if isinstance(node, TIR.Block):
            buffers = []
            for buffer in node.alloc_buffers:
                if buffer.data.same_as(roots[0]):
                    changed = True
                    buffer = TIR.decl_buffer([1], buffer.dtype, buffer.name, data=buffer.data, scope=buffer.scope())
                buffers.append(buffer)
            return TIR.Block(
                node.iter_vars,
                node.reads,
                node.writes,
                node.name_hint,
                node.body,
                node.init,
                buffers,
                node.match_buffers,
                node.annotations,
            )
        return None

    function = source.with_body(TIR.stmt_functor.ir_transform(source.body, None, shrink, ["tir.Allocate", "tir.Block"]))
    assert changed
    with pytest.raises(tvm.TVMError, match="footprint exceeds its real allocation|scratch|Scratch"):
        tvm.get_global_func("tl.transform.AnalyzeAscendAutoSchedule")(function)


@pytest.mark.parametrize("boundary", ["source", "planner"])
@pytest.mark.parametrize("damage", ["global", "half_typed", "source_overlap"])
def test_private_reduction_scratch_keeps_its_audited_role(reduction, boundary, damage):
    source = reduction[0].attrs["tl.ascend_auto_schedule"]["source_function"]
    changed = False

    def mutate(call):
        nonlocal changed
        if call_name(call) not in {"tl.ascend_reduce", "tl.ascend_reduce_fp32_v2"}:
            return None
        args = list(call.args)
        pointer = args[3]
        pointer_args = list(pointer.args)
        if damage == "global":
            pointer_args[1] = source.buffer_map[source.params[0]].data
        elif damage == "half_typed":
            pointer_args[0] = TIR.Call("float16", pointer_args[0].op, [])
            pointer_args[3] = pointer_args[3] * 2
        else:
            pointer_args[1] = args[2].args[1]
        args[3] = TIR.Call(pointer.dtype, pointer.op, pointer_args)
        changed = True
        return TIR.Call(call.dtype, call.op, args)

    function = rewrite_calls(source, mutate)
    assert changed
    with (
        tvm.transform.PassContext(config=configs()),
        pytest.raises(tvm.TVMError, match="scratch|Scratch"),
    ):
        if boundary == "source":
            tvm.get_global_func("tl.transform.AnalyzeAscendAutoSchedule")(function)
        else:
            tilelang.transform.AscendAutoSchedule(TARGET, "A3")(tvm.IRModule.from_expr(function))


@pytest.mark.parametrize("damage", ["mode", "payload"])
def test_mask_reaching_setter_fault_is_rejected_after_legalization(reduction, damage):
    changed = False
    wanted = "tl.ascend_set_mask_mode" if damage == "mode" else "tl.ascend_set_mask_payload"

    def mutate(node):
        nonlocal changed
        if changed or not isinstance(node.value, TIR.Call) or call_name(node.value) != wanted:
            return None
        changed = True
        if damage == "mode":
            return TIR.Evaluate(0)
        args = list(node.value.args)
        args[0] = TIR.const(0, "uint64")
        return TIR.Evaluate(TIR.Call(node.value.dtype, node.value.op, args))

    function = reduction[1].with_body(TIR.stmt_functor.ir_transform(reduction[1].body, None, mutate, ["tir.Evaluate"]))
    assert changed
    with pytest.raises(tvm.TVMError, match="matching reaching setter"):
        verify_mask(function)


def test_same_pool_alias_cannot_overwrite_an_earlier_source_producer(gemm):
    function = gemm[0]
    storages = function.attrs["tl.ascend_auto_schedule"]["source_analysis"]["storages"]
    owners = [
        row
        for row in function.attrs[REMAP]
        if any(storage["data"].same_as(row["logical_data"]) and str(storage["scope"]) == "shared.l1" for storage in storages)
    ]
    assert len(owners) >= 2
    addresses = dict(function.attrs["address_map"].items())
    addresses[owners[1]["physical_data"]] = addresses[owners[0]["physical_data"]]
    with pytest.raises(tvm.TVMError, match="stale/overwritten physical producer"):
        verify(function.with_attr("address_map", addresses))


def test_actual_allocate_cannot_exceed_the_real_size_map(gemm):
    function = gemm[0]
    data = next(row["physical_data"] for row in function.attrs[REMAP] if int(row["versions"]) == 2)
    changed = False

    def enlarge(node):
        nonlocal changed
        if not node.buffer_var.same_as(data):
            return None
        extents = list(node.extents)
        extents[0] = extents[0] * 2
        changed = True
        return TIR.Allocate(node.buffer_var, node.dtype, extents, node.condition, node.body, node.annotations)

    function = function.with_body(TIR.stmt_functor.ir_transform(function.body, None, enlarge, ["tir.Allocate"]))
    assert changed
    with pytest.raises(tvm.TVMError, match="Allocate exceeds size_map"):
        verify(function)


def test_public_parameter_order_is_part_of_the_original_semantics(gemm):
    function = gemm[0]
    params = list(function.params)
    params[0], params[1] = params[1], params[0]
    changed = TIR.PrimFunc(params, function.body, function.ret_type, function.buffer_map, function.attrs)
    with pytest.raises(tvm.TVMError, match="ordered public parameter role"):
        verify(changed)


def test_empty_source_cannot_acquire_a_hardware_event():
    function = lower(make_gemm_kernel(trips=0))[0]
    event = TIR.Evaluate(TIR.Call("handle", tvm.ir.Op.get("tl.ascend_auto_barrier"), [TIR.StringImm("PIPE_ALL")]))
    body = TIR.SeqStmt(
        [
            function.body,
            TIR.AttrStmt(TIR.IntImm("int32", 0), "resource_scope", TIR.IntImm("int32", 0), event),
        ]
    )
    with pytest.raises(tvm.TVMError, match="empty source acquired hardware tasks/events"):
        verify(function.with_body(body))


def test_nonempty_source_requires_the_same_actual_vector_launch_axis(mixed):
    changed = False

    def shrink(node):
        nonlocal changed
        if node.attr_key == "thread_extent" and node.node.thread_tag in {
            "blockIdx.y",
            "threadIdx.x",
        }:
            assert not changed and int(node.value) == 2
            changed = True
            return TIR.AttrStmt(node.node, node.attr_key, TIR.IntImm(node.value.dtype, 1), node.body)
        return None

    function = mixed[0].with_body(TIR.stmt_functor.ir_transform(mixed[0].body, None, shrink, ["tir.AttrStmt"]))
    assert changed
    with pytest.raises(tvm.TVMError, match="physical launch mapping changed"):
        verify(function)


@pytest.mark.parametrize("damage", ["capacity", "dtype"])
def test_caller_workspace_buffer_abi_cannot_be_resized_or_retyped(mixed, damage):
    function = mixed[0]
    channel = function.attrs["tl.ascend_auto_schedule"]["source_contract"]["channel"]["data"]
    buffers = dict(function.buffer_map.items())
    matches = [param for param, buffer in buffers.items() if buffer.data.same_as(channel)]
    assert len(matches) == 1
    param = matches[0]
    buffer = buffers[param]
    shape = list(buffer.shape)
    dtype = buffer.dtype
    if damage == "capacity":
        shape[0] = shape[0] * 2
    else:
        assert dtype == "float32"
        dtype = "int32"
    buffers[param] = TIR.decl_buffer(shape, dtype, buffer.name, data=buffer.data, scope=buffer.scope())
    changed = TIR.PrimFunc(function.params, function.body, function.ret_type, buffers, function.attrs)
    with pytest.raises(tvm.TVMError, match="GM storage capacity/ABI|buffer ABI|parameter.*dtype"):
        verify(changed)
