"""Native multi-version contracts; synthetic costs are IR fixtures, not A3 evidence."""

from collections import Counter

import pytest

import tilelang
from tilelang import tvm
from test_s4_pipeline_ir import _fills, _loops, _plan, _solver
from test_s5_cost_model import seal, synthetic_profile


tir = tvm.tir
REMAP = "tl.ascend_pipeline_buffer_remap"


def fixture_profile():
    profile = synthetic_profile()
    profile["quantum_ns"] = 1
    profile["records"] = []
    for dtype, width in (("float16", 2), ("float32", 4)):
        for elements in (32, 64):
            for kind in ("gm_to_ub", "ub_to_gm", "fill", "add"):
                profile["records"].append(
                    {
                        "kind": kind,
                        "dtype": dtype,
                        "elements": elements,
                        "bytes": elements * width,
                        "rows": 1,
                        "latency": 4,
                        "ii": 1,
                        "latency_effective_ns": 4,
                        "ii_effective_ns": 1,
                    }
                )
    return seal(profile)


def configs(profile=None):
    from test_s4_kernels import vector_pass_configs

    result = vector_pass_configs()
    result["tl.ascend_pipeline_cost_model"] = fixture_profile() if profile is None else profile
    return result


def _value(expression, environment):
    value = tvm.arith.Analyzer().simplify(tir.stmt_functor.substitute(expression, environment))
    assert isinstance(value, tir.IntImm), value
    return int(value)


def _execute(statement, environment, callback):
    """Execute actual generated clocks, blocks and predicates, including short-trip guards."""
    if isinstance(statement, tir.SeqStmt):
        for child in statement.seq:
            _execute(child, environment, callback)
    elif isinstance(statement, tir.For):
        minimum = _value(statement.min, environment)
        extent = _value(statement.extent, environment)
        assert extent >= 0
        for index in range(minimum, minimum + extent):
            nested = dict(environment)
            nested[statement.loop_var] = tir.IntImm(statement.loop_var.dtype, index)
            _execute(statement.body, nested, callback)
    elif isinstance(statement, tir.BlockRealize):
        if _value(statement.predicate, environment):
            _execute(statement.block.body, environment, callback)
    elif isinstance(statement, tir.IfThenElse):
        selected = statement.then_case if _value(statement.condition, environment) else statement.else_case
        if selected is not None:
            _execute(selected, environment, callback)
    elif isinstance(statement, tir.AttrStmt):
        _execute(statement.body, environment, callback)
    elif isinstance(statement, tir.LetStmt):
        nested = dict(environment)
        nested[statement.var] = tir.IntImm(statement.var.dtype, _value(statement.value, environment))
        _execute(statement.body, nested, callback)
    elif isinstance(statement, (tir.Evaluate, tir.BufferStore)):
        if isinstance(statement, tir.Evaluate) and isinstance(statement.value, tir.IntImm):
            assert int(statement.value) == 0
        else:
            callback(statement, environment)
    else:
        raise AssertionError(type(statement).__name__)


def _legacy_probe(trip, minimum=0):
    index = tir.Var("logical_iteration", "int32")
    body = tir.SeqStmt([tir.Evaluate(tir.call_extern("handle", "observe_unit", unit, index)) for unit in range(3)])
    loop = tir.For(
        index,
        minimum,
        trip,
        tir.ForKind.SERIAL,
        body,
        annotations={"software_pipeline_stage": [0, 1, 2], "software_pipeline_order": [0, 1, 2]},
    )
    return tvm.IRModule.from_expr(tir.PrimFunc([], loop).with_attr("global_symbol", "main"))


@pytest.mark.parametrize("trip", [0, 1, 2, 4])
@pytest.mark.parametrize("minimum", [0, 5])
def test_legacy_actual_short_trip_predicates_execute_each_unit_once(trip, minimum):
    module = tilelang.transform.InjectSoftwarePipeline()(_legacy_probe(trip, minimum))
    trace = []

    def observe(statement, environment):
        call = statement.value
        assert call.args[0].value == "observe_unit"
        trace.append((_value(call.args[1], environment), _value(call.args[2], environment)))

    _execute(module["main"].body, {}, observe)
    assert Counter(trace) == Counter((unit, index) for index in range(minimum, minimum + trip) for unit in range(3))


def test_nonzero_minimum_pointer_and_buffer_select_the_same_physical_version():
    buffer = tir.decl_buffer((64,), "float32", name="local", scope="shared.ub")
    index = tir.Var("logical_iteration", "int32")
    write = tir.BufferStore(buffer, tir.const(1.0, "float32"), [0])
    read = tir.Evaluate(tir.call_extern("handle", "observe_slot", buffer.access_ptr("r"), tir.BufferLoad(buffer, [0])))
    block = tir.Block([], [], [], "units", tir.SeqStmt([write, read]), alloc_buffers=[buffer])
    loop = tir.For(
        index,
        5,
        4,
        tir.ForKind.SERIAL,
        tir.BlockRealize([], True, block),
        annotations={"software_pipeline_stage": [0, 1], "software_pipeline_order": [0, 1]},
    )
    function = tir.PrimFunc([], loop).with_attr("global_symbol", "main")
    module = tilelang.transform.InjectSoftwarePipeline()(tvm.IRModule.from_expr(function))
    slots = []

    def observe(statement, environment):
        if isinstance(statement, tir.BufferStore):
            return
        call = statement.value
        pointer, load = call.args[1:]
        slots.append((_value(pointer.args[2], environment) // 64, _value(load.indices[0], environment)))

    _execute(module["main"].body, {}, observe)
    assert slots == [(0, 0), (1, 1), (0, 0), (1, 1)]
    assert int(module["main"].attrs["buffer_versions"][buffer.data]) == 2


@pytest.mark.parametrize("bound", [1, 2, 3])
@pytest.mark.parametrize("trip", [0, 1, 4])
def test_modulo_native_publishes_actual_versions_and_physical_geometry(bound, trip):
    planned = _plan(_fills(stages=bound, extent=trip), configs())
    candidate = planned["main"].attrs["tl.ascend_constraint_plan"]["candidates"][0]
    assert int(candidate["ii"]) > 0
    assert all(1 <= int(value) <= bound for value in candidate["buffer_versions"])
    if trip:
        loop = _loops(planned["main"])[0]
        assert "software_pipeline_stage" in loop.annotations
        assert str(candidate["execution"]) == "injected"
    else:
        assert not _loops(planned["main"])
        assert str(candidate["execution"]) == "zero_trip_noop"
    injected = tilelang.transform.InjectSoftwarePipeline()(planned)
    remaps = injected["main"].attrs[REMAP]
    assert len(remaps) == (3 if trip else 0)
    for remap in remaps:
        versions, stride = int(remap["versions"]), int(remap["version_stride_bytes"])
        assert versions <= bound
        assert stride == int(remap["version_span_bytes"]) == 256
        assert int(remap["total_bytes"]) == versions * stride
        assert remap["logical_data"].same_as(remap["physical_data"])


@pytest.mark.parametrize("bound", [2, 3])
def test_explicit_false_offset_keeps_all_stages_zero(bound):
    planned = _plan(_fills(stages=bound, annotations={"enable_offset": False}), configs())
    assert [int(value) for value in _loops(planned["main"])[0].annotations["software_pipeline_stage"]] == [0, 0, 0]


@pytest.mark.parametrize(
    "mutation,error",
    [("missing", "exactly one matching"), ("unit", "nanosecond"), ("zero", "latency/issue")],
)
def test_native_cost_records_are_matched_without_fallback(mutation, error):
    profile = fixture_profile()
    if mutation == "missing":
        profile["records"] = [record for record in profile["records"] if record["kind"] != "fill"]
    elif mutation == "unit":
        profile["unit"] = "cycles"
    else:
        for record in profile["records"]:
            record["latency"] = 0
    # Direct native invocation deliberately bypasses Python artifact validation.
    from tilelang.transform.ascend_schedule_solver_ffi import register_solver

    register_solver()
    with (
        tvm.transform.PassContext(config=configs(profile)),
        pytest.raises(tvm.TVMError, match=error),
    ):
        tvm.get_global_func("tl.transform.PipelinePlanning")()(_fills(stages=2))


def test_s5_no_profile_is_a_diagnostic_not_unit_cost_fallback():
    with pytest.raises(tvm.TVMError, match="explicit calibrated cost profile"):
        _plan(_fills(stages=2))


def test_existing_solver_callback_still_gets_independent_verification():
    def invalid(problem):
        count = len(problem["latencies"])
        return tvm.runtime.convert(
            {
                "schema_version": 1,
                "status": "SAT",
                "reason": "",
                "start_times": [0] * count,
                "order": list(range(count)),
                "ii": 1,
                "buffer_versions": [1] * len(problem["buffer_sizes"]),
            }
        )

    with _solver(invalid), pytest.raises(tvm.TVMError, match="Invalid consumed modulo plan"):
        _plan(_fills(stages=2), configs())


def test_actual_version_override_is_consumed_by_injector():
    def valid(problem):
        return tvm.runtime.convert(
            {
                "schema_version": 1,
                "status": "SAT",
                "reason": "",
                "start_times": [0, 1, 2],
                "order": [0, 1, 2],
                "ii": 4,
                "buffer_versions": [3, 3, 3],
            }
        )

    with _solver(valid):
        planned = _plan(_fills(stages=3), configs())
    module = tilelang.transform.InjectSoftwarePipeline()(planned)
    assert {int(record["versions"]) for record in module["main"].attrs[REMAP]} == {3}


def _manual(stages, *, dependent=False, minimum=0):
    module = _fills(stages=3, dependent=dependent, minimum=minimum, annotations={"enable_offset": True})
    ordinal = 0

    def mark(statement):
        nonlocal ordinal
        if not isinstance(statement.value, tir.Call):
            return statement
        stage = stages[ordinal]
        ordinal += 1
        return tir.AttrStmt(
            tvm.runtime.convert({"stage": tir.IntImm("int64", stage)}),
            "tl.ascend_pipeline_task_hint",
            tir.const(1, "int32"),
            statement,
        )

    function = module["main"]
    body = tir.stmt_functor.ir_transform(function.body, None, mark, ["tir.Evaluate"])
    if dependent:

        def allocation(block):
            return tir.Block(
                block.iter_vars,
                block.reads,
                block.writes,
                block.name_hint,
                block.body,
                block.init,
                [block.alloc_buffers[0]],
                block.match_buffers,
                block.annotations,
            )

        body = tir.stmt_functor.ir_transform(body, None, allocation, ["tir.Block"])
    return tvm.IRModule.from_expr(function.with_body(body))


def test_manual_stage_preserves_raw_values_for_independent_units():
    planned = _plan(_manual([2, 0, 1]), configs())
    assert [int(value) for value in _loops(planned["main"])[0].annotations["software_pipeline_stage"]] == [2, 0, 1]


def test_manual_stage_cannot_reinterpret_imperative_waw_order():
    with pytest.raises(tvm.TVMError, match="did not produce SAT"):
        _plan(_manual([2, 0, 0], dependent=True), configs())


@pytest.mark.parametrize("minimum", [-(2**31), 2**31 - 4])
def test_stage_skew_must_fit_real_loop_dtype_before_predicate_evaluation(minimum):
    with pytest.raises(tvm.TVMError, match="loop-update limits"):
        _plan(_manual([0, 1, 2], minimum=minimum), configs())


def test_multirow_ub_is_rejected_before_version_stride_materialization():
    with pytest.raises(tvm.TVMError, match="one contiguous 1D/1xL"):
        _plan(_fills(stages=2, allocation_shape=(2, 32)), configs())


def test_prepare_keeps_zero_and_one_trip_request_facts():
    for trip in (0, 1):
        module = _fills(stages=2, extent=trip)
        with tvm.transform.PassContext(config=configs()):
            prepared = tilelang.transform.AscendPrepareConstraintPipeline(tvm.target.Target({"kind": "llvm", "model": "ascendc"}))(module)
            facts = prepared["main"].attrs["tl.ascend_pipeline_degenerate_candidates"]
            assert len(facts) == 1 and int(facts[0]["extent"]) == trip
            planned = tilelang.transform.PipelinePlanning()(prepared)
        assert int(planned["main"].attrs["tl.ascend_constraint_plan"]["schema_version"]) == 3


@pytest.mark.parametrize("bound", [1, 2, 3])
@pytest.mark.parametrize("dtype", ["float16", "float32"])
def test_real_lowered_vector_pipeline_uses_the_selected_version_bound(bound, dtype):
    from test_s4_kernels import make_vector_kernel
    from test_s4_pipeline_ir import _before_planning

    module = _before_planning(make_vector_kernel(192, dtype))
    function = module["main"]

    def requested(loop):
        attributes = dict(loop.annotations)
        if "num_stages" in attributes:
            attributes["num_stages"] = bound
        return tir.For(
            loop.loop_var,
            loop.min,
            loop.extent,
            loop.kind,
            loop.body,
            loop.thread_binding,
            attributes,
        )

    body = tir.stmt_functor.ir_transform(function.body, None, requested, ["tir.For"])
    planned = _plan(tvm.IRModule.from_expr(function.with_body(body)), configs())
    candidate = planned["main"].attrs["tl.ascend_constraint_plan"]["candidates"][0]
    assert max(int(value) for value in candidate["buffer_versions"]) == bound
    injected = tilelang.transform.InjectSoftwarePipeline()(planned)
    assert max(int(record["versions"]) for record in injected["main"].attrs[REMAP]) == bound


@pytest.mark.parametrize("stage", [65, 2**31 - 1])
def test_raw_stage_horizon_is_rejected_before_injector_unroll(stage):
    with pytest.raises(tvm.TVMError, match="supported staging horizon"):
        _plan(_manual([stage, 0, 0]), configs())


def test_static_expansion_product_is_rejected_before_solver_and_injection():
    module = _manual([0, 0, 0])
    function = module["main"]

    def trip(loop):
        return tir.For(loop.loop_var, 0, 4096, loop.kind, loop.body, loop.thread_binding, loop.annotations)

    body = tir.stmt_functor.ir_transform(function.body, None, trip, ["tir.For"])
    with pytest.raises(
        tvm.TVMError,
        match="requested staging window exceeds|Lifetime hardware-task instance limit exceeded",
    ):
        _plan(tvm.IRModule.from_expr(function.with_body(body)), configs())


@pytest.mark.parametrize("trips", [0, 1, 2, 8])
@pytest.mark.parametrize("bound", [1, 2, 3])
def test_shared_kernel_complete_lowering_includes_final_sync_and_mask_proofs(trips, bound):
    from test_s5_kernels import make_pipeline_kernel

    with tvm.transform.PassContext(config=configs()):
        result = tilelang.lower(make_pipeline_kernel(trips=trips, num_stages=bound), target="ascendc", platform="A3")
    assert "auto main" in result.kernel_source or "void main" in result.kernel_source


def test_partial_write_without_undefined_reads_is_safe_to_plan():
    planned = _plan(_fills(stages=2, allocation_shape=(128,)), configs())
    assert int(planned["main"].attrs["tl.ascend_constraint_plan"]["schema_version"]) == 3


def test_partial_initialization_cannot_define_a_complete_read():
    from test_s2_task_ir import _add

    module = _fills(stages=2, allocation_shape=(128,))
    function = module["main"]
    block = _loops(function)[0].body.block
    statements = list(block.body.seq) + [_add(block.alloc_buffers[1], block.alloc_buffers[0], block.alloc_buffers[0], count=128)]
    replacement = tir.Block(
        block.iter_vars,
        block.reads,
        block.writes,
        block.name_hint,
        tir.SeqStmt(statements),
        alloc_buffers=block.alloc_buffers,
    )
    body = tir.stmt_functor.ir_transform(
        function.body,
        None,
        lambda node: tir.BlockRealize([], True, replacement),
        ["tir.BlockRealize"],
    )
    with pytest.raises(tvm.TVMError, match="read.*defin|uninitialized|undefined|readiness"):
        _plan(tvm.IRModule.from_expr(function.with_body(body)), configs())


def test_versioned_read_before_write_is_rejected_before_injection():
    from test_s2_task_ir import _add

    module = _fills(stages=2)
    function = module["main"]
    buffers = _loops(function)[0].body.block.alloc_buffers
    replaced = False

    def read_first(statement):
        nonlocal replaced
        if replaced:
            return statement
        replaced = True
        return _add(buffers[0], buffers[0], buffers[1])

    body = tir.stmt_functor.ir_transform(function.body, None, read_first, ["tir.Evaluate"])
    with pytest.raises(tvm.TVMError, match="read.*defin|uninitialized|undefined|readiness"):
        _plan(tvm.IRModule.from_expr(function.with_body(body)), configs())


def test_invalid_actual_version_budget_is_rejected_before_allocator():
    profile = fixture_profile()
    profile["records"].append(
        {
            "kind": "fill",
            "dtype": "float32",
            "elements": 8192,
            "bytes": 32768,
            "rows": 1,
            "latency": 4,
            "ii": 1,
            "latency_effective_ns": 4,
            "ii_effective_ns": 1,
        }
    )
    seal(profile)
    module = _fills(stages=3, allocation_shape=(8192,))
    function = module["main"]

    def count(statement):
        call = statement.value
        args = list(call.args)
        args[3] = tir.const(8192, "int32")
        return tir.Evaluate(tir.Call(call.dtype, call.op, args))

    body = tir.stmt_functor.ir_transform(function.body, None, count, ["tir.Evaluate"])

    def bad_budget(problem):
        return tvm.runtime.convert(
            {
                "schema_version": 1,
                "status": "SAT",
                "reason": "",
                "start_times": [0, 1, 2],
                "order": [0, 1, 2],
                "ii": 4,
                "buffer_versions": [2, 2, 2],
            }
        )

    with (
        _solver(bad_budget),
        pytest.raises(tvm.TVMError, match="Invalid consumed modulo plan.*Memory group"),
    ):
        _plan(tvm.IRModule.from_expr(function.with_body(body)), configs(profile))


@pytest.mark.parametrize("bound", [2, 3])
@pytest.mark.parametrize("domain,cores,group", [((3, 10), 8, 4), ((1, 3), 8, 2), ((1, 1), 1, 8)])
def test_guarded_persistent_waves_lower_with_per_logical_iteration_guards(bound, domain, cores, group):
    from test_s5_kernels import make_persistent_pipeline_kernel

    function = make_persistent_pipeline_kernel(domain=domain, cores=cores, group_size=group, num_stages=bound)
    breaks = []
    tir.stmt_functor.post_order_visit(
        function.body,
        lambda node: breaks.append(node) if isinstance(node, tir.Call) and getattr(node.op, "name", "") == "tl.loop_break" else None,
    )
    assert not breaks
    with tvm.transform.PassContext(config=configs()):
        result = tilelang.lower(function, target="ascendc", platform="A3")
    assert "break;" not in result.kernel_source


def test_reuse_constraints_cover_every_reader_and_writer_not_just_last_source_access():
    from test_s4_kernels import make_vector_kernel
    from test_s4_pipeline_ir import _before_planning
    from tilelang.transform.ascend_schedule_solver_ffi import _plain
    from tilelang.transform.ascend_z3_scheduler import solve_schedule

    module = _before_planning(make_vector_kernel())
    function = module["main"]

    def bound(loop):
        attributes = dict(loop.annotations)
        if "num_stages" in attributes:
            attributes["num_stages"] = 3
        return tir.For(
            loop.loop_var,
            loop.min,
            loop.extent,
            loop.kind,
            loop.body,
            loop.thread_binding,
            attributes,
        )

    body = tir.stmt_functor.ir_transform(function.body, None, bound, ["tir.For"])
    observed = []

    def inspect(problem):
        numeric = _plain(problem)
        reuse = [tuple(edge) for edge in numeric["data_deps"] if edge[2] < 0]
        observed.extend(reuse)
        # a: load/add; b: load/add; c: add/store. Every physical access
        # must finish before its slot is reused by a future logical iteration.
        assert len(reuse) == 6
        assert {(2, 2), (3, 2)} <= {(u, v) for u, v, _, _ in reuse}
        return tvm.runtime.convert(solve_schedule(numeric))

    with _solver(inspect):
        _plan(tvm.IRModule.from_expr(function.with_body(body)), configs())
    assert observed


def _public_serial_stage(stage):
    first = _fills(stages=0, annotations={"enable_offset": True})["main"]
    second = _fills(stages=1)["main"]
    baseline = tvm.IRModule.from_expr(first.with_body(tir.SeqStmt([first.body, second.body])))

    def mark(statement):
        task = tir.AttrStmt(tvm.runtime.convert({}), "tl.ascend_task", tir.const(1, "int32"), statement)
        return tir.AttrStmt(tir.IntImm("int64", stage), "tl.ascend_stage", tir.const(1, "int32"), task)

    body = tir.stmt_functor.ir_transform(first.body, None, mark, ["tir.Evaluate"])
    public = tvm.IRModule.from_expr(first.with_body(tir.SeqStmt([body, second.body])))
    return public, baseline


def _consume_public_hints(module):
    target = tvm.target.Target("c -keys=ascend -model=ascendc")
    module = tilelang.transform.AscendValidateScheduleHints()(module)
    return tilelang.transform.AscendLowerScheduleHints(target, "A3")(module)


def test_public_nonzero_stage_outside_positive_candidate_cannot_be_silently_erased():
    public, _ = _public_serial_stage(2)
    with tvm.transform.PassContext(config=configs()):
        lowered = _consume_public_hints(public)
        with pytest.raises(tvm.TVMError, match="Nonzero manual Stage is outside a positive"):
            tilelang.transform.PipelinePlanning()(lowered)


def test_public_stage_zero_outside_candidate_retains_serial_equivalence():
    from test_s4_kernels import vector_pass_configs

    public, baseline = _public_serial_stage(0)
    # Serial Stage(0) keeps the S3 contract without requiring an S5 profile.
    with tvm.transform.PassContext(config=vector_pass_configs()):
        marked = tilelang.transform.PipelinePlanning()(_consume_public_hints(public))["main"]
        plain = tilelang.transform.PipelinePlanning()(baseline)["main"]
    assert tvm.ir.structural_equal(marked.without_attr("tl.ascend_constraint_plan"), plain.without_attr("tl.ascend_constraint_plan"))
    source = marked.attrs["tl.ascend_constraint_plan"]["source_graph"]
    assert [int(task["manual_stage"]) for task in source["tasks"][:3]] == [0, 0, 0]


def test_internal_nonzero_stage_outside_candidate_cannot_bypass_the_gate():
    first = _manual([2, 0, 1])["main"]

    def serial(loop):
        annotations = dict(loop.annotations)
        annotations["num_stages"] = 0
        return tir.For(
            loop.loop_var,
            loop.min,
            loop.extent,
            loop.kind,
            loop.body,
            loop.thread_binding,
            annotations,
        )

    body = tir.stmt_functor.ir_transform(first.body, None, serial, ["tir.For"])
    second = _fills(stages=1)["main"]
    module = tvm.IRModule.from_expr(first.with_body(tir.SeqStmt([body, second.body])))
    with pytest.raises(tvm.TVMError, match="Nonzero manual Stage is outside a positive"):
        _plan(module, configs())


def test_guarded_persistent_affine_byte_bounds_use_the_joint_guard():
    from test_s5_kernels import make_persistent_pipeline_kernel

    with tvm.transform.PassContext(config=configs()):
        artifact = tilelang.lower(
            make_persistent_pipeline_kernel((9, 1), 4, 1, num_stages=3),
            target="ascendc",
            platform="A3",
        )
    assert artifact.kernel_source


@pytest.mark.parametrize("corruption", ["drop_guard", "shift_gm"])
def test_guarded_persistent_affine_bounds_still_reject_real_out_of_bounds(corruption):
    from test_s4_pipeline_ir import _before_planning
    from test_s5_kernels import make_persistent_pipeline_kernel

    with tvm.transform.PassContext(config=configs()):
        module = _before_planning(make_persistent_pipeline_kernel((9, 1), 4, 1))
    function = module["main"]
    if corruption == "drop_guard":
        body = tir.stmt_functor.ir_transform(function.body, None, lambda node: node.then_case, ["tir.IfThenElse"])
    else:
        source = next(buffer.data for buffer in function.buffer_map.values() if buffer.name == "A")

        def shift(call):
            if getattr(call.op, "name", "") != "tir.tvm_access_ptr" or not call.args[1].same_as(source):
                return call
            args = list(call.args)
            args[2] = args[2] + 192
            return tir.Call(call.dtype, call.op, args)

        body = tir.stmt_functor.ir_transform(function.body, None, shift, ["tir.Call"])
    with pytest.raises(tvm.TVMError, match="cannot prove access bounds"):
        _plan(tvm.IRModule.from_expr(function.with_body(body)), configs())
