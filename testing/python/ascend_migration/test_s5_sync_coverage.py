"""Version renaming must preserve source values and physical completion."""

from collections import Counter

import pytest

import tilelang
from tilelang import tvm

from testing.python.ascend_migration.test_s2_task_ir import _buffer, _fill, _function
from testing.python.ascend_migration.test_s4_sync_coverage import _barrier, _flag

tir = tvm.tir
TARGET = tvm.target.Target("c -keys=ascend -model=ascendc")


def _copy(src, dst, *, load, count=64, src_offset=0, dst_offset=0):
    element = "half" if src.dtype == "float16" else "float"
    name = "copy_gm_to_ub" if load else "copy_ub_to_gm"
    args = [
        src.access_ptr("r", offset=src_offset),
        dst.access_ptr("w", offset=dst_offset),
        tir.const(count, "int32"),
        tir.const(1, "int32"),
        tir.const(count, "int32"),
    ]
    if load:
        args.append(tir.const(0.0, src.dtype))
    args.append(tir.const(count, "int32"))
    return tir.Evaluate(tir.call_extern("handle", f"{name}<{element},{count}>", *args))


def _task(statement, unit, iteration):
    value = tir.Cast("int64", iteration) if isinstance(iteration, tir.PrimExpr) else tir.const(iteration, "int64")
    body = tir.AttrStmt(tir.const(0, "int64"), "tl.ascend_constraint_iteration", value, statement)
    return tir.AttrStmt(tir.const(unit, "int64"), "tl.ascend_constraint_task_id", 1, body)


def _fixture(
    versions=2,
    trips=3,
    dtype="float32",
    minimum=0,
    *,
    overwrite=False,
    wrong_slot=False,
    read_first=False,
    write_count=64,
    short_allocation=False,
    elide_empty=False,
):
    source = _buffer("input", "global", dtype, shape=(max(trips, 1) * 64,))
    output = _buffer("output", "global", dtype, shape=(max(trips, 1) * 64,))
    logical = _buffer("logical", dtype=dtype, shape=(64,))
    physical_shape = (versions, 64)
    physical = _buffer("physical", dtype=dtype, shape=(64,) if short_allocation else physical_shape)
    i = tir.Var("i", "int32")
    offset = (i - minimum) * 64
    write = _copy(source, logical, load=True, count=write_count, src_offset=offset)
    read = _copy(logical, output, load=False, dst_offset=offset)
    operations = [read, write] if read_first else [write, read]
    loop = tir.For(i, minimum, trips, tir.ForKind.SERIAL, tir.SeqStmt(operations))
    original = _function([source, output, logical], loop, noalias=True)
    graph = tilelang.transform.AnalyzeAscendSchedule(original, "A3")
    marked = [_task(statement, index, i) for index, statement in enumerate(operations)]
    original = _function(
        [source, output, logical],
        tir.For(i, minimum, trips, tir.ForKind.SERIAL, tir.SeqStmt(marked)),
        noalias=True,
    )
    if overwrite:
        order = [(True, index) for index in range(trips)]
        order += [(False, index) for index in range(trips)]
    else:
        order = []
        for clock in range(trips + versions - 1):
            if clock < trips:
                order.append((True, clock))
            consumer = clock - versions + 1
            if 0 <= consumer < trips:
                order.append((False, consumer))
    emitted = []
    for load, index in order:
        slot = index % versions
        if not load and wrong_slot:
            slot = (slot + 1) % versions
        statement = (
            _copy(
                source,
                physical,
                load=True,
                count=write_count,
                src_offset=index * 64,
                dst_offset=slot * 64,
            )
            if load
            else _copy(physical, output, load=False, src_offset=slot * 64, dst_offset=index * 64)
        )
        unit = int(not load) if not read_first else int(load)
        # Synthetic barriers isolate the value/slot proof from event insertion.
        # The verifier itself never inserts a barrier or repairs corrupted IR.
        emitted.extend([_barrier("PIPE_ALL"), _task(statement, unit, minimum + index)])
    body = emitted if emitted else tir.Evaluate(0)
    buffers = [source, output] if elide_empty and not trips else [source, output, physical]
    final = _function(buffers, body, noalias=True)
    bytes_per_element = tvm.DataType(dtype).bits // 8
    stride = 64 * bytes_per_element
    final = final.with_attr("address_map", {} if elide_empty and not trips else {physical.data: 0})
    final = final.with_attr("size_map", {} if elide_empty and not trips else {physical.data: versions * stride})
    remap = {
        "logical_data": logical.data,
        "physical_data": physical.data,
        "loop_id": 0,
        "versions": versions,
        "version_stride_bytes": stride,
        "version_span_bytes": stride,
        "total_bytes": versions * stride,
        "logical_elem_offset": 0,
        "selector_origin": minimum,
        "logical_shape": [64],
        "physical_shape": list(physical_shape),
    }
    final = final.with_attr("tl.ascend_pipeline_buffer_remap", [remap])
    return final.with_attr(
        "tl.ascend_constraint_plan",
        {"schema_version": 2, "source_graph": graph, "source_function": original, "candidates": []},
    )


def _verify(function):
    before = tvm.ir.save_json(function)
    with tvm.transform.PassContext(config={"tl.ascend_pipeline_planner": "constraint"}):
        result = tilelang.transform.AscendVerifyConstraintSync(TARGET, "A3")(tvm.IRModule({"main": function}))["main"]
    assert result.same_as(function)
    assert tvm.ir.save_json(function) == before


@pytest.mark.parametrize("dtype", ["float16", "float32"])
@pytest.mark.parametrize(
    "versions,trips",
    [(1, 0), (1, 1), (1, 4), (2, 0), (2, 1), (2, 2), (2, 5), (3, 0), (3, 1), (3, 2), (3, 5)],
)
def test_real_remap_allows_pipeline_renaming_and_short_trips(dtype, versions, trips):
    _verify(_fixture(versions, trips, dtype))


@pytest.mark.parametrize("versions", [1, 2, 3])
def test_nonzero_loop_origin_uses_logical_iteration_identity(versions):
    _verify(_fixture(versions, versions + 1, minimum=2))


def test_zero_trip_allocations_can_be_eliminated():
    _verify(_fixture(3, 0, elide_empty=True))


@pytest.mark.parametrize("versions", [2, 3])
def test_wrong_version_slot_is_not_hidden_by_same_var_or_equal_byte_length(versions):
    with pytest.raises(tvm.TVMError, match="logical effect remapping"):
        _verify(_fixture(versions, versions + 1, wrong_slot=True))


@pytest.mark.parametrize("versions", [2, 3])
def test_overwritten_value_is_rejected_even_when_all_memory_operations_complete(versions):
    with pytest.raises(tvm.TVMError, match="wrong reaching writer|overwritten slot"):
        _verify(_fixture(versions, versions + 1, overwrite=True))


@pytest.mark.parametrize("versions", [1, 2, 3])
def test_read_before_full_definition_is_not_reinterpreted_as_another_iteration(versions):
    with pytest.raises(tvm.TVMError, match="no source full writer"):
        _verify(_fixture(versions, 2, read_first=True))


def test_partial_writes_do_not_define_a_whole_version():
    with pytest.raises(tvm.TVMError, match="fixed whole logical region"):
        _verify(_fixture(2, 2, write_count=32))


@pytest.mark.parametrize("versions", [2, 3])
def test_claimed_unused_versions_must_still_be_physically_allocated(versions):
    with pytest.raises(tvm.TVMError, match="allocation shape"):
        _verify(_fixture(versions, 1, short_allocation=True))


@pytest.mark.parametrize("field,value", [("version_stride_bytes", 128), ("versions", 4), ("selector_origin", 9)])
def test_inconsistent_remap_geometry_is_rejected(field, value):
    function = _fixture(2, 2)
    remap = dict(function.attrs["tl.ascend_pipeline_buffer_remap"][0].items())
    remap[field] = value
    with pytest.raises(tvm.TVMError, match="physical version geometry|version selector origin"):
        _verify(function.with_attr("tl.ascend_pipeline_buffer_remap", [remap]))


@pytest.mark.parametrize("duplicate", [False, True])
def test_event_tokens_are_not_reset_at_iteration_or_pipeline_boundaries(duplicate):
    function = _fixture(2, 3)
    prefix = [_flag("set", "MTE2_V", 0)]
    if duplicate:
        prefix.append(_flag("set", "MTE2_V", 0))
    body = tir.AttrStmt(
        tir.const(0, "int32"),
        "resource_scope",
        1,
        tir.SeqStmt([*prefix, function.body, _flag("wait", "MTE2_V", 0)]),
    )
    function = function.with_body(body)
    if duplicate:
        with pytest.raises(tvm.TVMError, match="Duplicate set"):
            _verify(function)
    else:
        _verify(function)


def test_zero_trip_does_not_waive_event_drain():
    function = _fixture(3, 0)
    body = tir.AttrStmt(tir.const(0, "int32"), "resource_scope", 1, _flag("set", "MTE2_V", 0))
    with pytest.raises(tvm.TVMError, match="Undrained"):
        _verify(function.with_body(body))


def _narrow_guard(buffer, expression, dtype="int8"):
    return tir.IfThenElse(tir.Cast(dtype, expression) >= 0, _fill(buffer), None)


def _analyze_domains(function):
    before = tvm.ir.save_json(function)
    result = tilelang.transform.AnalyzeAscendSchedule(function, "A3")
    assert tvm.ir.save_json(function) == before
    return result


@pytest.mark.parametrize("dtype", ["int8", "uint8", "int32"])
def test_static_loop_and_launch_domains_prove_lossless_narrowing(dtype):
    buffer = _buffer("domain_buffer")
    iteration = tir.Var("iteration", "int64")
    core = tir.Var("core", "int32")
    body = _narrow_guard(buffer, iteration * 4 + tir.Cast("int64", core), dtype)
    body = tir.For(iteration, 0, 8, tir.ForKind.SERIAL, body)
    axis = tir.IterVar(tvm.ir.Range(0, 4), core, tir.IterVar.ThreadIndex, "blockIdx.x")
    body = tir.AttrStmt(axis, "thread_extent", 4, body)
    assert len(_analyze_domains(_function([buffer], body))["tasks"]) == 1


@pytest.mark.parametrize("source_dtype", ["int64", "uint64", "float32"])
def test_unbounded_or_noninteger_narrowing_remains_unsupported(source_dtype):
    buffer = _buffer("domain_buffer")
    value = tir.Var("unbounded", source_dtype)
    function = _function([buffer], _narrow_guard(buffer, value, "int32"), scalars=[value])
    with pytest.raises(tvm.TVMError, match="uint64|narrowing/lossy"):
        _analyze_domains(function)


@pytest.mark.parametrize("unknown_inner", [False, True])
def test_nested_reused_var_shadows_outer_domain_even_when_inner_is_unknown(unknown_inner):
    buffer = _buffer("domain_buffer")
    iteration = tir.Var("iteration", "int64")
    extent = tir.Var("unknown_extent", "int64")
    inner = tir.For(
        iteration,
        0 if unknown_inner else 1000,
        extent if unknown_inner else 2,
        tir.ForKind.SERIAL,
        _narrow_guard(buffer, iteration),
    )
    body = tir.For(iteration, 0, 4, tir.ForKind.SERIAL, inner)
    function = _function([buffer], body, scalars=[extent] if unknown_inner else [])
    with pytest.raises(tvm.TVMError, match="narrowing/lossy"):
        _analyze_domains(function)


def test_same_name_distinct_var_does_not_inherit_loop_domain():
    buffer = _buffer("domain_buffer")
    bound = tir.Var("iteration", "int64")
    unrelated = tir.Var("iteration", "int64")
    body = tir.For(bound, 0, 4, tir.ForKind.SERIAL, _narrow_guard(buffer, unrelated))
    with pytest.raises(tvm.TVMError, match="narrowing/lossy"):
        _analyze_domains(_function([buffer], body, scalars=[unrelated]))


def test_exited_inner_domain_cannot_prove_an_unbound_sibling_cast():
    buffer = _buffer("domain_buffer")
    iteration = tir.Var("iteration", "int64")
    body = tir.SeqStmt(
        [
            tir.For(iteration, 0, 4, tir.ForKind.SERIAL, _fill(buffer)),
            _narrow_guard(buffer, iteration),
        ]
    )
    with pytest.raises(tvm.TVMError, match="narrowing/lossy"):
        _analyze_domains(_function([buffer], body, scalars=[iteration]))


def test_nested_reused_var_restores_the_outer_domain_on_exit():
    buffer = _buffer("domain_buffer")
    iteration = tir.Var("iteration", "int64")
    inner = tir.For(iteration, 1000, 2, tir.ForKind.SERIAL, _fill(buffer))
    body = tir.For(
        iteration,
        0,
        4,
        tir.ForKind.SERIAL,
        tir.SeqStmt([inner, _narrow_guard(buffer, iteration)]),
    )
    assert len(_analyze_domains(_function([buffer], body))["tasks"]) == 2


def test_real_injected_prologue_guards_exclude_negative_logical_iterations():
    from test_s4_pipeline_ir import _plan
    from test_s5_versions_ir import _execute, _manual, _value, configs

    module = _manual([2, 0, 1])

    def one_trip(loop):
        return tir.For(
            loop.loop_var,
            loop.min,
            1,
            loop.kind,
            loop.body,
            loop.thread_binding,
            loop.annotations,
        )

    function = module["main"]
    body = tir.stmt_functor.ir_transform(function.body, None, one_trip, ["tir.For"])
    planned = _plan(tvm.IRModule.from_expr(function.with_body(body)), configs())
    function = tilelang.transform.InjectSoftwarePipeline()(planned)["main"]
    units = [int(unit) for unit in planned["main"].attrs["tl.ascend_constraint_plan"]["candidates"][0]["unit_ids"]]

    def observe_task(attribute):
        if attribute.attr_key != "tl.ascend_constraint_task_id":
            return attribute
        iterations = []
        tir.stmt_functor.post_order_visit(
            attribute.body,
            lambda node: (
                iterations.append(node.value)
                if isinstance(node, tir.AttrStmt) and node.attr_key == "tl.ascend_constraint_iteration"
                else None
            ),
        )
        assert len(iterations) == 1
        return tir.Evaluate(tir.call_extern("handle", "observe_unit", attribute.node, iterations[0]))

    # Replace only terminal hardware tasks in a test copy; retain the injector's
    # actual clocks, predicates and substituted logical iteration expressions.
    probe = tir.stmt_functor.ir_transform(function.body, None, observe_task, ["tir.AttrStmt"])

    def trace(statement):
        result = []

        def observe(terminal, environment):
            call = terminal.value
            assert call.args[0].value == "observe_unit"
            result.append((_value(call.args[1], environment), _value(call.args[2], environment)))

        _execute(statement, {}, observe)
        return result

    assert Counter(trace(probe)) == Counter((unit, 0) for unit in units)

    def ignore_predicate(statement):
        if isinstance(statement, tir.IfThenElse):
            assert statement.else_case is None
            return statement.then_case
        return tir.BlockRealize(statement.iter_values, True, statement.block)

    unchecked_probe = tir.stmt_functor.ir_transform(probe, None, ignore_predicate, ["tir.IfThenElse", "tir.BlockRealize"])
    assert any(iteration < 0 for _, iteration in trace(unchecked_probe))
