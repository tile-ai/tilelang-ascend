# Copyright (c) Tile-AI Organization.
# Licensed under the MIT License.
"""Independent CPU semantics tests for the migrated Ascend sync engine.

The expected ordering comes from each fixture's accesses, never the old pass.
No kernel is compiled or executed on a device in this module.
"""

from collections import defaultdict
from collections.abc import Mapping

import pytest

import tilelang
from tilelang import tvm

tir = tvm.tir
_SYNC_OPS = {
    "tl.ascend_auto_barrier",
    "tl.ascend_auto_set_flag",
    "tl.ascend_auto_wait_flag",
}


@pytest.fixture(autouse=True)
def _requires_proof_service() -> None:
    available = tvm.get_global_func("tl.ascend_sync.Z3ProofServiceAvailable")
    if not available():
        pytest.skip("requires USE_ASCEND_SYNC_Z3=ON and the migrated sync engine")


def _buffers() -> dict[str, tir.Buffer]:
    return {
        name: tir.decl_buffer((64,), "float32", name=name, scope=scope)
        for name, scope in (
            ("a", "shared.ub"),
            ("b", "shared.ub"),
            ("tmp", "shared.ub"),
            ("input", "global"),
            ("output", "global"),
        )
    }


def _copy(name: str, source: tir.Buffer, destination: tir.Buffer) -> tir.Evaluate:
    return tir.Evaluate(tir.call_extern("handle", name, source.access_ptr("r"), destination.access_ptr("w")))


def _fill(destination: tir.Buffer) -> tir.Evaluate:
    return tir.Evaluate(tir.call_extern("handle", "AscendC::Duplicate", destination.access_ptr("w")))


def _function(
    buffers: dict[str, tir.Buffer],
    statements: list[tir.Stmt],
    *,
    b_offset: int = 0,
    scalars: tuple[tir.Var, ...] = (),
    resource: int = 1,
    noalias: bool = True,
) -> tir.PrimFunc:
    body = statements[0] if len(statements) == 1 else tir.SeqStmt(statements)
    body = tir.AttrStmt(tir.IntImm("int32", 0), "resource_scope", resource, body)
    addresses = {"a": 0, "b": b_offset, "tmp": 4096}
    for name in addresses:
        buffer = buffers[name]
        body = tir.Allocate(buffer.data, buffer.dtype, [64], tir.const(True, "bool"), body)
    parameters = [buffers["input"], buffers["output"]]
    function = tir.PrimFunc(
        [buffer.data for buffer in parameters] + list(scalars),
        body,
        buffer_map={buffer.data: buffer for buffer in parameters},
    )
    function = function.with_attr("address_map", {buffers[name].data: value for name, value in addresses.items()})
    function = function.with_attr("size_map", {buffers[name].data: 256 for name in addresses})
    return function.with_attr("tir.noalias", noalias)


def _run(function: tir.PrimFunc, *, ordinary: bool = True, vs: bool = True) -> tir.PrimFunc:
    target = tvm.target.Target("c -keys=ascend -model=ascendc")
    with tvm.transform.PassContext(config={"tl.ascend_auto_sync": ordinary, "tl.ascend_auto_sync_vs": vs}):
        module = tilelang.transform.AscendInsertSync(target, "A3")(tvm.IRModule({"main": function}))
    return module["main"]


def _calls(function: tir.PrimFunc) -> list[tir.Call]:
    calls = []

    def collect(node: object) -> None:
        if isinstance(node, tir.Evaluate) and isinstance(node.value, tir.Call):
            calls.append(node.value)

    tir.stmt_functor.post_order_visit(function.body, collect)
    return calls


def _op_name(call: tir.Call) -> str:
    return call.op.name if isinstance(call.op, tvm.ir.Op) else str(call.op)


def _assert_var_map_unchanged(actual: Mapping[tir.Var, tir.PrimExpr], expected: Mapping[tir.Var, tir.PrimExpr]) -> None:
    # Legacy SEqual on a bare Var-keyed Map requires a surrounding binder and
    # can return false even for map compared with itself. Check identities and
    # values directly, without allowing free-variable renaming.
    assert len(actual) == len(expected)
    for key, value in expected.items():
        matching = [candidate for candidate in actual if candidate.same_as(key)]
        assert len(matching) == 1, f"logical pointer identity changed: {key}"
        assert tvm.ir.structural_equal(actual[matching[0]], value)


def _assert_handoff(function: tir.PrimFunc, event: str, producer: tir.Stmt, consumer: tir.Stmt) -> None:
    calls = _calls(function)
    producer_index = next(index for index, call in enumerate(calls) if tvm.ir.structural_equal(call, producer.value))
    consumer_index = next(index for index, call in enumerate(calls) if tvm.ir.structural_equal(call, consumer.value))
    for index in range(producer_index + 1, consumer_index):
        call = calls[index]
        if _op_name(call) != "tl.ascend_auto_set_flag" or call.args[0].value != event:
            continue
        for wait in calls[index + 1 : consumer_index]:
            if (
                _op_name(wait) == "tl.ascend_auto_wait_flag"
                and wait.args[0].value == event
                and tvm.ir.structural_equal(call.args[1], wait.args[1])
            ):
                return
    pytest.fail(f"missing paired {event} completion between producer and consumer")


@pytest.mark.parametrize("constant_index", [False, True])
def test_unsigned_wrapping_index_keeps_scalar_vector_dependency(constant_index: bool) -> None:
    buffers = _buffers()
    value = tir.Var("unsigned_index", "uint32")
    index = tir.IntImm("uint32", 0) if constant_index else value + tir.IntImm("uint32", 1)
    store = tir.BufferStore(buffers["a"], tir.FloatImm("float32", 1), [index])
    fill = _fill(buffers["a"])
    branch = tir.IfThenElse(value == tir.IntImm("uint32", (1 << 32) - 1), tir.SeqStmt([store, fill]), None)
    result = _run(_function(buffers, [branch], scalars=(value,)))
    events = [(call.args[0].value, int(call.args[1])) for call in _calls(result) if _op_name(call) == "tl.ascend_auto_set_flag"]
    assert any(event == "S_V" for event, _ in events)
    for event, event_id in events:
        assert any(
            _op_name(call) == "tl.ascend_auto_wait_flag" and call.args[0].value == event and int(call.args[1]) == event_id
            for call in _calls(result)
        )


@pytest.mark.parametrize("hazard", ["RAW", "WAR", "WAW"])
def test_shared_physical_address_is_ordered_without_merging_logical_vars(hazard: str) -> None:
    buffers = _buffers()
    if hazard == "RAW":
        first = _copy("copy_gm_to_ub", buffers["input"], buffers["a"])
        second = _copy("copy_ub_to_gm", buffers["b"], buffers["output"])
        event = "MTE2_MTE3"
    elif hazard == "WAR":
        first = _copy("copy_ub_to_gm", buffers["a"], buffers["output"])
        second = _fill(buffers["b"])
        event = "MTE3_V"
    else:
        first = _copy("copy_gm_to_ub", buffers["input"], buffers["a"])
        second = _fill(buffers["b"])
        event = "MTE2_V"
    original = _function(buffers, [first, second])
    result = _run(original)
    _assert_handoff(result, event, first, second)
    computations = [call for call in _calls(result) if _op_name(call) not in _SYNC_OPS]
    assert len(computations) == 2
    assert tvm.ir.structural_equal(computations[0], first.value)
    assert tvm.ir.structural_equal(computations[1], second.value)
    _assert_var_map_unchanged(result.attrs["address_map"], original.attrs["address_map"])
    _assert_var_map_unchanged(result.attrs["size_map"], original.attrs["size_map"])


def test_disjoint_final_allocations_do_not_create_a_false_handoff() -> None:
    buffers = _buffers()
    first = _copy("copy_ub_to_gm", buffers["a"], buffers["output"])
    second = _fill(buffers["b"])
    result = _run(_function(buffers, [first, second], b_offset=256))
    assert not [call for call in _calls(result) if _op_name(call) in _SYNC_OPS]


def test_dma_footprint_uses_padded_allocation_instead_of_pointer_extent() -> None:
    buffers = _buffers()
    first = _copy("copy_ub_to_gm", buffers["a"], buffers["output"])
    second = _fill(buffers["b"])
    original = _function(buffers, [first, second], b_offset=384)
    original = original.with_attr("size_map", {buffers["a"].data: 512, buffers["b"].data: 256, buffers["tmp"].data: 256})
    _assert_handoff(_run(original), "MTE3_V", first, second)


@pytest.mark.parametrize("noalias", [False, True])
def test_gm_parameter_identity_needs_an_explicit_noalias_contract(noalias: bool) -> None:
    buffers = _buffers()
    first = _copy("copy_gm_to_ub", buffers["input"], buffers["a"])
    second = _copy("copy_ub_to_gm", buffers["b"], buffers["output"])
    result = _run(_function(buffers, [first, second], b_offset=256, noalias=noalias))
    if noalias:
        assert not [call for call in _calls(result) if _op_name(call) in _SYNC_OPS]
    else:
        _assert_handoff(result, "MTE2_MTE3", first, second)


def test_same_vector_pipe_dependency_is_covered_by_unified_engine() -> None:
    buffers = _buffers()
    first = _fill(buffers["a"])
    second = _copy("copy_ub_to_ub", buffers["a"], buffers["tmp"])
    result = _run(_function(buffers, [first, second], b_offset=256))
    calls = _calls(result)
    first_index = next(i for i, call in enumerate(calls) if tvm.ir.structural_equal(call, first.value))
    last_index = next(i for i, call in enumerate(calls) if tvm.ir.structural_equal(call, second.value))
    assert any(
        _op_name(call) == "tl.ascend_auto_barrier" and call.args[0].value == "PIPE_V" for call in calls[first_index + 1 : last_index]
    )


def test_cube_scope_preserves_copy_route_and_completion() -> None:
    buffers = _buffers()
    buffers["a"] = tir.decl_buffer((64,), "float32", name="a_l1", scope="shared.l1")
    buffers["b"] = tir.decl_buffer((64,), "float32", name="b_l0a", scope="wmma.matrix_a")
    first = _copy("copy_gm_to_l1", buffers["input"], buffers["a"])
    second = _copy("copy_l1_to_l0a", buffers["a"], buffers["b"])
    result = _run(_function(buffers, [first, second], resource=0))
    _assert_handoff(result, "MTE2_MTE1", first, second)


def test_explicit_scope_conflicting_with_operation_owner_is_rejected() -> None:
    buffers = _buffers()
    vector_copy = _copy("copy_gm_to_ub", buffers["input"], buffers["a"])
    with pytest.raises(tvm.error.TVMError, match="(?i)(resource_scope|resource|scope|owner)"):
        _run(_function(buffers, [vector_copy], resource=0))


def _executed_events(
    function: tir.PrimFunc,
    scalar_values: dict[tir.Var, int],
    computations: list[tir.Call] | None = None,
) -> list[tuple]:
    """Expand only this fixture's pure control flow, without executing its ops."""
    events = []
    analyzer = tvm.arith.Analyzer()

    def integer(expression: tir.PrimExpr, bindings: dict[tir.Var, int]) -> int:
        values = {var: tir.const(value, var.dtype) for var, value in bindings.items()}
        constant = analyzer.simplify(tir.stmt_functor.substitute(expression, values))
        assert isinstance(constant, tir.IntImm), f"fixture control did not reduce: {constant}"
        return int(constant.value)

    def visit(statement: tir.Stmt, bindings: dict[tir.Var, int]) -> None:
        if isinstance(statement, tir.SeqStmt):
            for child in statement.seq:
                visit(child, bindings)
        elif isinstance(statement, tir.For):
            minimum = integer(statement.min, bindings)
            for index in range(minimum, minimum + integer(statement.extent, bindings)):
                visit(statement.body, {**bindings, statement.loop_var: index})
        elif isinstance(statement, tir.IfThenElse):
            branch = statement.then_case if integer(statement.condition, bindings) else statement.else_case
            if branch is not None:
                visit(branch, bindings)
        elif isinstance(statement, tir.LetStmt):
            visit(statement.body, {**bindings, statement.var: integer(statement.value, bindings)})
        elif isinstance(statement, (tir.Allocate, tir.AttrStmt, tir.DeclBuffer)):
            visit(statement.body, bindings)
        elif isinstance(statement, tir.Evaluate) and isinstance(statement.value, tir.Call):
            call = statement.value
            if _op_name(call) in {"tl.ascend_auto_set_flag", "tl.ascend_auto_wait_flag"}:
                events.append((_op_name(call), str(call.args[0].value), integer(call.args[1], bindings)))
            elif _op_name(call) not in _SYNC_OPS and computations is not None:
                computations.append(call)

    visit(function.body, scalar_values)
    return events


@pytest.mark.parametrize("trip_count", [0, 1, 3])
def test_dynamic_loop_events_are_balanced_and_body_runs_the_requested_count(
    trip_count: int,
) -> None:
    buffers = _buffers()
    count = tir.Var("count", "int32")
    write = _copy("copy_gm_to_ub", buffers["input"], buffers["a"])
    read = _copy("copy_ub_to_gm", buffers["a"], buffers["output"])
    loop = tir.For(tir.Var("i", "int32"), 0, count, tir.ForKind.SERIAL, tir.SeqStmt([write, read]))
    result = _run(_function(buffers, [loop], b_offset=256, scalars=(count,)))
    computations = []
    events = _executed_events(result, {count: trip_count}, computations)
    assert len(computations) == 2 * trip_count
    assert sum(tvm.ir.structural_equal(call, write.value) for call in computations) == trip_count
    assert sum(tvm.ir.structural_equal(call, read.value) for call in computations) == trip_count
    if trip_count > 0:
        assert events
    # A zero-trip loop may still have an outer seed/drain pair. It is safe only
    # when each wait has a producer and all event credits are drained below.
    credits = defaultdict(int)
    for operation, event, event_id in events:
        key = (event, event_id)
        if operation == "tl.ascend_auto_set_flag":
            credits[key] += 1
        else:
            assert credits[key] > 0, f"wait has no producer: {key}"
            credits[key] -= 1
    assert not any(credits.values()), "every set must be drained before kernel exit"


def test_optional_branch_does_not_emit_events_on_the_inactive_path() -> None:
    buffers = _buffers()
    condition = tir.Var("condition", "bool")
    write = _copy("copy_gm_to_ub", buffers["input"], buffers["a"])
    read = _copy("copy_ub_to_gm", buffers["a"], buffers["output"])
    branch = tir.IfThenElse(condition, tir.SeqStmt([write, read]), tir.Evaluate(0))
    result = _run(_function(buffers, [branch], b_offset=256, scalars=(condition,)))
    assert not _executed_events(result, {condition: 0})
    assert _executed_events(result, {condition: 1})


@pytest.mark.parametrize("missing", ["address_map", "size_map"])
def test_required_physical_maps_are_not_guessed(missing: str) -> None:
    buffers = _buffers()
    original = _function(buffers, [_fill(buffers["a"])]).without_attr(missing)
    with pytest.raises(tvm.error.TVMError, match="(?i)(address_map|size_map|physical)"):
        _run(original)


def test_unknown_side_effect_is_rejected() -> None:
    buffers = _buffers()
    unknown = tir.Evaluate(tir.call_extern("handle", "unknown_dma", buffers["a"].access_ptr("rw")))
    with pytest.raises(tvm.error.TVMError, match="(?i)(unknown|unsupported|effect|contract)"):
        _run(_function(buffers, [unknown]))


def test_materialized_versions_without_epoch_provenance_are_rejected() -> None:
    buffers = _buffers()
    original = _function(buffers, [_fill(buffers["a"])]).with_attr("buffer_versions", {buffers["a"].data: 2})
    with pytest.raises(tvm.error.TVMError, match="(?i)(provenance|version|pipeline|epoch)"):
        _run(original)


def test_repeated_pass_is_idempotent_but_changed_address_facts_are_rejected() -> None:
    buffers = _buffers()
    original = _function(
        buffers,
        [_copy("copy_ub_to_gm", buffers["a"], buffers["output"]), _fill(buffers["b"])],
    )
    first = _run(original)
    second = _run(first)
    assert tvm.ir.structural_equal(first, second)
    changed = first.with_attr("address_map", {buffers["a"].data: 0, buffers["b"].data: 256, buffers["tmp"].data: 4096})
    with pytest.raises(tvm.error.TVMError, match="(?i)(changed|pre-sync|synchronization)"):
        _run(changed)


def test_all_sync_options_off_preserves_input_without_proof_marker() -> None:
    buffers = _buffers()
    original = _function(buffers, [_fill(buffers["a"])])
    result = _run(original, ordinary=False, vs=False)
    assert tvm.ir.structural_equal(original, result)
    assert result.attrs.get("tl.ascend_sync_proof") is None


def _manual_flag(kind: str, event_id: int = 0) -> tir.Evaluate:
    return tir.Evaluate(
        tir.Call(
            "handle",
            tvm.ir.Op.get(f"tl.ascend_{kind}_flag"),
            [tir.StringImm("MTE2"), tir.StringImm("V"), tir.IntImm("int32", event_id)],
        )
    )


def test_forward_manual_flag_pair_reserves_its_event_id() -> None:
    buffers = _buffers()
    load = _copy("copy_gm_to_ub", buffers["input"], buffers["a"])
    fill = _fill(buffers["a"])
    original = _function(buffers, [_manual_flag("set"), _manual_flag("wait"), load, fill])
    result = _run(original)
    _assert_handoff(result, "MTE2_V", load, fill)
    generated = [call for call in _calls(result) if call.op.name == "tl.ascend_auto_set_flag" and call.args[0].value == "MTE2_V"]
    assert generated
    assert all(int(call.args[1]) != 0 for call in generated)


@pytest.mark.parametrize("protocol", ["backward", "unmatched", "cross_loop"])
def test_unproven_manual_schedule_cannot_create_an_automatic_wait_cycle(protocol: str) -> None:
    buffers = _buffers()
    if protocol == "backward":
        statements = [
            _manual_flag("wait"),
            _copy("copy_ub_to_ub", buffers["a"], buffers["b"]),
            _copy("copy_gm_to_ub", buffers["input"], buffers["a"]),
            _manual_flag("set"),
        ]
    elif protocol == "unmatched":
        statements = [_manual_flag("set"), _fill(buffers["a"])]
    else:
        loop = tir.For(tir.Var("i", "int32"), 0, 2, tir.ForKind.SERIAL, _manual_flag("wait"))
        statements = [_manual_flag("set"), loop]
    with pytest.raises(tvm.error.TVMError, match="(?i)(manual|producer|balanced)"):
        _run(_function(buffers, statements))
