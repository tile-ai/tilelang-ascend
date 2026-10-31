# Copyright (c) Tile-AI Organization.
# Licensed under the MIT License.
"""CPU-only checks for legacy conditional-domain admission to upstream sync.

No kernel is compiled or executed on a device. These fixtures independently
describe physical storage and verify admission before dependency closure can
turn a conditionally executed task into an unconditional completion proof.
"""

from collections import defaultdict

import pytest

import tilelang
from tilelang import tvm

tir = tvm.tir


@pytest.fixture(autouse=True)
def _requires_proof_service() -> None:
    if not tvm.get_global_func("tl.ascend_sync.Z3ProofServiceAvailable")():
        pytest.skip("requires USE_ASCEND_SYNC_Z3=ON and the migrated sync engine")


def _buffers() -> dict[str, tir.Buffer]:
    return {
        name: tir.decl_buffer((64,), "float32", name=name, scope=scope)
        for name, scope in (
            ("a", "shared.ub"),
            ("b", "shared.ub"),
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
    scalars: tuple[tir.Var, ...],
    *,
    b_offset: int = 256,
) -> tir.PrimFunc:
    body = statements[0] if len(statements) == 1 else tir.SeqStmt(statements)
    body = tir.AttrStmt(tir.IntImm("int32", 0), "resource_scope", 1, body)
    for name in ("a", "b"):
        buffer = buffers[name]
        body = tir.Allocate(buffer.data, buffer.dtype, [64], tir.const(True, "bool"), body)
    parameters = [buffers["input"], buffers["output"]]
    function = tir.PrimFunc(
        [buffer.data for buffer in parameters] + list(scalars),
        body,
        buffer_map={buffer.data: buffer for buffer in parameters},
    )
    return (
        function.with_attr("address_map", {buffers["a"].data: 0, buffers["b"].data: b_offset})
        .with_attr("size_map", {buffers[name].data: 256 for name in ("a", "b")})
        .with_attr("tir.noalias", True)
    )


def _run(function: tir.PrimFunc) -> tir.PrimFunc:
    target = tvm.target.Target("c -keys=ascend -model=ascendc")
    with tvm.transform.PassContext(config={"tl.ascend_auto_sync": True, "tl.ascend_auto_sync_vs": True}):
        module = tilelang.transform.AscendInsertSync(target, "A3")(tvm.IRModule({"main": function}))
    return module["main"]


def _execution_trace(function: tir.PrimFunc, values: dict[tir.Var, int]) -> list[tir.Call]:
    """Expand the original control flow; synthetic proxy indices cannot exist."""
    result = []
    analyzer = tvm.arith.Analyzer()

    def integer(expression: tir.PrimExpr, bindings: dict[tir.Var, int]) -> int:
        substitution = {var: tir.const(value, var.dtype) for var, value in bindings.items()}
        constant = analyzer.simplify(tir.stmt_functor.substitute(expression, substitution))
        assert isinstance(constant, tir.IntImm), f"unbound control expression: {constant}"
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
            if isinstance(call.op, tvm.ir.Op) and call.op.name in {
                "tl.ascend_auto_set_flag",
                "tl.ascend_auto_wait_flag",
            }:
                # Freeze a possibly loop-dependent event ID at its issue point.
                call = tir.Call(
                    call.dtype,
                    call.op,
                    [call.args[0], tir.IntImm("int32", integer(call.args[1], bindings))],
                )
            result.append(call)

    visit(function.body, values)
    return result


def _assert_complete_and_balanced(function: tir.PrimFunc, trace: list[tir.Call]) -> int:
    """Check binary event tokens and actual access hazards using pipe completion.

    A set completes prior commands on its source pipe. A wait imports those
    completions into its destination pipe. Issue order alone never discharges
    an access conflict; a same-pipe barrier must establish completion instead.
    """
    tokens = {}
    issued = defaultdict(set)
    completed = defaultdict(set)
    accesses = defaultdict(list)
    addresses = function.attrs["address_map"]
    computations = 0
    for call in trace:
        name = call.op.name
        if name == "tl.ascend_auto_barrier":
            pipe = call.args[0].value.removeprefix("PIPE_")
            completed[pipe].update(issued[pipe])
            continue
        if name in {"tl.ascend_auto_set_flag", "tl.ascend_auto_wait_flag"}:
            event = call.args[0].value
            source, destination = event.split("_")
            event_id = int(call.args[1])
            assert 0 <= event_id < 8
            key = (event, event_id)
            if name == "tl.ascend_auto_set_flag":
                assert key not in tokens, f"set overwrites an unconsumed event: {key}"
                tokens[key] = issued[source] | completed[source]
            else:
                assert key in tokens, f"wait has no producer: {key}"
                completed[destination].update(tokens.pop(key))
            continue
        assert name == "tir.call_extern", f"unmodelled fixture operation: {call}"
        helper = call.args[0].value
        pipe = {
            "AscendC::Duplicate": "V",
            "copy_gm_to_ub": "MTE2",
            "copy_ub_to_gm": "MTE3",
        }[helper]
        current = computations
        computations += 1
        current_accesses = {}
        for argument in call.args[1:]:
            if not isinstance(argument, tir.Call) or argument.op.name != "tir.tvm_access_ptr":
                continue
            data = argument.args[1]
            key = ("ub", int(addresses[data])) if data in addresses else ("gm", data)
            current_accesses[key] = current_accesses.get(key, 0) | int(argument.args[4])
        for key, mask in current_accesses.items():
            for previous, previous_mask in accesses[key]:
                if mask & 2 or previous_mask & 2:
                    assert previous in completed[pipe], (
                        f"{helper} command {current} lacks completion of conflicting command {previous} on {key}"
                    )
            accesses[key].append((current, mask))
        issued[pipe].add(current)
    assert not tokens, f"undrained events at kernel exit: {tokens}"
    return computations


def _assert_no_proxy_indices(function: tir.PrimFunc, expected_for_count: int) -> None:
    variables = []
    loops = []

    def collect(node: object) -> None:
        if isinstance(node, tir.Var):
            variables.append(node)
        elif isinstance(node, tir.For):
            loops.append(node)

    tir.stmt_functor.post_order_visit(function.body, collect)
    assert not any(var.name.startswith("__ascend_if_proxy_") for var in variables)
    assert len(loops) == expected_for_count, "the original If must remain an If"


@pytest.mark.parametrize("physical_alias", [False, True])
@pytest.mark.parametrize("condition_value", [False, True])
def test_optional_producer_has_a_complete_handshake_on_both_paths(physical_alias: bool, condition_value: bool) -> None:
    buffers = _buffers()
    condition = tir.Var("condition", "bool")
    destination = buffers["b"] if physical_alias else buffers["a"]
    statements = [
        _fill(buffers["a"]),
        tir.IfThenElse(condition, _copy("copy_gm_to_ub", buffers["input"], destination), None),
        _copy("copy_ub_to_gm", buffers["a"], buffers["output"]),
    ]
    original = _function(buffers, statements, (condition,), b_offset=0 if physical_alias else 256)
    result = _run(original)
    _assert_no_proxy_indices(result, expected_for_count=0)
    trace = _execution_trace(result, {condition: condition_value})
    # Even without the optional MTE2 command, an intervening handshake must
    # preserve the V fill's completion before the unconditional MTE3 reader.
    assert _assert_complete_and_balanced(result, trace) == 2 + int(condition_value)


@pytest.mark.parametrize("trip_count", [0, 1, 3, 4])
def test_alternating_optional_producer_preserves_completion_and_event_balance(
    trip_count: int,
) -> None:
    buffers = _buffers()
    count = tir.Var("count", "int32")
    index = tir.Var("i", "int32")
    body = tir.SeqStmt(
        [
            tir.IfThenElse(index % 2 == 0, _copy("copy_gm_to_ub", buffers["input"], buffers["a"]), None),
            _copy("copy_ub_to_gm", buffers["a"], buffers["output"]),
        ]
    )
    loop = tir.For(index, 0, count, tir.ForKind.SERIAL, body)
    result = _run(_function(buffers, [loop], (count,)))
    _assert_no_proxy_indices(result, expected_for_count=1)
    trace = _execution_trace(result, {count: trip_count})
    assert _assert_complete_and_balanced(result, trace) == trip_count + (trip_count + 1) // 2


@pytest.mark.parametrize("trip_count", [0, 1, 2, 3, 5])
def test_else_branch_carries_are_ordered_across_different_iterations(trip_count: int) -> None:
    buffers = _buffers()
    count = tir.Var("count", "int32")
    index = tir.Var("i", "int32")
    branch = tir.IfThenElse(
        index % 2 == 0,
        _copy("copy_gm_to_ub", buffers["input"], buffers["a"]),
        _fill(buffers["a"]),
    )
    loop = tir.For(index, 0, count, tir.ForKind.SERIAL, branch)
    result = _run(_function(buffers, [loop], (count,)))
    _assert_no_proxy_indices(result, expected_for_count=1)
    trace = _execution_trace(result, {count: trip_count})
    # The two writes exclude each other within one iteration, but that alone
    # cannot establish completion before the other branch in a later epoch.
    assert _assert_complete_and_balanced(result, trace) == trip_count


@pytest.mark.parametrize("outer,inner", [(False, False), (False, True), (True, False), (True, True)])
def test_nested_else_decisions_preserve_completion_on_every_path(outer: bool, inner: bool) -> None:
    buffers = _buffers()
    first = tir.Var("first", "bool")
    second = tir.Var("second", "bool")
    load = _copy("copy_gm_to_ub", buffers["input"], buffers["a"])
    fill = _fill(buffers["a"])
    branch = tir.IfThenElse(
        first,
        tir.IfThenElse(second, load, fill),
        tir.IfThenElse(second, fill, load),
    )
    original = _function(
        buffers,
        [_fill(buffers["a"]), branch, _copy("copy_ub_to_gm", buffers["a"], buffers["output"])],
        (first, second),
    )
    result = _run(original)
    _assert_no_proxy_indices(result, expected_for_count=0)
    trace = _execution_trace(result, {first: outer, second: inner})
    assert _assert_complete_and_balanced(result, trace) == 3


@pytest.mark.parametrize("then_count,else_count", [(0, 0), (0, 3), (1, 1), (2, 3)])
def test_each_else_arm_keeps_its_own_nested_loop_protocol(then_count: int, else_count: int) -> None:
    buffers = _buffers()
    outer = tir.Var("outer", "int32")
    then_index = tir.Var("then_index", "int32")
    else_index = tir.Var("else_index", "int32")
    store = _copy("copy_ub_to_gm", buffers["a"], buffers["output"])
    then_body = tir.For(
        then_index,
        0,
        then_count,
        tir.ForKind.SERIAL,
        tir.SeqStmt([_copy("copy_gm_to_ub", buffers["input"], buffers["a"]), store]),
    )
    else_body = tir.For(
        else_index,
        0,
        else_count,
        tir.ForKind.SERIAL,
        tir.SeqStmt([_fill(buffers["a"]), store]),
    )
    loop = tir.For(
        outer,
        0,
        3,
        tir.ForKind.SERIAL,
        tir.IfThenElse(outer % 2 == 0, then_body, else_body),
    )
    result = _run(_function(buffers, [loop], ()))
    _assert_no_proxy_indices(result, expected_for_count=3)
    assert _assert_complete_and_balanced(result, _execution_trace(result, {})) == (4 * then_count + 2 * else_count)


def test_many_else_boundaries_fail_closed_when_flag_reuse_is_not_proven() -> None:
    buffers = _buffers()
    condition = tir.Var("condition", "bool")
    statements = []
    for _ in range(10):
        store = _copy("copy_ub_to_gm", buffers["a"], buffers["output"])
        statements.append(
            tir.IfThenElse(
                condition,
                tir.SeqStmt([_copy("copy_gm_to_ub", buffers["input"], buffers["a"]), store]),
                tir.SeqStmt([_fill(buffers["a"]), store]),
            )
        )
    # The current cross-control lifetime proof does not merge these ten
    # independent channels into the eight physical slots. Capacity must fail
    # closed rather than overlap IDs or spill into a hidden hardware pool.
    with pytest.raises(tvm.error.TVMError, match="event capacity exhausted"):
        _run(_function(buffers, statements, (condition,)))


@pytest.mark.parametrize("condition_value", [False, True])
def test_else_protocols_reuse_events_across_complete_outer_iterations(
    condition_value: bool,
) -> None:
    buffers = _buffers()
    condition = tir.Var("condition", "bool")
    statements = []
    for _ in range(3):
        store = _copy("copy_ub_to_gm", buffers["a"], buffers["output"])
        statements.append(
            tir.IfThenElse(
                condition,
                tir.SeqStmt([_copy("copy_gm_to_ub", buffers["input"], buffers["a"]), store]),
                tir.SeqStmt([_fill(buffers["a"]), store]),
            )
        )
    loop = tir.For(tir.Var("i", "int32"), 0, 3, tir.ForKind.SERIAL, tir.SeqStmt(statements))
    result = _run(_function(buffers, [loop], (condition,)))
    _assert_no_proxy_indices(result, expected_for_count=1)
    trace = _execution_trace(result, {condition: condition_value})
    assert _assert_complete_and_balanced(result, trace) == 18
    event = "MTE2_MTE3" if condition_value else "V_MTE3"
    sets = [int(call.args[1]) for call in trace if call.op.name == "tl.ascend_auto_set_flag" and call.args[0].value == event]
    assert len(sets) >= 9
    assert len(set(sets)) < len(sets), "the bounded pool must reuse completed event lifetimes"


def test_tasks_with_the_same_if_guard_remain_supported() -> None:
    buffers = _buffers()
    condition = tir.Var("condition", "bool")
    branch = tir.IfThenElse(
        condition,
        tir.SeqStmt(
            [
                _copy("copy_gm_to_ub", buffers["input"], buffers["a"]),
                _copy("copy_ub_to_gm", buffers["a"], buffers["output"]),
            ]
        ),
        None,
    )
    result = _run(_function(buffers, [branch], (condition,)))
    assert result.attrs.get("tl.ascend_sync_proof") is not None


def test_guard_normalization_uses_an_empty_analysis_context() -> None:
    buffers = _buffers()
    condition = tir.Var("condition", "bool")
    statements = [
        tir.IfThenElse(condition, _copy("copy_gm_to_ub", buffers["input"], buffers["a"]), None),
        tir.IfThenElse(
            tir.And(condition, tir.const(True, "bool")),
            _copy("copy_ub_to_gm", buffers["a"], buffers["output"]),
            None,
        ),
    ]
    result = _run(_function(buffers, statements, (condition,)))
    assert result.attrs.get("tl.ascend_sync_proof") is not None


def test_mixed_guards_on_proven_disjoint_physical_allocations_remain_supported() -> None:
    buffers = _buffers()
    condition = tir.Var("condition", "bool")
    statements = [
        _fill(buffers["a"]),
        tir.IfThenElse(condition, _copy("copy_gm_to_ub", buffers["input"], buffers["b"]), None),
        _copy("copy_ub_to_gm", buffers["a"], buffers["output"]),
    ]
    result = _run(_function(buffers, statements, (condition,)))
    assert result.attrs.get("tl.ascend_sync_proof") is not None


def test_exclusive_branches_without_a_shared_loop_can_reuse_physical_storage() -> None:
    buffers = _buffers()
    condition = tir.Var("condition", "bool")
    branch = tir.IfThenElse(
        condition,
        _copy("copy_gm_to_ub", buffers["input"], buffers["a"]),
        _fill(buffers["a"]),
    )
    result = _run(_function(buffers, [branch], (condition,)))
    assert result.attrs.get("tl.ascend_sync_proof") is not None


def test_loop_branches_on_statically_disjoint_allocations_remain_supported() -> None:
    buffers = _buffers()
    index = tir.Var("i", "int32")
    branch = tir.IfThenElse(index % 2 == 0, _fill(buffers["a"]), _fill(buffers["b"]))
    loop = tir.For(index, 0, 3, tir.ForKind.SERIAL, branch)
    result = _run(_function(buffers, [loop], (), b_offset=256))
    assert result.attrs.get("tl.ascend_sync_proof") is not None
    fills = []

    def collect(node: object) -> None:
        if (
            isinstance(node, tir.Evaluate)
            and isinstance(node.value, tir.Call)
            and isinstance(node.value.op, tvm.ir.Op)
            and node.value.op.name == "tir.call_extern"
            and node.value.args[0].value == "AscendC::Duplicate"
        ):
            fills.append(node.value)

    tir.stmt_functor.post_order_visit(result.body, collect)
    assert len(fills) == 2
