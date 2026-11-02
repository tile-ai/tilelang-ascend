# Copyright (c) Tile-AI Organization.
# Licensed under the MIT License.
"""Prove local manual protocols and independently check reconstructed dependencies."""

import pytest

from tilelang import tvm

from test_ascend_upstream_sync_insert import _buffers, _calls, _copy, _function, _run
from test_ascend_upstream_sync_pipeline import _execute_protocol

tir = tvm.tir
_MANUAL = {"tl.ascend_set_flag", "tl.ascend_wait_flag"}
_AUTO = {"tl.ascend_auto_set_flag", "tl.ascend_auto_wait_flag"}


@pytest.fixture(autouse=True)
def _requires_proof_service():
    if not tvm.get_global_func("tl.ascend_sync.Z3ProofServiceAvailable")():
        pytest.skip("requires USE_ASCEND_SYNC_Z3=ON")


def _flag(kind, event_id=0, source="MTE3", destination="MTE2"):
    if isinstance(event_id, int):
        event_id = tir.IntImm("int32", event_id)
    return tir.Evaluate(
        tir.Call(
            "handle",
            tvm.ir.Op.get(f"tl.ascend_{kind}_flag"),
            [tir.StringImm(source), tir.StringImm(destination), event_id],
        ),
    )


def _payload(buffers):
    return [
        _copy("copy_gm_to_ub", buffers["input"], buffers["a"]),
        tir.Evaluate(
            tir.Call(
                "handle",
                tvm.ir.Op.get("tl.ascend_abs"),
                [buffers["b"].access_ptr("w"), buffers["a"].access_ptr("r"), 64],
            ),
        ),
        _copy("copy_ub_to_gm", buffers["b"], buffers["output"]),
    ]


def _protocol(trip_count, *, slots=2):
    buffers = _buffers()
    iteration = tir.Var("iteration", "int32")
    event_id = iteration % slots if slots > 1 else 0
    body = tir.SeqStmt([_flag("wait", event_id), *_payload(buffers), _flag("set", event_id)])
    loop = tir.For(iteration, 0, trip_count, tir.ForKind.SERIAL, body)
    statements = [
        *[_flag("set", slot) for slot in range(slots)],
        loop,
        *[_flag("wait", slot) for slot in range(slots)],
    ]
    scalars = (trip_count,) if isinstance(trip_count, tir.Var) else ()
    return _function(buffers, statements, b_offset=256, scalars=scalars)


def _assert_automatic_only(result):
    calls = _calls(result)
    assert not any(getattr(call.op, "name", "") in _MANUAL for call in calls)
    assert any(getattr(call.op, "name", "") in _AUTO for call in calls)


@pytest.mark.parametrize("trip_count", [0, 1, 2, 3, 5, 9])
def test_static_modulo_protocol_reconstructs_real_memory_dependencies(trip_count):
    original = _protocol(trip_count)
    result = _run(original)
    if trip_count:
        _assert_automatic_only(result)
    else:
        assert not any(getattr(call.op, "name", "") in _MANUAL for call in _calls(result))
    assert len([node for node in _statements(result.body) if isinstance(node, tir.For)]) == 1
    for _invocation in range(3):
        _events, computations = _execute_protocol(result, 0)
        assert computations == (
            {
                name: trip_count
                for name in (
                    "copy_gm_to_ub",
                    "tl.ascend_abs",
                    "copy_ub_to_gm",
                )
            }
            if trip_count
            else {}
        )


def _statements(body):
    nodes = []
    tir.stmt_functor.post_order_visit(body, nodes.append)
    return nodes


def test_dynamic_loop_uses_token_invariant_and_preserves_zero_trip_path():
    count = tir.Var("count", "int32")
    result = _run(_protocol(count, slots=1))
    _assert_automatic_only(result)
    for trip_count in (0, 1, 2, 5):
        specialized = result.with_body(
            tir.stmt_functor.substitute(result.body, {count: tir.IntImm("int32", trip_count)}),
        )
        for _invocation in range(3):
            _events, computations = _execute_protocol(specialized, 0)
            assert computations["copy_gm_to_ub"] == trip_count
            assert computations["tl.ascend_abs"] == trip_count
            assert computations["copy_ub_to_gm"] == trip_count


def test_byte_completion_oracle_rejects_missing_reconstructed_sync():
    result = _run(_protocol(3))

    def remove(statement):
        if (
            isinstance(statement, tir.Evaluate)
            and isinstance(statement.value, tir.Call)
            and getattr(statement.value.op, "name", "") in _AUTO
        ):
            return tir.Evaluate(0)
        return None

    corrupted = result.with_body(
        tir.stmt_functor.ir_transform(result.body, remove, None, ["tir.Evaluate"]),
    )
    with pytest.raises(AssertionError, match="memory dependency missing"):
        _execute_protocol(corrupted, 0)


@pytest.mark.parametrize(
    "bad",
    [
        "future_producer",
        "duplicate_set",
        "missing_drain",
        "unknown_id",
        "unknown_branch",
        "different_dynamic_exit",
    ],
)
def test_unproven_protocols_remain_rejected(bad):
    buffers = _buffers()
    count = tir.Var("count", "int32")
    iteration = tir.Var("iteration", "int32")
    if bad == "future_producer":
        body = tir.SeqStmt([_flag("wait"), *_payload(buffers), _flag("set")])
        statements = [tir.For(iteration, 0, 2, tir.ForKind.SERIAL, body)]
    elif bad == "duplicate_set":
        body = tir.SeqStmt([_flag("set"), _flag("wait")])
        statements = [
            _flag("set"),
            tir.For(iteration, 0, 2, tir.ForKind.SERIAL, body),
            _flag("wait"),
        ]
    elif bad == "missing_drain":
        body = tir.SeqStmt([_flag("wait"), *_payload(buffers), _flag("set")])
        statements = [_flag("set"), tir.For(iteration, 0, 2, tir.ForKind.SERIAL, body)]
    elif bad == "unknown_id":
        statements = [
            tir.For(
                iteration,
                0,
                count,
                tir.ForKind.SERIAL,
                tir.SeqStmt([_flag("set", iteration % 2), _flag("wait", iteration % 2)]),
            )
        ]
    elif bad == "unknown_branch":
        statements = [tir.IfThenElse(count > 0, _flag("set"), None), _flag("wait")]
    else:
        statements = [_flag("set"), tir.For(iteration, 0, count, tir.ForKind.SERIAL, _flag("wait"))]
    with pytest.raises(tvm.error.TVMError, match="(?i)manual event normalization"):
        _run(_function(buffers, statements, b_offset=256, scalars=(count,)))


def test_unknown_effect_is_not_admitted_by_a_valid_manual_protocol():
    buffers = _buffers()
    iteration = tir.Var("iteration", "int32")
    unknown = tir.Evaluate(
        tir.call_extern("handle", "unmodelled_memory_writer", buffers["a"].access_ptr("w")),
    )
    body = tir.SeqStmt([_flag("wait"), unknown, _flag("set")])
    statements = [_flag("set"), tir.For(iteration, 0, 2, tir.ForKind.SERIAL, body), _flag("wait")]
    with pytest.raises(tvm.error.TVMError, match="(?i)(unknown|configured|effects)"):
        _run(_function(buffers, statements))


def test_manual_protocol_is_not_removed_with_ordinary_sync_disabled():
    with pytest.raises(tvm.error.TVMError, match="(?i)(manual|balanced)"):
        _run(_protocol(3), ordinary=False)


@pytest.mark.parametrize("then_state", ["same", "different"])
def test_unknown_if_requires_equal_token_states_on_all_paths(then_state):
    buffers = _buffers()
    condition = tir.Var("condition", "bool")
    then_case = tir.SeqStmt([_flag("wait"), *_payload(buffers), _flag("set")])
    else_case = (
        tir.SeqStmt(
            [_flag("wait"), *_payload(buffers), _flag("set")],
        )
        if then_state == "same"
        else _flag("wait")
    )
    statements = [_flag("set"), tir.IfThenElse(condition, then_case, else_case), _flag("wait")]
    original = _function(buffers, statements, b_offset=256, scalars=(condition,))
    if then_state == "different":
        with pytest.raises(tvm.error.TVMError, match="(?i)(identical token|before its producer)"):
            _run(original)
        return
    result = _run(original)
    _assert_automatic_only(result)
    for decision in (False, True):
        specialized = result.with_body(
            tir.stmt_functor.substitute(result.body, {condition: tir.const(decision)}),
        )
        _execute_protocol(specialized, 0)


def test_repeated_pure_guard_correlates_initialization_and_consumption():
    buffers = _buffers()
    count = tir.Var("count", "int32")
    load, absolute, store = _payload(buffers)
    statements = [
        tir.IfThenElse(
            count > 0,
            tir.SeqStmt([load, _flag("set", source="MTE2", destination="V")]),
            None,
        ),
        tir.IfThenElse(
            count > 0,
            tir.SeqStmt([_flag("wait", source="MTE2", destination="V"), absolute, store]),
            None,
        ),
    ]
    result = _run(_function(buffers, statements, b_offset=256, scalars=(count,)))
    _assert_automatic_only(result)
    for value in (-1, 0, 1, 7):
        specialized = result.with_body(
            tir.stmt_functor.substitute(result.body, {count: tir.IntImm("int32", value)}),
        )
        _events, computations = _execute_protocol(specialized, 0)
        assert computations["copy_gm_to_ub"] == int(value > 0)


def test_repeated_mutable_guard_does_not_prove_a_stable_event_condition():
    buffers = _buffers()
    condition = tir.BufferLoad(buffers["input"], [0]) > tir.FloatImm("float32", 0)
    statements = [
        tir.IfThenElse(condition, _flag("set"), None),
        tir.IfThenElse(condition, _flag("wait"), None),
    ]
    with pytest.raises(tvm.error.TVMError, match="(?i)manual event normalization"):
        _run(_function(buffers, statements))


def _local_protocol_statements(buffers, *, minimum=0, extent=2, event_id=0):
    iteration = tir.Var("manual_protocol_iteration", "int32")
    body = tir.SeqStmt([_flag("wait", event_id), *_payload(buffers), _flag("set", event_id)])
    return [
        _flag("set"),
        tir.For(iteration, minimum, extent, tir.ForKind.SERIAL, body),
        _flag("wait"),
    ]


def test_normalized_manual_protocol_cannot_mix_cube_and_vector_memory_effects():
    buffers = _buffers()
    l1 = tir.decl_buffer((64,), "float32", name="manual_cube_l1", scope="shared.l1")
    cube_copy = tir.AttrStmt(
        tir.IntImm("int32", 0),
        "resource_scope",
        0,
        _copy("copy_gm_to_l1", buffers["input"], l1),
    )
    original = _function(
        buffers,
        [cube_copy, *_local_protocol_statements(buffers)],
        b_offset=256,
    )
    original = original.with_body(
        tir.Allocate(l1.data, l1.dtype, [64], tir.const(True), original.body),
    )
    original = original.with_attr(
        "address_map",
        {**dict(original.attrs["address_map"]), l1.data: 0},
    )
    original = original.with_attr("size_map", {**dict(original.attrs["size_map"]), l1.data: 256})
    with pytest.raises(tvm.error.TVMError, match="one execution resource"):
        _run(original)


def test_normalized_manual_event_owner_must_match_the_real_memory_owner():
    buffers = _buffers()
    iteration = tir.Var("manual_owner_iteration", "int32")

    def cube_flag(kind):
        return tir.AttrStmt(
            tir.IntImm("int32", 0),
            "resource_scope",
            0,
            _flag(kind, source="M", destination="MTE1"),
        )

    body = tir.SeqStmt([cube_flag("wait"), *_payload(buffers), cube_flag("set")])
    original = _function(
        buffers,
        [cube_flag("set"), tir.For(iteration, 0, 2, tir.ForKind.SERIAL, body), cube_flag("wait")],
        b_offset=256,
    )
    with pytest.raises(tvm.error.TVMError, match="(?i)manual event pipes disagree"):
        _run(original)


def test_cross_core_flags_are_not_consumed_as_a_local_manual_protocol():
    buffers = _buffers()
    cross_set = tir.Evaluate(
        tir.Call(
            "handle",
            tvm.ir.Op.get("tl.ascend_set_cross_flag"),
            [tir.StringImm("FIX"), tir.IntImm("int32", 3), tir.IntImm("int32", 2)],
        ),
    )
    cross_wait = tir.Evaluate(
        tir.Call(
            "handle",
            tvm.ir.Op.get("tl.ascend_wait_cross_flag"),
            [tir.IntImm("int32", 3), tir.StringImm("MTE2")],
        ),
    )
    original = _function(
        buffers,
        [cross_set, *_local_protocol_statements(buffers), cross_wait],
        b_offset=256,
    )
    with pytest.raises(tvm.error.TVMError, match="cross-core publication"):
        _run(original)


def test_automatic_flags_are_not_silently_removed_with_a_manual_protocol():
    buffers = _buffers()
    automatic = [
        tir.Evaluate(
            tir.Call(
                "handle",
                tvm.ir.Op.get(f"tl.ascend_auto_{kind}_flag"),
                [tir.StringImm("MTE2_V"), tir.IntImm("int32", 7)],
            ),
        )
        for kind in ("set", "wait")
    ]
    original = _function(
        buffers,
        [*_local_protocol_statements(buffers), *automatic],
        b_offset=256,
    )
    with pytest.raises(tvm.error.TVMError, match="mixed explicit and automatic"):
        _run(original)


@pytest.mark.parametrize("arithmetic", ["add", "multiply", "narrow_cast"])
def test_event_id_cannot_hide_an_overflow_before_modulo(arithmetic):
    buffers = _buffers()
    if arithmetic == "add":
        # Raw constructors preserve the overflowing intermediate until the pass.
        overflowing = tir.Add(tir.IntImm("int32", 2147483647), tir.IntImm("int32", 1))
    elif arithmetic == "multiply":
        overflowing = tir.Mul(tir.IntImm("int32", 1073741824), tir.IntImm("int32", 2))
    else:
        overflowing = tir.Cast("int8", tir.IntImm("int32", 256))
    event_id = tir.FloorMod(overflowing, tir.IntImm(overflowing.dtype, 8))
    original = _function(
        buffers,
        _local_protocol_statements(buffers, event_id=event_id),
        b_offset=256,
    )
    with pytest.raises(tvm.error.TVMError, match="integer overflow in protocol expression"):
        _run(original)


def test_protocol_loop_endpoint_must_fit_the_original_loop_variable():
    buffers = _buffers()
    original = _function(
        buffers,
        _local_protocol_statements(buffers, minimum=2147483647, extent=2),
        b_offset=256,
    )
    with pytest.raises(tvm.error.TVMError, match="nonrepresentable protocol loop endpoint"):
        _run(original)


def test_static_manual_protocol_has_a_bounded_compile_time_proof_cost():
    buffers = _buffers()
    original = _function(
        buffers,
        _local_protocol_statements(buffers, extent=1048576),
        b_offset=256,
    )
    with pytest.raises(tvm.error.TVMError, match="protocol execution budget exceeded"):
        _run(original)


def test_nested_flag_call_is_rejected_before_its_effect_can_be_lost():
    buffers = _buffers()
    iteration = tir.Var("nested_manual_iteration", "int32")
    nested = tir.LetStmt(
        tir.Var("nested_manual_result", "handle"),
        _flag("set").value,
        tir.SeqStmt([_flag("wait"), *_payload(buffers)]),
    )
    original = _function(
        buffers,
        [tir.For(iteration, 0, 2, tir.ForKind.SERIAL, nested)],
        b_offset=256,
    )
    with pytest.raises(
        tvm.error.TVMError,
        match="hard flags must be standalone Evaluate calls|snapshot requires a scalar value",
    ):
        _run(original)


@pytest.mark.parametrize("protocol", ["large_loop", "many_conditions", "automatic_flags"])
def test_previously_supported_balanced_protocol_keeps_its_original_events(protocol):
    buffers = _buffers()
    count = tir.Var("count", "int32")
    iteration = tir.Var("iteration", "int32")
    payload = [*_payload(buffers), _flag("set"), _flag("wait")]
    if protocol == "many_conditions":
        # Keep the conditional event protocol independent of memory hazards
        # between nine optional payloads, which have a separate slot limit.
        statements = [
            *_payload(buffers),
            *[
                tir.IfThenElse(
                    count > index,
                    tir.SeqStmt([_flag("set"), _flag("wait")]),
                    None,
                )
                for index in range(9)
            ],
        ]
    else:
        if protocol == "automatic_flags":
            payload += [
                tir.Evaluate(
                    tir.Call(
                        "handle",
                        tvm.ir.Op.get(f"tl.ascend_auto_{kind}_flag"),
                        [tir.StringImm("MTE2_V"), tir.IntImm("int32", 7)],
                    ),
                )
                for kind in ("set", "wait")
            ]
        extent = 1000000 if protocol == "large_loop" else 2
        statements = [tir.For(iteration, 0, extent, tir.ForKind.SERIAL, tir.SeqStmt(payload))]
    original = _function(buffers, statements, b_offset=256, scalars=(count,))
    result = _run(original)
    expected = [call for call in _calls(original) if getattr(call.op, "name", "") in _MANUAL]
    actual = [call for call in _calls(result) if getattr(call.op, "name", "") in _MANUAL]
    assert len(actual) == len(expected)
    assert all(tvm.ir.structural_equal(before, after) for before, after in zip(expected, actual))


def test_unscoped_and_explicit_vector_flags_share_the_same_physical_slot():
    buffers = _buffers()
    iteration = tir.Var("iteration", "int32")
    nested_pair = tir.AttrStmt(
        tir.IntImm("int32", 0),
        "resource_scope",
        1,
        tir.SeqStmt(
            [
                _flag("set", source="MTE2", destination="V"),
                _flag("wait", source="MTE2", destination="V"),
            ],
        ),
    )
    body = tir.SeqStmt(
        [
            _flag("wait", 1),
            _flag("set", source="MTE2", destination="V"),
            nested_pair,
            _flag("wait", source="MTE2", destination="V"),
            *_payload(buffers),
            _flag("set", 1),
        ],
    )
    original = _function(
        buffers,
        [_flag("set", 1), tir.For(iteration, 0, 1, tir.ForKind.SERIAL, body), _flag("wait", 1)],
        b_offset=256,
    )
    scopes = [
        node
        for node in _statements(original.body)
        if isinstance(
            node,
            tir.AttrStmt,
        )
        and node.attr_key == "resource_scope"
    ]
    outer = scopes[-1]

    def remove_outer(statement):
        if statement.same_as(outer):
            return statement.body
        return None

    original = original.with_body(
        tir.stmt_functor.ir_transform(original.body, remove_outer, None, ["tir.AttrStmt"]),
    )
    with pytest.raises(tvm.error.TVMError, match="duplicate SetFlag"):
        _run(original)
