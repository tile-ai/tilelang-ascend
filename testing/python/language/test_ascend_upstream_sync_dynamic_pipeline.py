# Copyright (c) Tile-AI Organization.
# Licensed under the MIT License.
"""Real dynamic SIP partitioning, metadata propagation and protocol oracles.

The completion oracle uses actual byte addresses and binary events, independently
of owner/clock annotations. Runtime dimensions are substituted only for execution
of the final IR, without another optimization or synchronization pass.
"""

import pytest

import tilelang
from tilelang import language as T
from tilelang import tvm

from test_ascend_pipeline_ring_provenance import _nodes, _pipelined_abs, _pre_sync
from test_ascend_upstream_sync_pipeline import _assert_balanced_events, _execute_protocol

tir = tvm.tir
_RINGS = "tl.ascend_physical_ring_provenance"
_VERSIONS = "buffer_versions"
_OWNERS = "tl.ascend_physical_pipeline_owners"
_EXTENT = "tl.ascend_physical_pipeline_extent"
_SEGMENT = "tl.ascend_physical_pipeline_segment"
_STAGE = "tl.ascend_physical_pipeline_stage"
_CLOCK = "tl.ascend_physical_ring_clock"


@pytest.fixture(scope="module", autouse=True)
def _requires_proof_service() -> None:
    if not tvm.get_global_func("tl.ascend_sync.Z3ProofServiceAvailable")():
        pytest.skip("requires USE_ASCEND_SYNC_Z3=ON")


def _run(function: tir.PrimFunc) -> tir.PrimFunc:
    target = tvm.target.Target({"kind": "llvm", "model": "ascendc"})
    with tvm.transform.PassContext(config={"tl.ascend_auto_sync": True, "tl.ascend_auto_sync_vs": True}):
        return tilelang.transform.AscendInsertSync(target, "A3")(tvm.IRModule({"main": function}))["main"]


def _attributes(function: tir.PrimFunc, key: str) -> list[tir.AttrStmt]:
    return [node for node in _nodes(function.body, tir.AttrStmt) if node.attr_key == key]


def _dimension(function: tir.PrimFunc) -> tir.Var:
    extents = _attributes(function, _EXTENT)
    assert len(extents) == 1 and isinstance(extents[0].value, tir.Var)
    return extents[0].value


def _operation(call: tir.Call) -> str:
    if call.op.name == "tir.call_extern":
        return call.args[0].value.split("<", 1)[0].removeprefix("tl::ascend::")
    return call.op.name


def _contains_operation(statement: tir.Stmt, name: str) -> bool:
    return any(_operation(node) == name for node in _nodes(statement, tir.Call))


def _dynamic_programs(num_stages):
    result = {}
    for planning in (False, True):
        program = _pipelined_abs(T.symbolic("n"), num_stages).with_attr("tir.noalias", True)
        before, snapshots = _pre_sync(program, memory_planning=planning)
        result[planning] = (before, snapshots, _run(before))
    return result


@pytest.fixture(scope="module")
def dynamic_programs():
    return _dynamic_programs(2)


@pytest.fixture(scope="module")
def three_stage_programs():
    return _dynamic_programs(3)


def _specialize_runtime_body(function, dimension, trip_count):
    return function.with_body(tir.stmt_functor.substitute(function.body, {dimension: tir.IntImm(dimension.dtype, trip_count)}))


def _assert_runtime_protocol(before, after, trip_count, vector_id, num_stages):
    executed = _specialize_runtime_body(after, _dimension(before), trip_count)
    events, computations = _execute_protocol(executed, vector_id)
    for operation in ("copy_gm_to_ub", "tl.ascend_abs", "copy_ub_to_gm"):
        assert computations[operation] == trip_count
    if trip_count:
        _assert_balanced_events(events, set())
    else:
        assert not events, "zero work must not leave a speculative protocol outside its case"
    for name in ("address_map", "size_map", _VERSIONS, _RINGS, _OWNERS):
        expected = before.attrs[name]
        actual = after.attrs[name]
        assert len(expected) == len(actual)
        for key, value in expected.items():
            assert key in actual
            assert tvm.ir.structural_equal(actual[key], value)
    assert int(next(iter(after.attrs[_OWNERS].values()))[1]) == num_stages
    logical_id, ring = next(iter(after.attrs[_RINGS].items()))
    assert int(ring[1]) == int(after.attrs[_VERSIONS][logical_id]) == num_stages
    assert len(_nodes(before.body, tir.Allocate)) == len(_nodes(after.body, tir.Allocate))
    assert not any(var.name.startswith("__ascend_if_proxy_") for var in _nodes(after.body, tir.Var)), (
        "pipeline cases and bounds must never become synthetic epoch loops"
    )
    assert tvm.ir.structural_equal(after, _run(after))


@pytest.mark.parametrize("memory_planning", [False, True])
@pytest.mark.parametrize("trip_count", [0, 1, 2, 3, 5, 33])
@pytest.mark.parametrize("vector_id", [0, 1])
def test_dynamic_pipeline_preserves_computation_bytes_and_binary_tokens(
    dynamic_programs, memory_planning: bool, trip_count: int, vector_id: int
) -> None:
    before, _, after = dynamic_programs[memory_planning]
    _assert_runtime_protocol(before, after, trip_count, vector_id, 2)


@pytest.mark.parametrize("memory_planning", [False, True])
@pytest.mark.parametrize("trip_count", [0, 1, 2, 3, 4, 7, 33])
@pytest.mark.parametrize("vector_id", [0, 1])
def test_three_stage_dynamic_pipeline_preserves_true_ring_and_protocol(
    three_stage_programs, memory_planning: bool, trip_count: int, vector_id: int
) -> None:
    before, _, after = three_stage_programs[memory_planning]
    _assert_runtime_protocol(before, after, trip_count, vector_id, 3)


@pytest.mark.parametrize("memory_planning", [False, True])
def test_dynamic_pipeline_metadata_survives_all_relevant_passes(dynamic_programs, memory_planning: bool) -> None:
    before, snapshots, _ = dynamic_programs[memory_planning]
    required = {
        "tl.InjectSoftwarePipeline",
        "tl.AscendLowerOpaqueBlock",
        "tl.FlattenBuffer",
        "tl.AscendStorageRewrite",
        "tir.HoistIfThenElse",
        "tl.AscendMemoryPlanning",
    }
    assert required.issubset(snapshots)
    for stage in required:
        function = snapshots[stage]
        assert len(function.attrs[_OWNERS]) == 1, stage
        owner, record = next(iter(function.attrs[_OWNERS].items()))
        assert [int(value) for value in record] == [1, 2, 0, 3], stage
        scopes = _attributes(function, _EXTENT)
        assert len(scopes) == 1 and scopes[0].node[0].same_as(owner), stage
        assert isinstance(scopes[0].value, tir.Var), stage
        frames = _attributes(function, _STAGE)
        assert frames and all(frame.node[0].same_as(owner) for frame in frames), stage
    stores = [frame for frame in _attributes(before, _STAGE) if _contains_operation(frame.body, "copy_ub_to_gm")]
    assert len(stores) == 3, "steady and both epilogue B1 stores need stage clocks"


def test_legacy_default_does_not_publish_dynamic_partition_metadata() -> None:
    program = _pipelined_abs(T.symbolic("n"), 2)
    before, _ = _pre_sync(program, backend="legacy")
    assert _OWNERS not in before.attrs
    assert not _attributes(before, _EXTENT)
    assert not _attributes(before, _STAGE)


@pytest.mark.parametrize("num_stages", [2, 3])
@pytest.mark.parametrize("memory_planning", [False, True])
@pytest.mark.parametrize("vector_id", [0, 1])
@pytest.mark.parametrize("event", ["MTE2_V", "V_MTE2", "V_MTE3", "MTE3_V"])
def test_dynamic_completion_oracle_rejects_deleted_whole_event_pairs(
    dynamic_programs, three_stage_programs, num_stages, memory_planning, vector_id, event
) -> None:
    programs = dynamic_programs if num_stages == 2 else three_stage_programs
    before, _, after = programs[memory_planning]

    def remove_pair(statement):
        if isinstance(statement, tir.Evaluate) and isinstance(statement.value, tir.Call):
            call = statement.value
            if _operation(call) in ("tl.ascend_auto_set_flag", "tl.ascend_auto_wait_flag") and call.args[0].value == event:
                return tir.Evaluate(0)
        return None

    body = tir.stmt_functor.ir_transform(after.body, remove_pair, None, ["tir.Evaluate"])
    corrupted = _specialize_runtime_body(after.with_body(body), _dimension(before), 33)
    # Removing both set and wait preserves binary token balance. Actual byte
    # completion must still expose the lost RAW/WAR dependency independently.
    with pytest.raises(AssertionError, match="memory dependency missing"):
        _execute_protocol(corrupted, vector_id)


@pytest.mark.parametrize("memory_planning", [False, True])
def test_three_stage_slot_cannot_hide_a_different_logical_clock(three_stage_programs, memory_planning) -> None:
    before = three_stage_programs[memory_planning][0]
    malformed = _rewrite_attribute(
        before,
        _CLOCK,
        lambda attr: tir.AttrStmt(attr.node, attr.attr_key, attr.value + 3, attr.body),
    )
    with pytest.raises(tvm.error.TVMError, match="ring clock disagrees"):
        _run(malformed)


def _rewrite_attribute(function: tir.PrimFunc, key: str, replacement) -> tir.PrimFunc:
    changed = False

    def rewrite(statement):
        nonlocal changed
        if not changed and isinstance(statement, tir.AttrStmt) and statement.attr_key == key:
            result = replacement(statement)
            if result is not None:
                changed = True
                return result
        return None

    body = tir.stmt_functor.ir_transform(function.body, None, rewrite, ["tir.AttrStmt"])
    assert changed, f"mutation did not find {key}"
    return function.with_body(body)


def _replace_record(function: tir.PrimFunc, name: str, index: int, value) -> tir.PrimFunc:
    records = dict(function.attrs[name].items())
    key = next(iter(records))
    record = list(records[key])
    record[index] = value
    records[key] = record
    return function.with_attr(name, records)


@pytest.mark.parametrize("mutation", ["owner_schema", "ring_schema", "lead", "version_table"])
def test_bad_dynamic_owner_or_version_certificates_are_rejected(dynamic_programs, mutation) -> None:
    before = dynamic_programs[False][0]
    if mutation == "owner_schema":
        malformed = _replace_record(before, _OWNERS, 0, tir.IntImm("int32", 9))
    elif mutation == "ring_schema":
        malformed = _replace_record(before, _RINGS, 0, tir.IntImm("int32", 9))
    elif mutation == "lead":
        malformed = _replace_record(before, _OWNERS, 1, tir.IntImm("int32", 3))
    else:
        versions = dict(before.attrs[_VERSIONS].items())
        versions[next(iter(before.attrs[_RINGS]))] = 3
        malformed = before.with_attr(_VERSIONS, versions)
    with pytest.raises(tvm.error.TVMError, match="(?i)(schema|lead|version|provenance)"):
        _run(malformed)


@pytest.mark.parametrize("metadata", [_OWNERS, _RINGS])
def test_multiple_owners_or_rings_are_not_silently_coalesced(dynamic_programs, metadata) -> None:
    before = dynamic_programs[False][0]
    values = dict(before.attrs[metadata].items())
    first_identity, first_record = next(iter(values.items()))
    identity = tir.Var("unrelated_identity", first_identity.dtype)
    values[identity] = first_record
    malformed = before.with_attr(metadata, values)
    if metadata == _RINGS:
        # Keep each ring certificate internally consistent so this tests the
        # family cardinality gate, rather than a missing-version rejection.
        versions = dict(before.attrs[_VERSIONS].items())
        versions[identity] = first_record[1]
        malformed = malformed.with_attr(_VERSIONS, versions)
    with pytest.raises(tvm.error.TVMError, match="(?i)(exactly one|one physical ring)"):
        _run(malformed)


@pytest.mark.parametrize("metadata", [_EXTENT, _STAGE])
def test_extent_and_stage_clock_changes_invalidate_the_bounds_certificate(dynamic_programs, metadata) -> None:
    before = dynamic_programs[False][0]
    malformed = _rewrite_attribute(
        before,
        metadata,
        lambda attr: tir.AttrStmt(attr.node, attr.attr_key, attr.value + 2, attr.body),
    )
    with pytest.raises(tvm.error.TVMError, match="(?i)(clock|bounds|extent|segment)"):
        _run(malformed)


def test_same_physical_slot_cannot_hide_a_different_logical_ring_clock(dynamic_programs) -> None:
    before = dynamic_programs[False][0]
    malformed = _rewrite_attribute(
        before,
        _CLOCK,
        lambda attr: tir.AttrStmt(attr.node, attr.attr_key, attr.value + 2, attr.body),
    )
    with pytest.raises(tvm.error.TVMError, match="ring clock disagrees"):
        _run(malformed)


def test_missing_b1_store_stage_clock_is_rejected(dynamic_programs) -> None:
    before = dynamic_programs[False][0]
    malformed = _rewrite_attribute(
        before,
        _STAGE,
        lambda attr: attr.body if _contains_operation(attr.body, "copy_ub_to_gm") else None,
    )
    with pytest.raises(tvm.error.TVMError, match="(?i)(unmarked|metadata|stage)"):
        _run(malformed)


def test_b1_store_must_use_its_exact_clock_bounds_not_a_stronger_guard(dynamic_programs) -> None:
    before = dynamic_programs[False][0]
    extent = _dimension(before)
    analyzer = tvm.arith.Analyzer()

    def change(attr):
        if int(attr.node[1]) != 2 or not _contains_operation(attr.body, "copy_ub_to_gm"):
            return None
        if not analyzer.can_prove_equal(attr.value, extent - 1):
            return None
        changed = False

        def guard(node):
            nonlocal changed
            if isinstance(node, tir.IfThenElse) and not changed:
                changed = True
                return tir.IfThenElse(extent >= 2, node.then_case, node.else_case)
            return None

        body = tir.stmt_functor.ir_transform(attr.body, None, guard, ["tir.IfThenElse"])
        assert changed
        return tir.AttrStmt(attr.node, attr.attr_key, attr.value, body)

    malformed = _rewrite_attribute(before, _STAGE, change)
    with pytest.raises(tvm.error.TVMError, match="bounds guard is not equivalent"):
        _run(malformed)


def test_bounds_equivalence_must_not_erase_an_opaque_guard_effect(dynamic_programs) -> None:
    before = dynamic_programs[False][0]

    def change(attr):
        changed = False

        def guard(node):
            nonlocal changed
            if isinstance(node, tir.IfThenElse) and not changed:
                changed = True
                effect = tir.call_extern("int32", "opaque_guard_effect")
                # Two executions of an opaque call may return different
                # values. Even a mathematical tautology cannot discard them.
                condition = tir.And(node.condition, tir.Or(effect == 0, effect != 0))
                return tir.IfThenElse(condition, node.then_case, node.else_case)
            return None

        body = tir.stmt_functor.ir_transform(attr.body, None, guard, ["tir.IfThenElse"])
        assert changed
        return tir.AttrStmt(attr.node, attr.attr_key, attr.value, body)

    malformed = _rewrite_attribute(before, _STAGE, change)
    assert _contains_operation(malformed.body, "opaque_guard_effect")
    with pytest.raises(tvm.error.TVMError, match="(?i)pure"):
        _run(malformed)


@pytest.mark.parametrize("field", ["owner_extent", "stage_clock", "loop_min", "loop_extent"])
def test_schedule_proof_inputs_are_pure_before_simplification(dynamic_programs, field) -> None:
    before = dynamic_programs[False][0]

    def with_effect(value):
        effect = tir.call_extern(value.dtype, "opaque_schedule_effect")
        return tir.Add(value, tir.Mul(tir.IntImm(value.dtype, 0), effect))

    if field in ("owner_extent", "stage_clock"):
        key = _EXTENT if field == "owner_extent" else _STAGE
        malformed = _rewrite_attribute(
            before,
            key,
            lambda attr: tir.AttrStmt(attr.node, attr.attr_key, with_effect(attr.value), attr.body),
        )
    else:
        changed = False

        def rewrite(node):
            nonlocal changed
            if isinstance(node, tir.For) and not changed:
                changed = True
                return tir.For(
                    node.loop_var,
                    with_effect(node.min) if field == "loop_min" else node.min,
                    with_effect(node.extent) if field == "loop_extent" else node.extent,
                    node.kind,
                    node.body,
                    node.thread_binding,
                    node.annotations,
                    node.span,
                )
            return None

        body = tir.stmt_functor.ir_transform(before.body, None, rewrite, ["tir.For"])
        assert changed
        malformed = before.with_body(body)
    assert _contains_operation(malformed.body, "opaque_schedule_effect")
    with pytest.raises(tvm.error.TVMError, match="(?i)pure"):
        _run(malformed)


def test_user_ring_if_cannot_borrow_an_injector_bounds_marker(dynamic_programs) -> None:
    before = dynamic_programs[False][0]
    predicate = tir.Var("user_predicate", "bool")
    malformed = _rewrite_attribute(
        before,
        _CLOCK,
        lambda attr: tir.AttrStmt(attr.node, attr.attr_key, attr.value, tir.IfThenElse(predicate, attr.body, None)),
    )
    malformed = tir.PrimFunc(
        [*malformed.params, predicate],
        malformed.body,
        malformed.ret_type,
        malformed.buffer_map,
        malformed.attrs,
    )
    with pytest.raises(tvm.error.TVMError, match="(?i)(user If|unsupported control)"):
        _run(malformed)


def test_async_protocol_hidden_inside_stage_payload_is_rejected(dynamic_programs) -> None:
    before = dynamic_programs[False][0]
    malformed = _rewrite_attribute(
        before,
        _CLOCK,
        lambda attr: tir.AttrStmt(
            attr.node,
            attr.attr_key,
            attr.value,
            tir.AttrStmt(tir.IntImm("int32", 0), "async_scope", 1, attr.body),
        ),
    )
    with pytest.raises(tvm.error.TVMError, match="(?i)async"):
        _run(malformed)


@pytest.mark.parametrize("wrapper", ["for", "if", "extra"])
def test_outer_semantic_control_and_extra_computation_remain_unsupported(dynamic_programs, wrapper) -> None:
    before = dynamic_programs[False][0]
    if wrapper == "for":
        body = tir.For(tir.Var("outer", "int32"), 0, 2, tir.ForKind.SERIAL, before.body)
    elif wrapper == "if":
        body = tir.IfThenElse(_dimension(before) > 0, before.body, None)
    else:
        body = tir.SeqStmt([before.body, tir.Evaluate(tir.call_extern("handle", "extra_effect"))])
    with pytest.raises(tvm.error.TVMError, match="(?i)(outer semantic|extra computation)"):
        _run(before.with_body(body))


def test_static_zero_trip_preserves_true_versions_without_requiring_dead_maps() -> None:
    before, _ = _pre_sync(_pipelined_abs(0, 2))
    after = _run(before)
    assert before.attrs[_VERSIONS]
    for key, version in before.attrs[_VERSIONS].items():
        assert int(after.attrs[_VERSIONS][key]) == int(version)
    assert not _nodes(after.body, tir.Call)


@pytest.mark.parametrize("operation", ["mutable_read", "opaque_effect"])
def test_dead_pipeline_metadata_does_not_hide_executable_wrapper_effects(operation) -> None:
    before, _ = _pre_sync(_pipelined_abs(0, 2))
    if operation == "mutable_read":
        source = tir.decl_buffer((1,), "int32", name="runtime_scalar")
        wrapped = tir.LetStmt(tir.Var("saved", "int32"), source[0], before.body)
        buffer_map = dict(before.buffer_map.items())
        buffer_map[source.data] = source
        malformed = tir.PrimFunc([*before.params, source.data], wrapped, before.ret_type, buffer_map, before.attrs)
    else:
        malformed = before.with_body(
            tir.LetStmt(
                tir.Var("saved", "int32"),
                tir.call_extern("int32", "opaque_runtime_effect"),
                before.body,
            )
        )
    # With executable work, the absent dead allocation/maps cannot use the
    # empty-body exception. The strict importer must check the real program.
    with pytest.raises(tvm.error.TVMError):
        _run(malformed)


@pytest.mark.parametrize(
    "mutation",
    ["schema", "versions", "owner", "slot_stride", "orders", "missing_versions", "clock_id"],
)
def test_zero_work_does_not_hide_corrupted_provenance(mutation) -> None:
    before, _ = _pre_sync(_pipelined_abs(0, 2))
    if mutation == "schema":
        malformed = _replace_record(before, _RINGS, 0, tir.IntImm("int32", 9))
    elif mutation == "owner":
        malformed = _replace_record(before, _OWNERS, 0, tir.IntImm("int32", 9))
    elif mutation == "slot_stride":
        malformed = _replace_record(before, _RINGS, 4, tir.IntImm("int64", 0))
    elif mutation == "orders":
        malformed = _replace_record(before, _OWNERS, 3, tir.IntImm("int32", 0))
    elif mutation == "missing_versions":
        malformed = before.without_attr(_VERSIONS)
    elif mutation == "clock_id":
        malformed = before.with_body(tir.AttrStmt([tir.Var("unknown_ring", "handle")], _CLOCK, 0, before.body))
    else:
        versions = dict(before.attrs[_VERSIONS].items())
        versions[next(iter(before.attrs[_RINGS]))] = 3
        malformed = before.with_attr(_VERSIONS, versions)
    with pytest.raises(tvm.error.TVMError, match="(?i)(schema|version|provenance|owner)"):
        _run(malformed)
