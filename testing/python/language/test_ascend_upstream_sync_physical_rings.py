# Copyright (c) Tile-AI Organization.
# Licensed under the MIT License.
"""CPU tests for the materialized UB provenance admission contract.

These fixtures state physical allocation maps and injector clocks explicitly.
They do not compile a kernel, invoke a legacy sync pass, or claim pipeline
performance. Actual injector propagation has a separate test suite.
"""

import pytest

import tilelang
from tilelang import tvm

tir = tvm.tir
PROVENANCE = "tl.ascend_physical_ring_provenance"
CLOCK = "tl.ascend_physical_ring_clock"


@pytest.fixture(autouse=True)
def _requires_proof_service() -> None:
    if not tvm.get_global_func("tl.ascend_sync.Z3ProofServiceAvailable")():
        pytest.skip("requires USE_ASCEND_SYNC_Z3=ON and the migrated sync engine")


def _function(
    *,
    clock: int | tir.PrimExpr | None = 0,
    pointer_offset: int | tir.PrimExpr = 0,
    pointer_extent: int = 64,
    versions: int = 2,
    base_bits: int = 0,
    stride_bits: int = 2048,
    element_bits: int = 32,
    original_min: int = 0,
    allocation_elements: int | None = None,
    allocation_bytes: int | None = None,
    result_address: int | None = None,
    duplicate_identity: bool = False,
    scalar_access: bool = False,
    loop_var: tir.Var | None = None,
    copy_columns: int = 64,
    abs_count: int = 64,
) -> tir.PrimFunc:
    if allocation_elements is None:
        allocation_elements = versions * 64
    if allocation_bytes is None:
        allocation_bytes = versions * 256
    if result_address is None:
        result_address = allocation_bytes
    source = tir.decl_buffer((max(64, copy_columns),), "float32", name="input")
    output = tir.decl_buffer((64,), "float32", name="output")
    physical = tir.decl_buffer((allocation_elements,), "float32", name="physical", scope="shared.ub")
    # The stable identity differs from the final pointer after StorageRewrite.
    logical = tir.decl_buffer((64,), "float32", name="logical", scope="shared.ub")
    result = tir.decl_buffer((max(64, abs_count),), "float32", name="result", scope="shared.ub")
    load = tir.Evaluate(
        tir.call_extern(
            "handle",
            f"tl::ascend::copy_gm_to_ub<float, {copy_columns}>",
            source.access_ptr("r"),
            physical.access_ptr("w", offset=pointer_offset, extent=pointer_extent),
            copy_columns,
            1,
            copy_columns,
            tir.const(0, "float32"),
            64,
        )
    )
    calculate = tir.Evaluate(
        tir.Call(
            "handle",
            tvm.ir.Op.get("tl.ascend_abs"),
            [
                result.access_ptr("w"),
                physical.access_ptr("r", offset=pointer_offset, extent=64),
                abs_count,
            ],
        )
    )
    if scalar_access:
        calculate = tir.BufferStore(physical, tir.const(1, "float32"), [0])
    body = tir.SeqStmt([load, calculate])
    if clock is not None:
        body = tir.AttrStmt(tvm.runtime.convert([logical.data]), CLOCK, clock, body)
    if loop_var is not None:
        body = tir.For(loop_var, 0, 2, tir.ForKind.SERIAL, body)
    store = tir.Evaluate(tir.call_extern("handle", "copy_ub_to_gm", result.access_ptr("r"), output.access_ptr("w")))
    body = tir.SeqStmt([body, store])
    body = tir.AttrStmt(tir.IntImm("int32", 0), "resource_scope", 1, body)
    for buffer in (physical, result):
        body = tir.Allocate(buffer.data, buffer.dtype, buffer.shape, tir.const(True, "bool"), body)
    fields = [
        tir.IntImm("int32", 1),
        tir.IntImm("int32", versions),
        physical.data,
        tir.IntImm("int64", base_bits),
        tir.IntImm("int64", stride_bits),
        tir.IntImm("int64", element_bits),
        tir.IntImm("int32", original_min),
    ]
    provenance = {logical.data: fields}
    version_map = {logical.data: versions, result.data: 1}
    if duplicate_identity:
        duplicate = tir.decl_buffer((64,), "float32", name="other", scope="shared.ub")
        provenance[duplicate.data] = fields
        version_map[duplicate.data] = versions
    return (
        tir.PrimFunc(
            [source.data, output.data],
            body,
            buffer_map={source.data: source, output.data: output},
        )
        .with_attr("address_map", {physical.data: 0, result.data: result_address})
        .with_attr("size_map", {physical.data: allocation_bytes, result.data: max(64, abs_count) * 4})
        .with_attr("buffer_versions", version_map)
        .with_attr(PROVENANCE, provenance)
        .with_attr("tir.noalias", True)
    )


def _run(function: tir.PrimFunc) -> tir.PrimFunc:
    target = tvm.target.Target("c -keys=ascend -model=ascendc")
    with tvm.transform.PassContext(config={"tl.ascend_auto_sync": True, "tl.ascend_auto_sync_vs": True}):
        result = tilelang.transform.AscendInsertSync(target, "A3")(tvm.IRModule({"main": function}))
    return result["main"]


def _assert_mte2_vector_handoff(function: tir.PrimFunc) -> None:
    result = _run(function)
    # Legacy TVM StructuralEqual cannot reliably compare a standalone Var-key
    # map, even with itself. Check logical identities and every fact directly.
    for attribute in ("buffer_versions", PROVENANCE):
        actual, expected = result.attrs[attribute], function.attrs[attribute]
        assert len(actual) == len(expected)
        for storage, facts in expected.items():
            assert storage in actual
            if attribute == "buffer_versions":
                assert int(actual[storage]) == int(facts)
            else:
                assert len(actual[storage]) == len(facts)
                for observed, original in zip(actual[storage], facts):
                    if isinstance(original, tir.Var):
                        assert observed.same_as(original)
                    else:
                        tvm.ir.assert_structural_equal(observed, original)
    assert result.attrs.get("tl.ascend_sync_proof") is not None
    calls = []

    def collect(node: object) -> None:
        if isinstance(node, tir.Call) and isinstance(node.op, tvm.ir.Op):
            calls.append(node)

    tir.stmt_functor.post_order_visit(result.body, collect)
    sets = [call for call in calls if call.op.name == "tl.ascend_auto_set_flag" and call.args[0].value == "MTE2_V"]
    waits = [call for call in calls if call.op.name == "tl.ascend_auto_wait_flag" and call.args[0].value == "MTE2_V"]
    assert sets and waits
    assert {int(call.args[1]) for call in sets} == {int(call.args[1]) for call in waits}


@pytest.mark.parametrize("versions", [2, 3])
def test_ring_keeps_logical_identity_after_physical_root_rewrite(versions: int) -> None:
    _assert_mte2_vector_handoff(_function(versions=versions))


@pytest.mark.parametrize("pointer_extent", [0, 1])
def test_ring_footprint_covers_the_complete_slot_even_with_short_access_extent(
    pointer_extent: int,
) -> None:
    # Actual full-mask copy writes 64 elements. Its descriptive access_ptr
    # extent must never shrink the graph footprint to one element.
    _assert_mte2_vector_handoff(_function(pointer_extent=pointer_extent))


@pytest.mark.parametrize("versions", [2, 3])
def test_simplified_physical_offset_is_proven_against_the_skewed_clock(versions: int) -> None:
    index = tir.Var("tile", "int32")
    # With 0 <= tile < 2, tile*64 equals ((tile+B) % B)*64. This is the
    # actual simplified steady-loop form produced by the old injector.
    _assert_mte2_vector_handoff(_function(clock=index + versions, pointer_offset=index * 64, loop_var=index, versions=versions))


@pytest.mark.parametrize("wrap", [1, 2])
def test_last_slot_and_wrapped_slot_use_the_real_version_count(wrap: int) -> None:
    versions = 3
    _assert_mte2_vector_handoff(
        _function(
            clock=wrap * versions + versions - 1,
            pointer_offset=(versions - 1) * 64,
            versions=versions,
        )
    )


@pytest.mark.parametrize(
    "kwargs,diagnostic",
    [
        ({"clock": None}, "missing injector clock provenance"),
        ({"clock": -1}, "clock must be provably nonnegative"),
        ({"pointer_offset": 64}, "clock-selected slot"),
        ({"pointer_offset": 1, "pointer_extent": 1}, "clock-selected slot"),
        ({"pointer_extent": 65}, "clock-selected slot"),
        ({"versions": 9}, "eight-slot.*event capacity"),
        ({"versions": 3, "allocation_bytes": 512}, "allocation/maps"),
        ({"versions": 4, "allocation_elements": 128}, "allocation shape/maps"),
        ({"original_min": 1}, "original loop minimum zero"),
        ({"base_bits": 4}, "aligned"),
        ({"stride_bits": 0}, "aligned"),
        ({"stride_bits": 2016}, "32-byte alignment"),
        ({"element_bits": 64}, "element bits.*dtype"),
        ({"allocation_bytes": 256}, "allocation/maps"),
        ({"allocation_elements": 64}, "allocation shape/maps"),
        ({"duplicate_identity": True}, "Multiple logical physical rings"),
        ({"result_address": 0}, "aliases another logical allocation"),
        ({"scalar_access": True}, "scalar accesses"),
    ],
)
def test_unknown_or_inconsistent_ring_contract_is_rejected(kwargs: dict[str, object], diagnostic: str) -> None:
    with pytest.raises(tvm.error.TVMError, match=diagnostic):
        _run(_function(**kwargs))


@pytest.mark.parametrize("versions", [2, 3])
@pytest.mark.parametrize("kwargs", [{"copy_columns": 128}, {"abs_count": 65}])
def test_real_helper_footprint_cannot_borrow_an_adjacent_ring_slot(
    kwargs: dict[str, int],
    versions: int,
) -> None:
    # Both pointers declare extent64 and start at slot0. The real helper ABI,
    # independently of these descriptors, would touch slot1 as well.
    with pytest.raises(tvm.error.TVMError, match="(?i)physical ring.*(footprint|slot)"):
        _run(_function(**kwargs, versions=versions))


def test_ring_provenance_requires_the_injector_version_table() -> None:
    with pytest.raises(tvm.error.TVMError, match="matching buffer_versions metadata"):
        _run(_function().with_attr("buffer_versions", {}))


def test_valid_b3_ring_cannot_spill_past_the_actual_reserved_event_pool() -> None:
    original = _function(versions=3)
    flags = []
    for event_id in range(8):
        for kind in ("set", "wait"):
            flags.append(
                tir.Evaluate(
                    tir.Call(
                        "handle",
                        tvm.ir.Op.get(f"tl.ascend_{kind}_flag"),
                        [tir.StringImm("MTE2"), tir.StringImm("V"), tir.IntImm("int32", event_id)],
                    )
                )
            )

    def reserve(statement):
        if isinstance(statement, tir.AttrStmt) and statement.attr_key == "resource_scope":
            return tir.AttrStmt(
                statement.node,
                statement.attr_key,
                statement.value,
                tir.SeqStmt([*flags, statement.body]),
            )
        return None

    original = original.with_body(tir.stmt_functor.ir_transform(original.body, reserve, None, ["tir.AttrStmt"]))
    with pytest.raises(tvm.error.TVMError, match="(?i)(capacity exhausted|exhausted.*event slots)"):
        _run(original)


@pytest.mark.parametrize(
    "kwargs,diagnostic",
    [
        ({"result_address": 0}, "aliases another logical allocation"),
        ({"scalar_access": True}, "scalar accesses"),
    ],
)
def test_b3_preserves_alias_and_scalar_storage_restrictions(kwargs, diagnostic):
    with pytest.raises(tvm.error.TVMError, match=diagnostic):
        _run(_function(versions=3, **kwargs))


def test_b3_preserves_guard_free_admission():
    original = _function(versions=3)

    def guarded(statement):
        if isinstance(statement, tir.AttrStmt) and statement.attr_key == CLOCK:
            body = tir.IfThenElse(tir.const(True, "bool"), statement.body, None)
            return tir.AttrStmt(statement.node, statement.attr_key, statement.value, body)
        return None

    original = original.with_body(tir.stmt_functor.ir_transform(original.body, guarded, None, ["tir.AttrStmt"]))
    with pytest.raises(tvm.error.TVMError, match="requires guard-free tasks"):
        _run(original)


def test_b3_preserves_static_loop_bounds():
    index = tir.Var("tile", "int32")
    extent = tir.Var("runtime_extent", "int32")
    original = _function(versions=3, clock=index, pointer_offset=(index % 3) * 64, loop_var=index)

    def dynamic(statement):
        if isinstance(statement, tir.For):
            return tir.For(statement.loop_var, statement.min, extent, statement.kind, statement.body)
        return None

    body = tir.stmt_functor.ir_transform(original.body, dynamic, None, ["tir.For"])
    original = tir.PrimFunc([*original.params, extent], body, original.ret_type, original.buffer_map, original.attrs)
    with pytest.raises(tvm.error.TVMError, match="requires static For bounds"):
        _run(original)
