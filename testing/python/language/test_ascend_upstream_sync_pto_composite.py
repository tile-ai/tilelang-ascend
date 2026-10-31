# Copyright (c) Tile-AI Organization.
# Licensed under the MIT License.
"""CPU integration tests for the A3 PTO flat gemm_v0 contract.

These tests transform real lowered GEMM IR without compiling device source or
executing a kernel. Expected hidden event IDs come from the audited PTO helper
ABI; the legacy synchronization pass is not used as an oracle.
"""

from itertools import permutations

import pytest

from tilelang import tvm

from test_ascend_pipeline_ring_provenance import _nodes, _pre_sync
from test_ascend_upstream_sync_composite import (
    _changed_call,
    _events,
    _gemms,
    _manual_flag,
    _rewrite_gemm,
    _run,
)
from test_tilelang_ascend_language_gemm_v0_n_tiling import gemm_nt

tir = tvm.tir
_HIDDEN_DIRECTIONS = ("MTE2_MTE1", "FIX_M", "M_MTE1", "MTE1_M", "MTE1_MTE2", "M_FIX")
_PROFILE = "PTO flat gemm_v0 contract"


@pytest.fixture(scope="module", autouse=True)
def _requires_proof_service() -> None:
    if not tvm.get_global_func("tl.ascend_sync.Z3ProofServiceAvailable")():
        pytest.skip("requires USE_ASCEND_SYNC_Z3=ON")


@pytest.fixture(scope="module")
def pto_flat_gemms():
    # Both K values are divisible by the factory's default kL0Size=128.
    # K128 has one internal step; K256 exercises the audited internal reuse.
    return {k: _pre_sync(gemm_nt(16, 16, k, "float16", "float").with_attr("tir.noalias", True))[0] for k in (128, 256)}


def _operation(call: tir.Call) -> str:
    name = getattr(call.op, "name", "")
    if name == "tir.call_extern":
        return call.args[0].value.split("<", 1)[0].removeprefix("tl::ascend::")
    return name


def _assert_hidden_ids_are_reserved(function: tir.PrimFunc) -> None:
    generated = _events(function)
    assert function.attrs.get("tl.ascend_sync_proof") is not None
    for _, event, event_id in generated:
        assert 0 <= event_id < 8
        if event in _HIDDEN_DIRECTIONS:
            assert event_id != 1, f"generated event collided with the PTO helper: {event}#1"


def _assert_helper_protocol(function: tir.PrimFunc, k: int) -> None:
    from test_ascend_upstream_sync_pto_composite_protocol import _execute, _program

    def without_manual_pair(statement):
        if (
            isinstance(statement, tir.Evaluate)
            and isinstance(statement.value, tir.Call)
            and _operation(statement.value) in ("tl.ascend_set_flag", "tl.ascend_wait_flag")
        ):
            return tir.Evaluate(tir.IntImm("int32", 0))
        return None

    # Manual pairs are checked separately. Removing them verifies that the
    # helper establishes completion without borrowing their event credits.
    body = tir.stmt_functor.ir_transform(function.body, without_manual_pair, None, ["tir.Evaluate"])
    queues = _program(function.with_body(body), k)
    for priority in permutations(range(len(queues))):
        _execute(queues, priority)


def _assert_original_computation_and_maps(before: tir.PrimFunc, after: tir.PrimFunc) -> None:
    assert len(_gemms(before)) == len(_gemms(after)) == 1
    tvm.ir.assert_structural_equal(_gemms(before)[0], _gemms(after)[0])
    for original, synchronized in zip(_gemms(before)[0].args[1:4], _gemms(after)[0].args[1:4]):
        assert original.args[1].same_as(synchronized.args[1])
        tvm.ir.assert_structural_equal(original, synchronized)
    for name in ("address_map", "size_map", "buffer_versions"):
        assert (name in before.attrs) == (name in after.attrs)
        if name not in before.attrs:
            continue
        expected, actual = before.attrs[name], after.attrs[name]
        assert len(expected) == len(actual)
        for root, value in expected.items():
            assert root in actual
            tvm.ir.assert_structural_equal(value, actual[root])
    operations = {"copy_gm_to_l1", "tl.ascend_gemm_v0", "copy_l0c_to_gm"}
    old_calls = [call for call in _nodes(before.body, tir.Call) if _operation(call) in operations]
    new_calls = [call for call in _nodes(after.body, tir.Call) if _operation(call) in operations]
    assert len(old_calls) == len(new_calls) == 4
    for old, new in zip(old_calls, new_calls):
        tvm.ir.assert_structural_equal(old, new)


@pytest.mark.parametrize("k", [128, 256])
@pytest.mark.parametrize("target_kind", ["c", "llvm"])
def test_pto_flat_gemm_preserves_one_calculation_and_routes_acquire_completion(pto_flat_gemms, k: int, target_kind: str) -> None:
    before = pto_flat_gemms[k]
    assert not _nodes(before.body, tir.For)
    assert not _nodes(before.body, tir.IfThenElse)
    assert not _events(before)
    after = _run(before, model="pto", target_kind=target_kind)
    _assert_original_computation_and_maps(before, after)
    _assert_hidden_ids_are_reserved(after)
    # The helper owns its MTE2 acquire and FIX completion handshakes. Verify
    # the actual asynchronous protocol instead of requiring redundant flags.
    _assert_helper_protocol(after, k)
    tvm.ir.assert_structural_equal(after, _run(after, model="pto", target_kind=target_kind))


def _with_balanced_manual_pair(function, event, event_id, position):
    pair = [_manual_flag(kind, event, event_id) for kind in ("set", "wait")]

    def rewrite(call):
        computation = tir.Evaluate(call)
        return tir.SeqStmt([*pair, computation] if position == "before" else [computation, *pair])

    return _rewrite_gemm(function, rewrite)


@pytest.mark.parametrize("event", _HIDDEN_DIRECTIONS)
@pytest.mark.parametrize("position", ["before", "after"])
def test_manual_pair_cannot_borrow_hidden_pto_id_one(pto_flat_gemms, event, position) -> None:
    original = _with_balanced_manual_pair(pto_flat_gemms[128], event, 1, position)
    with pytest.raises(
        tvm.error.TVMError,
        match="(?i)Manual event conflicts with composite gemm_v0 hidden event pool",
    ):
        _run(original, model="pto", target_kind="llvm")


@pytest.mark.parametrize("position", ["before", "after"])
def test_nonconflicting_forward_manual_pair_remains_reserved(pto_flat_gemms, position) -> None:
    original = _with_balanced_manual_pair(pto_flat_gemms[128], "MTE2_MTE1", 7, position)
    after = _run(original, model="pto", target_kind="llvm")
    _assert_original_computation_and_maps(original, after)
    _assert_hidden_ids_are_reserved(after)
    ids = [event_id for _, event, event_id in _events(after) if event == "MTE2_MTE1"]
    assert all(event_id not in (1, 7) for event_id in ids)
    manual = [call for call in _nodes(after.body, tir.Call) if _operation(call) in ("tl.ascend_set_flag", "tl.ascend_wait_flag")]
    assert len(manual) == 2
    assert {_operation(call) for call in manual} == {"tl.ascend_set_flag", "tl.ascend_wait_flag"}
    for call in manual:
        assert [argument.value for argument in call.args[:2]] == ["MTE2", "MTE1"]
        assert int(call.args[2]) == 7
    _assert_helper_protocol(after, 128)


def _with_parameter(function: tir.PrimFunc, parameter: tir.Var) -> tir.PrimFunc:
    return tir.PrimFunc(
        [*function.params, parameter],
        function.body,
        function.ret_type,
        function.buffer_map,
        function.attrs,
    )


@pytest.mark.parametrize("change", ["k64", "m32", "kl0_64"])
def test_pto_flat_contract_accepts_dimension_derived_shapes(pto_flat_gemms, change) -> None:
    if change in ("k64", "m32"):
        m, k = (16, 64) if change == "k64" else (32, 128)
        original = _pre_sync(gemm_nt(m, 16, k, "float16", "float"))[0]
    else:
        original = _rewrite_gemm(
            pto_flat_gemms[128],
            lambda call: tir.Evaluate(_changed_call(call, template="gemm_v0<half, float, 16, 16, 128, false, false, 64>")),
        )
    after = _run(original, model="pto", target_kind="llvm")
    _assert_original_computation_and_maps(original, after)
    _assert_hidden_ids_are_reserved(after)
    tvm.ir.assert_structural_equal(after, _run(after, model="pto", target_kind="llvm"))


@pytest.mark.parametrize(
    "change,diagnostic",
    [
        ("m17", "Composite gemm_v0 allocation does not cover its padded physical footprint"),
        ("n17", "Composite gemm_v0 allocation does not cover its padded physical footprint"),
        ("k17", _PROFILE + ": requires static 16-aligned"),
        ("kl0_17", "Composite gemm_v0 kL0Size must be 16-aligned and at most 4095"),
        ("kl0_4096", "Composite gemm_v0 kL0Size must be 16-aligned and at most 4095"),
        ("transpose_a", _PROFILE + ": requires FP16-to-FP32 and no transpose"),
        ("transpose_b", _PROFILE + ": requires FP16-to-FP32 and no transpose"),
        ("clear_false", _PROFILE + ": operand disagrees with its semantic descriptor"),
        ("clear_dynamic", _PROFILE + ": operand disagrees with its semantic descriptor"),
    ],
)
def test_pto_flat_profile_rejects_unvalidated_shapes_and_helper_modes(pto_flat_gemms, change, diagnostic) -> None:
    original = pto_flat_gemms[128]
    # Padded M/N footprints and kL0Size are rejected by common GEMM ABI
    # checks before the PTO descriptor audit, so assert their own diagnostics.
    if change in ("m17", "n17", "k17"):
        m, n, k = {"m17": (17, 16, 128), "n17": (16, 17, 128), "k17": (16, 16, 17)}[change]
        original = _pre_sync(gemm_nt(m, n, k, "float16", "float"))[0]
    elif change == "clear_false":
        original = _rewrite_gemm(original, lambda call: tir.Evaluate(_changed_call(call, clear=tir.const(False, "bool"))))
    elif change == "clear_dynamic":
        clear = tir.Var("runtime_clear", "bool")
        original = _with_parameter(
            _rewrite_gemm(original, lambda call: tir.Evaluate(_changed_call(call, clear=clear))),
            clear,
        )
    else:
        transpose_a, transpose_b = change == "transpose_a", change == "transpose_b"
        k_l0 = {"kl0_17": 17, "kl0_4096": 4096}.get(change, 128)
        template = f"gemm_v0<half, float, 16, 16, 128, {str(transpose_a).lower()}, {str(transpose_b).lower()}, {k_l0}>"
        original = _rewrite_gemm(original, lambda call: tir.Evaluate(_changed_call(call, template=template)))
    with pytest.raises(tvm.error.TVMError, match=diagnostic):
        _run(original, model="pto", target_kind="llvm")


@pytest.mark.parametrize(
    "change,diagnostic",
    [
        ("partial_n", "Composite gemm_v0 currently requires static n_actual == N"),
        ("dynamic_n", "Composite gemm_v0 currently requires static n_actual == N"),
        ("pointer_dtype", "Composite gemm_v0 pointer dtype disagrees"),
        ("footprint", "Composite gemm_v0 allocation does not cover its padded physical footprint"),
        ("base_offset", "Composite gemm_v0 currently requires base allocation pointers"),
    ],
)
def test_pto_retains_common_typed_and_physical_input_checks(pto_flat_gemms, change, diagnostic) -> None:
    original = pto_flat_gemms[128]
    if change in ("partial_n", "dynamic_n"):
        n_actual = tir.IntImm("int32", 8) if change == "partial_n" else tir.Var("runtime_n", "int32")
        original = _rewrite_gemm(original, lambda call: tir.Evaluate(_changed_call(call, n_actual=n_actual)))
        if change == "dynamic_n":
            original = _with_parameter(original, n_actual)
    elif change == "footprint":
        sizes = dict(original.attrs["size_map"].items())
        sizes[_gemms(original)[0].args[1].args[1]] = 32
        original = original.with_attr("size_map", sizes)
    else:

        def rewrite(call):
            arguments = list(call.args)
            pointer = arguments[1]
            fields = list(pointer.args)
            if change == "pointer_dtype":
                fields[0] = tir.Call("float32", tvm.ir.Op.get("tir.type_annotation"), [])
            else:
                fields[2] = fields[2] + 16
            arguments[1] = tir.Call(pointer.dtype, pointer.op, fields, pointer.span)
            return tir.Evaluate(tir.Call(call.dtype, call.op, arguments, call.span))

        original = _rewrite_gemm(original, rewrite)
    with pytest.raises(tvm.error.TVMError, match=diagnostic):
        _run(original, model="pto", target_kind="llvm")


@pytest.mark.parametrize(
    "change,diagnostic",
    [
        ("outer_for", _PROFILE),
        ("outer_if", _PROFILE + "|Composite gemm_v0 external If/While"),
        ("ring_b2", _PROFILE + "|Composite gemm_v0 physical B>1/version"),
        ("user_l0ab", _PROFILE + "|Composite gemm_v0 owns hidden L0A/L0B pools"),
        ("vector_owner", _PROFILE + "|Composite gemm_v0 execution owner must be Cube"),
    ],
)
def test_pto_flat_profile_keeps_control_storage_and_cube_ownership_boundaries(pto_flat_gemms, change, diagnostic) -> None:
    original = pto_flat_gemms[128]
    if change == "outer_for":
        original = _rewrite_gemm(
            original,
            lambda call: tir.For(tir.Var("outer_k", "int32"), 0, 2, tir.ForKind.SERIAL, tir.Evaluate(call)),
        )
    elif change == "outer_if":
        original = _rewrite_gemm(original, lambda call: tir.IfThenElse(tir.const(True, "bool"), tir.Evaluate(call), None))
    elif change == "ring_b2":
        original = original.with_attr("buffer_versions", {_gemms(original)[0].args[1].args[1]: 2})
    elif change == "user_l0ab":
        scratch = tir.decl_buffer((64,), "float16", name="user_l0ab", scope="wmma.matrix_a")
        original = original.with_body(tir.Allocate(scratch.data, scratch.dtype, [64], tir.const(True, "bool"), original.body))
    else:
        original = _rewrite_gemm(
            original,
            lambda call: tir.AttrStmt(tir.IntImm("int32", 0), "resource_scope", 1, tir.Evaluate(call)),
        )
    with pytest.raises(tvm.error.TVMError, match=diagnostic):
        _run(original, model="pto", target_kind="llvm")
