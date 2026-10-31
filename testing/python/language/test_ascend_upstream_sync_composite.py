# Copyright (c) Tile-AI Organization.
# Licensed under the MIT License.
"""CPU integration checks for the audited flat AscendC gemm_v0 contract.

These tests lower existing kernels and run synchronization IR passes. They do
not compile device source or execute a kernel. Expected event reservations come
from the helper ABI, without using the legacy synchronization pass as an oracle.
"""

import pytest

import tilelang
from tilelang import tvm

from test_ascend_pipeline_ring_provenance import _nodes, _pre_sync

tir = tvm.tir
_GEMM = "tl.ascend_gemm_v0"
_RESERVED = {
    "MTE2_MTE1": {0},
    "FIX_M": {0},
    "M_MTE1": {0, 1},
    "MTE1_M": {0, 1},
    "MTE1_MTE2": {0},
    "M_FIX": {0},
}


def _gemms(function):
    return [call for call in _nodes(function.body, tir.Call) if getattr(call.op, "name", None) == _GEMM]


def _events(function):
    return [
        (call.op.name, call.args[0].value, int(call.args[1]))
        for call in _nodes(function.body, tir.Call)
        if getattr(call.op, "name", None) in ("tl.ascend_auto_set_flag", "tl.ascend_auto_wait_flag")
    ]


def _run(function, *, platform="A3", model="ascendc", target_kind="c"):
    target = tvm.target.Target(f"{target_kind} -keys=ascend -model={model}")
    with tvm.transform.PassContext(config={"tl.ascend_auto_sync": True, "tl.ascend_auto_sync_vs": True}):
        return tilelang.transform.AscendInsertSync(target, platform)(tvm.IRModule({"main": function}))["main"]


def _rewrite_gemm(function, rewrite):
    def visit(statement):
        if (
            isinstance(statement, tir.Evaluate)
            and isinstance(statement.value, tir.Call)
            and getattr(statement.value.op, "name", None) == _GEMM
        ):
            return rewrite(statement.value)
        return None

    return function.with_body(tir.stmt_functor.ir_transform(function.body, visit, None, ["tir.Evaluate"]))


def _changed_call(call, *, template=None, n_actual=None, clear=None):
    arguments = list(call.args)
    if template is not None:
        arguments[0] = tir.StringImm(template)
    if n_actual is not None:
        arguments[5] = n_actual
    if clear is not None:
        arguments[4] = clear
    return tir.Call(call.dtype, call.op, arguments, call.span)


def _manual_flag(kind, event, event_id):
    source, destination = event.split("_", 1)
    return tir.Evaluate(
        tir.Call(
            "handle",
            tvm.ir.Op.get(f"tl.ascend_{kind}_flag"),
            [tir.StringImm(source), tir.StringImm(destination), tir.IntImm("int32", event_id)],
        )
    )


@pytest.fixture(scope="module")
def flat_gemms():
    if not tvm.get_global_func("tl.ascend_sync.Z3ProofServiceAvailable")():
        pytest.skip("requires USE_ASCEND_SYNC_Z3=ON")
    # The factory has no manual flags/barriers. K=256 exercises the helper's
    # internal L0 ping-pong, while the outer TIR remains a flat single call.
    from test_tilelang_ascend_language_gemm_v0_n_tiling import gemm_nt

    return {k: _pre_sync(gemm_nt(16, 16, k, "float16", "float"))[0] for k in (64, 256)}


@pytest.mark.parametrize("k", [64, 256])
@pytest.mark.parametrize("target_kind", ["c", "llvm"])
@pytest.mark.parametrize("model", ["ascendc", "auto"])
def test_flat_real_gemm_preserves_one_calculation_and_reserves_hidden_events(flat_gemms, k, target_kind, model):
    original = flat_gemms[k]
    assert len(_gemms(original)) == 1
    assert not _nodes(original.body, tir.For)
    assert not any(
        "flag" in str(getattr(call.op, "name", "")) or "barrier" in str(getattr(call.op, "name", ""))
        for call in _nodes(original.body, tir.Call)
    )
    result = _run(original, target_kind=target_kind, model=model)
    calculations = _gemms(result)
    assert len(calculations) == 1
    tvm.ir.assert_structural_equal(calculations[0], _gemms(original)[0])
    events = _events(result)
    assert events
    assert all(event_id not in _RESERVED.get(event, set()) for _, event, event_id in events)
    assert any(event == "MTE2_MTE1" for _, event, _ in events)
    assert any(event == "M_FIX" for _, event, _ in events)


def test_flat_accumulating_gemm_preserves_c_read_write_mask(flat_gemms):
    call = _gemms(flat_gemms[64])[0]
    pointer = call.args[3]
    arguments = list(pointer.args)
    arguments[4] = tir.IntImm("int32", 3)
    c_read_write = tir.Call(pointer.dtype, pointer.op, arguments, pointer.span)

    def rewrite(old):
        arguments = list(old.args)
        arguments[3] = c_read_write
        arguments[4] = tir.const(False, "bool")
        accumulating = tir.Call(old.dtype, old.op, arguments, old.span)
        return tir.SeqStmt([tir.Evaluate(old), tir.Evaluate(accumulating)])

    original = _rewrite_gemm(flat_gemms[64], rewrite)
    result = _run(original)
    assert len(_gemms(result)) == 2
    assert int(_gemms(result)[1].args[3].args[4]) == 3
    for calculation, expected in zip(_gemms(result), _gemms(original)):
        tvm.ir.assert_structural_equal(calculation, expected)
    assert any(event == "M_MTE1" and event_id not in (0, 1) for _, event, event_id in _events(result))


def test_real_int8_gemm_accepts_the_emitted_int_accumulator_alias():
    if not tvm.get_global_func("tl.ascend_sync.Z3ProofServiceAvailable")():
        pytest.skip("requires USE_ASCEND_SYNC_Z3=ON")
    from test_tilelang_ascend_language_gemm_v0_n_tiling import gemm_nt

    original = _pre_sync(gemm_nt(32, 32, 64, "int8", "int32"))[0]
    call = _gemms(original)[0]
    assert call.args[0].value.startswith("gemm_v0<int8_t, int,")
    result = _run(original)
    assert len(_gemms(result)) == 1
    tvm.ir.assert_structural_equal(_gemms(result)[0], call)
    assert any(event == "MTE2_MTE1" for _, event, _ in _events(result))
    assert any(event == "M_FIX" for _, event, _ in _events(result))


def test_int_accumulator_alias_still_requires_matching_pointer_dtype(flat_gemms):
    original = _rewrite_gemm(
        flat_gemms[64],
        lambda call: tir.Evaluate(_changed_call(call, template="gemm_v0<int8_t, int, 16, 16, 64, false, false, 64>")),
    )
    with pytest.raises(tvm.error.TVMError, match="pointer dtype disagrees"):
        _run(original)


@pytest.mark.parametrize(
    "event,event_id",
    [
        ("MTE2_MTE1", 0),
        ("FIX_M", 0),
        ("M_MTE1", 0),
        ("M_MTE1", 1),
        ("MTE1_M", 0),
        ("MTE1_M", 1),
        ("MTE1_MTE2", 0),
        ("M_FIX", 0),
    ],
)
def test_existing_manual_flags_cannot_borrow_helper_owned_ids(flat_gemms, event, event_id):
    original = _rewrite_gemm(
        flat_gemms[64],
        lambda call: tir.SeqStmt(
            [
                _manual_flag("set", event, event_id),
                _manual_flag("wait", event, event_id),
                tir.Evaluate(call),
            ]
        ),
    )
    with pytest.raises(tvm.error.TVMError, match="(?i)(composite.*hidden|hidden.*event|conflict)"):
        _run(original)


def test_nonconflicting_manual_id_stays_reserved_for_generated_flags(flat_gemms):
    original = _rewrite_gemm(
        flat_gemms[64],
        lambda call: tir.SeqStmt(
            [
                _manual_flag("set", "MTE2_MTE1", 7),
                _manual_flag("wait", "MTE2_MTE1", 7),
                tir.Evaluate(call),
            ]
        ),
    )
    result = _run(original)
    ids = [event_id for _, event, event_id in _events(result) if event == "MTE2_MTE1"]
    assert ids
    assert all(event_id not in (0, 7) for event_id in ids)


def test_external_k_loop_is_rejected_until_separate_hardware_acceptance(flat_gemms):
    original = _rewrite_gemm(
        flat_gemms[64],
        lambda call: tir.For(tir.Var("outer_k", "int32"), 0, 2, tir.ForKind.SERIAL, tir.Evaluate(call)),
    )
    with pytest.raises(tvm.error.TVMError, match="(?i)(composite.*For|external.*loop)"):
        _run(original)


def test_software_pipeline_annotation_is_not_treated_as_a_flat_helper(flat_gemms):
    original = _rewrite_gemm(
        flat_gemms[64],
        lambda call: tir.For(
            tir.Var("pipeline_k", "int32"),
            0,
            2,
            tir.ForKind.SERIAL,
            tir.Evaluate(call),
            annotations={"num_stages": 2},
        ),
    )
    with pytest.raises(tvm.error.TVMError, match="(?i)(composite.*pipeline|Cube-ring)"):
        _run(original)


def test_cube_ring_version_metadata_is_not_discarded(flat_gemms):
    call = _gemms(flat_gemms[64])[0]
    original = flat_gemms[64].with_attr("buffer_versions", {call.args[1].args[1]: 2})
    with pytest.raises(tvm.error.TVMError, match="(?i)(composite.*(ring|version)|B>1)"):
        _run(original)


@pytest.mark.parametrize("control", ["if", "while"])
def test_external_conditional_control_is_outside_the_first_flat_contract(flat_gemms, control):
    loop_enabled = tir.Var("loop_enabled", "bool")

    def rewrite(call):
        if control == "if":
            return tir.IfThenElse(tir.const(True, "bool"), tir.Evaluate(call), None)
        return tir.While(loop_enabled, tir.Evaluate(call))

    original = _rewrite_gemm(flat_gemms[64], rewrite)
    if control == "while":
        original = tir.PrimFunc(
            [*original.params, loop_enabled],
            original.body,
            original.ret_type,
            original.buffer_map,
            original.attrs,
        )
    with pytest.raises(tvm.error.TVMError, match="(?i)(composite.*If/While|contract.*flat)"):
        _run(original)


def test_user_l0ab_pool_cannot_alias_hidden_scratch(flat_gemms):
    user_l0 = tir.decl_buffer((64,), "float16", name="user_l0", scope="wmma.matrix_a")
    original = flat_gemms[64].with_body(tir.Allocate(user_l0.data, user_l0.dtype, [64], tir.const(True, "bool"), flat_gemms[64].body))
    with pytest.raises(tvm.error.TVMError, match="(?i)(hidden.*L0|user.*L0|pool)"):
        _run(original)


def test_bare_mma_mixing_requires_an_explicit_scratch_owner_model(flat_gemms):
    original = _rewrite_gemm(
        flat_gemms[64],
        lambda call: tir.SeqStmt(
            [
                tir.Evaluate(call),
                tir.Evaluate(
                    tir.Call(
                        "handle",
                        tvm.ir.Op.get("tl.ascend_mma"),
                        [
                            tir.StringImm("mma<half, float, 16, 16>"),
                            *list(call.args[1:4]),
                            tir.const(True, "bool"),
                            tir.IntImm("int32", 64),
                        ],
                    )
                ),
            ]
        ),
    )
    with pytest.raises(tvm.error.TVMError, match="(?i)(bare mma|hidden.*L0|pool)"):
        _run(original)


@pytest.mark.parametrize("platform,model", [("A2", "ascendc"), ("A5", "ascendc"), ("A2", "pto"), ("A5", "pto")])
def test_composite_target_contract_is_explicit(flat_gemms, platform, model):
    with pytest.raises(
        tvm.error.TVMError,
        match="(?i)(composite.*AscendC|AscendC.*A3|PTO flat gemm_v0 profile)",
    ):
        _run(flat_gemms[64], platform=platform, model=model)


def test_composite_accepts_a3_pto_semantic_contract(flat_gemms):
    original = flat_gemms[64]
    result = _run(original, model="pto", target_kind="llvm")
    assert result.attrs.get("tl.ascend_sync_proof") is not None
    assert len(_gemms(result)) == 1
    tvm.ir.assert_structural_equal(_gemms(original)[0], _gemms(result)[0])
    tvm.ir.assert_structural_equal(result, _run(result, model="pto", target_kind="llvm"))


@pytest.mark.parametrize(
    "template",
    [
        "gemm_v0<half, float, 0, 16, 64, false, false, 128>",
        "gemm_v0<half, float, dynamic_m, 16, 64, false, false, 128>",
    ],
)
def test_dynamic_or_nonpositive_template_shape_is_rejected(flat_gemms, template):
    original = _rewrite_gemm(flat_gemms[64], lambda call: tir.Evaluate(_changed_call(call, template=template)))
    with pytest.raises(tvm.error.TVMError, match="(?i)(static positive|static.*shape)"):
        _run(original)


def test_partial_n_actual_requires_an_independent_helper_contract(flat_gemms):
    original = _rewrite_gemm(
        flat_gemms[64],
        lambda call: tir.Evaluate(_changed_call(call, n_actual=tir.IntImm("int32", 8))),
    )
    with pytest.raises(tvm.error.TVMError, match="n_actual"):
        _run(original)


def test_nonzero_pointer_view_requires_an_independent_physical_footprint(flat_gemms):
    def rewrite(call):
        arguments = list(call.args)
        pointer = arguments[1]
        pointer_arguments = list(pointer.args)
        pointer_arguments[2] = tir.IntImm("int32", 16)
        arguments[1] = tir.Call(pointer.dtype, pointer.op, pointer_arguments, pointer.span)
        return tir.Evaluate(tir.Call(call.dtype, call.op, arguments, call.span))

    with pytest.raises(tvm.error.TVMError, match="(?i)(base allocation|subviews|footprint)"):
        _run(_rewrite_gemm(flat_gemms[64], rewrite))


def test_undersized_physical_allocation_is_not_trusted(flat_gemms):
    original = flat_gemms[64]
    sizes = dict(original.attrs["size_map"].items())
    sizes[_gemms(original)[0].args[1].args[1]] = 32
    with pytest.raises(tvm.error.TVMError, match="(?i)(padded physical footprint|allocation)"):
        _run(original.with_attr("size_map", sizes))


def test_gemm_execution_owner_cannot_be_vector(flat_gemms):
    original = _rewrite_gemm(
        flat_gemms[64],
        lambda call: tir.AttrStmt(tir.IntImm("int32", 0), "resource_scope", 1, tir.Evaluate(call)),
    )
    with pytest.raises(tvm.error.TVMError, match="(?i)(owner.*Cube|resource_scope)"):
        _run(original)


def test_untyped_extern_gemm_is_not_accepted_by_name(flat_gemms):
    original = _rewrite_gemm(
        flat_gemms[64],
        lambda call: tir.Evaluate(tir.call_extern("handle", "tl::ascend::" + call.args[0].value, *list(call.args[1:]))),
    )
    with pytest.raises(tvm.error.TVMError, match="(?i)(canonical typed|composite)"):
        _run(original)


def test_gemm_v1_stays_rejected_without_an_audited_helper(flat_gemms):
    original = _rewrite_gemm(
        flat_gemms[64],
        lambda call: tir.Evaluate(tir.Call("handle", tvm.ir.Op.get("tl.ascend_gemm_v1"), list(call.args[:5]))),
    )
    with pytest.raises(tvm.error.TVMError, match="(?i)(gemm_v1|no audited helper)"):
        _run(original)
