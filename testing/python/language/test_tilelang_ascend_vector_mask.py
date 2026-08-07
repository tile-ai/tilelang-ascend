"""Compiler-managed A2/A3 Ascend Vector-mask tests."""

from __future__ import annotations

import random

import pytest

import tilelang
import tilelang.language as T
from tilelang import tvm
from tvm import IRModule, tir
from tvm.ir import Op
from tvm.target import Target


ASCENDC = Target({"kind": "llvm", "model": "ascendc"})
PTO = Target({"kind": "llvm", "model": "pto"})


def _calls(node) -> list[tir.Call]:
    result = []

    def visit(current):
        if isinstance(current, tir.Call):
            result.append(current)

    tir.stmt_functor.post_order_visit(node, visit)
    return result


def _op_name(call: tir.Call) -> str | None:
    return call.op.name if isinstance(call.op, Op) else None


def _pass(func, transform, target=ASCENDC, platform="A2"):
    return transform(target, platform)(IRModule({"main": func}))["main"]


@T.prim_func
def _two_adds(
    a: T.Tensor((128,), "float32"),
    b: T.Tensor((128,), "float32"),
    c: T.Tensor((128,), "float32"),
):
    with T.Kernel(1, threads=1, is_npu=True) as _cid:
        a_ub = T.alloc_ub((128,), "float32")
        b_ub = T.alloc_ub((128,), "float32")
        c_ub = T.alloc_ub((128,), "float32")
        T.copy(a, a_ub)
        T.copy(b, b_ub)
        T.tile.add(c_ub, a_ub, b_ub)
        T.tile.add(a_ub, c_ub, b_ub)
        T.copy(c_ub, c)


@T.prim_func
def _full_normal_reduce(
    source: T.Tensor((128,), "float16"),
    result: T.Tensor((8,), "float16"),
):
    with T.Kernel(1, threads=1, is_npu=True) as _cid:
        source_ub = T.alloc_ub((128,), "float16")
        result_ub = T.alloc_ub((8,), "float16")
        T.copy(source, source_ub)
        T.tile.block_reduce_max(result_ub, source_ub, 1, 128, 1, 1, 8)
        T.copy(result_ub, result)


def test_selection_uses_fixed_selected_identity_and_preserves_operands():
    selected = _pass(
        _two_adds,
        tilelang.transform.AscendVectorInstructionSelection,
    )
    selected_calls = [call for call in _calls(selected.body) if _op_name(call) == "tl.ascend_add_raw_counter"]
    assert len(selected_calls) == 2
    assert all(len(call.args) == 4 for call in selected_calls)
    assert all(tvm.ir.structural_equal(call.args[-1], tir.IntImm("int32", 128)) for call in selected_calls)
    assert not any(_op_name(call) == "tl.ascend_add" for call in _calls(selected.body))


def test_selection_is_strictly_gated_to_a2_a3_ascendc():
    for target, platform in [(ASCENDC, "A5"), (PTO, "A2")]:
        result = _pass(
            _two_adds,
            tilelang.transform.AscendVectorInstructionSelection,
            target,
            platform,
        )
        names = {_op_name(call) for call in _calls(result.body)}
        assert "tl.ascend_add" in names
        assert "tl.ascend_add_raw_counter" not in names


def test_selection_accepts_symbolic_counter_from_dynamic_buffer_shape():
    length = tir.Var("length", "int32")
    buffer = tir.decl_buffer((length,), "float32", name="buffer")
    semantic = tir.Call(
        "handle",
        Op.get("tl.ascend_add"),
        [tir.IntImm("int32", 0)] * 3 + [length],
    )
    func = tir.PrimFunc(
        [length, buffer.data],
        tir.Evaluate(semantic),
        buffer_map={buffer.data: buffer},
    ).with_attr("global_symbol", "main")
    selected = _pass(func, tilelang.transform.AscendVectorInstructionSelection)
    call = next(call for call in _calls(selected.body) if _op_name(call) == "tl.ascend_add_raw_counter")
    assert tvm.ir.structural_equal(call.args[-1], length)


def test_selection_keeps_buffer_derived_let_counter_as_white_box_var():
    source = tir.decl_buffer((1,), "int32", name="source")
    length = tir.Var("length", "int32")
    value = tir.if_then_else(
        tir.BufferLoad(source, [0]) > 0,
        tir.IntImm("int32", 32),
        tir.IntImm("int32", 64),
    )
    semantic = tir.Call(
        "handle",
        Op.get("tl.ascend_add"),
        [tir.IntImm("int32", 0)] * 3 + [length],
    )
    func = tir.PrimFunc(
        [source.data],
        tir.LetStmt(length, value, tir.Evaluate(semantic)),
        buffer_map={source.data: source},
    ).with_attr("global_symbol", "main")

    selected = _pass(func, tilelang.transform.AscendVectorInstructionSelection)
    call = next(call for call in _calls(selected.body) if _op_name(call) == "tl.ascend_add_raw_counter")
    assert tvm.ir.structural_equal(call.args[-1], length)


def test_selection_rejects_symbolic_normal_payload():
    length = tir.Var("length", "uint32")
    semantic = tir.Call(
        "handle",
        Op.get("tl.ascend_fill_experiment"),
        [tir.IntImm("int32", 0)] * 3 + [length],
    )
    func = tir.PrimFunc([length], tir.Evaluate(semantic)).with_attr("global_symbol", "main")
    with pytest.raises(tvm.TVMError, match="NORMAL mask payload"):
        _pass(func, tilelang.transform.AscendVectorInstructionSelection)


def test_selection_accepts_large_uint_encoding_of_full_normal_mask():
    src = tir.decl_buffer((128,), "float16", name="src")
    dst = tir.decl_buffer((8,), "float16", name="dst")
    semantic = tir.Call(
        "handle",
        Op.get("tl.ascend_block_reduce_max"),
        [
            dst.access_ptr("w"),
            src.access_ptr("r"),
            tir.IntImm("int32", 1),
            tir.IntImm("int32", 128),
            tir.IntImm("int32", 1),
            tir.IntImm("int32", 1),
            tir.IntImm("int32", 8),
        ],
    )
    func = tir.PrimFunc(
        [src.data, dst.data],
        tir.Evaluate(semantic),
        buffer_map={src.data: src, dst.data: dst},
    ).with_attr("global_symbol", "main")

    selected = _pass(func, tilelang.transform.AscendVectorInstructionSelection)
    call = next(call for call in _calls(selected.body) if _op_name(call) == "tl.ascend_block_reduce_max_raw_normal")
    assert _op_name(call.args[-2]) == "tir.large_uint_imm"
    assert _op_name(call.args[-1]) == "tir.large_uint_imm"
    legalized = _pass(selected, tilelang.transform.AscendVectorMaskLegalize)
    _assert_consumer_contracts(legalized.body)


def test_selection_classifies_gather_mask_by_pattern_not_optional_tmp():
    src = tir.decl_buffer((128,), "float16", name="src")
    dst = tir.decl_buffer((128,), "float16", name="dst")
    pattern = tir.decl_buffer((8,), "uint32", name="pattern")
    tmp = tir.decl_buffer((256,), "uint8", name="tmp")
    common = [
        tir.StringImm("GatherMask<half>"),
        dst.access_ptr("w"),
        src.access_ptr("r"),
    ]
    fixed = tir.Evaluate(
        tir.Call(
            "handle",
            Op.get("tl.ascend_gather_mask"),
            [*common, tir.StringImm("P0101"), tmp.access_ptr("w")],
        )
    )
    custom = tir.Evaluate(
        tir.Call(
            "handle",
            Op.get("tl.ascend_gather_mask"),
            [*common, pattern.access_ptr("r"), tmp.access_ptr("w")],
        )
    )
    func = tir.PrimFunc(
        [src.data, dst.data, pattern.data, tmp.data],
        tir.SeqStmt([fixed, custom]),
        buffer_map={
            src.data: src,
            dst.data: dst,
            pattern.data: pattern,
            tmp.data: tmp,
        },
    ).with_attr("global_symbol", "main")

    selected = _pass(func, tilelang.transform.AscendVectorInstructionSelection)
    names = _managed_names(selected.body)
    assert names.count("tl.ascend_gather_mask_fixed_self_contained_normal") == 1
    assert names.count("tl.ascend_gather_mask_custom_composite") == 1


def test_selection_rejects_unclassified_extern_and_nested_semantic_call():
    unknown = tir.Evaluate(tir.call_extern("handle", "unreviewed_helper"))
    unknown_func = tir.PrimFunc([], unknown).with_attr("global_symbol", "main")
    with pytest.raises(tvm.TVMError, match="effect table"):
        _pass(unknown_func, tilelang.transform.AscendVectorInstructionSelection)

    semantic = tir.Call(
        "handle",
        Op.get("tl.ascend_add"),
        [tir.IntImm("int32", 0)] * 4,
    )
    nested = tir.Evaluate(tir.Call("handle", Op.get("tir.tvm_tuple"), [semantic]))
    nested_func = tir.PrimFunc([], nested).with_attr("global_symbol", "main")
    with pytest.raises(tvm.TVMError, match="direct value of an Evaluate"):
        _pass(nested_func, tilelang.transform.AscendVectorInstructionSelection)


def test_selection_requires_mask_effects_to_stay_in_aiv_resource_scope():
    resource = tir.Var("resource", "int32")
    semantic = tir.Evaluate(
        tir.Call(
            "handle",
            Op.get("tl.ascend_add"),
            [tir.IntImm("int32", 0)] * 4,
        )
    )

    aic_only = tir.PrimFunc(
        [],
        tir.AttrStmt(resource, "resource_scope", 0, semantic),
    ).with_attr("global_symbol", "main")
    with pytest.raises(tvm.TVMError, match="resource_scope=1"):
        _pass(aic_only, tilelang.transform.AscendVectorInstructionSelection)

    outside = tir.PrimFunc(
        [],
        tir.SeqStmt(
            [
                tir.AttrStmt(resource, "resource_scope", 1, tir.Evaluate(0)),
                semantic,
            ]
        ),
    ).with_attr("global_symbol", "main")
    with pytest.raises(tvm.TVMError, match="resource_scope=1"):
        _pass(outside, tilelang.transform.AscendVectorInstructionSelection)


def test_selection_rejects_overlapping_aic_aiv_resource_scopes():
    resource = tir.Var("resource", "int32")
    nested = tir.AttrStmt(
        resource,
        "resource_scope",
        1,
        tir.AttrStmt(resource, "resource_scope", 0, tir.Evaluate(0)),
    )
    func = tir.PrimFunc([], nested).with_attr("global_symbol", "main")
    with pytest.raises(tvm.TVMError, match="overlapping AIC/AIV"):
        _pass(func, tilelang.transform.AscendVectorInstructionSelection)


def test_legalizer_reuses_counter_mode_and_payload_in_straight_line():
    selected = _pass(
        _two_adds,
        tilelang.transform.AscendVectorInstructionSelection,
    )
    legalized = _pass(selected, tilelang.transform.AscendVectorMaskLegalize)
    names = [_op_name(call) for call in _calls(legalized.body)]
    assert names.count("tl.ascend_set_mask_mode") == 1
    assert names.count("tl.ascend_set_mask_payload") == 1
    assert names.count("tl.ascend_add_raw_counter") == 2


def _selected_counter(length: tir.PrimExpr) -> tir.Stmt:
    args = [tir.IntImm("int32", 0)] * 3 + [length]
    return tir.Evaluate(tir.Call("handle", Op.get("tl.ascend_add_raw_counter"), args))


def _selected_normal(lo: int, hi: int) -> tir.Stmt:
    args = [tir.IntImm("int32", 0)] * 4
    args.extend([tir.IntImm("uint64", lo), tir.IntImm("uint64", hi)])
    return tir.Evaluate(tir.Call("handle", Op.get("tl.ascend_block_reduce_max_raw_normal"), args))


def _legalize_stmt(stmt: tir.Stmt) -> tir.Stmt:
    func = tir.PrimFunc([], stmt).with_attr("global_symbol", "main")
    return _pass(func, tilelang.transform.AscendVectorMaskLegalize).body


def _managed_names(stmt: tir.Stmt) -> list[str | None]:
    return [_op_name(call) for call in _calls(stmt)]


def _same_fact(lhs, rhs) -> bool:
    if lhs is None or rhs is None:
        return lhs is rhs
    return tvm.ir.structural_equal(lhs, rhs)


def _merge_facts(lhs, rhs):
    return tuple(left if _same_fact(left, right) else None for left, right in zip(lhs, rhs))


def _assert_consumer_contracts(stmt: tir.Stmt, incoming=(None, None, None)):
    """Independently validate required mask facts in legalized test IR."""
    if isinstance(stmt, tir.SeqStmt):
        facts = incoming
        for child in stmt.seq:
            facts = _assert_consumer_contracts(child, facts)
        return facts
    if isinstance(stmt, tir.IfThenElse):
        then_out = _assert_consumer_contracts(stmt.then_case, incoming)
        else_out = incoming
        if stmt.else_case is not None:
            else_out = _assert_consumer_contracts(stmt.else_case, incoming)
        return _merge_facts(then_out, else_out)
    if isinstance(stmt, tir.For):
        effect_names = {
            "tl.ascend_add_raw_counter",
            "tl.ascend_block_reduce_max_raw_normal",
            "tl.ascend_src_code",
        }
        has_effect = any(_op_name(call) in effect_names for call in _calls(stmt.body))
        if not has_effect:
            _assert_consumer_contracts(stmt.body, incoming)
            return incoming
        _assert_consumer_contracts(stmt.body)
        return (None, None, None)
    if not isinstance(stmt, tir.Evaluate) or not isinstance(stmt.value, tir.Call):
        return incoming

    call = stmt.value
    name = _op_name(call)
    if name == "tl.ascend_set_mask_mode":
        mode = "counter" if call.args[0].value == 1 else "normal"
        return (mode, incoming[1], incoming[2])
    if name == "tl.ascend_set_mask_payload":
        return (incoming[0], call.args[0], call.args[1])
    if name == "tl.ascend_add_raw_counter":
        lo = tvm.arith.Analyzer().simplify(tir.Cast("uint64", call.args[-1]))
        required = ("counter", lo, tir.IntImm("uint64", 0))
    elif name == "tl.ascend_block_reduce_max_raw_normal":
        required = ("normal", call.args[-2], call.args[-1])
    else:
        return incoming
    assert all(_same_fact(actual, expected) for actual, expected in zip(incoming, required))
    return required


def test_mode_switch_keeps_payload_facts_and_repairs_only_required_fields():
    body = tir.SeqStmt(
        [
            _selected_counter(tir.IntImm("int32", 64)),
            _selected_normal(64, 0),
            _selected_counter(tir.IntImm("int32", 64)),
        ]
    )
    names = _managed_names(_legalize_stmt(body))
    assert names.count("tl.ascend_set_mask_mode") == 3
    assert names.count("tl.ascend_set_mask_payload") == 1


def test_equal_if_branches_preserve_must_facts_but_missing_else_does_not():
    cond = tir.Var("cond", "bool")
    warm = _selected_counter(tir.IntImm("int32", 64))
    equal_if = tir.IfThenElse(
        cond,
        _selected_counter(tir.IntImm("int32", 64)),
        _selected_counter(tir.IntImm("int32", 64)),
    )
    equal_body = tir.SeqStmt([warm, equal_if, _selected_counter(tir.IntImm("int32", 64))])
    equal_names = _managed_names(_legalize_stmt(equal_body))
    assert equal_names.count("tl.ascend_set_mask_payload") == 1

    missing_if = tir.IfThenElse(cond, _selected_counter(tir.IntImm("int32", 32)), None)
    missing_body = tir.SeqStmt([warm, missing_if, _selected_counter(tir.IntImm("int32", 64))])
    missing_names = _managed_names(_legalize_stmt(missing_body))
    assert missing_names.count("tl.ascend_set_mask_payload") == 3


def test_src_code_is_a_mask_fact_barrier():
    src_code = tir.Evaluate(tir.Call("handle", Op.get("tl.ascend_src_code"), [tir.StringImm("// opaque")]))
    body = tir.SeqStmt(
        [
            _selected_counter(tir.IntImm("int32", 64)),
            src_code,
            _selected_counter(tir.IntImm("int32", 64)),
        ]
    )
    names = _managed_names(_legalize_stmt(body))
    assert names.count("tl.ascend_set_mask_mode") == 2
    assert names.count("tl.ascend_set_mask_payload") == 2


def test_symbolic_counter_payload_stays_in_selected_ir_and_setter():
    length = tir.SizeVar("length", "int32")
    func = tir.PrimFunc([length], _selected_counter(length)).with_attr("global_symbol", "main")
    legalized = _pass(func, tilelang.transform.AscendVectorMaskLegalize)
    calls = _calls(legalized.body)
    selected = next(call for call in calls if _op_name(call) == "tl.ascend_add_raw_counter")
    setter = next(call for call in calls if _op_name(call) == "tl.ascend_set_mask_payload")
    assert tvm.ir.structural_equal(selected.args[-1], length)
    assert tvm.ir.structural_equal(setter.args[0], tir.Cast("uint64", length))


def test_legalizer_rejects_buffer_backed_payload():
    buffer = tir.decl_buffer((1,), "int32", name="buffer")
    payload = tir.BufferLoad(buffer, [0])
    func = tir.PrimFunc(
        [buffer.data],
        _selected_counter(payload),
        buffer_map={buffer.data: buffer},
    ).with_attr("global_symbol", "main")
    with pytest.raises(tvm.TVMError, match="buffer-backed"):
        _pass(func, tilelang.transform.AscendVectorMaskLegalize)


def test_let_projection_substitutes_only_pure_parent_scope_values():
    let_var = tir.Var("let_var", "int32")
    pure = tir.SeqStmt(
        [
            tir.LetStmt(
                let_var,
                tir.IntImm("int32", 64),
                _selected_counter(let_var),
            ),
            _selected_counter(tir.IntImm("int32", 64)),
        ]
    )
    pure_names = _managed_names(_legalize_stmt(pure))
    assert pure_names.count("tl.ascend_set_mask_payload") == 1

    buffer = tir.decl_buffer((1,), "int32", name="buffer")
    impure = tir.SeqStmt(
        [
            tir.LetStmt(
                let_var,
                tir.BufferLoad(buffer, [0]),
                _selected_counter(let_var),
            ),
            _selected_counter(tir.IntImm("int32", 64)),
        ]
    )
    func = tir.PrimFunc([buffer.data], impure, buffer_map={buffer.data: buffer}).with_attr("global_symbol", "main")
    names = _managed_names(_pass(func, tilelang.transform.AscendVectorMaskLegalize).body)
    assert names.count("tl.ascend_set_mask_payload") == 2


def test_loop_projection_drops_variant_facts_but_neutral_loop_preserves_them():
    loop_var = tir.Var("i", "int32")
    variant_loop = tir.For(
        loop_var,
        0,
        4,
        tir.ForKind.SERIAL,
        _selected_counter(loop_var + 1),
    )
    variant_body = tir.SeqStmt([variant_loop, _selected_counter(tir.IntImm("int32", 4))])
    variant_names = _managed_names(_legalize_stmt(variant_body))
    assert variant_names.count("tl.ascend_set_mask_mode") == 2
    assert variant_names.count("tl.ascend_set_mask_payload") == 2

    neutral_loop = tir.For(loop_var, 0, 4, tir.ForKind.SERIAL, tir.Evaluate(0))
    neutral_body = tir.SeqStmt(
        [
            _selected_counter(tir.IntImm("int32", 4)),
            neutral_loop,
            _selected_counter(tir.IntImm("int32", 4)),
        ]
    )
    neutral_names = _managed_names(_legalize_stmt(neutral_body))
    assert neutral_names.count("tl.ascend_set_mask_mode") == 1
    assert neutral_names.count("tl.ascend_set_mask_payload") == 1


def test_aiv_resource_scope_has_independent_entry_facts():
    resource = tir.Var("resource", "int32")
    first = tir.AttrStmt(
        resource,
        "resource_scope",
        1,
        _selected_counter(tir.IntImm("int32", 64)),
    )
    second = tir.AttrStmt(
        resource,
        "resource_scope",
        1,
        _selected_counter(tir.IntImm("int32", 64)),
    )
    names = _managed_names(_legalize_stmt(tir.SeqStmt([first, second])))
    assert names.count("tl.ascend_set_mask_mode") == 2
    assert names.count("tl.ascend_set_mask_payload") == 2


def test_legalizer_rejects_preexisting_internal_setter():
    setter = tir.Evaluate(
        tir.Call(
            "handle",
            Op.get("tl.ascend_set_mask_mode"),
            [tir.IntImm("int32", 1)],
        )
    )
    with pytest.raises(tvm.TVMError, match="already contains"):
        _legalize_stmt(setter)


@pytest.mark.parametrize("seed", range(20))
def test_random_straight_line_setter_count_matches_reference(seed):
    rng = random.Random(seed)
    lengths = [rng.choice([16, 32, 64, 128]) for _ in range(25)]
    body = tir.SeqStmt([_selected_counter(tir.IntImm("int32", length)) for length in lengths])
    legalized = _legalize_stmt(body)
    names = _managed_names(legalized)
    reference_payload_repairs = 1 + sum(a != b for a, b in zip(lengths, lengths[1:]))
    assert names.count("tl.ascend_set_mask_mode") == 1
    assert names.count("tl.ascend_set_mask_payload") == reference_payload_repairs
    _assert_consumer_contracts(legalized)

    before = [call for call in _calls(body) if _op_name(call) == "tl.ascend_add_raw_counter"]
    after = [call for call in _calls(legalized) if _op_name(call) == "tl.ascend_add_raw_counter"]
    assert len(before) == len(after)
    assert all(tvm.ir.structural_equal(lhs, rhs) for lhs, rhs in zip(before, after))


def _random_control_spec(rng: random.Random, depth: int, ordinal: list[int]):
    if depth == 0:
        if rng.randrange(4) == 0:
            return ("neutral",)
        return ("counter", rng.choice([16, 32, 64, 128]))
    kind = rng.choice(["counter", "seq", "if", "for"])
    if kind == "counter":
        return ("counter", rng.choice([16, 32, 64, 128]))
    ordinal[0] += 1
    current = ordinal[0]
    if kind == "seq":
        return (
            "seq",
            [_random_control_spec(rng, depth - 1, ordinal) for _ in range(rng.randint(1, 3))],
        )
    if kind == "if":
        else_spec = None
        if rng.randrange(3) != 0:
            else_spec = _random_control_spec(rng, depth - 1, ordinal)
        return (
            "if",
            current,
            _random_control_spec(rng, depth - 1, ordinal),
            else_spec,
        )
    return ("for", current, _random_control_spec(rng, depth - 1, ordinal))


def _control_spec_to_stmt(spec):
    kind = spec[0]
    if kind == "neutral":
        return tir.Evaluate(0)
    if kind == "counter":
        return _selected_counter(tir.IntImm("int32", spec[1]))
    if kind == "seq":
        statements = [_control_spec_to_stmt(item) for item in spec[1]]
        return statements[0] if len(statements) == 1 else tir.SeqStmt(statements)
    if kind == "if":
        else_case = None if spec[3] is None else _control_spec_to_stmt(spec[3])
        return tir.IfThenElse(
            tir.Var(f"condition_{spec[1]}", "bool"),
            _control_spec_to_stmt(spec[2]),
            else_case,
        )
    loop_var = tir.Var(f"loop_{spec[1]}", "int32")
    return tir.For(
        loop_var,
        0,
        3,
        tir.ForKind.SERIAL,
        _control_spec_to_stmt(spec[2]),
    )


def _reference_control(spec, incoming):
    kind = spec[0]
    if kind == "neutral":
        return (0, 0), incoming, False
    if kind == "counter":
        length = spec[1]
        mode_repairs = int(incoming[0] != "counter")
        payload_repairs = int(incoming[1] != length)
        return (mode_repairs, payload_repairs), ("counter", length, 0), True
    if kind == "seq":
        counts = [0, 0]
        facts = incoming
        has_effect = False
        for item in spec[1]:
            item_counts, facts, item_effect = _reference_control(item, facts)
            counts[0] += item_counts[0]
            counts[1] += item_counts[1]
            has_effect |= item_effect
        return tuple(counts), facts, has_effect
    if kind == "if":
        then_counts, then_out, then_effect = _reference_control(spec[2], incoming)
        if spec[3] is None:
            else_counts, else_out, else_effect = (0, 0), incoming, False
        else:
            else_counts, else_out, else_effect = _reference_control(spec[3], incoming)
        merged = tuple(lhs if lhs == rhs else None for lhs, rhs in zip(then_out, else_out))
        return (
            (then_counts[0] + else_counts[0], then_counts[1] + else_counts[1]),
            merged,
            then_effect or else_effect,
        )
    _, _, body = spec
    _, _, has_effect = _reference_control(body, incoming)
    if not has_effect:
        return (0, 0), incoming, False
    body_counts, _, _ = _reference_control(body, (None, None, None))
    return body_counts, (None, None, None), True


@pytest.mark.parametrize("seed", range(20))
def test_random_structured_setter_counts_match_reference(seed):
    spec = _random_control_spec(random.Random(seed), 3, [0])
    expected, _, _ = _reference_control(spec, (None, None, None))
    selected = _control_spec_to_stmt(spec)
    legalized = _legalize_stmt(selected)
    names = _managed_names(legalized)
    assert names.count("tl.ascend_set_mask_mode") == expected[0]
    assert names.count("tl.ascend_set_mask_payload") == expected[1]
    _assert_consumer_contracts(legalized)

    before = [call for call in _calls(selected) if _op_name(call) == "tl.ascend_add_raw_counter"]
    after = [call for call in _calls(legalized) if _op_name(call) == "tl.ascend_add_raw_counter"]
    assert len(before) == len(after)
    assert all(tvm.ir.structural_equal(lhs, rhs) for lhs, rhs in zip(before, after))


def test_full_pipeline_source_has_one_repair_for_two_equal_consumers():
    kernel = tilelang.compile(_two_adds, target="ascendc", platform="A2", out_idx=[2])
    source = kernel.get_kernel_source()
    assert source.count("AscendC::SetMaskCount();") == 1
    assert source.count("AscendC::SetVectorMask<uint8_t>((uint64_t)0, (uint64_t)128);") == 1
    assert source.count("AscendC::Add<float, false>") == 2
    assert "AscendC::Add(c_ub, a_ub, b_ub, 128)" not in source


def test_full_normal_payload_codegen_uses_unsigned_64_bit_literals():
    kernel = tilelang.compile(_full_normal_reduce, target="ascendc", platform="A2", out_idx=[1])
    source = kernel.get_kernel_source()
    assert source.count("AscendC::SetVectorMask<uint8_t>(0xffffffffffffffffULL, 0xffffffffffffffffULL);") == 1


def test_selected_identity_survives_phase2_and_legalizer_is_last_tir_pass(monkeypatch):
    snapshots = {}
    production_legalizer = tilelang.transform.AscendVectorMaskLegalize

    @tvm.transform.module_pass(opt_level=0)
    def capture_before(mod, _ctx):
        snapshots["before"] = mod
        return mod

    @tvm.transform.module_pass(opt_level=0)
    def capture_after(mod, _ctx):
        snapshots["after"] = mod
        return mod

    def captured_legalizer(target, platform):
        return tvm.transform.Sequential([capture_before, production_legalizer(target, platform), capture_after])

    monkeypatch.setattr(tilelang.transform, "AscendVectorMaskLegalize", captured_legalizer)
    artifact = tilelang.lower(_two_adds, target="ascendc", platform="A2")

    before_func = snapshots["before"].functions_items()[0][1]
    after_func = snapshots["after"].functions_items()[0][1]

    before_calls = [call for call in _calls(before_func.body) if _op_name(call) == "tl.ascend_add_raw_counter"]
    after_calls = [call for call in _calls(after_func.body) if _op_name(call) == "tl.ascend_add_raw_counter"]
    assert len(before_calls) == len(after_calls) == 2
    assert all(tvm.ir.structural_equal(lhs, rhs) for lhs, rhs in zip(before_calls, after_calls))
    assert all(call.args[-1].value == 128 for call in before_calls)
    assert tvm.ir.structural_equal(artifact.device_mod, snapshots["after"])
