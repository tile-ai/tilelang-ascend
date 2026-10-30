"""Read-only checking of actual compiler-managed mask setters and terminals."""

import pytest

import tilelang
from tilelang import tvm

from testing.python.ascend_migration.test_s2_task_ir import _add, _buffer, _fill, _function

tir = tvm.tir
TARGET = tvm.target.Target("c -keys=ascend -model=ascendc")


def _selected(operations):
    buffers, statements = [], []
    for index, (kind, count, dtype) in enumerate(operations):
        a, b = [_buffer(f"{name}{index}", dtype=dtype, shape=(256,)) for name in "ab"]
        buffers.extend([a, b])
        statements.append(_fill(a, count=count) if kind == "fill" else _add(a, b, b, count))
    function = _function(buffers, statements)
    return tilelang.transform.AscendVectorInstructionSelection(TARGET, "A3")(tvm.IRModule({"main": function}))["main"]


def _legalize(function, reuse=True):
    with tvm.transform.PassContext(config={"tl.ascend_pipeline_planner": "constraint", "tl.ascend_vector_mask_reuse": reuse}):
        result = tilelang.transform.AscendVectorMaskLegalize(TARGET, "A3")(tvm.IRModule({"main": function}))["main"]
    return result


def _verify(function):
    before = tvm.ir.save_json(function)
    result = tvm.get_global_func("tl.transform._AscendVerifyManagedMaskContract")(function)
    assert result.same_as(function)
    assert tvm.ir.save_json(function) == before


def _name(statement):
    if isinstance(statement, tir.Evaluate) and isinstance(statement.value, tir.Call):
        return getattr(statement.value.op, "name", "")
    return ""


def _change(function, setter, replacement, occurrence=0):
    observed = 0

    def rewrite(node):
        nonlocal observed
        if _name(node) != setter:
            return None
        index = observed
        observed += 1
        return replacement(node) if index == occurrence else None

    return function.with_body(tir.stmt_functor.ir_transform(function.body, None, rewrite, ["tir.Evaluate"]))


@pytest.mark.parametrize("reuse", [False, True])
@pytest.mark.parametrize("dtype", ["float16", "float32"])
def test_actual_mask_contract_accepts_mode_and_count_transitions(reuse, dtype):
    function = _selected([("add", 64, dtype), ("fill", 32, dtype), ("add", 96, dtype)])
    _verify(_legalize(function, reuse))


def test_mask_entry_state_is_unknown():
    with pytest.raises(tvm.TVMError, match="reaching setter"):
        _verify(_selected([("add", 64, "float32")]))


@pytest.mark.parametrize("corruption", ["drop_mode", "drop_payload", "wrong_mode", "wrong_low"])
def test_missing_or_wrong_reaching_setters_are_rejected(corruption):
    function = _legalize(_selected([("add", 64, "float32")]))
    setter = "tl.ascend_set_mask_mode" if "mode" in corruption else "tl.ascend_set_mask_payload"

    def replace(node):
        if corruption.startswith("drop"):
            return tir.Evaluate(0)
        args = list(node.value.args)
        args[0] = tir.const(1, "int32") if corruption == "wrong_mode" else tir.const(0, "uint64")
        return tir.Evaluate(tir.Call("handle", node.value.op, args))

    with pytest.raises(tvm.TVMError, match="reaching setter"):
        _verify(_change(function, setter, replace))


def test_fp16_high_mask_word_is_part_of_the_shared_state():
    function = _legalize(_selected([("add", 128, "float16")]))

    def clear_high(node):
        args = [node.value.args[0], tir.const(0, "uint64")]
        return tir.Evaluate(tir.Call("handle", node.value.op, args))

    with pytest.raises(tvm.TVMError, match="high word"):
        _verify(_change(function, "tl.ascend_set_mask_payload", clear_high))


def test_next_payload_overwrite_cannot_move_before_its_previous_consumer():
    function = _legalize(_selected([("add", 32, "float32"), ("add", 64, "float32")]))

    def rewrite(node):
        if not isinstance(node, tir.SeqStmt):
            return None
        statements = list(node.seq)
        terminals = [i for i, stmt in enumerate(statements) if _name(stmt) == "tl.ascend_add_raw"]
        payloads = [i for i, stmt in enumerate(statements) if _name(stmt) == "tl.ascend_set_mask_payload"]
        if len(terminals) == len(payloads) == 2:
            next_setter = statements.pop(payloads[1])
            statements.insert(terminals[0], next_setter)
            return tir.SeqStmt(statements)
        return None

    body = tir.stmt_functor.ir_transform(function.body, None, rewrite, ["tir.SeqStmt"])
    assert not tvm.ir.structural_equal(body, function.body)
    with pytest.raises(tvm.TVMError, match="low word"):
        _verify(function.with_body(body))


@pytest.mark.parametrize("trips", [0, 1, 3])
@pytest.mark.parametrize("reuse", [False, True])
def test_zero_short_loop_and_exit_mask_requirements_are_checked(trips, reuse):
    selected = _selected([("add", 64, "float32"), ("fill", 32, "float32")])
    statements = []
    tir.stmt_functor.post_order_visit(
        selected.body,
        lambda node: statements.append(node) if _name(node) in {"tl.ascend_add_raw", "tl.ascend_fill_raw_counter"} else None,
    )
    loop = tir.For(tir.Var("i", "int32"), 0, trips, tir.ForKind.SERIAL, tir.SeqStmt(statements))
    body = tir.AttrStmt(tir.const(0, "int32"), "resource_scope", 1, tir.SeqStmt([loop, statements[0]]))
    _verify(_legalize(selected.with_body(body), reuse))


@pytest.mark.parametrize("intervening_buffers", [1, 2, 3])
def test_mask_state_is_shared_across_independent_buffers(intervening_buffers):
    operations = [("add", 64, "float32")]
    operations += [("fill", 32, "float32")] * intervening_buffers
    operations += [("add", 64, "float32")]
    function = _legalize(_selected(operations))
    _verify(function)
    modes = []
    tir.stmt_functor.post_order_visit(
        function.body,
        lambda node: modes.append(node) if _name(node) == "tl.ascend_set_mask_mode" else None,
    )
    # NORMAL -> COUNTER -> NORMAL applies to one physical mask state, even
    # when the intervening operations use independent UB allocations.
    assert len(modes) == 3
    broken = _change(function, "tl.ascend_set_mask_mode", lambda _: tir.Evaluate(0), occurrence=2)
    with pytest.raises(tvm.TVMError, match="mode.*reaching setter"):
        _verify(broken)
