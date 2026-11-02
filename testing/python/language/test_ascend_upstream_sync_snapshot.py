# Copyright (c) Tile-AI Organization.
# Licensed under the MIT License.
"""CPU-only runtime-snapshot and lexical insertion tests for upstream sync.

These fixtures do not compile or run a device kernel. Buffer loads remain
real LetStmt values; proof-only unknown variables cannot satisfy these checks.
"""

from collections import defaultdict

import pytest

import tilelang
from tilelang import tvm

tir = tvm.tir


@pytest.fixture(autouse=True)
def _requires_proof_service() -> None:
    if not tvm.get_global_func("tl.ascend_sync.Z3ProofServiceAvailable")():
        pytest.skip("requires USE_ASCEND_SYNC_Z3=ON")


def _buffers() -> dict[str, tir.Buffer]:
    return {
        name: tir.decl_buffer((64,), "float32", name=name, scope=scope)
        for name, scope in (
            ("control", "shared.ub"),
            ("a", "shared.ub"),
            ("b", "shared.ub"),
            ("input", "global"),
            ("output", "global"),
        )
    }


def _fill(buffer: tir.Buffer, value: tir.PrimExpr | None = None) -> tir.Evaluate:
    if value is None:
        value = tir.FloatImm("float32", 1.0)
    return tir.Evaluate(tir.call_extern("handle", "AscendC::Duplicate", buffer.access_ptr("w"), value))


def _copy(name: str, source: tir.Buffer, destination: tir.Buffer) -> tir.Evaluate:
    return tir.Evaluate(tir.call_extern("handle", name, source.access_ptr("r"), destination.access_ptr("w")))


def _function(
    buffers: dict[str, tir.Buffer],
    statements: list[tir.Stmt],
    *,
    scalars: tuple[tir.Var, ...] = (),
) -> tir.PrimFunc:
    body = statements[0] if len(statements) == 1 else tir.SeqStmt(statements)
    body = tir.AttrStmt(tir.IntImm("int32", 0), "resource_scope", 1, body)
    addresses = {"control": 0, "a": 256, "b": 512}
    for name in addresses:
        buffer = buffers[name]
        body = tir.Allocate(buffer.data, buffer.dtype, [64], tir.const(True, "bool"), body)
    parameters = [buffers["input"], buffers["output"]]
    return tir.PrimFunc(
        [buffer.data for buffer in parameters] + list(scalars),
        body,
        buffer_map={buffer.data: buffer for buffer in parameters},
    ).with_attr(
        {
            "address_map": {buffers[name].data: offset for name, offset in addresses.items()},
            "size_map": {buffers[name].data: 256 for name in addresses},
            "tir.noalias": True,
        }
    )


def _run(function: tir.PrimFunc) -> tir.PrimFunc:
    target = tvm.target.Target("c -keys=ascend -model=ascendc")
    with tvm.transform.PassContext(config={"tl.ascend_auto_sync": True, "tl.ascend_auto_sync_vs": True}):
        module = tilelang.transform.AscendInsertSync(target, "A3")(tvm.IRModule({"main": function}))
    return module["main"]


def _nodes(statement: tir.Stmt, node_type: type) -> list:
    result = []

    def collect(node: object) -> None:
        if isinstance(node, node_type):
            result.append(node)

    tir.stmt_functor.post_order_visit(statement, collect)
    return result


def _flag_calls(statement: tir.Stmt, kind: str, event: str) -> list[tir.Call]:
    return [
        call
        for call in _nodes(statement, tir.Call)
        if isinstance(call.op, tvm.ir.Op) and call.op.name == f"tl.ascend_auto_{kind}_flag" and call.args[0].value == event
    ]


def _evaluation_order(statement: tir.Stmt) -> list:
    """Record a binding after its value and before its body, as TIR executes."""
    result = []

    def visit(node: tir.Stmt) -> None:
        if isinstance(node, tir.SeqStmt):
            for child in node.seq:
                visit(child)
        elif isinstance(node, tir.LetStmt):
            result.append(node.var)
            visit(node.body)
        elif isinstance(node, (tir.Allocate, tir.AttrStmt, tir.DeclBuffer)):
            visit(node.body)
        elif isinstance(node, tir.Evaluate) and isinstance(node.value, tir.Call):
            result.append(node.value)

    visit(statement)
    return result


def test_mutable_let_waits_before_the_read_and_sets_after_the_binding() -> None:
    buffers = _buffers()
    value = tir.Var("saved", "float32")
    original_value = tir.BufferLoad(buffers["control"], [0])
    definition = tir.LetStmt(
        value,
        original_value,
        tir.SeqStmt([_fill(buffers["control"]), _fill(buffers["b"], value)]),
    )
    result = _run(_function(buffers, [_fill(buffers["control"]), definition]))
    binding = next(node for node in _nodes(result.body, tir.LetStmt) if node.var.same_as(value))
    assert tvm.ir.structural_equal(binding.value, original_value)
    incoming = _flag_calls(result.body, "wait", "V_S")
    outgoing = _flag_calls(binding.body, "set", "S_V")
    assert incoming and outgoing, "snapshot reads must participate in real memory dependencies"
    order = _evaluation_order(result.body)
    binding_index = next(i for i, item in enumerate(order) if item.same_as(value))
    assert next(i for i, item in enumerate(order) if item.same_as(incoming[0])) < binding_index
    assert next(i for i, item in enumerate(order) if item.same_as(outgoing[0])) > binding_index
    assert len(_nodes(result.body, tir.BufferLoad)) == 1


def test_hardware_scalar_memory_operand_is_captured_once_before_the_operation() -> None:
    buffers = _buffers()
    scalar = tir.BufferLoad(buffers["control"], [0])
    result = _run(_function(buffers, [_fill(buffers["control"]), _fill(buffers["b"], scalar)]))
    bindings = _nodes(result.body, tir.LetStmt)
    assert len(bindings) == 1
    binding = bindings[0]
    assert tvm.ir.structural_equal(binding.value, scalar)
    calls = [call for call in _nodes(binding.body, tir.Call) if isinstance(call.op, tvm.ir.Op) and call.op.name == "tir.call_extern"]
    assert len(calls) == 1
    assert calls[0].args[2].same_as(binding.var)
    assert _flag_calls(result.body, "wait", "V_S")
    assert len(_nodes(result.body, tir.BufferLoad)) == 1


def _indirect_copy(buffers: dict[str, tir.Buffer], expression: tir.PrimExpr, field: int, direction: str) -> tir.Evaluate:
    pointer_buffer = buffers["input"] if direction == "MTE2" else buffers["output"]
    pointer = pointer_buffer.access_ptr("r" if direction == "MTE2" else "w")
    arguments = list(pointer.args)
    arguments[field] = expression
    pointer = tir.Call(pointer.dtype, pointer.op, arguments)
    local = buffers["a"].access_ptr("w" if direction == "MTE2" else "r")
    operands = (pointer, local) if direction == "MTE2" else (local, pointer)
    helper = "copy_gm_to_ub" if direction == "MTE2" else "copy_ub_to_gm"
    return tir.Evaluate(tir.call_extern("handle", helper, *operands))


@pytest.mark.parametrize("field", [2, 3], ids=["offset", "extent"])
@pytest.mark.parametrize("direction", ["MTE2", "MTE3"])
def test_pointer_memory_operand_is_read_once_before_dma(field: int, direction: str) -> None:
    buffers = _buffers()
    expression = tir.Cast("int32", tir.BufferLoad(buffers["control"], [0])) * 8
    result = _run(_function(buffers, [_fill(buffers["control"]), _indirect_copy(buffers, expression, field, direction)]))
    bindings = _nodes(result.body, tir.LetStmt)
    assert len(bindings) == 1
    binding = bindings[0]
    tvm.ir.assert_structural_equal(binding.value, expression)
    assert len(_nodes(result.body, tir.BufferLoad)) == 1
    helper = next(call for call in _nodes(binding.body, tir.Call) if call.op == tvm.ir.Op.get("tir.call_extern"))
    pointer = helper.args[1 if direction == "MTE2" else 2]
    assert pointer.args[field].same_as(binding.var)
    order = _evaluation_order(result.body)
    binding_index = next(i for i, item in enumerate(order) if item.same_as(binding.var))
    helper_index = next(i for i, item in enumerate(order) if item.same_as(helper))
    assert any(
        next(i for i, item in enumerate(order) if item.same_as(wait)) < binding_index for wait in _flag_calls(result.body, "wait", "V_S")
    )
    # The DMA consumes a scalar register defined by the preceding Let. This
    # def-use requires no S->DMA memory-completion event; V->S protects the
    # actual UB read, and export must keep the definition before its use.
    assert binding_index < helper_index
    tvm.ir.assert_structural_equal(result, _run(result))


def test_equal_pointer_memory_reads_after_an_index_update_get_distinct_snapshots() -> None:
    buffers = _buffers()
    expression = tir.Cast("int32", tir.BufferLoad(buffers["control"], [0])) * 8
    copy = _indirect_copy(buffers, expression, 2, "MTE2")
    result = _run(_function(buffers, [_fill(buffers["control"]), copy, _fill(buffers["control"], tir.FloatImm("float32", 2.0)), copy]))
    bindings = _nodes(result.body, tir.LetStmt)
    assert len(bindings) == 2
    assert not bindings[0].var.same_as(bindings[1].var)
    assert all(tvm.ir.structural_equal(binding.value, expression) for binding in bindings)
    # post_order_visit deduplicates shared expression objects. Count each
    # lexical binding's read instead: the same expression executes twice.
    assert sum(len(_nodes(tir.Evaluate(binding.value), tir.BufferLoad)) for binding in bindings) == 2
    assert _flag_calls(result.body, "wait", "S_V")


def test_pointer_fields_and_hardware_scalar_keep_their_original_operand_order() -> None:
    buffers = _buffers()
    reads = [tir.BufferLoad(buffers["control"], [index]) for index in range(3)]
    pointer = buffers["b"].access_ptr("w")
    fields = list(pointer.args)
    fields[2] = tir.Cast("int32", reads[0])
    fields[3] = tir.Cast("int32", reads[1])
    pointer = tir.Call(pointer.dtype, pointer.op, fields)
    operation = tir.Evaluate(tir.call_extern("handle", "AscendC::Duplicate", pointer, reads[2]))
    result = _run(_function(buffers, [_fill(buffers["control"]), operation]))
    bindings = {binding.var: binding.value for binding in _nodes(result.body, tir.LetStmt)}
    ordered_values = [bindings[node] for node in _evaluation_order(result.body) if isinstance(node, tir.Var) and node in bindings]
    assert len(ordered_values) == 3
    for actual, expected in zip(ordered_values, [fields[2], fields[3], reads[2]], strict=True):
        tvm.ir.assert_structural_equal(actual, expected)
    assert len(_nodes(result.body, tir.BufferLoad)) == 3


@pytest.mark.parametrize("expression", [tir.call_extern("int32", "unknown_index"), tir.call_intrin("int32", "tir.tvm_thread_context", 0)])
def test_opaque_or_implicit_state_pointer_operands_remain_rejected(expression: tir.PrimExpr) -> None:
    buffers = _buffers()
    with pytest.raises(tvm.error.TVMError, match="snapshot.*opaque|snapshot.*unmodelled"):
        _run(_function(buffers, [_indirect_copy(buffers, expression, 2, "MTE2")]))


def test_vector_pointer_memory_operand_is_not_captured_as_a_scalar() -> None:
    buffers = _buffers()
    expression = tir.Broadcast(tir.Cast("int32", tir.BufferLoad(buffers["control"], [0])), 2)
    with pytest.raises(tvm.error.TVMError, match="snapshot requires a scalar"):
        _run(_function(buffers, [_indirect_copy(buffers, expression, 2, "MTE2")]))


def test_pure_pointer_fields_keep_the_original_helper_call() -> None:
    buffers = _buffers()
    original = _indirect_copy(buffers, tir.IntImm("int32", 8), 2, "MTE2")
    result = _run(_function(buffers, [original]))
    assert not _nodes(result.body, tir.LetStmt)
    helper = next(call for call in _nodes(result.body, tir.Call) if call.op == tvm.ir.Op.get("tir.call_extern"))
    tvm.ir.assert_structural_equal(helper, original.value)


def test_mutable_if_condition_is_one_runtime_snapshot_with_pure_guards() -> None:
    buffers = _buffers()
    predicate = tir.BufferLoad(buffers["control"], [0]) > 0
    branch = tir.IfThenElse(
        predicate,
        tir.SeqStmt(
            [
                _copy("copy_gm_to_ub", buffers["input"], buffers["a"]),
                _copy("copy_ub_to_gm", buffers["a"], buffers["output"]),
            ]
        ),
        None,
    )
    result = _run(_function(buffers, [_fill(buffers["control"]), branch]))
    bindings = _nodes(result.body, tir.LetStmt)
    assert len(bindings) == 1
    assert tvm.ir.structural_equal(bindings[0].value, predicate)
    assert len(_nodes(result.body, tir.BufferLoad)) == 1
    for conditional in _nodes(result.body, tir.IfThenElse):
        assert not _nodes(tir.Evaluate(conditional.condition), tir.BufferLoad)
    assert _flag_calls(result.body, "wait", "V_S")
    assert tvm.ir.structural_equal(result, _run(result))


def test_equal_memory_reads_at_different_program_points_remain_distinct_snapshots() -> None:
    buffers = _buffers()
    first = tir.Var("first", "float32")
    second = tir.Var("second", "float32")
    read = tir.BufferLoad(buffers["control"], [0])
    body = tir.SeqStmt(
        [
            tir.BufferStore(buffers["control"], tir.FloatImm("float32", 2.0), [0]),
            tir.LetStmt(
                second,
                read,
                tir.SeqStmt(
                    [
                        tir.BufferStore(buffers["a"], first, [0]),
                        tir.BufferStore(buffers["b"], second, [0]),
                    ]
                ),
            ),
        ]
    )
    result = _run(
        _function(
            buffers,
            [
                tir.BufferStore(buffers["control"], tir.FloatImm("float32", 1.0), [0]),
                tir.LetStmt(first, read, body),
            ],
        )
    )
    bindings = _nodes(result.body, tir.LetStmt)
    assert len(bindings) == 2
    assert {node.var.name for node in bindings} == {"first", "second"}
    assert all(tvm.ir.structural_equal(node.value, read) for node in bindings)
    memory = {}

    def evaluate(expression: tir.PrimExpr, values: dict) -> float:
        if isinstance(expression, (tir.FloatImm, tir.IntImm)):
            return expression.value
        if isinstance(expression, tir.Var):
            return values[expression]
        assert isinstance(expression, tir.BufferLoad)
        return memory[expression.buffer.data]

    def execute(statement: tir.Stmt, values: dict) -> None:
        if isinstance(statement, tir.SeqStmt):
            for child in statement.seq:
                execute(child, values)
        elif isinstance(statement, (tir.Allocate, tir.AttrStmt, tir.DeclBuffer)):
            execute(statement.body, values)
        elif isinstance(statement, tir.LetStmt):
            captured = evaluate(statement.value, values)
            execute(statement.body, {**values, statement.var: captured})
        elif isinstance(statement, tir.BufferStore):
            memory[statement.buffer.data] = evaluate(statement.value, values)
        else:
            raise AssertionError(f"unexpected scalar-only fixture statement: {statement}")

    execute(result.body, {})
    assert memory[buffers["a"].data] == 1.0
    assert memory[buffers["b"].data] == 2.0


@pytest.mark.parametrize("position", ["let", "if"])
def test_opaque_scalar_expressions_are_not_treated_as_memory_snapshots(position: str) -> None:
    buffers = _buffers()
    dtype = "float32" if position == "let" else "bool"
    opaque = tir.call_extern(dtype, "unknown_scalar_side_effect")
    statement = (
        tir.LetStmt(tir.Var("value", dtype), opaque, _fill(buffers["a"]))
        if position == "let"
        else tir.IfThenElse(opaque, _fill(buffers["a"]), None)
    )
    with pytest.raises(tvm.error.TVMError, match="snapshot.*opaque"):
        _run(_function(buffers, [statement]))


def test_mutable_for_bounds_remain_explicitly_unsupported() -> None:
    buffers = _buffers()
    buffers["control"] = tir.decl_buffer((64,), "int32", name="control", scope="shared.ub")
    loop = tir.For(
        tir.Var("i", "int32"),
        0,
        tir.BufferLoad(buffers["control"], [0]),
        tir.ForKind.SERIAL,
        _fill(buffers["a"]),
    )
    with pytest.raises(tvm.error.TVMError, match="mutable For bounds.*unsupported"):
        _run(_function(buffers, [loop]))


def _interpret_scalar_program(function: tir.PrimFunc, parameters: dict | None = None) -> tuple:
    """Execute independent scalar TIR semantics and count actual memory reads."""
    memory = {}
    memory_reads = defaultdict(int)
    variable_reads = defaultdict(int)

    def evaluate(expression: tir.PrimExpr, values: dict):
        if isinstance(expression, (tir.FloatImm, tir.IntImm)):
            return expression.value
        if isinstance(expression, tir.Var):
            variable_reads[expression] += 1
            return values[expression]
        if isinstance(expression, tir.BufferLoad):
            memory_reads[expression.buffer.data] += 1
            index = int(evaluate(expression.indices[0], values))
            return memory[(expression.buffer.data, index)]
        if isinstance(expression, tir.Not):
            return not evaluate(expression.a, values)
        if isinstance(expression, tir.Cast):
            return evaluate(expression.value, values)
        operations = (
            (tir.Add, lambda a, b: a + b),
            (tir.Sub, lambda a, b: a - b),
            (tir.Mul, lambda a, b: a * b),
            (tir.FloorMod, lambda a, b: a % b),
            (tir.GT, lambda a, b: a > b),
            (tir.GE, lambda a, b: a >= b),
            (tir.LT, lambda a, b: a < b),
            (tir.LE, lambda a, b: a <= b),
            (tir.EQ, lambda a, b: a == b),
            (tir.NE, lambda a, b: a != b),
        )
        for kind, operation in operations:
            if isinstance(expression, kind):
                return operation(evaluate(expression.a, values), evaluate(expression.b, values))
        raise AssertionError(f"unmodelled scalar expression: {expression}")

    def execute(statement: tir.Stmt, values: dict) -> None:
        if isinstance(statement, tir.SeqStmt):
            for child in statement.seq:
                execute(child, values)
        elif isinstance(statement, (tir.Allocate, tir.AttrStmt, tir.DeclBuffer)):
            execute(statement.body, values)
        elif isinstance(statement, tir.LetStmt):
            captured = evaluate(statement.value, values)
            execute(statement.body, {**values, statement.var: captured})
        elif isinstance(statement, tir.For):
            minimum = int(evaluate(statement.min, values))
            extent = int(evaluate(statement.extent, values))
            for index in range(minimum, minimum + extent):
                execute(statement.body, {**values, statement.loop_var: index})
        elif isinstance(statement, tir.IfThenElse):
            selected = statement.then_case if evaluate(statement.condition, values) else statement.else_case
            if selected is not None:
                execute(selected, values)
        elif isinstance(statement, tir.BufferStore):
            index = int(evaluate(statement.indices[0], values))
            memory[(statement.buffer.data, index)] = evaluate(statement.value, values)
        elif isinstance(statement, tir.Evaluate):
            if isinstance(statement.value, tir.Call):
                assert statement.value.op.name in {
                    "tl.ascend_auto_barrier",
                    "tl.ascend_auto_set_flag",
                    "tl.ascend_auto_wait_flag",
                }
        else:
            raise AssertionError(f"unmodelled scalar statement: {statement}")

    execute(function.body, {} if parameters is None else parameters)
    return memory, memory_reads, variable_reads


@pytest.mark.parametrize("initial", [-1.0, 1.0])
def test_if_else_memory_condition_is_captured_before_either_branch_mutates_it(
    initial: float,
) -> None:
    buffers = _buffers()
    predicate = tir.BufferLoad(buffers["control"], [0]) > 0
    branch = tir.IfThenElse(
        predicate,
        tir.SeqStmt(
            [
                tir.BufferStore(buffers["control"], tir.FloatImm("float32", -1.0), [0]),
                tir.BufferStore(buffers["a"], tir.FloatImm("float32", 11.0), [0]),
            ]
        ),
        tir.SeqStmt(
            [
                tir.BufferStore(buffers["control"], tir.FloatImm("float32", 1.0), [0]),
                tir.BufferStore(buffers["a"], tir.FloatImm("float32", 22.0), [0]),
            ]
        ),
    )
    original = _function(
        buffers,
        [tir.BufferStore(buffers["control"], tir.FloatImm("float32", initial), [0]), branch],
    )
    result = _run(original)
    expected, expected_reads, _ = _interpret_scalar_program(original)
    actual, actual_reads, _ = _interpret_scalar_program(result)
    assert actual == expected
    assert actual[(buffers["a"].data, 0)] == (11.0 if initial > 0 else 22.0)
    assert actual_reads[buffers["control"].data] == expected_reads[buffers["control"].data] == 1
    bindings = _nodes(result.body, tir.LetStmt)
    assert len(bindings) == 1
    assert tvm.ir.structural_equal(bindings[0].value, predicate)
    assert all(node.else_case is None for node in _nodes(result.body, tir.IfThenElse))
    assert tvm.ir.structural_equal(result, _run(result))


@pytest.mark.parametrize("condition_value", [False, True])
def test_pure_if_else_condition_also_has_one_stable_runtime_binding(condition_value: bool) -> None:
    buffers = _buffers()
    condition = tir.Var("condition", "bool")
    original = _function(
        buffers,
        [
            tir.IfThenElse(
                condition,
                tir.BufferStore(buffers["a"], tir.FloatImm("float32", 1.0), [0]),
                tir.BufferStore(buffers["a"], tir.FloatImm("float32", 2.0), [0]),
            )
        ],
        scalars=(condition,),
    )
    result = _run(original)
    expected, _, _ = _interpret_scalar_program(original, {condition: condition_value})
    actual, _, reads = _interpret_scalar_program(result, {condition: condition_value})
    assert actual == expected
    assert reads[condition] == 1
    bindings = _nodes(result.body, tir.LetStmt)
    assert len(bindings) == 1
    assert bindings[0].value.same_as(condition)


@pytest.mark.parametrize("trip_count", [0, 1, 3, 4])
def test_loop_if_else_rechecks_memory_once_per_original_decision(trip_count: int) -> None:
    buffers = _buffers()
    control = buffers["control"]
    first = buffers["a"]
    second = buffers["b"]
    branch = tir.IfThenElse(
        tir.BufferLoad(control, [0]) > 0,
        tir.SeqStmt(
            [
                tir.BufferStore(control, tir.FloatImm("float32", -1.0), [0]),
                tir.BufferStore(first, tir.BufferLoad(first, [0]) + 1.0, [0]),
            ]
        ),
        tir.SeqStmt(
            [
                tir.BufferStore(control, tir.FloatImm("float32", 1.0), [0]),
                tir.BufferStore(second, tir.BufferLoad(second, [0]) + 1.0, [0]),
            ]
        ),
    )
    original = _function(
        buffers,
        [
            tir.BufferStore(control, tir.FloatImm("float32", 1.0), [0]),
            tir.BufferStore(first, tir.FloatImm("float32", 0.0), [0]),
            tir.BufferStore(second, tir.FloatImm("float32", 0.0), [0]),
            tir.For(tir.Var("i", "int32"), 0, trip_count, tir.ForKind.SERIAL, branch),
        ],
    )
    result = _run(original)
    expected, expected_reads, _ = _interpret_scalar_program(original)
    actual, actual_reads, _ = _interpret_scalar_program(result)
    assert actual == expected
    assert actual[(first.data, 0)] == (trip_count + 1) // 2
    assert actual[(second.data, 0)] == trip_count // 2
    assert actual_reads[control.data] == expected_reads[control.data] == trip_count
