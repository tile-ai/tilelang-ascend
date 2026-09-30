"""CPU-only regressions for outstanding accesses in AscendSyncInsert."""

import pytest

import tilelang
from tilelang import tvm

tir = tvm.tir


def _buffers():
    return {
        name: tir.decl_buffer((64,), "float32", name=name, scope=scope)
        for name, scope in (
            ("a", "shared.ub"),
            ("b", "shared.ub"),
            ("tmp", "shared.ub"),
            ("other", "shared.ub"),
            ("input", "global"),
            ("output", "global"),
        )
    }


def _copy(name, src, dst):
    return tir.Evaluate(tir.call_extern("handle", name, src.access_ptr("r"), dst.access_ptr("w")))


def _fill(dst):
    return tir.Evaluate(tir.call_extern("handle", "AscendC::Duplicate", dst.access_ptr("w")))


def _run(buffers, statements, b_offset=0, scalars=()):
    body = statements[0] if len(statements) == 1 else tir.SeqStmt(statements)
    offsets = {"a": 0, "b": b_offset, "tmp": 4096, "other": 8192}
    for name in offsets:
        buffer = buffers[name]
        body = tir.Allocate(buffer.data, buffer.dtype, [64], tir.const(True, "bool"), body)
    inputs = [buffers["input"], buffers["output"]]
    function = tir.PrimFunc(
        [buffer.data for buffer in inputs] + list(scalars),
        body,
        buffer_map={buffer.data: buffer for buffer in inputs},
    )
    function = function.with_attr("address_map", {buffers[name].data: value for name, value in offsets.items()})
    function = function.with_attr("size_map", {buffers[name].data: 256 for name in offsets})
    target = tvm.target.Target("c -keys=ascend -model=ascendc")
    with tvm.transform.PassContext(config={"tl.ascend_auto_sync": True}):
        module = tilelang.transform.AscendSyncInsert(target, "A3")(tvm.IRModule({"main": function}))
    return module["main"]


def _operations(function):
    if isinstance(function, list):
        return function
    operations = []

    def visit(node):
        if isinstance(node, tir.Evaluate) and isinstance(node.value, tir.Call):
            operations.append(node.value)
        elif isinstance(node, tir.BufferStore):
            operations.append(node)

    body = function.body if isinstance(function, tir.PrimFunc) else function
    tir.stmt_functor.post_order_visit(body, visit)
    return operations


def _handoffs(function):
    pending = {}
    handoffs = []
    for index, call in enumerate(_operations(function)):
        if not isinstance(call, tir.Call):
            continue
        if call.op == tvm.ir.Op.get("tl.ascend_auto_set_flag"):
            key = (str(call.args[0].value), int(call.args[1]))
            assert key not in pending, f"duplicate set without wait: {key}"
            pending[key] = index
        elif call.op == tvm.ir.Op.get("tl.ascend_auto_wait_flag"):
            key = (str(call.args[0].value), int(call.args[1]))
            assert key in pending, f"wait without matching earlier set: {key}"
            handoffs.append((key[0], pending.pop(key), index))
    assert not pending, f"set without matching wait: {pending}"
    return handoffs


def _events(function):
    return [event for event, _, _ in _handoffs(function)]


def _assert_handoff(function, event, producer, protected, *, occurrence=0):
    operations = _operations(function)

    def positions(stmt):
        value = stmt.value if isinstance(stmt, tir.Evaluate) else stmt
        return [index for index, op in enumerate(operations) if tvm.ir.structural_equal(op, value)]

    producer_index = positions(producer)[0]
    protected_index = positions(protected)[occurrence]
    matches = [
        (set_index, wait_index)
        for kind, set_index, wait_index in _handoffs(function)
        if kind == event and producer_index < set_index < wait_index < protected_index
    ]
    assert len(matches) == 1, f"expected one {event} handoff between producer and protected access"


def _reader_sequence(buffers, destination):
    return [
        _copy("copy_ub_to_gm", buffers["a"], buffers["output"]),
        _copy("copy_ub_to_ub", buffers["a"], buffers["tmp"]),
        _copy("copy_ub_to_ub", buffers["tmp"], buffers[destination]),
    ]


@pytest.mark.parametrize("destination,b_offset,needs_event", [("a", 0, True), ("b", 0, True), ("b", 128, True), ("b", 256, False)])
def test_outstanding_reader_survives_another_pipe_read(destination, b_offset, needs_event):
    buffers = _buffers()
    statements = _reader_sequence(buffers, destination)
    function = _run(buffers, statements, b_offset)
    assert ("MTE3_V" in _events(function)) == needs_event
    if needs_event:
        _assert_handoff(function, "MTE3_V", statements[0], statements[-1])


def test_read_read_does_not_require_synchronization():
    buffers = _buffers()
    function = _run(buffers, _reader_sequence(buffers, "b")[:2])
    assert not _events(function)
    assert "ascend_auto_barrier" not in function.script()


def test_writer_survives_same_pipe_read():
    buffers = _buffers()
    statements = [
        _fill(buffers["a"]),
        _copy("copy_ub_to_ub", buffers["a"], buffers["tmp"]),
        _copy("copy_ub_to_gm", buffers["a"], buffers["output"]),
    ]
    function = _run(buffers, statements)
    assert "V_MTE3" in _events(function)
    _assert_handoff(function, "V_MTE3", statements[0], statements[-1])


def test_new_reader_is_not_protected_by_old_alias_handoff():
    buffers = _buffers()
    statements = [
        _copy("copy_ub_to_gm", buffers["b"], buffers["output"]),
        _fill(buffers["b"]),
        *_reader_sequence(buffers, "b"),
    ]
    function = _run(buffers, statements)
    assert _events(function).count("MTE3_V") == 2
    _assert_handoff(function, "MTE3_V", statements[0], statements[1])
    _assert_handoff(function, "MTE3_V", statements[2], statements[-1])


@pytest.mark.parametrize("forward_order", [False, True])
def test_completion_paths_respect_event_order(forward_order):
    buffers = _buffers()
    vector_read = _copy("copy_ub_to_ub", buffers["a"], buffers["tmp"])
    other_handoff = [
        _fill(buffers["other"]),
        _copy("copy_ub_to_gm", buffers["other"], buffers["output"]),
    ]
    middle = [vector_read, *other_handoff] if forward_order else [*other_handoff, vector_read]
    statements = [
        _copy("copy_gm_to_ub", buffers["input"], buffers["a"]),
        *middle,
        _copy("copy_ub_to_gm", buffers["a"], buffers["output"]),
    ]
    function = _run(buffers, statements, b_offset=256)
    events = _events(function)
    assert "MTE2_V" in events and "V_MTE3" in events
    assert ("MTE2_MTE3" in events) != forward_order
    _assert_handoff(function, "MTE2_V", statements[0], vector_read)
    _assert_handoff(function, "V_MTE3", other_handoff[0], other_handoff[1])
    if not forward_order:
        _assert_handoff(function, "MTE2_MTE3", statements[0], statements[-1])


def test_outstanding_reader_survives_loop_back_edge():
    buffers = _buffers()
    # The reader is at the end of the iteration, the alias write at its start.
    read, vector_read, write = _reader_sequence(buffers, "b")
    loop = tir.For(tir.Var("i", "int32"), 0, 3, tir.ForKind.SERIAL, tir.SeqStmt([write, read, vector_read]))
    function = _run(buffers, [loop])
    assert "MTE3_V" in _events(function)
    loops = []
    tir.stmt_functor.post_order_visit(function.body, lambda node: loops.append(node) if isinstance(node, tir.For) else None)
    assert len(loops) == 1
    # Expand two iterations for the assertion: the previous iteration's read
    # must precede the handoff protecting the next iteration's overwrite.
    two_iterations = _operations(loops[0].body) * 2
    _assert_handoff(two_iterations, "MTE3_V", read, write, occurrence=1)


def test_branch_histories_keep_their_own_readers():
    buffers = _buffers()
    condition = tir.BufferLoad(buffers["input"], [0]) > 0
    branch = tir.IfThenElse(
        condition,
        tir.SeqStmt(_reader_sequence(buffers, "b")),
        tir.SeqStmt(_reader_sequence(buffers, "b")),
    )
    function = _run(buffers, [branch])
    assert _events(function).count("MTE3_V") == 2
    branches = []
    tir.stmt_functor.post_order_visit(function.body, lambda node: branches.append(node) if isinstance(node, tir.IfThenElse) else None)
    assert len(branches) == 1
    read, _, write = _reader_sequence(buffers, "b")
    for body in (branches[0].then_case, branches[0].else_case):
        _assert_handoff(body, "MTE3_V", read, write)


def test_scalar_alias_write_waits_for_all_readers():
    buffers = _buffers()
    statements = [
        *_reader_sequence(buffers, "b")[:2],
        tir.BufferStore(buffers["b"], tir.const(1, "float32"), [0]),
    ]
    function = _run(buffers, statements)
    assert {"MTE3_S", "V_S"}.issubset(_events(function))
    _assert_handoff(function, "MTE3_S", statements[0], statements[-1])
    _assert_handoff(function, "V_S", statements[1], statements[-1])


def _loop(extent, statements, name="i"):
    body = statements[0] if len(statements) == 1 else tir.SeqStmt(statements)
    return tir.For(tir.Var(name, "int32"), 0, extent, tir.ForKind.SERIAL, body)


def _outside_loops(function):
    """Return the operations that execute even when every loop is empty."""
    operations = []

    def visit(stmt):
        if isinstance(stmt, tir.For):
            return
        if isinstance(stmt, tir.SeqStmt):
            for child in stmt.seq:
                visit(child)
        elif isinstance(stmt, (tir.Allocate, tir.AttrStmt)):
            visit(stmt.body)
        else:
            operations.extend(_operations(stmt))

    visit(function.body)
    return operations


@pytest.mark.parametrize("trip_count", [0, 1, 3, "dynamic", "positive"])
def test_loop_exit_preserves_the_zero_trip_reader(trip_count):
    buffers = _buffers()
    n = tir.Var("n", "int32")
    if trip_count == "dynamic":
        extent = n
    elif trip_count == "positive":
        extent = tir.Max(n, 1)
    else:
        extent = trip_count
    read = _copy("copy_ub_to_gm", buffers["a"], buffers["output"])
    overwrite = _fill(buffers["b"])
    loop = _loop(extent, [overwrite, _copy("copy_gm_to_ub", buffers["input"], buffers["b"])])
    function = _run(buffers, [read, loop, overwrite], scalars=[n])
    outside = _outside_loops(function)
    needs_exit_handoff = trip_count in (0, "dynamic")
    assert ("MTE3_V" in _events(outside)) == needs_exit_handoff
    if needs_exit_handoff:
        _assert_handoff(outside, "MTE3_V", read, overwrite)
    if trip_count == 0:
        assert "MTE2_V" not in _events(outside)
        assert "for " not in function.script()
    else:
        # A possible last MTE2 writer still needs ordering on the taken path.
        assert "MTE2_V" in _events(outside)
        loops = []
        tir.stmt_functor.post_order_visit(function.body, lambda node: loops.append(node) if isinstance(node, tir.For) else None)
        assert len(loops) == 1
        assert "MTE3_V" in _events(loops[0].body)


def test_optional_loop_preserves_an_entry_writer_replaced_in_the_body():
    buffers = _buffers()
    n = tir.Var("n", "int32")
    entry_write = _copy("copy_gm_to_ub", buffers["input"], buffers["a"])
    loop_write = _fill(buffers["a"])
    read = _copy("copy_ub_to_gm", buffers["a"], buffers["output"])
    function = _run(buffers, [entry_write, _loop(n, [loop_write]), read], scalars=[n])
    outside = _outside_loops(function)
    assert {"MTE2_MTE3", "V_MTE3"}.issubset(_events(outside))
    _assert_handoff(outside, "MTE2_MTE3", entry_write, read)


def test_optional_loop_intersects_completion_for_different_writer_generations():
    buffers = _buffers()
    n = tir.Var("n", "int32")
    write = _copy("copy_gm_to_ub", buffers["input"], buffers["a"])
    read = _copy("copy_ub_to_gm", buffers["a"], buffers["output"])
    # The entry writer has completed to MTE3, but a new writer on the same pipe
    # exists only when the loop executes and has not completed to MTE3 yet.
    function = _run(buffers, [write, read, _loop(n, [write]), read], scalars=[n])
    assert _events(_outside_loops(function)).count("MTE2_MTE3") == 2


def test_optional_loop_keeps_completion_proven_before_entry():
    buffers = _buffers()
    n = tir.Var("n", "int32")
    read = _copy("copy_ub_to_gm", buffers["a"], buffers["output"])
    overwrite = _fill(buffers["b"])
    function = _run(buffers, [read, overwrite, _loop(n, [_fill(buffers["other"])]), overwrite], scalars=[n])
    # Entry and taken paths both know the original alias reader has completed.
    assert _events(_outside_loops(function)).count("MTE3_V") == 1


@pytest.mark.parametrize("layout", ["consecutive", "zero_consecutive", "nested", "positive_outer", "zero_outer"])
def test_optional_loops_compose_without_losing_the_entry_reader(layout):
    buffers = _buffers()
    n, m = tir.Var("n", "int32"), tir.Var("m", "int32")
    read = _copy("copy_ub_to_gm", buffers["a"], buffers["output"])
    overwrite = _fill(buffers["b"])
    body = [overwrite, _copy("copy_gm_to_ub", buffers["input"], buffers["b"])]
    inner = _loop(m, body, "j")
    if layout == "zero_consecutive":
        loops = [_loop(0, body), _loop(0, body, "j")]
    elif layout == "consecutive":
        loops = [_loop(n, body), inner]
    else:
        outer_extent = {"positive_outer": 2, "zero_outer": 0}.get(layout, n)
        loops = [_loop(outer_extent, [inner])]
    function = _run(buffers, [read, *loops, overwrite], scalars=[n, m])
    _assert_handoff(_outside_loops(function), "MTE3_V", read, overwrite)
