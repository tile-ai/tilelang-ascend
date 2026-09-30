"""CPU regressions for first-iteration exits and rebuilt loop handoffs."""

import pytest

from test_ascend_sync_insert import (
    _assert_handoff,
    _buffers,
    _copy,
    _events,
    _fill,
    _loop,
    _operations,
    _outside_loops,
    _run,
    tir,
)


def _late_completion(buffers, access_kind):
    if access_kind == "writer":
        producer = _copy("copy_ub_to_ub", buffers["other"], buffers["b"])
        body = [
            _copy("copy_gm_to_ub", buffers["input"], buffers["tmp"]),
            _copy("copy_ub_to_gm", buffers["other"], buffers["output"]),
            _copy("copy_gm_to_ub", buffers["input"], buffers["other"]),
        ]
        protected = _copy("copy_ub_to_gm", buffers["b"], buffers["output"])
        event = "V_MTE3"
    else:
        producer = _copy("copy_ub_to_ub", buffers["a"], buffers["tmp"])
        body = [
            _copy("copy_ub_to_gm", buffers["other"], buffers["output"]),
            _copy("copy_gm_to_ub", buffers["input"], buffers["other"]),
            _copy("copy_ub_to_gm", buffers["tmp"], buffers["output"]),
        ]
        protected = _copy("copy_gm_to_ub", buffers["input"], buffers["a"])
        event = "V_MTE2"
    return producer, body, protected, event


def _assert_no_full_barrier(function):
    for operation in _operations(function):
        if isinstance(operation, tir.Call) and str(operation.op) == "tl.ascend_auto_barrier":
            assert str(operation.args[0].value) != "PIPE_ALL"


def _loops(function):
    loops = []
    tir.stmt_functor.post_order_visit(
        function.body,
        lambda node: loops.append(node) if isinstance(node, tir.For) else None,
    )
    return loops


@pytest.mark.parametrize("trip_count", [1, "positive", "optional"])
@pytest.mark.parametrize("access_kind", ["writer", "reader"])
def test_loop_exit_does_not_borrow_a_second_iteration_handoff(trip_count, access_kind):
    buffers = _buffers()
    n = tir.Var("n", "int32")
    extent = {"positive": tir.Max(n, 1), "optional": n}.get(trip_count, trip_count)
    producer, body, protected, event = _late_completion(buffers, access_kind)
    function = _run(buffers, [producer, _loop(extent, body), protected], b_offset=256, scalars=[n])

    # The first iteration establishes V -> intermediate only after the
    # intermediate -> consumer handoff. Only a second iteration can chain them.
    _assert_handoff(_outside_loops(function), event, producer, protected)
    # Rebuilding must still keep the handoffs needed inside the real loop.
    events = _events(_loops(function)[0].body)
    assert {"MTE2_MTE3", "MTE3_MTE2"}.issubset(events)
    assert ("V_MTE2" if access_kind == "writer" else "V_MTE3") in events
    _assert_no_full_barrier(function)


@pytest.mark.parametrize("inner_optional", [False, True])
def test_nested_loop_exit_keeps_a_writer_created_in_the_outer_iteration(inner_optional):
    buffers = _buffers()
    n, m = tir.Var("n", "int32"), tir.Var("m", "int32")
    producer, body, protected, event = _late_completion(buffers, "writer")
    inner = _loop(m if inner_optional else tir.Max(m, 1), body, "j")
    outer = _loop(tir.Max(n, 1), [producer, inner])
    function = _run(buffers, [outer, protected], b_offset=256, scalars=[n, m])

    # The writer does not exist before the outer loop. Its exit guarantee must
    # cover an inner loop that ends after its first iteration (or is skipped).
    assert event in _events(_outside_loops(function))
    assert len(_loops(function)) == 2
    _events(function)  # Every generated set must retain its matching wait.
    _assert_no_full_barrier(function)


def test_positive_loop_preserves_completion_already_proven_before_entry():
    buffers = _buffers()
    n = tir.Var("n", "int32")
    read = _copy("copy_ub_to_gm", buffers["a"], buffers["output"])
    overwrite = _fill(buffers["b"])
    independent_loop = _loop(tir.Max(n, 1), [_fill(buffers["other"])])
    function = _run(buffers, [read, overwrite, independent_loop, overwrite], scalars=[n])

    # Both the first exit and every back edge preserve the completion already
    # established before entry. A conservative exit must not forget it.
    outside = _outside_loops(function)
    assert _events(outside).count("MTE3_V") == 1
    _assert_handoff(outside, "MTE3_V", read, overwrite)
    _assert_no_full_barrier(function)


@pytest.mark.parametrize("wrapped", [False, True])
def test_rebuilding_keeps_an_outer_back_edge_handoff_inside_the_inner_loop(wrapped):
    buffers = _buffers()
    initial_write = _copy("copy_gm_to_ub", buffers["input"], buffers["a"])
    read = _copy("copy_ub_to_gm", buffers["a"], buffers["output"])
    inner_body = tir.LetStmt(tir.Var("unused", "int32"), 0, read) if wrapped else read
    inner = _loop(2, [inner_body], "j")
    write = _fill(buffers["a"])
    function = _run(buffers, [initial_write, _loop(2, [inner, write])], b_offset=256)
    loops = _loops(function)

    assert len(loops) == 2
    # The first outer iteration needs MTE2 -> MTE3. Later ones also need
    # V -> MTE3; choosing only the first generated inner body loses that pair.
    assert {"MTE2_MTE3", "V_MTE3"}.issubset(_events(loops[0].body))
    _assert_handoff(function, "MTE2_MTE3", initial_write, read)
    two_outer_iterations = _operations(loops[1].body) * 2
    _assert_handoff(two_outer_iterations, "V_MTE3", write, read, occurrence=1)
    _assert_no_full_barrier(function)
