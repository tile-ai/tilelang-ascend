# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.

import pytest
import tilelang
import tilelang.language as T
import tvm
from tvm import tir


WAVE_SIZE = 20


def _persistent_kernel(rows, cols=2):
    @T.prim_func
    def main(O: T.Tensor((rows, cols), "int32")):
        with T.Kernel(WAVE_SIZE, is_npu=True) as (cid, _):
            for row, col in T.Persistent([rows, cols], WAVE_SIZE, cid):
                O[row, col] = row * cols + col

    return main


def _persistent_loop(func):
    loops = []

    def collect(node):
        if isinstance(node, tir.For):
            loops.append(node)

    tir.stmt_functor.post_order_visit(func.body, collect)
    assert len(loops) == 1
    return loops[0]


def _loop_break_calls(stmt):
    calls = []

    def collect(node):
        if isinstance(node, tir.Call) and getattr(node.op, "name", None) == "tl.loop_break":
            calls.append(node)

    tir.stmt_functor.post_order_visit(stmt, collect)
    return calls


def _block_index(func):
    indices = []

    def collect(node):
        if (
            isinstance(node, tir.AttrStmt)
            and node.attr_key == "thread_extent"
            and isinstance(node.node, tir.IterVar)
            and node.node.thread_tag == "blockIdx.x"
        ):
            indices.append(node.node.var)

    tir.stmt_functor.post_order_visit(func.body, collect)
    assert len(indices) == 1
    return indices[0]


def _global_index(func, loop):
    return loop.loop_var * WAVE_SIZE + _block_index(func)


def _assert_break_guard_precedes_tile_body(func, loop, domain_size):
    assert isinstance(loop.body, tir.SeqStmt)
    assert len(loop.body.seq) >= 2
    guard = loop.body.seq[0]
    assert isinstance(guard, tir.IfThenElse)
    assert len(_loop_break_calls(guard)) == 1
    expected_condition = tir.LE(domain_size, _global_index(func, loop))
    assert tvm.ir.structural_equal(guard.condition, expected_condition)
    for stmt in loop.body.seq[1:]:
        assert not _loop_break_calls(stmt)


def _assert_tile_body_is_predicated(func, loop, domain_size):
    assert isinstance(loop.body, tir.IfThenElse)
    assert not _loop_break_calls(loop.body)
    expected_condition = tir.LT(_global_index(func, loop), domain_size)
    assert tvm.ir.structural_equal(loop.body.condition, expected_condition)
    assert loop.body.else_case is None


@pytest.mark.parametrize(
    "rows,expected_waves,guard_kind",
    [
        (4, 1, "predicate"),  # 8 tiles: partial first wave (issue #1551).
        (10, 1, "none"),  # 20 tiles: exactly one full wave keeps the fast path.
        (11, 2, "break"),  # 22 tiles: partial final wave.
        (20, 2, "break"),  # 40 tiles: preserve the existing multi-wave guard.
    ],
)
def test_persistent_guard_for_static_domains(rows, expected_waves, guard_kind):
    func = _persistent_kernel(rows)
    loop = _persistent_loop(func)

    assert isinstance(loop.extent, tir.IntImm)
    assert loop.extent.value == expected_waves
    if guard_kind == "break":
        _assert_break_guard_precedes_tile_body(func, loop, rows * 2)
    elif guard_kind == "predicate":
        _assert_tile_body_is_predicated(func, loop, rows * 2)
    else:
        assert guard_kind == "none"
        assert not _loop_break_calls(loop)
        assert not isinstance(loop.body, tir.IfThenElse)


def test_persistent_guard_for_dynamic_domain():
    rows = T.symbolic("rows")
    func = _persistent_kernel(rows)
    loop = _persistent_loop(func)

    _assert_tile_body_is_predicated(func, loop, rows * 2)


@pytest.mark.parametrize(
    "domain,wave_size,index,group_size,error_match",
    [
        ([0, 2], WAVE_SIZE, 0, 8, "domain extents must be positive"),
        ([4, 2], 0, 0, 8, "wave_size must be positive"),
        ([4, 2], WAVE_SIZE, -1, 8, "index must satisfy"),
        ([4, 2], WAVE_SIZE, WAVE_SIZE, 8, "index must satisfy"),
        ([4, 2], WAVE_SIZE, 0, 0, "group_size must be positive"),
        ([4, 3], WAVE_SIZE, 0, 2, "last domain extent must be divisible"),
    ],
)
def test_persistent_rejects_invalid_static_arguments(domain, wave_size, index, group_size, error_match):
    with pytest.raises(tvm.TVMError, match=error_match):
        T.Persistent(domain, wave_size, index, group_size)


@pytest.mark.parametrize("target", ["ascendc", "pto"])
def test_partial_single_wave_guard_reaches_codegen(target):
    artifact = tilelang.lower(_persistent_kernel(4), target=target)

    assert "break;" not in artifact.kernel_source


@pytest.mark.parametrize("target", ["ascendc", "pto"])
def test_multi_wave_break_guard_reaches_codegen(target):
    artifact = tilelang.lower(_persistent_kernel(11), target=target)

    assert artifact.kernel_source.count("break;") == 1
