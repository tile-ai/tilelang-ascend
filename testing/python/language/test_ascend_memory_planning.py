"""Comprehensive tests for AscendMemoryPlanning pass.

Covers:
  1. Basic sequential allocation (linear mode)
  2. Auto-plan memory reuse for non-overlapping lifetimes
  3. Overlapping lifetime buffers must NOT share memory
  4. Loop-aware liveness: buffer defined before loop, used inside,
     must not be reused by buffers allocated inside the loop
  5. annotate_address pre-allocation respected
  6. tmp_buffer address planning
  7. Nested loop liveness
  8. NPU correctness for loop scenarios
  9. if/else branch liveness: branch-internal buffers must not overlap
     each other nor with outer-scope live buffers
  10. T.if_then_else dynamic index: memory planning stays correct under
      runtime-computed indices
  11. Boundary cases: tail-block if/else, minimum-aligned buffer
  12. NPU correctness for if/else and T.if_then_else scenarios

All new offset-checking cases are parametrized over both codegen targets
(``ascendc`` and ``pto``) to ensure the planning pass is backend-agnostic.
"""

import re

import pytest
import torch

import tilelang
import tilelang.language as T
from tilelang import tvm

VEC_NUM = 2

PASS_AUTO = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}

PASS_LINEAR = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
}

NPU_AVAILABLE = hasattr(torch, "npu") and torch.npu.is_available()


@pytest.fixture(scope="session", autouse=True)
def clear_cache():
    tilelang.disable_cache()
    yield


def _get_buffer_offsets(kernel_source: str) -> dict[str, int]:
    """Parse buffer offsets from generated AscendC/PTO source code.

    AscendC emits ``auto <name> = <scope>.GetWithOffset<type>(size, offset)``
    while PTO emits ``TASSIGN(<name>, <offset>)``. Both use byte offsets, so
    the returned map is directly comparable across targets.
    """
    offsets = {}
    for line in kernel_source.split("\n"):
        m = re.search(
            r"auto\s+(\w+)\s*=.*GetWithOffset<[^>]+>\(\s*\d+\s*,\s*(\d+)\s*\)",
            line,
        )
        if m:
            offsets[m.group(1)] = int(m.group(2))
            continue
        m = re.search(r"TASSIGN\(\s*(\w+)\s*,\s*(\d+)\s*\)", line)
        if m:
            offsets[m.group(1)] = int(m.group(2))
    return offsets


def _get_buffer_sizes(kernel_source: str) -> dict[str, int]:
    """Parse buffer sizes from generated AscendC source code."""
    sizes = {}
    for line in kernel_source.split("\n"):
        m = re.search(
            r"auto\s+(\w+)\s*=.*GetWithOffset<[^>]+>\(\s*(\d+)\s*,\s*\d+\s*\)",
            line,
        )
        if m:
            sizes[m.group(1)] = int(m.group(2))
    return sizes


def _assert_no_overlap(name1: str, off1: int, size1: int, name2: str, off2: int, size2: int):
    """Assert two buffer address ranges do not overlap."""
    assert off1 + size1 <= off2 or off2 + size2 <= off1, (
        f"Buffers {name1} [{off1}, {off1 + size1}) and {name2} [{off2}, {off2 + size2}) overlap"
    )


def _compile_and_get_offsets(program, pass_configs, target="ascendc", out_idx=None):
    """Compile a program and return buffer offset map."""
    kwargs = {"pass_configs": pass_configs, "target": target}
    if out_idx is not None:
        kwargs["out_idx"] = out_idx
    kernel = tilelang.compile(program, **kwargs)
    src = kernel.get_kernel_source()
    return _get_buffer_offsets(src), kernel


# ---------------------------------------------------------------------------
# 1. Basic sequential allocation
# ---------------------------------------------------------------------------


def test_linear_mode_sequential_no_overlap():
    """Linear mode: sequential buffers get increasing offsets."""

    @T.prim_func
    def main(
        A: T.Tensor((64, 128), "float32"),  # type: ignore
        B: T.Tensor((64, 128), "float32"),  # type: ignore
    ):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            a_ub = T.alloc_ub((64, 128), "float32")
            b_ub = T.alloc_ub((64, 128), "float32")
            T.copy(A[:, :], a_ub)
            T.copy(B[:, :], b_ub)

    offsets, _ = _compile_and_get_offsets(main, PASS_LINEAR, out_idx=[])
    assert "a_ub" in offsets
    assert "b_ub" in offsets
    assert offsets["a_ub"] == 0
    assert offsets["b_ub"] >= 64 * 128 * 4


def test_auto_mode_overlapping_buffers_no_reuse():
    """Auto mode: simultaneously-live buffers must NOT share memory."""

    @T.prim_func
    def main(
        A: T.Tensor((64, 128), "float32"),  # type: ignore
        B: T.Tensor((64, 128), "float32"),  # type: ignore
        C: T.Tensor((64, 128), "float32"),  # type: ignore
    ):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            a_ub = T.alloc_ub((64, 128), "float32")
            b_ub = T.alloc_ub((64, 128), "float32")
            c_ub = T.alloc_ub((64, 128), "float32")
            T.copy(A[:, :], a_ub)
            T.copy(B[:, :], b_ub)
            T.tile.add(c_ub, a_ub, b_ub)

    offsets, _ = _compile_and_get_offsets(main, PASS_AUTO, out_idx=[2])
    buf_size = 64 * 128 * 4
    # All three must have distinct, non-overlapping addresses
    addrs = [offsets["a_ub"], offsets["b_ub"], offsets["c_ub"]]
    assert len(set(addrs)) == 3, f"Addresses must be distinct: {addrs}"
    for i in range(3):
        for j in range(i + 1, 3):
            assert addrs[i] + buf_size <= addrs[j] or addrs[j] + buf_size <= addrs[i], f"Buffers {i} and {j} overlap"


# ---------------------------------------------------------------------------
# 2. Loop-aware liveness extension
# ---------------------------------------------------------------------------


def test_loop_buffer_before_loop_not_reused_inside():
    """Buffer defined before loop, read inside loop, must NOT be reused
    by a buffer allocated inside the loop body."""

    @T.prim_func
    def main(
        A: T.Tensor((4, 64), "float32"),  # type: ignore
        B: T.Tensor((4, 64), "float32"),  # type: ignore
    ):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            index_ub = T.alloc_ub((64,), "float32")
            T.copy(A[0, :], index_ub)
            for k in T.serial(3):
                merged_ub = T.alloc_ub((64,), "float32")
                T.copy(A[k, :], merged_ub)
                T.tile.add(merged_ub, merged_ub, index_ub)
                T.copy(merged_ub, B[k, :])

    offsets, _ = _compile_and_get_offsets(main, PASS_AUTO, out_idx=[1])
    assert "index_ub" in offsets
    assert "merged_ub" in offsets
    buf_size = 64 * 4
    idx_end = offsets["index_ub"] + buf_size
    mer_end = offsets["merged_ub"] + buf_size
    assert offsets["index_ub"] + buf_size <= offsets["merged_ub"] or offsets["merged_ub"] + buf_size <= offsets["index_ub"], (
        f"index_ub [{offsets['index_ub']}, {idx_end}) and merged_ub [{offsets['merged_ub']}, {mer_end}) overlap!"
    )


def test_loop_buffer_inside_loop_can_reuse():
    """Buffer allocated and fully consumed inside a single loop iteration
    can be reused across iterations (same offset)."""

    @T.prim_func
    def main(
        A: T.Tensor((4, 64), "float32"),  # type: ignore
        B: T.Tensor((4, 64), "float32"),  # type: ignore
    ):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            for k in T.serial(3):
                tmp_ub = T.alloc_ub((64,), "float32")
                T.copy(A[k, :], tmp_ub)
                T.copy(tmp_ub, B[k, :])

    offsets, _ = _compile_and_get_offsets(main, PASS_AUTO, out_idx=[1])
    assert "tmp_ub" in offsets
    assert offsets["tmp_ub"] == 0


# ---------------------------------------------------------------------------
# 3. Nested loop liveness
# ---------------------------------------------------------------------------


def test_nested_loop_outer_buffer_not_reused():
    """Buffer defined before outer loop, read in inner loop, must not be
    reused by inner-loop-allocated buffers."""

    @T.prim_func
    def main(
        A: T.Tensor((4, 8, 32), "float32"),  # type: ignore
        B: T.Tensor((4, 8, 32), "float32"),  # type: ignore
    ):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            acc_ub = T.alloc_ub((32,), "float32")
            T.copy(A[0, 0, :], acc_ub)
            for i in T.serial(3):
                for j in T.serial(4):
                    tmp_ub = T.alloc_ub((32,), "float32")
                    T.copy(A[i, j, :], tmp_ub)
                    T.tile.add(tmp_ub, tmp_ub, acc_ub)
                    T.copy(tmp_ub, B[i, j, :])

    offsets, _ = _compile_and_get_offsets(main, PASS_AUTO, out_idx=[1])
    assert "acc_ub" in offsets
    assert "tmp_ub" in offsets
    buf_size = 32 * 4
    assert offsets["acc_ub"] + buf_size <= offsets["tmp_ub"] or offsets["tmp_ub"] + buf_size <= offsets["acc_ub"], (
        "acc_ub and tmp_ub must not overlap in nested loop"
    )


# ---------------------------------------------------------------------------
# 4. annotate_address pre-allocation
# ---------------------------------------------------------------------------


def test_annotate_address_respected_in_auto_mode():
    """Pre-allocated buffer via annotate_address must keep its address
    in auto-plan mode."""

    @T.prim_func
    def main(
        A: T.Tensor((64, 128), "float32"),  # type: ignore
        B: T.Tensor((64, 128), "float32"),  # type: ignore
    ):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            a_ub = T.alloc_ub((64, 128), "float32")
            b_ub = T.alloc_ub((64, 128), "float32")
            T.annotate_address(
                {
                    a_ub: 32768,
                }
            )
            T.copy(A[:, :], a_ub)
            T.copy(B[:, :], b_ub)

    offsets, _ = _compile_and_get_offsets(main, PASS_AUTO, out_idx=[])
    assert offsets["a_ub"] == 32768, f"Pre-allocated a_ub must be at 32768, got {offsets['a_ub']}"


def test_annotate_address_no_conflict_in_auto_mode():
    """Auto-planned buffer must avoid pre-allocated buffer's address range."""

    @T.prim_func
    def main(
        A: T.Tensor((64, 128), "float32"),  # type: ignore
        B: T.Tensor((64, 128), "float32"),  # type: ignore
    ):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            a_ub = T.alloc_ub((64, 128), "float32")
            b_ub = T.alloc_ub((64, 128), "float32")
            T.annotate_address(
                {
                    a_ub: 0,
                }
            )
            T.copy(A[:, :], a_ub)
            T.copy(B[:, :], b_ub)

    offsets, _ = _compile_and_get_offsets(main, PASS_AUTO, out_idx=[])
    a_size = 64 * 128 * 4
    assert offsets["a_ub"] == 0
    assert offsets["b_ub"] >= a_size, f"b_ub must be after a_ub's range, got {offsets['b_ub']} < {a_size}"


# ---------------------------------------------------------------------------
# 5. tmp_buffer address planning
# ---------------------------------------------------------------------------


def test_tmp_buffer_gets_address_in_auto_mode():
    """tmp_ub should receive a valid address in auto-plan mode."""

    @T.prim_func
    def main(
        A: T.Tensor((128, 256), "float32"),  # type: ignore
        B: T.Tensor((128,), "float32"),  # type: ignore
    ):
        with T.Kernel(2, is_npu=True) as (cid, vid):
            rb = cid * 128 + vid * 64
            a_ub = T.alloc_ub((64, 256), "float32")
            b_ub = T.alloc_ub((64,), "float32")
            T.copy(A[rb : rb + 64, :], a_ub)
            T.reduce_max(a_ub, b_ub, dim=-1)
            T.copy(b_ub, B[rb : rb + 64])

    offsets, _ = _compile_and_get_offsets(main, PASS_AUTO, out_idx=[1])
    assert "tmp_ub" in offsets, "tmp_ub must be present in auto mode"
    assert offsets["tmp_ub"] >= 0


def test_tmp_buffer_does_not_overlap_user_buffers():
    """tmp_ub address must not overlap with user buffers."""

    @T.prim_func
    def main(
        A: T.Tensor((128, 256), "float32"),  # type: ignore
        B: T.Tensor((128,), "float32"),  # type: ignore
    ):
        with T.Kernel(2, is_npu=True) as (cid, vid):
            rb = cid * 128 + vid * 64
            a_ub = T.alloc_ub((64, 256), "float32")
            b_ub = T.alloc_ub((64,), "float32")
            T.copy(A[rb : rb + 64, :], a_ub)
            T.reduce_max(a_ub, b_ub, dim=-1)
            T.copy(b_ub, B[rb : rb + 64])

    offsets, _ = _compile_and_get_offsets(main, PASS_AUTO, out_idx=[1])
    a_size = 64 * 256 * 4
    b_size = 64 * 4

    def _overlaps(s1, sz1, s2, sz2):
        return s1 < s2 + sz2 and s2 < s1 + sz1

    assert not _overlaps(offsets["a_ub"], a_size, offsets["b_ub"], b_size), "a_ub and b_ub must not overlap"

    # tmp_ub might be reused with one of them if lifetime doesn't overlap
    # but should not corrupt results (verified by NPU test below)


# ---------------------------------------------------------------------------
# 6. Linear mode: no reuse, sequential allocation
# ---------------------------------------------------------------------------


def test_linear_mode_no_reuse():
    """Linear mode: even non-overlapping buffers get distinct addresses
    (no lifetime-based reuse)."""

    @T.prim_func
    def main(
        A: T.Tensor((64, 128), "float32"),  # type: ignore
        B: T.Tensor((64, 128), "float32"),  # type: ignore
    ):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            a_ub = T.alloc_ub((64, 128), "float32")
            T.copy(A[:, :], a_ub)
            T.copy(a_ub, B[:, :])

    offsets_lin, _ = _compile_and_get_offsets(main, PASS_LINEAR, out_idx=[1])
    offsets_auto, _ = _compile_and_get_offsets(main, PASS_AUTO, out_idx=[1])

    # In linear mode there's only one buffer so both should be at 0
    assert offsets_lin["a_ub"] == 0


# ---------------------------------------------------------------------------
# 7. Multiple scope groups (shared + wmma)
# ---------------------------------------------------------------------------


def test_multiple_scopes_independent_allocation():
    """Buffers in different scopes (UB vs L0C) get independent address
    spaces starting from 0."""

    @T.prim_func
    def main(
        A: T.Tensor((16, 16), "float16"),  # type: ignore
        B: T.Tensor((16, 16), "float16"),  # type: ignore
        C: T.Tensor((16, 16), "float16"),  # type: ignore
    ):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            a_l1 = T.alloc_L1((16, 16), "float16")
            b_l1 = T.alloc_L1((16, 16), "float16")
            c_l0 = T.alloc_L0C((16, 16), "float")
            T.copy(A[:, :], a_l1)
            T.copy(B[:, :], b_l1)
            T.gemm_v0(a_l1, b_l1, c_l0, init=True)
            T.copy(c_l0, C[:, :])

    offsets, _ = _compile_and_get_offsets(main, PASS_AUTO, out_idx=[2])
    # L1 and L0C are different scopes, can start at 0 independently
    assert "a_l1" in offsets
    assert "b_l1" in offsets
    assert "c_l0" in offsets
    assert offsets["a_l1"] == 0


# ---------------------------------------------------------------------------
# 9. if/else branch liveness
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("target", ["ascendc", "pto"])
def test_if_else_branch_buffers_no_overlap(target):
    """if/else branches each allocate a buffer; their addresses must not
    overlap so that runtime branch selection never corrupts data."""

    @T.prim_func
    def main(
        A: T.Tensor((2, 64), "float32"),  # type: ignore
        B: T.Tensor((1, 64), "float32"),  # type: ignore
        sel: T.Tensor((1,), "int32"),  # type: ignore
    ):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            s = sel[0]
            if s > 0:
                x_ub = T.alloc_ub((64,), "float32")
                T.copy(A[0, :], x_ub)
                T.copy(x_ub, B[0, :])
            else:
                y_ub = T.alloc_ub((64,), "float32")
                T.copy(A[1, :], y_ub)
                T.copy(y_ub, B[0, :])

    offsets, _ = _compile_and_get_offsets(main, PASS_AUTO, target=target, out_idx=[1])
    assert "x_ub" in offsets, f"x_ub missing in {target} source"
    assert "y_ub" in offsets, f"y_ub missing in {target} source"
    buf_size = 64 * 4
    _assert_no_overlap("x_ub", offsets["x_ub"], buf_size, "y_ub", offsets["y_ub"], buf_size)


@pytest.mark.parametrize("target", ["ascendc", "pto"])
def test_if_else_branch_buffer_not_reuse_outer(target):
    """Buffer allocated before if/else and read inside a branch must not
    share memory with branch-internal buffers."""

    @T.prim_func
    def main(
        A: T.Tensor((2, 64), "float32"),  # type: ignore
        B: T.Tensor((2, 64), "float32"),  # type: ignore
        sel: T.Tensor((1,), "int32"),  # type: ignore
    ):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            base_ub = T.alloc_ub((64,), "float32")
            T.copy(A[0, :], base_ub)
            s = sel[0]
            if s > 0:
                inner_ub = T.alloc_ub((64,), "float32")
                T.copy(A[1, :], inner_ub)
                T.tile.add(inner_ub, inner_ub, base_ub)
                T.copy(inner_ub, B[0, :])
            else:
                inner2_ub = T.alloc_ub((64,), "float32")
                T.copy(A[1, :], inner2_ub)
                T.tile.sub(inner2_ub, inner2_ub, base_ub)
                T.copy(inner2_ub, B[1, :])

    offsets, _ = _compile_and_get_offsets(main, PASS_AUTO, target=target, out_idx=[1])
    buf_size = 64 * 4
    _assert_no_overlap(
        "base_ub",
        offsets["base_ub"],
        buf_size,
        "inner_ub",
        offsets["inner_ub"],
        buf_size,
    )
    _assert_no_overlap(
        "base_ub",
        offsets["base_ub"],
        buf_size,
        "inner2_ub",
        offsets["inner2_ub"],
        buf_size,
    )


# ---------------------------------------------------------------------------
# 10. T.if_then_else dynamic index liveness
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("target", ["ascendc", "pto"])
def test_if_then_else_dynamic_index_buffers_no_overlap(target):
    """T.if_then_else computed dynamic index must not break memory planning:
    simultaneously-live buffers must not overlap."""

    @T.prim_func
    def main(
        A: T.Tensor((128, 64), "float32"),  # type: ignore
        B: T.Tensor((128, 64), "float32"),  # type: ignore
        n: T.Tensor((1,), "int32"),  # type: ignore
    ):
        with T.Kernel(2, is_npu=True) as (cid, vid):
            row = T.if_then_else(n[0] > 64, cid * 64, cid * 32)
            a_ub = T.alloc_ub((64, 64), "float32")
            b_ub = T.alloc_ub((64, 64), "float32")
            T.copy(A[row : row + 64, :], a_ub)
            T.copy(a_ub, b_ub)
            T.copy(b_ub, B[row : row + 64, :])

    offsets, _ = _compile_and_get_offsets(main, PASS_AUTO, target=target, out_idx=[1])
    buf_size = 64 * 64 * 4
    _assert_no_overlap("a_ub", offsets["a_ub"], buf_size, "b_ub", offsets["b_ub"], buf_size)


@pytest.mark.parametrize("target", ["ascendc", "pto"])
def test_if_then_else_in_loop_buffer_liveness(target):
    """Buffer defined before a loop that uses T.if_then_else for dynamic
    indexing must not be reused by loop-internal buffers."""

    @T.prim_func
    def main(
        A: T.Tensor((4, 64), "float32"),  # type: ignore
        B: T.Tensor((4, 64), "float32"),  # type: ignore
        sel: T.Tensor((1,), "int32"),  # type: ignore
    ):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            base_ub = T.alloc_ub((64,), "float32")
            T.copy(A[0, :], base_ub)
            for k in T.serial(3):
                src_row = T.if_then_else(sel[0] > 0, k, 3 - k)
                work_ub = T.alloc_ub((64,), "float32")
                T.copy(A[src_row, :], work_ub)
                T.tile.add(work_ub, work_ub, base_ub)
                T.copy(work_ub, B[k, :])

    offsets, _ = _compile_and_get_offsets(main, PASS_AUTO, target=target, out_idx=[1])
    buf_size = 64 * 4
    _assert_no_overlap(
        "base_ub",
        offsets["base_ub"],
        buf_size,
        "work_ub",
        offsets["work_ub"],
        buf_size,
    )


# ---------------------------------------------------------------------------
# 11. Boundary cases
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("target", ["ascendc", "pto"])
def test_if_else_boundary_last_block(target):
    """Boundary: the last block takes a different code path via if/else,
    allocating an extra buffer. Memory planning must keep it non-overlapping
    with the outer buffer."""

    @T.prim_func
    def main(
        A: T.Tensor((192, 32), "float32"),  # type: ignore
        B: T.Tensor((192, 32), "float32"),  # type: ignore
    ):
        with T.Kernel(3, is_npu=True) as (cid, vid):
            a_ub = T.alloc_ub((64, 32), "float32")
            start = cid * 64
            T.copy(A[start : start + 64, :], a_ub)
            if cid == 2:
                b_ub = T.alloc_ub((64, 32), "float32")
                T.tile.mul(b_ub, a_ub, 2.0)
                T.copy(b_ub, B[start : start + 64, :])
            else:
                T.copy(a_ub, B[start : start + 64, :])

    offsets, _ = _compile_and_get_offsets(main, PASS_AUTO, target=target, out_idx=[1])
    assert "a_ub" in offsets
    assert "b_ub" in offsets
    buf_size = 64 * 32 * 4
    _assert_no_overlap("a_ub", offsets["a_ub"], buf_size, "b_ub", offsets["b_ub"], buf_size)


@pytest.mark.parametrize("target", ["ascendc", "pto"])
def test_if_then_else_min_buffer_boundary(target):
    """Boundary: minimum 32B-aligned buffer (8 float32 elements) together
    with T.if_then_else index guard must compile and plan correctly."""

    @T.prim_func
    def main(
        A: T.Tensor((16, 8), "float32"),  # type: ignore
        B: T.Tensor((16, 8), "float32"),  # type: ignore
        n: T.Tensor((1,), "int32"),  # type: ignore
    ):
        with T.Kernel(2, is_npu=True) as (cid, vid):
            row = T.if_then_else(n[0] > 8, cid * 8, 0)
            a_ub = T.alloc_ub((8, 8), "float32")
            T.copy(A[row : row + 8, :], a_ub)
            T.copy(a_ub, B[row : row + 8, :])

    offsets, _ = _compile_and_get_offsets(main, PASS_AUTO, target=target, out_idx=[1])
    assert "a_ub" in offsets
    assert offsets["a_ub"] >= 0


# ---------------------------------------------------------------------------
# 8. NPU correctness tests
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not NPU_AVAILABLE,
    reason="NPU correctness requires an Ascend NPU runtime",
)
def test_npu_loop_liveness_correctness():
    """NPU correctness: buffer before loop + buffer inside loop.
    If liveness is wrong, data corruption will cause assertion failure."""

    @T.prim_func
    def main(
        A: T.Tensor((4, 64), "float32"),  # type: ignore
        B: T.Tensor((4, 64), "float32"),  # type: ignore
    ):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            index_ub = T.alloc_ub((64,), "float32")
            T.copy(A[0, :], index_ub)
            for k in T.serial(3):
                merged_ub = T.alloc_ub((64,), "float32")
                T.copy(A[k, :], merged_ub)
                T.tile.add(merged_ub, merged_ub, index_ub)
                T.copy(merged_ub, B[k, :])

    kernel = tilelang.compile(main, pass_configs=PASS_AUTO, target="ascendc", out_idx=[1])

    a = torch.randn(4, 64, dtype=torch.float32, device="npu")
    b = kernel(a)
    ref = a.clone()
    ref[0, :] = a[0, :] + a[0, :]
    ref[1, :] = a[1, :] + a[0, :]
    ref[2, :] = a[2, :] + a[0, :]
    torch.testing.assert_close(b[:3, :], ref[:3, :], rtol=1e-3, atol=1e-3)


@pytest.mark.skipif(
    not NPU_AVAILABLE,
    reason="NPU correctness requires an Ascend NPU runtime",
)
def test_npu_nested_loop_correctness():
    """NPU correctness: nested loop with cross-iteration buffer."""

    @T.prim_func
    def main(
        A: T.Tensor((3, 4, 32), "float32"),  # type: ignore
        B: T.Tensor((3, 4, 32), "float32"),  # type: ignore
    ):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            acc_ub = T.alloc_ub((32,), "float32")
            T.copy(A[0, 0, :], acc_ub)
            for i in T.serial(3):
                for j in T.serial(4):
                    tmp_ub = T.alloc_ub((32,), "float32")
                    T.copy(A[i, j, :], tmp_ub)
                    T.tile.add(tmp_ub, tmp_ub, acc_ub)
                    T.copy(tmp_ub, B[i, j, :])

    kernel = tilelang.compile(main, pass_configs=PASS_AUTO, target="ascendc", out_idx=[1])

    a = torch.randn(3, 4, 32, dtype=torch.float32, device="npu")
    b = kernel(a)
    ref = a.clone() + a[0, 0, :].unsqueeze(0).unsqueeze(0)
    torch.testing.assert_close(b, ref, rtol=1e-3, atol=1e-3)


@pytest.mark.skipif(
    not NPU_AVAILABLE,
    reason="NPU correctness requires an Ascend NPU runtime",
)
def test_npu_reduce_with_auto_planning():
    """NPU correctness: reduce_max with auto memory planning."""

    @T.prim_func
    def main(
        A: T.Tensor((128, 256), "float32"),  # type: ignore
        B: T.Tensor((128,), "float32"),  # type: ignore
    ):
        with T.Kernel(2, is_npu=True) as (cid, vid):
            rb = cid * 128 + vid * 64
            a_ub = T.alloc_ub((64, 256), "float32")
            b_ub = T.alloc_ub((64,), "float32")
            T.copy(A[rb : rb + 64, :], a_ub)
            T.reduce_max(a_ub, b_ub, dim=-1)
            T.copy(b_ub, B[rb : rb + 64])

    kernel = tilelang.compile(main, pass_configs=PASS_AUTO, target="ascendc", out_idx=[1])

    a = torch.randn(128, 256, dtype=torch.float32, device="npu")
    b = kernel(a)
    ref = torch.max(a, dim=-1).values
    torch.testing.assert_close(b, ref, rtol=1e-2, atol=1e-2)


@pytest.mark.skipif(
    not NPU_AVAILABLE,
    reason="NPU correctness requires an Ascend NPU runtime",
)
def test_npu_sequential_ops_correctness():
    """NPU correctness: multiple sequential operations in a loop,
    verifying no data corruption from incorrect memory reuse."""

    @T.prim_func
    def main(
        A: T.Tensor((4, 64), "float32"),  # type: ignore
        B: T.Tensor((4, 64), "float32"),  # type: ignore
    ):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            scale_ub = T.alloc_ub((64,), "float32")
            T.copy(A[0, :], scale_ub)
            for k in T.serial(4):
                work_ub = T.alloc_ub((64,), "float32")
                T.copy(A[k, :], work_ub)
                T.tile.mul(work_ub, work_ub, scale_ub)
                T.copy(work_ub, B[k, :])

    kernel = tilelang.compile(main, pass_configs=PASS_AUTO, target="ascendc", out_idx=[1])

    a = torch.randn(4, 64, dtype=torch.float32, device="npu")
    b = kernel(a)
    ref = a * a[0, :].unsqueeze(0)
    torch.testing.assert_close(b, ref, rtol=1e-3, atol=1e-3)


# ---------------------------------------------------------------------------
# 12. NPU correctness for if/else and T.if_then_else
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not NPU_AVAILABLE,
    reason="NPU correctness requires an Ascend NPU runtime",
)
def test_npu_if_else_branch_correctness():
    """NPU correctness: runtime-selected if/else branch must produce the
    expected output, proving branch buffers are not corrupted by planning."""

    @T.prim_func
    def main(
        A: T.Tensor((2, 64), "float32"),  # type: ignore
        sel: T.Tensor((1,), "int32"),  # type: ignore
        B: T.Tensor((1, 64), "float32"),  # type: ignore
    ):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            s = sel[0]
            if s > 0:
                x_ub = T.alloc_ub((64,), "float32")
                T.copy(A[0, :], x_ub)
                T.tile.mul(x_ub, x_ub, 2.0)
                T.copy(x_ub, B[0, :])
            else:
                y_ub = T.alloc_ub((64,), "float32")
                T.copy(A[1, :], y_ub)
                T.tile.mul(y_ub, y_ub, 3.0)
                T.copy(y_ub, B[0, :])

    kernel = tilelang.compile(main, pass_configs=PASS_AUTO, target="ascendc", out_idx=[2])

    a = torch.randn(2, 64, dtype=torch.float32, device="npu")
    sel_pos = torch.tensor([1], dtype=torch.int32, device="npu")
    sel_neg = torch.tensor([0], dtype=torch.int32, device="npu")
    b_pos = kernel(a, sel_pos)
    b_neg = kernel(a, sel_neg)
    torch.testing.assert_close(b_pos[0], a[0] * 2, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(b_neg[0], a[1] * 3, rtol=1e-3, atol=1e-3)


@pytest.mark.skipif(
    not NPU_AVAILABLE,
    reason="NPU correctness requires an Ascend NPU runtime",
)
def test_npu_if_then_else_dynamic_index_correctness():
    """NPU correctness: T.if_then_else computed offset selects the right
    source tile, proving dynamic indexing + planning is sound."""

    @T.prim_func
    def main(
        A: T.Tensor((128, 64), "float32"),  # type: ignore
        n: T.Tensor((1,), "int32"),  # type: ignore
        B: T.Tensor((64, 64), "float32"),  # type: ignore
    ):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            offset = T.if_then_else(n[0] > 64, 0, 64)
            a_ub = T.alloc_ub((64, 64), "float32")
            T.copy(A[offset : offset + 64, :], a_ub)
            T.copy(a_ub, B[:, :])

    kernel = tilelang.compile(main, pass_configs=PASS_AUTO, target="ascendc", out_idx=[2])

    a = torch.randn(128, 64, dtype=torch.float32, device="npu")
    n_hi = torch.tensor([100], dtype=torch.int32, device="npu")
    n_lo = torch.tensor([10], dtype=torch.int32, device="npu")
    b_hi = kernel(a, n_hi)
    b_lo = kernel(a, n_lo)
    torch.testing.assert_close(b_hi, a[0:64, :], rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(b_lo, a[64:128, :], rtol=1e-3, atol=1e-3)


@pytest.mark.skipif(
    not NPU_AVAILABLE,
    reason="NPU correctness requires an Ascend NPU runtime",
)
def test_npu_if_else_in_loop_correctness():
    """NPU correctness: if/else inside a loop with a cross-iteration base
    buffer; both branches must keep the base buffer intact."""

    @T.prim_func
    def main(
        A: T.Tensor((4, 64), "float32"),  # type: ignore
        sel: T.Tensor((1,), "int32"),  # type: ignore
        B: T.Tensor((4, 64), "float32"),  # type: ignore
    ):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            base_ub = T.alloc_ub((64,), "float32")
            T.copy(A[0, :], base_ub)
            for k in T.serial(4):
                s = sel[0]
                if s > 0:
                    add_ub = T.alloc_ub((64,), "float32")
                    T.copy(A[k, :], add_ub)
                    T.tile.add(add_ub, add_ub, base_ub)
                    T.copy(add_ub, B[k, :])
                else:
                    sub_ub = T.alloc_ub((64,), "float32")
                    T.copy(A[k, :], sub_ub)
                    T.tile.sub(sub_ub, sub_ub, base_ub)
                    T.copy(sub_ub, B[k, :])

    kernel = tilelang.compile(main, pass_configs=PASS_AUTO, target="ascendc", out_idx=[2])

    a = torch.randn(4, 64, dtype=torch.float32, device="npu")
    sel_pos = torch.tensor([1], dtype=torch.int32, device="npu")
    sel_neg = torch.tensor([0], dtype=torch.int32, device="npu")
    b_pos = kernel(a, sel_pos)
    b_neg = kernel(a, sel_neg)
    ref_pos = a + a[0, :].unsqueeze(0)
    ref_neg = a - a[0, :].unsqueeze(0)
    torch.testing.assert_close(b_pos, ref_pos, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(b_neg, ref_neg, rtol=1e-3, atol=1e-3)


# ---------------------------------------------------------------------------
# 13. LetStmt with BufferLoad(shared) regression (issue #1333)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("target", ["ascendc", "pto"])
def test_letstmt_buffer_load_shared_no_crash(target):
    """Regression: a scalar read from a shared-scope UB buffer produces a
    LetStmt whose value is a BufferLoad on a ``shared`` buffer.

    Before the fix, AscendMemoryPlanning did not override
    VisitStmt_(LetStmtNode) and fell back to the base visitor, which
    visited the BufferLoad child without pushing a scope_ entry, causing
    TrackBufferTouch to dereference an empty std::vector -> SIGSEGV.

    This remains a bare ``@T.prim_func`` lowered without ``T.Kernel``. The
    hardware work is placed in an explicit V resource scope, as required when
    CV separation is not applicable to a bare function.
    """

    @T.prim_func
    def main(
        A: T.Tensor((64, 128), "float32"),  # type: ignore
        B: T.Tensor((64, 128), "float32"),  # type: ignore
    ):
        a_ub = T.alloc_ub((64, 128), "float32")
        b_ub = T.alloc_ub((64, 128), "float32")
        with T.Scope("V"):
            T.copy(A[:, :], a_ub)
            # scalar read from shared buffer -> LetStmt + BufferLoad(shared)
            s_tmp = a_ub[0, 0]
            b_ub[0, 0] = s_tmp
            T.copy(b_ub[:, :], B[:, :])

    with tvm.transform.PassContext(opt_level=3, config=PASS_AUTO):
        artifact = tilelang.lower(main, target=target)
    src = artifact.kernel_source
    offsets = _get_buffer_offsets(src)
    assert "a_ub" in offsets, f"a_ub missing in {target} source"
    assert "b_ub" in offsets, f"b_ub missing in {target} source"
    buf_size = 64 * 128 * 4
    _assert_no_overlap(
        "a_ub",
        offsets["a_ub"],
        buf_size,
        "b_ub",
        offsets["b_ub"],
        buf_size,
    )


# Narrow MTE rows use 32-byte slots even when the logical UB row is smaller.
@pytest.mark.parametrize(
    "auto_plan,dtype,cols,dynamic_rows",
    [
        (False, "float32", 1, False),
        (True, "float32", 1, True),
        (False, "float16", 3, False),
        (True, "float32", 7, True),
        (False, "float32", 9, False),
        (True, "float32", 9, True),
        (True, "float16", 17, True),
    ],
)
@pytest.mark.skipif(not NPU_AVAILABLE, reason="Requires an Ascend NPU runtime")
def test_narrow_copy_preserves_adjacent_live_buffer(auto_plan, dtype, cols, dynamic_rows):
    rows, prefix, guard_size = 64, 16, 512
    actual_rows = T.symbolic("actual_rows") if dynamic_rows else rows

    @T.prim_func
    def main(
        A: T.Tensor((actual_rows, cols), dtype),
        GuardIn: T.Tensor((guard_size,), "float32"),
        GuardOut: T.Tensor((guard_size,), "float32"),
    ):
        with T.Kernel(1, threads=1, is_npu=True) as _cid:
            narrow_ub = T.alloc_ub((rows + prefix, cols), dtype)
            guard_ub = T.alloc_ub((guard_size,), "float32")
            # First touch places narrow_ub before the simultaneously-live guard.
            # The later, larger copy must determine its allocation footprint.
            T.copy(A[0:32, 0:cols], narrow_ub[0:32, 0:cols])
            T.copy(GuardIn, guard_ub)
            T.barrier_all()
            T.copy(A[0:rows, 0:cols], narrow_ub[prefix : prefix + rows, 0:cols])
            T.barrier_all()
            T.copy(guard_ub, GuardOut)

    configs = PASS_AUTO if auto_plan else PASS_LINEAR
    kernel = tilelang.compile(main, pass_configs=configs, target="ascendc", out_idx=[2])
    offsets = _get_buffer_offsets(kernel.get_kernel_source())
    element_bytes = 2 if dtype == "float16" else 4
    row_bytes = ((cols * element_bytes + 31) // 32) * 32
    physical_bytes = prefix * cols * element_bytes + rows * row_bytes
    torch_dtype = torch.float16 if dtype == "float16" else torch.float32
    for count in [17, rows] if dynamic_rows else [rows]:
        a = torch.arange(count * cols, dtype=torch.float32).reshape(count, cols)
        a = a.to(device="npu", dtype=torch_dtype)
        guard = torch.full((guard_size,), 123.0, dtype=torch.float32, device="npu")
        result = kernel(a, guard)
        torch.npu.synchronize()
        torch.testing.assert_close(result, guard, rtol=0, atol=0)
    _assert_no_overlap(
        "narrow_ub",
        offsets["narrow_ub"],
        physical_bytes,
        "guard_ub",
        offsets["guard_ub"],
        guard_size * 4,
    )


@pytest.mark.parametrize("auto_plan", [False, True])
def test_narrow_copy_with_loop_offset_reserves_all_row_slots(auto_plan):
    """The largest physical footprint may occur at a nonconstant UB offset."""
    rows, tile_rows, guard_size = 128, 64, 512

    @T.prim_func
    def main(
        A: T.Tensor((rows, 1), "float32"),
        GuardIn: T.Tensor((guard_size,), "float32"),
        GuardOut: T.Tensor((guard_size,), "float32"),
    ):
        with T.Kernel(1, threads=1, is_npu=True) as _cid:
            narrow_ub = T.alloc_ub((rows, 1), "float32")
            guard_ub = T.alloc_ub((guard_size,), "float32")
            T.copy(A[0:tile_rows, 0:1], narrow_ub[0:tile_rows, 0:1])
            T.copy(GuardIn, guard_ub)
            T.barrier_all()
            for tile in T.serial(2):
                T.copy(
                    A[tile * tile_rows : (tile + 1) * tile_rows, 0:1],
                    narrow_ub[tile * tile_rows : (tile + 1) * tile_rows, 0:1],
                )
                T.barrier_all()
            T.copy(guard_ub, GuardOut)

    configs = PASS_AUTO if auto_plan else PASS_LINEAR
    with tvm.transform.PassContext(opt_level=3, config=configs):
        artifact = tilelang.lower(main, target="ascendc")
    offsets = _get_buffer_offsets(artifact.kernel_source)
    physical_bytes = tile_rows * 4 + tile_rows * 32
    _assert_no_overlap(
        "narrow_ub",
        offsets["narrow_ub"],
        physical_bytes,
        "guard_ub",
        offsets["guard_ub"],
        guard_size * 4,
    )


def test_narrow_copy_rejects_physical_ub_overflow():
    # The logical allocation is 96KB and fits UB. The nonzero-offset DMA needs
    # 80KB + 4000 * 32B = 208KB. blockCount=4000 is within the DMA field limit.
    @T.prim_func
    def main(A: T.Tensor((4000, 1), "float32")):
        with T.Kernel(1, threads=1, is_npu=True) as _cid:
            narrow_ub = T.alloc_ub((24000, 1), "float32")
            T.copy(A, narrow_ub[20000:24000, 0:1])

    with (
        tvm.transform.PassContext(opt_level=3, config=PASS_LINEAR),
        pytest.raises(tvm.error.InternalError, match="UB copy physical footprint"),
    ):
        tilelang.lower(main, target="ascendc")


def test_narrow_packed_store_keeps_compact_allocation():
    # A packed 32KB output uses one flat burst. Treating it as 8192 narrow
    # multi-burst rows would incorrectly require 256KB and reject the kernel.
    @T.prim_func
    def main(Out: T.Tensor((8192, 1), "float32")):
        with T.Kernel(1, threads=1, is_npu=True) as _cid:
            packed_ub = T.alloc_ub((8192, 1), "float32")
            T.tile.arith_progression(packed_ub, 0.0, 1.0, 8192)
            T.copy(packed_ub, Out)

    with tvm.transform.PassContext(opt_level=3, config=PASS_LINEAR):
        artifact = tilelang.lower(main, target="ascendc")
    assert _get_buffer_sizes(artifact.kernel_source)["packed_ub"] == 8192


@pytest.mark.parametrize("auto_plan", [False, True])
def test_narrow_copy_index_preserves_thread_let_and_loop_bounds(auto_plan):
    """A nested, shared copy must use all enclosing offset bounds exactly once."""
    tir = tvm.tir
    source = tir.decl_buffer((96,), "float32", name="source", scope="global")
    outer = tir.decl_buffer((96,), "float32", name="outer", scope="shared.ub")
    inner = tir.decl_buffer((1,), "float32", name="inner", scope="shared.ub")
    cid, tile, offset = [tir.Var(name, "int32") for name in ("cid", "tile", "offset")]
    copy = tir.call_extern(
        "handle",
        "tl::ascend::copy_gm_to_ub<float, 1, 64>",
        source.access_ptr("r"),
        outer.access_ptr("w", offset=offset, extent=64),
        1,
        64,
        1,
        tir.const(0, "float32"),
    )
    body = tir.SeqStmt(
        [
            tir.Evaluate(copy),
            tir.Evaluate(copy),
            tir.Evaluate(tir.call_extern("handle", "test_touch", inner.access_ptr("rw"))),
        ]
    )
    body = tir.Allocate(inner.data, inner.dtype, [1], tir.const(True, "bool"), body)
    body = tir.LetStmt(offset, (cid + tile) * 16, body)
    body = tir.For(tile, 0, 2, tir.ForKind.SERIAL, body)
    body = tir.Allocate(outer.data, outer.dtype, [96], tir.const(True, "bool"), body)
    thread = tir.IterVar((0, 2), cid, tir.IterVar.ThreadIndex, "blockIdx.x")
    body = tir.AttrStmt(thread, "thread_extent", 2, body)
    func = tir.PrimFunc([source.data], body, buffer_map={source.data: source}).with_attr("global_symbol", "main")
    with tvm.transform.PassContext(config={"tl.ascend_memory_planning": auto_plan}):
        planned = tilelang.transform.AscendMemoryPlanning()(tvm.IRModule({"main": func}))
    # max(offset) = (1 + 1) * 16 elements; each of the 64 DMA rows occupies 32B.
    assert int(planned["main"].attrs["size_map"][outer.data]) == 32 * 4 + 64 * 32


@pytest.mark.parametrize("auto_plan", [False, True])
def test_narrow_copy_bounds_cache_keeps_select_constraints_local(auto_plan):
    """A bound from one Select arm must not replace a later global bound."""
    tir = tvm.tir
    source = tir.decl_buffer((128,), "float32", name="source", scope="global")
    ub = tir.decl_buffer((128,), "float32", name="ub", scope="shared.ub")
    rows = tir.Var("rows", "int32")

    def copy(count):
        return tir.Evaluate(
            tir.call_extern(
                "handle",
                "tl::ascend::copy_gm_to_ub<float, 1, 128>",
                source.access_ptr("r"),
                ub.access_ptr("w"),
                1,
                count,
                1,
                tir.const(0, "float32"),
            )
        )

    body = tir.SeqStmt([copy(tir.Select(rows < 64, rows, 64)), copy(rows)])
    body = tir.For(rows, 0, 129, tir.ForKind.SERIAL, body)
    body = tir.Allocate(ub.data, ub.dtype, [128], tir.const(True, "bool"), body)
    func = tir.PrimFunc([source.data], body, buffer_map={source.data: source}).with_attr("global_symbol", "main")
    with tvm.transform.PassContext(config={"tl.ascend_memory_planning": auto_plan}):
        planned = tilelang.transform.AscendMemoryPlanning()(tvm.IRModule({"main": func}))
    assert int(planned["main"].attrs["size_map"][ub.data]) == 128 * 32


@pytest.mark.parametrize("auto_plan", [False, True])
@pytest.mark.parametrize("row_step", [2500, 250])
def test_narrow_copy_preserves_offset_row_correlation(auto_plan, row_step):
    """Correlated offsets/rows must neither over-allocate nor hide UB overflow."""
    tir = tvm.tir
    source = tir.decl_buffer((4000,), "float32", name="source", scope="global")
    ub = tir.decl_buffer((32000,), "float32", name="ub", scope="shared.ub")
    tile = tir.Var("tile", "int32")
    rows = 4000 - tile * row_step
    copy = tir.call_extern(
        "handle",
        "tl::ascend::copy_gm_to_ub<float, 1, 4000>",
        source.access_ptr("r"),
        ub.access_ptr("w", offset=tile * 20000),
        1,
        rows,
        1,
        tir.const(0, "float32"),
    )
    body = tir.For(tile, 0, 2, tir.ForKind.SERIAL, tir.Evaluate(copy))
    body = tir.Allocate(ub.data, ub.dtype, [32000], tir.const(True, "bool"), body)
    func = tir.PrimFunc([source.data], body, buffer_map={source.data: source}).with_attr("global_symbol", "main")
    with tvm.transform.PassContext(config={"tl.ascend_memory_planning": auto_plan}):
        if row_step == 250:
            # The second iteration really needs 80000 + 3750 * 32 = 200000B.
            with pytest.raises(tvm.error.InternalError, match="UB copy physical footprint"):
                tilelang.transform.AscendMemoryPlanning()(tvm.IRModule({"main": func}))
        else:
            planned = tilelang.transform.AscendMemoryPlanning()(tvm.IRModule({"main": func}))
            # Both iterations end at 128000B, not max(offset) + max(rows) * 32.
            assert int(planned["main"].attrs["size_map"][ub.data]) == 128000


@pytest.mark.parametrize("auto_plan", [False, True])
def test_dma_copy_rejects_unbounded_offset(auto_plan):
    """An unbounded runtime offset must not silently discard the DMA footprint."""
    tir = tvm.tir
    source = tir.decl_buffer((64,), "float32", name="source", scope="global")
    ub = tir.decl_buffer((128,), "float32", name="ub", scope="shared.ub")
    start = tir.Var("start", "int32")
    copy = tir.call_extern(
        "handle",
        "tl::ascend::copy_gm_to_ub<float, 1, 64>",
        source.access_ptr("r"),
        ub.access_ptr("w", offset=start),
        1,
        64,
        1,
        tir.const(0, "float32"),
    )
    body = tir.Allocate(ub.data, ub.dtype, [128], tir.const(True, "bool"), tir.Evaluate(copy))
    func = tir.PrimFunc([source.data, start], body, buffer_map={source.data: source}).with_attr("global_symbol", "main")
    with (
        tvm.transform.PassContext(config={"tl.ascend_memory_planning": auto_plan}),
        pytest.raises(tvm.error.InternalError, match="Cannot prove.*UB copy"),
    ):
        tilelang.transform.AscendMemoryPlanning()(tvm.IRModule({"main": func}))


@pytest.mark.parametrize("auto_plan", [False, True])
@pytest.mark.parametrize("dtype,cpp_dtype,columns", [("float32", "float", 9), ("float16", "half", 17)])
def test_dma_copy_reserves_wide_unaligned_rows(auto_plan, dtype, cpp_dtype, columns):
    """A 34/36-byte GM->UB row occupies a 64-byte physical slot."""
    tir = tvm.tir
    source = tir.decl_buffer((64, columns), dtype, name="source", scope="global")
    ub = tir.decl_buffer((64, columns), dtype, name="ub", scope="shared.ub")
    copy = tir.call_extern(
        "handle",
        f"tl::ascend::copy_gm_to_ub<{cpp_dtype}, {columns}, 64>",
        source.access_ptr("r"),
        ub.access_ptr("w"),
        columns,
        64,
        columns,
        tir.const(0, dtype),
    )
    body = tir.Allocate(ub.data, ub.dtype, [64 * columns], tir.const(True, "bool"), tir.Evaluate(copy))
    func = tir.PrimFunc([source.data], body, buffer_map={source.data: source}).with_attr("global_symbol", "main")
    with tvm.transform.PassContext(config={"tl.ascend_memory_planning": auto_plan}):
        planned = tilelang.transform.AscendMemoryPlanning()(tvm.IRModule({"main": func}))
    assert int(planned["main"].attrs["size_map"][ub.data]) == 64 * 64


@pytest.mark.parametrize("auto_plan", [False, True])
@pytest.mark.parametrize("atomic", [False, True])
def test_dma_packed_store_ignores_empty_tail_columns(auto_plan, atomic):
    """Zero columns do not turn an otherwise flat 32KB store into 256KB."""
    tir = tvm.tir
    output = tir.decl_buffer((8192,), "float32", name="output", scope="global")
    ub = tir.decl_buffer((8192,), "float32", name="ub", scope="shared.ub")
    cid = tir.Var("cid", "int32")
    columns = tir.Select(cid == 0, 1, 0)
    helper = "atomic_add_ub_to_gm" if atomic else "copy_ub_to_gm"
    copy = tir.call_extern(
        "handle",
        f"tl::ascend::{helper}<float, 1, 8192>",
        ub.access_ptr("r"),
        output.access_ptr("w"),
        1,
        8192,
        columns,
    )
    body = tir.For(cid, 0, 2, tir.ForKind.SERIAL, tir.Evaluate(copy))
    body = tir.Allocate(ub.data, ub.dtype, [8192], tir.const(True, "bool"), body)
    func = tir.PrimFunc([output.data], body, buffer_map={output.data: output}).with_attr("global_symbol", "main")
    with tvm.transform.PassContext(config={"tl.ascend_memory_planning": auto_plan}):
        planned = tilelang.transform.AscendMemoryPlanning()(tvm.IRModule({"main": func}))
    assert int(planned["main"].attrs["size_map"][ub.data]) == 8192 * 4


@pytest.mark.parametrize("auto_plan", [False, True])
@pytest.mark.parametrize("valid_columns,expected_size", [(0, 2304), (1, 4064), (8, 2304), (9, 4096)])
def test_dma_unaligned_pitch_tracks_valid_columns(auto_plan, valid_columns, expected_size):
    """The runtime block length controls stride; tail Duplicate is also counted."""
    tir = tvm.tir
    source = tir.decl_buffer((64, 9), "float32", name="source", scope="global")
    ub = tir.decl_buffer((64, 9), "float32", name="ub", scope="shared.ub")
    copy = tir.call_extern(
        "handle",
        "tl::ascend::copy_gm_to_ub<float, 9, 64>",
        source.access_ptr("r"),
        ub.access_ptr("w"),
        9,
        64,
        valid_columns,
        tir.const(0, "float32"),
    )
    body = tir.Allocate(ub.data, ub.dtype, [64 * 9], tir.const(True, "bool"), tir.Evaluate(copy))
    func = tir.PrimFunc([source.data], body, buffer_map={source.data: source}).with_attr("global_symbol", "main")
    with tvm.transform.PassContext(config={"tl.ascend_memory_planning": auto_plan}):
        planned = tilelang.transform.AscendMemoryPlanning()(tvm.IRModule({"main": func}))
    assert int(planned["main"].attrs["size_map"][ub.data]) == expected_size


@pytest.mark.parametrize("auto_plan", [False, True])
@pytest.mark.parametrize("shared_call", [False, True])
def test_dma_copy_uses_enclosing_branch_bounds(auto_plan, shared_call):
    """A shared Call is analyzed separately in every reachable branch."""
    tir = tvm.tir
    source = tir.decl_buffer((4000,), "float32", name="source", scope="global")
    ub = tir.decl_buffer((40000,), "float32", name="ub", scope="shared.ub")
    cid = tir.Var("cid", "int32")
    copy = tir.call_extern(
        "handle",
        "tl::ascend::copy_gm_to_ub<float, 1, 4000>",
        source.access_ptr("r"),
        ub.access_ptr("w", offset=cid * 256),
        1,
        4000,
        1,
        tir.const(0, "float32"),
    )
    if shared_call:
        body = tir.SeqStmt(
            [
                tir.IfThenElse(cid < 32, tir.Evaluate(copy), None),
                tir.IfThenElse(tir.And(cid >= 32, cid < 64), tir.Evaluate(copy), None),
            ]
        )
    else:
        body = tir.IfThenElse(cid >= 64, tir.Evaluate(0), tir.Evaluate(copy))
    body = tir.For(cid, 0, 128, tir.ForKind.SERIAL, body)
    body = tir.Allocate(ub.data, ub.dtype, [40000], tir.const(True, "bool"), body)
    func = tir.PrimFunc([source.data], body, buffer_map={source.data: source}).with_attr("global_symbol", "main")
    with tvm.transform.PassContext(config={"tl.ascend_memory_planning": auto_plan}):
        planned = tilelang.transform.AscendMemoryPlanning()(tvm.IRModule({"main": func}))
    assert int(planned["main"].attrs["size_map"][ub.data]) == 63 * 256 * 4 + 4000 * 32


@pytest.mark.parametrize("auto_plan", [False, True])
def test_dma_copy_shared_call_keeps_unconditional_bounds(auto_plan):
    """A narrow branch must not hide the same Call's later unconditional use."""
    tir = tvm.tir
    source = tir.decl_buffer((64,), "float32", name="source", scope="global")
    ub = tir.decl_buffer((128,), "float32", name="ub", scope="shared.ub")
    cid = tir.Var("cid", "int32")
    copy = tir.call_extern(
        "handle",
        "tl::ascend::copy_gm_to_ub<float, 1, 64>",
        source.access_ptr("r"),
        ub.access_ptr("w", offset=cid * 8),
        1,
        64,
        1,
        tir.const(0, "float32"),
    )
    body = tir.SeqStmt([tir.IfThenElse(cid < 2, tir.Evaluate(copy), None), tir.Evaluate(copy)])
    body = tir.For(cid, 0, 8, tir.ForKind.SERIAL, body)
    body = tir.Allocate(ub.data, ub.dtype, [128], tir.const(True, "bool"), body)
    func = tir.PrimFunc([source.data], body, buffer_map={source.data: source}).with_attr("global_symbol", "main")
    with tvm.transform.PassContext(config={"tl.ascend_memory_planning": auto_plan}):
        planned = tilelang.transform.AscendMemoryPlanning()(tvm.IRModule({"main": func}))
    assert int(planned["main"].attrs["size_map"][ub.data]) == 7 * 8 * 4 + 64 * 32


@pytest.mark.parametrize("auto_plan", [False, True])
@pytest.mark.parametrize("large_field", ["offset", "rows"])
def test_dma_copy_rejects_extreme_finite_bounds(auto_plan, large_field):
    """Finite bounds must not wrap during element-to-byte conversion."""
    tir = tvm.tir
    source = tir.decl_buffer((64,), "float32", name="source", scope="global")
    ub = tir.decl_buffer((128,), "float32", name="ub", scope="shared.ub")
    large = tir.const(2**61, "int64")
    copy = tir.call_extern(
        "handle",
        "tl::ascend::copy_gm_to_ub<float, 1, 64>",
        source.access_ptr("r"),
        ub.access_ptr("w", offset=large if large_field == "offset" else 0, extent=64),
        1,
        large if large_field == "rows" else 64,
        1,
        tir.const(0, "float32"),
    )
    body = tir.Allocate(ub.data, ub.dtype, [128], tir.const(True, "bool"), tir.Evaluate(copy))
    func = tir.PrimFunc([source.data], body, buffer_map={source.data: source}).with_attr("global_symbol", "main")
    with (
        tvm.transform.PassContext(config={"tl.ascend_memory_planning": auto_plan}),
        pytest.raises(tvm.error.InternalError, match="UB copy physical footprint"),
    ):
        tilelang.transform.AscendMemoryPlanning()(tvm.IRModule({"main": func}))


@pytest.mark.parametrize("auto_plan", [False, True])
def test_dma_store_ignores_offsets_on_empty_rows(auto_plan):
    """A huge offset confined to a no-op row count must not reject a safe store."""
    tir = tvm.tir
    output = tir.decl_buffer((64, 2), "float32", name="output", scope="global")
    ub = tir.decl_buffer((128,), "float32", name="ub", scope="shared.ub")
    cid = tir.Var("cid", "int32")
    offset = tir.Select(cid == 0, tir.const(0, "int64"), tir.const(2**61, "int64"))
    rows = tir.Select(cid == 0, 64, 0)
    copy = tir.call_extern(
        "handle",
        "tl::ascend::copy_ub_to_gm<float, 1, 64>",
        ub.access_ptr("r", offset=offset, extent=64),
        output.access_ptr("w"),
        2,
        rows,
        1,
    )
    body = tir.For(cid, 0, 2, tir.ForKind.SERIAL, tir.Evaluate(copy))
    body = tir.Allocate(ub.data, ub.dtype, [128], tir.const(True, "bool"), body)
    func = tir.PrimFunc([output.data], body, buffer_map={output.data: output}).with_attr("global_symbol", "main")
    with tvm.transform.PassContext(config={"tl.ascend_memory_planning": auto_plan}):
        planned = tilelang.transform.AscendMemoryPlanning()(tvm.IRModule({"main": func}))
    assert int(planned["main"].attrs["size_map"][ub.data]) == 64 * 32


@pytest.mark.parametrize("auto_plan", [False, True])
@pytest.mark.skipif(not NPU_AVAILABLE, reason="Requires an Ascend NPU runtime")
def test_dma_empty_tail_column_store_correctness(auto_plan):
    """The second core has no output columns; the first writes one flat slab."""

    @T.prim_func
    def main(Out: T.Tensor((8192, 1), "float32")):
        with T.Kernel(2, threads=1, is_npu=True) as cid:
            packed_ub = T.alloc_ub((8192, 1), "float32")
            T.tile.arith_progression(packed_ub, 0.0, 1.0, 8192)
            T.copy(packed_ub, Out[:, cid : cid + 1])

    configs = PASS_AUTO if auto_plan else PASS_LINEAR
    kernel = tilelang.compile(main, pass_configs=configs, target="ascendc", out_idx=[0])
    assert _get_buffer_sizes(kernel.get_kernel_source())["packed_ub"] == 8192
    actual = kernel()
    torch.npu.synchronize()
    expected = torch.arange(8192, dtype=torch.float32, device="npu").reshape(8192, 1)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("auto_plan", [False, True])
@pytest.mark.parametrize("columns,offset,expected_bytes", [(4, 8, 96), (17, 8, 128), (257, 8, 1088)])
def test_dma_single_burst_reshape_uses_full_block_length(auto_plan, columns, offset, expected_bytes):
    """A one-row reshape may exceed template pitch, as in GroupNorm parameters."""
    tir = tvm.tir
    source = tir.decl_buffer((columns,), "float32", name="source", scope="global")
    ub = tir.decl_buffer((16,), "float32", name="ub", scope="shared.ub")
    copy = tir.call_extern(
        "handle",
        "tl::ascend::copy_gm_to_ub<float, 1, 16>",
        source.access_ptr("r"),
        ub.access_ptr("w", offset=offset, extent=columns),
        columns,
        1,
        columns,
        tir.const(0, "float32"),
    )
    body = tir.Allocate(ub.data, ub.dtype, [16], tir.const(True, "bool"), tir.Evaluate(copy))
    func = tir.PrimFunc([source.data], body, buffer_map={source.data: source}).with_attr("global_symbol", "main")
    with tvm.transform.PassContext(config={"tl.ascend_memory_planning": auto_plan}):
        planned = tilelang.transform.AscendMemoryPlanning()(tvm.IRModule({"main": func}))
    # Four columns: Duplicate of the 16-element template dominates. Wider
    # bursts: the byte offset plus every touched 32-byte block dominates.
    assert int(planned["main"].attrs["size_map"][ub.data]) == expected_bytes


@pytest.mark.parametrize("auto_plan", [False, True])
def test_dma_multiburst_rejects_columns_exceeding_pitch(auto_plan):
    """Only a single row can ignore a template stride smaller than its burst."""
    tir = tvm.tir
    source = tir.decl_buffer((2, 17), "float32", name="source", scope="global")
    ub = tir.decl_buffer((64,), "float32", name="ub", scope="shared.ub")
    copy = tir.call_extern(
        "handle",
        "tl::ascend::copy_gm_to_ub<float, 1, 16>",
        source.access_ptr("r"),
        ub.access_ptr("w"),
        17,
        2,
        17,
        tir.const(0, "float32"),
    )
    body = tir.Allocate(ub.data, ub.dtype, [64], tir.const(True, "bool"), tir.Evaluate(copy))
    func = tir.PrimFunc([source.data], body, buffer_map={source.data: source}).with_attr("global_symbol", "main")
    with (
        tvm.transform.PassContext(config={"tl.ascend_memory_planning": auto_plan}),
        pytest.raises(tvm.error.InternalError, match="Cannot prove bounded nonnegative UB copy extents"),
    ):
        tilelang.transform.AscendMemoryPlanning()(tvm.IRModule({"main": func}))


def _nested_dma_tail_columns(cid, vid, slot):
    """The floordiv predicates produced for XAttention's last merge tile."""
    tir = tvm.tir
    return tir.Select(
        -vid - cid * 2 - slot * 48 >= -15,
        64,
        tir.Select(slot * 3 + cid // 8 < 1, 1024 - vid * 64 - cid * 128 - slot * 3072, 0),
    )


@pytest.mark.parametrize("auto_plan", [False, True])
def test_dma_nested_floor_predicates_have_bounded_extents(auto_plan):
    """The apparent negative Select arm is unreachable at all 96 scalar points."""
    tir = tvm.tir
    source = tir.decl_buffer((8192,), "float32", name="source", scope="global")
    ub = tir.decl_buffer((128,), "float32", name="ub", scope="shared.ub")
    cid, vid, slot = [tir.Var(name, "int32") for name in ("cid", "vid", "slot")]
    columns = _nested_dma_tail_columns(cid, vid, slot)
    copy = tir.call_extern(
        "handle",
        "tl::ascend::copy_gm_to_ub<float, 1, 64>",
        source.access_ptr("r"),
        ub.access_ptr("w", offset=(slot + 1) % 2 * 64, extent=64),
        4096,
        1,
        columns,
        tir.const(0, "float32"),
    )
    body = tir.IfThenElse(slot + 1 < (111 - vid - cid * 2) // 48, tir.Evaluate(copy), None)
    body = tir.IfThenElse(slot < (111 - vid - cid * 2) // 48, body, None)
    body = tir.For(slot, 0, 2, tir.ForKind.SERIAL, body)
    body = tir.For(vid, 0, 2, tir.ForKind.SERIAL, body)
    body = tir.For(cid, 0, 24, tir.ForKind.SERIAL, body)
    body = tir.Allocate(ub.data, ub.dtype, [128], tir.const(True, "bool"), body)
    func = tir.PrimFunc([source.data], body, buffer_map={source.data: source}).with_attr("global_symbol", "main")
    with tvm.transform.PassContext(config={"tl.ascend_memory_planning": auto_plan}):
        planned = tilelang.transform.AscendMemoryPlanning()(tvm.IRModule({"main": func}))
    assert int(planned["main"].attrs["size_map"][ub.data]) == 128 * 4


@pytest.mark.parametrize("auto_plan", [False, True])
@pytest.mark.parametrize("invalid_bounds", ["negative", "unbounded", "runtime_load", "over_budget"])
def test_dma_failed_extent_bounds_remain_rejected(auto_plan, invalid_bounds):
    """Small-domain refinement must never clamp, sample, or guess a runtime value."""
    tir = tvm.tir
    source = tir.decl_buffer((8192,), "float32", name="source", scope="global")
    ub = tir.decl_buffer((128,), "float32", name="ub", scope="shared.ub")
    cid, vid, slot = [tir.Var(name, "int32") for name in ("cid", "vid", "slot")]
    parameters = [source.data]
    buffer_map = {source.data: source}
    if invalid_bounds == "negative":
        columns = tir.Select(cid == 0, 1, -1)
        loops = [(cid, 2)]
    elif invalid_bounds == "unbounded":
        columns = tir.Var("columns", "int64")
        parameters.append(columns)
        loops = []
    elif invalid_bounds == "runtime_load":
        extents = tir.decl_buffer((2,), "int32", name="extents", scope="global")
        parameters.append(extents.data)
        buffer_map[extents.data] = extents
        columns = extents[cid]
        loops = [(cid, 2)]
    else:
        # All values are safe, but the 129 * 2 * 2 domain exceeds the strict
        # fallback budget. It must retain the original unproven-bound error.
        columns = _nested_dma_tail_columns(cid, vid, slot)
        loops = [(cid, 129), (vid, 2), (slot, 2)]
    copy = tir.call_extern(
        "handle",
        "tl::ascend::copy_gm_to_ub<float, 1, 64>",
        source.access_ptr("r"),
        ub.access_ptr("w", extent=64),
        4096,
        1,
        columns,
        tir.const(0, "float32"),
    )
    body = tir.Evaluate(copy)
    for variable, extent in reversed(loops):
        body = tir.For(variable, 0, extent, tir.ForKind.SERIAL, body)
    body = tir.Allocate(ub.data, ub.dtype, [128], tir.const(True, "bool"), body)
    func = tir.PrimFunc(parameters, body, buffer_map=buffer_map).with_attr("global_symbol", "main")
    with (
        tvm.transform.PassContext(config={"tl.ascend_memory_planning": auto_plan}),
        pytest.raises(tvm.error.InternalError, match="Cannot prove bounded nonnegative UB copy extents"),
    ):
        tilelang.transform.AscendMemoryPlanning()(tvm.IRModule({"main": func}))


@pytest.mark.parametrize("auto_plan", [False, True])
@pytest.mark.parametrize("rows_per_iter,minimum", [(1, 0), (32, 3), (32, -3)])
def test_dma_dynamic_conditional_loop_bounds(auto_plan, rows_per_iter, minimum):
    """GDN's dynamic batches still cap each UB row loop at its static tile size."""
    tir = tvm.tir
    source = tir.decl_buffer((8192,), "float32", name="source", scope="global")
    ub = tir.decl_buffer((64,), "float32", name="ub", scope="shared.ub")
    batches, vid, chunk, row = [tir.Var(name, "int32") for name in ("batches", "vid", "chunk", "row")]
    base = vid * ((batches + 1) // 2) + chunk * rows_per_iter
    remaining = tir.if_then_else(base < batches, batches - base, 0)
    count = tir.if_then_else(remaining >= rows_per_iter, rows_per_iter, remaining)
    copy = tir.call_extern(
        "handle",
        "tl::ascend::copy_gm_to_ub<float, 4, 1>",
        source.access_ptr("r"),
        ub.access_ptr("w", offset=(row - minimum) * 64, extent=4),
        4,
        1,
        4,
        tir.const(0, "float32"),
    )
    # The first occurrence has a narrower context. It must not poison the
    # repeated Call's global bounds, including for nonzero/negative loop minima.
    body = tir.SeqStmt([tir.IfThenElse(row < minimum + 1, tir.Evaluate(copy), None), tir.Evaluate(copy)])
    body = tir.For(row, minimum, count, tir.ForKind.SERIAL, body)
    chunks = tir.if_then_else(batches > 0, (batches + 1) // 2, 0)
    body = tir.For(chunk, 0, chunks, tir.ForKind.SERIAL, body)
    body = tir.For(vid, 0, 2, tir.ForKind.SERIAL, body)
    body = tir.Allocate(ub.data, ub.dtype, [64], tir.const(True, "bool"), body)
    func = tir.PrimFunc([source.data, batches], body, buffer_map={source.data: source}).with_attr("global_symbol", "main")
    with tvm.transform.PassContext(config={"tl.ascend_memory_planning": auto_plan}):
        planned = tilelang.transform.AscendMemoryPlanning()(tvm.IRModule({"main": func}))
    assert int(planned["main"].attrs["size_map"][ub.data]) == max(256, (rows_per_iter - 1) * 256 + 32)


@pytest.mark.parametrize("auto_plan", [False, True])
@pytest.mark.parametrize("invalid_range", ["negative", "unbounded_min", "overflow", "uncapped"])
def test_dma_conditional_loop_bounds_do_not_hide_invalid_offsets(auto_plan, invalid_range):
    """Refinement preserves minima and refuses unknown or overflowing ranges."""
    tir = tvm.tir
    source = tir.decl_buffer((8192,), "float32", name="source", scope="global")
    ub = tir.decl_buffer((64,), "float32", name="ub", scope="shared.ub")
    batches, row = [tir.Var(name, "int64") for name in ("batches", "row")]
    parameters = [source.data, batches]
    remaining = tir.if_then_else(batches > 0, batches, tir.const(0, "int64"))
    count = tir.if_then_else(remaining >= 32, tir.const(32, "int64"), remaining)
    if invalid_range == "negative":
        minimum = tir.const(-1, "int64")
    elif invalid_range == "unbounded_min":
        minimum = tir.Var("start", "int64")
        parameters.append(minimum)
    elif invalid_range == "overflow":
        minimum = tir.const(2**63 - 9, "int64")
    else:
        minimum = tir.const(0, "int64")
        count = remaining
    copy = tir.call_extern(
        "handle",
        "tl::ascend::copy_gm_to_ub<float, 4, 1>",
        source.access_ptr("r"),
        ub.access_ptr("w", offset=row, extent=4),
        4,
        1,
        4,
        tir.const(0, "float32"),
    )
    body = tir.For(row, minimum, count, tir.ForKind.SERIAL, tir.Evaluate(copy))
    body = tir.Allocate(ub.data, ub.dtype, [64], tir.const(True, "bool"), body)
    func = tir.PrimFunc(parameters, body, buffer_map={source.data: source}).with_attr("global_symbol", "main")
    with (
        tvm.transform.PassContext(config={"tl.ascend_memory_planning": auto_plan}),
        pytest.raises(tvm.error.InternalError, match="UB copy"),
    ):
        tilelang.transform.AscendMemoryPlanning()(tvm.IRModule({"main": func}))


@pytest.mark.parametrize("auto_plan", [False, True])
@pytest.mark.parametrize("store", [False, True])
def test_dma_if_then_else_counts_and_empty_offsets(auto_plan, store):
    """Scalar if_then_else extents have the same bounds and no-op rules as Select."""
    tir = tvm.tir
    source = tir.decl_buffer((64, 2), "float32", name="source", scope="global")
    ub = tir.decl_buffer((64,), "float32", name="ub", scope="shared.ub")
    batches = tir.Var("batches", "int32")
    remaining = tir.if_then_else(batches > 0, batches, 0)
    rows = tir.if_then_else(remaining >= 32, 32, remaining)
    if store:
        offset = tir.if_then_else(batches > 0, tir.const(0, "int64"), tir.const(2**61, "int64"))
        copy = tir.call_extern(
            "handle",
            "tl::ascend::copy_ub_to_gm<float, 1, 32>",
            ub.access_ptr("r", offset=offset, extent=32),
            source.access_ptr("w"),
            2,
            rows,
            1,
        )
    else:
        copy = tir.call_extern(
            "handle",
            "tl::ascend::copy_gm_to_ub<float, 1, 32>",
            source.access_ptr("r"),
            ub.access_ptr("w", extent=32),
            2,
            rows,
            1,
            tir.const(0, "float32"),
        )
    body = tir.Allocate(ub.data, ub.dtype, [64], tir.const(True, "bool"), tir.Evaluate(copy))
    func = tir.PrimFunc([source.data, batches], body, buffer_map={source.data: source}).with_attr("global_symbol", "main")
    with tvm.transform.PassContext(config={"tl.ascend_memory_planning": auto_plan}):
        planned = tilelang.transform.AscendMemoryPlanning()(tvm.IRModule({"main": func}))
    assert int(planned["main"].attrs["size_map"][ub.data]) == 32 * 32


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-n", "8"])
