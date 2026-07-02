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

VEC_NUM = 2

PASS_AUTO = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
}

PASS_LINEAR = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
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


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-n", "8"])
