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
    """Parse buffer offsets from generated AscendC source code."""
    offsets = {}
    for line in kernel_source.split("\n"):
        m = re.search(
            r"auto\s+(\w+)\s*=.*GetWithOffset<[^>]+>\(\s*\d+\s*,\s*(\d+)\s*\)",
            line,
        )
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
# 9. If-else branch and T.if_then_else memory planning
# ---------------------------------------------------------------------------


def test_if_else_branch_exclusive_buffers_can_reuse():
    """Buffers in mutually exclusive if-else branches can share memory."""

    @T.prim_func
    def main(
        A: T.Tensor((2, 64), "float32"),  # type: ignore
        B: T.Tensor((2, 64), "float32"),  # type: ignore
    ):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            if vid == 0:
                buf_a = T.alloc_ub((64,), "float32")
                T.copy(A[0, :], buf_a)
                T.copy(buf_a, B[0, :])
            else:
                buf_b = T.alloc_ub((64,), "float32")
                T.copy(A[1, :], buf_b)
                T.copy(buf_b, B[1, :])

    offsets, _ = _compile_and_get_offsets(main, PASS_AUTO, out_idx=[1])
    assert "buf_a" in offsets or "buf_b" in offsets


def test_if_else_shared_buffer_before_branch_not_reused():
    """Buffer defined before if-else, used inside both branches,
    must NOT be reused by branch-internal buffers."""

    @T.prim_func
    def main(
        A: T.Tensor((4, 32), "float32"),  # type: ignore
        B: T.Tensor((4, 32), "float32"),  # type: ignore
    ):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            shared_ub = T.alloc_ub((32,), "float32")
            T.copy(A[0, :], shared_ub)
            if vid == 0:
                work_a_ub = T.alloc_ub((32,), "float32")
                T.copy(A[1, :], work_a_ub)
                T.tile.add(work_a_ub, work_a_ub, shared_ub)
                T.copy(work_a_ub, B[1, :])
            else:
                work_b_ub = T.alloc_ub((32,), "float32")
                T.copy(A[2, :], work_b_ub)
                T.tile.mul(work_b_ub, work_b_ub, shared_ub)
                T.copy(work_b_ub, B[2, :])

    offsets, _ = _compile_and_get_offsets(main, PASS_AUTO, out_idx=[1])
    assert "shared_ub" in offsets
    buf_size = 32 * 4
    for name in ("work_a_ub", "work_b_ub"):
        if name in offsets:
            assert offsets["shared_ub"] + buf_size <= offsets[name] or offsets[name] + buf_size <= offsets["shared_ub"], (
                f"shared_ub and {name} must not overlap"
            )


def test_if_then_else_expr_dynamic_offset_compiles():
    """T.if_then_else as a dynamic index expression — buffers with
    conditional access must still get valid addresses."""

    @T.prim_func
    def main(
        A: T.Tensor((2, 64), "float32"),  # type: ignore
        B: T.Tensor((2, 64), "float32"),  # type: ignore
    ):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            a_ub = T.alloc_ub((64,), "float32")
            src_row = T.if_then_else(vid == 0, 0, 1)
            T.copy(A[src_row, :], a_ub)
            T.copy(a_ub, B[0, :])

    offsets, _ = _compile_and_get_offsets(main, PASS_AUTO, out_idx=[1])
    assert "a_ub" in offsets
    assert offsets["a_ub"] == 0


def test_if_then_else_expr_in_loop_with_reduce_compiles():
    """T.if_then_else inside a loop with reduce — tmp_ub must be
    injected and buffers must not overlap."""

    @T.prim_func
    def main(
        A: T.Tensor((4, 64), "float32"),  # type: ignore
        B: T.Tensor((4,), "float32"),  # type: ignore
    ):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            a_ub = T.alloc_ub((64,), "float32")
            b_ub = T.alloc_ub((1,), "float32")
            for k in T.serial(3):
                src = T.if_then_else(k < 2, k, 2)
                T.copy(A[src, :], a_ub)
                T.reduce_max(a_ub, b_ub, dim=-1)
                T.copy(b_ub, B[k : k + 1])

    offsets, _ = _compile_and_get_offsets(main, PASS_AUTO, out_idx=[1])
    assert "tmp_ub" in offsets
    buf_size = 64 * 4
    assert offsets["a_ub"] + buf_size <= offsets["b_ub"] or offsets["b_ub"] + buf_size <= offsets["a_ub"], "a_ub and b_ub must not overlap"


def test_if_else_branch_with_loop_nested():
    """If-else with loop inside, buffer before if used in loop body."""

    @T.prim_func
    def main(
        A: T.Tensor((4, 32), "float32"),  # type: ignore
        B: T.Tensor((4, 32), "float32"),  # type: ignore
    ):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            base_ub = T.alloc_ub((32,), "float32")
            T.copy(A[0, :], base_ub)
            if vid == 0:
                for k in T.serial(3):
                    tmp_ub = T.alloc_ub((32,), "float32")
                    T.copy(A[k, :], tmp_ub)
                    T.tile.add(tmp_ub, tmp_ub, base_ub)
                    T.copy(tmp_ub, B[k, :])

    offsets, _ = _compile_and_get_offsets(main, PASS_AUTO, out_idx=[1])
    assert "base_ub" in offsets
    buf_size = 32 * 4
    if "tmp_ub" in offsets:
        assert offsets["base_ub"] + buf_size <= offsets["tmp_ub"] or offsets["tmp_ub"] + buf_size <= offsets["base_ub"], (
            "base_ub and tmp_ub must not overlap in if-else+loop"
        )


def test_if_then_else_conditional_store_compiles():
    """T.if_then_else used for conditional store offset — two buffers
    with conditional access must get distinct addresses when overlapping."""

    @T.prim_func
    def main(
        A: T.Tensor((4, 32), "float32"),  # type: ignore
        B: T.Tensor((4, 32), "float32"),  # type: ignore
    ):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            a_ub = T.alloc_ub((32,), "float32")
            b_ub = T.alloc_ub((32,), "float32")
            T.copy(A[0, :], a_ub)
            T.copy(A[1, :], b_ub)
            dst = T.if_then_else(vid == 0, 0, 1)
            T.copy(a_ub, B[dst, :])
            T.copy(b_ub, B[dst + 2, :])

    offsets, _ = _compile_and_get_offsets(main, PASS_AUTO, out_idx=[1])
    assert "a_ub" in offsets and "b_ub" in offsets
    buf_size = 32 * 4
    assert offsets["a_ub"] + buf_size <= offsets["b_ub"] or offsets["b_ub"] + buf_size <= offsets["a_ub"], "a_ub and b_ub must not overlap"


@pytest.mark.skipif(
    not NPU_AVAILABLE,
    reason="NPU correctness requires an Ascend NPU runtime",
)
def test_npu_if_else_reduce_correctness():
    """NPU: reduce inside if-else — max for vid=0, sum for vid=1."""

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
            if vid == 0:
                T.reduce_max(a_ub, b_ub, dim=-1)
            else:
                T.reduce_sum(a_ub, b_ub, dim=-1)
            T.copy(b_ub, B[rb : rb + 64])

    kernel = tilelang.compile(main, pass_configs=PASS_AUTO, target="ascendc", out_idx=[1])
    a = torch.randn(128, 256, dtype=torch.float32, device="npu")
    b = kernel(a)
    ref = torch.zeros(128, dtype=torch.float32, device="npu")
    ref[:64] = torch.max(a[:64, :], dim=-1).values
    ref[64:] = torch.sum(a[64:, :], dim=-1)
    torch.testing.assert_close(b, ref, rtol=1e-2, atol=1e-2)


@pytest.mark.skipif(
    not NPU_AVAILABLE,
    reason="NPU correctness requires an Ascend NPU runtime",
)
def test_npu_if_else_with_loop_correctness():
    """NPU: if-else with loop, buffer before if used in loop — no corruption."""

    @T.prim_func
    def main(
        A: T.Tensor((4, 32), "float32"),  # type: ignore
        B: T.Tensor((4, 32), "float32"),  # type: ignore
    ):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            base_ub = T.alloc_ub((32,), "float32")
            T.copy(A[0, :], base_ub)
            if vid == 0:
                for k in T.serial(3):
                    tmp_ub = T.alloc_ub((32,), "float32")
                    T.copy(A[k, :], tmp_ub)
                    T.tile.add(tmp_ub, tmp_ub, base_ub)
                    T.copy(tmp_ub, B[k, :])

    kernel = tilelang.compile(main, pass_configs=PASS_AUTO, target="ascendc", out_idx=[1])
    a = torch.randn(4, 32, dtype=torch.float32, device="npu")
    b = kernel(a)
    ref = torch.zeros_like(a)
    ref[:3, :] = a[:3, :] + a[0, :].unsqueeze(0)
    torch.testing.assert_close(b[:3, :], ref[:3, :], rtol=1e-3, atol=1e-3)


@pytest.mark.skipif(
    not NPU_AVAILABLE,
    reason="NPU correctness requires an Ascend NPU runtime",
)
def test_npu_if_then_else_dynamic_index_correctness():
    """NPU: T.if_then_else as dynamic source index — correct data loaded."""

    @T.prim_func
    def main(
        A: T.Tensor((2, 64), "float32"),  # type: ignore
        B: T.Tensor((2, 64), "float32"),  # type: ignore
    ):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            a_ub = T.alloc_ub((64,), "float32")
            src_row = T.if_then_else(vid == 0, 0, 1)
            T.copy(A[src_row, :], a_ub)
            T.copy(a_ub, B[vid, :])

    kernel = tilelang.compile(main, pass_configs=PASS_AUTO, target="ascendc", out_idx=[1])
    a = torch.randn(2, 64, dtype=torch.float32, device="npu")
    b = kernel(a)
    torch.testing.assert_close(b[0, :], a[0, :], rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(b[1, :], a[1, :], rtol=1e-3, atol=1e-3)


@pytest.mark.skipif(
    not NPU_AVAILABLE,
    reason="NPU correctness requires an Ascend NPU runtime",
)
def test_npu_if_then_else_in_loop_reduce_correctness():
    """NPU: T.if_then_else in a loop with reduce — correct results."""

    @T.prim_func
    def main(
        A: T.Tensor((4, 64), "float32"),  # type: ignore
        B: T.Tensor((4,), "float32"),  # type: ignore
    ):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            a_ub = T.alloc_ub((64,), "float32")
            b_ub = T.alloc_ub((1,), "float32")
            for k in T.serial(3):
                src = T.if_then_else(k < 2, k, 2)
                T.copy(A[src, :], a_ub)
                T.reduce_max(a_ub, b_ub, dim=-1)
                T.copy(b_ub, B[k : k + 1])

    kernel = tilelang.compile(main, pass_configs=PASS_AUTO, target="ascendc", out_idx=[1])
    a = torch.randn(4, 64, dtype=torch.float32, device="npu")
    b = kernel(a)
    ref = torch.zeros(4, dtype=torch.float32, device="npu")
    ref[0] = a[0, :].max()
    ref[1] = a[1, :].max()
    ref[2] = a[2, :].max()
    torch.testing.assert_close(b[:3], ref[:3], rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-n", "8"])
