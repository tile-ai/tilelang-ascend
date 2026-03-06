# Copyright (c) Huawei Technologies Co., Ltd. 2025.
"""
Cube core sliced-copy tests (nd2nz → dot → fixpipe) with static shapes.

For each rank (2D / 3D / 4D):
  1. Construct A with only one row-slice non-zero.
  2. Construct an N×N identity matrix B.
  3. The Cube pipeline loads the slice, multiplies by identity, and stores back.
  4. Verify only the target slice is written; all others remain zero.

Edge cases covered via pytest parametrize:
  - First / last row index
  - M = 1 (single row)
  - Leading dimensions = 1 (rank-reduction boundary)
  - Consecutive leading 1s (e.g. B=1, H=1, M=1 for 4D)
"""
import pytest
import torch
import torch_npu  # noqa: F401
import tilelang
import tilelang.language as T

from testcommon import assert_close, build_dtype_param_combos, gen_tensor

pytestmark = [pytest.mark.copy, pytest.mark.op("copy")]

IN_DTYPES = ["float16", "float32"]
OUT_DTYPES = ["float16", "float32"]
DTYPE_COMBOS = build_dtype_param_combos(IN_DTYPES, OUT_DTYPES)

# ---------------------------------------------------------------------------
# Kernel builders
# ---------------------------------------------------------------------------

@tilelang.jit(target="npuir")
def cube_sliced_copy_2d(M, N, idx, in_dtype="float16", out_dtype="float16", accum_dtype="float32"):
    @T.prim_func
    def cube_sliced_copy_2d(
        A_in: T.Tensor((M, N), in_dtype),
        B_in: T.Tensor((N, N), in_dtype),
        Out: T.Tensor((M, N), out_dtype),
    ):
        with T.Kernel(1, is_npu=True) as (cid, subid):
            l1_a = T.alloc_L1([1, N], in_dtype)
            l1_b = T.alloc_L1([N, N], in_dtype)
            l0_c = T.alloc_L0C([1, N], accum_dtype)

            with T.Scope("Cube"):
                tail_m = 1
                tail_n = N
                tail_k = N

                T.copy(A_in[idx:idx + tail_m, 0:tail_k], l1_a[0:tail_m, 0:tail_k])
                T.copy(B_in[0:tail_n, 0:tail_k], l1_b[0:tail_n, 0:tail_k])

                T.npuir_dot(
                    l1_a, l1_b, l0_c,
                    initC=True, b_transpose=True,
                    size=[tail_m, tail_k, tail_n],
                )

                with T.rs("PIPE_FIX"):
                    T.copy(l0_c[0:tail_m, 0:tail_n], Out[idx:idx + tail_m, 0:tail_n])

    return cube_sliced_copy_2d


@tilelang.jit(target="npuir")
def cube_sliced_copy_3d(B, M, N, b_idx, m_idx, in_dtype="float16", out_dtype="float16", accum_dtype="float32"):
    @T.prim_func
    def cube_sliced_copy_3d(
        A_in: T.Tensor((B, M, N), in_dtype),
        B_in: T.Tensor((N, N), in_dtype),
        Out: T.Tensor((B, M, N), out_dtype),
    ):
        with T.Kernel(1, is_npu=True) as (cid, subid):
            l1_a = T.alloc_L1([1, N], in_dtype)
            l1_b = T.alloc_L1([N, N], in_dtype)
            l0_c = T.alloc_L0C([1, N], accum_dtype)

            with T.Scope("Cube"):
                tail_m = 1
                tail_n = N
                tail_k = N

                T.copy(A_in[b_idx, m_idx:m_idx + tail_m, 0:tail_k], l1_a[0:tail_m, 0:tail_k])
                T.copy(B_in[0:tail_n, 0:tail_k], l1_b[0:tail_n, 0:tail_k])

                T.npuir_dot(
                    l1_a, l1_b, l0_c,
                    initC=True, b_transpose=True,
                    size=[tail_m, tail_k, tail_n],
                )

                with T.rs("PIPE_FIX"):
                    T.copy(l0_c[0:tail_m, 0:tail_n], Out[b_idx, m_idx:m_idx + tail_m, 0:tail_n])

    return cube_sliced_copy_3d


@tilelang.jit(target="npuir")
def cube_sliced_copy_4d(
    B, H, M, N, b_idx, h_idx, m_idx, in_dtype="float16", out_dtype="float16", accum_dtype="float32"
):
    @T.prim_func
    def cube_sliced_copy_4d(
        A_in: T.Tensor((B, H, M, N), in_dtype),
        B_in: T.Tensor((N, N), in_dtype),
        Out: T.Tensor((B, H, M, N), out_dtype),
    ):
        with T.Kernel(1, is_npu=True) as (cid, subid):
            l1_a = T.alloc_L1([1, N], in_dtype)
            l1_b = T.alloc_L1([N, N], in_dtype)
            l0_c = T.alloc_L0C([1, N], accum_dtype)

            with T.Scope("Cube"):
                tail_m = 1
                tail_n = N
                tail_k = N

                T.copy(A_in[b_idx, h_idx, m_idx:m_idx + tail_m, 0:tail_k], l1_a[0:tail_m, 0:tail_k])
                T.copy(B_in[0:tail_n, 0:tail_k], l1_b[0:tail_n, 0:tail_k])

                T.npuir_dot(
                    l1_a, l1_b, l0_c,
                    initC=True, b_transpose=True,
                    size=[tail_m, tail_k, tail_n],
                )

                with T.rs("PIPE_FIX"):
                    T.copy(l0_c[0:tail_m, 0:tail_n],
                           Out[b_idx, h_idx, m_idx:m_idx + tail_m, 0:tail_n])

    return cube_sliced_copy_4d


# ---------------------------------------------------------------------------
# 2D parametrized tests  —  A: [M, N]
# ---------------------------------------------------------------------------

SLICED_2D_CASES = [
    # (M,  N,  idx)
    (16, 32, 5),         # base case
    (16, 32, 0),         # first row
    (16, 32, 15),        # last row
    (1,  32, 0),         # M=1: single-row tensor
    (32, 16, 16),        # M > N, higher index
    (16,  8, 5),         # small N=8
    (1,   8, 0),         # small N=8, M=1
]

@pytest.mark.parametrize("in_dtype, out_dtype", DTYPE_COMBOS)
@pytest.mark.parametrize("M, N, idx", SLICED_2D_CASES)
def test_cube_sliced_copy_2d(M, N, idx, in_dtype, out_dtype):
    kernel = cube_sliced_copy_2d(M, N, idx, in_dtype, out_dtype)

    A = gen_tensor((M, N), in_dtype, kind="zeros")
    row = torch.arange(1, N + 1, dtype=A.dtype).npu()
    A[idx] = row

    B = torch.eye(N, dtype=A.dtype).npu()
    Out = gen_tensor((M, N), out_dtype, kind="zeros")

    kernel(A, B, Out)

    expected = gen_tensor((M, N), out_dtype, kind="zeros")
    expected[idx] = row.to(dtype=Out.dtype)
    assert_close(Out.cpu(), expected.cpu(), dtype=out_dtype, rtol=1e-5, atol=1e-5)


# ---------------------------------------------------------------------------
# 3D parametrized tests  —  A: [B, M, N]
# ---------------------------------------------------------------------------

SLICED_3D_CASES = [
    # (B,  M,  N,  b_idx, m_idx)
    (4,  16, 32, 1, 7),       # base case
    (1,  16, 32, 0, 7),       # B=1  (leading-1 rank reduction)
    (4,  1,  32, 2, 0),       # M=1
    (1,  1,  32, 0, 0),       # B=1, M=1  (consecutive leading 1s)
    (4,  16, 32, 3, 15),      # boundary indices (last B, last M)
    (4,  16, 32, 0, 0),       # all-zero indices
    (4,  16,  8, 1, 7),       # small N=8
    (1,  1,   8, 0, 0),       # small N=8, B=1, M=1
]

@pytest.mark.parametrize("in_dtype, out_dtype", DTYPE_COMBOS)
@pytest.mark.parametrize("B_dim, M, N, b_idx, m_idx", SLICED_3D_CASES)
def test_cube_sliced_copy_3d(B_dim, M, N, b_idx, m_idx, in_dtype, out_dtype):
    kernel = cube_sliced_copy_3d(B_dim, M, N, b_idx, m_idx, in_dtype, out_dtype)

    A = gen_tensor((B_dim, M, N), in_dtype, kind="zeros")
    row = torch.arange(1, N + 1, dtype=A.dtype).npu()
    A[b_idx, m_idx] = row

    B = torch.eye(N, dtype=A.dtype).npu()
    Out = gen_tensor((B_dim, M, N), out_dtype, kind="zeros")

    kernel(A, B, Out)

    expected = gen_tensor((B_dim, M, N), out_dtype, kind="zeros")
    expected[b_idx, m_idx] = row.to(dtype=Out.dtype)
    assert_close(Out.cpu(), expected.cpu(), dtype=out_dtype, rtol=1e-5, atol=1e-5)


# ---------------------------------------------------------------------------
# 4D parametrized tests  —  A: [B, H, M, N]
# ---------------------------------------------------------------------------

SLICED_4D_CASES = [
    # (B,  H,  M,  N,  b_idx, h_idx, m_idx)
    (2,  4,  16, 32, 1, 2, 5),       # base case
    (1,  4,  16, 32, 0, 2, 5),       # B=1
    (2,  1,  16, 32, 1, 0, 5),       # H=1
    (1,  1,  16, 32, 0, 0, 5),       # B=1, H=1  (two consecutive leading 1s)
    (1,  1,  1,  32, 0, 0, 0),       # B=1, H=1, M=1  (three consecutive 1s)
    (2,  4,  16, 32, 0, 0, 0),       # all-zero indices
    (2,  4,  16, 32, 1, 3, 15),      # boundary indices (last valid)
    (2,  4,  16,  8, 1, 2, 5),       # small N=8
    (1,  1,  1,   8, 0, 0, 0),       # small N=8, all leading 1s
]

@pytest.mark.parametrize("in_dtype, out_dtype", DTYPE_COMBOS)
@pytest.mark.parametrize("B_dim, H, M, N, b_idx, h_idx, m_idx", SLICED_4D_CASES)
def test_cube_sliced_copy_4d(B_dim, H, M, N, b_idx, h_idx, m_idx, in_dtype, out_dtype):
    kernel = cube_sliced_copy_4d(B_dim, H, M, N, b_idx, h_idx, m_idx, in_dtype, out_dtype)

    A = gen_tensor((B_dim, H, M, N), in_dtype, kind="zeros")
    row = torch.arange(1, N + 1, dtype=A.dtype).npu()
    A[b_idx, h_idx, m_idx] = row

    B = torch.eye(N, dtype=A.dtype).npu()
    Out = gen_tensor((B_dim, H, M, N), out_dtype, kind="zeros")

    kernel(A, B, Out)

    expected = gen_tensor((B_dim, H, M, N), out_dtype, kind="zeros")
    expected[b_idx, h_idx, m_idx] = row.to(dtype=Out.dtype)
    assert_close(Out.cpu(), expected.cpu(), dtype=out_dtype, rtol=1e-5, atol=1e-5)
