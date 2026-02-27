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
import tilelang
import tilelang.language as T

torch.npu.set_device(0)
tilelang.cache.clear_cache()

# ---------------------------------------------------------------------------
# Kernel builders
# ---------------------------------------------------------------------------

@tilelang.jit(out_idx=[-1], target="npuir")
def cube_sliced_copy_2d(M, N, idx, dtype="float16", accum_dtype="float32"):
    @T.prim_func
    def main(
        A_in: T.Tensor((M, N), dtype),
        B_in: T.Tensor((N, N), dtype),
        Out: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(1, is_npu=True) as (cid, subid):
            l1_a = T.alloc_L1([1, N], dtype)
            l1_b = T.alloc_L1([N, N], dtype)
            l0_c = T.alloc_L0C([1, N], accum_dtype)

            with T.Scope("Cube"):
                tail_m = 1
                tail_n = N
                tail_k = N

                T.npuir_load_nd2nz(A_in[idx, 0], l1_a, size=[tail_m, tail_k])
                T.npuir_load_nd2nz(B_in[0, 0], l1_b, size=[tail_n, tail_k])

                T.npuir_dot(
                    l1_a, l1_b, l0_c,
                    initC=True, b_transpose=True,
                    size=[tail_m, tail_k, tail_n],
                )

                with T.rs("PIPE_FIX"):
                    T.npuir_store_fixpipe(
                        l0_c, Out[idx, 0],
                        size=[tail_m, tail_n],
                        enable_nz2nd=True,
                    )

    return main


@tilelang.jit(out_idx=[-1], target="npuir")
def cube_sliced_copy_3d(B, M, N, b_idx, m_idx, dtype="float16", accum_dtype="float32"):
    @T.prim_func
    def main(
        A_in: T.Tensor((B, M, N), dtype),
        B_in: T.Tensor((N, N), dtype),
        Out: T.Tensor((B, M, N), dtype),
    ):
        with T.Kernel(1, is_npu=True) as (cid, subid):
            l1_a = T.alloc_L1([1, N], dtype)
            l1_b = T.alloc_L1([N, N], dtype)
            l0_c = T.alloc_L0C([1, N], accum_dtype)

            with T.Scope("Cube"):
                tail_m = 1
                tail_n = N
                tail_k = N

                T.npuir_load_nd2nz(A_in[b_idx, m_idx, 0], l1_a, size=[1, tail_m, tail_k])
                T.npuir_load_nd2nz(B_in[0, 0], l1_b, size=[tail_n, tail_k])

                T.npuir_dot(
                    l1_a, l1_b, l0_c,
                    initC=True, b_transpose=True,
                    size=[tail_m, tail_k, tail_n],
                )

                with T.rs("PIPE_FIX"):
                    T.npuir_store_fixpipe(
                        l0_c, Out[b_idx, m_idx, 0],
                        size=[1, tail_m, tail_n],
                        enable_nz2nd=True,
                    )

    return main


@tilelang.jit(out_idx=[-1], target="npuir")
def cube_sliced_copy_4d(B, H, M, N, b_idx, h_idx, m_idx, dtype="float16", accum_dtype="float32"):
    @T.prim_func
    def main(
        A_in: T.Tensor((B, H, M, N), dtype),
        B_in: T.Tensor((N, N), dtype),
        Out: T.Tensor((B, H, M, N), dtype),
    ):
        with T.Kernel(1, is_npu=True) as (cid, subid):
            l1_a = T.alloc_L1([1, N], dtype)
            l1_b = T.alloc_L1([N, N], dtype)
            l0_c = T.alloc_L0C([1, N], accum_dtype)

            with T.Scope("Cube"):
                tail_m = 1
                tail_n = N
                tail_k = N

                T.npuir_load_nd2nz(A_in[b_idx, h_idx, m_idx, 0], l1_a, size=[1, 1, tail_m, tail_k])
                T.npuir_load_nd2nz(B_in[0, 0], l1_b, size=[tail_n, tail_k])

                T.npuir_dot(
                    l1_a, l1_b, l0_c,
                    initC=True, b_transpose=True,
                    size=[tail_m, tail_k, tail_n],
                )

                with T.rs("PIPE_FIX"):
                    T.npuir_store_fixpipe(
                        l0_c, Out[b_idx, h_idx, m_idx, 0],
                        size=[1, 1, tail_m, tail_n],
                        enable_nz2nd=True,
                    )

    return main


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

@pytest.mark.parametrize("M, N, idx", SLICED_2D_CASES)
def test_cube_sliced_copy_2d(M, N, idx):
    kernel = cube_sliced_copy_2d(M, N, idx)

    A = torch.zeros(M, N, dtype=torch.float16).npu()
    row = torch.arange(1, N + 1, dtype=torch.float16).npu()
    A[idx] = row

    B = torch.eye(N, dtype=torch.float16).npu()
    Out = torch.zeros(M, N, dtype=torch.float16).npu()

    kernel(A, B, Out)

    expected = torch.zeros(M, N, dtype=torch.float16).npu()
    expected[idx] = row
    torch.testing.assert_close(Out, expected, rtol=1e-5, atol=1e-5)


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

@pytest.mark.parametrize("B_dim, M, N, b_idx, m_idx", SLICED_3D_CASES)
def test_cube_sliced_copy_3d(B_dim, M, N, b_idx, m_idx):
    kernel = cube_sliced_copy_3d(B_dim, M, N, b_idx, m_idx)

    A = torch.zeros(B_dim, M, N, dtype=torch.float16).npu()
    row = torch.arange(1, N + 1, dtype=torch.float16).npu()
    A[b_idx, m_idx] = row

    B = torch.eye(N, dtype=torch.float16).npu()
    Out = torch.zeros(B_dim, M, N, dtype=torch.float16).npu()

    kernel(A, B, Out)

    expected = torch.zeros(B_dim, M, N, dtype=torch.float16).npu()
    expected[b_idx, m_idx] = row
    torch.testing.assert_close(Out, expected, rtol=1e-5, atol=1e-5)


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

@pytest.mark.parametrize("B_dim, H, M, N, b_idx, h_idx, m_idx", SLICED_4D_CASES)
def test_cube_sliced_copy_4d(B_dim, H, M, N, b_idx, h_idx, m_idx):
    kernel = cube_sliced_copy_4d(B_dim, H, M, N, b_idx, h_idx, m_idx)

    A = torch.zeros(B_dim, H, M, N, dtype=torch.float16).npu()
    row = torch.arange(1, N + 1, dtype=torch.float16).npu()
    A[b_idx, h_idx, m_idx] = row

    B = torch.eye(N, dtype=torch.float16).npu()
    Out = torch.zeros(B_dim, H, M, N, dtype=torch.float16).npu()

    kernel(A, B, Out)

    expected = torch.zeros(B_dim, H, M, N, dtype=torch.float16).npu()
    expected[b_idx, h_idx, m_idx] = row
    torch.testing.assert_close(Out, expected, rtol=1e-5, atol=1e-5)


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
