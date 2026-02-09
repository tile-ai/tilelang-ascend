# Copyright (c) Huawei Technologies Co., Ltd. 2025.
import os
import torch
import tilelang
import tilelang.language as T

torch.npu.set_device(0)
tilelang.cache.clear_cache()

"""
This file focuses on testing the Cube core sliced path:
    GM (one row slice) --npuir_load_nd2nz--> L1
    L1 × L1 (via npuir_dot) --> L0C
    L0C --npuir_store_fixpipe--> GM (same row position)

Idea for 2D: construct A with shape [M, N], only row `idx` is non-zero;
construct B with shape [N, N] as an identity matrix.
In Cube scope we use:
    T.npuir_load_nd2nz(A[idx, 0], l1_a, [1, N])
    T.npuir_load_nd2nz(B[0, 0],   l1_b, [N, N])
    T.npuir_dot(l1_a, l1_b, l0_c, ..., b_transpose=True, size=[1, N, N])
    T.npuir_store_fixpipe(l0_c, Out[idx, 0], size=[1, N], enable_nz2nd=True)
Then the `idx`-th row of `Out` should equal the `idx`-th row of `A`, and all
other rows should remain zero.

We extend this pattern to:
- 3D: 3D GM (B, M, N) -> 2D L1 / 2D L0C -> 3D GM
- 4D: 4D GM (B, H, M, N) <-> 2D L1 / 2D L0C
"""


@tilelang.jit(out_idx=[-1], target="npuir")
def cube_sliced_copy_2d(M, N, idx, dtype="float16", accum_dtype="float32"):
    @T.prim_func
    def main(
        A_in: T.Tensor((M, N), dtype),
        B_in: T.Tensor((N, N), dtype),
        Out: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(1, is_npu=True) as (cid, subid):
            # Only one row slice is involved, so L1/L0C shapes are [1, N] or [N, N].
            l1_a = T.alloc_L1([1, N], dtype)
            l1_b = T.alloc_L1([N, N], dtype)
            l0_c = T.alloc_L0C([1, N], accum_dtype)

            with T.Scope("Cube"):
                tail_m = 1
                tail_n = N
                tail_k = N

                # GM -> L1: load the `idx`-th row of A and the full B (identity).
                T.npuir_load_nd2nz(A_in[idx, 0], l1_a, [tail_m, tail_k])
                T.npuir_load_nd2nz(B_in[0, 0], l1_b, [tail_n, tail_k])

                # L1 × L1 -> L0C: run through Cube dot to L0C.
                T.npuir_dot(
                    l1_a,
                    l1_b,
                    l0_c,
                    initC=True,
                    b_transpose=True,
                    size=[tail_m, tail_k, tail_n],
                )

                # L0C -> GM: use fixpipe to store back to the same row in GM.
                with T.rs("PIPE_FIX"):
                    T.npuir_store_fixpipe(
                        l0_c,
                        Out[idx, 0],
                        size=[tail_m, tail_n],
                        enable_nz2nd=True,
                    )

    return main


def test_cube_sliced_copy_2d():
    print("=" * 30 + " Running Cube Sliced Copy 2D Test " + "=" * 30)

    M, N = 16, 32
    idx = 5

    kernel = cube_sliced_copy_2d(M, N, idx)

    # A: only row `idx` is non-zero, others are zeros.
    A = torch.zeros(M, N).npu().half()
    row = torch.arange(1, N + 1, dtype=torch.float16).npu()
    A[idx] = row

    # B: identity matrix, so A[idx] @ B^T == A[idx].
    B = torch.eye(N, dtype=torch.float16).npu()

    Out = torch.zeros(M, N).npu().half()

    # Run kernel: the Cube pipeline should write back only the `idx`-th row.
    kernel(A, B, Out)

    expected = torch.zeros(M, N).npu().half()
    expected[idx] = row

    torch.testing.assert_close(Out, expected, rtol=1e-5, atol=1e-5)

    print("Cube Sliced Copy 2D Test Passed!")


@tilelang.jit(out_idx=[-1], target="npuir")
def cube_sliced_copy_3d(B, M, N, b_idx, m_idx, dtype="float16", accum_dtype="float32"):
    """
    3D GM -> 2D L1 / 2D L0C -> 3D GM on Cube.

    A_in: [B, M, N], only slice (b_idx, m_idx, :) is non-zero.
    B_in: [N, N] identity.
    We load A_in[b_idx, m_idx, :] to [1, N] L1, run dot with identity,
    and store the result back to Out[b_idx, m_idx, :].
    """

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

                # GM -> L1: 3D slice (b_idx, m_idx, :) to 2D L1.
                T.npuir_load_nd2nz(A_in[b_idx, m_idx, 0], l1_a, [tail_m, tail_k])
                T.npuir_load_nd2nz(B_in[0, 0], l1_b, [tail_n, tail_k])

                # L1 × L1 -> L0C.
                T.npuir_dot(
                    l1_a,
                    l1_b,
                    l0_c,
                    initC=True,
                    b_transpose=True,
                    size=[tail_m, tail_k, tail_n],
                )

                # L0C -> GM: back to the same (b_idx, m_idx, :) slice in 3D GM.
                with T.rs("PIPE_FIX"):
                    T.npuir_store_fixpipe(
                        l0_c,
                        Out[b_idx, m_idx, 0],
                        size=[tail_m, tail_n],
                        enable_nz2nd=True,
                    )

    return main


def test_cube_sliced_copy_3d():
    print("\n" + "=" * 30 + " Running Cube Sliced Copy 3D Test " + "=" * 30)

    B_dim, M, N = 4, 16, 32
    b_idx, m_idx = 1, 7

    kernel = cube_sliced_copy_3d(B_dim, M, N, b_idx, m_idx)

    # A: only slice (b_idx, m_idx, :) is non-zero.
    A = torch.zeros(B_dim, M, N).npu().half()
    row = torch.arange(1, N + 1, dtype=torch.float16).npu()
    A[b_idx, m_idx] = row

    # B: identity matrix on the last dimension.
    B = torch.eye(N, dtype=torch.float16).npu()

    Out = torch.zeros(B_dim, M, N).npu().half()

    kernel(A, B, Out)

    expected = torch.zeros(B_dim, M, N).npu().half()
    expected[b_idx, m_idx] = row

    torch.testing.assert_close(Out, expected, rtol=1e-5, atol=1e-5)

    print("Cube Sliced Copy 3D Test Passed!")


@tilelang.jit(out_idx=[-1], target="npuir")
def cube_sliced_copy_4d(B, H, M, N, b_idx, h_idx, m_idx, dtype="float16", accum_dtype="float32"):
    """
    4D GM <-> 2D L1 / 2D L0C on Cube.

    A_in: [B, H, M, N], only slice (b_idx, h_idx, m_idx, :) is non-zero.
    B_in: [N, N] identity.
    The selected 1×N slice is moved to 2D L1, computed in 2D L0C, and
    stored back into the same 4D GM position.
    """

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

                # GM -> L1: 4D slice (b_idx, h_idx, m_idx, :) to 2D L1.
                T.npuir_load_nd2nz(A_in[b_idx, h_idx, m_idx, 0], l1_a, [tail_m, tail_k])
                T.npuir_load_nd2nz(B_in[0, 0], l1_b, [tail_n, tail_k])

                # L1 × L1 -> L0C.
                T.npuir_dot(
                    l1_a,
                    l1_b,
                    l0_c,
                    initC=True,
                    b_transpose=True,
                    size=[tail_m, tail_k, tail_n],
                )

                # L0C -> GM: back to the same (b_idx, h_idx, m_idx, :) slice in 4D GM.
                with T.rs("PIPE_FIX"):
                    T.npuir_store_fixpipe(
                        l0_c,
                        Out[b_idx, h_idx, m_idx, 0],
                        size=[tail_m, tail_n],
                        enable_nz2nd=True,
                    )

    return main


def test_cube_sliced_copy_4d():
    print("\n" + "=" * 30 + " Running Cube Sliced Copy 4D Test " + "=" * 30)

    B_dim, H, M, N = 2, 4, 16, 32
    b_idx, h_idx, m_idx = 1, 2, 5

    kernel = cube_sliced_copy_4d(B_dim, H, M, N, b_idx, h_idx, m_idx)

    # A: only slice (b_idx, h_idx, m_idx, :) is non-zero.
    A = torch.zeros(B_dim, H, M, N).npu().half()
    row = torch.arange(1, N + 1, dtype=torch.float16).npu()
    A[b_idx, h_idx, m_idx] = row

    # B: identity matrix on the last dimension.
    B = torch.eye(N, dtype=torch.float16).npu()

    Out = torch.zeros(B_dim, H, M, N).npu().half()

    kernel(A, B, Out)

    expected = torch.zeros(B_dim, H, M, N).npu().half()
    expected[b_idx, h_idx, m_idx] = row

    torch.testing.assert_close(Out, expected, rtol=1e-5, atol=1e-5)

    print("Cube Sliced Copy 4D Test Passed!")


def main():
    test_cube_sliced_copy_2d()
    test_cube_sliced_copy_3d()
    test_cube_sliced_copy_4d()


if __name__ == "__main__":
    main()