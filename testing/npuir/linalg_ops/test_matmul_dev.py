# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
import os
import torch
import argparse

import tilelang
import tilelang.language as T

parser = argparse.ArgumentParser(description="NPU Kernel Compilation")
parser.add_argument("--M", type=int, default=16, help="")
parser.add_argument("--N", type=int, default=16, help="")
parser.add_argument("--K", type=int, default=16, help="")
parser.add_argument("--block_M", type=int, default=16, help="")
parser.add_argument("--block_N", type=int, default=16, help="")


def matmul(M, N, K, block_M, block_N):
    m_num = M // block_M
    n_num = N // block_N
    accum_dtype = "float32"

    @T.prim_func
    def main(
        A: T.Tensor((M, K), accum_dtype),
        B: T.Tensor((N, K), accum_dtype),
        C: T.Tensor((M, N), accum_dtype),
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, _):
            bx = block_M * (cid // n_num)
            by = block_N * (cid % n_num)

            A_shared = T.alloc_shared((block_M, K), accum_dtype)
            B_shared = T.alloc_shared((block_N, K), accum_dtype)
            C_shared = T.alloc_shared((block_M, block_N), accum_dtype)

            T.copy(A[bx:bx+block_M, 0:K], A_shared)
            T.copy(B[by:by+block_N, 0:K], B_shared)
            T.copy(C[bx:bx+block_M, by:by+block_N], C_shared)

            T.gemm(A_shared, B_shared, C_shared, initC=False, b_transpose=True)

            T.copy(C_shared, C[bx:bx+block_M, by:by+block_N])

    return main


def test_matmul(main_args):
    os.environ['TILELANG_ASCEND_MODE'] = "Developer"
    M, N, K = main_args.M, main_args.N, main_args.K
    block_M, block_N = main_args.block_M, main_args.block_N

    func = matmul(M, N, K, block_M, block_N)
    compiled_kernel = tilelang.compile(func, target="npuir")

    A = torch.randn((M, K), dtype = torch.float32).npu()
    B = torch.randn((N, K), dtype = torch.float32).npu()
    C = torch.ones((M, N), dtype = torch.float32).npu()

    compiled_kernel(A, B, C)

    ref = A @ B.t() + 1
    torch.testing.assert_close(C, ref, rtol=1e-2, atol=1e-2)
    print("\033[92mAll check passed!\033[0m")

if __name__ == "__main__":
    args = parser.parse_args()
    test_matmul(args)