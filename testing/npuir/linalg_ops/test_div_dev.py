# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
import os
import torch
import tilelang
import tilelang.language as T

def div_kernel(M, N,block_M, block_N):
    m_num = M // block_M
    n_num = N // block_N
    dtype = "float32"

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),
        B: T.Tensor((M, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, _):
            bx = block_M * (cid // n_num)
            by = block_N * (cid % n_num)

            A_shared = T.alloc_shared((block_M, block_N), dtype)
            B_shared = T.alloc_shared((block_M, block_N), dtype)
            C_shared = T.alloc_shared((block_M, block_N), dtype)

            T.copy(A[bx:bx + block_M, by:by + block_N], A_shared)
            T.copy(B[bx:bx + block_M, by:by + block_N], B_shared)

            T.npuir_div(A_shared, B_shared, C_shared)

            T.copy(C_shared, C[bx:bx + block_M, by:by + block_N])

    return main


def test_div_kernel():
    os.environ['TILELANG_ASCEND_MODE'] = "Developer"
    M, N = 16, 16
    block_M, block_N = 16, 16

    func = div_kernel(M, N, block_M, block_N)

    A = torch.randn((M, N), dtype = torch.float32).npu()
    B = torch.randn((M, N), dtype = torch.float32).npu()
    C = torch.zeros((M, N), dtype = torch.float32).npu()

    compiled = tilelang.compile(func, target="npuir")
    compiled(A, B, C)

    ref = A/B
    torch.testing.assert_close(C, ref, rtol=1e-2, atol=1e-2)
    print("\033[92mDIV check passed!\033[0m")

if __name__ == "__main__":
    test_div_kernel()