import sys
import os
import argparse
import torch

import tilelang
import tilelang.language as T

torch.npu.set_device(0)
tilelang.cache.clear_cache()

def vec_add(block_M, block_N):
    M = T.symbolic("M")
    N = T.symbolic("N")
    m_num = M // block_M
    n_num = N // block_N
    dtype = "float16"
    BLOCK_SIZE = 20

    @T.prim_func
    def main(
            A: T.Tensor((M, N), dtype),
            B: T.Tensor((N), dtype),
            C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(BLOCK_SIZE, is_npu=True) as (cid, _):
            A_VEC = T.alloc_ub((block_M, block_N), dtype)
            B_VEC = T.alloc_ub((block_N), dtype)
            C_VEC = T.alloc_ub((block_M, block_N), dtype)
            for i in T.serial(T.ceildiv(m_num*n_num, BLOCK_SIZE)):
                block_id = i * BLOCK_SIZE + cid
                if block_id < m_num * n_num:
                    block_id_m = block_id // n_num
                    block_id_n = block_id % n_num
                    bx = block_id_m * block_M
                    by = block_id_n * block_N
                    T.copy(A[bx, by], A_VEC)
                    T.copy(B[:block_N], B_VEC)
                    T.npuir_add(A_VEC, B_VEC[0], C_VEC)
                    T.copy(C_VEC, C[bx, by])

    return main

def run_test():
    M, N = 128, 256
    func = vec_add(32, 32)
    compiled_kernel = tilelang.compile(func, target='npuir')

    a = torch.randn(M, N).half().npu()
    b = torch.randn(N).half().npu()
    c = torch.randn(M, N).half().npu()

    torch.manual_seed(88888888)  # set the random seed for torch
    dtype = "float16"

    ref_output = a + b[0]
    compiled_kernel(a, b, c)
    print("Actual Result:")
    print(c)
    print("Expected Result:")
    print(ref_output)
    torch.testing.assert_close(c, ref_output, rtol=1e-2, atol=1e-2)
    print("\033[92mAll check passed!\033[0m")

if __name__ == "__main__":
    run_test()