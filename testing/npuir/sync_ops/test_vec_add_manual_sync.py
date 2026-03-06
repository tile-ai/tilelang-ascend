# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
import os
import torch

import tilelang
import tilelang.language as T

tilelang.cache.clear_cache()
torch.npu.set_device(0)

M = 1024
N = 1024
K = 1024

def vec_add(M, N, K, block_M, block_N, dtype="float16"):
    m_num = M // block_M
    n_num = N // block_N

    VEC_NUM = 2

    @T.prim_func
    def main(
            A: T.Tensor((M, K), dtype),
            B: T.Tensor((K, N), dtype),
            C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, _):
            bx_ = cid // n_num
            bx = bx_ * block_M
            by_ = cid % n_num
            by = by_ * block_N
            
            A_VEC = T.alloc_ub((block_M, block_N), dtype)
            B_VEC = T.alloc_ub((block_M, block_N), dtype)
            C_VEC = T.alloc_ub((block_M, block_N), dtype)
            with T.rs("PIPE_MTE2"):
                T.copy(A[bx, by], A_VEC)
                T.copy(B[bx, by], B_VEC)
                T.set_flag("PIPE_V", 0)
            
            with T.rs("PIPE_V"):
                T.wait_flag("PIPE_MTE2", 0)
                T.npuir_add(A_VEC, B_VEC, C_VEC)
                T.set_flag("PIPE_MTE3", 0)

            with T.rs("PIPE_MTE3"):
                T.wait_flag("PIPE_V", 0)
                T.copy(C_VEC, C[bx, by])
                        
    return main


def test_vec_add():
    func = vec_add(M, N, K, 128, 256)
    
    kernel = tilelang.compile(func, target="npuir")

    A = torch.randn([M, K]).half().npu()
    B = torch.randn([K, N]).half().npu()
    C = torch.randn([M, N]).half().npu()

    kernel(A, B, C)

    print(C)
    ref_cpu = torch.add(A, B)
    print(ref_cpu)

    torch.testing.assert_close(C, ref_cpu, rtol=1e-2, atol=1e-2)
    print("\033[92mAll check passed!\033[0m")

if __name__ == "__main__":
    test_vec_add()