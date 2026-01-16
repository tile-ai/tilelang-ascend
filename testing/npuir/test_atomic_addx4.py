# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
import os

import tilelang
import tilelang.language as T

import torch
import torch_npu

tilelang.cache.clear_cache()

dtype = "float32"


def run_atomic_addx4(M, N, block_M, block_N, dtype="float32"):
    m_num = M // block_M
    n_num = N // block_N
    @T.prim_func
    def atomicAddx4Program(
            A: T.Tensor((M, N), dtype),
            B: T.Tensor((M, N), dtype),
            shape_M: T.int32, shape_N: T.int32,
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, _):
            blockx = cid // n_num
            blocky = cid % n_num
            A_VEC = T.alloc_ub((1, 4), dtype)

            for i, j in T.Parallel(block_M, block_N // 4):
                bx = blockx * block_M + i
                by = blocky * block_N + j * 4
                t0 = shape_M - bx
                tile_size_M = T.min(block_M, t0)        

                t0 = shape_N - by
                tile_size_N = T.min(block_N, t0)   
                T.copy(A[bx, by], A_VEC, [1, 4])
                T.npuir_atomic_addx4(A_VEC, B[bx, by], [1, 4])            
            
    return atomicAddx4Program

def test_vec_atomic_addx4():
    torch.npu.set_device(0)
    
    M, N = 256, 256
    
    A = torch.randn(size=[M, N], dtype=eval("torch." + dtype)).npu()
    B = torch.zeros(size=[M, N], dtype=eval("torch." + dtype)).npu()
    ref_B = B.clone()

    for i in range(M):
        for j in range(0, N - 3, 4):
            ref_B[i, j] += A[i, j]
            ref_B[i, j + 1] += A[i, j + 1]
            ref_B[i, j + 2] += A[i, j + 2]
            ref_B[i, j + 3] += A[i, j + 3]

    func = run_atomic_addx4(M, N, block_M=16, block_N=16)
    compiled_kernel = tilelang.compile(func, target="npuir")
    
    print("\nRunning atomic addx4...")
    compiled_kernel(A, B, M, N)
    print(f"Expected: First few elements: {B[0:16]}")
    print(f"Actual: First few elements of B: {ref_B[0:16]}")    
    torch.testing.assert_close(B, ref_B, atol=1e-3, rtol=1e-3)

if __name__ == "__main__":
    test_vec_atomic_addx4()