# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
import os

import tilelang
import tilelang.language as T

import torch
import torch_npu

tilelang.cache.clear_cache()

dtype = "float32"

def vec_atomic_add_1d(N, block_size, dtype="float32"):
    n_blocks = N // block_size

    @T.prim_func
    def vecAtomicAdd1D(
            A: T.Tensor((N,), dtype),
            B: T.Tensor((N,), dtype),
            shape: T.int32,
    ):
        with T.Kernel(n_blocks, is_npu=True) as (bid, _):
            A_VEC = T.alloc_shared((block_size,), dtype)
            start = bid * block_size
            t0 = shape - start 
            tail_size = T.min(block_size, t0)
            T.copy(A[start], A_VEC, [tail_size])
    
            T.npuir_atomic_add(A_VEC, B[start], [tail_size])

    return vecAtomicAdd1D

def vec_atomic_add_2d(M, N, block_M, block_N, dtype="float32"):
    m_num = M // block_M
    n_num = N // block_N
    @T.prim_func
    def vecAtomicAdd2D(
            A: T.Tensor((M, N), dtype),
            B: T.Tensor((M, N), dtype),
            shape_M: T.int32, shape_N: T.int32,
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, _):
            blockx = cid // n_num
            bx = blockx * block_M
            blocky = cid % n_num
            by = blocky * block_N
            A_VEC = T.alloc_shared((block_M, block_N), dtype)

            t0 = shape_M - bx
            tile_size_M = T.min(block_M, t0)        

            t0 = shape_N - by
            tile_size_N = T.min(block_N, t0)   
            T.copy(A[bx, by], A_VEC, [tile_size_M, tile_size_N]) 
            T.npuir_atomic_add(A_VEC, B[bx, by], [tile_size_M, tile_size_N])           


    return vecAtomicAdd2D

def test_vec_atomic_add_1d():
    torch.npu.set_device(0)
    vec_size = 64

    A = torch.randn(size=[vec_size], dtype=eval("torch." + dtype)).npu()
    B = torch.randn(size=[vec_size], dtype=eval("torch." + dtype)).npu()
    expected = A + B

    func = vec_atomic_add_1d(vec_size, block_size=32)
    compiled_kernel = tilelang.compile(func, target="npuir")

    print("Running 1D atomic add...")
    compiled_kernel(A, B, vec_size)

    print("1D Verification:")
    print(f"Expected: First few elements: {expected[0:64]}")
    print(f"Actual: First few elements of B: {B[0:64]}")
    print(f"All elements equal to expected: {torch.allclose(B, expected)}")

def test_vec_atomic_add_2d():
    torch.npu.set_device(0)

    M, N = 256, 256

    A = torch.randn(size=[M, N], dtype=eval("torch." + dtype)).npu()
    B = torch.randn(size=[M, N], dtype=eval("torch." + dtype)).npu()
    expected = A + B

    func = vec_atomic_add_2d(M, N, block_M=16, block_N=16)
    compiled_kernel = tilelang.compile(func, target="npuir")

    print("\nRunning 2D atomic add...")
    compiled_kernel(A, B, M, N)

    print("2D Verification:")

    print(f"Expected: First row elements: {expected[0:8, 0:8]}")
    print(f"Actual: First row elements of B: {B[0:8, 0:8]}")
    print(f"All elements equal to expected: {torch.allclose(B, expected)}")

if __name__ == "__main__":
    os.environ['TILELANG_ASCEND_MODE'] = 'Developer'
    test_vec_atomic_add_1d()
    test_vec_atomic_add_2d()