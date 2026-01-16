# Copyright (c) Huawei Technologies Co., Ltd. 2025.
import os
import torch
import tilelang
import tilelang.language as T


tilelang.cache.clear_cache()
dtype="float16"

def copy_shape_1d_2d(M, N, block_M, block_N):

    @T.prim_func
    def copyShape(
            A: T.Tensor((M, N), dtype),
            B: T.Tensor((M, N), dtype),
            shape_M: T.int32, shape_N: T.int32,
    ):
        with T.Kernel(T.ceildiv(M, block_M) * T.ceildiv(N, block_N), is_npu=True) as (cid, _):
            blockx = cid // T.ceildiv(N, block_N)
            blocky = cid % T.ceildiv(N, block_N)
            by = blocky * block_N

            A_BUF = T.alloc_shared((block_N), dtype)

            for i in T.Parallel(block_M):
                bx = blockx * block_M + i 
                t0 = shape_N - by
                tile_size_N = T.min(block_N, t0) 
                T.copy(A[bx, by], A_BUF, [tile_size_N])  
                T.copy(A_BUF, B[bx, by], [tile_size_N])       

    return copyShape

def copy_shape_2d_3d(M, N, block_M, block_N):
    m_num = M // block_M
    n_num = N // block_N

    @T.prim_func
    def copyShape2D3D(
            A: T.Tensor((1, M, N), dtype),
            B: T.Tensor((1, M, N), dtype),
            shape_M: T.int32, shape_N: T.int32,
    ):
        with T.Kernel(T.ceildiv(M, block_M) * T.ceildiv(N, block_N), is_npu=True) as (cid, _):
            blockx = cid // T.ceildiv(N, block_N)
            blocky = cid % T.ceildiv(N, block_N)
            by = blocky * block_N

            A_BUF = T.alloc_shared((1, block_N), dtype)

            for i in T.Parallel(block_M):
                bx = blockx * block_M + i
                t0 = shape_N - by
                tile_size_N = T.min(block_N, t0) 
                T.copy(A[0, bx, by], A_BUF, [1, tile_size_N])  
                T.copy(A_BUF, B[0, bx, by], [1, tile_size_N])       
                
    return copyShape2D3D

def test_copy_shape_1d_2d():
    # In the futrue, Developer mode and Expert Mode will transition smoothly without
    # requiring explicit declarations.
    torch.npu.set_device(0)
    M = 8
    N = 8
    v1 = torch.randn(size=[M, N], dtype=eval("torch." + dtype)).npu()
    v2 = torch.zeros(size=[M, N], dtype=eval("torch." + dtype)).npu()
    v_ref = v1.clone()
    func = copy_shape_1d_2d(M, N, block_M=3, block_N=3)
    compiled_kernel = tilelang.compile(func, target="npuir")
    
    compiled_kernel(v1, v2, M, N)
    print(v_ref)
    print(v2)
    torch.testing.assert_close(v2, v_ref, rtol=1e-2, atol=1e-2)
    print("\033[92mAll check passed!\033[0m")

def test_copy_shape_2d_3d():
    # In the futrue, Developer mode and Expert Mode will transition smoothly without
    # requiring explicit declarations.
    M = 8
    N = 8
    # In the futrue, it will be optimized to automatically derive the workspace size.
    func = copy_shape_2d_3d(M, N, 3, 3)
    compiled_kernel = tilelang.compile(func, target="npuir")

    v1 = torch.randn(size=[1, M, N], dtype=eval("torch." + dtype)).npu()
    v2 = torch.randn(size=[1, M, N], dtype=eval("torch." + dtype)).npu()
    v_ref = v1.clone()
    compiled_kernel(v1, v2, M, N)

    print(v_ref)
    print(v2)
    torch.testing.assert_close(v2, v_ref, rtol=1e-2, atol=1e-2)
    print("\033[92mAll check passed!\033[0m")

if __name__ == "__main__":

    os.environ['TILELANG_ASCEND_MODE'] = 'Developer'

    test_copy_shape_1d_2d()
    test_copy_shape_2d_3d()

