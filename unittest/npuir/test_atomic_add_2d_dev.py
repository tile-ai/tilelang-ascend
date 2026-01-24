# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
import os
import filecmp

import tilelang
import tilelang.language as T

import torch
import torch_npu

tilelang.cache.clear_cache()

dtype = "float32"

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

def test_vec_atomic_add_2d():
    os.environ['TILELANG_ASCEND_MODE'] = 'Developer'
    M, N = 256, 256
    func = vec_atomic_add_2d(M, N, block_M=16, block_N=16)
    kernel = tilelang.engine.lower(func, target='npuir')
    curr_name = os.path.splitext(os.path.basename(__file__))[0][5:] + ".mlir"
    # Export to .mlir file
    output_file = './output/' + curr_name
    with open(output_file, 'w') as f:
        f.write(kernel)
    
    ref_file = "./mlir_files/" + curr_name
    # filecmp.cmp returns True if files are identical, False otherwise
    are_identical = filecmp.cmp(output_file, ref_file , shallow=False)
    # assertion for pytest
    assert are_identical, f"'{output_file}' and '{ref_file}' are not identical"

if __name__ == "__main__":
    test_vec_atomic_add_2d()