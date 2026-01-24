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
    M, N = 256, 256
    func = run_atomic_addx4(M, N, block_M=16, block_N=16)
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
    test_vec_atomic_addx4()