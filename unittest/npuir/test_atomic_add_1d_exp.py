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

def vec_atomic_add_1d(N, block_size, dtype="float32"):
    n_blocks = N // block_size
    
    @T.prim_func
    def vecAtomicAdd1D(
            A: T.Tensor((N,), dtype),
            B: T.Tensor((N,), dtype),
            shape: T.int32,
    ):
        with T.Kernel(n_blocks, is_npu=True) as (bid, _):
            A_VEC = T.alloc_ub((block_size,), dtype)
            # B_VEC = T.alloc_ub((block_size,), dtype)
            
            start = bid * block_size
            t0 = shape - start
            tail_size = T.min(block_size, t0)
            
            T.copy(A[start], A_VEC, [tail_size])

            T.npuir_atomic_add(A_VEC, B[start], [tail_size])
            
            # T.copy(A_VEC, B[start], [tail_size])

    return vecAtomicAdd1D

def test_vec_atomic_add_1d():
    vec_size = 64
    func = vec_atomic_add_1d(vec_size, block_size=32)
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
    test_vec_atomic_add_1d()
