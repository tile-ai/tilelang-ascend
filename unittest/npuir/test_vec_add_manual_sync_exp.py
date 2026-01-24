# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
import os
import torch
import filecmp

import tilelang
import tilelang.language as T

tilelang.cache.clear_cache()

M = 1024
N = 1024
K = 1024

def vec_add_manual_sync(M, N, K, block_M, block_N, dtype="float16"):
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


def test_vec_add_manual_sync():
    func = vec_add_manual_sync(M, N, K, 128, 256)
    kernel = tilelang.engine.lower(func, target='npuir')
    # print(kernel)

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
    test_vec_add_manual_sync()