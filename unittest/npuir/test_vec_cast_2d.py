# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
import os
import filecmp

import tilelang
import tilelang.language as T

tilelang.cache.clear_cache()

M = 512
N = 512

def vec_cast(M, N, block_M, block_N, src_dtype="float32", dst_dtype="float16"):
    m_num = M // block_M
    n_num = N // block_N

    @T.prim_func
    def main(
            A: T.Tensor((M, N), src_dtype),
            B: T.Tensor((M, N), dst_dtype),
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, _):
            bx_ = cid // n_num
            bx = bx_ * block_M
            by_ = cid % n_num
            by = by_ * block_N

            A_VEC = T.alloc_ub((block_M, block_N), src_dtype)
            B_VEC = T.alloc_ub((block_M, block_N), dst_dtype)
            T.copy(A[bx, by], A_VEC)
            T.npuir_cast(A_VEC, B_VEC, size=[128, 256], round_mode="round")
            T.copy(B_VEC, B[bx, by])

    return main

def test_vec_cast():
    func = vec_cast(M, N, 128, 256)
    kernel = tilelang.engine.lower(func)
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
    test_vec_cast()
