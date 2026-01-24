# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.\
import os
import filecmp

import tilelang
import tilelang.language as T

tilelang.cache.clear_cache()

M = 1024
N = 1024


def if_else(M, N, block_M, block_N, dtype="float16"):
    m_num = M // block_M
    n_num = N // block_N

    VEC_NUM = 2
    BLOCK_SIZE = 20
    step_core_num = 10

    @T.prim_func
    def main(
            A: T.Tensor((M, N), dtype),
            B: T.Tensor((M, N), dtype),
            C: T.Tensor((M, N), dtype),
            D: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(BLOCK_SIZE, is_npu=True) as (cid, _):
            bx_ = cid // n_num
            bx = bx_ * block_M
            by_ = cid % n_num
            by = by_* block_N
            A_VEC = T.alloc_ub((block_M, block_N), dtype)
            B_VEC = T.alloc_ub((block_M, block_N), dtype)
            C_VEC = T.alloc_ub((block_M, block_N), dtype)
            for i in T.serial(T.ceildiv(m_num*n_num, step_core_num)):
                block_id_base = i * step_core_num
                cid_in_step = cid % step_core_num
                block_id = block_id_base + cid_in_step
                if block_id < m_num*n_num:
                    block_id_m = block_id // n_num
                    block_id_n = block_id % n_num
                    bx = block_id_m * block_M
                    by = block_id_n * block_N
                    T.copy(A[bx, by], A_VEC)
                    T.copy(B[bx, by], B_VEC)
                    if cid < step_core_num:
                        T.npuir_add(A_VEC, B_VEC, C_VEC)
                    else:
                        T.npuir_sub(A_VEC, B_VEC, C_VEC)
                    if cid < step_core_num:
                        T.copy(C_VEC, C[bx, by])
                    else:
                        T.copy(C_VEC, D[bx, by])

    return main

def test_if_else():
    func = if_else(M, N, 64, 256)
    kernel = tilelang.engine.lower(func)
    print(kernel)

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
    test_if_else()
