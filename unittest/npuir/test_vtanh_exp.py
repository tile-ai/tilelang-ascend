# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.

import os
import torch
import tilelang
import tilelang.language as T
import filecmp

tilelang.cache.clear_cache()

M = 16
N = 16
BLOCK_M = 16
BLOCK_N = 16
DTYPE = "float16"

def generate_tensor_new(shape, dtype, data_range):
    return torch.empty(shape, dtype = dtype).uniform_(data_range[0], data_range[1])

def vec_tanh(M, N, block_M, block_N, dtype="float16"):
    m_num = M // block_M
    n_num = N // block_N
    BLOCK_SIZE = 8

    @T.prim_func
    def main(
            A: T.Tensor((M, N), dtype),
            B: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(BLOCK_SIZE, is_npu=True) as (cid, _):
            A_VEC = T.alloc_ub((block_M, block_N), dtype)
            B_VEC = T.alloc_ub((block_M, block_N), dtype)

            for i in T.serial(T.ceildiv(m_num * n_num, BLOCK_SIZE)):
                block_id = i * BLOCK_SIZE + cid
                if block_id < m_num * n_num:
                    block_id_m = block_id // n_num
                    block_id_n = block_id % n_num
                    bx = block_id_m * block_M
                    by = block_id_n * block_N

                    T.copy(A[bx, by], A_VEC)
                    T.npuir_vtanh(A_VEC, B_VEC)
                    T.copy(B_VEC, B[bx, by])
    return main


def test_vec_tanh():
    func = vec_tanh(M, N, BLOCK_M, BLOCK_N, DTYPE)
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
    test_vec_tanh()
