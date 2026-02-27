# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
import os
import filecmp

import torch
import torch_npu

import tilelang
import tilelang.language as T

tilelang.cache.clear_cache()

torch.npu.set_device(0)


def vec_transpose(block_M, block_N, dtype="float16"):
    
    BLOCK_SIZE = 1

    @T.prim_func
    def main(
            A: T.Tensor((block_M, block_N), dtype),
            C: T.Tensor((block_N, block_M), dtype),
    ):
        with T.Kernel(BLOCK_SIZE, is_npu=True) as (cid, _):
            
            A_VEC = T.alloc_ub((block_M, block_N), dtype)
            C_VEC = T.alloc_ub((block_N, block_M), dtype)
    
            T.copy(A, A_VEC)
            T.npuir_transpose(A_VEC[:block_M, :block_N], 
                           C_VEC[:block_N, :block_M], [1, 0])
            T.copy(C_VEC, C)

    return main

def test_vec_transpose():
    M, N = 32, 32
    torch.manual_seed(42)
    compile_kernel = vec_transpose(M, N)
    kernel = tilelang.compile(compile_kernel, target="npuir")

    A = torch.randn([M, N], dtype=torch.float16).npu()
    C = torch.zeros([N, M], dtype=torch.float16).npu()
    # print(kernel)
    ref_C = torch.transpose(A, 0, 1)
    kernel(A, C)

    print("verification")
    torch.testing.assert_close(C, ref_C, rtol=1e-2, atol=1e-2)
    print("test transpose success")

if __name__ == "__main__":
    os.environ['TILELANG_ASCEND_MODE'] = 'Expert'
    test_vec_transpose()
