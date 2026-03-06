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


def vec_gather(block_M, block_N, dim, dtype="float16"):
    
    BLOCK_SIZE = 1
    itype = "int32"

    @T.prim_func
    def main(
            A: T.Tensor((block_M, block_N), dtype),
            B: T.Tensor((block_M, block_N), itype),
            C: T.Tensor((block_M, block_N), dtype),
    ):
        with T.Kernel(BLOCK_SIZE, is_npu=True) as (cid, _):
            
            A_VEC = T.alloc_ub((block_M, block_N), dtype)
            index_VEC = T.alloc_ub((block_M, block_N), itype)
            C_VEC = T.alloc_ub((block_M, block_N), dtype)
    
            T.copy(A, A_VEC)
            T.copy(B, index_VEC)
            T.npuir_gather(A_VEC[:block_M, :block_N], 
                           C_VEC[:block_M, :block_N],
                           index_VEC)
            T.copy(C_VEC, C)

    return main

def test_vec_gather():
    M, N = 32, 32
    torch.manual_seed(42)
    compile_kernel = vec_gather(M, N, dim=1)
    kernel = tilelang.compile(compile_kernel, target="npuir")

    A = torch.randn([M, N], dtype=torch.float16).npu()
    B = torch.randint(0, N, [M, N], dtype=torch.int32).npu()
    C = torch.zeros([M, N], dtype=torch.float16).npu()
    # print(kernel)
    ref_C = torch.gather(A, dim=1, index=B)
    kernel(A, B, C)

    print("verification")
    torch.testing.assert_close(C, ref_C, rtol=1e-2, atol=1e-2)
    print("test gather success")

if __name__ == "__main__":
    os.environ['TILELANG_ASCEND_MODE'] = 'Expert'
    test_vec_gather()
