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


def vec_pad(block_M, block_N, dtype="float16"):
    
    BLOCK_SIZE = 1


    @T.prim_func
    def main(
            A: T.Tensor((block_M, block_N), dtype),
            C: T.Tensor((2*block_M, block_N), dtype),
    ):
        with T.Kernel(BLOCK_SIZE, is_npu=True) as (cid, _):
            
            A_VEC = T.alloc_ub((block_M, block_N), dtype)
            C_VEC = T.alloc_ub((2*block_M, block_N), dtype)
    
            T.copy(A, A_VEC)
            T.npuir_pad(A_VEC[:block_M, :block_N], 
                        C_VEC, T.float16(0),
                        [block_M/2, 0],
                        [block_M/2, 0])
            T.copy(C_VEC, C)

    return main

def test_vec_pad():
    M, N = 32, 32
    torch.manual_seed(42)
    compile_kernel = vec_pad(M, N)
    kernel = tilelang.compile(compile_kernel, target="npuir")

    A = torch.randn([M, N], dtype=torch.float16).npu()
    C = torch.zeros([2*M, N], dtype=torch.float16).npu()
    # print(kernel)
    ref_C = torch.nn.functional.pad(A, (0, 0, 16, 16), mode='constant', value=0)
    kernel(A, C)

    print("verification")
    torch.testing.assert_close(C, ref_C, rtol=1e-2, atol=1e-2)
    print("test pad success")

if __name__ == "__main__":
    os.environ['TILELANG_ASCEND_MODE'] = 'Expert'
    test_vec_pad()
