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


def vec_interleave(block_M, block_N, dtype="float16"):
    
    BLOCK_SIZE = 1

    @T.prim_func
    def main(
            A: T.Tensor((block_M, block_N), dtype),
            B: T.Tensor((block_N, block_N), dtype),
            C: T.Tensor((block_N, 2*block_N), dtype),
    ):
        with T.Kernel(BLOCK_SIZE, is_npu=True) as (cid, _):
            
            A_VEC = T.alloc_ub((block_M, block_N), dtype)
            B_VEC = T.alloc_ub((block_M, block_N), dtype)
            C_VEC = T.alloc_ub((block_M, 2*block_N), dtype)
    
            T.copy(A, A_VEC)
            T.copy(B, B_VEC)
            T.npuir_interleave(A_VEC[:block_M, :block_N], 
                               B_VEC[:block_M, :block_N],
                               C_VEC[:block_M, :2*block_N])
            T.copy(C_VEC, C)

    return main

def interleave_tensors(*tensors, dim=0):
    """
    tensor interleave func
    """
    stacked = torch.stack(tensors, dim=dim + 1)
    shape = list(stacked.shape)
    shape[dim] *= len(tensors)
    shape.pop(dim + 1)
    return stacked.view(shape)


def test_vec_interleave():
    M, N = 32, 32
    torch.manual_seed(42)
    compile_kernel = vec_interleave(M, N)
    kernel = tilelang.compile(compile_kernel, target="npuir")

    A = torch.randn([M, N], dtype=torch.float16).npu()
    B = torch.randn([M, N], dtype=torch.float16).npu()
    C = torch.zeros([M, 2*N], dtype=torch.float16).npu()
    # print(kernel)
    ref_C = interleave_tensors(A, B, dim=1)
    kernel(A, B, C)

    print("verification")
    torch.testing.assert_close(C, ref_C, rtol=1e-2, atol=1e-2)
    print("test interleave success")

if __name__ == "__main__":
    os.environ['TILELANG_ASCEND_MODE'] = 'Expert'
    test_vec_interleave()
