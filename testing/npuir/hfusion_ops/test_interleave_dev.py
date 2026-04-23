# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
import os

import tilelang
import tilelang.language as T

import torch
import torch_npu

tilelang.cache.clear_cache()
dtype = "float16"
M = 2
N = 32

def vinterleave_kernel(M, N, dtype):
    BLOCK_SIZE = 1

    @T.prim_func
    def main(A: T.Tensor((M, N), dtype),
             B: T.Tensor((M, N), dtype),
             C: T.Tensor((M, N * 2), dtype)):

        with T.Kernel(BLOCK_SIZE, is_npu=True) as (cid, _):

            A_ub = T.alloc_shared((M, N), dtype)
            B_ub = T.alloc_shared((M, N), dtype)
            C_ub = T.alloc_shared((M, N * 2), dtype)

            T.copy(A, A_ub)
            T.copy(B, B_ub)
            T.interleave(A_ub, B_ub, C_ub, channel_nums=2)
            T.copy(C_ub, C)

    return main

def test_vinterleave_kernel():
    torch.npu.set_device(0)
    os.environ['TILELANG_ASCEND_MODE'] = 'Developer'

    func = vinterleave_kernel(M, N, dtype)
    compiled_kernel = tilelang.compile(func, target="npuir")

    v1 = torch.randn(size=[M, N], dtype=eval("torch." + dtype)).npu()
    v2 = torch.randn(size=[M, N], dtype=eval("torch." + dtype)).npu()
    v3 = torch.randn(size=[M, N * 2], dtype=eval("torch." + dtype)).npu()

    v_ref = torch.cat([v1.unsqueeze(-1), v2.unsqueeze(-1)], dim=-1).flatten(-2)
    compiled_kernel(v1, v2, v3)

    torch.testing.assert_close(v_ref, v3, rtol=1e-2, atol=1e-2)
    print("Interleave Pass!")

if __name__ == "__main__":
    test_vinterleave_kernel()