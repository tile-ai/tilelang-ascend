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

def vdeinterleave_kernel(M, N, dtype):
    BLOCK_SIZE = 1

    @T.prim_func
    def main(
        C: T.Tensor((M, N * 2), dtype),
        A: T.Tensor((M, N), dtype),
        B: T.Tensor((M, N), dtype)
    ):
        with T.Kernel(BLOCK_SIZE, is_npu=True) as (cid, _):
            C_ub = T.alloc_shared((M, N * 2), dtype)
            A_ub = T.alloc_shared((M, N), dtype)
            B_ub = T.alloc_shared((M, N), dtype)

            T.copy(C, C_ub)
            T.deinterleave(C_ub, A_ub, B_ub, channel_nums=2)
            T.copy(A_ub, A)
            T.copy(B_ub, B)

    return main

def test_vdeinterleave_kernel():
    torch.npu.set_device(0)
    os.environ['TILELANG_ASCEND_MODE'] = 'Developer'

    func = vdeinterleave_kernel(M, N, dtype)
    compiled_kernel = tilelang.compile(func, target="npuir")

    v3 = torch.randn(size=[M, N * 2], dtype=eval("torch." + dtype)).npu()
    v1 = torch.randn(size=[M, N], dtype=eval("torch." + dtype)).npu()
    v2 = torch.randn(size=[M, N], dtype=eval("torch." + dtype)).npu()

    v_unflatten = v3.unflatten(-1, (N, 2))
    ref_a, ref_b = v_unflatten.split(1, dim=-1)
    ref_a = ref_a.squeeze(-1).to(dtype=eval("torch." + dtype))
    ref_b = ref_b.squeeze(-1).to(dtype=eval("torch." + dtype))

    compiled_kernel(v3, v1, v2)

    torch.testing.assert_close(v1, ref_a, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(v2, ref_b, rtol=1e-2, atol=1e-2)
    print("Deinterleave Pass!")

if __name__ == "__main__":
    test_vdeinterleave_kernel()