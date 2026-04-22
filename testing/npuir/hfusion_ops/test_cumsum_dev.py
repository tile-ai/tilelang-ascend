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
K = 64
N = 32


def vec_cumsum(M, K, N, dtype="float16"):
    @T.prim_func
    def main(A: T.Tensor((M, K, N), dtype),
             B: T.Tensor((M, K, N), dtype),
             ):
        with T.Kernel(1, is_npu=True) as (cid, _):
            a = T.alloc_shared((M, K, N), dtype)
            s = T.alloc_shared((M, K, N), dtype)

            T.copy(A, a)
            T.cumsum(a, s, dim=2)

            T.copy(s, B)

    return main


def test_cumsumt():
    torch.npu.set_device(0)
    os.environ['TILELANG_ASCEND_MODE'] = 'Developer'

    func = vec_cumsum(M, K, N)
    compiled_kernel = tilelang.compile(func, target="npuir")

    v1 = torch.randn(size=[M, K, N], dtype=eval("torch." + dtype)).npu()
    v2 = torch.randn(size=[M, K, N], dtype=eval("torch." + dtype)).npu()

    v_ref = torch.cumsum(v1, dim=2)
    compiled_kernel(v1, v2)

    torch.testing.assert_close(v_ref, v2, rtol=1e-2, atol=1e-2)
    print("Cumsum Pass!")


if __name__ == "__main__":
    test_cumsumt()