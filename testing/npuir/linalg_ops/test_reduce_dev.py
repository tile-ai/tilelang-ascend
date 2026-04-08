# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
import os

import tilelang
import tilelang.language as T

import torch
import torch_npu

tilelang.cache.clear_cache()

dtype = "float32"
accum_dtype = "float16"

M = 2
K = 64
N = 32


def vec_reduce_3d_to_2d_max(M, K, N, dtype="float32"):
    @T.prim_func
    def main_max(A: T.Tensor((M, K, N), dtype),
             B: T.Tensor((M, K, 1), dtype),
             ):
        with T.Kernel(1, is_npu=True) as (cid, _):
            a = T.alloc_shared((M, K, N), dtype)
            s = T.alloc_shared((M, K, 1), dtype)

            T.copy(A, a)
            T.reduce(a, s, dims=2, reduce_mode="max", clear=True)

            T.copy(s, B)

    return main_max


def vec_reduce_3d_to_2d_sum(M, K, N, dtype="float32"):
    @T.prim_func
    def main_sum(A: T.Tensor((M, K, N), dtype),
             B: T.Tensor((M, K, 1), dtype),
             ):
        with T.Kernel(1, is_npu=True) as (cid, _):
            a = T.alloc_shared((M, K, N), dtype)
            s = T.alloc_shared((M, K, 1), dtype)

            T.copy(A, a)
            T.reduce(a, s, dims=2, reduce_mode="sum", clear=True)

            T.copy(s, B)

    return main_sum


def test_vec_reduce_max():
    torch.npu.set_device(0)
    os.environ['TILELANG_ASCEND_MODE'] = 'Developer'

    func = vec_reduce_3d_to_2d_max(M, K, N)
    compiled_kernel = tilelang.compile(func, target="npuir")

    v1 = torch.randn(size=[M, K, N], dtype=eval("torch." + dtype)).npu()
    v2 = torch.randn(size=[M, K, 1], dtype=eval("torch." + dtype)).npu()

    v_ref = torch.max(v1, dim=2).values.reshape(M, K, 1)
    compiled_kernel(v1, v2)

    print("Max 参考结果 shape:", v_ref.shape)
    print("Max 计算结果 shape:", v2.shape)
    torch.testing.assert_close(v_ref, v2, rtol=1e-2, atol=1e-2)
    print("Max Reduce 校验通过")


def test_vec_reduce_sum():
    torch.npu.set_device(0)
    os.environ['TILELANG_ASCEND_MODE'] = 'Developer'

    func = vec_reduce_3d_to_2d_sum(M, K, N)
    compiled_kernel = tilelang.compile(func, target="npuir")

    v1 = torch.randn(size=[M, K, N], dtype=eval("torch." + dtype)).npu()
    v2 = torch.randn(size=[M, K, 1], dtype=eval("torch." + dtype)).npu()

    v_ref = torch.sum(v1, dim=2, keepdim=True)
    compiled_kernel(v1, v2)

    print("Sum 参考结果 shape:", v_ref.shape)
    print("Sum 计算结果 shape:", v2.shape)
    torch.testing.assert_close(v_ref, v2, rtol=1e-2, atol=1e-2)
    print("Sum Reduce 校验通过")


if __name__ == "__main__":
    test_vec_reduce_max()
    test_vec_reduce_sum()
    print("所有测试全部通过！")