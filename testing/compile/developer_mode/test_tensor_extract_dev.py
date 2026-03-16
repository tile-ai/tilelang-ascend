# Copyright (c) Huawei Technologies Co., Ltd. 2025.
import os
import pytest
import argparse
import torch

import tilelang

from testcommon import npuir_compile_to_bin

pytestmark = [pytest.mark.mode("Developer")]
import tilelang.language as T

tilelang.cache.clear_cache()

parser = argparse.ArgumentParser(description="NPU Kernel Compilation")
parser.add_argument("--M", type=int, default=4, help="")
parser.add_argument("--N", type=int, default=4, help="")
parser.add_argument("--n", type=int, default=32, help="")

def vec_add(M, N, n):
    dtype = "float32"

    @T.prim_func
    def add(
            A: T.Tensor((M, N), dtype),
            B: T.Tensor((3,), dtype),
            C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(n, is_npu=True) as (cid, _):
            i = cid
            A_ub = T.alloc_shared((1, N), dtype)
            B_ub = T.alloc_shared((3,), dtype)
            C_ub = T.alloc_shared((1, N), dtype)

            T.copy(A[i, :], A_ub)
            T.copy(B, B_ub)

            T.npuir_add(A_ub, B_ub[0], C_ub)

            T.copy(C_ub, C[i, :])

    return add

def generate_tensor(shape, dtype, clear=False):
    """generate tensor"""
    if clear:
        return torch.zeros(shape, dtype=eval("torch." + dtype))
    if dtype in ("float32", "float16", "bfloat16"):
        return torch.randn(size=shape, dtype=eval("torch." + dtype))
    if dtype in ("int32", "int64", "int16"):
        return torch.randint(low=0, high=2000, size=shape, dtype=eval("torch." + dtype))
    if dtype == "int8":
        return torch.randint(low=0, high=127, size=shape, dtype=eval("torch." + dtype))
    if dtype == "bool":
        return torch.randint(low=0, high=2, size=shape).bool()
    raise ValueError('Invalid parameter "dtype" is found : {}'.format(dtype))


def test_tensor_extract():
    main_args = parser.parse_args([])
    func = vec_add(
        main_args.M,
        main_args.N,
        main_args.n,

    )
    kernel = tilelang.engine.lower(func, target='npuir')
    # print(kernel)

    result = npuir_compile_to_bin(kernel)
    assert result is not None and len(result) > 0, "npuir compile failed or returned empty"

if __name__ == "__main__":
    test_tensor_extract()