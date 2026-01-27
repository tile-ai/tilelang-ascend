# Copyright (c) Huawei Technologies Co., Ltd. 2025.
import os
import argparse
import torch
import tilelang
import tilelang.language as T

from utils import assert_compile_to_kernel_o_success


def ref_program(x, y):
    return x + y[0, :]

def elementwise_add(M, N, block_M, block_N, in_dtype="float32", out_dtype="float32"):
    @T.prim_func
    def elemAdd(
            A: T.Tensor((M, N), in_dtype),
            B: T.Tensor((M, N), in_dtype),
            C: T.Tensor((M, N), out_dtype)
    ):
        with T.Kernel(T.ceildiv(N, block_N) * T.ceildiv(M, block_M), is_npu=True) as (cid, _):
            by = cid // T.ceildiv(N, block_N)
            bx = cid % T.ceildiv(N, block_N)

            A_shared = T.alloc_shared((block_M, block_N), in_dtype)
            B_shared = T.alloc_shared((block_M, block_N), in_dtype)
            C_local = T.alloc_fragment((block_M, block_N), out_dtype)
            C_shared = T.alloc_shared((block_M, block_N), out_dtype)

            T.copy(A[by * block_M, bx * block_N], A_shared)
            T.copy(B[0, bx * block_N], B_shared[0, :])
            T.npuir_add(A_shared[:, :], B_shared[0, :], C_local[:, :])
            T.copy(C_local, C_shared)
            T.copy(C_shared, C[by * block_M, bx * block_N])

    return elemAdd


def test_elementwise_add_compile():
    os.environ["TILELANG_ASCEND_MODE"] = "Developer"

    kernel = elementwise_add(M=256, N=256, block_M=64, block_N=64, in_dtype="float32", out_dtype="float32")
    o_bytes = assert_compile_to_kernel_o_success(kernel)
    assert o_bytes is not None and len(o_bytes) > 0


if __name__ == "__main__":
    test_elementwise_add_compile()
