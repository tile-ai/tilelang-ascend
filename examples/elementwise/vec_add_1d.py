# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
import os

import tilelang
import tilelang.language as T

import torch
import torch_npu

tilelang.cache.clear_cache()

dtype = "float32"
seq_len = 4096


def vec_add(N, block_N, dtype="float32"):
    n_num = N // block_N

    @T.prim_func
    def main(
            A: T.Tensor((N), dtype),
            B: T.Tensor((N), dtype),
            C: T.Tensor((N), dtype),
            shape: T.int32,
    ):
        with T.Kernel(n_num, is_npu=True) as (cid, _):
            A_VEC = T.alloc_ub((block_N), dtype)
            B_VEC = T.alloc_ub((block_N), dtype)
            C_VEC = T.alloc_ub((block_N), dtype)
            t0 = cid * block_N
            t0 = shape - t0
            tail_size = T.min(block_N, t0)
            T.copy(A[cid * block_N], A_VEC, [tail_size])
            T.copy(B[cid * block_N], B_VEC, [tail_size])

            T.npuir_add(A_VEC, B_VEC, C_VEC)
            T.copy(C_VEC, C[cid * block_N], [tail_size])

    return main


def test_vec_add():
    torch.npu.set_device(0)
    func = vec_add(seq_len, seq_len)
    compiled_kernel = tilelang.compile(func, target="npuir")

    v1 = torch.randn(size=[seq_len], dtype=eval("torch." + dtype)).npu()
    v2 = torch.randn(size=[seq_len], dtype=eval("torch." + dtype)).npu()
    v3 = torch.zeros(size=[seq_len], dtype=eval("torch." + dtype)).npu()

    y_ref = v1 + v2
    compiled_kernel(v1, v2, v3, seq_len)

    print(y_ref)
    print(v3)


if __name__ == "__main__":
    test_vec_add()
