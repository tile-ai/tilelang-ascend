# Copyright (c) Huawei Technologies Co., Ltd. 2025.
import torch
import pytest
import tilelang
import tilelang.language as T

import testcommon as tc


pytestmark = [pytest.mark.mode("Developer")]

M = 256
N = 256
BLOCK_M = 64
BLOCK_N = 64


def ref_program(x, y):
    block_row_indices = torch.arange(x.shape[0], device=x.device) // BLOCK_M * BLOCK_M
    return x + y.index_select(0, block_row_indices)


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
            T.copy(B[by * block_M, bx * block_N], B_shared)
            T.npuir_add(A_shared[:, :], B_shared[0, :], C_local[:, :])
            T.copy(C_local, C_shared)
            T.copy(C_shared, C[by * block_M, bx * block_N])

    return elemAdd


@pytest.mark.op("vadd_auto_brc_dev")
def test_vadd_auto_brc():
    a = torch.randn(M, N, dtype=torch.float32, device="npu")
    b = torch.randn(M, N, dtype=torch.float32, device="npu")
    c = torch.zeros(M, N, dtype=torch.float32, device="npu")

    func = elementwise_add(M, N, BLOCK_M, BLOCK_N, in_dtype="float32", out_dtype="float32")
    kernel = tilelang.compile(func, target="npuir")

    kernel(a, b, c)

    tc.assert_close(c.cpu(), ref_program(a, b).cpu(), rtol=1e-2, atol=1e-2)
