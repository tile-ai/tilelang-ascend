# Copyright (c) Huawei Technologies Co., Ltd. 2025.
import pytest
import torch
import torch_npu  # noqa: F401

import tilelang
import tilelang.language as T

from testcommon import assert_close, gen_tensor

pytestmark = [
    pytest.mark.op("reduce"),
    pytest.mark.mode("Developer"),
]

DTYPES = ["float16"]
ACCUM_DTYPES = ["float16"]


def row_reduce_sum_dev(M, N, block_M, dtype, accum_dtype):
    BLOCK_SIZE = 1

    @T.prim_func
    def rowReduceSumDevSolo(
        A: T.Tensor((M, N), dtype),
        B: T.Tensor((M, N), dtype),
        C: T.Tensor((M, N), dtype),
        D: T.Tensor((M, N), dtype),
        O: T.Tensor((M, 1), accum_dtype),
    ):
        with T.Kernel(BLOCK_SIZE, is_npu=True) as (cid, _):
            a = T.alloc_shared((M, N), dtype)
            b = T.alloc_shared((M, N), dtype)
            c = T.alloc_shared((M, N), dtype)
            d = T.alloc_shared((M, N), dtype)
            s = T.alloc_shared((M, 1), accum_dtype)
            T.copy(A, a)
            T.copy(B, b)
            T.copy(C, c)
            T.copy(D, d)
            T.reduce_abssum(a, s)
            T.reduce_max(b, s, clear=False)
            T.reduce_min(c, s, clear=False)
            T.reduce(d, s, dims=1, reduce_mode="sum", clear=False)
            T.copy(s, O)

    return rowReduceSumDevSolo


def _ref_row_reduce(A, B, C, D):
    res1 = torch.sum(torch.abs(A), dim=1, keepdim=True)
    res2 = torch.max(B, dim=1, keepdim=True).values
    res3 = torch.maximum(res1, res2)
    res4 = torch.min(C, dim=1, keepdim=True).values
    res5 = torch.minimum(res3, res4)
    return res5 + torch.sum(D, dim=1, keepdim=True)


@pytest.mark.parametrize(
    "dtype,accum_dtype", list(zip(DTYPES, ACCUM_DTYPES, strict=True))
)
def test_row_reduce_sum_dev(dtype, accum_dtype):
    M, N = 16, 16
    shape = (M, N)
    shape2 = (M, 1)
    A = gen_tensor(shape, dtype, kind="randn")
    B = gen_tensor(shape, dtype, kind="randn")
    C = gen_tensor(shape, dtype, kind="randn")
    D = gen_tensor(shape, dtype, kind="randn")
    O = gen_tensor(shape2, accum_dtype, kind="zeros")
    ref = _ref_row_reduce(A.cpu(), B.cpu(), C.cpu(), D.cpu())

    func = row_reduce_sum_dev(
        M=M, N=N, block_M=32, dtype=dtype, accum_dtype=accum_dtype
    )
    compiled = tilelang.compile(func, target="npuir")
    compiled(A, B, C, D, O)

    assert_close(O.cpu(), ref, dtype=accum_dtype, rtol=1e-2, atol=1e-2)
