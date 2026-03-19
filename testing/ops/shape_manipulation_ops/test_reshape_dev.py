# Copyright (c) Huawei Technologies Co., Ltd. 2025.
import pytest
import torch
import torch_npu  # noqa: F401

import tilelang
import tilelang.language as T

from testcommon import assert_close, gen_tensor

pytestmark = [
    pytest.mark.op("reshape"),
    pytest.mark.mode("Developer"),
]

DTYPES = ["float16"]


def reshape_dev(M, N, dtype):
    BLOCK_SIZE = 1

    @T.prim_func
    def reshapeDev(
        A: T.Tensor((M, N), dtype),
        B: T.Tensor((N, M), dtype),
    ):
        with T.Kernel(BLOCK_SIZE, is_npu=True) as (cid, _):
            a = T.alloc_shared((M, N), dtype)
            b = T.alloc_shared((N, M), dtype)
            T.copy(A, a)
            T.npuir_reshape(a, b)
            T.npuir_exp(b, b)
            T.copy(b, B)

    return reshapeDev


@pytest.mark.parametrize("dtype", DTYPES)
def test_reshape_dev(dtype):
    M, N = 8, 16
    A = gen_tensor((M, N), dtype, kind="randn")
    B = gen_tensor((N, M), dtype, kind="zeros")
    ref = torch.exp(A.cpu().reshape(N, M))

    func = reshape_dev(M=M, N=N, dtype=dtype)
    compiled = tilelang.compile(func, target="npuir")
    compiled(A, B)

    assert_close(B.cpu(), ref, dtype=dtype, rtol=1e-3, atol=1e-3)
