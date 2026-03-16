# Copyright (c) Huawei Technologies Co., Ltd. 2025.
import pytest
import torch
import torch_npu  # noqa: F401

import tilelang
import tilelang.language as T

from testcommon import assert_close, gen_tensor

pytestmark = [
    pytest.mark.op("cumsum"),
    pytest.mark.mode("Expert"),
]

DTYPES = ["float16"]
ACCUM_DTYPES = ["float16"]


def cumsum_kernel(M, N, dim, reverse, dtype, accum_dtype):
    BLOCK_SIZE = 1

    @T.prim_func
    def cumsumExp(
        src: T.Tensor((M, N), dtype),
        dst: T.Tensor((M, N), accum_dtype),
    ):
        with T.Kernel(BLOCK_SIZE, is_npu=True) as (cid, _):
            src_ub = T.alloc_ub((M, N), dtype)
            dst_ub = T.alloc_ub((M, N), accum_dtype)
            T.copy(src, src_ub)
            T.cumsum(src_ub, dst_ub, dim=dim, reverse=reverse)
            T.copy(dst_ub, dst)

    return cumsumExp


@pytest.mark.parametrize("dtype,accum_dtype", list(zip(DTYPES, ACCUM_DTYPES)))
def test_cumsum_exp(dtype, accum_dtype):
    M, N, dim, reverse = 4, 4, 0, False
    src = gen_tensor((M, N), dtype, kind="randn")
    dst = gen_tensor((M, N), accum_dtype, kind="zeros")
    ref = torch.cumsum(src.cpu(), dim=dim)

    func = cumsum_kernel(M=M, N=N, dim=dim, reverse=reverse, dtype=dtype, accum_dtype=accum_dtype)
    compiled = tilelang.compile(func, target="npuir")
    compiled(src, dst)

    assert_close(dst.cpu(), ref, dtype=accum_dtype, rtol=1e-3, atol=1e-3)
