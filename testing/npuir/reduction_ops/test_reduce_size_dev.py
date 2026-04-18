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


@tilelang.jit(target="npuir")
def slice_reduce(block_M, block_N, dtype="float16"):
    M = T.symbolic("M")
    N = T.symbolic("N")
    BLOCK_SIZE = 1

    @T.prim_func
    def sliceReduceSizeDev(
        Input: T.Tensor((M, N), dtype),
        Output: T.Tensor((1, N), dtype),
    ):
        with T.Kernel(BLOCK_SIZE, is_npu=True) as (cid, _):
            src = T.alloc_shared([block_M, block_N], dtype=dtype)
            dst = T.alloc_shared([1, block_N], dtype=dtype)
            for i in T.serial(T.ceildiv(N, block_N)):
                offset_n = i * block_N
                remain_n = T.min(block_N, N - offset_n)
                value_zero = 0
                T.npuir_brc(value_zero, dst)
                for j in T.serial(T.ceildiv(M, block_M)):
                    offset_m = j * block_M
                    remain_m = T.min(block_M, M - offset_m)
                    T.copy(
                        Input[
                            offset_m : offset_m + remain_m,
                            offset_n : offset_n + remain_n,
                        ],
                        src[0:remain_m, 0:remain_n],
                    )
                    T.npuir_reduce(
                        src,
                        dst,
                        dims=0,
                        reduce_mode="sum",
                        size=[remain_m, remain_n],
                        clear=False,
                    )
                T.copy(dst[0, 0:remain_n], Output[0:1, offset_n : offset_n + remain_n])

    return sliceReduceSizeDev


@pytest.mark.parametrize("dtype", DTYPES)
def test_slice_reduce_size_case1(dtype):
    kernel = slice_reduce(32, 32)
    M, N = 17, 256
    input_t = gen_tensor((M, N), dtype, kind="randn")
    output = gen_tensor((1, N), dtype, kind="randn")
    kernel(input_t, output)
    ref = torch.sum(input_t.cpu(), dim=0, keepdim=True)
    assert_close(output.cpu(), ref, dtype=dtype, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("dtype", DTYPES)
def test_slice_reduce_size_case2(dtype):
    kernel = slice_reduce(32, 32)
    M, N = 39, 466
    input_t = gen_tensor((M, N), dtype, kind="randn")
    output = gen_tensor((1, N), dtype, kind="randn")
    kernel(input_t, output)
    ref = torch.sum(input_t.cpu(), dim=0, keepdim=True)
    assert_close(output.cpu(), ref, dtype=dtype, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("dtype", DTYPES)
def test_slice_reduce_size_case3(dtype):
    kernel = slice_reduce(32, 32)
    M, N = 77, 283
    input_t = gen_tensor((M, N), dtype, kind="randn")
    output = gen_tensor((1, N), dtype, kind="randn")
    kernel(input_t, output)
    ref = torch.sum(input_t.cpu(), dim=0, keepdim=True)
    assert_close(output.cpu(), ref, dtype=dtype, rtol=1e-2, atol=1e-2)
