import pytest
import torch
import torch_npu  # noqa: F401

import tilelang
import tilelang.language as T

from testcommon import assert_close, gen_tensor


pytestmark = [
    pytest.mark.op("if"),
    pytest.mark.mode("Developer"),
]

DTYPES = ["float16"]
N_CASES = [64, 133, 397, 499]
M = 20
BLOCK_N = 32


@tilelang.jit(target="npuir")
def if_no_yield(M, block_N, dtype="float16", indexType="int32"):
    block_size = 1
    N = T.symbolic("N")

    @T.prim_func
    def if_no_yield_(
        Input: T.Tensor((M, N), dtype),
        Index: T.Tensor((M), indexType),
        Output: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(block_size, is_npu=True) as (cid, _):
            src = T.alloc_shared((1, block_N), dtype)
            dst = T.alloc_shared((1, block_N), dtype)
            idx = T.alloc_shared((M), indexType)
            T.copy(Index, idx)
            n_num = T.ceildiv(N, block_N)
            for i in T.serial(M):
                for idx_n in T.serial(n_num):
                    offset_n = idx_n * block_N
                    remain_n = T.min(N - offset_n, block_N)
                    if idx[i] == 1:
                        T.copy(Input[i:i + 1, offset_n:offset_n + remain_n], src)
                        T.npuir_add(src, src, dst)
                        T.copy(dst, Output[i:i + 1, offset_n:offset_n + remain_n])
                    else:
                        value_zero = 0
                        T.npuir_brc(value_zero, dst)
                        T.copy(dst, Output[i:i + 1, offset_n:offset_n + remain_n])

    return if_no_yield_


def if_no_yield_torch(Input: torch.Tensor, Index: torch.Tensor, block_N: int) -> torch.Tensor:
    M, N = Input.shape
    Output = torch.zeros_like(Input)
    for i in range(M):
        if Index[i].item() == 1:
            for idx_n in range(0, N, block_N):
                offset_n = idx_n
                remain_n = min(N - offset_n, block_N)
                src = Input[i, offset_n:offset_n + remain_n]
                dst = src + src
                Output[i, offset_n:offset_n + remain_n] = dst
        else:
            for idx_n in range(0, N, block_N):
                offset_n = idx_n
                remain_n = min(N - offset_n, block_N)
                Output[i, offset_n:offset_n + remain_n] = 0
    return Output


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("N", N_CASES)
def test_if_no_yield_dev(dtype, N):
    kernel = if_no_yield(M, BLOCK_N, dtype)
    input_tensor = gen_tensor((M, N), dtype, kind="randn")
    index_tensor = gen_tensor((M,), "int32", kind="randint", low=0, high=2)
    output_tensor = gen_tensor((M, N), dtype, kind="randn")

    output_ref = if_no_yield_torch(input_tensor, index_tensor, BLOCK_N)
    kernel(input_tensor, index_tensor, output_tensor)

    assert_close(output_tensor.cpu(), output_ref.cpu(), dtype=dtype, rtol=1e-2, atol=1e-2)
