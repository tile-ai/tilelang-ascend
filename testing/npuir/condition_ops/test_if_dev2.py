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
def if_have_yield(M, block_N, dtype="float16", indexType="int32"):
    block_size = 1
    N = T.symbolic("N")

    @T.prim_func
    def if_have_yield_(
        Input: T.Tensor((M, N), dtype),
        Index: T.Tensor((M), indexType),
        Output: T.Tensor((1, N), dtype),
    ):
        with T.Kernel(block_size, is_npu=True) as (cid, _):
            src = T.alloc_shared((1, block_N), dtype)
            dst = T.alloc_shared((1, block_N), dtype)
            idx = T.alloc_shared((M), indexType)
            T.copy(Index, idx)
            n_num = T.ceildiv(N, block_N)

            for idx_n in T.serial(n_num):
                value_zero = 0
                T.npuir_brc(value_zero, dst)
                offset_n = idx_n * block_N
                remain_n = T.min(N - offset_n, block_N)
                for i in T.serial(M):
                    if idx[i] == 1:
                        T.copy(
                            Input[i : i + 1, offset_n : offset_n + remain_n],
                            src[0, 0:remain_n],
                        )
                        T.npuir_add(src, dst, dst)
                T.copy(dst[0, 0:remain_n], Output[0:1, offset_n : offset_n + remain_n])

    return if_have_yield_


def if_have_yield_torch(
    Input: torch.Tensor, Index: torch.Tensor, block_N: int
) -> torch.Tensor:
    M, N = Input.shape
    Output = torch.zeros((1, N), dtype=Input.dtype, device=Input.device)
    n_num = (N + block_N - 1) // block_N
    for idx_n in range(n_num):
        offset_n = idx_n * block_N
        remain_n = min(N - offset_n, block_N)
        dst = torch.zeros(remain_n, dtype=Input.dtype, device=Input.device)
        for i in range(M):
            if Index[i].item() == 1:
                src = Input[i, offset_n : offset_n + remain_n]
                dst = dst + src

        Output[0, offset_n : offset_n + remain_n] = dst

    return Output


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("N", N_CASES)
def test_if_have_yield_dev(dtype, N):
    kernel = if_have_yield(M, BLOCK_N, dtype)
    input_tensor = gen_tensor((M, N), dtype, kind="randn")
    index_tensor = gen_tensor((M,), "int32", kind="randint", low=0, high=2)
    output_tensor = gen_tensor((1, N), dtype, kind="randn")

    output_ref = if_have_yield_torch(input_tensor, index_tensor, BLOCK_N)
    kernel(input_tensor, index_tensor, output_tensor)

    assert_close(
        output_tensor.cpu(), output_ref.cpu(), dtype=dtype, rtol=1e-2, atol=1e-2
    )
