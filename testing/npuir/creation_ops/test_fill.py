import pytest
import torch
import torch_npu  # noqa: F401

import tilelang
import tilelang.language as T

from testcommon import assert_close, gen_tensor


pytestmark = [
    pytest.mark.op("fill"),
    pytest.mark.mode("Expert"),
]

DTYPES = ["float16"]
FILL_CASES = [(512, 512, 512, 128, 256)]


def vec_fill(M, N, K, block_M, block_N, dtype="float16"):
    m_num = M // block_M
    n_num = N // block_N
    block_size = 20

    @T.prim_func
    def vecFillCreation(
        A: T.Tensor((M, K), dtype),
    ):
        with T.Kernel(block_size, is_npu=True) as (cid, _):
            A_VEC = T.alloc_ub((block_M, block_N), dtype)
            for i in T.serial(T.ceildiv(m_num * n_num, block_size)):
                block_id = i * block_size + cid
                if block_id < m_num * n_num:
                    block_id_m = block_id // n_num
                    block_id_n = block_id % n_num
                    bx = block_id_m * block_M
                    by = block_id_n * block_N
                    T.copy(A[bx, by], A_VEC)
                    fill_value = 3
                    T.npuir_fill(A_VEC, fill_value)
                    T.copy(A_VEC, A[bx, by])

    return vecFillCreation


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("M, N, K, block_M, block_N", FILL_CASES)
def test_fill(dtype, M, N, K, block_M, block_N):
    kernel = tilelang.compile(vec_fill(M, N, K, block_M, block_N, dtype), target="npuir")

    a = gen_tensor((M, K), dtype, kind="randn")
    ref_output = torch.full((M, K), fill_value=3, dtype=torch.float16, device=a.device)

    kernel(a)
    assert_close(a.cpu(), ref_output.cpu(), dtype=dtype, rtol=1e-2, atol=1e-2)
