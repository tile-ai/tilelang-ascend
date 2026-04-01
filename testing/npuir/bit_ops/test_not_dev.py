import pytest
import torch

import tilelang
import tilelang.language as T

from testcommon import assert_close, gen_tensor


pytestmark = [
    pytest.mark.op("not"),
    pytest.mark.mode("Developer"),
]

GATHER_CASES = [(16, 16)]
DTYPES = ["float16"]


def vec_not(block_M, block_N, dtype="float16"):
    block_size = 1

    @T.prim_func
    def NotKernel(
        A: T.Tensor((block_M, block_N), dtype), B: T.Tensor((block_M, block_N), dtype)
    ):
        with T.Kernel(block_size, is_npu=True) as (cid, _):
            A_VEC = T.alloc_shared((block_M, block_N), dtype)
            B_VEC = T.alloc_shared((block_M, block_N), dtype)

            T.copy(A, A_VEC)
            T.npuir_not(A_VEC, B_VEC)
            T.copy(B_VEC, B)

    return NotKernel


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("M, N", GATHER_CASES)
def test_vec_not(dtype, M, N):
    compile_kernel = vec_not(M, N, dtype=dtype)
    kernel = tilelang.compile(compile_kernel, target="npuir")

    a = gen_tensor((M, N), dtype, kind="randn")
    b = gen_tensor((M, N), dtype, kind="zeros")
    kernel(a, b)

    ref = (~a.view(torch.int16)).view(torch.float16)
    assert_close(b.cpu(), ref.cpu(), dtype=dtype, rtol=1e-2, atol=1e-2)
