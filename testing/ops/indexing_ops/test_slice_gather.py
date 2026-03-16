import pytest
import torch
import torch_npu  # noqa: F401

import tilelang
import tilelang.language as T

from testcommon import assert_close, gen_tensor


pytestmark = [
    pytest.mark.op("gather"),
    pytest.mark.mode("Expert"),
]

GATHER_CASES = [(32, 32, 1)]
DTYPES = ["float16"]


def vec_gather(block_M, block_N, dim, dtype="float16"):
    block_size = 1
    itype = "int32"

    @T.prim_func
    def sliceGatherKernel(
        A: T.Tensor((block_M, block_N), dtype),
        B: T.Tensor((block_M, block_N), itype),
        C: T.Tensor((block_M, block_N), dtype),
    ):
        with T.Kernel(block_size, is_npu=True) as (cid, _):
            A_VEC = T.alloc_ub((block_M, block_N), dtype)
            index_VEC = T.alloc_ub((block_M, block_N), itype)
            C_VEC = T.alloc_ub((block_M, block_N), dtype)

            T.copy(A, A_VEC)
            T.copy(B, index_VEC)
            T.npuir_gather(
                A_VEC[:block_M, :block_N],
                C_VEC[:block_M, :block_N],
                index_VEC,
            )
            T.copy(C_VEC, C)

    return sliceGatherKernel


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("M, N, dim", GATHER_CASES)
def test_vec_gather(dtype, M, N, dim):
    compile_kernel = vec_gather(M, N, dim=dim, dtype=dtype)
    kernel = tilelang.compile(compile_kernel, target="npuir")

    a = gen_tensor((M, N), dtype, kind="randn")
    b = gen_tensor((M, N), "int32", kind="randint", low=0, high=N)
    c = gen_tensor((M, N), dtype, kind="zeros")

    ref_c = torch.gather(a, dim=dim, index=b)
    kernel(a, b, c)

    assert_close(c.cpu(), ref_c.cpu(), dtype=dtype, rtol=1e-2, atol=1e-2)
