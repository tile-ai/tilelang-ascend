import torch
import tilelang
import tilelang.language as T

import pytest
from testcommon import assert_close

pytestmark = [pytest.mark.op("vadd_1x1"), pytest.mark.mode("Developer")]

DTYPES = ["float16", "float32"]


def vadd_1x1_kernel(M, N, block_M, block_N, dtype="float16"):
    grid_M = (M + block_M - 1) // block_M
    grid_N = (N + block_N - 1) // block_N

    @T.prim_func
    def vadd_1x1(
        A: T.Tensor((1, 1), dtype),
        B: T.Tensor((M, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(grid_M * grid_N, is_npu=True) as (cid, _):
            blockx = cid % T.ceildiv(N, block_N)
            bx = blockx * block_M
            blocky = cid // T.ceildiv(N, block_N)
            by = blocky * block_N
            A_VEC = T.alloc_shared([1, 1], dtype)
            B_VEC = T.alloc_shared([block_M, block_N], dtype)
            C_VEC = T.alloc_shared([block_M, block_N], dtype)

            T.copy(A[0, 0], A_VEC)
            T.copy(B[bx : bx + block_M, by : by + block_N], B_VEC)
            T.npuir_add(A_VEC, B_VEC, C_VEC)
            T.copy(C_VEC[:block_M, :block_N], C[bx : bx + block_M, by : by + block_N])

    return vadd_1x1


@pytest.mark.parametrize("dtype", DTYPES)
def test_vadd_1x1(dtype):
    M, N = 128, 128
    func = vadd_1x1_kernel(128, 128, 32, 32, dtype)
    compiled_kernel = tilelang.compile(func, target="npuir")
    dType = eval("torch." + dtype)
    a = torch.ones((1, 1), dtype=dType).npu()
    b = torch.ones((M, N), dtype=dType).npu()
    c = torch.empty((M, N), dtype=dType).npu()
    compiled_kernel(a, b, c)
    ref_c = a + b
    assert_close(c, ref_c, rtol=1e-2, atol=1e-2)
