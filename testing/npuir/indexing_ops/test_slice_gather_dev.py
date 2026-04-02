import pytest
import torch
import torch_npu  # noqa: F401

import tilelang
import tilelang.language as T

from testcommon import assert_close, gen_tensor


pytestmark = [
    pytest.mark.op("gather"),
    pytest.mark.mode("Developer"),
]

GATHER_CASES = [(256, 256, 32, 32, 1)]
DTYPES = ["float16"]


def vec_gather(M, N, block_M, block_N, dim, dtype="float16"):
    itype = "int32"
    n_num = N // block_N

    @T.prim_func
    def sliceGatherKernel(
        A: T.Tensor((M, N), dtype),
        B: T.Tensor((M, N), itype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(M, block_M) * T.ceildiv(N, block_N), is_npu=True) as (
            cid,
            _,
        ):
            block_y = cid % n_num
            block_x = cid // n_num
            bx = block_x * block_M
            by = block_y * block_N
            A_VEC = T.alloc_shared([block_M, block_N], dtype)
            B_VEC = T.alloc_shared([block_M, block_N], itype)
            C_VEC = T.alloc_shared([block_M, block_N], dtype)
            T.copy(A[bx : bx + block_M, by : by + block_N], A_VEC)
            T.copy(B[bx : bx + block_M, by : by + block_N], B_VEC)
            T.gather(A_VEC, C_VEC, B_VEC)
            T.copy(C_VEC, C[bx : bx + block_M, by : by + block_N])

    return sliceGatherKernel


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("M, N, block_M, block_N, dim", GATHER_CASES)
def test_vec_gather(dtype, M, N, block_M, block_N, dim):
    compile_kernel = vec_gather(M, N, block_M, block_N, dim=dim, dtype=dtype)
    kernel = tilelang.compile(compile_kernel, target="npuir")

    a = gen_tensor((M, N), dtype, kind="randn")
    b = gen_tensor((M, N), "int32", kind="randint", low=0, high=block_N)
    c = gen_tensor((M, N), dtype, kind="zeros")

    ref_c = torch.zeros_like(a)

    for bx in range(0, M, block_M):
        for by in range(0, N, block_N):
            a_tile = a[bx : bx + block_M, by : by + block_N]
            b_tile = b[bx : bx + block_M, by : by + block_N]
            ref_c[bx : bx + block_M, by : by + block_N] = torch.gather(
                a_tile, dim=1, index=b_tile
            )
    kernel(a, b, c)

    assert_close(c.cpu(), ref_c.cpu(), dtype=dtype, rtol=1e-2, atol=1e-2)
