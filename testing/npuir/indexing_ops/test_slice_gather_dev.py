import pytest
import torch

import tilelang
import tilelang.language as T

from testcommon import assert_close, gen_tensor


pytestmark = [
    pytest.mark.op("gather"),
    pytest.mark.mode("Developer"),
]

GATHER_CASES = [(260, 260, 32, 32, 1)]
DTYPES = ["float16"]

torch.npu.set_device(11)


def vec_gather(M, N, block_M, block_N, dim, dtype="float16"):
    itype = "int32"
    m_num = T.ceildiv(M, block_M)
    n_num = T.ceildiv(N, block_N)

    @T.prim_func
    def sliceGatherKernel(
        A: T.Tensor((M, N), dtype),
        B: T.Tensor((M, N), itype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (
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

            valid_m = T.min(block_M, M - bx)
            valid_n = T.min(block_N, N - by)

            T.copy(A[bx : bx + valid_m, by : by + valid_n], A_VEC)
            T.copy(B[bx : bx + valid_m, by : by + valid_n], B_VEC)
            T.clear(C_VEC)
            T.gather(
                A_VEC[:valid_m, :valid_n],
                C_VEC[:valid_m, :valid_n],
                B_VEC[:valid_m, :valid_n],
            )
            T.copy(C_VEC[:valid_m, :valid_n], C[bx : bx + valid_m, by : by + valid_n])

    return sliceGatherKernel


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("M, N, block_M, block_N, dim", GATHER_CASES)
def test_vec_gather(dtype, M, N, block_M, block_N, dim):
    compile_kernel = vec_gather(M, N, block_M, block_N, dim=dim, dtype=dtype)
    kernel = tilelang.compile(compile_kernel, target="npuir")

    a = gen_tensor((M, N), dtype, kind="randn")

    b = torch.empty((M, N), dtype=torch.int32, device=a.device)
    for bx in range(0, M, block_M):
        for by in range(0, N, block_N):
            valid_m = min(block_M, M - bx)
            valid_n = min(block_N, N - by)
            b[bx : bx + valid_m, by : by + valid_n] = torch.randint(
                low=0,
                high=valid_n,
                size=(valid_m, valid_n),
                dtype=torch.int32,
                device=a.device,
            )

    c = gen_tensor((M, N), dtype, kind="zeros")

    ref_c = torch.zeros_like(a)

    for bx in range(0, M, block_M):
        for by in range(0, N, block_N):
            valid_m = min(block_M, M - bx)
            valid_n = min(block_N, N - by)

            a_tile = a[bx : bx + valid_m, by : by + valid_n]
            b_tile = b[bx : bx + valid_m, by : by + valid_n]

            ref_c[bx : bx + valid_m, by : by + valid_n] = torch.gather(
                a_tile, dim=1, index=b_tile
            )

    kernel(a, b, c)

    assert_close(c.cpu(), ref_c.cpu(), dtype=dtype, rtol=1e-2, atol=1e-2)
