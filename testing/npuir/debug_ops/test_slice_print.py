import pytest
import torch
import torch_npu  # noqa: F401

import tilelang
import tilelang.language as T

from testcommon import assert_close, gen_tensor


pytestmark = [
    pytest.mark.op("print"),
    pytest.mark.mode("Developer"),
]

DTYPES = ["float32"]
PRINT_CASES = [(256, 256, 32, 32)]


@tilelang.jit(target="npuir")
def vec_add_2d(block_M, block_N, dtype="float32"):
    M = T.symbolic("M")
    N = T.symbolic("N")

    @T.prim_func
    def vecAdd2D(
        A: T.Tensor((M, N), dtype),
        B: T.Tensor((M, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N) * T.ceildiv(M, block_M), is_npu=True) as (cid, _):
            blockx = cid % T.ceildiv(N, block_N)
            bx = blockx * block_M
            blocky = cid // T.ceildiv(N, block_N)
            by = blocky * block_N
            A_VEC = T.alloc_shared([block_M, block_N], dtype)
            B_VEC = T.alloc_shared([block_M, block_N], dtype)
            C_VEC = T.alloc_shared([block_M, block_N], dtype)

            t0 = M - bx
            tile_size_M = T.min(block_M, t0)

            t0 = N - by
            tile_size_N = T.min(block_N, t0)
            T.copy(A[bx:bx + tile_size_M, by:by + tile_size_N], A_VEC[:tile_size_M, :tile_size_N])
            T.copy(B[bx:bx + tile_size_M, by:by + tile_size_N], B_VEC[:tile_size_M, :tile_size_N])
            T.npuir_add(A_VEC, B_VEC, C_VEC)
            T.print(C_VEC[:4, :4])
            T.copy(C_VEC[:tile_size_M, :tile_size_N], C[bx:bx + tile_size_M, by:by + tile_size_N])

    return vecAdd2D


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("M, N, block_M, block_N", PRINT_CASES)
def test_vec_add_2d(dtype, M, N, block_M, block_N):
    a = gen_tensor((M, N), dtype, kind="ones")
    b = gen_tensor((M, N), dtype, kind="ones")
    c = gen_tensor((M, N), dtype, kind="zeros")
    expected = a + b

    func = vec_add_2d(block_M, block_N, dtype)
    func(a, b, c)

    assert_close(c.cpu(), expected.cpu(), dtype=dtype)
