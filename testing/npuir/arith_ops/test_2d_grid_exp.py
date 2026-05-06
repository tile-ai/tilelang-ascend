# Copyright (c) Huawei Technologies Co., Ltd. 2025.
import tilelang
import tilelang.language as T
import pytest

from testcommon import assert_close, gen_tensor

pytestmark = [
    pytest.mark.op("2d_grid_exp"),
    pytest.mark.mode("Expert"),
]

DTYPES = ["float16", "float32"]


@tilelang.jit(target="npuir")
def grid_2d_demo_exp(M, N, block_M, block_N, dtype="float16"):
    @T.prim_func
    def grid_2d_exp(
        A: T.Tensor((M, N), dtype),
        B: T.Tensor((M, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), is_npu=True) as (
            bx,
            by,
            _,
        ):
            idx_x = bx * block_N
            idx_y = by * block_M

            A_VEC = T.alloc_ub([block_M, block_N], dtype)
            B_VEC = T.alloc_ub([block_M, block_N], dtype)
            C_VEC = T.alloc_ub([block_M, block_N], dtype)

            T.copy(A[idx_y : idx_y + block_M, idx_x : idx_x + block_N], A_VEC)
            T.copy(B[idx_y : idx_y + block_M, idx_x : idx_x + block_N], B_VEC)
            T.npuir_add(A_VEC, B_VEC, C_VEC)
            T.copy(C_VEC, C[idx_y : idx_y + block_M, idx_x : idx_x + block_N])

    return grid_2d_exp


@pytest.mark.parametrize("dtype", DTYPES)
def test_grid_2d_exp(dtype):
    M, N = 128, 128
    block_M, block_N = 32, 32
    A = gen_tensor((M, N), dtype=dtype, kind="randn")
    B = gen_tensor((M, N), dtype=dtype, kind="randn")
    C = gen_tensor((M, N), dtype=dtype, kind="zeros")
    ref_C = A + B

    grid_2d_exp = grid_2d_demo_exp(M, N, block_M, block_N, dtype=dtype)
    grid_2d_exp(A, B, C)
    assert_close(C.cpu(), ref_C.cpu(), atol=1e-2, rtol=1e-2)
