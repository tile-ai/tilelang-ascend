# Copyright (c) Huawei Technologies Co., Ltd. 2025.
import torch
import pytest
import tilelang
import tilelang.language as T

from testcommon import assert_close, gen_tensor

pytestmark = [
    pytest.mark.op("gemm_2d_grid_dynamic"),
    pytest.mark.mode("Developer"),
]

TEST_SHAPES = [(1024, 512, 2048), (512, 1024, 512), (512, 512, 1280)]


def matmul_dyn_func_dev(
    block_M, block_N, block_K, dtype="float16", accum_dtype="float32"
):
    M = T.symbolic("M")
    N = T.symbolic("N")
    K = T.symbolic("K")

    a_shape = (M, K)
    b_shape = (K, N)
    c_shape = (M, N)

    @T.prim_func
    def matmul_dev_kernel(
        A: T.Tensor(a_shape, dtype),
        B: T.Tensor(b_shape, dtype),
        C: T.Tensor(c_shape, accum_dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), is_npu=True) as (
            bx,
            by,
            _,
        ):
            idx = bx * block_N
            idy = by * block_M
            A_buf = T.alloc_shared((block_M, block_K), dtype)
            B_buf = T.alloc_shared((block_K, block_N), dtype)
            C_buf = T.alloc_shared((block_M, block_N), accum_dtype)
            value_zero = 0
            T.npuir_brc(value_zero, C_buf)
            for k in T.Pipelined(T.ceildiv(K, block_K)):
                T.copy(A[idy, k * block_K], A_buf)
                T.copy(B[k * block_K, idx], B_buf)
                T.gemm(A_buf, B_buf, C_buf)
            T.copy(C_buf, C[idy, idx])

    return matmul_dev_kernel


@pytest.mark.parametrize("M,N,K", TEST_SHAPES)
def test_matmul_dyn_dev(M, N, K):
    block_M = 128
    block_N = 256
    block_K = 32
    dtype = "float16"
    accum_dtype = "float32"
    A = gen_tensor((M, K), dtype, kind="randn")
    B = gen_tensor((K, N), dtype, kind="randn")
    C = gen_tensor((M, N), accum_dtype, kind="zeros")

    program = matmul_dyn_func_dev(block_M, block_N, block_K, dtype, accum_dtype)
    kernel = tilelang.compile(program, target="npuir")
    kernel(A, B, C)
    ref_C = torch.matmul(A.cpu(), B.cpu()).to(torch.float32)
    assert_close(C.cpu(), ref_C, rtol=1e-2, atol=1e-2)
