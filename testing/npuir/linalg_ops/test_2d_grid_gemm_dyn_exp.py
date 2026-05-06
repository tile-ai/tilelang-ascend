# Copyright (c) Huawei Technologies Co., Ltd. 2025.
import torch
import pytest
import tilelang
import tilelang.language as T

from testcommon import assert_close, gen_tensor

pytestmark = [
    pytest.mark.op("gemm_2d_grid_dynamic"),
    pytest.mark.mode("Expert"),
]

SHAPES = [(1024, 512, 2048), (512, 1024, 512), (512, 512, 1280)]


@tilelang.jit(target="npuir")
def matmul_dyn_func_exp(block_M, block_N, K_L1, dtype="float16", accum_dtype="float32"):
    M = T.symbolic("M")
    N = T.symbolic("N")
    K = T.symbolic("K")

    a_shape = (M, K)
    b_shape = (K, N)
    c_shape = (M, N)

    @T.prim_func
    def matmul_exp_kernel(
        A: T.Tensor(a_shape, dtype),
        B: T.Tensor(b_shape, dtype),
        C: T.Tensor(c_shape, accum_dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), is_npu=True) as (
            idy,
            idx,
            _,
        ):
            with T.Scope("Cube"):
                bx = idx * block_M
                by = idy * block_N
                A_BUF = T.alloc_L1([block_M, K_L1], dtype)
                B_BUF = T.alloc_L1([K_L1, block_N], dtype)
                C_BUF = T.alloc_L0C([block_M, block_N], accum_dtype)
                remain_M = T.min(M - bx, block_M)
                remain_N = T.min(N - by, block_N)
                for i in T.serial(T.ceildiv(K, K_L1)):
                    remain_K = T.min(K - i * K_L1, K_L1)
                    T.load_nd2nz(A[bx, i * K_L1], A_BUF, [remain_M, remain_K])
                    T.load_nd2nz(B[i * K_L1, by], B_BUF, [remain_K, remain_N])
                    if i == 0:
                        T.gemm(
                            A_BUF,
                            B_BUF,
                            C_BUF,
                            initC=True,
                            b_transpose=False,
                            size=[remain_M, remain_K, remain_N],
                        )
                    else:
                        T.gemm(
                            A_BUF,
                            B_BUF,
                            C_BUF,
                            initC=False,
                            b_transpose=False,
                            size=[remain_M, remain_K, remain_N],
                        )
                    T.store_fixpipe(
                        C_BUF, C[bx, by], size=[remain_M, remain_N], enable_nz2nd=True
                    )

    return matmul_exp_kernel


@pytest.mark.parametrize("M,N,K", SHAPES)
def test_matmul_dyn_exp(M, N, K):
    block_M = 128
    block_N = 256
    block_K = 32
    dtype = "float16"
    accum_dtype = "float32"
    A = gen_tensor((M, K), dtype, kind="randn")
    B = gen_tensor((K, N), dtype, kind="randn")
    C = gen_tensor((M, N), accum_dtype, kind="zeros")

    kernel = matmul_dyn_func_exp(block_M, block_N, block_K, dtype, accum_dtype)
    kernel(A, B, C)
    ref_C = torch.matmul(A.cpu(), B.cpu()).to(torch.float32)
    assert_close(C.cpu(), ref_C.cpu(), rtol=1e-2, atol=1e-2)
