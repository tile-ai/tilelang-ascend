import pytest
import torch
import torch_npu  # noqa: F401

import tilelang
import tilelang.language as T

from testcommon import assert_close, gen_tensor


MATMUL_CASES = [
    (1024, 512, 2048),
    (512, 1024, 512),
]
MINICV_CASES = [(1024, 1024, 512, 128, 256)]


@tilelang.jit(out_idx=[-1], target="npuir")
def matmul(block_M, block_N, K_L1, dtype="float16", accum_dtype="float32"):
    M = T.symbolic("M")
    N = T.symbolic("N")
    K = T.symbolic("K")

    @T.prim_func
    def outIdxMatmul(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(M, block_M) * T.ceildiv(N, block_N), is_npu=True) as (cid, _):
            with T.Scope("Cube"):
                bx = cid // T.ceildiv(N, block_N) * block_M
                by = cid % T.ceildiv(N, block_N) * block_N
                A_BUF = T.alloc_L1([block_M, K_L1], dtype)
                B_BUF = T.alloc_L1([K_L1, block_N], dtype)
                C_BUF = T.alloc_L0C([block_M, block_N], accum_dtype)

                remain_M = T.min(M - bx, block_M)
                remain_N = T.min(N - by, block_N)

                for i in T.serial(T.ceildiv(K, K_L1)):
                    remain_K = T.min(K - i * K_L1, K_L1)
                    T.npuir_load_nd2nz(A[bx, i * K_L1], A_BUF, [remain_M, remain_K])
                    T.npuir_load_nd2nz(B[i * K_L1, by], B_BUF, [remain_K, remain_N])

                    if i == 0:
                        T.npuir_dot(
                            A_BUF,
                            B_BUF,
                            C_BUF,
                            initC=True,
                            b_transpose=False,
                            size=[remain_M, remain_K, remain_N],
                        )
                    else:
                        T.npuir_dot(
                            A_BUF,
                            B_BUF,
                            C_BUF,
                            initC=False,
                            b_transpose=False,
                            size=[remain_M, remain_K, remain_N],
                        )

                    T.npuir_store_fixpipe(
                        C_BUF,
                        C[bx, by],
                        size=[remain_M, remain_N],
                        enable_nz2nd=True,
                    )

    return outIdxMatmul


@tilelang.jit(out_idx=[-2, -1], target="npuir")
def minicv(M, N, K, block_M, block_N, dtype="float16", inner_dtype="float32"):
    m_num = M // block_M
    n_num = N // block_N

    @T.prim_func
    def outIdxMinicv(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), inner_dtype),
        D: T.Tensor((M, N), inner_dtype),
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, _):
            blockx = cid // n_num
            bx = blockx * block_M
            blocky = cid % n_num
            by = blocky * block_N

            A_BUF = T.alloc_shared((block_M, K), dtype)
            B_BUF = T.alloc_shared((K, block_N), dtype)
            C_BUF = T.alloc_fragment((block_M, block_N), inner_dtype)
            D_BUF = T.alloc_fragment((block_M, block_N), inner_dtype)

            T.copy(A[bx:bx + block_M, 0:K], A_BUF)
            T.copy(B[0:K, by:by + block_N], B_BUF)

            T.gemm(A_BUF, B_BUF, C_BUF, [block_M, K, block_N], initC=True)
            T.vexp(C_BUF, D_BUF)

            T.copy(C_BUF, C[bx:bx + block_M, by:by + block_N])
            T.copy(D_BUF, D[bx:bx + block_M, by:by + block_N])

    return outIdxMinicv


@pytest.mark.op("matmul")
@pytest.mark.mode("Expert")
@pytest.mark.parametrize("M, N, K", MATMUL_CASES)
def test_mat_mul(M, N, K):
    func = matmul(128, 256, 16)
    a = gen_tensor((M, K), "float16", kind="randn")
    b = gen_tensor((K, N), "float16", kind="randn")

    c = func(a, b)
    ref_c = a @ b

    assert_close(c.cpu(), ref_c.cpu(), dtype="float16", rtol=1e-2, atol=1e-2)


@pytest.mark.op("exp")
@pytest.mark.mode("Developer")
@pytest.mark.parametrize("M, N, K, block_M, block_N", MINICV_CASES)
def test_minicv(monkeypatch, M, N, K, block_M, block_N):
    monkeypatch.setenv("TILELANG_ASCEND_WORKSPACE_SIZE", str(M * N))
    func = minicv(M, N, K, block_M, block_N)

    v1 = gen_tensor((M, K), "float16", kind="randn")
    v2 = gen_tensor((K, N), "float16", kind="randn")

    y_ref = torch.exp(v1.to(torch.float32) @ v2.to(torch.float32))
    _, v4 = func(v1, v2)

    assert_close(v4.cpu(), y_ref.cpu(), dtype="float32", rtol=1e-2, atol=1e-2)
