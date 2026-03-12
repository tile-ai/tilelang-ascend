# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
import pytest
import torch
import torch_npu  # noqa: F401

import tilelang
import tilelang.language as T

from testcommon import assert_close, gen_tensor

pytestmark = [
    pytest.mark.op("gemm"),
    pytest.mark.mode("Expert"),
]


def matmul(M, N, K, block_M, block_N, block_K, in_dtype, out_dtype, num_stages):
    m_num = M // block_M
    n_num = N // block_N

    @T.prim_func
    def sliceGemmExp(
        A: T.Tensor((M, K), in_dtype),
        B: T.Tensor((K, N), in_dtype),
        C: T.Tensor((M, N), out_dtype),
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, _):
            with T.Scope("Cube"):
                bx = cid // n_num * block_M
                by = cid % n_num * block_N

                A_buf = T.alloc_L1((block_M, block_K), in_dtype)
                B_buf = T.alloc_L1((block_K, block_N), in_dtype)
                C_buf = T.alloc_L0C((block_M, block_N), out_dtype)

                for ki in T.serial(T.ceildiv(K, block_K)):
                    T.load_nd2nz(A[bx, ki * block_K], A_buf, [block_M, block_K])
                    T.load_nd2nz(B[ki * block_K, by], B_buf, [block_K, block_N])

                    T.gemm(
                        A_buf,
                        B_buf,
                        C_buf,
                        initC=ki == 0,
                        b_transpose=False,
                        size=[block_M, block_K, block_N],
                    )

                T.store_fixpipe(
                    C_buf,
                    C[bx, by],
                    size=[block_M, block_N],
                    enable_nz2nd=True,
                )

    return sliceGemmExp


DTYPE_CASES = [("float16", "float32")]


@pytest.mark.parametrize("in_dtype,out_dtype", DTYPE_CASES)
def test_matmul_exp(in_dtype, out_dtype):
    M, K, N = 256, 512, 256
    A = gen_tensor((M, K), in_dtype, kind="randn")
    B = gen_tensor((K, N), in_dtype, kind="randn")
    C = gen_tensor((M, N), out_dtype, kind="zeros")
    ref_C = torch.matmul(A.cpu(), B.cpu()).to(torch.float32)

    program = matmul(
        M=M,
        N=N,
        K=K,
        block_M=32,
        block_K=32,
        block_N=32,
        in_dtype=in_dtype,
        out_dtype=out_dtype,
        num_stages=0,
    )
    compiled = tilelang.compile(program, target="npuir")
    compiled(A, B, C)

    assert_close(C.cpu(), ref_C, dtype=out_dtype, rtol=1e-2, atol=1e-2)
