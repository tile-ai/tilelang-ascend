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
    pytest.mark.mode("Developer"),
]


def matmul(M, N, K, block_M, block_N, block_K, in_dtype, out_dtype, num_stages):
    A_shape = (M, K)
    B_shape = (K, N)
    A_shared_shape = (block_M, block_K)
    B_shared_shape = (block_K, block_N)

    @T.prim_func
    def sliceGemmDev(
        A: T.Tensor(A_shape, in_dtype),
        B: T.Tensor(B_shape, in_dtype),
        C: T.Tensor((M, N), out_dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N) * T.ceildiv(M, block_M), is_npu=True) as (cid, _):
            blockx = cid % T.ceildiv(N, block_N)
            bx = blockx * block_M
            blocky = cid // T.ceildiv(N, block_N)
            by = blocky * block_N
            A_shared = T.alloc_shared(A_shared_shape, in_dtype)
            B_shared = T.alloc_shared(B_shared_shape, in_dtype)
            C_local = T.alloc_shared((block_M, block_N), out_dtype)
            value_zero = 0
            T.npuir_brc(value_zero, C_local)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                T.copy(A[by, k * block_K], A_shared, size=[block_M, block_K])
                T.copy(B[k * block_K, bx], B_shared, size=[block_K, block_N])
                T.gemm(
                    A_shared[:block_M, :block_K],
                    B_shared[:block_K, :block_N],
                    C_local[:block_M, :block_N],
                )
            T.copy(C_local, C[by, bx], size=[block_M, block_N])

    return sliceGemmDev


DTYPE_CASES = [("float16", "float32")]


@pytest.mark.parametrize("in_dtype,out_dtype", DTYPE_CASES)
def test_matmul(in_dtype, out_dtype):
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
