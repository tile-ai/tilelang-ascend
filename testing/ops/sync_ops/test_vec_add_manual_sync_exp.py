# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
import pytest
import torch
import torch_npu  # noqa: F401

import tilelang
import tilelang.language as T

from testcommon import assert_close, gen_tensor

pytestmark = [
    pytest.mark.op("sync"),
    pytest.mark.mode("Expert"),
]

DTYPES = ["float16"]


def vec_add(M, N, K, block_M, block_N, dtype="float16"):
    m_num = M // block_M
    n_num = N // block_N

    @T.prim_func
    def vecAddManualSyncExp(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, _):
            bx_ = cid // n_num
            bx = bx_ * block_M
            by_ = cid % n_num
            by = by_ * block_N
            A_VEC = T.alloc_ub((block_M, block_N), dtype)
            B_VEC = T.alloc_ub((block_M, block_N), dtype)
            C_VEC = T.alloc_ub((block_M, block_N), dtype)
            with T.rs("PIPE_MTE2"):
                T.copy(A[bx, by], A_VEC)
                T.copy(B[bx, by], B_VEC)
                T.set_flag("PIPE_V", 0)
            with T.rs("PIPE_V"):
                T.wait_flag("PIPE_MTE2", 0)
                T.npuir_add(A_VEC, B_VEC, C_VEC)
                T.set_flag("PIPE_MTE3", 0)
            with T.rs("PIPE_MTE3"):
                T.wait_flag("PIPE_V", 0)
                T.copy(C_VEC, C[bx, by])

    return vecAddManualSyncExp


@pytest.mark.parametrize("dtype", DTYPES)
def test_vec_add_manual_sync(dtype):
    M, N, K = 1024, 1024, 1024
    A = gen_tensor((M, K), dtype, kind="randn")
    B = gen_tensor((K, N), dtype, kind="randn")
    C = gen_tensor((M, N), dtype, kind="randn")
    ref = torch.add(A.cpu(), B.cpu())

    func = vec_add(M=M, N=N, K=K, block_M=128, block_N=256, dtype=dtype)
    compiled = tilelang.compile(func, target="npuir")
    compiled(A, B, C)

    assert_close(C.cpu(), ref, dtype=dtype, rtol=1e-2, atol=1e-2)
