# Copyright (c) Huawei Technologies Co., Ltd. 2025.
import pytest
import torch
import torch_npu  # noqa: F401

import tilelang
import tilelang.language as T

from testcommon import gen_tensor

pytestmark = [
    pytest.mark.op("brc"),
    pytest.mark.mode("Expert"),
]

DTYPES = ["float16"]


def vec_brc_exp(M, N, K, block_M, block_N):
    m_num = M // block_M
    n_num = N // block_N
    dtype = "float16"
    BLOCK_SIZE = 20

    @T.prim_func
    def brcFillDynExp(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((1, block_N), dtype),
        C: T.Tensor((M, block_N), dtype),
    ):
        with T.Kernel(BLOCK_SIZE, is_npu=True) as (cid, _):
            A_VEC = T.alloc_ub((block_M, block_N), dtype)
            B_VEC = T.alloc_ub((1, block_N), dtype)
            C_VEC = T.alloc_ub((M, block_N), dtype)
            T.copy(B, B_VEC)
            T.copy(C, C_VEC)
            for i in T.serial(T.ceildiv(m_num * n_num, BLOCK_SIZE)):
                block_id = i * BLOCK_SIZE + cid
                if block_id < m_num * n_num:
                    block_id_m = block_id // n_num
                    block_id_n = block_id % n_num
                    bx = block_id_m * block_M
                    by = block_id_n * block_N
                    T.npuir_brc(C_VEC[0:1, :], A_VEC)
                    T.copy(A_VEC, A[bx, by])

    return brcFillDynExp


@pytest.mark.parametrize("dtype", DTYPES)
def test_vec_brc_exp(dtype):
    torch.manual_seed(88888888)
    M, N, K = 512, 512, 512
    block_M, block_N = 32, 32
    a = gen_tensor((M, K), dtype, kind="randn")
    b = gen_tensor((1, block_N), dtype, kind="randn")
    c = gen_tensor((M, block_N), dtype, kind="randn")

    func = vec_brc_exp(M=M, N=N, K=K, block_M=block_M, block_N=block_N)
    compiled = tilelang.compile(func, target="npuir")
    compiled(a, b, c)
