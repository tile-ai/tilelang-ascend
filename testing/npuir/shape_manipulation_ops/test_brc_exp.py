# Copyright (c) Huawei Technologies Co., Ltd. 2025.
import pytest
import torch
import torch_npu  # noqa: F401

import tilelang
import tilelang.language as T

from testcommon import assert_close, gen_tensor

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
    def brcExpVec(A: T.Tensor((M, K), dtype)):
        with T.Kernel(BLOCK_SIZE, is_npu=True) as (cid, _):
            A_VEC = T.alloc_ub((block_M, block_N), dtype)
            for i in T.serial(T.ceildiv(m_num * n_num, BLOCK_SIZE)):
                block_id = i * BLOCK_SIZE + cid
                if block_id < m_num * n_num:
                    block_id_m = block_id // n_num
                    block_id_n = block_id % n_num
                    bx = block_id_m * block_M
                    by = block_id_n * block_N
                    T.copy(A[bx, by], A_VEC)
                    brc_value = 3
                    T.npuir_brc(brc_value, A_VEC)
                    T.copy(A_VEC, A[bx, by])

    return brcExpVec


@pytest.mark.parametrize("dtype", DTYPES)
def test_vec_brc_exp(dtype):
    M, N, K = 512, 512, 512
    shape = (M, K)
    a = gen_tensor(shape, dtype, kind="randn")
    ref = torch.full(shape, 3.0, dtype=torch.float16, device="npu")

    func = vec_brc_exp(M=M, N=N, K=K, block_M=128, block_N=256)
    compiled = tilelang.compile(func, target="npuir")
    compiled(a)

    assert_close(a.cpu(), ref.cpu(), dtype=dtype, rtol=1e-2, atol=1e-2)
