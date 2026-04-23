# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
import pytest
import torch
import torch_npu  # noqa: F401

import tilelang
import tilelang.language as T

from testcommon import assert_close, gen_tensor

pytestmark = [
    pytest.mark.op("transpose"),
    pytest.mark.mode("Expert"),
]

DTYPES = ["float16"]


def vec_transpose_exp(block_M, block_N, dtype="float16"):
    BLOCK_SIZE = 1

    @T.prim_func
    def sliceTransposeExp(
        A: T.Tensor((block_M, block_N), dtype),
        C: T.Tensor((block_N, block_M), dtype),
    ):
        with T.Kernel(BLOCK_SIZE, is_npu=True) as (cid, _):
            A_VEC = T.alloc_ub((block_M, block_N), dtype)
            C_VEC = T.alloc_ub((block_N, block_M), dtype)
            T.copy(A, A_VEC)
            T.npuir_transpose(
                A_VEC[:block_M, :block_N],
                C_VEC[:block_N, :block_M],
                [1, 0],
            )
            T.copy(C_VEC, C)

    return sliceTransposeExp


@pytest.mark.parametrize("dtype", DTYPES)
def test_vec_transpose_exp(dtype):
    M, N = 32, 32
    torch.manual_seed(42)
    A = gen_tensor((M, N), dtype, kind="randn")
    C = gen_tensor((N, M), dtype, kind="zeros")
    ref_C = torch.transpose(A.cpu(), 0, 1)

    func = vec_transpose_exp(32, 32)
    compiled = tilelang.compile(func, target="npuir")
    compiled(A, C)

    assert_close(C.cpu(), ref_C, dtype=dtype, rtol=1e-2, atol=1e-2)
