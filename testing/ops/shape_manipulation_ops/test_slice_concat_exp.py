# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
import pytest
import torch
import torch_npu  # noqa: F401

import tilelang
import tilelang.language as T

from testcommon import assert_close, gen_tensor

pytestmark = [
    pytest.mark.op("concat"),
    pytest.mark.mode("Expert"),
]

DTYPES = ["float16"]


def vec_concat(block_M, block_N, dim, dtype="float16"):
    BLOCK_SIZE = 1

    @T.prim_func
    def sliceConcatExp(
        A: T.Tensor((block_M, block_N), dtype),
        B: T.Tensor((block_M, block_N), dtype),
        C: T.Tensor((block_M, 2 * block_N), dtype),
    ):
        with T.Kernel(BLOCK_SIZE, is_npu=True) as (cid, _):
            A_VEC = T.alloc_ub((block_M, block_N), dtype)
            B_VEC = T.alloc_ub((block_M, block_N), dtype)
            C_VEC = T.alloc_ub((block_M, 2 * block_N), dtype)
            T.copy(A, A_VEC)
            T.copy(B, B_VEC)
            T.npuir_concat(
                A_VEC[:block_M, :block_N],
                B_VEC[:block_M, :block_N],
                C_VEC[:block_M, : 2 * block_N],
                dim,
            )
            T.copy(C_VEC, C)

    return sliceConcatExp


@pytest.mark.parametrize("dtype", DTYPES)
def test_vec_concat(dtype):
    M, N = 32, 32
    torch.manual_seed(42)
    A = gen_tensor((M, N), dtype, kind="randn")
    B = gen_tensor((M, N), dtype, kind="randn")
    C = gen_tensor((M, 2 * N), dtype, kind="zeros")
    ref_C = torch.cat((A.cpu(), B.cpu()), dim=1)

    func = vec_concat(32, 32, dim=1)
    compiled = tilelang.compile(func, target="npuir")
    compiled(A, B, C)

    assert_close(C.cpu(), ref_C, dtype=dtype, rtol=1e-2, atol=1e-2)
