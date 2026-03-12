# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
import pytest
import torch
import torch_npu  # noqa: F401

import tilelang
import tilelang.language as T

from testcommon import assert_close, gen_tensor

pytestmark = [
    pytest.mark.op("interleave"),
    pytest.mark.mode("Developer"),
]

DTYPES = ["float16"]


def interleave_tensors(*tensors, dim=0):
    stacked = torch.stack(tensors, dim=dim + 1)
    shape = list(stacked.shape)
    shape[dim] *= len(tensors)
    shape.pop(dim + 1)
    return stacked.view(shape)


def vec_interleave(block_M, block_N, dtype="float16"):
    BLOCK_SIZE = 1

    @T.prim_func
    def sliceInterleaveDev(
        A: T.Tensor((block_M, block_N), dtype),
        B: T.Tensor((block_M, block_N), dtype),
        C: T.Tensor((block_M, 2 * block_N), dtype),
    ):
        with T.Kernel(BLOCK_SIZE, is_npu=True) as (cid, _):
            A_VEC = T.alloc_shared((block_M, block_N), dtype)
            B_VEC = T.alloc_shared((block_M, block_N), dtype)
            C_VEC = T.alloc_shared((block_M, 2 * block_N), dtype)
            T.copy(A, A_VEC)
            T.copy(B, B_VEC)
            T.npuir_interleave(
                A_VEC[:block_M, :block_N],
                B_VEC[:block_M, :block_N],
                C_VEC[:block_M, : 2 * block_N],
            )
            T.copy(C_VEC, C)

    return sliceInterleaveDev


@pytest.mark.parametrize("dtype", DTYPES)
def test_vec_interleave_dev(dtype):
    M, N = 32, 32
    torch.manual_seed(42)
    A = gen_tensor((M, N), dtype, kind="randn")
    B = gen_tensor((M, N), dtype, kind="randn")
    C = gen_tensor((M, 2 * N), dtype, kind="zeros")
    ref_C = interleave_tensors(A.cpu(), B.cpu(), dim=1)

    func = vec_interleave(32, 32)
    compiled = tilelang.compile(func, target="npuir")
    compiled(A, B, C)

    assert_close(C.cpu(), ref_C, dtype=dtype, rtol=1e-2, atol=1e-2)
