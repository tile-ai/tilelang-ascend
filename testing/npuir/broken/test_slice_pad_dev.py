# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
import pytest
import torch
import torch_npu  # noqa: F401

import tilelang
import tilelang.language as T

from testcommon import assert_close, gen_tensor

pytestmark = [
    pytest.mark.op("pad"),
    pytest.mark.mode("Developer"),
]

DTYPES = ["float16"]


def vec_pad(block_M, block_N, dtype="float16"):
    BLOCK_SIZE = 1

    @T.prim_func
    def slicePadDev(
        A: T.Tensor((block_M, block_N), dtype),
        C: T.Tensor((2 * block_M, block_N), dtype),
    ):
        with T.Kernel(BLOCK_SIZE, is_npu=True) as (cid, _):
            A_VEC = T.alloc_shared((block_M, block_N), dtype)
            C_VEC = T.alloc_shared((2 * block_M, block_N), dtype)
            T.copy(A, A_VEC)
            T.npuir_pad(
                A_VEC[:block_M, :block_N],
                C_VEC,
                T.float16(0),
                [block_M / 2, 0],
                [block_M / 2, 0],
            )
            T.copy(C_VEC, C)

    return slicePadDev


@pytest.mark.parametrize("dtype", DTYPES)
def test_vec_pad_dev(dtype):
    M, N = 32, 32
    torch.manual_seed(42)
    A = gen_tensor((M, N), dtype, kind="randn")
    C = gen_tensor((2 * M, N), dtype, kind="zeros")
    ref_C = torch.nn.functional.pad(A.cpu(), (0, 0, 16, 16), mode="constant", value=0)

    func = vec_pad(32, 32)
    compiled = tilelang.compile(func, target="npuir")
    compiled(A, C)

    assert_close(C.cpu(), ref_C, dtype=dtype, rtol=1e-2, atol=1e-2)
