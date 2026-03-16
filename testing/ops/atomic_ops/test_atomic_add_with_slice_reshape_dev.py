import pytest
import torch_npu  # noqa: F401

import tilelang
import tilelang.language as T

from testcommon import assert_close, gen_tensor

pytestmark = [
    pytest.mark.op("atomic_add"),
    pytest.mark.mode("Developer"),
]

DTYPES = ["float32"]
ATOMIC_ADD_4D_CASES = [(2, 64, 4, 128)]


def simple_4d_atomic_add_kernel(B, S, H, D, dtype="float32"):
    @T.prim_func
    def atomic_add_4d(
        A: T.Tensor((B, S, H, D), dtype),
        B_tensor: T.Tensor((B, S, H, D), dtype),
        shape_B: T.int32,
        shape_H: T.int32,
    ):
        with T.Kernel(B * H, is_npu=True) as (cid, _):
            b_idx = cid // shape_H
            h_idx = cid % shape_H
            tile = T.alloc_shared((S, D), dtype)
            T.copy(A[b_idx, 0:S, h_idx, 0:D], tile)
            T.npuir_atomic_add(B_tensor[b_idx, 0:S, h_idx, 0:D], tile)

    return atomic_add_4d


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("B, S, H, D", ATOMIC_ADD_4D_CASES)
def test_atomic_add_with_slice_reshape_dev(dtype, B, S, H, D):
    A = gen_tensor((B, S, H, D), dtype, kind="randn")
    B_tensor = gen_tensor((B, S, H, D), dtype, kind="randn")
    expected = A + B_tensor

    func = simple_4d_atomic_add_kernel(B, S, H, D, dtype=dtype)
    compiled_kernel = tilelang.compile(func, target="npuir")
    compiled_kernel(A, B_tensor, B, H)

    assert_close(B_tensor.cpu(), expected.cpu(), dtype=dtype, rtol=1e-5, atol=1e-8)
