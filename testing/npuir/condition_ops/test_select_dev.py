import pytest
import torch
import torch_npu  # noqa: F401

import tilelang
import tilelang.language as T

from testcommon import assert_close, gen_tensor


pytestmark = [
    pytest.mark.op("select"),
    pytest.mark.mode("Developer"),
]

DTYPES = ["float16"]
SELECT_CASES = [(8, 32, 8)]


def select_kernel(M, N, block_M, dtype="float16"):
    grid_M = (M + block_M - 1) // block_M

    @T.prim_func
    def selectFullKernel(
        A: T.Tensor((N,), dtype),
        B: T.Tensor((N,), dtype),
        Out: T.Tensor((N,), dtype),
    ):
        with T.Kernel(grid_M, is_npu=True) as (bx, _):
            cond_ub = T.alloc_shared((N,), "bool")
            acc_A = T.alloc_shared((N,), dtype)
            acc_B = T.alloc_shared((N,), dtype)
            out_ub = T.alloc_shared((N,), dtype)

            T.copy(A, acc_A)
            T.copy(B, acc_B)

            T.npuir_cmp(acc_A, acc_B, cond_ub, "ge")

            for i in T.serial(block_M):
                T.npuir_select(
                    cond_ub,
                    acc_A,
                    acc_B,
                    out_ub,
                )

                T.copy(out_ub, Out)

    return selectFullKernel


def select_partial_kernel(M, N, block_M, dtype="float16"):
    grid_M = (M + block_M - 1) // block_M

    @T.prim_func
    def selectPartialKernel(
        A: T.Tensor((N,), dtype),
        B: T.Tensor((N,), dtype),
        Out: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(grid_M, is_npu=True) as (bx, _):
            cond_ub = T.alloc_shared((N,), "bool")
            acc_A = T.alloc_shared((N,), dtype)
            acc_B = T.alloc_shared((N,), dtype)
            out_ub = T.alloc_shared((M, N), dtype)

            T.copy(A, acc_A)
            T.copy(B, acc_B)

            T.npuir_cmp(acc_A, acc_B, cond_ub, "ge")

            for i in T.serial(block_M):
                T.npuir_select(
                    cond_ub,
                    acc_A,
                    acc_B,
                    out_ub[i, :],
                )

                T.copy(out_ub, Out)

    return selectPartialKernel


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("M, N, block_M", SELECT_CASES)
def test_select_dev(dtype, M, N, block_M):
    full_kernel = tilelang.compile(select_kernel(M, N, block_M, dtype), target="npuir")
    partial_kernel = tilelang.compile(select_partial_kernel(M, N, block_M, dtype), target="npuir")

    a = gen_tensor((N,), dtype, kind="randn")
    b = gen_tensor((N,), dtype, kind="randn")

    out_full = gen_tensor((N,), dtype, kind="zeros")
    full_kernel(a, b, out_full)
    ref_full = torch.where(a.cpu() >= b.cpu(), a.cpu(), b.cpu())
    assert_close(out_full.cpu(), ref_full, dtype=dtype, rtol=1e-3, atol=1e-3)

    out_partial = gen_tensor((M, N), dtype, kind="zeros")
    partial_kernel(a, b, out_partial)
    ref_partial = torch.where(
        (a.cpu() >= b.cpu())[None, :],
        a.cpu()[None, :],
        b.cpu()[None, :],
    ).expand(M, -1)
    assert_close(out_partial.cpu(), ref_partial, dtype=dtype, rtol=1e-3, atol=1e-3)
