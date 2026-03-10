import torch
import torch_npu
import tilelang
import tilelang.language as T
import os

import pytest
import testcommon as tc

DTYPE_CASES = ["int32"]

def pow_int_kernel(M, N, block_M, dtype):
    grid_M = (M + block_M - 1) // block_M

    @T.prim_func
    def vecPowDev(
        A: T.Tensor((N,), dtype),   # base: int32
        B: T.Tensor((N,), dtype),   # exponent: int32
        Out: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(grid_M, is_npu=True) as (bx, _):
            # UB buffers
            acc_A  = T.alloc_shared((N,), dtype)
            acc_B  = T.alloc_shared((N,), dtype)
            out_ub = T.alloc_shared((M, N), dtype)

            # GM -> UB
            T.copy(A, acc_A)
            T.copy(B, acc_B)

            # Each row: elementwise pow
            for i in T.serial(block_M):
                T.npuir_pow(acc_A, acc_B, out_ub[i, :])

            # UB -> GM
            T.copy(out_ub, Out)

    return vecPowDev


def reference(A, B, M):
    ref = torch.pow(A, B)[None, :].expand(M, -1)
    return ref

@pytest.mark.mode("Developer")
@pytest.mark.op("vpow_dev")
@pytest.mark.parametrize("dtype", DTYPE_CASES)
def test_pow_dev(dtype):
    dataType = tc.resolve_dtype(dtype)
    M, N = 4, 32
    block_M = 4

    A = torch.randint(
        low=0, high=5, size=(N,), dtype=dataType
    ).npu()

    B = torch.randint(
        low=0, high=4, size=(N,), dtype=dataType
    ).npu()

    Out = torch.zeros((M, N), dtype=dataType).npu()

    func = pow_int_kernel(M, N, block_M, dtype)
    compiled = tilelang.compile(func, target="npuir")

    compiled(A, B, Out)

    ref = reference(A.cpu(), B.cpu(), M)
    tc.assert_close(Out.cpu(), ref, atol=0, rtol=0)

