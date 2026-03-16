import torch
import torch_npu
import tilelang
import tilelang.language as T
import pytest
import os

import testcommon as tc

tilelang.cache.clear_cache()

pytestmark = pytest.mark.mode("Developer")

DTYPE_CASES = ["float16", "float32"]

M, N = 4, 64
block_M = 4

def binary_kernel(M, N, block_M, op, dtype="float16"):
    grid_M = (M + block_M - 1) // block_M

    @T.prim_func
    def binaryArithFullDev(
        A: T.Tensor((N,), dtype),
        B: T.Tensor((N,), dtype),
        Out: T.Tensor((N,), dtype),
    ):
        with T.Kernel(grid_M, is_npu=True) as (bx, _):
            # UB buffers
            acc_A = T.alloc_shared((N,), dtype)
            acc_B = T.alloc_shared((N,), dtype)
            out_ub = T.alloc_shared((N,), dtype)

            # GM -> UB
            T.copy(A, acc_A)
            T.copy(B, acc_B)

            # Each row: elementwise binary op
            for i in T.serial(block_M):
                if op == "add":
                    T.npuir_add(acc_A, acc_B, out_ub)
                elif op == "sub":
                    T.npuir_sub(acc_A, acc_B, out_ub)
                elif op == "mul":
                    T.npuir_mul(acc_A, acc_B, out_ub)
                elif op == "div":
                    T.npuir_div(acc_A, acc_B, out_ub)
                else:
                    T.assert_(False, "Unsupported op")

            # UB -> GM
            T.copy(out_ub, Out)

    return binaryArithFullDev

def binary_partial_kernel(M, N, block_M, op, dtype="float16"):
    grid_M = (M + block_M - 1) // block_M

    @T.prim_func
    def binaryArithPartialDev(
        A: T.Tensor((N,), dtype),
        B: T.Tensor((N,), dtype),
        Out: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(grid_M, is_npu=True) as (bx, _):
            # UB buffers
            acc_A = T.alloc_shared((N,), dtype)
            acc_B = T.alloc_shared((N,), dtype)
            out_ub = T.alloc_shared((M, N), dtype)

            # GM -> UB
            T.copy(A, acc_A)
            T.copy(B, acc_B)

            # Each row: elementwise binary op
            for i in T.serial(block_M):
                if op == "add":
                    T.npuir_add(acc_A, acc_B, out_ub[i, :])
                elif op == "sub":
                    T.npuir_sub(acc_A, acc_B, out_ub[i, :])
                elif op == "mul":
                    T.npuir_mul(acc_A, acc_B, out_ub[i, :])
                elif op == "div":
                    T.npuir_div(acc_A, acc_B, out_ub[i, :])
                else:
                    T.assert_(False, "Unsupported op")

            # UB -> GM
            T.copy(out_ub, Out)

    return binaryArithPartialDev


def reference(A, B, M, op):
    if op == "add":
        ref = (A + B)[None, :].expand(M, -1)
    elif op == "sub":
        ref = (A - B)[None, :].expand(M, -1)
    elif op == "mul":
        ref = (A * B)[None, :].expand(M, -1)
    elif op == "div":
        ref = (A / B)[None, :].expand(M, -1)
    else:
        raise ValueError(op)
    return ref

@pytest.mark.op("vadd")
@pytest.mark.parametrize("dtype", DTYPE_CASES)
def test_vadd(dtype):
    datatype = tc.DTYPE_MAP[dtype]
    A = torch.randn(N, dtype=datatype).npu()
    B = torch.randn(N, dtype=datatype).npu()
    Out = torch.zeros((N,), dtype=datatype).npu()

    func = binary_kernel(M, N, block_M, "add", dtype)
    compiled = tilelang.compile(func, target="npuir")

    compiled(A, B, Out)
    ref = A + B
    tc.assert_close(Out.cpu(), ref.cpu(), rtol=1e-2, atol=1e-2, equal_nan=True)

@pytest.mark.op("vsub")
@pytest.mark.parametrize("dtype", DTYPE_CASES)
def test_vsub(dtype):
    datatype = tc.DTYPE_MAP[dtype]
    A = torch.randn(N, dtype=datatype).npu()
    B = torch.randn(N, dtype=datatype).npu()
    Out = torch.zeros((N,), dtype=datatype).npu()
    func = binary_kernel(M, N, block_M, "sub", dtype)
    compiled = tilelang.compile(func, target="npuir")
    compiled(A, B, Out)
    ref = A - B
    tc.assert_close(Out.cpu(), ref.cpu(), rtol=1e-2, atol=1e-2, equal_nan=True)

@pytest.mark.op("vmul")
@pytest.mark.parametrize("dtype", DTYPE_CASES)
def test_vmul(dtype):
    datatype = tc.DTYPE_MAP[dtype]
    A = torch.randn(N, dtype=datatype).npu()
    B = torch.randn(N, dtype=datatype).npu()
    Out = torch.zeros((N,), dtype=datatype).npu()
    func = binary_kernel(M, N, block_M, "mul", dtype)
    compiled = tilelang.compile(func, target="npuir")
    compiled(A, B, Out)
    ref = A * B
    tc.assert_close(Out.cpu(), ref.cpu(), rtol=1e-2, atol=1e-2, equal_nan=True)

@pytest.mark.op("vdiv")
@pytest.mark.parametrize("dtype", DTYPE_CASES)
def test_vdiv(dtype):
    datatype = tc.DTYPE_MAP[dtype]
    A = torch.randn(N, dtype=datatype).npu()
    B = torch.randn(N, dtype=datatype).npu()
    Out = torch.zeros((N,), dtype=datatype).npu()
    func = binary_kernel(M, N, block_M, "div", dtype)
    compiled = tilelang.compile(func, target="npuir")
    compiled(A, B, Out)
    ref = A / B
    tc.assert_close(Out.cpu(), ref.cpu(), rtol=1e-2, atol=1e-2, equal_nan=True)

# Partial binary op tests

@pytest.mark.op("vadd_partial")
@pytest.mark.parametrize("dtype", DTYPE_CASES)
def test_vadd_partial(dtype):
    datatype = tc.DTYPE_MAP[dtype]
    A = torch.randn(N, dtype=datatype).npu()
    B = torch.randn(N, dtype=datatype).npu()
    Out = torch.zeros((M, N), dtype=datatype).npu()

    func = binary_partial_kernel(M, N, block_M, "add", dtype)
    compiled = tilelang.compile(func, target="npuir")

    compiled(A, B, Out)
    ref = reference(A.cpu(), B.cpu(), M, "add")
    tc.assert_close(Out.cpu(), ref.cpu(), rtol=1e-2, atol=1e-2, equal_nan=True)

@pytest.mark.op("vsub_partial")
@pytest.mark.parametrize("dtype", DTYPE_CASES)
def test_vsub_partial(dtype):
    datatype = tc.DTYPE_MAP[dtype]
    A = torch.randn(N, dtype=datatype).npu()
    B = torch.randn(N, dtype=datatype).npu()
    Out = torch.zeros((M, N), dtype=datatype).npu()

    func = binary_partial_kernel(M, N, block_M, "sub", dtype)
    compiled = tilelang.compile(func, target="npuir")

    compiled(A, B, Out)
    ref = reference(A.cpu(), B.cpu(), M, "sub")
    tc.assert_close(Out.cpu(), ref.cpu(), rtol=1e-2, atol=1e-2, equal_nan=True)

@pytest.mark.op("vmul_partial")
@pytest.mark.parametrize("dtype", DTYPE_CASES)
def test_vmul_partial(dtype):
    datatype = tc.DTYPE_MAP[dtype]
    A = torch.randn(N, dtype=datatype).npu()
    B = torch.randn(N, dtype=datatype).npu()
    Out = torch.zeros((M, N), dtype=datatype).npu()

    func = binary_partial_kernel(M, N, block_M, "mul", dtype)
    compiled = tilelang.compile(func, target="npuir")

    compiled(A, B, Out)
    ref = reference(A.cpu(), B.cpu(), M, "mul")
    tc.assert_close(Out.cpu(), ref.cpu(), rtol=1e-2, atol=1e-2, equal_nan=True)

@pytest.mark.op("vdiv_partial")
@pytest.mark.parametrize("dtype", DTYPE_CASES)
def test_vdiv_partial(dtype):   
    datatype = tc.DTYPE_MAP[dtype]
    A = torch.randn(N, dtype=datatype).npu()
    B = torch.randn(N, dtype=datatype).npu()
    Out = torch.zeros((M, N), dtype=datatype).npu()

    func = binary_partial_kernel(M, N, block_M, "div", dtype)
    compiled = tilelang.compile(func, target="npuir")

    compiled(A, B, Out)
    ref = reference(A.cpu(), B.cpu(), M, "div")
    tc.assert_close(Out.cpu(), ref.cpu(), rtol=1e-2, atol=1e-2, equal_nan=True)

