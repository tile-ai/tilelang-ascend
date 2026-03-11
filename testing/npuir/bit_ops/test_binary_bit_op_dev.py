import pytest
import torch
import torch_npu  # noqa: F401

import tilelang
import tilelang.language as T

from testcommon import assert_close, gen_tensor


pytestmark = [pytest.mark.mode("Developer")]

DTYPES = ["int32"]
BINARY_CASES = [(4, 32, 4)]


def binary_kernel(M, N, block_M, op_name):
    grid_M = (M + block_M - 1) // block_M

    @T.prim_func
    def bitBinaryFullDev(
        A: T.Tensor((N,), "int32"),
        B: T.Tensor((N,), "int32"),
        Out: T.Tensor((N,), "int32"),
    ):
        with T.Kernel(grid_M, is_npu=True) as (bx, _):
            acc_A = T.alloc_shared((N,), "int32")
            acc_B = T.alloc_shared((N,), "int32")
            out_ub = T.alloc_shared((N,), "int32")

            T.copy(A, acc_A)
            T.copy(B, acc_B)

            for i in T.serial(block_M):
                if op_name == "max":
                    T.npuir_max(acc_A, acc_B, out_ub)
                elif op_name == "min":
                    T.npuir_min(acc_A, acc_B, out_ub)
                elif op_name == "and":
                    T.npuir_and(acc_A, acc_B, out_ub)
                elif op_name == "or":
                    T.npuir_or(acc_A, acc_B, out_ub)
                elif op_name == "xor":
                    T.npuir_xor(acc_A, acc_B, out_ub)
                elif op_name == "shl":
                    T.npuir_shl(acc_A, acc_B, out_ub)
                elif op_name == "shr":
                    T.npuir_shr(acc_A, acc_B, out_ub)
                else:
                    raise ValueError(f"Unsupported op: {op_name}")

            T.copy(out_ub, Out)

    return bitBinaryFullDev


def binary_partial_kernel(M, N, block_M, op_name):
    grid_M = (M + block_M - 1) // block_M

    @T.prim_func
    def bitBinaryPartialDev(
        A: T.Tensor((N,), "int32"),
        B: T.Tensor((N,), "int32"),
        Out: T.Tensor((M, N), "int32"),
    ):
        with T.Kernel(grid_M, is_npu=True) as (bx, _):
            acc_A = T.alloc_shared((N,), "int32")
            acc_B = T.alloc_shared((N,), "int32")
            out_ub = T.alloc_shared((M, N), "int32")

            T.copy(A, acc_A)
            T.copy(B, acc_B)

            for i in T.serial(block_M):
                if op_name == "max":
                    T.npuir_max(acc_A, acc_B, out_ub[i, :])
                elif op_name == "min":
                    T.npuir_min(acc_A, acc_B, out_ub[i, :])
                elif op_name == "and":
                    T.npuir_and(acc_A, acc_B, out_ub[i, :])
                elif op_name == "or":
                    T.npuir_or(acc_A, acc_B, out_ub[i, :])
                elif op_name == "xor":
                    T.npuir_xor(acc_A, acc_B, out_ub[i, :])
                elif op_name == "shl":
                    T.npuir_shl(acc_A, acc_B, out_ub[i, :])
                elif op_name == "shr":
                    T.npuir_shr(acc_A, acc_B, out_ub[i, :])
                else:
                    raise ValueError(f"Unsupported op: {op_name}")

            T.copy(out_ub, Out)

    return bitBinaryPartialDev


def compute_expected(A, B, op_name):
    if op_name == "max":
        return torch.max(A, B)
    if op_name == "min":
        return torch.min(A, B)
    if op_name == "and":
        return A & B
    if op_name == "or":
        return A | B
    if op_name == "xor":
        return A ^ B
    if op_name == "shl":
        return A << B
    if op_name == "shr":
        return A >> B
    raise ValueError(f"Unsupported op: {op_name}")


def run_binary_case(M, N, block_M, dtype, op_name):
    A = gen_tensor((N,), dtype, kind="randint", low=0, high=10)
    B = gen_tensor((N,), dtype, kind="randint", low=0, high=10)

    out_full = gen_tensor((N,), dtype, kind="zeros")
    full_kernel = tilelang.compile(binary_kernel(M, N, block_M, op_name), target="npuir")
    full_kernel(A, B, out_full)

    expected_full = compute_expected(A.cpu(), B.cpu(), op_name)
    assert_close(out_full.cpu(), expected_full.cpu(), dtype=dtype)

    out_partial = gen_tensor((M, N), dtype, kind="zeros")
    partial_kernel = tilelang.compile(binary_partial_kernel(M, N, block_M, op_name), target="npuir")
    partial_kernel(A, B, out_partial)

    expected_partial = expected_full[None, :].expand(M, -1)
    assert_close(out_partial.cpu(), expected_partial.cpu(), dtype=dtype)


@pytest.mark.op("max")
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("M, N, block_M", BINARY_CASES)
def test_binary_max_dev(M, N, block_M, dtype):
    run_binary_case(M, N, block_M, dtype, "max")


@pytest.mark.op("min")
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("M, N, block_M", BINARY_CASES)
def test_binary_min_dev(M, N, block_M, dtype):
    run_binary_case(M, N, block_M, dtype, "min")


@pytest.mark.op("and")
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("M, N, block_M", BINARY_CASES)
def test_binary_and_dev(M, N, block_M, dtype):
    run_binary_case(M, N, block_M, dtype, "and")


@pytest.mark.op("or")
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("M, N, block_M", BINARY_CASES)
def test_binary_or_dev(M, N, block_M, dtype):
    run_binary_case(M, N, block_M, dtype, "or")


@pytest.mark.op("xor")
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("M, N, block_M", BINARY_CASES)
def test_binary_xor_dev(M, N, block_M, dtype):
    run_binary_case(M, N, block_M, dtype, "xor")


@pytest.mark.op("shl")
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("M, N, block_M", BINARY_CASES)
def test_binary_shl_dev(M, N, block_M, dtype):
    run_binary_case(M, N, block_M, dtype, "shl")


@pytest.mark.op("shr")
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("M, N, block_M", BINARY_CASES)
def test_binary_shr_dev(M, N, block_M, dtype):
    run_binary_case(M, N, block_M, dtype, "shr")
