import torch
import torch_npu
import tilelang
import tilelang.language as T
import pytest
import os

import testcommon as tc

tilelang.cache.clear_cache()

pytestmark = pytest.mark.mode("Developer")

BITWISE_DTYPE_CASES = ["int32", "int16", "uint32", "uint16"]
DTYPE_CASES = ["float16", "float32"] + BITWISE_DTYPE_CASES

M, N = 4, 64
block_M = 4

# ---------- Arithmetic Kernels (existing) ----------
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


def make_integer_tensor(shape, dtype, *, low, high):
    return torch.randint(low, high, shape, dtype=tc.DTYPE_MAP[dtype]).npu()


def compute_uint_reference(lhs, rhs, op):
    lhs_u = lhs.to(torch.int64)
    rhs_u = rhs.to(torch.int64)
    if op == "add":
        return lhs_u + rhs_u
    if op == "sub":
        return lhs_u - rhs_u
    if op == "mul":
        return lhs_u * rhs_u
    if op == "div":
        return lhs_u // rhs_u
    if op == "and":
        return lhs_u & rhs_u
    if op == "or":
        return lhs_u | rhs_u
    if op == "xor":
        return lhs_u ^ rhs_u
    if op == "shl":
        return lhs_u << rhs_u
    if op == "shr":
        return lhs_u >> rhs_u
    raise ValueError(op)


def compute_int_reference(lhs, rhs, op):
    if op == "add":
        return lhs + rhs
    if op == "sub":
        return lhs - rhs
    if op == "mul":
        return lhs * rhs
    if op == "div":
        return lhs / rhs
    if op == "and":
        return lhs & rhs
    if op == "or":
        return lhs | rhs
    if op == "xor":
        return lhs ^ rhs
    if op == "shl":
        return lhs << rhs
    if op == "shr":
        return lhs >> rhs
    raise ValueError(op)

# ---------- New: Bitwise Kernels (AND, OR, XOR) ----------
def bitwise_kernel(M, N, block_M, op, dtype="int32"):
    grid_M = (M + block_M - 1) // block_M

    @T.prim_func
    def bitwiseFullDev(
        A: T.Tensor((N,), dtype),
        B: T.Tensor((N,), dtype),
        Out: T.Tensor((N,), dtype),
    ):
        with T.Kernel(grid_M, is_npu=True) as (bx, _):
            acc_A = T.alloc_shared((N,), dtype)
            acc_B = T.alloc_shared((N,), dtype)
            out_ub = T.alloc_shared((N,), dtype)

            T.copy(A, acc_A)
            T.copy(B, acc_B)

            for i in T.serial(block_M):
                if op == "and":
                    T.npuir_and(acc_A, acc_B, out_ub)
                elif op == "or":
                    T.npuir_or(acc_A, acc_B, out_ub)
                elif op == "xor":
                    T.npuir_xor(acc_A, acc_B, out_ub)
                else:
                    T.assert_(False, "Unsupported bitwise op")

            T.copy(out_ub, Out)

    return bitwiseFullDev

def bitwise_partial_kernel(M, N, block_M, op, dtype="int32"):
    grid_M = (M + block_M - 1) // block_M

    @T.prim_func
    def bitwisePartialDev(
        A: T.Tensor((N,), dtype),
        B: T.Tensor((N,), dtype),
        Out: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(grid_M, is_npu=True) as (bx, _):
            acc_A = T.alloc_shared((N,), dtype)
            acc_B = T.alloc_shared((N,), dtype)
            out_ub = T.alloc_shared((M, N), dtype)

            T.copy(A, acc_A)
            T.copy(B, acc_B)

            for i in T.serial(block_M):
                if op == "and":
                    T.npuir_and(acc_A, acc_B, out_ub[i, :])
                elif op == "or":
                    T.npuir_or(acc_A, acc_B, out_ub[i, :])
                elif op == "xor":
                    T.npuir_xor(acc_A, acc_B, out_ub[i, :])
                else:
                    T.assert_(False, "Unsupported bitwise op")

            T.copy(out_ub, Out)

    return bitwisePartialDev

# ---------- New: Shift Kernels (SHL, SHR) ----------
def shift_kernel(M, N, block_M, op, dtype="int32"):
    grid_M = (M + block_M - 1) // block_M

    @T.prim_func
    def shiftFullDev(
        A: T.Tensor((N,), dtype),
        B: T.Tensor((N,), dtype),  # shift amounts
        Out: T.Tensor((N,), dtype),
    ):
        with T.Kernel(grid_M, is_npu=True) as (bx, _):
            acc_A = T.alloc_shared((N,), dtype)
            acc_B = T.alloc_shared((N,), dtype)
            out_ub = T.alloc_shared((N,), dtype)

            T.copy(A, acc_A)
            T.copy(B, acc_B)

            for i in T.serial(block_M):
                if op == "shl":
                    T.npuir_shl(acc_A, acc_B, out_ub)
                elif op == "shr":
                    T.npuir_shr(acc_A, acc_B, out_ub)
                else:
                    T.assert_(False, "Unsupported shift op")

            T.copy(out_ub, Out)

    return shiftFullDev

def shift_partial_kernel(M, N, block_M, op, dtype="int32"):
    grid_M = (M + block_M - 1) // block_M

    @T.prim_func
    def shiftPartialDev(
        A: T.Tensor((N,), dtype),
        B: T.Tensor((N,), dtype),
        Out: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(grid_M, is_npu=True) as (bx, _):
            acc_A = T.alloc_shared((N,), dtype)
            acc_B = T.alloc_shared((N,), dtype)
            out_ub = T.alloc_shared((M, N), dtype)

            T.copy(A, acc_A)
            T.copy(B, acc_B)

            for i in T.serial(block_M):
                if op == "shl":
                    T.npuir_shl(acc_A, acc_B, out_ub[i, :])
                elif op == "shr":
                    T.npuir_shr(acc_A, acc_B, out_ub[i, :])
                else:
                    T.assert_(False, "Unsupported shift op")

            T.copy(out_ub, Out)

    return shiftPartialDev

# ---------- Reference functions ----------
def reference_arith(A, B, M, op):
    if op == "add":
        ref = (A + B)[None, :].expand(M, -1)
    elif op == "sub":
        ref = (A - B)[None, :].expand(M, -1)
    elif op == "mul":
        ref = (A * B)[None, :].expand(M, -1)
    elif op == "div":
        ref = (A / B)[None, :].expand(M, -1)
    elif op == "add" and A.dtype.is_unsigned:
        ref = (A.to(torch.int64) + B.to(torch.int64))[None, :].expand(M, -1).to(A.dtype)
    elif op == "sub" and A.dtype.is_unsigned:
        ref = (A.to(torch.int64) - B.to(torch.int64))[None, :].expand(M, -1).to(A.dtype)
    elif op == "mul" and A.dtype.is_unsigned:
        ref = (A.to(torch.int64) * B.to(torch.int64))[None, :].expand(M, -1).to(A.dtype)
    elif op == "div" and A.dtype.is_unsigned:
        ref = (A.to(torch.int64) // B.to(torch.int64))[None, :].expand(M, -1).to(A.dtype)
    else:
        raise ValueError(op)
    return ref

def reference_bitwise(A, B, M, op):
    if op == "and":
        ref = torch.bitwise_and(A, B)[None, :].expand(M, -1)
    elif op == "or":
        ref = torch.bitwise_or(A, B)[None, :].expand(M, -1)
    elif op == "xor":
        ref = torch.bitwise_xor(A, B)[None, :].expand(M, -1)
    elif op == "shl":
        ref = (A << B)[None, :].expand(M, -1)
    elif op == "shr":
        ref = (A >> B)[None, :].expand(M, -1)
    else:
        raise ValueError(op)
    return ref

def reference_shift(A, B, M, op, bitwidth=32):
    # B is shift amount, clamp to 0..bitwidth-1 to avoid undefined behavior
    B_clamped = B.clamp(0, bitwidth - 1)
    if op == "shl":
        ref = (A << B_clamped)[None, :].expand(M, -1)
    elif op == "shr":
        ref = (A >> B_clamped)[None, :].expand(M, -1)
    else:
        raise ValueError(op)
    return ref


def reference_integer(A, B, M, op):
    if A.dtype.is_unsigned:
        return compute_uint_reference(A, B, op)[None, :].expand(M, -1).to(A.dtype)
    return compute_int_reference(A, B, op)[None, :].expand(M, -1)

# ---------- Arithmetic Tests (unchanged) ----------
@pytest.mark.op("vadd")
@pytest.mark.parametrize("dtype", DTYPE_CASES)
def test_vadd(dtype):
    datatype = tc.DTYPE_MAP[dtype]
    if dtype in BITWISE_DTYPE_CASES:
        # For bitwise ops, use integer inputs
        A = torch.randint(torch.iinfo(datatype).min, torch.iinfo(datatype).max, (N,), dtype=datatype).npu()
        B = torch.randint(torch.iinfo(datatype).min, torch.iinfo(datatype).max, (N,), dtype=datatype).npu()
    else:
        A = torch.randn(N, dtype=datatype).npu()
        B = torch.randn(N, dtype=datatype).npu()
    Out = torch.zeros((N,), dtype=datatype).npu()

    func = binary_kernel(M, N, block_M, "add", dtype)
    compiled = tilelang.compile(func, target="npuir")

    compiled(A, B, Out)
    if dtype.startswith("uint"):
        ref = (A.long() + B.long()).to(datatype)
    else:
        ref = A + B
    if dtype in BITWISE_DTYPE_CASES:
        # For bitwise ops, expect exact match
        assert(torch.equal(Out.cpu(), ref.cpu()))
    else:
        tc.assert_close(Out.cpu(), ref.cpu(), rtol=1e-2, atol=1e-2, equal_nan=True)

@pytest.mark.op("vsub")
@pytest.mark.parametrize("dtype", DTYPE_CASES)
def test_vsub(dtype):
    datatype = tc.DTYPE_MAP[dtype]
    if dtype in BITWISE_DTYPE_CASES:
        # For bitwise ops, use integer inputs
        A = torch.randint(torch.iinfo(datatype).min, torch.iinfo(datatype).max, (N,), dtype=datatype).npu()
        B = torch.randint(torch.iinfo(datatype).min, torch.iinfo(datatype).max, (N,), dtype=datatype).npu()
    else:
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
    if dtype in BITWISE_DTYPE_CASES:
        # For bitwise ops, use integer inputs
        A = torch.randint(torch.iinfo(datatype).min, torch.iinfo(datatype).max, (N,), dtype=datatype).npu()
        B = torch.randint(torch.iinfo(datatype).min, torch.iinfo(datatype).max, (N,), dtype=datatype).npu()
    else:
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
    if dtype in BITWISE_DTYPE_CASES:
        # For bitwise ops, use integer inputs
        A = torch.randint(torch.iinfo(datatype).min, torch.iinfo(datatype).max, (N,), dtype=datatype).npu()
        B = torch.randint(torch.iinfo(datatype).min, torch.iinfo(datatype).max, (N,), dtype=datatype).npu()
    else:
        A = torch.randn(N, dtype=datatype).npu()
        B = torch.randn(N, dtype=datatype).npu()
    Out = torch.zeros((N,), dtype=datatype).npu()
    func = binary_kernel(M, N, block_M, "div", dtype)
    compiled = tilelang.compile(func, target="npuir")
    compiled(A, B, Out)
    ref = A / B
    tc.assert_close(Out.cpu(), ref.cpu(), rtol=1e-2, atol=1e-2, equal_nan=True)

# Partial arithmetic
@pytest.mark.op("vadd_partial")
@pytest.mark.parametrize("dtype", DTYPE_CASES)
def test_vadd_partial(dtype):
    datatype = tc.DTYPE_MAP[dtype]
    if dtype in BITWISE_DTYPE_CASES:
        # For bitwise ops, use integer inputs
        A = torch.randint(torch.iinfo(datatype).min, torch.iinfo(datatype).max, (N,), dtype=datatype).npu()
        B = torch.randint(torch.iinfo(datatype).min, torch.iinfo(datatype).max, (N,), dtype=datatype).npu()
    else:
        A = torch.randn(N, dtype=datatype).npu()
        B = torch.randn(N, dtype=datatype).npu()
    Out = torch.zeros((M, N), dtype=datatype).npu()

    func = binary_partial_kernel(M, N, block_M, "add", dtype)
    compiled = tilelang.compile(func, target="npuir")

    compiled(A, B, Out)
    ref = reference_arith(A.cpu(), B.cpu(), M, "add")
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
    ref = reference_arith(A.cpu(), B.cpu(), M, "sub")
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
    ref = reference_arith(A.cpu(), B.cpu(), M, "mul")
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
    ref = reference_arith(A.cpu(), B.cpu(), M, "div")
    tc.assert_close(Out.cpu(), ref.cpu(), rtol=1e-2, atol=1e-2, equal_nan=True)


# ---------- New Tests: Bitwise AND, OR, XOR ----------
@pytest.mark.op("vand")
@pytest.mark.parametrize("dtype", BITWISE_DTYPE_CASES)
def test_vand(dtype):
    datatype = tc.DTYPE_MAP[dtype]
    # Generate random integers within reasonable range for the dtype
    if dtype == "int32":
        A = torch.randint(-(2**31), 2**31-1, (N,), dtype=datatype).npu()
        B = torch.randint(-(2**31), 2**31-1, (N,), dtype=datatype).npu()
    else:  # int16, int8 fallback
        A = torch.randint(-128, 127, (N,), dtype=datatype).npu()
        B = torch.randint(-128, 127, (N,), dtype=datatype).npu()
    Out = torch.zeros((N,), dtype=datatype).npu()

    func = bitwise_kernel(M, N, block_M, "and", dtype)
    compiled = tilelang.compile(func, target="npuir")
    compiled(A, B, Out)
    ref = torch.bitwise_and(A.cpu(), B.cpu())
    tc.assert_close(Out.cpu(), ref.cpu(), rtol=0, atol=0)

@pytest.mark.op("vor")
@pytest.mark.parametrize("dtype", BITWISE_DTYPE_CASES)
def test_vor(dtype):
    datatype = tc.DTYPE_MAP[dtype]
    if dtype == "int32":
        A = torch.randint(-(2**31), 2**31-1, (N,), dtype=datatype).npu()
        B = torch.randint(-(2**31), 2**31-1, (N,), dtype=datatype).npu()
    else:
        A = torch.randint(-128, 127, (N,), dtype=datatype).npu()
        B = torch.randint(-128, 127, (N,), dtype=datatype).npu()
    Out = torch.zeros((N,), dtype=datatype).npu()

    func = bitwise_kernel(M, N, block_M, "or", dtype)
    compiled = tilelang.compile(func, target="npuir")
    compiled(A, B, Out)
    ref = torch.bitwise_or(A.cpu(), B.cpu())
    tc.assert_close(Out.cpu(), ref.cpu(), rtol=0, atol=0)

@pytest.mark.op("vxor")
@pytest.mark.parametrize("dtype", BITWISE_DTYPE_CASES)
def test_vxor(dtype):
    datatype = tc.DTYPE_MAP[dtype]
    if dtype == "int32":
        A = torch.randint(-(2**31), 2**31-1, (N,), dtype=datatype).npu()
        B = torch.randint(-(2**31), 2**31-1, (N,), dtype=datatype).npu()
    else:
        A = torch.randint(-128, 127, (N,), dtype=datatype).npu()
        B = torch.randint(-128, 127, (N,), dtype=datatype).npu()
    Out = torch.zeros((N,), dtype=datatype).npu()

    func = bitwise_kernel(M, N, block_M, "xor", dtype)
    compiled = tilelang.compile(func, target="npuir")
    compiled(A, B, Out)
    ref = torch.bitwise_xor(A.cpu(), B.cpu())
    tc.assert_close(Out.cpu(), ref.cpu(), rtol=0, atol=0)

# Partial bitwise tests
@pytest.mark.op("vand_partial")
@pytest.mark.parametrize("dtype", BITWISE_DTYPE_CASES)
def test_vand_partial(dtype):
    datatype = tc.DTYPE_MAP[dtype]
    if dtype == "int32":
        A = torch.randint(-(2**31), 2**31-1, (N,), dtype=datatype).npu()
        B = torch.randint(-(2**31), 2**31-1, (N,), dtype=datatype).npu()
    else:
        A = torch.randint(-128, 127, (N,), dtype=datatype).npu()
        B = torch.randint(-128, 127, (N,), dtype=datatype).npu()
    Out = torch.zeros((M, N), dtype=datatype).npu()

    func = bitwise_partial_kernel(M, N, block_M, "and", dtype)
    compiled = tilelang.compile(func, target="npuir")
    compiled(A, B, Out)
    ref = reference_bitwise(A.cpu(), B.cpu(), M, "and")
    tc.assert_close(Out.cpu(), ref.cpu(), rtol=0, atol=0)

@pytest.mark.op("vor_partial")
@pytest.mark.parametrize("dtype", BITWISE_DTYPE_CASES)
def test_vor_partial(dtype):
    datatype = tc.DTYPE_MAP[dtype]
    if dtype == "int32":
        A = torch.randint(-(2**31), 2**31-1, (N,), dtype=datatype).npu()
        B = torch.randint(-(2**31), 2**31-1, (N,), dtype=datatype).npu()
    else:
        A = torch.randint(-128, 127, (N,), dtype=datatype).npu()
        B = torch.randint(-128, 127, (N,), dtype=datatype).npu()
    Out = torch.zeros((M, N), dtype=datatype).npu()

    func = bitwise_partial_kernel(M, N, block_M, "or", dtype)
    compiled = tilelang.compile(func, target="npuir")
    compiled(A, B, Out)
    ref = reference_bitwise(A.cpu(), B.cpu(), M, "or")
    tc.assert_close(Out.cpu(), ref.cpu(), rtol=0, atol=0)

@pytest.mark.op("vxor_partial")
@pytest.mark.parametrize("dtype", BITWISE_DTYPE_CASES)
def test_vxor_partial(dtype):
    datatype = tc.DTYPE_MAP[dtype]
    if dtype == "int32":
        A = torch.randint(-(2**31), 2**31-1, (N,), dtype=datatype).npu()
        B = torch.randint(-(2**31), 2**31-1, (N,), dtype=datatype).npu()
    else:
        A = torch.randint(-128, 127, (N,), dtype=datatype).npu()
        B = torch.randint(-128, 127, (N,), dtype=datatype).npu()
    Out = torch.zeros((M, N), dtype=datatype).npu()

    func = bitwise_partial_kernel(M, N, block_M, "xor", dtype)
    compiled = tilelang.compile(func, target="npuir")
    compiled(A, B, Out)
    ref = reference_bitwise(A.cpu(), B.cpu(), M, "xor")
    tc.assert_close(Out.cpu(), ref.cpu(), rtol=0, atol=0)

# ---------- New Tests: Shift Left and Shift Right ----------
@pytest.mark.op("vshl")
@pytest.mark.parametrize("dtype", BITWISE_DTYPE_CASES)
def test_vshl(dtype):
    datatype = tc.DTYPE_MAP[dtype]
    bitwidth = 32 if dtype == "int32" else 16 if dtype == "int16" else 8
    A = torch.randint(-(2**(bitwidth-1)), 2**(bitwidth-1)-1, (N,), dtype=datatype).npu()
    # shift amounts between 0 and bitwidth-1
    B = torch.randint(0, bitwidth, (N,), dtype=torch.int32).npu()
    Out = torch.zeros((N,), dtype=datatype).npu()

    func = shift_kernel(M, N, block_M, "shl", dtype)
    compiled = tilelang.compile(func, target="npuir")
    compiled(A, B, Out)
    # clamp shift amounts for reference
    B_cpu = B.cpu().clamp(0, bitwidth - 1)
    ref = (A.cpu() << B_cpu)
    tc.assert_close(Out.cpu(), ref.cpu(), rtol=0, atol=0)

@pytest.mark.op("vshr")
@pytest.mark.parametrize("dtype", BITWISE_DTYPE_CASES)
def test_vshr(dtype):
    datatype = tc.DTYPE_MAP[dtype]
    bitwidth = 32 if dtype == "int32" else 16 if dtype == "int16" else 8
    A = torch.randint(-(2**(bitwidth-1)), 2**(bitwidth-1)-1, (N,), dtype=datatype).npu()
    B = torch.randint(0, bitwidth, (N,), dtype=torch.int32).npu()
    Out = torch.zeros((N,), dtype=datatype).npu()

    func = shift_kernel(M, N, block_M, "shr", dtype)
    compiled = tilelang.compile(func, target="npuir")
    compiled(A, B, Out)
    B_cpu = B.cpu().clamp(0, bitwidth - 1)
    ref = (A.cpu() >> B_cpu)
    tc.assert_close(Out.cpu(), ref.cpu(), rtol=0, atol=0)

# Partial shift tests
@pytest.mark.op("vshl_partial")
@pytest.mark.parametrize("dtype", BITWISE_DTYPE_CASES)
def test_vshl_partial(dtype):
    datatype = tc.DTYPE_MAP[dtype]
    bitwidth = 32 if dtype == "int32" else 16 if dtype == "int16" else 8
    A = torch.randint(-(2**(bitwidth-1)), 2**(bitwidth-1)-1, (N,), dtype=datatype).npu()
    B = torch.randint(0, bitwidth, (N,), dtype=torch.int32).npu()
    Out = torch.zeros((M, N), dtype=datatype).npu()

    func = shift_partial_kernel(M, N, block_M, "shl", dtype)
    compiled = tilelang.compile(func, target="npuir")
    compiled(A, B, Out)
    ref = reference_shift(A.cpu(), B.cpu(), M, "shl", bitwidth)
    tc.assert_close(Out.cpu(), ref.cpu(), rtol=0, atol=0)

@pytest.mark.op("vshr_partial")
@pytest.mark.parametrize("dtype", BITWISE_DTYPE_CASES)
def test_vshr_partial(dtype):
    datatype = tc.DTYPE_MAP[dtype]
    bitwidth = 32 if dtype == "int32" else 16 if dtype == "int16" else 8
    A = torch.randint(-(2**(bitwidth-1)), 2**(bitwidth-1)-1, (N,), dtype=datatype).npu()
    B = torch.randint(0, bitwidth, (N,), dtype=torch.int32).npu()
    Out = torch.zeros((M, N), dtype=datatype).npu()

    func = shift_partial_kernel(M, N, block_M, "shr", dtype)
    compiled = tilelang.compile(func, target="npuir")
    compiled(A, B, Out)
    ref = reference_shift(A.cpu(), B.cpu(), M, "shr", bitwidth)
    tc.assert_close(Out.cpu(), ref.cpu(), rtol=0, atol=0)