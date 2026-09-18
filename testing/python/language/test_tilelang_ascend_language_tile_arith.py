import argparse

import pytest
import torch

import tilelang
import tilelang.language as T

"""
Test suite for T.tile.add/sub/mul/div APIs.

Covers:
  - tensor-tensor form (AscendC::Add/Sub/Mul/Div):
    float16/float32 default, int16/int32 low_priority
  - scalar form (AscendC::Adds/Muls, sub via negated scalar, div via reciprocal):
    float16/float32 default, int16/int32 low_priority
  - BufferLoad scalar form (1D single element):
    float16/float32 default; int16/int32 add/mul low_priority;
    sub int16/int32 is ascendc compile-fail -> pto only
  - 2D whole-row slice regions (BufferRegion, flash-attention style)
  - in-place aliasing (dst==src0 / dst==src1 / src0==src1)
  - size mismatch validation (dst vs src0, src1 BufferRegion) and the
    unvalidated src1-Buffer size case
  - div integer dtype unsupported (compile error on both backends)

See api/api_docs/T.tile.{add,sub,mul,div}.md for the verified constraint set.
"""

TORCH_DTYPE = {
    "float16": torch.float16,
    "float32": torch.float32,
    "int16": torch.int16,
    "int32": torch.int32,
}
RTOL = 1e-2
ATOL = 1e-2

PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}

# -----------------------------------------------------------------------------
# Kernel builders
# -----------------------------------------------------------------------------


def _make_data(M, N, dtype):
    """Random a and positive b; b >= 0.5 keeps the division bounded."""
    torch_dt = TORCH_DTYPE[dtype]
    if dtype.startswith("float"):
        a = torch.randn(M, N, dtype=torch_dt)
        b = torch.rand(M, N, dtype=torch_dt) + 0.5
    else:
        a = torch.randint(-100, 100, (M, N), dtype=torch_dt)
        b = torch.randint(1, 100, (M, N), dtype=torch_dt)
    return a, b


def binary_tensor_kernel(op, M, N, dtype):
    """dst/src0/src1 are whole 2D buffers."""

    @T.prim_func
    def main(A: T.Tensor((M, N), dtype), B: T.Tensor((M, N), dtype), C: T.Tensor((M, N), dtype)):
        with T.Kernel(1, is_npu=True) as (cid, _):
            a_ub = T.alloc_ub((M, N), dtype)
            b_ub = T.alloc_ub((M, N), dtype)
            c_ub = T.alloc_ub((M, N), dtype)
            T.copy(A, a_ub)
            T.copy(B, b_ub)
            getattr(T.tile, op)(c_ub, a_ub, b_ub)
            T.copy(c_ub, C)

    return main


def binary_scalar_kernel(op, M, N, scalar, dtype):
    @T.prim_func
    def main(A: T.Tensor((M, N), dtype), C: T.Tensor((M, N), dtype)):
        with T.Kernel(1, is_npu=True) as (cid, _):
            a_ub = T.alloc_ub((M, N), dtype)
            c_ub = T.alloc_ub((M, N), dtype)
            T.copy(A, a_ub)
            getattr(T.tile, op)(c_ub, a_ub, scalar)
            T.copy(c_ub, C)

    return main


def binary_buffload_kernel(op, M, N, m, dtype):
    """src1 is a 1D scalar buffer element (S[3])."""

    @T.prim_func
    def main(A: T.Tensor((M, N), dtype), S: T.Tensor((m,), dtype), C: T.Tensor((M, N), dtype)):
        with T.Kernel(1, is_npu=True) as (cid, _):
            a_ub = T.alloc_ub((M, N), dtype)
            s_ub = T.alloc_ub((m,), dtype)
            c_ub = T.alloc_ub((M, N), dtype)
            T.copy(A, a_ub)
            T.copy(S, s_ub)
            getattr(T.tile, op)(c_ub, a_ub, s_ub[3])
            T.copy(c_ub, C)

    return main


def binary_row_slice_kernel(op, M, N, m, dtype):
    """2D whole-row slices as dst/src0 (flash-attention style)."""

    @T.prim_func
    def main(A: T.Tensor((M, N), dtype), S: T.Tensor((m,), dtype), C: T.Tensor((M, N), dtype)):
        with T.Kernel(1, is_npu=True) as (cid, _):
            a_ub = T.alloc_ub((M, N), dtype)
            s_ub = T.alloc_ub((m,), dtype)
            c_ub = T.alloc_ub((M, N), dtype)
            T.copy(A, a_ub)
            T.copy(S, s_ub)
            for h_i in range(M):
                getattr(T.tile, op)(c_ub[h_i, :], a_ub[h_i, :], s_ub[h_i % m])
            T.copy(c_ub, C)

    return main


def binary_inplace_kernel(op, form, M, N, dtype):
    """In-place aliasing: dst==src0, dst==src1, or src0==src1 (distinct GM outs)."""

    @T.prim_func
    def main(A: T.Tensor((M, N), dtype), B: T.Tensor((M, N), dtype), C: T.Tensor((M, N), dtype)):
        with T.Kernel(1, is_npu=True) as (cid, _):
            a_ub = T.alloc_ub((M, N), dtype)
            b_ub = T.alloc_ub((M, N), dtype)
            c_ub = T.alloc_ub((M, N), dtype)
            T.copy(A, a_ub)
            T.copy(B, b_ub)
            if form == "dst=src0":
                getattr(T.tile, op)(a_ub, a_ub, b_ub)
                T.copy(a_ub, C)
            elif form == "dst=src1":
                getattr(T.tile, op)(b_ub, a_ub, b_ub)
                T.copy(b_ub, C)
            else:  # "src0=src1"
                getattr(T.tile, op)(c_ub, b_ub, b_ub)
                T.copy(c_ub, C)

    return main


def binary_tensor_mismatch_kernel(dtype):
    """dst/src0 sizes differ -> Python assert at trace time."""

    @T.prim_func
    def main(A: T.Tensor((64,), dtype), B: T.Tensor((128,), dtype), C: T.Tensor((64,), dtype)):
        with T.Kernel(1, is_npu=True) as (cid, _):
            a_ub = T.alloc_ub((64,), dtype)
            b_ub = T.alloc_ub((128,), dtype)
            c_ub = T.alloc_ub((64,), dtype)
            T.copy(A, a_ub)
            T.copy(B, b_ub)
            T.tile.add(c_ub, b_ub, a_ub)
            T.copy(c_ub, C)

    return main


def binary_region_mismatch_kernel(dtype):
    """src1 BufferRegion size differs -> Python assert at trace time."""

    @T.prim_func
    def main(A: T.Tensor((64,), dtype), B: T.Tensor((128,), dtype), C: T.Tensor((64,), dtype)):
        with T.Kernel(1, is_npu=True) as (cid, _):
            a_ub = T.alloc_ub((64,), dtype)
            b_ub = T.alloc_ub((128,), dtype)
            c_ub = T.alloc_ub((64,), dtype)
            T.copy(A, a_ub)
            T.copy(B, b_ub)
            T.tile.add(c_ub, a_ub, b_ub[0:32])
            T.copy(c_ub, C)

    return main


def binary_buffer_mismatch_kernel(dtype):
    """src1 Buffer size differs -> NOT validated, only dst.size elements used."""

    @T.prim_func
    def main(A: T.Tensor((64,), dtype), B: T.Tensor((128,), dtype), C: T.Tensor((64,), dtype)):
        with T.Kernel(1, is_npu=True) as (cid, _):
            a_ub = T.alloc_ub((64,), dtype)
            b_ub = T.alloc_ub((128,), dtype)
            c_ub = T.alloc_ub((64,), dtype)
            T.copy(A, a_ub)
            T.copy(B, b_ub)
            T.tile.add(c_ub, a_ub, b_ub)
            T.copy(c_ub, C)

    return main


# -----------------------------------------------------------------------------
# Run-test helpers
# -----------------------------------------------------------------------------


def run_test_tensor(op, M, N, dtype, target):
    a, b = _make_data(M, N, dtype)
    kernel = binary_tensor_kernel(op, M, N, dtype)
    func = tilelang.compile(kernel, out_idx=[-1], pass_configs=PASS_CONFIGS, target=target)
    out = func(a.npu(), b.npu())
    torch.npu.synchronize()
    ref = {"add": a + b, "sub": a - b, "mul": a * b, "div": a / b}[op]
    torch.testing.assert_close(out.cpu().float(), ref.cpu().float(), rtol=RTOL, atol=ATOL)


def run_test_scalar(op, M, N, scalar, dtype, target):
    a, _ = _make_data(M, N, dtype)
    kernel = binary_scalar_kernel(op, M, N, scalar, dtype)
    func = tilelang.compile(kernel, out_idx=[-1], pass_configs=PASS_CONFIGS, target=target)
    out = func(a.npu())
    torch.npu.synchronize()
    ref = {"add": a + scalar, "sub": a - scalar, "mul": a * scalar, "div": a / scalar}[op]
    torch.testing.assert_close(out.cpu().float(), ref.cpu().float(), rtol=RTOL, atol=ATOL)


def run_test_buffload(op, M, N, m, dtype, target):
    a, b = _make_data(M, N, dtype)
    if dtype.startswith("float"):
        values = [1.5, 2.0, 0.5, 3.0, 4.0, -1.0, 0.25, 0.75]
    else:
        values = [1, 2, 3, 4, 5, 6, 7, 8]
    s = torch.tensor(values, dtype=TORCH_DTYPE[dtype])[:m]
    kernel = binary_buffload_kernel(op, M, N, m, dtype)
    func = tilelang.compile(kernel, out_idx=[-1], pass_configs=PASS_CONFIGS, target=target)
    out = func(a.npu(), s.npu())
    torch.npu.synchronize()
    ref = {"add": a + s[3], "sub": a - s[3], "mul": a * s[3], "div": a / s[3]}[op]
    torch.testing.assert_close(out.cpu().float(), ref.cpu().float(), rtol=RTOL, atol=ATOL)


def run_test_row_slice(op, M, N, m, dtype, target):
    a, _ = _make_data(M, N, dtype)
    if dtype.startswith("float"):
        values = [1.5, 2.0, 0.5, 3.0, 4.0, -1.0, 0.25, 0.75]
    else:
        values = [1, 2, 3, 4, 5, 6, 7, 8]
    s = torch.tensor(values, dtype=TORCH_DTYPE[dtype])[:m]
    kernel = binary_row_slice_kernel(op, M, N, m, dtype)
    func = tilelang.compile(kernel, out_idx=[-1], pass_configs=PASS_CONFIGS, target=target)
    out = func(a.npu(), s.npu())
    torch.npu.synchronize()
    ref = torch.stack(
        [{"add": a[h] + s[h % m], "sub": a[h] - s[h % m], "mul": a[h] * s[h % m], "div": a[h] / s[h % m]}[op] for h in range(M)]
    )
    torch.testing.assert_close(out.cpu().float(), ref.cpu().float(), rtol=RTOL, atol=ATOL)


def run_test_inplace(op, form, M, N, dtype, target):
    a, b = _make_data(M, N, dtype)
    kernel = binary_inplace_kernel(op, form, M, N, dtype)
    func = tilelang.compile(kernel, out_idx=[-1], pass_configs=PASS_CONFIGS, target=target)
    out = func(a.npu(), b.npu())
    torch.npu.synchronize()
    if form in ("dst=src0", "dst=src1"):
        ref = {"add": a + b, "sub": a - b, "mul": a * b, "div": a / b}[op]
    else:  # src0=src1
        ref = {"add": b + b, "sub": b - b, "mul": b * b, "div": b / b}[op]
    torch.testing.assert_close(out.cpu().float(), ref.cpu().float(), rtol=RTOL, atol=ATOL)


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture(scope="session", autouse=True)
def clear_cache():
    """Clear tilelang cache before tests."""
    tilelang.cache.clear_cache()
    yield


@pytest.fixture
def setup_random_seed():
    """Set random seed for reproducibility."""
    torch.manual_seed(0)
    yield


# -----------------------------------------------------------------------------
# Test cases: tensor-tensor form
# -----------------------------------------------------------------------------

TENSOR_INT_DTYPES = [
    pytest.param("int16", marks=pytest.mark.low_priority),
    pytest.param("int32", marks=pytest.mark.low_priority),
]
FLOAT_DTYPES = ["float32", pytest.param("float16", marks=pytest.mark.low_priority)]
# add/sub/mul/div share the same binary_op path; only `add` stays in the
# PR-triggered set as the representative, sub/mul/div run in the full suite.
ARITH_OPS = [
    "add",
    pytest.param("sub", marks=pytest.mark.low_priority),
    pytest.param("mul", marks=pytest.mark.low_priority),
]


@pytest.mark.usefixtures("setup_random_seed")
@pytest.mark.parametrize("op", ARITH_OPS)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES + TENSOR_INT_DTYPES)
@pytest.mark.parametrize("target", ["ascendc", pytest.param("pto", marks=pytest.mark.low_priority)])
def test_tile_arith_tensor(op, dtype, target):
    """Tensor-tensor form: float16/float32/int16/int32."""
    run_test_tensor(op, 64, 128, dtype, target)


@pytest.mark.usefixtures("setup_random_seed")
@pytest.mark.low_priority
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("target", ["ascendc", "pto"])
def test_tile_arith_tensor_div(dtype, target):
    """Tensor-tensor div: floating point only."""
    run_test_tensor("div", 64, 128, dtype, target)


@pytest.mark.usefixtures("setup_random_seed")
@pytest.mark.low_priority
@pytest.mark.parametrize("dtype", ["int16", "int32"])
@pytest.mark.parametrize("target", ["ascendc", "pto"])
def test_tile_arith_div_int_compile_error(dtype, target):
    """Div with integer dtypes fails to compile (AscendC Div static_assert)."""
    with pytest.raises(RuntimeError):
        kernel = binary_tensor_kernel("div", 64, 128, dtype)
        tilelang.compile(kernel, out_idx=[-1], pass_configs=PASS_CONFIGS, target=target)


# -----------------------------------------------------------------------------
# Test cases: scalar form
# -----------------------------------------------------------------------------


@pytest.mark.usefixtures("setup_random_seed")
@pytest.mark.parametrize("op", ARITH_OPS)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES + TENSOR_INT_DTYPES)
@pytest.mark.parametrize("target", ["ascendc", pytest.param("pto", marks=pytest.mark.low_priority)])
def test_tile_arith_scalar(op, dtype, target):
    """Scalar immediate: dst = src0 op scalar."""
    scalar = 2 if dtype in ("int16", "int32") else 2.0
    run_test_scalar(op, 64, 128, scalar, dtype, target)


@pytest.mark.usefixtures("setup_random_seed")
@pytest.mark.low_priority
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("target", ["ascendc", "pto"])
def test_tile_arith_scalar_div(dtype, target):
    """Scalar div (implemented as multiply by reciprocal)."""
    run_test_scalar("div", 64, 128, 2.0, dtype, target)


# -----------------------------------------------------------------------------
# Test cases: BufferLoad scalar form
# -----------------------------------------------------------------------------


@pytest.mark.usefixtures("setup_random_seed")
@pytest.mark.parametrize("op", ARITH_OPS + [pytest.param("div", marks=pytest.mark.low_priority)])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("target", ["ascendc", pytest.param("pto", marks=pytest.mark.low_priority)])
def test_tile_arith_buffload(op, dtype, target):
    """1D BufferLoad scalar: dst = src0 op S[3]."""
    run_test_buffload(op, 64, 128, 8, dtype, target)


@pytest.mark.usefixtures("setup_random_seed")
@pytest.mark.low_priority
@pytest.mark.parametrize("op", ["add", "mul"])
@pytest.mark.parametrize("dtype", ["int16", "int32"])
@pytest.mark.parametrize("target", ["ascendc", "pto"])
def test_tile_arith_buffload_int_add_mul(op, dtype, target):
    """BufferLoad scalar with integer dtypes: add/mul work on both backends."""
    run_test_buffload(op, 64, 128, 8, dtype, target)


@pytest.mark.usefixtures("setup_random_seed")
@pytest.mark.low_priority
@pytest.mark.parametrize("target", [pytest.param("pto", marks=pytest.mark.low_priority)])
def test_tile_arith_buffload_sub_int(target):
    """BufferLoad scalar sub with int16 on pto works.

    ascendc is excluded here: its codegen mixes a float scalar with int
    tensors (codegen_ascend.cc SubsOpCodegen) -> no matching Adds overload.
    See doc constraint 8 and the separate compile-error test below.
    """
    run_test_buffload("sub", 64, 128, 8, "int16", target)


@pytest.mark.usefixtures("setup_random_seed")
@pytest.mark.low_priority
def test_tile_arith_buffload_sub_int_ascendc_compile_fail():
    """BufferLoad scalar sub with int16 fails to compile on ascendc."""
    with pytest.raises(RuntimeError):
        kernel = binary_buffload_kernel("sub", 64, 128, 8, "int16")
        tilelang.compile(kernel, out_idx=[-1], pass_configs=PASS_CONFIGS, target="ascendc")


# -----------------------------------------------------------------------------
# Test cases: 2D whole-row slices
# -----------------------------------------------------------------------------


@pytest.mark.usefixtures("setup_random_seed")
@pytest.mark.parametrize("op", ARITH_OPS + [pytest.param("div", marks=pytest.mark.low_priority)])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("target", ["ascendc", pytest.param("pto", marks=pytest.mark.low_priority)])
def test_tile_arith_row_slice_2d(op, dtype, target):
    """2D whole-row slice regions with a per-row BufferLoad scalar."""
    run_test_row_slice(op, 4, 128, 4, dtype, target)


# -----------------------------------------------------------------------------
# Test cases: in-place aliasing
# -----------------------------------------------------------------------------

INPLACE_FORMS = [
    "dst=src0",
    pytest.param("dst=src1", marks=pytest.mark.low_priority),
    pytest.param("src0=src1", marks=pytest.mark.low_priority),
]


@pytest.mark.usefixtures("setup_random_seed")
@pytest.mark.parametrize("op", ARITH_OPS + [pytest.param("div", marks=pytest.mark.low_priority)])
@pytest.mark.parametrize("form", INPLACE_FORMS)
@pytest.mark.parametrize("target", ["ascendc", pytest.param("pto", marks=pytest.mark.low_priority)])
def test_tile_arith_inplace(op, form, target):
    """dst may alias src0 / src1; src0 may alias src1 (separate GM outputs)."""
    run_test_inplace(op, form, 4, 128, "float32", target)


# -----------------------------------------------------------------------------
# Test cases: size validation
# -----------------------------------------------------------------------------


def test_tile_arith_size_mismatch_src0():
    """dst vs src0 size mismatch raises at trace time (Python assert)."""
    with pytest.raises(RuntimeError):
        binary_tensor_mismatch_kernel("float16")


def test_tile_arith_size_mismatch_region():
    """src1 BufferRegion size mismatch raises at trace time (Python assert)."""
    with pytest.raises(RuntimeError):
        binary_region_mismatch_kernel("float16")


@pytest.mark.usefixtures("setup_random_seed")
@pytest.mark.parametrize("target", ["ascendc", pytest.param("pto", marks=pytest.mark.low_priority)])
def test_tile_arith_buffer_size_unvalidated(target):
    """src1 Buffer size is NOT validated: only dst.size elements are used.

    a(64) + b(128): the kernel silently computes a + b[0:64].
    """
    dtype = "float16"
    torch.manual_seed(0)
    a = torch.randn(64, dtype=torch.float16)
    b = torch.rand(128, dtype=torch.float16) + 0.5
    kernel = binary_buffer_mismatch_kernel(dtype)
    func = tilelang.compile(kernel, out_idx=[-1], pass_configs=PASS_CONFIGS, target=target)
    out = func(a.npu(), b.npu())
    torch.npu.synchronize()
    torch.testing.assert_close(out.cpu().float(), (a + b[:64]).float(), rtol=RTOL, atol=ATOL)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="T.tile.add/sub/mul/div test suite")
    parser.add_argument("--target", type=str, choices=["ascendc", "pto"], default="ascendc")
    parser.add_argument("--op", type=str, choices=["add", "sub", "mul", "div"], default="add")
    parser.add_argument("--mode", type=str, choices=["tensor", "scalar", "buffload", "slice", "inplace"], default="tensor")
    args = parser.parse_args()

    torch.manual_seed(0)
    tilelang.cache.clear_cache()

    print("=" * 60)
    print(f"T.tile.{args.op} test: target={args.target}, mode={args.mode}")
    print("=" * 60)

    if args.mode == "tensor":
        run_test_tensor(args.op, 64, 128, "float32", args.target)
    elif args.mode == "scalar":
        run_test_scalar(args.op, 64, 128, 2.0, "float32", args.target)
    elif args.mode == "buffload":
        run_test_buffload(args.op, 64, 128, 8, "float32", args.target)
    elif args.mode == "slice":
        run_test_row_slice(args.op, 4, 128, 4, "float32", args.target)
    else:
        run_test_inplace(args.op, "dst=src0", 4, 128, "float32", args.target)
    print(f"  {args.op} ({args.mode}) PASSED")

    print("\nAll requested cases passed.")
