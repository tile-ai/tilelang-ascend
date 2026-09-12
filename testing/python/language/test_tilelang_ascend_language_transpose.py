import argparse

import pytest
import tilelang
import tilelang.language as T
import torch

tir = tilelang.tvm.tir

PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}

TORCH_DTYPE = {
    "float16": torch.float16,
    "float32": torch.float32,
    "int16": torch.int16,
    "int32": torch.int32,
    "uint16": torch.uint16,
    "uint32": torch.uint32,
    "int8": torch.int8,
    "bfloat16": torch.bfloat16,
    "int64": torch.int64,
}


def _assert_close(actual, expected, dtype, rtol=1e-2, atol=1e-2):
    if dtype in ("uint16", "uint32"):
        cast = "int16" if dtype == "uint16" else "int32"
        torch.testing.assert_close(
            actual.to(getattr(torch, cast)),
            expected.to(getattr(torch, cast)),
            rtol=rtol,
            atol=atol,
        )
    else:
        torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)


def transpose_kernel(M, N, dtype="float16"):
    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),  # type: ignore
        B: T.Tensor((N, M), dtype),  # type: ignore
    ):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            a_ub = T.alloc_ub((M, N), dtype)
            b_ub = T.alloc_ub((N, M), dtype)

            T.copy(A, a_ub)
            T.tile.transpose(b_ub, a_ub)
            T.copy(b_ub, B)

    return main


def transpose_inplace_kernel(M, dtype="float16"):
    @T.prim_func
    def main(
        A: T.Tensor((M, M), dtype),  # type: ignore
        B: T.Tensor((M, M), dtype),  # type: ignore
    ):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            a_ub = T.alloc_ub((M, M), dtype)

            T.copy(A, a_ub)
            T.tile.transpose(a_ub, a_ub)
            T.copy(a_ub, B)

    return main


def run_test_transpose(M, N, dtype, target):
    torch.manual_seed(0)
    tilelang.cache.clear_cache()

    func = tilelang.compile(
        transpose_kernel(M, N, dtype),
        out_idx=[-1],
        pass_configs=PASS_CONFIGS,
        target=target,
    )

    torch_dtype = TORCH_DTYPE[dtype]
    if dtype in ("int8", "int16", "int32", "uint16", "uint32", "int64"):
        lo = -100 if dtype in ("int8", "int16", "int32", "int64") else 0
        hi = 100 if dtype in ("int8", "int16", "int32", "int64") else 200
        a = torch.randint(lo, hi, (M, N), dtype=torch_dtype).npu()
    else:
        a = torch.randn(M, N, dtype=torch_dtype).npu()

    torch.npu.synchronize()
    b = func(a)
    ref_b = a.T.contiguous()
    _assert_close(b, ref_b, dtype)


# -----------------------------------------------------------------------------
# 16x16 hardware-instruction path (AscendC::Transpose)
# -----------------------------------------------------------------------------
transpose_dtype_target_params = [
    pytest.param("int16", "ascendc", marks=pytest.mark.low_priority),
    pytest.param("int16", "pto", marks=pytest.mark.low_priority),
    pytest.param("uint16", "ascendc", marks=pytest.mark.low_priority),
    pytest.param("uint16", "pto", marks=pytest.mark.low_priority),
    ("float16", "ascendc"),
    pytest.param("float16", "pto", marks=pytest.mark.low_priority),
    pytest.param("int32", "ascendc", marks=pytest.mark.low_priority),
    pytest.param("int32", "pto", marks=pytest.mark.low_priority),
    pytest.param("uint32", "ascendc", marks=pytest.mark.low_priority),
    pytest.param("uint32", "pto", marks=pytest.mark.low_priority),
    pytest.param("float32", "ascendc", marks=pytest.mark.low_priority),
    pytest.param("float32", "pto", marks=pytest.mark.low_priority),
]


@pytest.mark.parametrize("dtype,target", transpose_dtype_target_params)
@pytest.mark.parametrize("shape", [(16, 16)])
def test_transpose_16x16(dtype, target, shape):
    M, N = shape
    run_test_transpose(M, N, dtype, target)


# -----------------------------------------------------------------------------
# Block-transpose path (TransDataTo5HD): B16, H/W multiples of 16, non-16x16
# -----------------------------------------------------------------------------
@pytest.mark.parametrize(
    "dtype",
    ["float16", pytest.param("int16", marks=pytest.mark.low_priority), pytest.param("uint16", marks=pytest.mark.low_priority)],
)
@pytest.mark.parametrize("target", ["ascendc", pytest.param("pto", marks=pytest.mark.low_priority)])
@pytest.mark.parametrize(
    "shape",
    [
        (32, 32),
        pytest.param((64, 64), marks=pytest.mark.low_priority),
        pytest.param((32, 16), marks=pytest.mark.low_priority),
        pytest.param((16, 32), marks=pytest.mark.low_priority),
        pytest.param((64, 32), marks=pytest.mark.low_priority),
        pytest.param((32, 64), marks=pytest.mark.low_priority),
        pytest.param((48, 48), marks=pytest.mark.low_priority),
        pytest.param((16, 48), marks=pytest.mark.low_priority),
        pytest.param((48, 16), marks=pytest.mark.low_priority),
        pytest.param((128, 128), marks=pytest.mark.low_priority),
    ],
)
def test_transpose_block_b16(dtype, target, shape):
    M, N = shape
    run_test_transpose(M, N, dtype, target)


# -----------------------------------------------------------------------------
# Block-transpose path (TransDataTo5HD): B32, H/W multiples of 16
# -----------------------------------------------------------------------------
@pytest.mark.parametrize(
    "dtype",
    ["float32", pytest.param("int32", marks=pytest.mark.low_priority), pytest.param("uint32", marks=pytest.mark.low_priority)],
)
@pytest.mark.parametrize("target", ["ascendc", pytest.param("pto", marks=pytest.mark.low_priority)])
@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((16, 16), marks=pytest.mark.low_priority),
        (32, 32),
        pytest.param((32, 16), marks=pytest.mark.low_priority),
        pytest.param((16, 32), marks=pytest.mark.low_priority),
        pytest.param((48, 48), marks=pytest.mark.low_priority),
        pytest.param((64, 64), marks=pytest.mark.low_priority),
        pytest.param((64, 32), marks=pytest.mark.low_priority),
        pytest.param((32, 64), marks=pytest.mark.low_priority),
    ],
)
def test_transpose_block_b32(dtype, target, shape):
    M, N = shape
    run_test_transpose(M, N, dtype, target)


# -----------------------------------------------------------------------------
# B32 scalar-fallback path: H/W multiples of 8 but not 16 (e.g. 8, 24)
# -----------------------------------------------------------------------------
@pytest.mark.parametrize(
    "dtype",
    ["float32", pytest.param("int32", marks=pytest.mark.low_priority), pytest.param("uint32", marks=pytest.mark.low_priority)],
)
@pytest.mark.parametrize("target", ["ascendc", pytest.param("pto", marks=pytest.mark.low_priority)])
@pytest.mark.parametrize(
    "shape",
    [
        (16, 8),
        pytest.param((8, 16), marks=pytest.mark.low_priority),
        pytest.param((24, 24), marks=pytest.mark.low_priority),
    ],
)
def test_transpose_b32_scalar_fallback(dtype, target, shape):
    M, N = shape
    run_test_transpose(M, N, dtype, target)


# -----------------------------------------------------------------------------
# Scalar fallback path: int8 (32-byte aligned => H/W multiples of 32)
# -----------------------------------------------------------------------------
@pytest.mark.parametrize("dtype", ["int8"])
@pytest.mark.parametrize("target", ["ascendc", pytest.param("pto", marks=pytest.mark.low_priority)])
@pytest.mark.parametrize(
    "shape",
    [
        (32, 32),
        pytest.param((64, 64), marks=pytest.mark.low_priority),
        pytest.param((32, 64), marks=pytest.mark.low_priority),
    ],
)
def test_transpose_fallback_int8(dtype, target, shape):
    M, N = shape
    run_test_transpose(M, N, dtype, target)


# -----------------------------------------------------------------------------
# Scalar fallback path: bfloat16 (32-byte aligned => H/W multiples of 16)
# -----------------------------------------------------------------------------
@pytest.mark.parametrize("dtype", ["bfloat16"])
@pytest.mark.parametrize("target", ["ascendc", pytest.param("pto", marks=pytest.mark.low_priority)])
@pytest.mark.parametrize(
    "shape",
    [
        (16, 16),
        pytest.param((32, 32), marks=pytest.mark.low_priority),
        pytest.param((16, 32), marks=pytest.mark.low_priority),
    ],
)
def test_transpose_fallback_bfloat16(dtype, target, shape):
    M, N = shape
    run_test_transpose(M, N, dtype, target)


# -----------------------------------------------------------------------------
# Scalar fallback path: int64 (ascendc only; pto TTRANS does not support B64)
# -----------------------------------------------------------------------------
@pytest.mark.low_priority
@pytest.mark.parametrize("dtype", ["int64"])
@pytest.mark.parametrize("target", ["ascendc", pytest.param("pto", marks=pytest.mark.ci_skip)])
@pytest.mark.parametrize("shape", [(16, 16), (32, 32)])
def test_transpose_fallback_int64(dtype, target, shape):
    M, N = shape
    run_test_transpose(M, N, dtype, target)


# -----------------------------------------------------------------------------
# Non-aligned shape raises ValueError at compile time
# -----------------------------------------------------------------------------
@pytest.mark.parametrize(
    "shape,dtype",
    [
        pytest.param((20, 36), "float16", id="20x36-f16", marks=pytest.mark.low_priority),
        pytest.param((17, 33), "float16", id="17x33-f16", marks=pytest.mark.low_priority),
        pytest.param((24, 40), "float16", id="24x40-f16", marks=pytest.mark.low_priority),
        pytest.param((16, 33), "float16", id="16x33-f16", marks=pytest.mark.low_priority),
        pytest.param((33, 16), "float16", id="33x16-f16", marks=pytest.mark.low_priority),
        pytest.param((16, 16), "int8", id="16x16-i8", marks=pytest.mark.low_priority),
        pytest.param((17, 17), "float32", id="17x17-f32"),
    ],
)
def test_transpose_non_aligned_raises(shape, dtype):
    M, N = shape
    src = tir.decl_buffer((M, N), dtype)
    dst = tir.decl_buffer((N, M), dtype)
    with pytest.raises(ValueError, match="32-byte alignment"):
        T.tile.transpose(dst, src)


# -----------------------------------------------------------------------------
# In-place transpose (dst == src): confirms doc constraint 5 — unsupported.
# Most dispatch paths (transpose_block, scalar) produce wrong results when
# dst == src. The 16x16 B16 AscendC::Transpose path may coincidentally pass,
# so xfail is non-strict.
# -----------------------------------------------------------------------------
@pytest.mark.low_priority
@pytest.mark.xfail(strict=False, reason="in-place transpose unsupported per doc constraint 5")
@pytest.mark.parametrize("dtype", ["float16", "float32", "int8"])
@pytest.mark.parametrize("target", ["ascendc"])
@pytest.mark.parametrize("shape", [(16, 16), (32, 32)])
def test_transpose_inplace_unsupported(dtype, target, shape):
    M, _ = shape
    torch.manual_seed(0)
    tilelang.cache.clear_cache()

    func = tilelang.compile(
        transpose_inplace_kernel(M, dtype),
        out_idx=[-1],
        pass_configs=PASS_CONFIGS,
        target=target,
    )

    torch_dtype = TORCH_DTYPE[dtype]
    if dtype == "int8":
        a = torch.randint(-100, 100, (M, M), dtype=torch_dtype).npu()
    else:
        a = torch.randn(M, M, dtype=torch_dtype).npu()

    torch.npu.synchronize()
    b = func(a)
    ref_b = a.T.contiguous()
    _assert_close(b, ref_b, dtype)


# -----------------------------------------------------------------------------
# Standalone command-line entry point
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=32)
    parser.add_argument("--n", type=int, default=32)
    parser.add_argument("--dtype", type=str, default="float16")
    parser.add_argument("--target", type=str, choices=["ascendc", "pto"], default="ascendc")
    args = parser.parse_args()

    run_test_transpose(args.m, args.n, args.dtype, args.target)
