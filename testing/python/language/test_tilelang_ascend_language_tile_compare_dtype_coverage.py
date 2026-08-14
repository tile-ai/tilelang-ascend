"""
T.tile.compare dtype coverage tests.

Existing coverage in test_tilelang_ascend_language_elementwise.py:
- float/float16 x ascendc/pto x LT mode x 2D (tensor-tensor, tensor-scalar, with slice variants)
- out_dtype: int8/uint8

This file supplements with:
1. 1D tensor-tensor path (float32/float16 x ascendc/pto x LT)
2. 1D tensor-scalar path (float32/float16 x ascendc/pto x LT)
3. Exception boundary tests (dtype mismatch, invalid mode, non-256-byte alignment)

Known limitations (documented in docs/api_docs/T.tile.compare.md):
- int32 x non-EQ modes: compiles but produces incorrect output (CANN vcmpv int32 limitation)
- bfloat16/int8/uint8/int16/uint16/uint32: not supported (CANN vcmpv only accepts half/float/int32)
- Shape mismatch between src0 and src1: not validated at compile time (no error raised)
- dst dtype != uint8: not validated at compile time (produces incorrect results)
"""

import pytest
import tilelang
import tilelang.language as T
import torch


@pytest.fixture(scope="module", autouse=True)
def _disable_cache():
    tilelang.disable_cache()
    yield
    tilelang.enable_cache()


PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}

TORCH_DTYPE_MAP = {
    "float16": torch.float16,
    "float32": torch.float32,
}

MODE_TORCH_MAP = {
    "EQ": torch.eq,
    "NE": torch.ne,
    "GT": torch.gt,
    "GE": torch.ge,
    "LT": torch.lt,
    "LE": torch.le,
}

N = 256  # elements; 256-byte aligned for all supported dtypes


def _gen_input_1d(dtype, n=N):
    td = TORCH_DTYPE_MAP[dtype]
    a = (torch.arange(n, dtype=torch.float32) * 0.1).to(td)
    b = (torch.arange(n, dtype=torch.float32) * 0.1 + 0.05).to(td)
    return a.npu(), b.npu()


def _ref_bitpack(mask_flat, n_elements):
    """Pack boolean mask into uint8 bytes (LSB-first within each byte)."""
    n_bytes = n_elements // 8
    bits = mask_flat[:n_elements].to(torch.uint8).reshape(n_bytes, 8)
    weights = torch.tensor([1, 2, 4, 8, 16, 32, 64, 128], dtype=torch.uint8)
    return (bits * weights).sum(dim=1, dtype=torch.uint8)


# ---------------------------------------------------------------------------
# Functional: 1D tensor-tensor (float32/float16 x ascendc/pto x LT)
# ---------------------------------------------------------------------------

_DTYPE_TENSOR_1D = [
    "float32",
    pytest.param("float16", marks=pytest.mark.low_priority),
]


@pytest.mark.parametrize("dtype", _DTYPE_TENSOR_1D)
@pytest.mark.parametrize(
    "target",
    ["ascendc", pytest.param("pto", marks=pytest.mark.low_priority)],
)
def test_compare_tensor_1d(dtype, target):
    mode = "LT"

    @T.prim_func
    def main(
        A: T.Tensor((N,), dtype),  # type: ignore
        B: T.Tensor((N,), dtype),  # type: ignore
        C: T.Tensor((N // 8,), "uint8"),  # type: ignore
    ):
        with T.Kernel(1, is_npu=True) as (_, vid):
            if vid == 0:
                a_ub = T.alloc_ub((N,), dtype)
                b_ub = T.alloc_ub((N,), dtype)
                c_ub = T.alloc_ub((N // 8,), "uint8")
                T.copy(A, a_ub)
                T.copy(B, b_ub)
                T.tile.compare(c_ub, a_ub, b_ub, mode)
                T.copy(c_ub, C)

    kernel = tilelang.compile(main, out_idx=[-1], pass_configs=PASS_CONFIGS, target=target)
    a, b = _gen_input_1d(dtype)
    c = kernel(a, b)
    torch.npu.synchronize()
    mask = MODE_TORCH_MAP[mode](a.cpu(), b.cpu())
    ref = _ref_bitpack(mask, N)
    torch.testing.assert_close(c.cpu(), ref, rtol=0, atol=0)


# ---------------------------------------------------------------------------
# Functional: 1D tensor-scalar (float32/float16 x ascendc/pto x LT)
# ---------------------------------------------------------------------------

_DTYPE_SCALAR_1D = [
    "float32",
    pytest.param("float16", marks=pytest.mark.low_priority),
]


@pytest.mark.parametrize("dtype", _DTYPE_SCALAR_1D)
@pytest.mark.parametrize(
    "target",
    ["ascendc", pytest.param("pto", marks=pytest.mark.low_priority)],
)
def test_compare_scalar_1d(dtype, target):
    mode = "LT"
    scalar_val = 0.5

    @T.prim_func
    def main(
        A: T.Tensor((N,), dtype),  # type: ignore
        C: T.Tensor((N // 8,), "uint8"),  # type: ignore
    ):
        with T.Kernel(1, is_npu=True) as (_, vid):
            if vid == 0:
                a_ub = T.alloc_ub((N,), dtype)
                c_ub = T.alloc_ub((N // 8,), "uint8")
                T.copy(A, a_ub)
                T.tile.compare(c_ub, a_ub, scalar_val, mode)
                T.copy(c_ub, C)

    kernel = tilelang.compile(main, out_idx=[-1], pass_configs=PASS_CONFIGS, target=target)
    a, _ = _gen_input_1d(dtype)
    c = kernel(a)
    torch.npu.synchronize()
    td = TORCH_DTYPE_MAP[dtype]
    scalar_t = torch.tensor(scalar_val, dtype=td)
    mask = MODE_TORCH_MAP[mode](a.cpu(), scalar_t)
    ref = _ref_bitpack(mask, N)
    torch.testing.assert_close(c.cpu(), ref, rtol=0, atol=0)


# ---------------------------------------------------------------------------
# Exception boundary: dtype mismatch (default — no LP)
# ---------------------------------------------------------------------------


def test_compare_dtype_mismatch_raises():
    """src0 and src1 must have the same dtype; mismatch should fail at compile time."""

    @T.prim_func
    def main(
        A: T.Tensor((N,), "float16"),  # type: ignore
        B: T.Tensor((N,), "float32"),  # type: ignore
        C: T.Tensor((N // 8,), "uint8"),  # type: ignore
    ):
        with T.Kernel(1, is_npu=True) as (_, vid):
            if vid == 0:
                a_ub = T.alloc_ub((N,), "float16")
                b_ub = T.alloc_ub((N,), "float32")
                c_ub = T.alloc_ub((N // 8,), "uint8")
                T.copy(A, a_ub)
                T.copy(B, b_ub)
                T.tile.compare(c_ub, a_ub, b_ub, "LT")
                T.copy(c_ub, C)

    with pytest.raises(RuntimeError, match="Compilation Failed"):  # noqa: B017
        tilelang.compile(main, out_idx=[-1], pass_configs=PASS_CONFIGS, target="ascendc")


# ---------------------------------------------------------------------------
# Exception boundary: invalid mode (default — no LP)
# ---------------------------------------------------------------------------


def test_compare_invalid_mode_raises():
    """mode must be one of EQ/NE/GT/GE/LT/LE; invalid mode should raise during TIR construction."""

    with pytest.raises(RuntimeError, match="DiagnosticError"):  # noqa: B017

        @T.prim_func
        def main(
            A: T.Tensor((N,), "float32"),  # type: ignore
            B: T.Tensor((N,), "float32"),  # type: ignore
            C: T.Tensor((N // 8,), "uint8"),  # type: ignore
        ):
            with T.Kernel(1, is_npu=True) as (_, vid):
                if vid == 0:
                    a_ub = T.alloc_ub((N,), "float32")
                    b_ub = T.alloc_ub((N,), "float32")
                    c_ub = T.alloc_ub((N // 8,), "uint8")
                    T.copy(A, a_ub)
                    T.copy(B, b_ub)
                    T.tile.compare(c_ub, a_ub, b_ub, "XX")
                    T.copy(c_ub, C)


# ---------------------------------------------------------------------------
# Exception boundary: non-256-byte alignment (default — no LP)
# ---------------------------------------------------------------------------


def test_compare_alignment_raises():
    """src0 total bytes must be 256-byte aligned; non-aligned should fail at compile time."""

    @T.prim_func
    def main(
        A: T.Tensor((48,), "float32"),  # type: ignore
        B: T.Tensor((48,), "float32"),  # type: ignore
        C: T.Tensor((6,), "uint8"),  # type: ignore
    ):
        with T.Kernel(1, is_npu=True) as (_, vid):
            if vid == 0:
                a_ub = T.alloc_ub((48,), "float32")
                b_ub = T.alloc_ub((48,), "float32")
                c_ub = T.alloc_ub((6,), "uint8")
                T.copy(A, a_ub)
                T.copy(B, b_ub)
                T.tile.compare(c_ub, a_ub, b_ub, "LT")
                T.copy(c_ub, C)

    with pytest.raises(RuntimeError, match="alignment"):  # noqa: B017
        tilelang.compile(main, out_idx=[-1], pass_configs=PASS_CONFIGS, target="ascendc")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-n", "8"])
