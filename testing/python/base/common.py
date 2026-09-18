"""
Shared test utilities for TileLang-Ascend API tests.

Import from test files:
    from base import TOLERANCE, DTYPE_MAP, DEFAULT_PASS_CONFIGS, assert_close_npu, make_test_data
"""

import torch
import tilelang


# ---------------------------------------------------------------------------
# Precision tolerance table — graded by dtype (NOT a flat 1e-2 for all)
# ---------------------------------------------------------------------------
TOLERANCE = {
    "float32": {"rtol": 1e-5, "atol": 1e-5},
    "float": {"rtol": 1e-5, "atol": 1e-5},
    "float16": {"rtol": 1e-3, "atol": 1e-3},
    "bfloat16": {"rtol": 7.8e-3, "atol": 7.8e-3},
    "int8": {"rtol": 0, "atol": 0},
    "int16": {"rtol": 0, "atol": 0},
    "int32": {"rtol": 0, "atol": 0},
    "uint8": {"rtol": 0, "atol": 0},
    "uint16": {"rtol": 0, "atol": 0},
    "uint32": {"rtol": 0, "atol": 0},
}


# ---------------------------------------------------------------------------
# TileLang dtype string  →  torch dtype
# ---------------------------------------------------------------------------
DTYPE_MAP = {
    "float": torch.float32,
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "int8": torch.int8,
    "int16": torch.int16,
    "int32": torch.int32,
    "uint8": torch.uint8,
    "uint16": torch.uint16,
    "uint32": torch.uint32,
}


# ---------------------------------------------------------------------------
# Assertion helper — handles uint16/uint32 that torch-npu can't compare
# ---------------------------------------------------------------------------
def assert_close_npu(actual, expected, dtype, **override):
    """Compare NPU output with golden reference using dtype-graded tolerance.

    Usage:
        assert_close_npu(result, expected, "float16")
        assert_close_npu(result, expected, "float32", rtol=1e-4)  # override
    """
    tol = {**TOLERANCE.get(dtype, {"rtol": 1e-2, "atol": 1e-2}), **override}
    tol.setdefault("equal_nan", True)
    # torch-npu doesn't support isclose for uint16/uint32
    if dtype in ("uint16", "uint32"):
        int_dtype = getattr(torch, dtype.replace("uint", "int"))
        actual = actual.to(int_dtype)
        expected = expected.to(int_dtype)
    torch.testing.assert_close(actual, expected, **tol)


# ---------------------------------------------------------------------------
# Test data generator — integers for int types, randn for float types
# ---------------------------------------------------------------------------
def make_test_data(shape, dtype, *, low=-100, high=100, device="npu"):
    """Generate random test data appropriate for the dtype.

    - Integer types: uniform random in [low, high]
    - Float types: standard normal
    """
    torch_dtype = DTYPE_MAP[dtype]
    if dtype in ("int8", "int16", "int32", "uint8", "uint16", "uint32"):
        return torch.randint(low, high, shape, dtype=torch_dtype, device=device)
    return torch.randn(shape, dtype=torch_dtype, device=device)


# ---------------------------------------------------------------------------
# Default pass_configs for Developer mode tests
# ---------------------------------------------------------------------------
DEFAULT_PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}


def skip_if_missing(op, attr):
    import pytest

    if not getattr(op, attr):
        pytest.skip(f"No {attr}")
