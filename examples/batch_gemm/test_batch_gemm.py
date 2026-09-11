import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import torch


def _check_precision(actual, golden, dtype):
    table = {
        "float16": (2**-14, 2**-9, 0.1),
        "bfloat16": (2**-10, 2**-6, 1.0),
        "float32": (2**-16, 2**-10, 0.01),
        "hifloat32": (2**-16, 2**-10, 0.01),
        "float8_e4m3": (2**-4, 2**-2, 1.0),
        "float8_e5m2": (2**-3, 2**-1, 0.1),
    }
    actual, golden = actual.detach().cpu(), golden.detach().cpu()
    if actual.shape != golden.shape:
        return False, 0.0, float("inf")
    name = str(dtype).replace("torch.", "")
    name = "float8_e4m3" if name.startswith("float8_e4m3") else ("float8_e5m2" if name.startswith("float8_e5m2") else name)
    if name in {"int8", "int16", "int32", "int64", "uint8"}:
        mismatch = (actual != golden).sum().item()
        total = max(actual.numel(), 1)
        return mismatch == 0, 1.0 - mismatch / total, 0.0 if mismatch == 0 else float("inf")
    atol, rtol, limit = table.get(name, table["float16"])
    actual, golden = actual.float(), golden.float()
    special = ~torch.isfinite(golden)
    if special.any() and (
        not torch.equal(torch.isnan(actual[special]), torch.isnan(golden[special]))
        or not torch.equal(torch.isinf(actual[special]), torch.isinf(golden[special]))
        or not torch.equal(actual[special][torch.isinf(golden[special])], golden[special][torch.isinf(golden[special])])
    ):
        return False, 0.0, float("inf")
    finite = torch.isfinite(golden)
    if not finite.any():
        return True, 1.0, 0.0
    error = (actual[finite] - golden[finite]).abs()
    error = torch.where(torch.isfinite(error), error, torch.full_like(error, float("inf")))
    ratio, max_abs = (error <= atol + rtol * golden[finite].abs()).float().mean().item(), error.max().item()
    return ratio >= 0.99 and max_abs <= limit, ratio, max_abs


def _load_batch_gemm_example() -> ModuleType:
    source = Path(__file__).with_name("batch_gemm.py")
    spec = importlib.util.spec_from_file_location("_batch_gemm_example_for_test", source)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load example module: {source}")

    module = importlib.util.module_from_spec(spec)
    original_argv = sys.argv
    try:
        # batch_gemm.py parses arguments at import time. Hide Pytest arguments
        # while loading it without changing the original Example.
        sys.argv = [str(source)]
        spec.loader.exec_module(module)
    finally:
        sys.argv = original_argv

    return module


def test_batch_gemm_accuracy() -> None:
    example = _load_batch_gemm_example()

    batch = 8
    m = 1024
    n = 1024
    k = 1024

    kernel = example.batch_matmul(
        batch,
        m,
        n,
        k,
        block_M=128,
        block_N=256,
        K_L1=64,
    )

    torch.manual_seed(0)
    lhs = torch.randn(batch, m, k).half().npu()
    rhs = torch.randn(batch, k, n).half().npu()

    actual = kernel(lhs, rhs)
    expected = torch.matmul(lhs, rhs)

    passed, ratio, max_abs = _check_precision(actual, expected, actual.dtype)
    assert passed, f"dtype={actual.dtype}, matched_ratio={ratio:.4f}, max_abs_error={max_abs:.6e}"
