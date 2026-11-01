import importlib.util
import sys
from pathlib import Path
from types import ModuleType
import torch


def _check_precision(actual, golden, dtype):
    table = {
        "float16": (2**-14, 2**-9, 1e-1, 0.99),
        "bfloat16": (2**-10, 2**-6, 1e0, 0.99),
        "float32": (2**-16, 2**-10, 1e-2, 0.99),
        "hifloat32": (2**-16, 2**-10, 1e-2, 0.99),
        "float8_e4m3": (2**-4, 2**-2, 1e0, 0.99),
        "float8_e5m2": (2**-3, 2**-1, 1e-1, 0.99),
    }
    actual, golden = actual.detach().cpu(), golden.detach().cpu()
    if dtype in {"int8", "int16", "int32", "int64", "uint8"}:
        assert torch.equal(actual, golden), "integer output must match exactly"
        return
    atol, rtol, max_abs_limit, required_ratio = table[dtype]
    actual, golden = actual.float(), golden.float()
    special = ~torch.isfinite(golden)
    assert torch.equal(torch.isnan(actual[special]), torch.isnan(golden[special]))
    assert torch.equal(torch.isinf(actual[special]), torch.isinf(golden[special]))
    finite = torch.isfinite(golden)
    if not finite.any():
        return
    errors = (actual[finite] - golden[finite]).abs()
    ratio = (errors <= atol + rtol * golden[finite].abs()).float().mean().item()
    max_abs = errors.max().item()
    assert ratio >= required_ratio and max_abs <= max_abs_limit, f"matched_ratio={ratio:.4f}, max_abs_error={max_abs:.6e}"


def _load_int8_gemm_example() -> ModuleType:
    source = Path(__file__).with_name("int8_gemm.py")
    spec = importlib.util.spec_from_file_location("_int8_gemm_example_for_test", source)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load example module: {source}")

    module = importlib.util.module_from_spec(spec)
    original_argv = sys.argv
    try:
        sys.argv = [str(source)]
        spec.loader.exec_module(module)
    finally:
        sys.argv = original_argv
    return module


def test_int8_gemm_accuracy() -> None:
    example = _load_int8_gemm_example()

    m = 1024
    n = 1024
    k = 1024

    torch.manual_seed(42)
    a_bf16 = torch.randn(m, k, dtype=torch.bfloat16, device="npu")
    a_fp32 = a_bf16.float()
    a_abs_max = torch.max(torch.abs(a_fp32), dim=1, keepdim=True)[0]
    a_abs_max = torch.clamp(a_abs_max, min=1e-4)
    a_scales = a_abs_max / 127.0
    a_scaled = a_fp32 / a_scales
    a_int8 = torch.clamp(a_scaled, -128, 127).round().to(torch.int8)

    b_bf16 = torch.randn(n, k, dtype=torch.bfloat16, device="npu")
    b_fp32 = b_bf16.float()
    b_abs_max = torch.max(torch.abs(b_fp32), dim=1, keepdim=True)[0]
    b_abs_max = torch.clamp(b_abs_max, min=1e-4)
    b_scales = b_abs_max / 127.0
    b_scaled = b_fp32 / b_scales
    b_int8 = torch.clamp(b_scaled, -128, 127).round().to(torch.int8)

    kernel = example.int8_gemm_kernel_corrected(
        n,
        k,
        block_M=64,
        block_N=64,
        block_K=64,
        out_dtype=example.BF16,
    )

    actual = kernel(a_int8, b_int8, a_scales, b_scales)
    torch.npu.synchronize()

    expected = example.int8_gemm_torch_optimized(
        a_int8,
        a_scales,
        b_int8,
        b_scales,
        out_dtype=torch.bfloat16,
    )
    torch.npu.synchronize()

    _check_precision(actual, expected, "bfloat16")
