import importlib.util
import sys
from pathlib import Path
from types import ModuleType
import torch


def _load_rope_example() -> ModuleType:
    source = Path(__file__).with_name("rope.py")
    spec = importlib.util.spec_from_file_location("_pos_rope_example_for_test", source)
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


def _check_precision(actual, golden, dtype):
    table = {
        "float16": (2**-14, 2**-9, 1e-1),
        "bfloat16": (2**-10, 2**-6, 1e0),
        "float32": (2**-16, 2**-10, 1e-2),
        "hifloat32": (2**-16, 2**-10, 1e-2),
        "float8_e4m3": (2**-4, 2**-2, 1e0),
        "float8_e4m3fn": (2**-4, 2**-2, 1e0),
        "float8_e5m2": (2**-3, 2**-1, 1e-1),
    }
    actual, golden = actual.detach().cpu(), golden.detach().cpu()
    if actual.shape != golden.shape:
        return False, 0.0, float("inf")
    if not (actual.is_floating_point() or golden.is_floating_point()):
        mismatches = (actual != golden).sum().item()
        return mismatches == 0, 1.0 - mismatches / max(actual.numel(), 1), 0.0 if mismatches == 0 else float("inf")
    atol, rtol, limit = table.get(str(dtype).replace("torch.", ""), table["float16"])
    actual, golden = actual.float(), golden.float()
    special = ~torch.isfinite(golden)
    if special.any() and (
        not torch.equal(torch.isnan(actual[special]), torch.isnan(golden[special]))
        or not torch.equal(torch.isinf(actual[special]), torch.isinf(golden[special]))
    ):
        return False, 0.0, float("inf")
    finite = torch.isfinite(golden)
    if not finite.any():
        return True, 1.0, 0.0
    error = (actual[finite] - golden[finite]).abs()
    error = torch.where(torch.isfinite(error), error, torch.full_like(error, float("inf")))
    ratio, maximum = (error <= atol + rtol * golden[finite].abs()).float().mean().item(), error.max().item()
    return ratio >= 0.99 and maximum <= limit, ratio, maximum


def test_rope_accuracy() -> None:

    example = _load_rope_example()

    batch_size = 16
    head_num = 64
    hidden_size = 512
    rope_dim = 256

    torch.manual_seed(42)
    example.tilelang.disable_cache()

    dtype = torch.float16
    device = "npu"

    x = torch.randn((batch_size, head_num, hidden_size), device=device, dtype=dtype)
    sin = torch.randn((batch_size, rope_dim), device=device, dtype=dtype)
    cos = torch.randn((batch_size, rope_dim), device=device, dtype=dtype)

    dim_start = hidden_size - rope_dim
    expected = x.clone()
    expected_part = expected[..., dim_start:]
    expected[..., dim_start:] = example.torch_rope_ref(
        expected_part.to(torch.float32),
        sin.to(torch.float32),
        cos.to(torch.float32),
    )

    actual = x.clone()
    example.tilelang_apply_rope_partial_in_place(actual, sin, cos)

    passed, ratio, max_abs = _check_precision(actual, expected, actual.dtype)
    assert passed, f"dtype={actual.dtype}, matched_ratio={ratio:.4f}, max_abs_error={max_abs:.6e}"
