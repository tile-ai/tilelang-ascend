import importlib.util
import sys
from pathlib import Path
from types import ModuleType
import torch


def _check_precision(actual, golden):
    a, g = actual.detach().cpu(), golden.detach().cpu()
    if a.shape != g.shape:
        return False, 0.0, float("inf")
    if not (a.is_floating_point() or g.is_floating_point()):
        mismatches = (a != g).sum().item()
        return mismatches == 0, 1.0 - mismatches / max(a.numel(), 1), 0.0 if mismatches == 0 else float("inf")
    table = {
        "float16": (2**-14, 2**-9, 1e-1),
        "bfloat16": (2**-10, 2**-6, 1e0),
        "float32": (2**-16, 2**-10, 1e-2),
        "hifloat32": (2**-16, 2**-10, 1e-2),
        "float8_e4m3": (2**-4, 2**-2, 1e0),
        "float8_e4m3fn": (2**-4, 2**-2, 1e0),
        "float8_e5m2": (2**-3, 2**-1, 1e-1),
    }
    atol, rtol, limit = table.get(str(g.dtype).removeprefix("torch."), table["float16"])
    a, g = a.float(), g.float()
    special = ~torch.isfinite(g)
    if special.any() and (
        not torch.equal(torch.isnan(a[special]), torch.isnan(g[special]))
        or not torch.equal(torch.isinf(a[special]), torch.isinf(g[special]))
    ):
        return False, 0.0, float("inf")
    finite = torch.isfinite(g)
    if not finite.any():
        return True, 1.0, 0.0
    err = (a[finite] - g[finite]).abs()
    err = torch.where(torch.isfinite(err), err, torch.full_like(err, float("inf")))
    ratio = (err <= atol + rtol * g[finite].abs()).float().mean().item()
    maximum = err.max().item()
    return ratio >= 0.99 and maximum <= limit, ratio, maximum


def _load_rms_rope_fused_mask_example() -> ModuleType:
    source = Path(__file__).with_name("rms_rope_fused_mask.py")
    spec = importlib.util.spec_from_file_location("_rms_rope_fused_mask_example_for_test", source)
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


def test_rms_rope_fused_mask_accuracy() -> None:
    import torch

    example = _load_rms_rope_fused_mask_example()

    batch_size = 16
    head_num = 64
    head_dim = 512
    rope_dim = 256
    eps = 1e-6

    torch.manual_seed(42)
    example.tilelang.disable_cache()

    dtype = torch.float16
    device = "npu"
    example.device = device

    q = torch.randn((batch_size, head_num, head_dim), device=device, dtype=dtype)
    sin = torch.randn((batch_size, rope_dim), device=device, dtype=dtype)
    cos = torch.randn((batch_size, rope_dim), device=device, dtype=dtype)

    dim_start = head_dim - rope_dim
    expected = example.rms_norm_reference(q, head_dim, eps)
    expected_part = expected[..., dim_start:]
    expected[..., dim_start:] = example.rope_reference(
        expected_part.to(torch.float32),
        cos.to(torch.float32),
        sin.to(torch.float32),
    )

    actual = example.tilelang_rms_rope_fused(q.clone(), sin, cos, eps)

    passed, ratio, max_abs = _check_precision(actual, expected)
    assert passed, f"matched_ratio={ratio:.4f}, max_abs_error={max_abs:.6e}"
