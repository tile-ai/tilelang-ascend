import importlib.util
import sys
from pathlib import Path
from types import ModuleType


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

    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)
