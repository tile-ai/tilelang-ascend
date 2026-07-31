import importlib.util
import sys
from pathlib import Path
from types import ModuleType


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


def test_rope_accuracy() -> None:
    import torch

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

    torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-3)
