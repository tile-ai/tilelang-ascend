import importlib.util
import sys
from pathlib import Path
from types import ModuleType


def _load_rms_norm_example() -> ModuleType:
    source = Path(__file__).with_name("rms_norm.py")
    spec = importlib.util.spec_from_file_location("_rms_norm_example_for_test", source)
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


def test_rms_norm_accuracy() -> None:
    import torch

    example = _load_rms_norm_example()

    variance_epsilon = 1e-6

    torch.manual_seed(0)
    q = torch.randn(16, 64, 512, dtype=torch.float16, device="npu")

    actual = example.tilelang_q_rms(q, variance_epsilon)
    expected = example.rms_norm_reference(q, variance_epsilon)

    torch.testing.assert_close(actual.cpu(), expected.cpu(), rtol=1e-2, atol=1e-2)
