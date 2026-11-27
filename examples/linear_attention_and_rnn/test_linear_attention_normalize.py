import importlib.util
import sys
from pathlib import Path
from types import ModuleType


def _run_example() -> ModuleType:
    source = Path(__file__).with_name("linear_attention_normalize.py")
    spec = importlib.util.spec_from_file_location("_linear_attention_normalize_for_test", source)
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


def test_linear_attention_normalize() -> None:
    # This example has no main guard: it builds its tensors, runs the kernel and
    # compares against ref_linear_attention at module level, so executing the
    # module is running the check and a mismatch surfaces as the AssertionError
    # torch.testing raises.
    _run_example()
