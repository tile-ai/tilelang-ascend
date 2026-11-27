import importlib.util
import sys
from pathlib import Path
from types import ModuleType


def _run_example() -> ModuleType:
    source = Path(__file__).with_name("opt_gdn_full.py")
    spec = importlib.util.spec_from_file_location("_opt_gdn_full_for_test", source)
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


def test_opt_gdn_full() -> None:
    # This example has no main guard: it builds its tensors, runs the kernel and
    # compares against ref_seq_gdn at module level, so executing the
    # module is running the check and a mismatch surfaces as the AssertionError
    # torch.testing raises.
    _run_example()
