import importlib.util
import sys
from pathlib import Path
from types import ModuleType


def _run_example() -> ModuleType:
    source = Path(__file__).with_name("example_sparse_flash_attn_mask_pa.py")
    spec = importlib.util.spec_from_file_location("_example_sparse_flash_attn_mask_pa_for_test", source)
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


def test_sparse_flash_attn_mask_pa() -> None:
    import torch

    # This example has no main guard: it builds its tensors, runs the kernel and
    # compares against ref_sparse_attention_fwd_interface at module level, so
    # executing the module is running the check and a mismatch surfaces as the
    # AssertionError that torch.testing raises. It also sets the default device
    # while doing so, which is restored here because a test shares its process.
    previous = torch.get_default_device()
    try:
        _run_example()
    finally:
        torch.set_default_device(previous)
