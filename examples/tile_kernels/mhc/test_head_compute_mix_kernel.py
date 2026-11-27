import importlib.util
import sys
from pathlib import Path
from types import ModuleType


def _load_example() -> ModuleType:
    source = Path(__file__).with_name("head_compute_mix_kernel.py")
    spec = importlib.util.spec_from_file_location("_mhc_head_compute_mix_example_for_test", source)
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


# The example asserts against mhc_head_compute_mix_ref on the way forward and
# against autograd on the way back, and it places every tensor explicitly, so
# these call it rather than restate the reshape arithmetic that surrounds the
# kernel. Loading the example under a private module name keeps Pytest from
# collecting its functions twice.
def test_head_compute_mix_forward() -> None:
    example = _load_example()
    example.test_fwd()


def test_head_compute_mix_backward() -> None:
    example = _load_example()
    example.test_bwd()
