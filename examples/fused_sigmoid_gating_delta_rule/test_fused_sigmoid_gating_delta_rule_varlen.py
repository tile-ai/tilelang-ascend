import importlib.util
import sys
from pathlib import Path
from types import ModuleType


def _load_example() -> ModuleType:
    source = Path(__file__).with_name("fused_sigmoid_gating_delta_rule_varlen.py")
    spec = importlib.util.spec_from_file_location("_fused_sigmoid_gating_example_for_test", source)
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


# main() takes its shapes as parameters and asserts the output and the final
# state against golden(), so this drives it directly. The example's own guard
# wraps the same call in a thread pool, which only matters when several cases
# run at once and there is one.
def test_fused_sigmoid_gating_delta_rule_varlen() -> None:
    example = _load_example()

    # A hundred short sequences of alternating length, which is what makes this
    # the varlen case rather than a padded batch.
    example.main(seqlens=[4, 8] * 50, nk=16, nv=32, dk=128, dv=128)
