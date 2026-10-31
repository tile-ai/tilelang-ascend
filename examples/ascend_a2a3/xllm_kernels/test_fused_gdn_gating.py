import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest


FUSED_GDN_GATING_NUM_BATCHES = [
    pytest.param(1, id="batch1"),
    pytest.param(16, id="batch16"),
    pytest.param(275, id="batch275"),
    pytest.param(4096, id="batch4096"),
    pytest.param(65536, id="batch65536"),
]

FUSED_GDN_GATING_NUM_HEADS = [
    pytest.param(4, id="heads4"),
    pytest.param(8, id="heads8"),
    pytest.param(16, id="heads16"),
    pytest.param(24, id="heads24"),
]


def _load_fused_gdn_gating_example() -> ModuleType:
    source = Path(__file__).with_name("fused_gdn_gating.py")
    spec = importlib.util.spec_from_file_location("_fused_gdn_gating_example_for_test", source)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load example module: {source}")

    module = importlib.util.module_from_spec(spec)
    original_argv = sys.argv
    original_sys_path = list(sys.path)
    try:
        sys.argv = [str(source)]
        sys.path.insert(0, str(source.parent))
        sys.path.insert(0, str(source.parents[2]))
        spec.loader.exec_module(module)
    finally:
        sys.argv = original_argv
        sys.path[:] = original_sys_path
    return module


@pytest.mark.parametrize("num_batches", FUSED_GDN_GATING_NUM_BATCHES)
@pytest.mark.parametrize("num_heads", FUSED_GDN_GATING_NUM_HEADS)
def test_fused_gdn_gating_accuracy(num_batches: int, num_heads: int) -> None:
    example = _load_fused_gdn_gating_example()

    example._run_ref_check(
        num_batches=num_batches,
        num_heads=num_heads,
        compile_max_batch=example.DEFAULT_MAX_BATCH,
        softplus_beta=1.0,
        softplus_threshold=20.0,
    )
