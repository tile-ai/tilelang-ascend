import importlib.util
from pathlib import Path
from types import ModuleType

import pytest


def _load_example() -> ModuleType:
    source = Path(__file__).with_name("mtgr_ragged_segment_attention.py")
    spec = importlib.util.spec_from_file_location("_mtgr_ragged_segment_attention_for_test", source)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load example module: {source}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# A ragged batch: the segment lengths of each request, the rule each segment is
# scored under, and how much of the request was already cached. The first is
# the example's own; the second is the same shape of problem an order of
# magnitude smaller, which is enough to catch a kernel that mis-indexes a
# segment boundary without paying for the long one twice.
CONFIGS = {
    "example": {
        "H": 8,
        "D": 128,
        "seg_lengths": [[1600, 8] + [5] * 1 + [1200]],
        "rules": [0, 1] + [2] * 1 + [2],
        "matched_prefix_arr": [0],
    },
    "short_segments": {
        "H": 8,
        "D": 128,
        "seg_lengths": [[256, 8, 5, 128]],
        "rules": [0, 1, 2, 2],
        "matched_prefix_arr": [0],
    },
}


@pytest.mark.parametrize("case", sorted(CONFIGS), ids=sorted(CONFIGS))
def test_mtgr_ragged_segment_attention_accuracy(case: str) -> None:
    # No default device is set here, unlike the tests of the examples that build
    # their tensors without naming one: this example moves each of its own to
    # the accelerator and keeps the mask on the host, so a default would put the
    # two on different devices. The example's __main__ does not set one either.
    _load_example().test(CONFIGS[case])
