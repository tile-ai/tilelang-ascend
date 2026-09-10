import importlib.util
from pathlib import Path
from types import ModuleType

import pytest
import torch


def _load_example() -> ModuleType:
    source = Path(__file__).with_name("example_topk_selector.py")
    spec = importlib.util.spec_from_file_location("_example_topk_selector_for_test", source)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load example module: {source}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("batch", "n", "top_k"),
    [
        (4, 4 * 1024, 2048),
        (1024, 1024, 128),
    ],
    ids=["wide_rows_topk2048", "many_rows_topk128"],
)
def test_topk_selector_accuracy(batch: int, n: int, top_k: int) -> None:
    example = _load_example()

    # Both cases are the ones the example runs itself. check_case compares the
    # selected indices against torch.topk and asserts the agreement is above
    # 99%, which is the tolerance the kernel is written to: it selects by a
    # threshold rather than a full sort, so ties near the cut can land either
    # way. The example's seed is kept so the draw is the one it was tuned on.
    torch.manual_seed(0)
    example.check_case(batch, n, top_k)
