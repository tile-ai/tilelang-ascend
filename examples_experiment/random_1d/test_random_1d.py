import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest
import torch


def _load_random_1d_example() -> ModuleType:
    source = Path(__file__).with_name("random_1d.py")
    spec = importlib.util.spec_from_file_location("_random_1d_example_for_test", source)
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


@pytest.fixture(scope="module")
def random_1d_example():
    return _load_random_1d_example()


_CONFIGS = [
    (1024, 42),
    (256, 42),
    (1024, 0),
    (500, 42),
    (100, 123),
]


@pytest.mark.skipif(
    not hasattr(torch, "npu") or not torch.npu.is_available(),
    reason="requires an Ascend NPU",
)
@pytest.mark.parametrize("M, seed", _CONFIGS)
def test_random_1d_accuracy(random_1d_example, M, seed):
    func = random_1d_example.random_1d(
        M,
        random_1d_example.BLOCK_SIZE,
        seed,
        random_1d_example.LCG_A,
        random_1d_example.LCG_C,
    )
    output = func()
    output_truncated = output[:M]
    ref_output = random_1d_example.reference_random_1d(M, seed)
    torch.testing.assert_close(output_truncated.cpu(), ref_output, rtol=0, atol=0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
