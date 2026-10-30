import importlib.util
import sys
from pathlib import Path
from types import ModuleType


def _load_batch_gemm_example() -> ModuleType:
    source = Path(__file__).with_name("batch_gemm.py")
    spec = importlib.util.spec_from_file_location("_batch_gemm_example_for_test", source)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load example module: {source}")

    module = importlib.util.module_from_spec(spec)
    original_argv = sys.argv
    try:
        # batch_gemm.py parses arguments at import time. Hide Pytest arguments
        # while loading it without changing the original Example.
        sys.argv = [str(source)]
        spec.loader.exec_module(module)
    finally:
        sys.argv = original_argv

    return module


def test_batch_gemm_accuracy() -> None:
    import torch

    example = _load_batch_gemm_example()

    batch = 8
    m = 1024
    n = 1024
    k = 1024

    kernel = example.batch_matmul(
        batch,
        m,
        n,
        k,
        block_M=128,
        block_N=256,
        K_L1=64,
    )

    torch.manual_seed(0)
    lhs = torch.randn(batch, m, k).half().npu()
    rhs = torch.randn(batch, k, n).half().npu()

    actual = kernel(lhs, rhs)
    expected = torch.matmul(lhs, rhs)

    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)
