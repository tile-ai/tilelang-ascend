import importlib.util
import sys
from pathlib import Path
from types import ModuleType


def _load_example() -> ModuleType:
    source = Path(__file__).with_name("gdn_chunk_scaled_dot_kkt.py")
    spec = importlib.util.spec_from_file_location("_gdn_kkt_example_for_test", source)
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


def test_gdn_chunk_scaled_dot_kkt_accuracy() -> None:
    import torch

    example = _load_example()

    batch = 2
    heads = 32
    seq_len = 256
    head_dim = 64
    chunk = 32
    block_k = 32

    torch.manual_seed(0)
    k = torch.randn((batch, heads, seq_len, head_dim)).npu().to(torch.float16)
    beta = torch.randn((batch, heads, seq_len)).npu().to(torch.float16)
    g = torch.randn((batch, heads, seq_len)).npu().to(torch.float)

    actual = example.kkt(k, beta, g, chunk, block_k)
    expected = example.ref_kkt(k, beta, g, chunk)

    torch.testing.assert_close(actual.cpu(), expected.cpu(), rtol=1e-5, atol=1e-5)
