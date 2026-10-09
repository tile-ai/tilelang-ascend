import importlib.util
import sys
from pathlib import Path
from types import ModuleType


def _load_example() -> ModuleType:
    source = Path(__file__).with_name("opt_gdn_chunk_scaled_dot_kkt.py")
    spec = importlib.util.spec_from_file_location("_opt_gdn_kkt_example_for_test", source)
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


def test_opt_gdn_chunk_scaled_dot_kkt_accuracy() -> None:
    import torch

    example = _load_example()

    # The kernel walks CC chunks per iteration, so seq_len is a multiple of
    # chunk far larger than the plain gdn variant uses.
    batch = 2
    heads = 16
    seq_len = 16384
    key_dim = 128
    chunk = 128

    torch.manual_seed(0)
    k = torch.randn((batch, heads, seq_len, key_dim)).npu().to(torch.float16)
    beta = torch.rand((batch, heads, seq_len)).npu().to(torch.float16)
    g = torch.randn((batch, heads, seq_len)).npu().to(torch.float)

    actual = example.kkt(k, beta, g, chunk)
    expected = example.ref_kkt(k, beta, g, chunk)

    torch.testing.assert_close(actual.cpu(), expected.cpu(), rtol=1e-3, atol=1e-3)
