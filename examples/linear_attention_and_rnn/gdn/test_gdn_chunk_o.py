import importlib.util
import sys
from pathlib import Path
from types import ModuleType


def _load_example() -> ModuleType:
    source = Path(__file__).with_name("gdn_chunk_o.py")
    spec = importlib.util.spec_from_file_location("_gdn_chunk_o_example_for_test", source)
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


def test_gdn_chunk_o_accuracy() -> None:
    import torch
    import torch.nn.functional as F

    example = _load_example()

    batch = 2
    heads = 32
    seq_len = 256
    key_dim = 64
    value_dim = 64
    chunk = 32
    block_k = 32
    block_v = 32
    chunk_num = (seq_len + chunk - 1) // chunk

    torch.manual_seed(0)
    q = torch.randn((batch, heads, seq_len, key_dim)).npu().to(torch.float16)
    k = torch.randn((batch, heads, seq_len, key_dim)).npu().to(torch.float16)
    v = torch.randn((batch, heads, seq_len, value_dim)).npu().to(torch.float16)
    s = torch.randn((batch, heads, chunk_num, key_dim, value_dim)).npu().to(torch.float16)
    g = torch.randn((batch, heads, seq_len)).npu().to(torch.float)

    q = F.normalize(q, dim=-1, p=2)
    k = F.normalize(k, dim=-1, p=2)

    actual = example.chunk_o(q, k, v, s, g, chunk, block_k, block_v)
    expected = example.ref_chunk_o(q, k, v, s, g, chunk)

    torch.testing.assert_close(actual.cpu(), expected.cpu(), rtol=1e-5, atol=1e-5)
