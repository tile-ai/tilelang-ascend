import importlib.util
import sys
from pathlib import Path
from types import ModuleType


def _load_example() -> ModuleType:
    source = Path(__file__).with_name("opt_gdn_wy_fast.py")
    spec = importlib.util.spec_from_file_location("_opt_gdn_wy_fast_example_for_test", source)
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


def test_opt_gdn_wy_fast_accuracy() -> None:
    import torch

    example = _load_example()

    batch = 2
    heads = 16
    seq_len = 16384
    key_dim = 128
    value_dim = 128
    chunk = 128

    torch.manual_seed(0)
    k = torch.randn((batch, heads, seq_len, key_dim)).npu().to(torch.float16)
    v = torch.randn((batch, heads, seq_len, value_dim)).npu().to(torch.float16)
    beta = torch.rand((batch, heads, seq_len)).npu().to(torch.float16)
    g = torch.randn((batch, heads, seq_len)).npu().to(torch.float)
    a = torch.randn((batch, heads, seq_len, chunk)).npu().to(torch.float16)

    w, u = example.wy_fast(k, v, beta, g, a, chunk)
    ref_w, ref_u = example.ref_wy_fast(k, v, beta, g, a, chunk)

    torch.testing.assert_close(w.cpu(), ref_w.cpu(), rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(u.cpu(), ref_u.cpu(), rtol=1e-5, atol=1e-5)
