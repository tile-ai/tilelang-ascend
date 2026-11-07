import importlib.util
import sys
from pathlib import Path
from types import ModuleType


def _load_example() -> ModuleType:
    source = Path(__file__).with_name("opt_gdn_chunk_h.py")
    spec = importlib.util.spec_from_file_location("_opt_gdn_chunk_h_example_for_test", source)
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


def test_opt_gdn_chunk_h_accuracy() -> None:
    import torch
    import torch.nn.functional as F

    example = _load_example()

    batch = 2
    heads = 16
    seq_len = 16384
    key_dim = 128
    value_dim = 128
    chunk = 128

    torch.manual_seed(0)
    k = torch.randn((batch, heads, seq_len, key_dim)).npu().to(torch.float16)
    w = torch.randn((batch, heads, seq_len, key_dim)).npu().to(torch.float16)
    u = torch.randn((batch, heads, seq_len, value_dim)).npu().to(torch.float16)
    g = torch.randn((batch, heads, seq_len)).npu().to(torch.float)

    # The recurrence expects the gate already accumulated per chunk.
    g = example.ref_chunk_cumsum(F.logsigmoid(g), chunk)
    k = F.normalize(k, dim=-1, p=2)
    w = F.normalize(w, dim=-1, p=2)

    s, new_v, final_s = example.chunk_h(k, w, u, g, chunk)
    ref_s, ref_new_v, ref_final_s = example.ref_chunk_h(k, w, u, g, chunk)

    torch.testing.assert_close(s.cpu(), ref_s.cpu(), rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(new_v.cpu(), ref_new_v.cpu(), rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(final_s.cpu(), ref_final_s.cpu(), rtol=1e-5, atol=1e-5)
