import importlib.util
import sys
from pathlib import Path
from types import ModuleType


def _load_sparse_attention_example() -> ModuleType:
    source = Path(__file__).with_name("sparse_attention.py")
    spec = importlib.util.spec_from_file_location("_sparse_attention_example_for_test", source)
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


def test_sparse_attention_accuracy() -> None:
    import torch

    example = _load_sparse_attention_example()

    dtype = torch.bfloat16
    batch = 1
    query_len = 256
    kv_len = 256
    heads = 64
    head_dim = 512
    topk = 128

    torch.manual_seed(42)
    inputs = example.make_random_test_inputs(batch, query_len, kv_len, heads, head_dim, topk, dtype)
    query = inputs["q"]
    kv = inputs["kv"]
    attn_sink = inputs["attn_sink"]
    topk_idxs = inputs["topk_idxs"]
    softmax_scale = head_dim**-0.5

    kernel = example.sparse_attn_kernel(h=heads, d=head_dim, scale=softmax_scale)
    actual = kernel(query, kv, attn_sink, topk_idxs)
    torch.npu.synchronize()

    expected = example.sparse_attn(query, kv, attn_sink, topk_idxs, softmax_scale)
    torch.testing.assert_close(expected, actual, rtol=1e-2, atol=1e-2)
