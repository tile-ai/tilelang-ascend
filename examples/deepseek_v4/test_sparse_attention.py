import importlib.util
import sys
from pathlib import Path
from types import ModuleType
import torch


def _check_precision(actual, golden, dtype):
    table = {
        "float16": (2**-14, 2**-9, 1e-1, 0.99),
        "bfloat16": (2**-10, 2**-6, 1e0, 0.99),
        "float32": (2**-16, 2**-10, 1e-2, 0.99),
        "hifloat32": (2**-16, 2**-10, 1e-2, 0.99),
        "float8_e4m3": (2**-4, 2**-2, 1e0, 0.99),
        "float8_e5m2": (2**-3, 2**-1, 1e-1, 0.99),
    }
    actual, golden = actual.detach().cpu(), golden.detach().cpu()
    if dtype in {"int8", "int16", "int32", "int64", "uint8"}:
        assert torch.equal(actual, golden), "integer output must match exactly"
        return
    atol, rtol, max_abs_limit, required_ratio = table[dtype]
    actual, golden = actual.float(), golden.float()
    special = ~torch.isfinite(golden)
    assert torch.equal(torch.isnan(actual[special]), torch.isnan(golden[special]))
    assert torch.equal(torch.isinf(actual[special]), torch.isinf(golden[special]))
    finite = torch.isfinite(golden)
    if not finite.any():
        return
    errors = (actual[finite] - golden[finite]).abs()
    ratio = (errors <= atol + rtol * golden[finite].abs()).float().mean().item()
    max_abs = errors.max().item()
    assert ratio >= required_ratio and max_abs <= max_abs_limit, f"matched_ratio={ratio:.4f}, max_abs_error={max_abs:.6e}"


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
    _check_precision(actual, expected, "bfloat16")
