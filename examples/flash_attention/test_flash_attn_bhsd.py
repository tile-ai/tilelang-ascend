import importlib.util
import sys
from pathlib import Path
from types import ModuleType
import torch


def _check_precision(a, b):
    a, b = a.detach().cpu(), b.detach().cpu()
    if a.shape != b.shape:
        raise AssertionError("shape mismatch")
    if not a.dtype.is_floating_point:
        if not torch.equal(a, b):
            raise AssertionError("integer mismatch")
        return
    name = str(a.dtype).replace("torch.", "")
    table = {"float16": (2**-14, 2**-9, 1e-1), "bfloat16": (2**-10, 2**-6, 1e0), "float32": (2**-16, 2**-10, 1e-2)}
    atol, rtol, limit = table.get(name, table["float16"])
    a, b = a.float(), b.float()
    if not (
        torch.equal(torch.isnan(a), torch.isnan(b))
        and torch.equal(torch.isposinf(a), torch.isposinf(b))
        and torch.equal(torch.isneginf(a), torch.isneginf(b))
    ):
        raise AssertionError("special values differ")
    finite = torch.isfinite(b)
    if finite.any():
        err = torch.where(torch.isfinite(a[finite]), (a[finite] - b[finite]).abs(), torch.full_like(b[finite], float("inf")))
        if (err <= atol + rtol * b[finite].abs()).float().mean().item() < 0.99 or err.max().item() > limit:
            raise AssertionError("precision mismatch")


def _load_flash_attn_example() -> ModuleType:
    source = Path(__file__).with_name("flash_attn_bhsd.py")
    spec = importlib.util.spec_from_file_location("_flash_attn_bhsd_example_for_test", source)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load example module: {source}")

    module = importlib.util.module_from_spec(spec)
    original_argv = sys.argv
    try:
        # flash_attn_bhsd.py parses arguments in its __main__ block. Hide Pytest
        # arguments while loading it without changing the original Example.
        sys.argv = [str(source)]
        spec.loader.exec_module(module)
    finally:
        sys.argv = original_argv

    return module


def _reference_flash_attn(query, key, value):
    # Mirrors ref_flash_attn defined inside the example's __main__ block, which
    # is not reachable after importing the module.
    query = query.float()
    key = key.float()
    value = value.float()

    scores = torch.einsum("bhsd,bhkd->bhsk", query, key) * (1.0 / query.shape[-1]) ** 0.5
    scores = scores.softmax(dim=-1)
    out = torch.einsum("bhsk,bhkd->bhsd", scores, value)
    return out.to(torch.float16)


def test_flash_attn_bhsd_accuracy() -> None:
    import torch

    example = _load_flash_attn_example()

    batch = 1
    seq_len = 128
    heads = 1
    dim = 512

    kernel = example.flash_attention_fwd(
        batch=batch,
        seq_len=seq_len,
        heads=heads,
        dim=dim,
    )

    torch.manual_seed(0)
    query = torch.randn((batch, heads, seq_len, dim), dtype=torch.float16, device="npu")
    key = torch.randn((batch, heads, seq_len, dim), dtype=torch.float16, device="npu")
    value = torch.randn((batch, heads, seq_len, dim), dtype=torch.float16, device="npu")

    actual = kernel(query, key, value)
    expected = _reference_flash_attn(query, key, value)
    torch.npu.synchronize()

    _check_precision(actual, expected)
