import importlib.util
import sys
from pathlib import Path
from types import ModuleType


def _load_example() -> ModuleType:
    source = Path(__file__).with_name("chunk_gated_delta_rule.py")
    spec = importlib.util.spec_from_file_location("_chunk_gated_delta_rule_example_for_test", source)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load example module: {source}")

    module = importlib.util.module_from_spec(spec)
    original_argv = sys.argv
    try:
        # The example parses arguments at import time.
        sys.argv = [str(source)]
        spec.loader.exec_module(module)
    finally:
        sys.argv = original_argv

    return module


def test_chunk_gated_delta_rule_accuracy() -> None:
    import torch

    example = _load_example()

    # The kernel only takes the varlen layout, so the batch dimension is one and
    # the sequences are concatenated behind cu_seqlens.
    seqlens = [2048]
    heads = 8
    grouped_heads = 4
    key_dim = 128
    value_dim = 128

    total = sum(seqlens)
    sequences = len(seqlens)
    cu_seqlens = torch.tensor([0] + [sum(seqlens[: i + 1]) for i in range(len(seqlens))], dtype=torch.int32).npu()

    # The recurrence multiplies these together over 2048 steps, so the example
    # keeps the inputs small; drawn from randn they overflow float16.
    torch.manual_seed(41)
    k = torch.rand(1, total, grouped_heads, key_dim, dtype=torch.float16).npu() * 0.01
    w = torch.rand(1, total, heads, key_dim, dtype=torch.float16).npu() * 0.01
    u = torch.rand(1, total, heads, value_dim, dtype=torch.float16).npu() * 0.01
    g = torch.rand(1, total, heads, dtype=torch.float32).npu() * -1.0
    initial_state = torch.rand(1, sequences, heads, key_dim, value_dim, dtype=torch.float16).npu() * 0.01

    h, v_new, ht = example.chunk_gated_delta_rule_fwd_h(
        k, w, u, g, initial_state=initial_state, output_final_state=True, cu_seqlens=cu_seqlens
    )
    ref_h, ref_v_new, ref_ht = example.ref_chunk_gated_delta_rule(
        k.cpu(),
        w.cpu(),
        u.cpu(),
        g.cpu(),
        initial_state=initial_state.cpu(),
        output_final_state=True,
        cu_seqlens=cu_seqlens.cpu(),
    )

    torch.testing.assert_close(h.cpu(), ref_h.cpu(), rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(v_new.cpu(), ref_v_new.cpu(), rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(ht.cpu(), ref_ht.cpu(), rtol=1e-5, atol=1e-5)
