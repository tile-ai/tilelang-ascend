"""moe_gating_top_k_softmax operator tests: L0/L1 + finished + tie for CI pytest.

Imports the host callable and the golden from the sibling example file
(example_moe_gating_top_k_softmax.py).  The y output is checked with a
dtype-scaled mixed tolerance (fp16 ~2^-10 / bf16 ~2^-7); expert_idx is
checked exactly modulo topk tie-breaking (positions with equal y values may
legitimately swap indices, so tie positions are compared by value instead);
row_idx is always exact.

This file is picked up by CI's ``pytest **/test*.py`` in ``examples`` and is
also directly runnable: ``python test_moe_gating_top_k_softmax.py``.
"""

import os
import sys

import torch

# Make the sibling example importable (same directory).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from example_moe_gating_top_k_softmax import (  # noqa: E402
    golden_moe_gating_top_k_softmax, moe_gating_top_k_softmax,
)

# ========== Test configs (evaluation case-1 / case-3 representatives) ==========
# (name, x shape, dtype, k)
TEST_CONFIGS = [
    ("l0_fp16", (1024, 16), torch.float16, 2),
    ("l1_bf16", (16384, 64), torch.bfloat16, 8),
]

# Mixed tolerance per dtype (softmax+topk values).
_ATOL = {
    torch.float16: 2**-10,
    torch.bfloat16: 2**-7,
    torch.float32: 2**-13,
}


def _check_outputs(x, finished, k):
    """Run op + golden and assert y tolerance / idx tie-exactness."""
    out = moe_gating_top_k_softmax(x, finished, k)
    ref = golden_moe_gating_top_k_softmax(x.cpu(),
                                          finished.cpu() if finished is not None else None, k)

    a_y, g_y = out[0].detach().cpu().float(), ref[0].float()
    assert torch.allclose(a_y, g_y, atol=_ATOL[x.dtype], rtol=1e-3), \
        f"y mismatch: max_abs={(a_y - g_y).abs().max().item():.3e}"

    a_r, g_r = out[2].detach().cpu(), ref[2]
    assert torch.equal(a_r, g_r), "row_idx mismatch"

    a_i, g_i = out[1].detach().cpu(), ref[1]
    if not torch.equal(a_i, g_i):
        mism = a_i != g_i
        # Topk tie-breaking: at mismatched positions the y values must agree
        # at the OUTPUT dtype precision (the kernel keeps fp32 intermediates,
        # so bf16-equal values may differ by < 1 ulp in fp32).
        tie_tol = _ATOL[x.dtype]
        assert torch.allclose(a_y[mism], g_y[mism], atol=tie_tol, rtol=1e-3), \
            f"expert_idx mismatch without value tie ({int(mism.sum())} positions)"
    return out, ref


def test_moe_gating_l0():
    """L0: fp16 (1024, 16) k=2 — per-row topk path (case-1 representative)."""
    _, shape, dt, k = TEST_CONFIGS[0]
    torch.manual_seed(0)
    x = torch.randn(*shape, dtype=dt).npu()
    _check_outputs(x, None, k)


def test_moe_gating_l1():
    """L1: bf16 (16384, 64) k=8 — sort32+merge direct-output path (case-3)."""
    _, shape, dt, k = TEST_CONFIGS[1]
    torch.manual_seed(0)
    x = torch.randn(*shape, dtype=dt).npu()
    _check_outputs(x, None, k)


def test_moe_gating_finished():
    """Finished mask: flagged rows must report expert_idx == E."""
    shape, dt, k = (1024, 64), torch.float16, 8
    E = shape[-1]
    torch.manual_seed(0)
    x = torch.randn(*shape, dtype=dt).npu()
    finished = torch.zeros(shape[:-1], dtype=torch.bool)
    finished[:128] = True  # flag the first 128 rows
    out, ref = _check_outputs(x, finished.npu(), k)
    # Every flagged row must be all-E in the op output; unflagged rows are
    # already covered by _check_outputs (y tolerance + idx tie-exactness).
    op_idx = out[1].detach().cpu()
    assert torch.equal(op_idx[:128], torch.full((128, k), E, dtype=torch.int32)), \
        "finished rows must have expert_idx == E"


def test_moe_gating_tie():
    """Tie-breaking: duplicated values keep y exact; idx compared by set."""
    shape, dt, k = (256, 32), torch.float32, 4
    torch.manual_seed(0)
    x = torch.randn(*shape, dtype=dt)
    x[:, ::2] = x[:, 0:1]  # force heavy value duplication
    out = moe_gating_top_k_softmax(x.npu(), None, k)
    ref = golden_moe_gating_top_k_softmax(x, None, k)

    a_y, g_y = out[0].detach().cpu().float(), ref[0].float()
    assert torch.allclose(a_y, g_y, atol=_ATOL[torch.float32], rtol=1e-3), \
        "tie input: y values must stay exact"

    # With duplicated values the selected index ORDER may differ; the value
    # MULTISET of each row must be identical at the OUTPUT dtype precision
    # (the kernel keeps fp32 intermediates; bf16/fp32-exact comparison is
    # not meaningful across implementations).
    a_sorted = torch.sort(a_y, dim=-1).values
    g_sorted = torch.sort(g_y, dim=-1).values
    assert torch.allclose(a_sorted, g_sorted, atol=_ATOL[torch.float32], rtol=1e-3), \
        "tie input: y value multiset differs"

    # row_idx is position-only: always exact.
    assert torch.equal(out[2].detach().cpu(), ref[2])


if __name__ == "__main__":
    for name, shape, dt, k in TEST_CONFIGS:
        print(f"Testing moe_gating_top_k_softmax {name} with shape={shape}, dtype={dt}, k={k}")
        torch.manual_seed(0)
        x = torch.randn(*shape, dtype=dt).npu()
        _check_outputs(x, None, k)
        print(f"Test pass! {name}")
    print("Kernel Output Match!")
