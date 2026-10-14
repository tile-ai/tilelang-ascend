"""moe_re_routing operator tests: L0/L1 + main() for CI pytest.

Imports the host callable and the golden from the sibling example file
(example_moe_re_routing.py).  Test cases assert exact-match precision on all
four outputs (permute_tokens / permute_per_token_scales / permute_token_idx /
expert_token_num) using the mixed-tolerance standard;

This file is picked up by CI's ``pytest **/test*.py`` in ``examples`` and is
also directly runnable: ``python test_moe_re_routing.py``.
"""

import os
import sys

import torch

# Make the sibling example importable (same directory).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from example_moe_re_routing import (  # noqa: E402
    golden_moe_re_routing,
    moe_re_routing,
)

# ========== Test configs (uniform cnt, as evaluation rewrites it) ==========
# (name, A, H, N, E, tokens dtype, cnt dtype, with scale)
TEST_CONFIGS = [
    ("l0_fp16", 1024, 512, 8, 8, torch.float16, torch.int32, True),
    ("l1_bf16", 16384, 1024, 16, 16, torch.bfloat16, torch.int64, True),
]


def _uniform_cnt(A, N, E, cnt_dtype, device):
    """Counts with uniform base and the remainder on the last cell (Sum == A)."""
    base = A // (N * E)
    cnt = torch.full((N * E,), base, dtype=cnt_dtype, device=device)
    cnt[-1] += A - base * N * E
    return cnt.reshape(N, E)


def _assert_exact(out, ref):
    """All four outputs of a pure permutation must be bit-exact."""
    for name, a, g in zip(
        ["permute_tokens", "permute_per_token_scales", "permute_token_idx", "expert_token_num"],
        out,
        ref,
    ):
        assert a.detach().cpu().shape == g.cpu().shape, f"{name} shape mismatch"
        assert torch.equal(a.detach().cpu(), g.cpu()), f"{name} mismatch"


def _check_all_outputs(tokens, cnt, scales=None):
    out = moe_re_routing(tokens, cnt, scales)
    ref = golden_moe_re_routing(tokens, cnt, scales)
    _assert_exact(out, ref)


def test_moe_re_routing_l0():
    """L0: fp16 + int32 + scales (case-1 representative)."""
    name, A, H, N, E, td, cd, with_scale = TEST_CONFIGS[0]
    tokens = torch.randn(A, H, dtype=td).npu()
    cnt = _uniform_cnt(A, N, E, cd, "npu")
    scales = torch.randn(A, dtype=torch.float32).npu() if with_scale else None
    _check_all_outputs(tokens, cnt, scales)


def test_moe_re_routing_l1():
    """L1: bf16 + int64 + scales (case-2 representative)."""
    name, A, H, N, E, td, cd, with_scale = TEST_CONFIGS[1]
    tokens = torch.randn(A, H, dtype=td).npu()
    cnt = _uniform_cnt(A, N, E, cd, "npu")
    scales = torch.randn(A, dtype=torch.float32).npu() if with_scale else None
    _check_all_outputs(tokens, cnt, scales)


def test_moe_re_routing_noscale():
    """No per_token_scales path: output scales must be zeros and all else exact."""
    A, H, N, E = 1024, 512, 8, 8
    tokens = torch.randn(A, H, dtype=torch.float16).npu()
    cnt = _uniform_cnt(A, N, E, torch.int32, "npu")
    out = moe_re_routing(tokens, cnt, None)
    ref = golden_moe_re_routing(tokens, cnt, None)
    _assert_exact(out, ref)
    assert torch.equal(out[1].detach().cpu(), torch.zeros(A, dtype=torch.float32))


if __name__ == "__main__":
    for name, A, H, N, E, td, cd, with_scale in TEST_CONFIGS:
        print(f"Testing moe_re_routing {name} with A={A}, H={H}, N={N}, E={E}")
        tokens = torch.randn(A, H, dtype=td).npu()
        cnt = _uniform_cnt(A, N, E, cd, "npu")
        scales = torch.randn(A, dtype=torch.float32).npu() if with_scale else None
        _check_all_outputs(tokens, cnt, scales)
        print(f"Test pass! {name} all outputs exact")
    print("Kernel Output Match!")
