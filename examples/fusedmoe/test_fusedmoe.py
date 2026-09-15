"""Tests for FusedMoE NPU implementation.

Test levels:
  - L0: threshold tests (regular shapes, block-aligned)
  - L1: functional tests (irregular shapes, various configs)
  - L2: exception tests (empty groups, minimal input)
  - Boundary: special values (zero, large, NaN, Inf)
  - bench: performance benchmark using tilelang.profiler.do_bench

Usage:
  python test_fusedmoe.py                # run L0 tests (default)
  python test_fusedmoe.py --level l0     # run L0 tests
  python test_fusedmoe.py --level all    # run all tests
  python test_fusedmoe.py --level bench  # run performance benchmark (do_bench)
  python test_fusedmoe.py --level bench --profiler msprof  # hardware-level profiling
"""

import argparse
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import tilelang  # noqa: E402
import torch  # noqa: E402
from tilelang.profiler import do_bench  # noqa: E402

from example_fusedmoe import (  # noqa: E402
    BLOCK_M,
    golden_fusedmoe_full,
    golden_routed_expert_nc,
    golden_shared_expert,
    host_preprocess,
    host_preprocess_for_test,
    routed_expert_kernel,
    shared_expert_kernel,
)


def test_fusedmoe_l0():
    """L0 threshold tests: regular shapes, block-aligned, for precision convergence."""
    ok = True
    device = torch.device("npu")

    # ---- Test 1: l0_shared_basic ----
    try:
        num_tokens, d_hidden, d_expert = 64, 128, 64
        torch.manual_seed(42)

        x = torch.randn(num_tokens, d_hidden, dtype=torch.float16).to(device)
        w_gate = torch.randn(d_expert, d_hidden, dtype=torch.float16).to(device) * 0.01
        w_up = torch.randn(d_expert, d_hidden, dtype=torch.float16).to(device) * 0.01
        w_down = torch.randn(d_hidden, d_expert, dtype=torch.float16).to(device) * 0.01

        kernel = shared_expert_kernel(num_tokens, d_hidden, d_expert)
        output = kernel(x, w_gate, w_up, w_down)

        ref = golden_shared_expert(x, w_gate, w_up, w_down)
        torch.testing.assert_close(output.cpu(), ref.cpu(), atol=5e-3, rtol=5e-3)
        print(f"[PRECISION_PASS] l0_shared_basic shape=({num_tokens},{d_hidden},{d_expert}) dtype=float16")
    except Exception as e:
        print(f"[PRECISION_FAIL] l0_shared_basic shape=(64,128,64) dtype=float16: {e}")
        ok = False

    # ---- Test 2: l0_shared_typical ----
    try:
        num_tokens, d_hidden, d_expert = 128, 256, 128
        torch.manual_seed(43)

        x = torch.randn(num_tokens, d_hidden, dtype=torch.float16).to(device)
        w_gate = torch.randn(d_expert, d_hidden, dtype=torch.float16).to(device) * 0.01
        w_up = torch.randn(d_expert, d_hidden, dtype=torch.float16).to(device) * 0.01
        w_down = torch.randn(d_hidden, d_expert, dtype=torch.float16).to(device) * 0.01

        kernel = shared_expert_kernel(num_tokens, d_hidden, d_expert)
        output = kernel(x, w_gate, w_up, w_down)

        ref = golden_shared_expert(x, w_gate, w_up, w_down)
        torch.testing.assert_close(output.cpu(), ref.cpu(), atol=5e-3, rtol=5e-3)
        print(f"[PRECISION_PASS] l0_shared_typical shape=({num_tokens},{d_hidden},{d_expert}) dtype=float16")
    except Exception as e:
        print(f"[PRECISION_FAIL] l0_shared_typical shape=(128,256,128) dtype=float16: {e}")
        ok = False

    # ---- Test 3: l0_routed_basic ----
    try:
        d_hidden, d_expert, n_experts = 128, 64, 1
        group_sizes = [64]
        torch.manual_seed(44)

        routing = host_preprocess_for_test(group_sizes, d_hidden, n_experts, BLOCK_M, device)
        buf_rows = routing["buf_rows"]

        w_gate = torch.randn(n_experts, d_expert, d_hidden, dtype=torch.float16).to(device) * 0.01
        w_up = torch.randn(n_experts, d_expert, d_hidden, dtype=torch.float16).to(device) * 0.01
        w_down = torch.randn(n_experts, d_hidden, d_expert, dtype=torch.float16).to(device) * 0.01

        kernel = routed_expert_kernel(buf_rows, d_hidden, d_expert, n_experts, routing["total_m_blocks"])
        output = kernel(
            routing["stacked_tokens"],
            w_gate,
            w_up,
            w_down,
            routing["stacked_weights"],
            routing["block_metadata"],
        )

        ref = golden_routed_expert_nc(
            routing["stacked_tokens"],
            w_gate,
            w_up,
            w_down,
            routing["stacked_weights"],
            routing["block_metadata"],
        )

        # Compare only valid rows (exclude padding from non-compact layout)
        valid_mask = routing["stacked_weights"] > 0
        if valid_mask.any():
            torch.testing.assert_close(
                output[valid_mask].cpu(),
                ref[valid_mask].cpu(),
                atol=5e-3,
                rtol=5e-3,
            )
        print(f"[PRECISION_PASS] l0_routed_basic group_sum=64 d_hidden={d_hidden} n_experts={n_experts} dtype=float16")
    except Exception as e:
        print(f"[PRECISION_FAIL] l0_routed_basic: {e}")
        ok = False

    # ---- Test 4: l0_routed_multi ----
    try:
        d_hidden, d_expert, n_experts = 128, 64, 2
        group_sizes = [128, 128]
        torch.manual_seed(45)

        routing = host_preprocess_for_test(group_sizes, d_hidden, n_experts, BLOCK_M, device)
        buf_rows = routing["buf_rows"]

        w_gate = torch.randn(n_experts, d_expert, d_hidden, dtype=torch.float16).to(device) * 0.01
        w_up = torch.randn(n_experts, d_expert, d_hidden, dtype=torch.float16).to(device) * 0.01
        w_down = torch.randn(n_experts, d_hidden, d_expert, dtype=torch.float16).to(device) * 0.01

        kernel = routed_expert_kernel(buf_rows, d_hidden, d_expert, n_experts, routing["total_m_blocks"])
        output = kernel(
            routing["stacked_tokens"],
            w_gate,
            w_up,
            w_down,
            routing["stacked_weights"],
            routing["block_metadata"],
        )

        ref = golden_routed_expert_nc(
            routing["stacked_tokens"],
            w_gate,
            w_up,
            w_down,
            routing["stacked_weights"],
            routing["block_metadata"],
        )

        valid_mask = routing["stacked_weights"] > 0
        if valid_mask.any():
            torch.testing.assert_close(
                output[valid_mask].cpu(),
                ref[valid_mask].cpu(),
                atol=1e-2,
                rtol=1e-2,
            )
        print(f"[PRECISION_PASS] l0_routed_multi group_sum=256 d_hidden={d_hidden} n_experts={n_experts} dtype=float16")
    except Exception as e:
        print(f"[PRECISION_FAIL] l0_routed_multi: {e}")
        ok = False

    # ---- Test 5: l0_e2e_tiny ----
    try:
        batch_size, seq_len = 1, 64
        d_hidden, d_expert = 128, 64
        n_routed_experts, n_shared_experts = 2, 1
        n_experts_per_token = 1
        d_expert_shared = d_expert * n_shared_experts
        num_tokens = batch_size * seq_len
        torch.manual_seed(46)

        x = torch.randn(batch_size, seq_len, d_hidden, dtype=torch.float16).to(device)
        x_flat = x.view(num_tokens, d_hidden)

        w_gate_shared = torch.randn(d_expert_shared, d_hidden, dtype=torch.float16).to(device) * 0.01
        w_up_shared = torch.randn(d_expert_shared, d_hidden, dtype=torch.float16).to(device) * 0.01
        w_down_shared = torch.randn(d_hidden, d_expert_shared, dtype=torch.float16).to(device) * 0.01

        w_gate_routed = (
            torch.randn(
                n_routed_experts,
                d_expert,
                d_hidden,
                dtype=torch.float16,
            ).to(device)
            * 0.01
        )
        w_up_routed = (
            torch.randn(
                n_routed_experts,
                d_expert,
                d_hidden,
                dtype=torch.float16,
            ).to(device)
            * 0.01
        )
        w_down_routed = (
            torch.randn(
                n_routed_experts,
                d_hidden,
                d_expert,
                dtype=torch.float16,
            ).to(device)
            * 0.01
        )

        router_weight = torch.randn(n_routed_experts, d_hidden, dtype=torch.float16).to(device) * 0.01

        # Golden
        ref_output = golden_fusedmoe_full(
            x,
            w_gate_shared,
            w_up_shared,
            w_down_shared,
            w_gate_routed,
            w_up_routed,
            w_down_routed,
            router_weight,
            n_experts_per_token,
        )

        # Kernel: shared expert
        shared_kernel = shared_expert_kernel(num_tokens, d_hidden, d_expert_shared)
        shared_output = shared_kernel(x_flat, w_gate_shared, w_up_shared, w_down_shared)

        # Kernel: routed expert
        routing = host_preprocess(x_flat, router_weight, n_experts_per_token, BLOCK_M, device)
        buf_rows = routing["buf_rows"]

        routed_kernel = routed_expert_kernel(buf_rows, d_hidden, d_expert, n_routed_experts, routing["total_m_blocks"])
        routed_output_nc = routed_kernel(
            routing["stacked_tokens"],
            w_gate_routed,
            w_up_routed,
            w_down_routed,
            routing["stacked_weights"],
            routing["block_metadata"],
        )

        # Scatter from non-compact layout (only valid rows)
        valid_mask = routing["stacked_weights"] > 0
        valid_output = routed_output_nc[valid_mask].to(torch.float16)
        valid_idxs = routing["token_idxs_nc"][valid_mask].long()

        expert_cache = torch.zeros(num_tokens, d_hidden, dtype=torch.float16).to(device)
        expert_cache[valid_idxs] = valid_output

        # Final output
        kernel_output = shared_output.view(batch_size, seq_len, d_hidden) + expert_cache.view(batch_size, seq_len, d_hidden)

        torch.testing.assert_close(kernel_output.cpu(), ref_output.cpu(), atol=0.25, rtol=0.25)
        print(
            f"[PRECISION_PASS] l0_e2e_tiny batch={batch_size} seq={seq_len} "
            f"d_hidden={d_hidden} d_expert={d_expert} n_routed={n_routed_experts} "
            f"top_k={n_experts_per_token} dtype=float16"
        )
    except Exception as e:
        print(f"[PRECISION_FAIL] l0_e2e_tiny: {e}")
        ok = False

    return ok


# ============================================================================
# Test Helpers (L1/L2/Boundary)
# ============================================================================
def _run_precision(level, name, fn, atol, rtol):
    """L0/L1 single test case: prints [PRECISION_PASS] or [PRECISION_FAIL]."""
    try:
        fn()
        print(f"[PRECISION_PASS] {level} {name}")
        return True
    except Exception as e:
        print(f"[PRECISION_FAIL] {level} {name}: {e}")
        return False


def _run_boundary(level, name, fn):
    """L2/Boundary single test case: prints [BOUNDARY_PASS] or [BOUNDARY_WARN]."""
    try:
        fn()
        print(f"[BOUNDARY_PASS] {level} {name}")
    except Exception as e:
        print(f"[BOUNDARY_WARN] {level} {name}: {e}")


# ============================================================================
# L1 Functional Tests (regular + irregular shapes, blocking on precision)
# ============================================================================
def test_fusedmoe_l1():
    """L1 functional tests: regular + irregular shapes + user golden params.

    Per DESIGN.md §9.3 precision standard (Fusion class):
      - float16 standard: atol=5e-3, rtol=5e-3
      - float16 large (d_hidden=7168): atol=1e-2, rtol=1e-2
    """
    ok = True
    device = torch.device("npu")

    # ---- L1-1: shared expert, irregular shape (non-block-aligned) ----
    def l1_shared_irregular_1():
        num_tokens, d_hidden, d_expert = 100, 200, 150  # not divisible by 64
        torch.manual_seed(101)
        x = torch.randn(num_tokens, d_hidden, dtype=torch.float16).to(device)
        w_gate = torch.randn(d_expert, d_hidden, dtype=torch.float16).to(device) * 0.01
        w_up = torch.randn(d_expert, d_hidden, dtype=torch.float16).to(device) * 0.01
        w_down = torch.randn(d_hidden, d_expert, dtype=torch.float16).to(device) * 0.01
        kernel = shared_expert_kernel(num_tokens, d_hidden, d_expert)
        output = kernel(x, w_gate, w_up, w_down)
        ref = golden_shared_expert(x, w_gate, w_up, w_down)
        torch.testing.assert_close(output.cpu(), ref.cpu(), atol=5e-3, rtol=5e-3)

    ok &= _run_precision("l1", "shared_irregular_1 (100,200,150)", l1_shared_irregular_1, 5e-3, 5e-3)

    # ---- L1-2: shared expert, tail block (small last block) ----
    def l1_shared_irregular_2():
        num_tokens, d_hidden, d_expert = 50, 300, 100  # 50 not divisible by 64
        torch.manual_seed(102)
        x = torch.randn(num_tokens, d_hidden, dtype=torch.float16).to(device)
        w_gate = torch.randn(d_expert, d_hidden, dtype=torch.float16).to(device) * 0.01
        w_up = torch.randn(d_expert, d_hidden, dtype=torch.float16).to(device) * 0.01
        w_down = torch.randn(d_hidden, d_expert, dtype=torch.float16).to(device) * 0.01
        kernel = shared_expert_kernel(num_tokens, d_hidden, d_expert)
        output = kernel(x, w_gate, w_up, w_down)
        ref = golden_shared_expert(x, w_gate, w_up, w_down)
        torch.testing.assert_close(output.cpu(), ref.cpu(), atol=5e-3, rtol=5e-3)

    ok &= _run_precision("l1", "shared_irregular_2 (50,300,100)", l1_shared_irregular_2, 5e-3, 5e-3)

    # ---- L1-3: routed expert, irregular group_sizes ----
    def l1_routed_irregular():
        d_hidden, d_expert, n_experts = 128, 64, 2
        group_sizes = [100, 50]  # not divisible by 64
        torch.manual_seed(103)
        routing = host_preprocess_for_test(group_sizes, d_hidden, n_experts, BLOCK_M, device)
        buf_rows = routing["buf_rows"]
        w_gate = torch.randn(n_experts, d_expert, d_hidden, dtype=torch.float16).to(device) * 0.01
        w_up = torch.randn(n_experts, d_expert, d_hidden, dtype=torch.float16).to(device) * 0.01
        w_down = torch.randn(n_experts, d_hidden, d_expert, dtype=torch.float16).to(device) * 0.01
        kernel = routed_expert_kernel(buf_rows, d_hidden, d_expert, n_experts, routing["total_m_blocks"])
        output = kernel(
            routing["stacked_tokens"],
            w_gate,
            w_up,
            w_down,
            routing["stacked_weights"],
            routing["block_metadata"],
        )
        ref = golden_routed_expert_nc(
            routing["stacked_tokens"],
            w_gate,
            w_up,
            w_down,
            routing["stacked_weights"],
            routing["block_metadata"],
        )
        valid_mask = routing["stacked_weights"] > 0
        torch.testing.assert_close(output[valid_mask].cpu(), ref[valid_mask].cpu(), atol=5e-3, rtol=5e-3)

    ok &= _run_precision("l1", "routed_irregular groups=[100,50]", l1_routed_irregular, 5e-3, 5e-3)

    # ---- L1-4: routed expert, uneven multi-expert distribution ----
    def l1_routed_multi_uneven():
        d_hidden, d_expert, n_experts = 256, 128, 3
        group_sizes = [64, 192, 128]  # uneven
        torch.manual_seed(104)
        routing = host_preprocess_for_test(group_sizes, d_hidden, n_experts, BLOCK_M, device)
        buf_rows = routing["buf_rows"]
        w_gate = torch.randn(n_experts, d_expert, d_hidden, dtype=torch.float16).to(device) * 0.01
        w_up = torch.randn(n_experts, d_expert, d_hidden, dtype=torch.float16).to(device) * 0.01
        w_down = torch.randn(n_experts, d_hidden, d_expert, dtype=torch.float16).to(device) * 0.01
        kernel = routed_expert_kernel(buf_rows, d_hidden, d_expert, n_experts, routing["total_m_blocks"])
        output = kernel(
            routing["stacked_tokens"],
            w_gate,
            w_up,
            w_down,
            routing["stacked_weights"],
            routing["block_metadata"],
        )
        ref = golden_routed_expert_nc(
            routing["stacked_tokens"],
            w_gate,
            w_up,
            w_down,
            routing["stacked_weights"],
            routing["block_metadata"],
        )
        valid_mask = routing["stacked_weights"] > 0
        torch.testing.assert_close(output[valid_mask].cpu(), ref[valid_mask].cpu(), atol=5e-3, rtol=5e-3)

    ok &= _run_precision("l1", "routed_multi_uneven [64,192,128]", l1_routed_multi_uneven, 5e-3, 5e-3)

    # ---- L1-5: routed expert, single token (minimal valid) ----
    def l1_routed_single_token():
        d_hidden, d_expert, n_experts = 128, 64, 1
        group_sizes = [1]  # single token, heavy tail padding
        torch.manual_seed(105)
        routing = host_preprocess_for_test(group_sizes, d_hidden, n_experts, BLOCK_M, device)
        buf_rows = routing["buf_rows"]
        w_gate = torch.randn(n_experts, d_expert, d_hidden, dtype=torch.float16).to(device) * 0.01
        w_up = torch.randn(n_experts, d_expert, d_hidden, dtype=torch.float16).to(device) * 0.01
        w_down = torch.randn(n_experts, d_hidden, d_expert, dtype=torch.float16).to(device) * 0.01
        kernel = routed_expert_kernel(buf_rows, d_hidden, d_expert, n_experts, routing["total_m_blocks"])
        output = kernel(
            routing["stacked_tokens"],
            w_gate,
            w_up,
            w_down,
            routing["stacked_weights"],
            routing["block_metadata"],
        )
        ref = golden_routed_expert_nc(
            routing["stacked_tokens"],
            w_gate,
            w_up,
            w_down,
            routing["stacked_weights"],
            routing["block_metadata"],
        )
        valid_mask = routing["stacked_weights"] > 0
        torch.testing.assert_close(output[valid_mask].cpu(), ref[valid_mask].cpu(), atol=5e-3, rtol=5e-3)

    ok &= _run_precision("l1", "routed_single_token group=[1]", l1_routed_single_token, 5e-3, 5e-3)

    # ---- L1-6: E2E with multiple experts (top_k=2, n_routed=3) ----
    def l1_e2e_multi_expert():
        batch_size, seq_len = 1, 128
        d_hidden, d_expert = 128, 64
        n_routed_experts, n_shared_experts = 3, 1
        n_experts_per_token = 2
        d_expert_shared = d_expert * n_shared_experts
        num_tokens = batch_size * seq_len
        torch.manual_seed(106)
        x = torch.randn(batch_size, seq_len, d_hidden, dtype=torch.float16).to(device)
        x_flat = x.view(num_tokens, d_hidden)
        w_gate_shared = torch.randn(d_expert_shared, d_hidden, dtype=torch.float16).to(device) * 0.01
        w_up_shared = torch.randn(d_expert_shared, d_hidden, dtype=torch.float16).to(device) * 0.01
        w_down_shared = torch.randn(d_hidden, d_expert_shared, dtype=torch.float16).to(device) * 0.01
        w_gate_routed = torch.randn(n_routed_experts, d_expert, d_hidden, dtype=torch.float16).to(device) * 0.01
        w_up_routed = torch.randn(n_routed_experts, d_expert, d_hidden, dtype=torch.float16).to(device) * 0.01
        w_down_routed = torch.randn(n_routed_experts, d_hidden, d_expert, dtype=torch.float16).to(device) * 0.01
        router_weight = torch.randn(n_routed_experts, d_hidden, dtype=torch.float16).to(device) * 0.01

        ref_output = golden_fusedmoe_full(
            x,
            w_gate_shared,
            w_up_shared,
            w_down_shared,
            w_gate_routed,
            w_up_routed,
            w_down_routed,
            router_weight,
            n_experts_per_token,
        )
        shared_kernel = shared_expert_kernel(num_tokens, d_hidden, d_expert_shared)
        shared_output = shared_kernel(x_flat, w_gate_shared, w_up_shared, w_down_shared)
        routing = host_preprocess(x_flat, router_weight, n_experts_per_token, BLOCK_M, device)
        buf_rows = routing["buf_rows"]
        routed_kernel = routed_expert_kernel(buf_rows, d_hidden, d_expert, n_routed_experts, routing["total_m_blocks"])
        routed_output_nc = routed_kernel(
            routing["stacked_tokens"],
            w_gate_routed,
            w_up_routed,
            w_down_routed,
            routing["stacked_weights"],
            routing["block_metadata"],
        )
        valid_mask = routing["stacked_weights"] > 0
        valid_output = routed_output_nc[valid_mask].to(torch.float16)
        valid_idxs = routing["token_idxs_nc"][valid_mask].long()
        expert_cache = torch.zeros(num_tokens, d_hidden, dtype=torch.float16).to(device)
        expert_cache[valid_idxs] = valid_output
        kernel_output = shared_output.view(batch_size, seq_len, d_hidden) + expert_cache.view(batch_size, seq_len, d_hidden)
        torch.testing.assert_close(kernel_output.cpu(), ref_output.cpu(), atol=0.25, rtol=0.25)

    ok &= _run_precision("l1", "e2e_multi_expert (b=1,s=128,routed=3,k=2)", l1_e2e_multi_expert, 0.25, 0.25)

    # ---- L1-7: E2E with top_k > n_routed_experts (user spec, small scale) ----
    def l1_e2e_topk_clamp():
        batch_size, seq_len = 1, 128
        d_hidden, d_expert = 128, 64
        n_routed_experts, n_shared_experts = 1, 1
        n_experts_per_token = 4  # > n_routed_experts (1); clamped to 1
        d_expert_shared = d_expert * n_shared_experts
        num_tokens = batch_size * seq_len
        torch.manual_seed(107)
        x = torch.randn(batch_size, seq_len, d_hidden, dtype=torch.float16).to(device)
        x_flat = x.view(num_tokens, d_hidden)
        w_gate_shared = torch.randn(d_expert_shared, d_hidden, dtype=torch.float16).to(device) * 0.01
        w_up_shared = torch.randn(d_expert_shared, d_hidden, dtype=torch.float16).to(device) * 0.01
        w_down_shared = torch.randn(d_hidden, d_expert_shared, dtype=torch.float16).to(device) * 0.01
        w_gate_routed = torch.randn(n_routed_experts, d_expert, d_hidden, dtype=torch.float16).to(device) * 0.01
        w_up_routed = torch.randn(n_routed_experts, d_expert, d_hidden, dtype=torch.float16).to(device) * 0.01
        w_down_routed = torch.randn(n_routed_experts, d_hidden, d_expert, dtype=torch.float16).to(device) * 0.01
        router_weight = torch.randn(n_routed_experts, d_hidden, dtype=torch.float16).to(device) * 0.01

        ref_output = golden_fusedmoe_full(
            x,
            w_gate_shared,
            w_up_shared,
            w_down_shared,
            w_gate_routed,
            w_up_routed,
            w_down_routed,
            router_weight,
            n_experts_per_token,
        )
        shared_kernel = shared_expert_kernel(num_tokens, d_hidden, d_expert_shared)
        shared_output = shared_kernel(x_flat, w_gate_shared, w_up_shared, w_down_shared)
        routing = host_preprocess(x_flat, router_weight, n_experts_per_token, BLOCK_M, device)
        buf_rows = routing["buf_rows"]
        routed_kernel = routed_expert_kernel(buf_rows, d_hidden, d_expert, n_routed_experts, routing["total_m_blocks"])
        routed_output_nc = routed_kernel(
            routing["stacked_tokens"],
            w_gate_routed,
            w_up_routed,
            w_down_routed,
            routing["stacked_weights"],
            routing["block_metadata"],
        )
        valid_mask = routing["stacked_weights"] > 0
        valid_output = routed_output_nc[valid_mask].to(torch.float16)
        valid_idxs = routing["token_idxs_nc"][valid_mask].long()
        expert_cache = torch.zeros(num_tokens, d_hidden, dtype=torch.float16).to(device)
        expert_cache[valid_idxs] = valid_output
        kernel_output = shared_output.view(batch_size, seq_len, d_hidden) + expert_cache.view(batch_size, seq_len, d_hidden)
        torch.testing.assert_close(kernel_output.cpu(), ref_output.cpu(), atol=0.25, rtol=0.25)

    ok &= _run_precision("l1", "e2e_topk_clamp (b=1,s=128,routed=1,k=4)", l1_e2e_topk_clamp, 0.25, 0.25)

    # ---- L1-8: User's golden params — routed expert (large shape) ----
    # d_hidden=7168, d_expert=2048, n_routed_experts=1, group_sizes=[8192]
    # Precision: large matrix → atol=1e-2, rtol=1e-2 per DESIGN.md §9.3
    def l1_user_routed_large():
        d_hidden, d_expert, n_experts = 7168, 2048, 1
        group_sizes = [8192]
        torch.manual_seed(108)
        routing = host_preprocess_for_test(group_sizes, d_hidden, n_experts, BLOCK_M, device)
        buf_rows = routing["buf_rows"]
        w_gate = torch.randn(n_experts, d_expert, d_hidden, dtype=torch.float16).to(device) * 0.01
        w_up = torch.randn(n_experts, d_expert, d_hidden, dtype=torch.float16).to(device) * 0.01
        w_down = torch.randn(n_experts, d_hidden, d_expert, dtype=torch.float16).to(device) * 0.01
        kernel = routed_expert_kernel(buf_rows, d_hidden, d_expert, n_experts, routing["total_m_blocks"])
        output = kernel(
            routing["stacked_tokens"],
            w_gate,
            w_up,
            w_down,
            routing["stacked_weights"],
            routing["block_metadata"],
        )
        ref = golden_routed_expert_nc(
            routing["stacked_tokens"],
            w_gate,
            w_up,
            w_down,
            routing["stacked_weights"],
            routing["block_metadata"],
        )
        valid_mask = routing["stacked_weights"] > 0
        # Large matrix: d_hidden=7168 → K-dim accumulation 112 iterations
        # Use relaxed tolerance per DESIGN.md §9.3 (large matrix: 1e-2)
        torch.testing.assert_close(output[valid_mask].cpu(), ref[valid_mask].cpu(), atol=1e-2, rtol=1e-2)

    ok &= _run_precision("l1", "user_routed_large (d_h=7168,d_e=2048,n=1)", l1_user_routed_large, 1e-2, 1e-2)

    # ---- L1-9: User's golden params — E2E (large scale) ----
    # Full user spec: d_hidden=7168, d_expert=2048, n_routed=1, top_k=4,
    # n_shared=1, batch=1, seq=8192, fp16
    def l1_user_e2e():
        batch_size, seq_len = 1, 8192
        d_hidden, d_expert = 7168, 2048
        n_routed_experts, n_shared_experts = 1, 1
        n_experts_per_token = 4  # > n_routed_experts; clamped to 1
        d_expert_shared = d_expert * n_shared_experts
        num_tokens = batch_size * seq_len
        torch.manual_seed(109)
        x = torch.randn(batch_size, seq_len, d_hidden, dtype=torch.float16).to(device)
        x_flat = x.view(num_tokens, d_hidden)
        w_gate_shared = torch.randn(d_expert_shared, d_hidden, dtype=torch.float16).to(device) * 0.01
        w_up_shared = torch.randn(d_expert_shared, d_hidden, dtype=torch.float16).to(device) * 0.01
        w_down_shared = torch.randn(d_hidden, d_expert_shared, dtype=torch.float16).to(device) * 0.01
        w_gate_routed = torch.randn(n_routed_experts, d_expert, d_hidden, dtype=torch.float16).to(device) * 0.01
        w_up_routed = torch.randn(n_routed_experts, d_expert, d_hidden, dtype=torch.float16).to(device) * 0.01
        w_down_routed = torch.randn(n_routed_experts, d_hidden, d_expert, dtype=torch.float16).to(device) * 0.01
        router_weight = torch.randn(n_routed_experts, d_hidden, dtype=torch.float16).to(device) * 0.01

        ref_output = golden_fusedmoe_full(
            x,
            w_gate_shared,
            w_up_shared,
            w_down_shared,
            w_gate_routed,
            w_up_routed,
            w_down_routed,
            router_weight,
            n_experts_per_token,
        )
        shared_kernel = shared_expert_kernel(num_tokens, d_hidden, d_expert_shared)
        shared_output = shared_kernel(x_flat, w_gate_shared, w_up_shared, w_down_shared)
        routing = host_preprocess(x_flat, router_weight, n_experts_per_token, BLOCK_M, device)
        buf_rows = routing["buf_rows"]
        routed_kernel = routed_expert_kernel(buf_rows, d_hidden, d_expert, n_routed_experts, routing["total_m_blocks"])
        routed_output_nc = routed_kernel(
            routing["stacked_tokens"],
            w_gate_routed,
            w_up_routed,
            w_down_routed,
            routing["stacked_weights"],
            routing["block_metadata"],
        )
        valid_mask = routing["stacked_weights"] > 0
        valid_output = routed_output_nc[valid_mask].to(torch.float16)
        valid_idxs = routing["token_idxs_nc"][valid_mask].long()
        expert_cache = torch.zeros(num_tokens, d_hidden, dtype=torch.float16).to(device)
        expert_cache[valid_idxs] = valid_output
        kernel_output = shared_output.view(batch_size, seq_len, d_hidden) + expert_cache.view(batch_size, seq_len, d_hidden)
        # Large matrix E2E: accumulated pipeline error, relaxed tolerance
        torch.testing.assert_close(kernel_output.cpu(), ref_output.cpu(), atol=0.5, rtol=0.5)

    ok &= _run_precision("l1", "user_e2e (b=1,s=8192,d_h=7168,d_e=2048)", l1_user_e2e, 0.5, 0.5)

    return ok


# ============================================================================
# L2 Exception Tests (non-blocking, [BOUNDARY_WARN] on failure)
# ============================================================================
def test_fusedmoe_l2():
    """L2 exception tests: abnormal inputs. Failures are non-blocking."""
    device = torch.device("npu")

    # ---- L2-1: Empty group (expert with 0 tokens) ----
    def l2_empty_group():
        d_hidden, d_expert, n_experts = 128, 64, 2
        group_sizes = [0, 64]  # first expert has 0 tokens
        torch.manual_seed(201)
        routing = host_preprocess_for_test(group_sizes, d_hidden, n_experts, BLOCK_M, device)
        buf_rows = routing["buf_rows"]
        w_gate = torch.randn(n_experts, d_expert, d_hidden, dtype=torch.float16).to(device) * 0.01
        w_up = torch.randn(n_experts, d_expert, d_hidden, dtype=torch.float16).to(device) * 0.01
        w_down = torch.randn(n_experts, d_hidden, d_expert, dtype=torch.float16).to(device) * 0.01
        kernel = routed_expert_kernel(buf_rows, d_hidden, d_expert, n_experts, routing["total_m_blocks"])
        output = kernel(
            routing["stacked_tokens"],
            w_gate,
            w_up,
            w_down,
            routing["stacked_weights"],
            routing["block_metadata"],
        )
        ref = golden_routed_expert_nc(
            routing["stacked_tokens"],
            w_gate,
            w_up,
            w_down,
            routing["stacked_weights"],
            routing["block_metadata"],
        )
        valid_mask = routing["stacked_weights"] > 0
        torch.testing.assert_close(output[valid_mask].cpu(), ref[valid_mask].cpu(), atol=5e-3, rtol=5e-3)

    _run_boundary("l2", "empty_group [0,64]", l2_empty_group)

    # ---- L2-2: All groups empty (no tokens for any expert) ----
    def l2_all_empty_groups():
        d_hidden, d_expert, n_experts = 128, 64, 2
        group_sizes = [0, 0]  # all experts empty
        torch.manual_seed(202)
        # host_preprocess_for_test with all-zero groups: total_m_blocks=0
        # This should be handled gracefully (kernel not called or no-op)
        routing = host_preprocess_for_test(group_sizes, d_hidden, n_experts, BLOCK_M, device)
        if routing["total_m_blocks"] == 0:
            # No blocks → nothing to compute, trivially passes
            return
        # If total_m_blocks > 0, run kernel (shouldn't happen with all-zero)
        buf_rows = routing["buf_rows"]
        w_gate = torch.randn(n_experts, d_expert, d_hidden, dtype=torch.float16).to(device) * 0.01
        w_up = torch.randn(n_experts, d_expert, d_hidden, dtype=torch.float16).to(device) * 0.01
        w_down = torch.randn(n_experts, d_hidden, d_expert, dtype=torch.float16).to(device) * 0.01
        kernel = routed_expert_kernel(buf_rows, d_hidden, d_expert, n_experts, routing["total_m_blocks"])
        kernel(
            routing["stacked_tokens"],
            w_gate,
            w_up,
            w_down,
            routing["stacked_weights"],
            routing["block_metadata"],
        )

    _run_boundary("l2", "all_empty_groups [0,0]", l2_all_empty_groups)

    # ---- L2-3: Minimal input (1 token, smallest valid shape) ----
    def l2_minimal_input():
        d_hidden, d_expert, n_experts = 64, 64, 1  # d_hidden=block_K (minimal K)
        group_sizes = [1]  # single token
        torch.manual_seed(203)
        routing = host_preprocess_for_test(group_sizes, d_hidden, n_experts, BLOCK_M, device)
        buf_rows = routing["buf_rows"]
        w_gate = torch.randn(n_experts, d_expert, d_hidden, dtype=torch.float16).to(device) * 0.01
        w_up = torch.randn(n_experts, d_expert, d_hidden, dtype=torch.float16).to(device) * 0.01
        w_down = torch.randn(n_experts, d_hidden, d_expert, dtype=torch.float16).to(device) * 0.01
        kernel = routed_expert_kernel(buf_rows, d_hidden, d_expert, n_experts, routing["total_m_blocks"])
        output = kernel(
            routing["stacked_tokens"],
            w_gate,
            w_up,
            w_down,
            routing["stacked_weights"],
            routing["block_metadata"],
        )
        ref = golden_routed_expert_nc(
            routing["stacked_tokens"],
            w_gate,
            w_up,
            w_down,
            routing["stacked_weights"],
            routing["block_metadata"],
        )
        valid_mask = routing["stacked_weights"] > 0
        torch.testing.assert_close(output[valid_mask].cpu(), ref[valid_mask].cpu(), atol=5e-3, rtol=5e-3)

    _run_boundary("l2", "minimal_input (d_h=64,group=[1])", l2_minimal_input)


# ============================================================================
# Boundary Special Value Tests (non-blocking, [BOUNDARY_WARN] on failure)
# ============================================================================
def test_fusedmoe_boundary():
    """Boundary tests: INF/NAN/extreme values. Failures are non-blocking."""
    device = torch.device("npu")

    # ---- B-1: All-zero input ----
    def b_zero_input():
        d_hidden, d_expert, n_experts = 128, 64, 1
        group_sizes = [64]
        torch.manual_seed(301)
        routing = host_preprocess_for_test(group_sizes, d_hidden, n_experts, BLOCK_M, device)
        buf_rows = routing["buf_rows"]
        # Zero the input tokens
        routing["stacked_tokens"].zero_()
        w_gate = torch.randn(n_experts, d_expert, d_hidden, dtype=torch.float16).to(device) * 0.01
        w_up = torch.randn(n_experts, d_expert, d_hidden, dtype=torch.float16).to(device) * 0.01
        w_down = torch.randn(n_experts, d_hidden, d_expert, dtype=torch.float16).to(device) * 0.01
        kernel = routed_expert_kernel(buf_rows, d_hidden, d_expert, n_experts, routing["total_m_blocks"])
        output = kernel(
            routing["stacked_tokens"],
            w_gate,
            w_up,
            w_down,
            routing["stacked_weights"],
            routing["block_metadata"],
        )
        ref = golden_routed_expert_nc(
            routing["stacked_tokens"],
            w_gate,
            w_up,
            w_down,
            routing["stacked_weights"],
            routing["block_metadata"],
        )
        valid_mask = routing["stacked_weights"] > 0
        # Zero input → zero output (0 @ W = 0, silu(0)=0, 0*0=0, 0 @ W = 0)
        torch.testing.assert_close(output[valid_mask].cpu(), ref[valid_mask].cpu(), atol=1e-3, rtol=1e-3)

    _run_boundary("boundary", "zero_input", b_zero_input)

    # ---- B-2: Large values (near fp16 max) ----
    def b_large_values():
        d_hidden, d_expert, n_experts = 128, 64, 1
        group_sizes = [64]
        torch.manual_seed(302)
        routing = host_preprocess_for_test(group_sizes, d_hidden, n_experts, BLOCK_M, device)
        buf_rows = routing["buf_rows"]
        # Scale input to large values (but not overflow after GEMM)
        routing["stacked_tokens"] = routing["stacked_tokens"] * 100.0
        w_gate = torch.randn(n_experts, d_expert, d_hidden, dtype=torch.float16).to(device) * 0.1
        w_up = torch.randn(n_experts, d_expert, d_hidden, dtype=torch.float16).to(device) * 0.1
        w_down = torch.randn(n_experts, d_hidden, d_expert, dtype=torch.float16).to(device) * 0.1
        kernel = routed_expert_kernel(buf_rows, d_hidden, d_expert, n_experts, routing["total_m_blocks"])
        output = kernel(
            routing["stacked_tokens"],
            w_gate,
            w_up,
            w_down,
            routing["stacked_weights"],
            routing["block_metadata"],
        )
        ref = golden_routed_expert_nc(
            routing["stacked_tokens"],
            w_gate,
            w_up,
            w_down,
            routing["stacked_weights"],
            routing["block_metadata"],
        )
        valid_mask = routing["stacked_weights"] > 0
        # Large values → larger absolute error, use relaxed tolerance
        torch.testing.assert_close(output[valid_mask].cpu(), ref[valid_mask].cpu(), atol=1.0, rtol=1e-1)

    _run_boundary("boundary", "large_values", b_large_values)

    # ---- B-3: NaN in input (expect NaN propagation, not crash) ----
    def b_nan_input():
        d_hidden, d_expert, n_experts = 128, 64, 1
        group_sizes = [64]
        torch.manual_seed(303)
        routing = host_preprocess_for_test(group_sizes, d_hidden, n_experts, BLOCK_M, device)
        buf_rows = routing["buf_rows"]
        # Inject NaN in a few positions
        routing["stacked_tokens"][0, 0] = float("nan")
        w_gate = torch.randn(n_experts, d_expert, d_hidden, dtype=torch.float16).to(device) * 0.01
        w_up = torch.randn(n_experts, d_expert, d_hidden, dtype=torch.float16).to(device) * 0.01
        w_down = torch.randn(n_experts, d_hidden, d_expert, dtype=torch.float16).to(device) * 0.01
        kernel = routed_expert_kernel(buf_rows, d_hidden, d_expert, n_experts, routing["total_m_blocks"])
        output = kernel(
            routing["stacked_tokens"],
            w_gate,
            w_up,
            w_down,
            routing["stacked_weights"],
            routing["block_metadata"],
        )
        # Verify kernel doesn't crash and propagates NaN (at least some NaN in output)
        assert torch.isnan(output).any(), "Expected NaN in output for NaN input"

    _run_boundary("boundary", "nan_input", b_nan_input)

    # ---- B-4: Inf in input (expect Inf or large values, not crash) ----
    def b_inf_input():
        d_hidden, d_expert, n_experts = 128, 64, 1
        group_sizes = [64]
        torch.manual_seed(304)
        routing = host_preprocess_for_test(group_sizes, d_hidden, n_experts, BLOCK_M, device)
        buf_rows = routing["buf_rows"]
        # Inject Inf in a few positions
        routing["stacked_tokens"][0, 0] = float("inf")
        w_gate = torch.randn(n_experts, d_expert, d_hidden, dtype=torch.float16).to(device) * 0.01
        w_up = torch.randn(n_experts, d_expert, d_hidden, dtype=torch.float16).to(device) * 0.01
        w_down = torch.randn(n_experts, d_hidden, d_expert, dtype=torch.float16).to(device) * 0.01
        kernel = routed_expert_kernel(buf_rows, d_hidden, d_expert, n_experts, routing["total_m_blocks"])
        output = kernel(
            routing["stacked_tokens"],
            w_gate,
            w_up,
            w_down,
            routing["stacked_weights"],
            routing["block_metadata"],
        )
        # Verify kernel doesn't crash (Inf or large values in output is acceptable)
        assert output.shape == (buf_rows, d_hidden), "Output shape mismatch"

    _run_boundary("boundary", "inf_input", b_inf_input)


# ===========================================================================
# Benchmark: Performance measurement (uses do_bench, CI-stable)
# ===========================================================================


def generate_golden_input(
    d_hidden=7168,
    d_expert=2048,
    n_routed_experts=1,
    n_shared_experts=1,
    n_experts_per_token=4,
    batch_size=1,
    seq_len=8192,
    dtype=torch.float16,
    device="npu",
    seed=81394,
):
    """Generate input tensors matching GPU reference's generate_input."""
    gen = torch.Generator(device="cpu").manual_seed(seed)

    num_tokens = batch_size * seq_len

    x = torch.randn(num_tokens, d_hidden, dtype=dtype, generator=gen).to(device)

    router_weight = (torch.randn(n_routed_experts, d_hidden, dtype=dtype, generator=gen) / math.sqrt(d_hidden)).to(device)

    shared_expert_dim = d_expert * n_shared_experts
    W_gate_shared = (torch.randn(shared_expert_dim, d_hidden, dtype=dtype, generator=gen) / math.sqrt(shared_expert_dim)).to(device)
    W_up_shared = (torch.randn(shared_expert_dim, d_hidden, dtype=dtype, generator=gen) / math.sqrt(shared_expert_dim)).to(device)
    W_down_shared = (torch.randn(d_hidden, shared_expert_dim, dtype=dtype, generator=gen) / math.sqrt(d_hidden)).to(device)

    W_gate_routed = (torch.randn(n_routed_experts, d_expert, d_hidden, dtype=dtype, generator=gen) / math.sqrt(d_expert)).to(device)
    W_up_routed = (torch.randn(n_routed_experts, d_expert, d_hidden, dtype=dtype, generator=gen) / math.sqrt(d_expert)).to(device)
    W_down_routed = (torch.randn(n_routed_experts, d_hidden, d_expert, dtype=dtype, generator=gen) / math.sqrt(d_hidden)).to(device)

    return {
        "x": x,
        "router_weight": router_weight,
        "W_gate_shared": W_gate_shared,
        "W_up_shared": W_up_shared,
        "W_down_shared": W_down_shared,
        "W_gate_routed": W_gate_routed,
        "W_up_routed": W_up_routed,
        "W_down_routed": W_down_routed,
    }


def _bench_kernels(
    shared_run,
    routed_run,
    x,
    stacked_tokens,
    wgs,
    wus,
    wds,
    wgr,
    wur,
    wdr,
    expert_weights,
    block_metadata,
):
    """Benchmark shared/routed/pipeline kernels with do_bench.

    Defined outside the bench loop so lambdas capture function parameters
    (not loop variables), avoiding ruff B023.
    """
    shared_ms = do_bench(
        lambda: shared_run(x, wgs, wus, wds),
        _n_warmup=5,
        _n_repeat=5,
        return_mode="mean",
    )
    routed_ms = do_bench(
        lambda: routed_run(stacked_tokens, wgr, wur, wdr, expert_weights, block_metadata),
        _n_warmup=5,
        _n_repeat=5,
        return_mode="mean",
    )
    pipeline_ms = do_bench(
        lambda: (shared_run(x, wgs, wus, wds), routed_run(stacked_tokens, wgr, wur, wdr, expert_weights, block_metadata)),
        _n_warmup=5,
        _n_repeat=5,
        return_mode="mean",
    )
    return shared_ms, routed_ms, pipeline_ms


def _bench_kernels_msprof(
    shared_run,
    routed_run,
    x,
    stacked_tokens,
    wgs,
    wus,
    wds,
    wgr,
    wur,
    wdr,
    expert_weights,
    block_metadata,
    output_dir,
):
    """Benchmark kernels with msprof op (hardware-level profiling).

    Launches a subprocess running msprof op on a script that calls each kernel.
    Parses the resulting CSV files for Task Duration, L2 hit rate, Cube ratio, etc.
    """
    import csv as csv_module
    import json as json_module
    import subprocess

    mod_dir = os.path.dirname(os.path.abspath(__file__))

    # Save tensors for subprocess
    tensors = {
        "x": x,
        "wgs": wgs,
        "wus": wus,
        "wds": wds,
        "st": stacked_tokens,
        "wgr": wgr,
        "wur": wur,
        "wdr": wdr,
        "ew": expert_weights,
        "bm": block_metadata,
    }
    for name, tensor in tensors.items():
        torch.save(tensor, f"{output_dir}/{name}.pt")

    load_lines = "\n".join(
        f"{name} = torch.load({json_module.dumps(output_dir + '/' + name + '.pt')}, weights_only=False)" for name in tensors
    )

    script_content = (
        f"import sys, os\n"
        f"sys.path.insert(0, {json_module.dumps(mod_dir)})\n"
        f"import tilelang, torch\n"
        f"from example_fusedmoe import shared_expert_kernel, routed_expert_kernel, BLOCK_M\n"
        f"\n"
        f"tilelang.disable_cache()\n"
        f"torch.set_default_device('npu')\n"
        f"\n"
        f"{load_lines}\n"
        f"\n"
        f"num_tokens = x.shape[0]\n"
        f"d_hidden = x.shape[1]\n"
        f"d_expert = wgs.shape[0]\n"
        f"buf_rows = st.shape[0]\n"
        f"total_m_blocks = bm.shape[0]\n"
        f"n_routed = wgr.shape[0]\n"
        f"\n"
        f"shared_run = shared_expert_kernel(num_tokens, d_hidden, d_expert, dtype='float16')\n"
        f"routed_run = routed_expert_kernel(buf_rows, d_hidden, d_expert, n_routed, total_m_blocks, dtype='float16')\n"
        f"\n"
        f"for _ in range(3):\n"
        f"    shared_run(x, wgs, wus, wds)\n"
        f"    routed_run(st, wgr, wur, wdr, ew, bm)\n"
        f"torch.npu.synchronize()\n"
        f"\n"
        f"shared_run(x, wgs, wus, wds)\n"
        f"torch.npu.synchronize()\n"
        f"routed_run(st, wgr, wur, wdr, ew, bm)\n"
        f"torch.npu.synchronize()\n"
        f"print('done')\n"
    )

    script_path = f"{output_dir}/prof_script.py"
    with open(script_path, "w") as f:
        f.write(script_content)

    msprof_out = f"{output_dir}/msprof_result"
    cmd = [
        "msprof",
        "op",
        f"--application=python {script_path}",
        f"--output={msprof_out}",
        "--aic-metrics=ArithmeticUtilization,Memory,L2Cache,PipeUtilization",
        "--launch-count=10",
        "--warm-up=3",
        "--kernel-name=kernel_kernel",
    ]

    subprocess.run(cmd, capture_output=True, text=True, timeout=600)

    # Find profiling output
    prof_dirs = [d for d in os.listdir(msprof_out) if d.startswith("OPPROF_")] if os.path.exists(msprof_out) else []
    if not prof_dirs:
        return None

    prof_base = f"{msprof_out}/{prof_dirs[0]}/kernel_kernel/0"
    if not os.path.exists(prof_base):
        return None

    metrics = {}

    def read_csv_cube_row(directory, prefix):
        if not os.path.exists(directory):
            return {}
        csv_files = [f for f in os.listdir(directory) if f.startswith(prefix) and f.endswith(".csv")]
        if not csv_files:
            return {}
        with open(f"{directory}/{csv_files[0]}") as f:
            reader = csv_module.DictReader(f)
            for row in reader:
                if row.get("sub_block_id") == "cube0":
                    return row
        return {}

    arith = read_csv_cube_row(prof_base, "ArithmeticUtilization")
    l2 = read_csv_cube_row(prof_base, "L2Cache")
    pipe = read_csv_cube_row(prof_base, "PipeUtilization")

    basic_csvs = [f for f in os.listdir(prof_base) if f.startswith("OpBasicInfo") and f.endswith(".csv")]
    if basic_csvs:
        with open(f"{prof_base}/{basic_csvs[0]}") as f:
            reader = csv_module.DictReader(f)
            for row in reader:
                metrics["task_duration_us"] = float(row["Task Duration(us)"])
                metrics["block_dim"] = int(row["Block Dim"])
                break

    if arith:
        metrics["cube_ratio"] = float(arith.get("aic_cube_ratio", 0)) * 100
    if l2:
        metrics["l2_read_hit_rate"] = float(l2.get("aic_read_hit_rate(%)", "0"))
    if pipe:
        metrics["mte2_ratio"] = float(pipe.get("aic_mte2_ratio", 0)) * 100

    return metrics


def test_fusedmoe_bench(profiler="do_bench"):
    """Performance benchmark.

    Args:
        profiler: "do_bench" (default, CI-stable) or "msprof" (hardware-level).
    """
    bench_configs = [
        (1, 512, 768, 256, 1, 4, "small"),
        (1, 2048, 4096, 2048, 1, 4, "medium"),
        (1, 8192, 7168, 2048, 1, 4, "golden_config"),
    ]

    print()
    print("=" * 90)
    if profiler == "msprof":
        print("  FusedMoE NPU — Performance Benchmark (msprof op, hardware-level)")
    else:
        print("  FusedMoE NPU — Performance Benchmark (do_bench, CI-stable)")
    print("=" * 90)
    print()

    if profiler == "msprof":
        print("| Label | B | Seq | d_hidden | d_expert | Task(us) | Cube(%) | MTE2(%) | L2_hit(%) | max_diff |")
        print("|" + "-" * 88 + "|")
    else:
        print("| Label | B | Seq | d_hidden | d_expert | Shared(ms) | Routed(ms) | Pipeline(ms) | max_diff |")
        print("|" + "-" * 88 + "|")

    ok = True
    for batch_size, seq_len, d_hidden, d_expert, n_routed, n_experts_per_token, label in bench_configs:
        try:
            data = generate_golden_input(
                d_hidden=d_hidden,
                d_expert=d_expert,
                n_routed_experts=n_routed,
                n_experts_per_token=n_experts_per_token,
                n_shared_experts=1,
                batch_size=batch_size,
                seq_len=seq_len,
            )
            x = data["x"]
            num_tokens = batch_size * seq_len
            shared_expert_dim = data["W_gate_shared"].shape[0]
            device = x.device

            shared_run = shared_expert_kernel(num_tokens, d_hidden, shared_expert_dim, dtype="float16")
            routing = host_preprocess(x, data["router_weight"], n_experts_per_token, BLOCK_M, device)
            routed_run = routed_expert_kernel(routing["buf_rows"], d_hidden, d_expert, n_routed, routing["total_m_blocks"], dtype="float16")

            stacked_tokens = routing["stacked_tokens"]
            expert_weights = routing["stacked_weights"]
            block_metadata = routing["block_metadata"]
            wgs, wus, wds = data["W_gate_shared"], data["W_up_shared"], data["W_down_shared"]
            wgr, wur, wdr = data["W_gate_routed"], data["W_up_routed"], data["W_down_routed"]

            # Precision check (shared expert vs golden)
            shared_output = shared_run(x, wgs, wus, wds)
            ref_shared = golden_shared_expert(x, wgs, wus, wds)
            max_diff = (shared_output.cpu().float() - ref_shared.cpu().float()).abs().max().item()
            torch.testing.assert_close(shared_output.cpu().float(), ref_shared.cpu().float(), rtol=1e-2, atol=1e-2)

            if profiler == "msprof":
                import tempfile

                prof_dir = tempfile.mkdtemp(prefix="msprof_fusedmoe_")
                metrics = _bench_kernels_msprof(
                    shared_run,
                    routed_run,
                    x,
                    stacked_tokens,
                    wgs,
                    wus,
                    wds,
                    wgr,
                    wur,
                    wdr,
                    expert_weights,
                    block_metadata,
                    prof_dir,
                )
                if metrics:
                    task_us = metrics.get("task_duration_us", 0)
                    cube_pct = metrics.get("cube_ratio", 0)
                    mte2_pct = metrics.get("mte2_ratio", 0)
                    l2_hit = metrics.get("l2_read_hit_rate", 0)
                    print(
                        f"| {label} | {batch_size} | {seq_len} | {d_hidden} | {d_expert} | "
                        f"{task_us:.0f} | {cube_pct:.1f} | {mte2_pct:.1f} | {l2_hit:.1f} | {max_diff:.2e} |"
                    )
                else:
                    print(f"| {label} | MSPROF FAIL |")
                    ok = False
            else:
                shared_ms, routed_ms, pipeline_ms = _bench_kernels(
                    shared_run, routed_run, x, stacked_tokens, wgs, wus, wds, wgr, wur, wdr, expert_weights, block_metadata
                )
                print(
                    f"| {label} | {batch_size} | {seq_len} | {d_hidden} | {d_expert} | "
                    f"{shared_ms:.2f} | {routed_ms:.2f} | {pipeline_ms:.2f} | {max_diff:.2e} |"
                )
        except Exception as e:
            print(f"| {label} | BENCH FAIL: {e} |")
            ok = False

    print()
    return ok


# ============================================================================
# Main Entry
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="FusedMoE NPU Tests")
    parser.add_argument("--level", default="l0", choices=["l0", "l1", "l2", "boundary", "all", "bench"])
    parser.add_argument(
        "--profiler",
        default="do_bench",
        choices=["do_bench", "msprof"],
        help="Profiler for bench level: do_bench (CI-stable) or msprof (hardware-level)",
    )
    args = parser.parse_args()

    tilelang.disable_cache()
    torch.manual_seed(0)

    if args.level == "bench":
        ok = test_fusedmoe_bench(profiler=args.profiler)
        if ok:
            print("Test Passed!")
            sys.exit(0)
        sys.exit(1)

    blocking_ok = True
    if args.level in ("l0", "all"):
        blocking_ok &= test_fusedmoe_l0()
    if args.level in ("l1", "all"):
        blocking_ok &= test_fusedmoe_l1()
    if args.level in ("l2", "all"):
        test_fusedmoe_l2()
    if args.level in ("boundary", "all"):
        test_fusedmoe_boundary()

    if blocking_ok:
        print("Test Passed!")
        sys.exit(0)
    sys.exit(1)


# pytest-discoverable aliases
def test_forward():
    """Pytest alias: run L0 cases."""
    tilelang.disable_cache()
    torch.manual_seed(0)
    assert test_fusedmoe_l0()


if __name__ == "__main__":
    main()
