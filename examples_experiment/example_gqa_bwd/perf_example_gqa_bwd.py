"""
Performance benchmark for GQA Flash Attention (github version).

Tests:
  1. TileLang Forward v1 (flashattn_fwd)
  2. TileLang Forward v4 (flashattn_fwd_v4, L0 double buffer)
  3. TileLang Backward pipeline (flashattn_bwd_pipeline)
  4. PyTorch baseline (forward + backward)

Usage:
  python perf_example_gqa_bwd.py
  python perf_example_gqa_bwd.py --causal
  python perf_example_gqa_bwd.py --batch 4 --n_ctx 2048
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import tilelang
import torch
import torch.nn.functional as F
from tilelang.profiler import do_bench  # noqa: E402

from example_gqa_bwd import (  # noqa: E402
    flashattn_fwd,
    flashattn_fwd_v4,
    flashattn_bwd_preprocess,
    flashattn_bwd_pipeline,
    ref_program,
    ref_bwd,
    NUM_CORES,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--h", type=int, default=32)
    parser.add_argument("--n_ctx", type=int, default=1024)
    parser.add_argument("--d_head_qk", type=int, default=192)
    parser.add_argument("--d_head_v", type=int, default=128)
    parser.add_argument("--groups", type=int, default=16)
    parser.add_argument("--causal", action="store_true")
    args = parser.parse_args()

    tilelang.disable_cache()
    torch.set_default_device("npu")
    torch.manual_seed(42)

    B = args.batch
    H = args.h
    N = args.n_ctx
    D_qk = args.d_head_qk
    D_v = args.d_head_v
    groups = args.groups
    causal = args.causal
    H_kv = H // groups

    # FLOPS calculation
    fwd_flops = 2.0 * B * H * N * N * (D_qk + D_v)
    bwd_flops = 2.0 * B * H * N * N * (3 * D_qk + 2 * D_v)
    total_flops = fwd_flops + bwd_flops
    if causal:
        fwd_flops *= 0.5
        bwd_flops *= 0.5
        total_flops *= 0.5

    # ---- Allocate tensors (BHSD layout) ----
    Q = torch.randn(B, H, N, D_qk, dtype=torch.float16, device="npu")
    K = torch.randn(B, H_kv, N, D_qk, dtype=torch.float16, device="npu")
    V = torch.randn(B, H_kv, N, D_v, dtype=torch.float16, device="npu")
    dO = torch.randn(B, H, N, D_v, dtype=torch.float16, device="npu")

    print()
    print("=" * 70)
    print(f"  Config: B={B} H={H} H_kv={H_kv} N={N} D_qk={D_qk} D_v={D_v}")
    print(f"          groups={groups} causal={causal} dtype=fp16")
    print("=" * 70)

    # ============================================================
    # 0. Correctness check before bench (so we don't bench a broken kernel)
    #    Forward v4 vs PyTorch golden + Backward pipeline vs PyTorch golden.
    # ============================================================
    atol = 1e-2
    bM_v4, bN_v4 = 32, 64
    num_stages_fwd = 8
    cross_interval_fwd = 2
    fwd_v4_mod = flashattn_fwd_v4(
        B,
        H,
        N,
        D_qk,
        D_v,
        causal,
        bM_v4,
        bN_v4,
        groups,
        num_stages_fwd,
        cross_interval_fwd,
    )
    ws1_fwd = torch.empty(NUM_CORES, num_stages_fwd, bM_v4, bN_v4, dtype=torch.float32, device="npu")
    ws2_fwd = torch.empty(NUM_CORES, num_stages_fwd, bM_v4, bN_v4, dtype=torch.float16, device="npu")
    ws3_fwd = torch.empty(NUM_CORES, num_stages_fwd, bM_v4, D_v, dtype=torch.float32, device="npu")
    O_npu, lse_npu = fwd_v4_mod(Q, K, V, ws1_fwd, ws2_fwd, ws3_fwd)
    torch.npu.synchronize()

    O_ref = ref_program(Q, K, V, causal, groups)
    fwd_max_diff = (O_npu.float() - O_ref.float()).abs().max().item()
    print(f"  correctness: fwd_max_diff={fwd_max_diff:.6e} (atol={atol})")
    if fwd_max_diff >= atol:
        print(f"  [ERROR] forward correctness check failed: max_diff={fwd_max_diff} >= atol={atol}")
        sys.exit(1)

    # Backward correctness
    prep_mod = flashattn_bwd_preprocess(B, H, N, D_v, blk=32)
    Delta_npu = prep_mod(O_npu, dO)
    torch.npu.synchronize()

    D_qk_padded = ((D_qk + 127) // 128) * 128
    bM_bwd, bN_bwd = 64, 64 if causal else 32
    num_stages_bwd = 8
    dQ_raw = torch.zeros(B, H, N, D_qk_padded, dtype=torch.float32, device="npu")
    dK_raw = torch.zeros(B, H_kv, N, D_qk_padded, dtype=torch.float32, device="npu")
    dV_raw = torch.zeros(B, H_kv, N, D_v, dtype=torch.float32, device="npu")
    bwd_block_num = (N // bM_bwd) * H * B
    ws1_bwd = torch.empty(bwd_block_num, num_stages_bwd, bM_bwd, bN_bwd, dtype=torch.float32, device="npu")
    ws2_bwd = torch.empty(bwd_block_num, num_stages_bwd, bM_bwd, bN_bwd, dtype=torch.float16, device="npu")
    ws3_bwd = torch.empty(bwd_block_num, num_stages_bwd, bN_bwd, max(D_qk_padded, D_v), dtype=torch.float32, device="npu")
    bwd_mod = flashattn_bwd_pipeline(B, H, N, D_qk, D_v, causal, bM_bwd, bN_bwd, groups, num_stages_bwd)

    Q_padded = torch.zeros(B, H, N, D_qk_padded, dtype=torch.float16, device="npu")
    Q_padded[:, :, :, :D_qk] = Q
    K_padded = torch.zeros(B, H_kv, N, D_qk_padded, dtype=torch.float16, device="npu")
    K_padded[:, :, :, :D_qk] = K
    bwd_mod(Q_padded, K_padded, V, dO, lse_npu, Delta_npu, dQ_raw, dK_raw, dV_raw, ws1_bwd, ws2_bwd, ws3_bwd)
    torch.npu.synchronize()

    dQ_ref, dK_ref, dV_ref = ref_bwd(Q, K, V, dO, causal, groups)
    bwd_max_diff = max(
        (dV_raw.half().float() - dV_ref.float()).abs().max().item(),
        (dK_raw[:, :, :, :D_qk].half().float() - dK_ref.float()).abs().max().item(),
        (dQ_raw[:, :, :, :D_qk].half().float() - dQ_ref.float()).abs().max().item(),
    )
    print(f"  correctness: bwd_max_diff={bwd_max_diff:.6e} (atol={atol})")
    if bwd_max_diff >= atol:
        print(f"  [ERROR] backward correctness check failed: max_diff={bwd_max_diff} >= atol={atol}")
        sys.exit(1)
    print(f"  correctness: PASS (fwd={fwd_max_diff:.6e}, bwd={bwd_max_diff:.6e})")

    # ============================================================
    # Benchmark using tilelang.profiler.do_bench (standard NPU profiler).
    # Uses _n_warmup=5, _n_repeat=5 (same as perf_gqa_fwd_varlen.py).
    # do_bench handles NPU synchronization properly between iterations,
    # avoiding aicore timeout from tight 250-iter loops.
    # ============================================================

    # 1. TileLang Forward v1
    D_qk_padded_check = ((D_qk + 127) // 128) * 128
    skip_v1 = causal and D_qk_padded_check > 128
    if skip_v1:
        lat_fwd_v1 = float("nan")
        print(f"  [INFO] Skipping Forward v1 (causal + D_qk_padded={D_qk_padded_check} > 128, known limitation)")
    else:
        bM_v1, bN_v1 = 64, 64
        fwd_v1_mod = flashattn_fwd(B, H, N, D_qk, D_v, causal, bM_v1, bN_v1, groups)
        lat_fwd_v1 = do_bench(lambda: fwd_v1_mod(Q, K, V), _n_warmup=5, _n_repeat=5, return_mode="mean")

    # 2. TileLang Forward v4
    lat_fwd_v4 = do_bench(
        lambda: fwd_v4_mod(Q, K, V, ws1_fwd, ws2_fwd, ws3_fwd),
        _n_warmup=5,
        _n_repeat=5,
        return_mode="mean",
    )

    # 3. TileLang Backward (pipeline)
    #    dK/dV use atomic_add to GM — must zero before each call to prevent
    #    accumulation across bench iterations.
    def _run_bwd():
        dK_raw.zero_()
        dV_raw.zero_()
        bwd_mod(Q_padded, K_padded, V, dO, lse_npu, Delta_npu, dQ_raw, dK_raw, dV_raw, ws1_bwd, ws2_bwd, ws3_bwd)

    lat_bwd = do_bench(_run_bwd, _n_warmup=5, _n_repeat=5, return_mode="mean")

    # 4. PyTorch baseline
    q_r = Q.float()
    k_r = K.float().repeat_interleave(groups, dim=1)
    v_r = V.float().repeat_interleave(groups, dim=1)

    def _run_ref_fwd():
        scores = torch.matmul(q_r, k_r.transpose(-2, -1)) * (1.0 / D_qk**0.5)
        if causal:
            mask = torch.tril(torch.ones(N, N, device=scores.device, dtype=torch.bool))
            scores = scores.masked_fill(~mask.unsqueeze(0).unsqueeze(0), float("-inf"))
        P = F.softmax(scores, dim=-1)
        torch.matmul(P, v_r)

    lat_ref_fwd = do_bench(_run_ref_fwd, _n_warmup=5, _n_repeat=5, return_mode="mean")

    def _run_ref_fwd_bwd():
        q2 = Q.float().requires_grad_(True)
        k2 = K.float().repeat_interleave(groups, dim=1).requires_grad_(True)
        v2 = V.float().repeat_interleave(groups, dim=1).requires_grad_(True)
        scores = torch.matmul(q2, k2.transpose(-2, -1)) * (1.0 / D_qk**0.5)
        if causal:
            mask = torch.tril(torch.ones(N, N, device=scores.device, dtype=torch.bool))
            scores = scores.masked_fill(~mask.unsqueeze(0).unsqueeze(0), float("-inf"))
        P = F.softmax(scores, dim=-1)
        O2 = torch.matmul(P, v2)
        O2.backward(dO.float())

    lat_ref_e2e = do_bench(_run_ref_fwd_bwd, _n_warmup=3, _n_repeat=3, return_mode="mean")

    # ============================================================
    # Print results
    # ============================================================
    print()
    print(f"  {'Kernel':<32} {'Latency':>10} {'TFlops':>10}")
    print(f"  {'-' * 55}")
    if skip_v1:
        print(f"  {'TileLang Forward v1':<32} {'SKIPPED':>10}         -")
    else:
        print(f"  {'TileLang Forward v1':<32} {lat_fwd_v1:>8.2f} ms  {fwd_flops / lat_fwd_v1 * 1e-9:>8.2f}")
    print(f"  {'TileLang Forward v4':<32} {lat_fwd_v4:>8.2f} ms  {fwd_flops / lat_fwd_v4 * 1e-9:>8.2f}")
    print(f"  {'TileLang Backward (pipeline)':<32} {lat_bwd:>8.2f} ms  {bwd_flops / lat_bwd * 1e-9:>8.2f}")
    print(f"  {'TileLang Fwd(v4)+Bwd (raw)':<32} {lat_fwd_v4 + lat_bwd:>8.2f} ms  {total_flops / (lat_fwd_v4 + lat_bwd) * 1e-9:>8.2f}")
    print(f"  {'-' * 55}")
    print(f"  {'PyTorch Forward only':<32} {lat_ref_fwd:>8.2f} ms  {fwd_flops / lat_ref_fwd * 1e-9:>8.2f}")
    print(f"  {'PyTorch Fwd+Bwd (e2e)':<32} {lat_ref_e2e:>8.2f} ms  {total_flops / lat_ref_e2e * 1e-9:>8.2f}")
    print(f"  {'-' * 55}")
    sp_fwd_v4 = lat_ref_fwd / lat_fwd_v4
    sp_e2e_v4 = lat_ref_e2e / (lat_fwd_v4 + lat_bwd)
    print(f"  Speedup (v4 forward vs PyTorch fwd):  {sp_fwd_v4:.2f}x")
    print(f"  Speedup (v4 fwd+bwd vs PyTorch e2e):  {sp_e2e_v4:.2f}x")
    if not skip_v1:
        print(f"  v4 vs v1 forward speedup:             {lat_fwd_v1 / lat_fwd_v4:.2f}x")
    print("=" * 70)

    # CI compatibility: bench_test.sh marks a script PASSED only if stdout
    # contains "Test Passed!" / "Kernel Output Match!". Correctness was
    # already verified above (fwd_max_diff < atol and bwd_max_diff < atol).
    print("Test Passed!")


if __name__ == "__main__":
    main()
