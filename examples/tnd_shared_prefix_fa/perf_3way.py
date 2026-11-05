"""3-way perf benchmark: TileLang Expert kernel vs PyTorch (CANN) baseline.

Kernel-only: only the kernel call inside the do_bench lambda (inputs
pre-allocated, metadata reused).
E2E: metadata construction + kernel launch (per-call recompute), matching a
real inference step on the deduplicated TND layout.
PyTorch (CANN): per-request torch_npu.npu_fusion_attention with
KV = concat(shared prefix, own private) — the natural fallback when no fused
op exists for the deduplicated layout.

Usage:
    python perf_3way.py
"""

import math
import os
import sys

import tilelang
import torch
import torch_npu
from tilelang.profiler import do_bench

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tnd_shared_prefix_fa_expert import tnd_shared_prefix_fa_expert
from tnd_shared_prefix_fa_developer import build_block_metadata

CONFIGS = [
    # (tag, batch, q_head, kv_head, head_dim, sp_len, priv_len)
    ("small", 2, 4, 2, 64, 24, 150),
    ("business", 10, 14, 2, 64, 24, 150),
    ("p99", 10, 14, 2, 64, 24, 218),
]


def make_cann_baseline(Q, KS, VS, KP, VP, sp_len, priv_len, batch, q_head, kv_head, sm_scale):
    """Per-request npu_fusion_attention on the deduplicated TND layout.

    Each request must see shared prefix KV + its own private KV, so the
    fallback without a fused op is: concat per request, layout-convert to
    BNSD, call the CANN flash attention op, loop over requests.
    """
    group = q_head // kv_head

    def run():
        outs = []
        priv_start = 0
        for _b in range(batch):
            k_tnd = torch.cat([KS, KP[priv_start : priv_start + priv_len]], dim=0)
            v_tnd = torch.cat([VS, VP[priv_start : priv_start + priv_len]], dim=0)
            k_bnsd = k_tnd.repeat_interleave(group, dim=1).permute(1, 0, 2).unsqueeze(0)
            v_bnsd = v_tnd.repeat_interleave(group, dim=1).permute(1, 0, 2).unsqueeze(0)
            q_seg = Q[sp_len + priv_start : sp_len + priv_start + priv_len]
            q_bnsd = q_seg.permute(1, 0, 2).unsqueeze(0)
            o = torch_npu.npu_fusion_attention(
                q_bnsd,
                k_bnsd,
                v_bnsd,
                q_head,
                padding_mask=None,
                atten_mask=None,
                scale=sm_scale,
                keep_prob=1.0,
                input_layout="BNSD",
                pre_tockens=65535,
                next_tockens=65535,
                sparse_mode=0,
            )[0]
            outs.append(o.squeeze(0).permute(1, 0, 2))
            priv_start += priv_len
        return torch.cat(outs, dim=0)

    return run


def run_one(tag, batch, q_head, kv_head, head_dim, sp_len, priv_len):
    private_q_lens = [priv_len] * batch
    total_q = sp_len + sum(private_q_lens)
    total_private_kv = sum(private_q_lens)
    max_private_kv_len = max(private_q_lens)
    total_q_blocks = math.ceil(sp_len / 128) + sum(math.ceil(l / 128) for l in private_q_lens)
    sm_scale = 1.0 / (head_dim**0.5)
    block_M, block_N = 128, 64

    Q = torch.randn(total_q, q_head, head_dim, dtype=torch.float16).npu()
    KS = torch.randn(sp_len, kv_head, head_dim, dtype=torch.float16).npu()
    VS = torch.randn(sp_len, kv_head, head_dim, dtype=torch.float16).npu()
    KP = torch.randn(total_private_kv, kv_head, head_dim, dtype=torch.float16).npu()
    VP = torch.randn(total_private_kv, kv_head, head_dim, dtype=torch.float16).npu()
    bm = build_block_metadata(sp_len, private_q_lens, block_M, "npu")

    kernel = tnd_shared_prefix_fa_expert(
        q_head=q_head,
        kv_head=kv_head,
        head_dim=head_dim,
        shared_prefix_len=sp_len,
        max_private_kv_len=max_private_kv_len,
        total_q=total_q,
        total_private_kv=total_private_kv,
        total_q_blocks=total_q_blocks,
        block_M=block_M,
        block_N=block_N,
        sm_scale=sm_scale,
    )
    torch.npu.synchronize()

    def kernel_only():
        kernel(Q, KS, VS, KP, VP, bm)

    def e2e():
        bm2 = build_block_metadata(sp_len, private_q_lens, block_M, "cpu").npu()
        kernel(Q, KS, VS, KP, VP, bm2)

    baseline = make_cann_baseline(Q, KS, VS, KP, VP, sp_len, priv_len, batch, q_head, kv_head, sm_scale)

    lat_kernel = do_bench(kernel_only, _n_warmup=5, _n_repeat=20, return_mode="mean")
    lat_e2e = do_bench(e2e, _n_warmup=5, _n_repeat=10, return_mode="mean")
    lat_cann = do_bench(baseline, _n_warmup=3, _n_repeat=5, return_mode="mean")

    k_spd = lat_cann / lat_kernel
    e_spd = lat_cann / lat_e2e
    print(f"{tag:12s} {lat_kernel:>10.2f}ms {lat_e2e:>10.2f}ms {lat_cann:>12.2f}ms {k_spd:>8.2f}x {e_spd:>8.2f}x")


def main():
    tilelang.disable_cache()
    torch.manual_seed(0)

    print(f"{'':12s} {'Kernel-only':>12s} {'E2E':>12s} {'PyTorch(CANN)':>14s} {'Kern spd':>9s} {'E2E spd':>9s}")
    print("-" * 72)
    for cfg in CONFIGS:
        run_one(*cfg)
    print("Test Passed!")


if __name__ == "__main__":
    main()
