"""fla_chunk_inter example test: representative L0 + bench cases with a PyTorch golden.

All fp32: q/k/g [B,S,H,DK], v/dv/do [B,S,H,DV], h/dh [B,S//64,H,DK,DV]
-> dq/dk/dw/dg [B,S,H,DK].

Precision: mixed-tolerance dual gate inlined
(|actual - golden| <= atol + rtol * |golden| element-wise, plus
matched_ratio >= 0.99 and max_abs_error <= 1e-2 overall).
"""

import os
import sys

import tilelang
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from example_fla_chunk_inter import fla_chunk_inter_bwd  # noqa: E402


# ============================================================================
# Golden reference (PyTorch chunk loop, fp32 domain, torch.exp2 semantics)
# ============================================================================
def fla_chunk_inter_ref(q, k, v, g, h, dv, do, dh, scale=1.0, chunk_size=64):
    """fla_chunk_inter golden. All fp32 -> (dq, dk, dw, dg), each [B,S,H,DK]."""
    B, S, H, DK = q.shape
    C = chunk_size
    assert S % C == 0 and h.shape[1] == S // C, "S must be divisible by chunk_size"
    NT = S // C
    qf, kf, vf, gf = q.float(), k.float(), v.float(), g.float()
    do_f, dv_f, h_f, dh_f = do.float(), dv.float(), h.float(), dh.float()
    dq = torch.empty(B, S, H, DK, dtype=torch.float32)
    dk = torch.empty_like(dq)
    dw = torch.empty_like(dq)
    dg = torch.empty_like(dq)
    for i in range(NT):
        sl = slice(i * C, (i + 1) * C)
        qc, kc, gc = qf[:, sl], kf[:, sl], gf[:, sl]  # [B,C,H,DK]
        vc, doc, dvc = vf[:, sl], do_f[:, sl], dv_f[:, sl]  # [B,C,H,DV]
        hi, dhi = h_f[:, i], dh_f[:, i]  # [B,H,DK,DV]
        gn = gf[:, (i + 1) * C - 1]  # [B,H,DK] (in-domain: always chunk last row)
        # Three GEMMs (fp32)
        dq_raw = torch.einsum("bthv,bhkv->bthk", doc, hi)  # do @ h^T
        dk_raw = torch.einsum("bthv,bhkv->bthk", vc, dhi)  # v  @ dh^T
        dw_blk = -torch.einsum("bthv,bhkv->bthk", dvc, hi)  # -(dv @ h^T)
        # Gating
        dq_blk = dq_raw * scale * torch.exp2(gc)  # [B,C,H,DK]
        dk_blk = dk_raw * torch.exp2(gn.unsqueeze(1) - gc)  # gn broadcast along chunk rows
        # dgk and dg
        dgk = (hi * dhi).sum(-1) * torch.exp2(gn)  # [B,H,DK]
        pp = dgk + (dk_blk * kc).sum(1)  # sum_t dk*k (gated dk)
        dg_blk = qc * dq_blk - kc * dk_blk  # [B,C,H,DK]
        dg_blk[:, -1] += pp  # m_last row injection (in-domain: always last row)
        dq[:, sl], dk[:, sl], dw[:, sl], dg[:, sl] = dq_blk, dk_blk, dw_blk, dg_blk
    return dq, dk, dw, dg


# ============================================================================
# Input construction (CPU randn aligned with the upstream prepare_input;
# single H2D per tensor)
# ============================================================================
def make_inputs(B, S, H, DK, DV, chunk, seed):
    gen = torch.Generator().manual_seed(seed)
    NT = S // chunk

    def rnd(*shape):
        return torch.randn(*shape, generator=gen, dtype=torch.float32)

    cpu = (
        rnd(B, S, H, DK),
        rnd(B, S, H, DK),
        rnd(B, S, H, DV),
        rnd(B, S, H, DK),
        rnd(B, NT, H, DK, DV),
        rnd(B, S, H, DV),
        rnd(B, S, H, DV),
        rnd(B, NT, H, DK, DV),
    )
    npu = tuple(t.npu() for t in cpu)
    return npu, cpu


def check_output(name, out, ref, atol, rtol, max_abs_limit, required_ratio):
    """Inlined mixed-tolerance dual gate (element-wise |a-g| <= atol+rtol*|g|,
    overall matched_ratio >= required and max_abs <= limit)."""
    y_cpu = out.detach().cpu().float()
    ref_cpu = ref.detach().cpu().float()
    assert torch.equal(torch.isfinite(y_cpu), torch.isfinite(ref_cpu)), (
        f"{name} non-finite structure mismatch: actual {(~torch.isfinite(y_cpu)).sum().item()}"
        f" vs golden {(~torch.isfinite(ref_cpu)).sum().item()} non-finite elements"
    )
    m = torch.isfinite(ref_cpu)
    abs_err = (y_cpu[m] - ref_cpu[m]).abs()
    ratio = (abs_err <= (atol + rtol * ref_cpu[m].abs())).float().mean().item()
    max_abs = abs_err.max().item()
    assert ratio >= required_ratio and max_abs <= max_abs_limit, f"{name} precision fail: ratio={ratio:.4f} max_abs={max_abs:.3e}"
    print(f"Test pass! {name}: matched_ratio={ratio:.4f} max_abs={max_abs:.3e}")


if __name__ == "__main__":
    tilelang.disable_cache()
    torch.manual_seed(0)

    # Representative configs: L0 minimal gate + bench shape
    test_configs = [
        ("l0_min", (1, 128, 2, 64, 64), 1001, "L0"),
        ("l1_bench", (1, 8192, 64, 128, 128), 2102, "L1"),
    ]

    for name, shape, seed, level in test_configs:
        B, S, H, DK, DV = shape
        print(f"Testing fla_chunk_inter {level} {name} with shape={shape}")
        npu_in, cpu_in = make_inputs(B, S, H, DK, DV, 64, seed)
        dq, dk, dw, dg = fla_chunk_inter_bwd(*npu_in)
        torch.npu.synchronize()
        print("Init successful!")
        ref = fla_chunk_inter_ref(*cpu_in)
        # fp32 mixed tolerance
        atol, rtol, max_abs_limit, required_ratio = 2**-16, 2**-10, 1e-2, 0.99
        for out_name, out, golden in zip(("dq", "dk", "dw", "dg"), (dq, dk, dw, dg), ref):
            check_output(out_name, out, golden, atol, rtol, max_abs_limit, required_ratio)

    print("Kernel Output Match!")
