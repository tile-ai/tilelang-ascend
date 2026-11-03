"""fla_chunk_intra example test: representative L0 + L1 cases with PyTorch golden.

fwd: q/k/beta bf16 + gk fp32 + Akk_diag fp32 -> (Aqk, Akk) bf16
bwd: all fp32 -> (dq2, dk2, db2, dg2) fp32
     (+ host post-processing db = db2.sum(0) + db,
      dg = reverse chunk-local cumsum(dg2) -- upstream wrapper parity,
      not part of the kernels)

Precision: mixed-tolerance dual gate inlined per dtype
(|actual - golden| <= atol + rtol * |golden| element-wise, plus
matched_ratio >= 0.99 and max_abs_error <= limit overall).
"""

import os
import sys

import tilelang
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from example_fla_chunk_intra import (  # noqa: E402
    fla_chunk_intra_bwd_intra,
    fla_chunk_intra_fwd_inter_solve_fused,
)


# ============================================================================
# Chunk-local cumsum (upstream test-input construction semantics)
# ============================================================================
def chunk_local_cumsum(g, chunk=64, reverse=False):
    B, S, H, D = g.shape
    x = g.view(B, S // chunk, chunk, H, D)
    if reverse:
        x = x.flip(2).cumsum(2).flip(2)
    else:
        x = x.cumsum(2)
    return x.view(B, S, H, D)


# ============================================================================
# Golden reference: fwd (PyTorch fp64 rewrite)
# Unwritten regions (Aqk diagonal + block upper triangle) are explicit zeros.
# ============================================================================
def fla_chunk_intra_fwd_ref(q, k, gk, beta, Akk_diag, chunk_size=64, sub_chunk_size=16, scale=1.0):
    """fwd golden. q/k [B,S,H,DK] bf16, gk fp32, beta [B,S,H] bf16,
    Akk_diag [B,S,H,BC] fp32 -> (Aqk, Akk) [B,S,H,CS] bf16. CS=64, BC=16."""
    B, S, H, DK = q.shape
    CS, BC = chunk_size, sub_chunk_size
    assert S % CS == 0 and CS % BC == 0
    qf, kf, bf, gf, adf = (t.double() for t in (q, k, beta, gk, Akk_diag))
    Aqk = torch.zeros(B, S, H, CS, dtype=torch.float64)
    Akk_out = torch.zeros(B, S, H, CS, dtype=torch.float64)
    for c in range(S // CS):
        s0 = c * CS
        g_c, k_c, q_c = gf[:, s0 : s0 + CS], kf[:, s0 : s0 + CS], qf[:, s0 : s0 + CS]
        # (1) off-diagonal Aqk blocks (direct 2^(g_r - g_j) form, fp64-safe)
        for s2 in range(1, 4):
            r0 = s2 * BC
            for s1 in range(0, s2):
                j0 = s1 * BC
                e = torch.exp2(g_c[:, r0 : r0 + BC].unsqueeze(2) - g_c[:, j0 : j0 + BC].unsqueeze(1))
                val = torch.einsum("brhd,bjhd,brjhd->brjh", q_c[:, r0 : r0 + BC], k_c[:, j0 : j0 + BC], e) * scale
                Aqk[:, s0 + r0 : s0 + r0 + BC, :, j0 : j0 + BC] = val.permute(0, 1, 3, 2)
        # (2) M assembly: M[s,s] = I + strict_tril(A_s); M[s2,s1] = Akk-beta'[s2,s1]
        M = torch.zeros(B, CS, CS, H, dtype=torch.float64)
        Iblk = torch.eye(BC, dtype=torch.float64)
        Str = torch.tril(torch.ones(BC, BC, dtype=torch.float64), -1)
        for s in range(4):
            r0 = s * BC
            A_s = adf[:, s0 + r0 : s0 + r0 + BC, :, :].permute(0, 1, 3, 2)
            M[:, r0 : r0 + BC, r0 : r0 + BC, :] = A_s * Str[None, :, :, None] + Iblk[None, :, :, None]
            for s2 in range(s + 1, 4):
                r2 = s2 * BC
                e = torch.exp2(g_c[:, r2 : r2 + BC].unsqueeze(2) - g_c[:, r0 : r0 + BC].unsqueeze(1))
                akk = torch.einsum("brhd,bjhd,brjhd->brjh", k_c[:, r2 : r2 + BC], k_c[:, r0 : r0 + BC], e)
                M[:, r2 : r2 + BC, r0 : r0 + BC, :] = akk * bf[:, s0 + r2 : s0 + r2 + BC].unsqueeze(2)
        # (3) Akk = M^{-1} (unit lower block-triangular solve)
        Mi = M.permute(0, 3, 1, 2)
        eye = torch.eye(CS, dtype=torch.float64).expand(B, H, CS, CS).contiguous()
        Ai = torch.linalg.solve_triangular(Mi, eye, upper=False, unitriangular=True)
        Akk_out[:, s0 : s0 + CS, :, :] = Ai.permute(0, 2, 1, 3)
    return Aqk.to(q.dtype), Akk_out.to(q.dtype)


# ============================================================================
# Golden reference: bwd (unified-mask direct form, fp64)
# Returns (dq2, dk2, dg2, dbp); dbp is the db2.sum(0) equivalent (NK-free).
# ============================================================================
def fla_chunk_intra_bwd_ref(q, k, g, beta, dAqk, dAkk, dq, dk, db, dg, chunk_size=64, sub_chunk_size=16):
    """bwd golden. all fp32 -> (dq2, dk2, dg2, dbp) fp64."""
    B, S, H, DK = q.shape
    CS = chunk_size
    qf, kf, bf, gf = (t.double() for t in (q, k, beta, g))
    dAf, dKf, dqf, dkf, dbf, dgf = (t.double() for t in (dAqk, dAkk, dq, dk, db, dg))
    dq2 = torch.zeros(B, S, H, DK, dtype=torch.float64)
    dk2 = torch.zeros(B, S, H, DK, dtype=torch.float64)
    dg2 = torch.zeros(B, S, H, DK, dtype=torch.float64)
    dbp = torch.zeros(B, S, H, dtype=torch.float64)
    for c in range(S // CS):
        s0 = c * CS
        g_c, k_c, q_c, b_c = gf[:, s0 : s0 + CS], kf[:, s0 : s0 + CS], qf[:, s0 : s0 + CS], bf[:, s0 : s0 + CS]
        dA_c, dK_c = dAf[:, s0 : s0 + CS], dKf[:, s0 : s0 + CS]
        ar = torch.arange(CS)
        for i in range(CS):
            m_lo = (ar <= i).double()  # j <= i: (A)+(B) unified mask
            m_up = (ar >= i).double()  # j >= i: (D)+(E) unified mask
            eij = torch.exp2(g_c[:, i].unsqueeze(1) - g_c)  # 2^(g_i-g_j) [B,CS,H,DK]
            eji = torch.exp2(g_c - g_c[:, i].unsqueeze(1))  # 2^(g_j-g_i)
            # (A)+(B): dq2_pre[i] / dk2pre[i] (pre-beta)
            dq2[:, s0 + i] = torch.einsum("bhj,bjhk->bhk", dA_c[:, i] * m_lo, k_c * eij)
            dk2pre = torch.einsum("bhj,bjhk->bhk", dK_c[:, i] * m_lo, k_c * eij)
            # (D)+(E): dkt[i] (transposed side; beta_j folded into the dK term)
            dkt = torch.einsum("bjh,bjhk->bhk", dK_c[:, :, :, i] * m_up.unsqueeze(-1) * b_c, k_c * eji) + torch.einsum(
                "bjh,bjhk->bhk", dA_c[:, :, :, i] * m_up.unsqueeze(-1), q_c * eji
            )
            # tail ((F) post-beta; dg2 uses q*dq2_pre before the +dq add)
            dbp[:, s0 + i] = (dk2pre * k_c[:, i]).sum(-1)
            dk2b = dk2pre * b_c[:, i].unsqueeze(-1)
            dg2[:, s0 + i] = q_c[:, i] * dq2[:, s0 + i] + (dk2b - dkt) * k_c[:, i] + dgf[:, s0 + i]
            dq2[:, s0 + i] = dq2[:, s0 + i] + dqf[:, s0 + i]
            dk2[:, s0 + i] = dk2b + dkf[:, s0 + i] + dkt
    return dq2, dk2, dg2, dbp


# ============================================================================
# Input construction (CPU randn + single H2D; upstream prepare_input parity)
# ============================================================================
def make_fwd_inputs(B, S, H, DK, chunk, sub, seed):
    gen = torch.Generator().manual_seed(seed)
    q = torch.randn(B, S, H, DK, generator=gen).to(torch.bfloat16)
    k = torch.randn(B, S, H, DK, generator=gen).to(torch.bfloat16)
    beta = torch.randn(B, S, H, generator=gen).to(torch.bfloat16)
    gk = torch.nn.functional.logsigmoid(torch.randn(B, S, H, DK, generator=gen, dtype=torch.float32))
    gk = chunk_local_cumsum(gk, chunk)
    Akk_diag = torch.ones(B, S, H, sub, dtype=torch.float32)
    cpu = (q, k, gk, beta, Akk_diag)
    npu = tuple(t.npu() for t in cpu)
    return npu, cpu


def make_bwd_inputs(B, S, H, DK, chunk, seed):
    gen = torch.Generator().manual_seed(seed)
    q = torch.randn(B, S, H, DK, generator=gen, dtype=torch.float32)
    k = torch.randn(B, S, H, DK, generator=gen, dtype=torch.float32)
    g = torch.randn(B, S, H, DK, generator=gen, dtype=torch.float32)  # NOT cumsum'd (upstream parity)
    beta = torch.randn(B, S, H, generator=gen, dtype=torch.float32)
    dAqk = torch.randn(B, S, H, chunk, generator=gen, dtype=torch.float32)
    dAkk = torch.randn(B, S, H, chunk, generator=gen, dtype=torch.float32)
    dq = torch.randn(B, S, H, DK, generator=gen, dtype=torch.float32)
    dk = torch.randn(B, S, H, DK, generator=gen, dtype=torch.float32)
    db = torch.randn(B, S, H, generator=gen, dtype=torch.float32)
    dg = torch.randn(B, S, H, DK, generator=gen, dtype=torch.float32)
    cpu = (q, k, g, beta, dAqk, dAkk, dq, dk, db, dg)
    npu = tuple(t.npu() for t in cpu)
    return npu, cpu


def check_output(name, out, ref, atol, rtol, max_abs_limit, required_ratio):
    """Inlined mixed-tolerance dual gate (element-wise |a-g| <= atol+rtol*|g|,
    overall matched_ratio >= required and max_abs <= limit)."""
    y_cpu = out.detach().cpu().float()
    ref_cpu = ref.detach().cpu().float()
    m = torch.isfinite(ref_cpu)
    abs_err = (y_cpu[m] - ref_cpu[m]).abs()
    ratio = (abs_err <= (atol + rtol * ref_cpu[m].abs())).float().mean().item()
    max_abs = abs_err.max().item()
    assert ratio >= required_ratio and max_abs <= max_abs_limit, f"{name} precision fail: ratio={ratio:.4f} max_abs={max_abs:.3e}"
    print(f"Test pass! {name}: matched_ratio={ratio:.4f} max_abs={max_abs:.3e}")


if __name__ == "__main__":
    tilelang.cache.clear_cache()
    torch.manual_seed(0)

    # Representative configs: L0 minimal gates + L1 benchmark shapes
    test_configs = [
        ("l0_fwd_min", "fwd", (1, 64, 2, 128), 1001, "L0"),
        ("l0_bwd_min", "bwd", (1, 64, 2, 128), 1003, "L0"),
        ("l1_fwd_bench", "fwd", (1, 8192, 64, 128), 2101, "L1"),
        ("l1_bwd_bench", "bwd", (1, 8192, 8, 128), 2103, "L1"),
    ]

    for _name, kind, shape, seed, level in test_configs:
        B, S, H, DK = shape
        print(f"Testing fla_chunk_intra {kind} {level} with shape={shape}")
        if kind == "fwd":
            npu_in, cpu_in = make_fwd_inputs(B, S, H, DK, 64, 16, seed)
            aqk, akk = fla_chunk_intra_fwd_inter_solve_fused(*npu_in)
            torch.npu.synchronize()
            print("Init successful!")
            gaqk, gakk = fla_chunk_intra_fwd_ref(*cpu_in)
            # bf16 mixed tolerance
            atol, rtol, max_abs_limit, required_ratio = 2**-10, 2**-6, 1e0, 0.99
            check_output("Aqk", aqk, gaqk, atol, rtol, max_abs_limit, required_ratio)
            check_output("Akk", akk, gakk, atol, rtol, max_abs_limit, required_ratio)
        else:
            npu_in, cpu_in = make_bwd_inputs(B, S, H, DK, 64, seed)
            dq2, dk2, db2, dg2 = fla_chunk_intra_bwd_intra(*npu_in)
            torch.npu.synchronize()
            print("Init successful!")
            gdq2, gdk2, gdg2, gdbp = fla_chunk_intra_bwd_ref(*cpu_in)
            # host post-processing on both sides (upstream wrapper parity)
            db = db2.sum(0) + npu_in[8]
            dg = chunk_local_cumsum(dg2, 64, reverse=True)
            gdb = gdbp + cpu_in[8].double()
            gdg = chunk_local_cumsum(gdg2, 64, reverse=True)
            # fp32 mixed tolerance
            atol, rtol, max_abs_limit, required_ratio = 2**-16, 2**-10, 1e-2, 0.99
            check_output("dq2", dq2, gdq2, atol, rtol, max_abs_limit, required_ratio)
            check_output("dk2", dk2, gdk2, atol, rtol, max_abs_limit, required_ratio)
            check_output("dg2", dg2, gdg2, atol, rtol, max_abs_limit, required_ratio)
            check_output("db_post", db, gdb, atol, rtol, max_abs_limit, required_ratio)
            check_output("dg_post", dg, gdg, atol, rtol, max_abs_limit, required_ratio)

    print("Kernel Output Match!")
