"""fla_chunk_o example test: representative L0 + L1 cases with PyTorch golden.

q/v/A/h bf16 + g fp32 -> o bf16 (B, S, H, DV).

Precision: mixed-tolerance dual gate inlined
(|actual - golden| <= atol + rtol * |golden| element-wise, plus
matched_ratio >= 0.99 and max_abs_error <= 1e0 overall).
"""

import os
import sys

import tilelang
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from example_fla_chunk_o import fla_chunk_o  # noqa: E402


# ============================================================================
# Golden reference (PyTorch chunk loop, fp32 domain, torch.exp2 semantics)
# ============================================================================
def fla_chunk_o_ref(q, v, g, A, h, scale=None, chunk_size=64):
    """fla_chunk_o golden. q[B,S,H,DK]/v[B,S,H,DV]/A[B,S,H,C]/h[B,NT,H,DK,DV]
    bf16, g[B,S,H,DK] fp32 -> o[B,S,H,DV] bf16. C = chunk_size = A.shape[-1]."""
    B, S, H, DK = q.shape
    DV = v.shape[-1]
    C = A.shape[-1]
    assert chunk_size == C and S % C == 0, "S must be divisible by chunk_size"
    if scale is None:
        scale = 1.0 / DK**0.5
    NT = S // C
    qf, vf, Af, hf = q.float(), v.float(), A.float(), h.float()
    M = torch.tril(torch.ones(C, C, dtype=torch.bool), 0)  # incl diag (t >= j)
    o = torch.empty(B, S, H, DV, dtype=v.dtype)
    for i in range(NT):
        sl = slice(i * C, (i + 1) * C)
        qc, gc = qf[:, sl], g[:, sl]  # [B,C,H,DK]
        Ac, vc, hc = Af[:, sl], vf[:, sl], hf[:, i]  # [B,C,H,C]/[B,C,H,DV]/[B,H,DK,DV]
        qg = qc * scale * torch.exp2(gc)  # [B,C,H,DK]
        o_inter = torch.einsum("bchk,bhkd->bchd", qg, hc)  # inter-chunk
        # intra-chunk (tril incl diag over the chunk dims (t, j); upstream
        # semantics m_s[t, j] = t >= j, b_o += dot(where(m_s, A, 0), v)).
        # where (not * mask): the strict upper triangle must be zeroed even
        # when A carries Inf/NaN there (Inf*0 = NaN would leak through a
        # multiplicative mask).
        Am = torch.where(M[None, :, None, :], Ac, torch.zeros((), dtype=Ac.dtype))  # [B,C,H,C]
        o_intra = torch.einsum("bthj,bjhd->bthd", Am, vc)
        o[:, sl] = (o_inter + o_intra).to(v.dtype)
    return o


# ============================================================================
# Input construction (CPU randn + single H2D; aligned with the upstream
# prepare_input: A/h are randn too -- the operator only consumes them)
# ============================================================================
def make_inputs(B, S, H, DK, DV, chunk, seed):
    gen = torch.Generator().manual_seed(seed)
    q = torch.randn(B, S, H, DK, generator=gen)
    v = torch.randn(B, S, H, DV, generator=gen)
    g = torch.randn(B, S, H, DK, generator=gen)
    A = torch.randn(B, S, H, chunk, generator=gen)
    h = torch.randn(B, S // chunk, H, DK, DV, generator=gen)
    q, v, A, h = (t.to(torch.bfloat16) for t in (q, v, A, h))
    cpu = (q, v, g, A, h)
    npu = tuple(t.npu() for t in cpu)
    return npu, cpu


def check_output(name, out, ref, atol, rtol, max_abs_limit, required_ratio):
    """Inlined mixed-tolerance dual gate (element-wise |a-g| <= atol+rtol*|g|,
    overall matched_ratio >= required and max_abs <= limit)."""
    y_cpu = out.detach().cpu().float()
    ref_cpu = ref.detach().cpu().float()
    # Structural gate: actual's non-finite positions must match the golden's
    # (catches e.g. an Inf*0=NaN leaking through a multiplicative mask).
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
    tilelang.cache.clear_cache()
    torch.manual_seed(0)

    # Representative configs: L0 minimal gate + L1 benchmark shape
    test_configs = [
        ("l0_min", (1, 128, 2, 64, 64), 1001, "L0"),
        ("l1_bench", (1, 8192, 64, 128, 128), 2102, "L1"),
    ]

    for name, shape, seed, level in test_configs:
        B, S, H, DK, DV = shape
        print(f"Testing fla_chunk_o {level} {name} with shape={shape}")
        npu_in, cpu_in = make_inputs(B, S, H, DK, DV, 64, seed)
        o = fla_chunk_o(*npu_in)
        torch.npu.synchronize()
        print("Init successful!")
        ref = fla_chunk_o_ref(*cpu_in)
        # bf16 mixed tolerance
        atol, rtol, max_abs_limit, required_ratio = 2**-10, 2**-6, 1e0, 0.99
        check_output("o", o, ref, atol, rtol, max_abs_limit, required_ratio)

    # Regression (PR review): Inf confined to the STRICT UPPER triangle of A
    # must be zeroed by the mask (upstream tl.where semantics). With q=h=g=0,
    # v=1 and A=0 except upper-triangle Inf, the output must be exactly 0
    # everywhere; a multiplicative mask turns Inf into NaN (Inf*0) and
    # pollutes the affected chunk rows.
    B, S, H, DK, DV, CS = 1, 64, 1, 64, 64, 64
    q = torch.zeros(B, S, H, DK)
    v = torch.ones(B, S, H, DV)
    g = torch.zeros(B, S, H, DK)
    A = torch.zeros(B, S, H, CS)
    h = torch.zeros(B, S // CS, H, DK, DV)
    A[0, 0, 0, 1] = float("inf")  # strict upper triangle of chunk 0
    cpu = (q.to(torch.bfloat16), v.to(torch.bfloat16), g, A.to(torch.bfloat16), h.to(torch.bfloat16))
    npu = tuple(t.npu() for t in cpu)
    print(f"Testing fla_chunk_o regression upper-tri Inf with shape={(B, S, H, DK, DV)}")
    o = fla_chunk_o(*npu)
    torch.npu.synchronize()
    print("Init successful!")
    ref = fla_chunk_o_ref(*cpu)
    atol, rtol, max_abs_limit, required_ratio = 2**-10, 2**-6, 1e0, 0.99
    check_output("o", o, ref, atol, rtol, max_abs_limit, required_ratio)

    print("Kernel Output Match!")
