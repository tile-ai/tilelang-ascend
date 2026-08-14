"""KDA L0: the recurrent (token-by-token) forward kernel.

This is the decode path: short sequences and single tokens, with
``initial_state`` in and ``final_state`` out, the state accumulated in fp32.
It is validated against two independent goldens -- the small-shape PyTorch
recurrence in ``kda_ref.py`` and FLA's ``naive_recurrent_kda``.

Three steps per token, matching ``kda_ref.kda_ref`` line for line:

    S <- Diag(exp(g_t)) S                    # per-channel row scaling
    S <- S + beta_t * k_t (v_t - S^T k_t)^T  # delta rule: only the residual is written
    o_t = S^T (scale * q_t)

Parallel decomposition
----------------------
``grid = B * HV``; one block per (batch, value head).  The two vector cores
split the V axis in half via ``vid``, each holding one ``[K, BV]`` half of the
state.

Splitting along V rather than K is the load-bearing choice: ``S^T k`` in the
delta rule reduces along K, so a K-split would need a cross-block reduction
every token.  A V-split leaves each half self-contained -- the two cores never
communicate.

Everything runs on the vector cores; the Cube is unused.  Every token-level
operation is matrix-vector shaped (M = 1) and cannot fill the Cube's 16x16x16
fractal.  Decode is memory-bound in any case.
"""

import torch
import tilelang
from tilelang import language as T

import kda_ref

pass_configs = {tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True}


@tilelang.jit(out_idx=[-2, -1], pass_configs=pass_configs)
def kda_recurrent_ker(B, SEQ, H, HV, K, V, scale, dtype="float16", accum_dtype="float"):
    VEC_NUM = 2
    BV = V // VEC_NUM
    GRP = HV // H  # every GRP value heads share one qk head

    @T.prim_func
    def main(
        Q: T.Tensor([B, SEQ, H, K], dtype),  # type: ignore
        Kt: T.Tensor([B, SEQ, H, K], dtype),  # type: ignore
        Vt: T.Tensor([B, SEQ, HV, V], dtype),  # type: ignore
        G: T.Tensor([B, SEQ, HV, K], accum_dtype),  # type: ignore
        # beta is padded to 8 fp32 slots on its last axis.  UB wants 32B
        # alignment, but beta only needs one fp32 per step; a [1] buffer is 4B
        # and skews the address of every allocation after it ("The UB address
        # accessed by the VEC instruction is not aligned" in practice).  So one
        # read pulls a full 8 fp32 = 32B and uses lane 0 only; the padding
        # zeros keep that read in bounds.  The same tail-padding trick is used
        # in fused_sigmoid_gating_delta_rule_varlen.py.
        Beta: T.Tensor([B, SEQ, HV, 8], accum_dtype),  # type: ignore
        S0: T.Tensor([B, HV, K, V], accum_dtype),  # type: ignore  initial state; the host guarantees it is present
        O: T.Tensor([B, SEQ, HV, V], dtype),  # type: ignore
        SF: T.Tensor([B, HV, K, V], accum_dtype),  # type: ignore  final state
    ):
        with T.Kernel(B * HV, is_npu=True) as (cid, vid):
            bz = cid // HV  # batch
            hv = cid % HV  # value head
            hq = hv // GRP  # the qk head it maps to (GVA)
            vo = vid * BV  # this vector core's V offset

            s_ub = T.alloc_ub([K, BV], accum_dtype)  # the state, resident in fp32
            prod_ub = T.alloc_ub([K, BV], accum_dtype)  # scratch tile for the reductions
            ku_ub = T.alloc_ub([K, BV], accum_dtype)  # outer product k (x) u

            g_ub = T.alloc_ub([K], accum_dtype)
            k_ub = T.alloc_ub([K], accum_dtype)
            q_ub = T.alloc_ub([K], accum_dtype)
            row_half = T.alloc_ub([K], dtype)

            ks_ub = T.alloc_ub([BV], accum_dtype)  # S^T k
            u_ub = T.alloc_ub([BV], accum_dtype)  # residual * beta
            o_ub = T.alloc_ub([BV], accum_dtype)
            col_half = T.alloc_ub([BV], dtype)

            beta_ub = T.alloc_ub([8], accum_dtype)  # 32B, see the note on Beta above

            with T.Scope("V"):
                T.copy(S0[bz, hv, 0, vo], s_ub)  # load the initial state

                for t in T.serial(SEQ):
                    # ---- 1) per-channel decay: row i of S is scaled by exp(g[i])
                    T.copy(G[bz, t, hv, 0], g_ub)
                    for i in T.Parallel(K):
                        g_ub[i] = T.exp(g_ub[i])
                    for i, j in T.Parallel(K, BV):
                        s_ub[i, j] = s_ub[i, j] * g_ub[i]

                    # ---- 2) delta rule
                    T.copy(Kt[bz, t, hq, 0], row_half)
                    T.copy(row_half, k_ub)

                    # S^T k reduces along K (dim=0 reduces the first axis, giving
                    # [BV]).  reduce_sum defaults to clear=True and initialises its
                    # output, so no manual zeroing is needed
                    # (tilelang/language/reduce.py:79; src/op/reduce.cc:143).
                    # Accumulate semantics would need an explicit clear=False.
                    #
                    # Note this is "copy, then multiply in place" rather than one
                    # statement: when a 1-D buffer is broadcast by an *outer* loop
                    # variable, the destination has to be the tile being scaled
                    # (in place).  Writing into a different buffer trips a UB
                    # alignment fault.
                    T.copy(s_ub, prod_ub)
                    for i, j in T.Parallel(K, BV):
                        prod_ub[i, j] = prod_ub[i, j] * k_ub[i]
                    T.reduce_sum(prod_ub, ks_ub, dim=0)

                    T.copy(Vt[bz, t, hv, vo], col_half)
                    T.copy(col_half, u_ub)
                    for j in T.Parallel(BV):
                        u_ub[j] = u_ub[j] - ks_ub[j]  # prediction residual

                    T.copy(Beta[bz, t, hv, 0], beta_ub)
                    for j in T.Parallel(BV):
                        u_ub[j] = u_ub[j] * beta_ub[0]  # write strength

                    # The outer product k (x) u is built in two passes: first tile
                    # u across the whole buffer using the *inner* variable (in
                    # place + inner variable, legal), then scale by k using the
                    # *outer* variable (in place + outer variable, legal).  The
                    # one-liner ku[i,j] = k[i]*u[j] is out-of-place broadcast by an
                    # outer variable, which does not lower.
                    T.tile.fill(ku_ub, 0.0)
                    for i, j in T.Parallel(K, BV):
                        ku_ub[i, j] = ku_ub[i, j] + u_ub[j]
                    for i, j in T.Parallel(K, BV):
                        ku_ub[i, j] = ku_ub[i, j] * k_ub[i]
                    for i, j in T.Parallel(K, BV):
                        s_ub[i, j] = s_ub[i, j] + ku_ub[i, j]

                    # ---- 3) read out o = S^T (scale * q)
                    T.copy(Q[bz, t, hq, 0], row_half)
                    T.copy(row_half, q_ub)
                    for i in T.Parallel(K):
                        q_ub[i] = q_ub[i] * scale
                    T.copy(s_ub, prod_ub)  # copy first, then scale in place -- as above
                    for i, j in T.Parallel(K, BV):
                        prod_ub[i, j] = prod_ub[i, j] * q_ub[i]
                    T.reduce_sum(prod_ub, o_ub, dim=0)

                    T.copy(o_ub, col_half)
                    T.copy(col_half, O[bz, t, hv, vo])

                T.copy(s_ub, SF[bz, hv, 0, vo])  # emit the final state

    return main


def kda_recurrent(q, k, v, g, beta, scale=None, initial_state=None, output_final_state=False):
    """Host wrapper.  Semantics match kda_ref.kda_ref."""
    B, SEQ, H, K = q.shape
    HV, V = v.shape[2], v.shape[-1]
    assert V % 32 == 0, "the V/2 slice each vector core gets must stay 32B aligned"
    if scale is None:
        scale = K**-0.5

    # An absent initial state is passed as zeros: the kernel reads it
    # unconditionally, which avoids compiling a variant for the optional input.
    s0 = initial_state.float() if initial_state is not None else torch.zeros((B, HV, K, V), device=q.device, dtype=torch.float)

    # Each beta element gets its own 8-fp32 (32B) slot; the reason is in the
    # kernel comment on Beta.  Padding to [B,SEQ,HV,8] and not [B,SEQ,HV+8]:
    # the latter starts head hv at byte offset 4*hv, which is not a multiple of
    # 32, so it would be misaligned all the same.
    beta_p = torch.zeros((B, SEQ, HV, 8), device=q.device, dtype=torch.float)
    beta_p[..., 0] = beta.float()

    dt = {torch.float16: "float16", torch.bfloat16: "bfloat16"}[q.dtype]
    o, sf = kda_recurrent_ker(B, SEQ, H, HV, K, V, float(scale), dtype=dt)(q, k, v, g, beta_p, s0)
    return o, (sf if output_final_state else None)


# ----------------------------------------------------------------------- test
def _relerr(x, r):
    r = r.float()
    return (x.float() - r).abs().max().item() / max(r.abs().max().item(), 1e-9)


def _case(B, SEQ, H, HV, K, V, gate, with_state, dtype):
    q, k, v, g, beta, s0 = kda_ref.make_inputs(B, SEQ, H, HV, K, V, dtype=dtype, gate=gate, with_state=with_state)

    o, sf = kda_recurrent(q, k, v, g, beta, initial_state=s0, output_final_state=True)
    ro, rs = kda_ref.kda_ref(q, k, v, g, beta, initial_state=s0, output_final_state=True)

    eo, es = _relerr(o, ro), _relerr(sf, rs)
    tol = 5e-3 if dtype == torch.float16 else 3e-2  # bf16 keeps only 8 mantissa bits
    ok = eo < tol and es < tol and torch.isfinite(o.float()).all()
    tag = "bf16" if dtype == torch.bfloat16 else "fp16"
    print(
        f"  B{B} T{SEQ:<4d} H{H} HV{HV} K{K:<3d} V{V:<3d} {tag} {gate:8s} "
        f"state={'Y' if with_state else 'N'}  o={eo:.2e} S={es:.2e}  "
        f"{'ok' if ok else 'FAIL'}"
    )
    return ok


def main():
    tilelang.disable_cache()
    torch.manual_seed(0)

    ok = True
    print("== shape coverage (fp16, normal gate, no initial state) ==")
    for B, SEQ, H, HV, K, V in [
        (1, 1, 1, 1, 64, 64),  # pure decode
        (1, 7, 2, 2, 64, 64),  # very short sequence
        (4, 64, 4, 4, 64, 64),  # B=4, multi-head
        (2, 70, 2, 4, 64, 64),  # non-divisible length + GVA (HV=2H)
        (1, 33, 1, 4, 128, 128),  # K3 head dim + GVA (HV=4H)
    ]:
        ok &= _case(B, SEQ, H, HV, K, V, "normal", False, torch.float16)

    print("== four gate regimes x with / without initial state ==")
    for gate in ("keep", "normal", "forget", "extreme"):
        for ws in (False, True):
            ok &= _case(2, 32, 2, 4, 64, 64, gate, ws, torch.float16)

    print("== bf16 ==")
    for gate in ("normal", "forget"):
        ok &= _case(2, 32, 2, 4, 64, 64, gate, True, torch.bfloat16)

    if ok:
        print("Kernel Output Match!")
    else:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
