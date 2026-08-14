"""KDA L1 stage 5 (chunk_h) on the frozen L0 interface.

What this stage computes, per (batch, value head, chunk n):

    states[n] = S                               entry state of chunk n
    V'[n]     = U[n] - W[n] S                   pseudo-values
    S         = Diag(e^{G_C[n]}) S + kg[n]^T V'[n]

with the decayed key folded in this kernel, from K and G directly:

    kg = K . e^{G_C - G}                        G_C = G of the chunk's last row

This is the only chunk-serial stage of the pipeline: chunk n+1 needs the state
produced by chunk n.  The serial chain is not broken up; instead the grid runs
B * HV * bv_num independent streams so batches, value heads and value blocks
keep the cores busy.

Two differences from GDN, both of them a scalar widened into a K-vector:
  * the state decay was `S *= e^{g_last}` (one scalar for the whole matrix); it
    is now a per-row scale, row d of S carries e^{G_C[d]}.
  * kg was a row broadcast (`k[i, :] *= c_i`); it is now a full elementwise
    product over [C, K].

Where the decay is folded matters.  This kernel puts it on the *key* side, as
gdn_chunk_h.py does.  Some delta-rule kernels fold the equivalent scalar onto
v_new instead; the two agree only for a scalar gate.  With a per-channel gate
the decay lives on the K axis, which V' does not have, so folding it onto V'
would be wrong.

Numerics
--------
Both exponents are differences taken in the non-increasing direction
(G is a chunk-local cumsum of g <= 0, so G_C <= G_i for every i):

    G_C - G_i <= 0      and      G_C <= 0

so both e^{...} land in (0, 1] and neither can overflow.  Nothing here is ever
a ratio of two exponentials, and no mask is applied after an exp -- the two
ways this pipeline produces NaN.  The cumsum restarting at every chunk boundary
is what bounds how far the exponents can travel; that is the first line of
defence, not an optimisation.

Interface (FLA contract; frozen on day 1 of the port)
---------------------------------------------
    Kt      [B, SEQ, H,  K]     dtype   qk head axis, read through hq = hv // GRP
    W       [B, SEQ, HV, K]     dtype   from wy_fast
    U       [B, SEQ, HV, V]     dtype   from wy_fast
    G       [B, SEQ, HV, K]     fp32    log-domain chunk-local cumsum, <= 0
    S0      [B, HV, K, V]       fp32    initial state, host guarantees non-null
    ->
    states  [B, HV, N, K, V]    dtype   entry state per chunk, N = SEQ // C
    Vt      [B, SEQ, HV, V]     dtype   pseudo-values (a gemm operand downstream)
    SF      [B, HV, K, V]       fp32    final state

Why states is dtype and SF is fp32, even though both hold a state: SF is
user-facing, it is the relay between two calls, and rounding it would make a
two-segment run disagree with a one-shot run -- so it comes straight out of the
fp32 accumulator.  states is an internal handoff to chunk_o, which feeds it to
the Cube as `(Q . e^G) S` and therefore needs dtype anyway (see the S operand
of gdn_chunk_o.py, and of kda_chunk_o.py here).  Emitting fp32 there would buy
chunk_o an extra cast plus a GM round trip, and would not change O by one bit.
It is also written from the *same* rounded tile this kernel hands the Cube for
W S, so the two stages contract against a bit-identical S.

The token axis is dim 1 and HV sits *between* T and K, so every [C, K] tile is
a strided DMA: C rows of K contiguous elements, HV*K elements apart.  That is
expressed by slicing the token axis explicitly (`X[bz, t0 : t0 + C, hv, :]`),
which makes the copy's extents [1, C, 1, K]; the row pitch HV*K is then derived
by the backend (src/op/ascend.cc: compute_strideN collapses the unit-extent
axes after the sliced one into the stride).  Indexing without the slice
(`X[bz, t0, hv, 0]`) maps the destination onto the *trailing* two axes instead
and would silently read C consecutive heads.  1-D reads of a single row need no
slice, which is why glast/gexp use the plain form.

beta and scale do not appear in this stage: kg carries no beta (see
kda_l1_ref.ref_wy_fast) and scale rides on q, which chunk_o consumes.

Known limitations (first pass)
------------------------------
  * requires SEQ % C == 0.  Tail blocks are handled by the reference
    (kda_l1_ref.kda_chunk_ref pads the token axis) but not yet by this kernel;
    fixed length first, varlen later, per the task spec.
  * the two gemm operands that carry the state (S into `W S`, and V' into
    `kg^T V'`) are `dtype`, because that is what the Cube takes.  The
    recurrence itself is fp32 in UB, and the two L0C->GM workspaces are fp32,
    so the only rounding per chunk is on the gemm inputs.  A high/low split of
    S into two dtype halves would remove even that, at the cost of doubling the
    gemm count; not worth it for a first pass.
  * V must be a multiple of BV, and C and K must be even (the two vector cores
    split C for the token tiles and K for the state tiles).
  * BV defaults to min(V, 64), i.e. K = V = 128 runs with the state sharded in
    two along V.  BV = V would fit UB at C = 32 but not at C = 64 (180992 B
    against a 196352 B ceiling, too little headroom), and the wrapper asserts
    rather than letting it become an aicore exception.
"""

import os
import sys

import torch
import tilelang
from tilelang import language as T

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kda_l1_ref  # noqa: E402

# Only AUTO_SYNC, matching the six GDN kernels in the repo.  MEMORY_PLANNING is
# deliberately off: on the backward bwd_dot kernel it aliased a reduction target
# with a temporary tile, and the store wrote zeroes while the registers were
# right -- a failure mode that looks like a math bug.
pass_configs = {tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True}

UB_LIMIT = 196352  # bytes per AIV
UB_MARGIN = 16384  # headroom for compiler-allocated temporaries


def ub_bytes(C, K, BV, itemsize):
    """UB footprint of one AIV, in bytes.  Mirrors the allocations below."""
    CV, KV = C // 2, K // 2
    f4 = 4
    return (
        3 * CV * K * f4  # g_ub, coeff_ub, k_ub
        + K * f4  # glast_ub
        + KV * f4  # gexp_ub
        + 2 * KV * BV * f4  # s_ub, kv_ub
        + 2 * CV * BV * f4  # u_ub, ws_ub
        + CV * K * itemsize  # k_half
        + KV * BV * itemsize  # s_half
        + CV * BV * itemsize  # u_half
    )


@tilelang.jit(out_idx=[-3, -2, -1], workspace_idx=[-7, -6, -5, -4], pass_configs=pass_configs)
def chunk_h_ker(B, SEQ, H, HV, K, V, C, BV, dtype="float16", accum_dtype="float"):
    N_CHUNK = SEQ // C  # this pass requires SEQ % C == 0
    BV_NUM = V // BV
    VEC_NUM = 2  # 910B: one Cube core, two Vector cores
    CV = C // VEC_NUM  # token rows per vector core
    KV = K // VEC_NUM  # state rows per vector core
    GRP = HV // H  # GVA: GRP value heads per qk head
    NBLK = B * HV * BV_NUM

    @T.prim_func
    def main(
        Kt: T.Tensor([B, SEQ, H, K], dtype),  # type: ignore
        W: T.Tensor([B, SEQ, HV, K], dtype),  # type: ignore
        U: T.Tensor([B, SEQ, HV, V], dtype),  # type: ignore
        G: T.Tensor([B, SEQ, HV, K], accum_dtype),  # type: ignore
        S0: T.Tensor([B, HV, K, V], accum_dtype),  # type: ignore
        # Four scratch workspaces.  Every one of them is written before it is
        # read inside the same chunk iteration, so all four may be framework
        # allocated (torch.empty).  A workspace that needed zeroing could NOT
        # be listed in workspace_idx -- it would come back as dirty memory.
        # The old [B,H,L,DK]-layout version had to pass the state workspace in
        # as a host-zeroed input for exactly that reason; supporting S0 removed
        # the need, because the vector core now writes the state workspace at
        # the top of every iteration, including the first.
        ws_wS: T.Tensor([NBLK, C, BV], accum_dtype),  # type: ignore  W S
        ws_kg: T.Tensor([NBLK, C, K], dtype),  # type: ignore         kg
        ws_st: T.Tensor([NBLK, K, BV], dtype),  # type: ignore        S as a gemm operand
        ws_kv: T.Tensor([NBLK, K, BV], accum_dtype),  # type: ignore  kg^T V'
        states: T.Tensor([B, HV, N_CHUNK, K, V], dtype),  # type: ignore
        Vt: T.Tensor([B, SEQ, HV, V], dtype),  # type: ignore
        SF: T.Tensor([B, HV, K, V], accum_dtype),  # type: ignore
    ):
        with T.Kernel(NBLK, is_npu=True) as (cid, vid):
            bx = cid % BV_NUM  # value block
            hv = (cid // BV_NUM) % HV  # value head
            bz = cid // (BV_NUM * HV)  # batch
            hq = hv // GRP  # qk head feeding this value head
            vo = bx * BV  # value offset of this block
            ko = vid * KV  # state rows of this vector core
            co = vid * CV  # token rows of this vector core

            s_l1 = T.alloc_L1([K, BV], dtype)
            w_l1 = T.alloc_L1([C, K], dtype)
            k_l1 = T.alloc_L1([C, K], dtype)
            v_l1 = T.alloc_L1([C, BV], dtype)
            wS_l0 = T.alloc_L0C([C, BV], accum_dtype)
            kv_l0 = T.alloc_L0C([K, BV], accum_dtype)

            g_ub = T.alloc_ub([CV, K], accum_dtype)
            coeff_ub = T.alloc_ub([CV, K], accum_dtype)
            k_ub = T.alloc_ub([CV, K], accum_dtype)
            glast_ub = T.alloc_ub([K], accum_dtype)
            gexp_ub = T.alloc_ub([KV], accum_dtype)
            s_ub = T.alloc_ub([KV, BV], accum_dtype)
            kv_ub = T.alloc_ub([KV, BV], accum_dtype)
            u_ub = T.alloc_ub([CV, BV], accum_dtype)
            ws_ub = T.alloc_ub([CV, BV], accum_dtype)
            k_half = T.alloc_ub([CV, K], dtype)
            s_half = T.alloc_ub([KV, BV], dtype)
            u_half = T.alloc_ub([CV, BV], dtype)

            # ---------------------------------------------------------- Cube
            # Flag protocol, same four ids and pipes as gdn_chunk_h.py:
            #   3 (V->C) ws_st holds S_n        1 (V->C) ws_kg / Vt hold kg, V'
            #   0 (C->V) ws_wS holds W S        2 (C->V) ws_kv holds kg^T V'
            # Each id is set by both AIVs (or by the AIC) and waited on by the
            # other side, which is what serialises the two concurrent streams.
            with T.Scope("C"):
                for ci in T.serial(N_CHUNK):
                    t0 = ci * C

                    T.wait_cross_flag(3)
                    T.copy(ws_st[cid, 0, 0], s_l1)
                    T.copy(W[bz, t0 : t0 + C, hv, :], w_l1)
                    T.gemm_v0(w_l1, s_l1, wS_l0, init=True)  # W S
                    T.copy(wS_l0, ws_wS[cid, 0, 0])
                    T.set_cross_flag("FIX", 0)

                    T.wait_cross_flag(1)
                    T.copy(ws_kg[cid, 0, 0], k_l1)
                    T.copy(Vt[bz, t0 : t0 + C, hv, vo : vo + BV], v_l1)
                    T.gemm_v0(k_l1, v_l1, kv_l0, transpose_A=True, init=True)
                    T.copy(kv_l0, ws_kv[cid, 0, 0])  # kg^T V'
                    T.set_cross_flag("FIX", 2)

            # -------------------------------------------------------- Vector
            with T.Scope("V"):
                T.copy(S0[bz, hv, ko, vo], s_ub)  # fp32 state, stays resident

                for ci in T.serial(N_CHUNK):
                    t0 = ci * C
                    tv = t0 + co  # this core's first token

                    # entry state of this chunk: one rounded tile, published
                    # twice -- to chunk_o as `states`, and to the Cube as the
                    # right operand of W S.
                    T.copy(s_ub, s_half)
                    T.copy(s_half, states[bz, hv, ci, ko, vo])
                    T.copy(s_half, ws_st[cid, ko, 0])
                    T.set_cross_flag("MTE3", 3)

                    # kg = K . e^{G_C - G}.  GDN broadcasts one scalar per row
                    # here; KDA needs the whole [CV, K] tile.
                    T.copy(Kt[bz, tv : tv + CV, hq, :], k_half)
                    T.copy(k_half, k_ub)
                    T.copy(G[bz, tv : tv + CV, hv, :], g_ub)
                    T.copy(G[bz, t0 + C - 1, hv, 0], glast_ub)  # G_C, all K
                    T.copy(G[bz, t0 + C - 1, hv, ko], gexp_ub)  # G_C, my rows
                    for a, b in T.Parallel(CV, K):
                        coeff_ub[a, b] = glast_ub[b] - g_ub[a, b]  # <= 0
                    for a, b in T.Parallel(CV, K):
                        coeff_ub[a, b] = T.exp(coeff_ub[a, b])
                    for a, b in T.Parallel(CV, K):
                        k_ub[a, b] = k_ub[a, b] * coeff_ub[a, b]
                    T.copy(k_ub, k_half)

                    # V' = U - W S
                    T.copy(U[bz, tv : tv + CV, hv, vo : vo + BV], u_half)
                    T.copy(u_half, u_ub)
                    T.wait_cross_flag(0)
                    T.copy(ws_wS[cid, co, 0], ws_ub)
                    for a, b in T.Parallel(CV, BV):
                        u_ub[a, b] = u_ub[a, b] - ws_ub[a, b]
                    T.copy(u_ub, u_half)
                    T.copy(u_half, Vt[bz, tv : tv + CV, hv, vo : vo + BV])
                    T.copy(k_half, ws_kg[cid, co, 0])
                    T.set_cross_flag("MTE3", 1)

                    # per-channel state decay: row d of S carries e^{G_C[d]}.
                    # Broadcasting a 1-D buffer along the *outer* loop variable
                    # has to be in place -- writing to another buffer does not
                    # pass the UB alignment check.
                    for i in T.Parallel(KV):
                        gexp_ub[i] = T.exp(gexp_ub[i])
                    for i, j in T.Parallel(KV, BV):
                        s_ub[i, j] = s_ub[i, j] * gexp_ub[i]

                    T.wait_cross_flag(2)
                    T.copy(ws_kv[cid, ko, 0], kv_ub)
                    for i, j in T.Parallel(KV, BV):
                        s_ub[i, j] = s_ub[i, j] + kv_ub[i, j]

                T.copy(s_ub, SF[bz, hv, ko, vo])

    return main


# ----------------------------------------------------------------- host side
_DTYPE_STR = {torch.float16: "float16", torch.bfloat16: "bfloat16"}


def chunk_h(kt, w, u, g_cumsum, C=64, BV=None, initial_state=None):
    """Host wrapper.  Semantics match kda_l1_ref.ref_chunk_h.

    Only zero-filling the absent initial state, a dtype table lookup and the
    block-size choice happen here.  No transpose, no reshape, no staging copy:
    the kernel indexes the external [B, SEQ, HV, *] layout directly, which is
    the task's gate against hiding kernel cost on the host.
    """
    B, SEQ, H, K = kt.shape
    HV, V = u.shape[2], u.shape[-1]

    assert SEQ % C == 0, f"first pass needs SEQ % C == 0, got SEQ={SEQ} C={C}"
    assert HV % H == 0, "HV must be divisible by H (GVA)"
    assert C % 2 == 0 and K % 2 == 0, "the two vector cores split C and K"
    assert w.dtype == u.dtype == kt.dtype, "Kt / W / U must share one dtype"
    assert kt.dtype in _DTYPE_STR, f"unsupported dtype {kt.dtype}"
    assert g_cumsum.dtype == torch.float32, "G is fp32 (log domain)"
    for name, x in (("Kt", kt), ("W", w), ("U", u), ("G", g_cumsum)):
        assert x.is_contiguous(), f"{name} must be contiguous"

    if BV is None:
        # Sharding S along V is lossless (the recurrence is independent per
        # value column); it only repeats the per-token kg/G preparation.  64
        # keeps K=V=128 comfortably inside UB for both C=32 and C=64.
        BV = min(V, 64)
    assert V % BV == 0 and BV % 16 == 0, f"illegal BV={BV} for V={V}"

    need = ub_bytes(C, K, BV, kt.element_size())
    assert need <= UB_LIMIT - UB_MARGIN, f"UB overflow: {need} B needed, limit {UB_LIMIT} B (margin {UB_MARGIN} B) for C={C} K={K} BV={BV}"

    # Absent initial state is passed as zeros so the kernel can read it
    # unconditionally; the same choice as the L0 kernel.
    s0 = initial_state.float() if initial_state is not None else torch.zeros((B, HV, K, V), device=kt.device, dtype=torch.float32)
    assert s0.shape == (B, HV, K, V) and s0.is_contiguous(), "S0 layout"

    # SEQ == 0 slips past the assert above (0 % C == 0) and would launch a
    # zero-block grid over unwritten outputs.  A zero-length sequence is legal
    # input.  This is the one stage where the answer is not simply "empty": no
    # token is consumed, so the state must pass through untouched and the final
    # state IS the initial state.  Cloned rather than aliased so a caller that
    # relays it into the next segment cannot mutate its own input.
    if SEQ == 0:
        return (
            torch.empty((B, HV, 0, K, V), device=kt.device, dtype=kt.dtype),
            torch.empty((B, 0, HV, V), device=kt.device, dtype=kt.dtype),
            s0.clone(),
        )

    ker = chunk_h_ker(B, SEQ, H, HV, K, V, C, BV, dtype=_DTYPE_STR[kt.dtype])
    states, vt, sf = ker(kt, w, u, g_cumsum, s0)
    return states, vt, sf


# ---------------------------------------------------------------------- test
def _relerr(x, r):
    r = r.float()
    return (x.float() - r).abs().max().item() / max(r.abs().max().item(), 1e-9)


def _case(B, SEQ, H, HV, K, V, C, gate, with_state, dtype):
    q, k, v, g, beta, s0 = kda_l1_ref.make_inputs(B, SEQ, H, HV, K, V, dtype=dtype, gate=gate, with_state=with_state)

    gold = kda_l1_ref.stage_tensors(q, k, v, g, beta, C=C, initial_state=s0)
    # stage_tensors returns external [B, SEQ, HV, *] views produced by a
    # transpose, so they are not contiguous; the kernels that will feed this
    # stage in the real pipeline emit contiguous tensors, hence .contiguous()
    # here is test scaffolding, not a host-side layout change.
    W = gold["W"].contiguous().to(dtype)
    U = gold["U"].contiguous().to(dtype)
    G = gold["G"].contiguous()

    states, vt, sf = chunk_h(k, W, U, G, C=C, initial_state=s0)

    eS = _relerr(states, gold["states"])
    eV = _relerr(vt, gold["Vt"])
    eF = _relerr(sf, gold["SF"])
    # bf16 has 8 mantissa bits against fp16's 11, so rounding the two dtype
    # gemm operands costs about 8x more on the same shapes.
    tol = 2e-2 if dtype == torch.float16 else 6e-2
    finite = torch.isfinite(states.float()).all() and torch.isfinite(vt.float()).all() and torch.isfinite(sf.float()).all()
    ok = bool(finite) and max(eS, eV, eF) < tol
    tag = "bf16" if dtype == torch.bfloat16 else "fp16"
    print(
        f"  B{B} T{SEQ:<4d} H{H} HV{HV} K{K:<3d} V{V:<3d} C{C:<2d} {tag} "
        f"{gate:8s} state={'Y' if with_state else 'N'}  "
        f"S={eS:.2e} V'={eV:.2e} SF={eF:.2e}  {'ok' if ok else 'FAIL'}"
    )
    return ok


def main():
    tilelang.disable_cache()
    torch.manual_seed(0)

    ok = True
    print("== HV == H and HV == 2H, C = 32 / 64, two gate levels (fp16) ==")
    for B, SEQ, H, HV, K, V, C, gate, ws in [
        (1, 128, 1, 1, 64, 64, 64, "normal", False),  # HV == H
        (1, 128, 1, 1, 64, 64, 64, "normal", True),  # HV == H  + initial_state
        (2, 128, 2, 4, 64, 64, 64, "normal", True),  # HV == 2H
        (2, 128, 2, 4, 64, 64, 32, "forget", True),  # HV == 2H + C=32 + forget
        (1, 256, 1, 1, 64, 64, 32, "forget", False),  # 8 chunks of serial chain
    ]:
        ok &= _case(B, SEQ, H, HV, K, V, C, gate, ws, torch.float16)

    print("== K3 spec K = V = 128 (fp16) ==")
    for B, SEQ, H, HV, K, V, C, gate, ws in [
        (1, 128, 1, 1, 128, 128, 64, "normal", True),
        (1, 128, 2, 4, 128, 128, 64, "forget", True),  # + GVA
        (1, 128, 1, 1, 128, 128, 32, "forget", False),
    ]:
        ok &= _case(B, SEQ, H, HV, K, V, C, gate, ws, torch.float16)

    print("== dtype passthrough (bf16) ==")
    for B, SEQ, H, HV, K, V, C, gate, ws in [
        (2, 128, 2, 4, 64, 64, 64, "normal", True),
        (1, 128, 1, 1, 128, 128, 64, "forget", True),
    ]:
        ok &= _case(B, SEQ, H, HV, K, V, C, gate, ws, torch.bfloat16)

    if ok:
        print("Kernel Output Match!")
    else:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
