"""KDA L1 stage 6: the output.

    O = (scale.Q . e^{G}) S  +  Aqk V'
    Aqk[i, j] = sum_d q[i, d] k[j, d] e^{G[i, d] - G[j, d]},   j <= i

Two decayed terms that need opposite treatment:

  * the inter-chunk term only needs Q scaled by e^{G}.  Inside a chunk G <= 0,
    so that factor is bounded by 1 -- a plain elementwise multiply followed by a
    cube matmul against the chunk's entry state.
  * the intra-chunk term is the same contraction as stage 2 (kkt), with q in
    place of the first k and an *inclusive* mask (i >= j, kkt uses i > j).  It
    inherits the same constraint: with a per-channel gate the decay sits inside
    the sum over d, so it cannot be applied after the matmul, and the mask has
    to go into the exponent before exp().

Anchored blocking (the reason this stage is not a one-liner)
-----------------------------------------------------------
Folding e^{+G_i} into q and e^{-G_j} into k would make Aqk a single matmul, but
e^{-G_j} grows without bound down the chunk.  So the chunk is cut into blocks of
BC rows, exactly like ``kda_l1_ref._decayed_dot``:

    row block a = [a*BC, (a+1)*BC), anchor row ar = a*BC

    columns j < ar   (off-diagonal strip):  both folded factors are bounded,
        qf[i] = (scale q[i]) e^{G_i - G_ar}   (i >= ar  =>  exponent <= 0)
        kf[j] = k[j]         e^{G_ar - G_j}   (j <  ar  =>  exponent <= 0)
        one cube matmul qf @ kf^T gives the whole strip.

    columns j in [ar, ar+BC)  (diagonal block):  no anchor bounds both sides, so
        it is evaluated row by row on the vector cores, directly as
        e^{G_i - G_j} restricted to j <= i (exponent <= 0 by construction).

    columns j >= ar+BC:  above the diagonal, must be zero.

The mask is folded into kf's exponent *and* multiplied in afterwards, so the
columns j >= ar of the strip matmul come out as exact zeros rather than as
garbage that a later multiply-by-mask would have to kill (0 * inf = NaN).  The
vector cores then overwrite only the BC diagonal columns of each row.

Pipeline (three flags, the same count as gdn_chunk_o, but starting on the vector
side because with a per-channel gate *both* matmul operands have to be built
there -- GDN can start on the cube because its decay is a scalar post-scale)
------------------------------------------------------------------------------
    V  build qg, qf, kf[a]                        -> ws_qg / ws_qf / ws_kf, flag 0
    C  NB strip matmuls qf[a] @ kf[a]^T           -> ws_ao,               flag 1
    V  patch the BC diagonal columns of each row  -> ws_aq,               flag 2
    C  qg @ S accumulated with Aqk @ V' in one L0C tile, fixpipe straight to O

Both matmuls accumulate into the same L0C accumulator, so the sum of the two
terms never touches the vector cores and O is written by the fixpipe directly.

Interface
---------
Frozen FLA contract, [B, SEQ, HV, *] layout.  Note the token axis is *not*
innermost: taking a [C, K] tile is a strided move -- K contiguous elements per
row, HV*K elements between rows -- expressed as ``X[b, t0:t0+C, h, 0:K]``.

    Q, Kt   [B, SEQ, H,  K]     dtype     GVA: read with hq = hv // GRP
    Vt      [B, SEQ, HV, V]     dtype     pseudo-values V' from stage 5
    S       [B, HV, N, K, V]    dtype     per-chunk entry states from stage 5
    G       [B, SEQ, HV, K]     fp32      chunk-local cumulative log gate, <= 0
    MskInc  [C, C]              fp32      i >= j
    MskStr  [C, C]              fp32      i >  j
    O       [B, SEQ, HV, V]     dtype
    scale   K ** -0.5, a compile-time constant, multiplied into q here

Beta is not in the list: the step size is already inside A, and through A inside
W / U / V', so stage 6 never sees it.  qg and Aqk are built here from Q, Kt and G
rather than taken as inputs, which is what makes this a single kernel launch.

Cross-core flags 0 (V->C), 1 (C->V), 2 (V->C).  The chain is strictly
alternating, so there is no cycle to deadlock on: V sets 0 then waits 1; C waits
0, sets 1, waits 2; V sets 2.  Both vector cores set every flag and the cube
waits once, exactly as in gdn_chunk_o where the cube also consumes a [C, C] tile
written half by each core.

Known limitations of this first pass
------------------------------------
  * ``SEQ % C == 0`` is required.  Tail blocks are handled by the reference but
    not here (task spec: fixed length first, varlen later).
  * ``C % (BC * 2) == 0`` is required, i.e. C >= 32 for BC = 16: the two vector
    cores split the anchor blocks, and each core's row half must be a whole
    number of anchor blocks.
  * S and V' are taken in the *input* dtype, not fp32.  They are cube operands,
    so they have to be fp16/bf16 before the matmul anyway; stage 5 writes them
    out of L0C / UB and can emit them already cast at no cost.  Only the states
    the frozen contract names (S0 / SF) stay fp32.
  * K and V must be multiples of 16 (fractal granularity) and HV % H == 0.

UB accounting, worst case in scope (C=64, K=V=128, VEC_NUM=2 -> CV=32, BC=16)
----------------------------------------------------------------------------
    g_ub     [C, K]  fp32   32768      chunk-resident G
    k_ub     [C, K]  fp32   32768      chunk-resident K
    kh_ub    [C, K]  fp16   16384      K load staging, reused for the kf store
    fold_ub  [C, K]  fp32   32768      kf[a] under construction
    qb_ub    [BC, K] fp32    8192      scale * q block
    qb_half  [BC, K] fp16    4096
    pb_ub    [BC, K] fp32    8192      exponent / product tile
    ob_ub    [BC, K] fp32    8192      folded q block
    ob_half  [BC, K] fp16    4096
    ah_ub    [CV, C] fp16    4096      this core's rows of Aqk
    grow/qrow/qrow_half/mcol/mrow/red  1664
                          -------
                          153216  of 196352 (no buffer named tmp_ub, and every
                                   T.Parallel is single-op so the backend never
                                   has to allocate a compound temporary)
"""

import os
import sys

import torch
import tilelang
from tilelang import language as T

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kda_l1_ref  # noqa: E402

# Only AUTO_SYNC, same as the six GDN kernels.  MEMORY_PLANNING is deliberately
# off: on the backward bwd_dot kernel it aliased a reduction target with a live
# temporary, which showed up as correct registers but an all-zero store.
pass_configs = {tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True}

VEC_NUM = 2
UB_LIMIT = 196352


def ub_bytes(C, K, V, BC, elem):
    """Per-core UB footprint, mirroring the alloc_ub list in the kernel.

    ``elem`` is the input itemsize (2 for fp16 / bf16).  Worth checking on the
    host: C=128 with K=128 blows the budget on the three [C, K] fp32 tiles
    alone, and the failure mode on device is an aicore exception, not a message.
    """
    CV = C // VEC_NUM
    f32 = 4
    return (
        3 * C * K * f32  # g_ub, k_ub, fold_ub
        + C * K * elem  # kh_ub
        + 3 * BC * K * f32  # qb_ub, pb_ub, ob_ub
        + 2 * BC * K * elem  # qb_half, ob_half
        + CV * C * elem  # ah_ub
        + 2 * K * f32
        + K * elem  # grow_ub, qrow_ub, qrow_half
        + C * f32
        + 2 * BC * f32
    )  # mcol_ub, mrow_ub, red_ub


@tilelang.jit(out_idx=[-1], workspace_idx=[-6, -5, -4, -3, -2], pass_configs=pass_configs)
def chunk_o_ker(B, SEQ, H, HV, K, V, C, scale, BC=16, dtype="float16", accum_dtype="float"):
    assert SEQ % C == 0, f"first pass needs SEQ % C == 0, got SEQ={SEQ} C={C}"
    assert C % (BC * VEC_NUM) == 0, f"need C % {BC * VEC_NUM} == 0, got C={C}"
    assert HV % H == 0, "HV must be divisible by H (GVA)"

    N = SEQ // C  # chunks per sequence
    CV = C // VEC_NUM  # rows per vector core
    NB = C // BC  # anchor blocks per chunk
    NBV = NB // VEC_NUM  # anchor blocks built by each vector core
    NAB = CV // BC  # anchor blocks inside one core's rows
    GRP = HV // H  # value heads sharing one qk head

    @T.prim_func
    def main(
        Q: T.Tensor([B, SEQ, H, K], dtype),  # type: ignore
        Kt: T.Tensor([B, SEQ, H, K], dtype),  # type: ignore
        Vt: T.Tensor([B, SEQ, HV, V], dtype),  # type: ignore  pseudo-values V'
        S: T.Tensor([B, HV, N, K, V], dtype),  # type: ignore  entry states
        G: T.Tensor([B, SEQ, HV, K], accum_dtype),  # type: ignore
        MskInc: T.Tensor([C, C], accum_dtype),  # type: ignore  i >= j
        MskStr: T.Tensor([C, C], accum_dtype),  # type: ignore  i >  j
        ws_qg: T.Tensor([B * HV * N, C, K], dtype),  # type: ignore
        ws_qf: T.Tensor([B * HV * N, C, K], dtype),  # type: ignore
        ws_kf: T.Tensor([B * HV * N, NB, C, K], dtype),  # type: ignore
        ws_ao: T.Tensor([B * HV * N, C, C], dtype),  # type: ignore  strip matmuls
        ws_aq: T.Tensor([B * HV * N, C, C], dtype),  # type: ignore  full Aqk
        O: T.Tensor([B, SEQ, HV, V], dtype),  # type: ignore
    ):
        with T.Kernel(B * HV * N, is_npu=True) as (cid, vid):
            n = cid % N  # chunk index
            hv = (cid // N) % HV  # value head
            bz = (cid // N) // HV  # batch
            hq = hv // GRP  # qk head (GVA)
            t0 = n * C  # first token of this chunk
            r0 = vid * CV  # first chunk-local row of this core

            # ---- L1 / L0C: cube operands and accumulators
            qf_l1 = T.alloc_L1([BC, K], dtype)
            kf_l1 = T.alloc_L1([C, K], dtype)
            qg_l1 = T.alloc_L1([C, K], dtype)
            s_l1 = T.alloc_L1([K, V], dtype)
            aq_l1 = T.alloc_L1([C, C], dtype)
            v_l1 = T.alloc_L1([C, V], dtype)
            strip_l0 = T.alloc_L0C([BC, C], accum_dtype)
            o_l0 = T.alloc_L0C([C, V], accum_dtype)

            # ---- UB: see the accounting in the module docstring
            g_ub = T.alloc_ub([C, K], accum_dtype)
            k_ub = T.alloc_ub([C, K], accum_dtype)
            kh_ub = T.alloc_ub([C, K], dtype)
            fold_ub = T.alloc_ub([C, K], accum_dtype)

            qb_ub = T.alloc_ub([BC, K], accum_dtype)
            qb_half = T.alloc_ub([BC, K], dtype)
            pb_ub = T.alloc_ub([BC, K], accum_dtype)
            ob_ub = T.alloc_ub([BC, K], accum_dtype)
            ob_half = T.alloc_ub([BC, K], dtype)

            ah_ub = T.alloc_ub([CV, C], dtype)

            grow_ub = T.alloc_ub([K], accum_dtype)
            qrow_ub = T.alloc_ub([K], accum_dtype)
            qrow_half = T.alloc_ub([K], dtype)
            mcol_ub = T.alloc_ub([C], accum_dtype)
            mrow_ub = T.alloc_ub([BC], accum_dtype)
            red_ub = T.alloc_ub([BC], accum_dtype)

            with T.Scope("V"):
                # ============ phase 1: build the three cube operands ==========
                # Chunk-resident G and K.  [C, *] over the token axis is a
                # strided move: the row length is K, the row pitch is HV*K
                # (H*K for the qk-head tensors).
                T.copy(G[bz, t0 : t0 + C, hv, 0:K], g_ub)
                T.copy(Kt[bz, t0 : t0 + C, hq, 0:K], kh_ub)
                T.copy(kh_ub, k_ub)

                for ab in range(NAB):
                    row = r0 + ab * BC  # chunk-local first row of the block

                    T.copy(Q[bz, t0 + row : t0 + row + BC, hq, 0:K], qb_half)
                    T.copy(qb_half, qb_ub)
                    for i, d in T.Parallel(BC, K):
                        qb_ub[i, d] = qb_ub[i, d] * scale

                    # inter-chunk operand qg = (scale q) . e^{G}, exponent <= 0.
                    # This block's rows of G are re-read from global memory
                    # rather than sliced out of the resident g_ub: a UB->UB copy
                    # at a *runtime* row offset (row depends on vid) is the one
                    # pattern kda_chunk_scaled_dot_kkt.py documents as not doing
                    # what it looks like, and no kernel in this repo or in the
                    # GDN reference uses it.  A strided GM read at a runtime
                    # offset is proven -- it is what the Q load above does.
                    T.copy(G[bz, t0 + row : t0 + row + BC, hv, 0:K], pb_ub)
                    for i, d in T.Parallel(BC, K):
                        pb_ub[i, d] = T.exp(pb_ub[i, d])
                    for i, d in T.Parallel(BC, K):
                        ob_ub[i, d] = qb_ub[i, d] * pb_ub[i, d]
                    T.copy(ob_ub, ob_half)
                    T.copy(ob_half, ws_qg[cid, row, 0])

                    # strip operand qf = (scale q) . e^{G - G_anchor}.  Rows of
                    # the block are at or below the anchor, so the exponent is
                    # <= 0; qg cannot be reused for this (dividing it by
                    # e^{G_anchor} is exactly the overflow we are avoiding).
                    T.copy(G[bz, t0 + row, hv, 0], grow_ub)
                    T.copy(G[bz, t0 + row : t0 + row + BC, hv, 0:K], pb_ub)
                    for i, d in T.Parallel(BC, K):
                        pb_ub[i, d] = pb_ub[i, d] - grow_ub[d]
                    for i, d in T.Parallel(BC, K):
                        pb_ub[i, d] = T.exp(pb_ub[i, d])
                    for i, d in T.Parallel(BC, K):
                        ob_ub[i, d] = qb_ub[i, d] * pb_ub[i, d]
                    T.copy(ob_ub, ob_half)
                    T.copy(ob_half, ws_qf[cid, row, 0])

                # strip operand kf[a] = K . e^{G_anchor - G} for j < anchor,
                # zero elsewhere.  The two cores split the anchor blocks.
                for ai in range(NBV):
                    a = vid + ai * VEC_NUM
                    ar = a * BC  # anchor row of this block

                    T.copy(G[bz, t0 + ar, hv, 0], grow_ub)
                    # strictly-lower row ar is the indicator of j < ar; for
                    # ar == 0 it is all zeros, so that block's matmul yields
                    # exact zeros instead of reading dirty workspace memory.
                    T.copy(MskStr[ar, 0], mcol_ub)
                    for j, d in T.Parallel(C, K):
                        fold_ub[j, d] = grow_ub[d] - g_ub[j, d]
                    for j, d in T.Parallel(C, K):
                        fold_ub[j, d] = fold_ub[j, d] * mcol_ub[j]
                    for j, d in T.Parallel(C, K):
                        fold_ub[j, d] = T.exp(fold_ub[j, d])
                    for j, d in T.Parallel(C, K):
                        fold_ub[j, d] = fold_ub[j, d] * k_ub[j, d]
                    for j, d in T.Parallel(C, K):
                        fold_ub[j, d] = fold_ub[j, d] * mcol_ub[j]
                    T.copy(fold_ub, kh_ub)
                    T.copy(kh_ub, ws_kf[cid, a, 0, 0])

                T.set_cross_flag("MTE3", 0)

                # ============ phase 3: patch the diagonal blocks =============
                T.wait_cross_flag(1)
                T.copy(ws_ao[cid, r0, 0], ah_ub)

                for ab in range(NAB):
                    a0 = r0 + ab * BC  # anchor row of this block
                    for rr in range(BC):
                        i_loc = a0 + rr  # chunk-local row being fixed

                        T.copy(Q[bz, t0 + i_loc, hq, 0], qrow_half)
                        T.copy(qrow_half, qrow_ub)
                        for d in T.Parallel(K):
                            qrow_ub[d] = qrow_ub[d] * scale
                        T.copy(G[bz, t0 + i_loc, hv, 0], grow_ub)
                        # inclusive row i_loc restricted to this block's columns
                        T.copy(MskInc[i_loc, a0], mrow_ub)

                        for jj, d in T.Parallel(BC, K):
                            pb_ub[jj, d] = grow_ub[d] - g_ub[a0 + jj, d]
                        # mask folded in before exp(): the j > i half of the
                        # exponent is positive and would overflow first
                        for jj, d in T.Parallel(BC, K):
                            pb_ub[jj, d] = pb_ub[jj, d] * mrow_ub[jj]
                        for jj, d in T.Parallel(BC, K):
                            pb_ub[jj, d] = T.exp(pb_ub[jj, d])
                        for jj, d in T.Parallel(BC, K):
                            pb_ub[jj, d] = pb_ub[jj, d] * qrow_ub[d]
                        for jj, d in T.Parallel(BC, K):
                            pb_ub[jj, d] = pb_ub[jj, d] * k_ub[a0 + jj, d]
                        # and again after exp(), to zero the masked columns --
                        # every factor is finite here, so this multiply is safe
                        for jj, d in T.Parallel(BC, K):
                            pb_ub[jj, d] = pb_ub[jj, d] * mrow_ub[jj]

                        # reduce_sum clears its output itself (clear=True by
                        # default), so no manual fill
                        T.reduce_sum(pb_ub, red_ub, dim=-1)
                        # BC fp16 = 32 B exactly, and a0 is a multiple of BC
                        T.copy(red_ub, ah_ub[ab * BC + rr, a0])

                T.copy(ah_ub, ws_aq[cid, r0, 0])
                T.set_cross_flag("MTE3", 2)

            with T.Scope("C"):
                # ============ phase 2: the off-diagonal strips ================
                T.wait_cross_flag(0)
                for a in range(NB):
                    T.copy(ws_qf[cid, a * BC, 0], qf_l1)
                    T.copy(ws_kf[cid, a, 0, 0], kf_l1)
                    T.gemm_v0(qf_l1, kf_l1, strip_l0, transpose_B=True, init=True)
                    T.copy(strip_l0, ws_ao[cid, a * BC, 0])
                T.set_cross_flag("FIX", 1)

                # ============ phase 4: both output terms in one accumulator ===
                T.wait_cross_flag(2)
                T.copy(ws_qg[cid, 0, 0], qg_l1)
                T.copy(S[bz, hv, n, 0, 0], s_l1)
                T.gemm_v0(qg_l1, s_l1, o_l0, init=True)

                T.copy(ws_aq[cid, 0, 0], aq_l1)
                T.copy(Vt[bz, t0 : t0 + C, hv, 0:V], v_l1)
                T.gemm_v0(aq_l1, v_l1, o_l0, init=False)

                # fixpipe straight into O: a [C, V] tile with row pitch HV*V
                T.copy(o_l0, O[bz, t0 : t0 + C, hv, 0:V])

    return main


def chunk_o(q, k, vnew, states, G, C=64, BC=16, scale=None):
    """Host wrapper.  O = (scale.Q . e^G) S + Aqk V', all external layout.

    q, k    [B, SEQ, H,  K]   dtype
    vnew    [B, SEQ, HV, V]   dtype   V' from stage 5
    states  [B, HV, N, K, V]  dtype   per-chunk entry states from stage 5
    G       [B, SEQ, HV, K]   fp32    stage 1 output

    The host only builds the two constant masks and looks the dtype up; no
    transposes, no reshapes, no state shuffling.
    """
    B, SEQ, H, K = q.shape
    HV, V = vnew.shape[2], vnew.shape[-1]
    assert SEQ % C == 0, f"first pass needs SEQ % C == 0, got SEQ={SEQ} C={C}"
    assert C % (BC * VEC_NUM) == 0, f"need C % {BC * VEC_NUM} == 0, got C={C}"
    assert K % 16 == 0 and V % 16 == 0, "K and V must be multiples of 16"
    # The dtype is threaded into the kernel as a compile-time constant, so all
    # four Cube operands have to agree on it.  Kept on one line deliberately:
    # ruff 0.6.5 (requirements-lint.txt) and current ruff (what the CI action
    # installs) wrap a split assert in opposite directions, so a single line
    # under the 140-column limit is the only form both leave alone.
    assert k.dtype == q.dtype and vnew.dtype == q.dtype and states.dtype == q.dtype, "Q / Kt / V' / S must share one dtype"
    assert G.shape == (B, SEQ, HV, K), f"G must be [B, SEQ, HV, K], got {tuple(G.shape)}"
    assert states.shape == (B, HV, SEQ // C, K, V), f"states must be [B, HV, SEQ//C, K, V], got {tuple(states.shape)}"

    elem = torch.finfo(q.dtype).bits // 8
    need = ub_bytes(C, K, V, BC, elem)
    assert need <= UB_LIMIT, f"UB budget {need} > {UB_LIMIT} for C={C} K={K}; lower C"

    # SEQ == 0 slips past the assert above (0 % C == 0) and would launch a
    # zero-block grid over an unwritten output.  A zero-length sequence is legal
    # input; there is no query to read out, so O is empty along the token axis.
    if SEQ == 0:
        return torch.empty((B, 0, HV, V), device=q.device, dtype=q.dtype)

    if scale is None:
        scale = K**-0.5

    idx = torch.arange(C, device=q.device)
    msk_inc = (idx[:, None] >= idx[None, :]).float()  # i >= j
    msk_str = (idx[:, None] > idx[None, :]).float()  # i >  j

    dt = {torch.float16: "float16", torch.bfloat16: "bfloat16"}[q.dtype]
    ker = chunk_o_ker(B, SEQ, H, HV, K, V, C, float(scale), BC=BC, dtype=dt)
    return ker(q, k, vnew, states, G, msk_inc, msk_str)


# ----------------------------------------------------------------- test
def _relerr(x, r):
    r = r.float()
    return (x.float() - r).abs().max().item() / max(r.abs().max().item(), 1e-9)


def _case(B, SEQ, H, HV, K, V, C, gate, dtype=torch.float16, BC=16):
    q, k, v, g, beta, _ = kda_l1_ref.make_inputs(B, SEQ, H, HV, K, V, dtype=dtype, gate=gate)
    st = kda_l1_ref.stage_tensors(q, k, v, g, beta, C=C, BC=BC)

    # stage_tensors works in the internal [B, HV, T, *] layout and transposes on
    # the way out, so its tensors are views; the kernel adapter needs contiguous
    # ones.  That materialisation belongs to the reference, not to this kernel's
    # host path -- in the fused pipeline stage 5 writes V' / states straight out
    # in external layout and in the input dtype, and chunk_o() below does no
    # tensor work at all beyond the two constant masks.
    got = chunk_o(q, k, st["Vt"].to(dtype).contiguous(), st["states"].to(dtype).contiguous(), st["G"].contiguous(), C=C, BC=BC)

    err = _relerr(got, st["o"])
    finite = torch.isfinite(got.float()).all().item()
    tol = 3e-2 if dtype == torch.float16 else 6e-2  # bf16 keeps 8 mantissa bits
    ok = finite and err < tol
    tag = "bf16" if dtype == torch.bfloat16 else "fp16"
    print(
        f"  B{B} T{SEQ:<4d} H{H} HV{HV} K{K:<3d} V{V:<3d} C{C:<2d} {tag} "
        f"{gate:8s} relerr={err:.2e} finite={finite}  {'ok' if ok else 'FAIL'}"
    )
    return ok


def main():
    tilelang.disable_cache()
    torch.manual_seed(0)

    ok = True
    print("== HV == H and HV == 2H, C = 32 and C = 64, two gate levels ==")
    for gate in ("normal", "forget"):
        ok &= _case(1, 128, 2, 2, 64, 64, 64, gate)  # HV == H,  C = 64
        ok &= _case(2, 128, 2, 4, 64, 64, 64, gate)  # HV == 2H, C = 64
        ok &= _case(1, 128, 2, 2, 64, 64, 32, gate)  # HV == H,  C = 32
        ok &= _case(2, 256, 2, 4, 64, 64, 32, gate)  # HV == 2H, C = 32

    print("== K3 head dim K = V = 128 ==")
    ok &= _case(1, 256, 1, 1, 128, 128, 64, "forget")
    ok &= _case(1, 128, 1, 2, 128, 128, 64, "normal")  # K3 + GVA
    ok &= _case(1, 128, 1, 1, 128, 128, 32, "forget")

    print("== gates that underflow inside a chunk (the NaN trap) ==")
    ok &= _case(1, 128, 1, 2, 64, 64, 64, "extreme")
    ok &= _case(1, 128, 1, 1, 64, 64, 32, "keep")

    print("== bf16 (dtype is threaded through from the inputs) ==")
    ok &= _case(2, 128, 2, 4, 64, 64, 64, "normal", dtype=torch.bfloat16)
    ok &= _case(1, 128, 1, 1, 128, 128, 64, "forget", dtype=torch.bfloat16)

    if ok:
        print("Kernel Output Match!")
    else:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
