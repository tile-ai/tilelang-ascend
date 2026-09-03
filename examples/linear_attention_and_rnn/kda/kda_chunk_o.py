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
BC rows, exactly like ``kda_chunk_ref._decayed_dot``:

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

Ragged tail
-----------
``SEQ % C != 0`` is supported.  This stage has the most index sites of the six,
but no new maths -- everything is either clamped for free or clamped by hand.

  * The two chunk-resident [C, K] loads and the final ``O`` store are multi-row
    on the token axis and are clamped by ``compute_valid_extent``.  Because the
    ``O`` store is a clamped l0c2gm, ``O`` stays [B, SEQ, HV, V] and needs no
    padded allocation.
  * Four single-row reads (the two anchor gates, and the Q / G rows of the
    diagonal patch) are NOT bounds-checked -- their region has the token axis on
    a unit-extent dim -- so they are index-clamped to the last valid row by
    ``tok()``.  Clamping is the right fix *here* and would be wrong in stage 2:
    everything these rows produce lands in UB or in ``ws_*`` rows >= R and is
    discarded by the clamped store, so nothing is ever written to GM at a
    clamped index.
  * Three BC-row block loads can start past ``SEQ`` entirely, which would clamp
    ``validRow`` to 0.  They are zero-filled and then skipped rather than issued
    with ``blockCount = 0``.

The cross-core flags are never skipped -- only DMAs are.

Known limitations of this first pass
------------------------------------
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

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import kda_chunk_ref  # noqa: E402
import kda_varlen as _VL  # noqa: E402

# Only AUTO_SYNC, same as the six GDN kernels.  MEMORY_PLANNING is deliberately
# off: on the backward bwd_dot kernel it aliased a reduction target with a live
# temporary, which showed up as correct registers but an all-zero store.
pass_configs = {tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True}

# MEMORY_PLANNING (live-range based UB reuse) has always been off in this repo:
# VERSION.md records it aliasing a reduction target with a scratch tile on the
# backward dot, so the registers were right and the store wrote zeros.
#
# It is exposed as a switch here because probe_stacked_escalate.py measured what
# the wide [2*BC, K] tile in phase 3 would need: 14.4-19.5 KB of *implicit*
# compiler scratch on top of the declared buffers, against the 9.2 KB this kernel
# has spare at K=128 (the static check passes and the device raises an aivector
# error).  With the pass on, that configuration fits.
#
# It stays off by default because phase 3 contains a T.reduce_sum, exactly the
# kind of construct that failed before.  Turning it on requires a full test run
# plus kda_full's bit-identity criterion for varlen.
if os.environ.get("KDA_CHUNKO_MEMPLAN", "0") == "1":
    pass_configs[tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING] = True


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
        4 * C * K * f32  # g_ub, k_ub, fold_ub, mcol_b
        + C * K * elem  # kh_ub
        + 3 * BC * K * f32  # qb_ub, pb_ub, ob_ub
        + 2 * BC * K * elem  # qb_half, ob_half
        + CV * C * elem  # ah_ub
        + 2 * K * f32
        + C * f32
        + 2 * BC * f32
    )  # mcol_ub, mrow_ub, red_ub


@tilelang.jit(out_idx=[-1], workspace_idx=[-6, -5, -4, -3, -2], pass_configs=pass_configs)
def chunk_o_ker(B, SEQ, H, HV, K, V, C, scale, BC=16, dtype="float16", accum_dtype="float"):
    assert C % (BC * VEC_NUM) == 0, f"need C % {BC * VEC_NUM} == 0, got C={C}"
    assert HV % H == 0, "HV must be divisible by H (GVA)"

    # ceil, not floor: the last chunk may be ragged.  SEQ is a Python int at
    # trace time, so this stays a compile-time constant, and all five
    # workspace first dims follow from N.
    N = -(-SEQ // C)  # chunks per sequence
    R = SEQ % C  # 0 when aligned; else the valid row count of the last chunk
    RAGGED = R != 0
    # Every token index below is clamped to the last valid row on a ragged
    # chunk.  Clamping is SAFE in this stage and would NOT be in stage 2: what
    # these rows produce lands in ah_ub (UB) or in ws_* rows >= R, and the only
    # GM store is the final l0c2gm of O, which is itself clamped -- so nothing
    # is ever written to GM at a clamped index.
    CV = C // VEC_NUM  # rows per vector core
    NB = C // BC  # anchor blocks per chunk
    NBV = NB // VEC_NUM  # anchor blocks built by each vector core
    NAB = CV // BC  # anchor blocks inside one core's rows
    GRP = HV // H  # value heads sharing one qk head

    def tok(idx):
        """Clamp a single-row token index to the last row that exists.

        Single-row reads have region extents [1, 1, 1, *]; find_active_dim_indices
        keeps only the last two *active* dims, so the token axis is folded into
        the base address and is never bounds-checked.  Unlike stage 2, clamping
        (rather than shortening a loop) is the right fix here: every value these
        rows produce ends up in UB or in ws_* rows >= R, and the only GM store is
        the final clamped l0c2gm of O.  Nothing is written at a clamped index.
        """
        return T.if_then_else(idx < SEQ, idx, SEQ - 1) if RAGGED else idx

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
            ob_half = T.alloc_ub([BC, K], dtype)

            ah_ub = T.alloc_ub([CV, C], dtype)

            mcol_ub = T.alloc_ub([C], accum_dtype)
            # The diagonal block's inclusive mask, read once as a block.
            # BC * BC fp32 = 1 KB.
            mblk_ub = T.alloc_ub([BC, BC], accum_dtype)
            # One materialised-broadcast tile for the kf build's column mask.
            # The diagonal patch does NOT need one -- its two masks became a
            # clamp and a post-reduction [BC] multiply, both free.
            mcol_b = T.alloc_ub([C, K], accum_dtype)
            red_ub = T.alloc_ub([BC], accum_dtype)
            ob_ub = T.alloc_ub([BC, K], accum_dtype)

            with T.Scope("V"):
                # ============ phase 1: build the three cube operands ==========
                # Chunk-resident G and K.  [C, *] over the token axis is a
                # strided move: the row length is K, the row pitch is HV*K
                # (H*K for the qk-head tensors).
                if RAGGED and n == N - 1:
                    # Both loads are clamped to the R valid rows, so the tail
                    # rows keep stale UB -- and g_ub is exponentiated while
                    # kh_ub becomes a cube operand, so a garbage row turns into
                    # inf and then NaN.  Zero is also the right filler: g = 0
                    # leaves the decay at 1 and k = 0 contributes nothing.
                    T.tile.fill(g_ub, 0.0)
                    T.tile.fill(kh_ub, 0)

                T.copy(G[bz, t0 : t0 + C, hv, 0:K], g_ub)
                T.copy(Kt[bz, t0 : t0 + C, hq, 0:K], kh_ub)
                T.copy(kh_ub, k_ub)

                for ab in range(NAB):
                    row = r0 + ab * BC  # chunk-local first row of the block

                    # Ragged tail: this BC-row block can start past SEQ, which
                    # would clamp validRow to 0, i.e. DataCopyExtParams with
                    # blockCount 0 -- outside the documented [1, 4095].  Fill
                    # first, then issue the DMA only if the block has any real
                    # row; the tile is exponentiated below, so a stale row
                    # would become inf.
                    if RAGGED and n == N - 1:
                        T.tile.fill(qb_half, 0)
                    if (not RAGGED) or t0 + row < SEQ:
                        T.copy(Q[bz, t0 + row : t0 + row + BC, hq, 0:K], qb_half)
                    T.copy(qb_half, qb_ub)
                    for i, d in T.Parallel(BC, K):
                        qb_ub[i, d] = qb_ub[i, d] * scale

                    # inter-chunk operand qg = (scale q) . e^{G}, exponent <= 0.
                    # Read straight out of the resident g_ub at a run-time row
                    # offset instead of re-reading the block from GM.  An older
                    # note here warned that UB->UB at a run-time row offset is
                    # unreliable -- that is about T.copy; reading at a run-time row
                    # offset *inside* T.Parallel (`g_ub[row + i, d]`) is the form
                    # phase 3 of this file has always used.
                    # It also drops the T.tile.fill: g_ub is already zeroed
                    # wholesale on the ragged branch, so a tail row reads 0 and
                    # exp(0) = 1 -- bit-identical to the old fill-then-exp.
                    for i, d in T.Parallel(BC, K):
                        pb_ub[i, d] = T.exp(g_ub[row + i, d])
                    for i, d in T.Parallel(BC, K):
                        ob_ub[i, d] = qb_ub[i, d] * pb_ub[i, d]
                    T.copy(ob_ub, ob_half)
                    T.copy(ob_half, ws_qg[cid, row, 0])

                    # strip operand qf = (scale q) . e^{G - G_anchor}.  Rows of
                    # the block are at or below the anchor, so the exponent is
                    # <= 0; qg cannot be reused for this (dividing it by
                    # e^{G_anchor} is exactly the overflow we are avoiding).
                    # Both the anchor row and the block come from the resident
                    # g_ub, saving one per-row GM read, one whole-block GM re-read
                    # and one fill.
                    T.tile.broadcast(ob_ub, g_ub[row, 0:K], axis=0)
                    for i, d in T.Parallel(BC, K):
                        pb_ub[i, d] = g_ub[row + i, d] - ob_ub[i, d]
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

                    # strictly-lower row ar is the indicator of j < ar; for
                    # ar == 0 it is all zeros, so that block's matmul yields
                    # exact zeros instead of reading dirty workspace memory.
                    T.copy(MskStr[ar, 0], mcol_ub)
                    # Materialised broadcast.  An operand indexed by an INNER
                    # variable alone lowers to one narrow instruction per row in
                    # this dialect: 64 x Sub(128) instead of a single Sub(8192).
                    # The destination is fold_ub itself -- already this line's
                    # target, so no extra UB (only 10,368 B are spare here and a
                    # fresh [C, K] fp32 tile would need 32,768 B).
                    T.tile.broadcast(fold_ub, g_ub[ar, 0:K], axis=0)
                    for j, d in T.Parallel(C, K):
                        fold_ub[j, d] = fold_ub[j, d] - g_ub[j, d]
                    # ★ Overflow guard as a clamp (free), same argument as the
                    # diagonal patch: for j < ar the exponent is already <= 0
                    # because G is non-increasing, and for j >= ar it is >= 0,
                    # where the clamp gives exactly the 0 the mask gave.
                    T.tile.min(fold_ub, fold_ub, 0.0)
                    for j, d in T.Parallel(C, K):
                        fold_ub[j, d] = T.exp(fold_ub[j, d])
                    for j, d in T.Parallel(C, K):
                        fold_ub[j, d] = fold_ub[j, d] * k_ub[j, d]
                    # ★ The discarding mask.  Unlike the diagonal patch there is
                    # NO reduction after this -- fold_ub goes straight to the
                    # cube as a [C, K] operand -- so it cannot be moved past one.
                    # Materialise the broadcast instead: mcol_ub is [C] indexed
                    # by the OUTER variable, so as written it was one narrow op
                    # per row with a PipeBarrier<PIPE_ALL> and a GetValue each.
                    T.tile.broadcast(mcol_b, mcol_ub, axis=1)
                    for j, d in T.Parallel(C, K):
                        fold_ub[j, d] = fold_ub[j, d] * mcol_b[j, d]
                    T.copy(fold_ub, kh_ub)
                    T.copy(kh_ub, ws_kf[cid, a, 0, 0])

                T.set_cross_flag("MTE3", 0)

                # ============ phase 3: patch the diagonal blocks =============
                T.wait_cross_flag(1)
                T.copy(ws_ao[cid, r0, 0], ah_ub)
                # MskInc[i_loc, a0] is just "jj <= rr" -- independent of a0 and of
                # vid -- so it is identically the [BC, BC] top-left corner of
                # MskInc, a compile-time constant.  Read it once and index
                # mblk_ub[rr, jj] afterwards: rr comes from range(BC) and is a
                # compile-time constant while jj is the inner variable, so there is
                # neither a broadcast nor any scalar traffic.  This used to be one
                # narrow GM read per row, 32 per block.
                T.copy(MskInc[0, 0], mblk_ub)

                for ab in range(NAB):
                    a0 = r0 + ab * BC  # anchor row of this block

                    # One block read of Q replaces BC narrow per-row GM reads, BC
                    # fp16->fp32 casts and BC scale multiplies.  The guard is word
                    # for word the one phase 1 uses for its Q block, so the ragged
                    # tail behaves identically (out-of-range rows are filled with 0
                    # rather than clamped to the last row): their results land past
                    # row R of ah_ub, and the final store of O is clamped, so they
                    # never leave.
                    if RAGGED and n == N - 1:
                        T.tile.fill(qb_half, 0)
                    if (not RAGGED) or t0 + a0 < SEQ:
                        T.copy(Q[bz, t0 + a0 : t0 + a0 + BC, hq, 0:K], qb_half)
                    T.copy(qb_half, qb_ub)
                    for i, d in T.Parallel(BC, K):
                        qb_ub[i, d] = qb_ub[i, d] * scale

                    for rr in range(BC):
                        i_loc = a0 + rr  # chunk-local row being fixed

                        # Two UB->UB row copies used to sit here purely to give
                        # the broadcasts a 1-D source.  Feeding the row slice to
                        # the broadcast directly removes both of them (32 loop
                        # iterations x 2 per block).
                        #
                        # Materialised broadcast, straight into pb_ub (which is
                        # rewritten in full).  This site is in the diagonal patch
                        # and runs BC rows x NAB blocks = 32 times per block, an
                        # order of magnitude more often than the kf build's.
                        T.tile.broadcast(pb_ub, g_ub[i_loc, 0:K], axis=0)
                        for jj, d in T.Parallel(BC, K):
                            pb_ub[jj, d] = pb_ub[jj, d] - g_ub[a0 + jj, d]
                        T.tile.min(pb_ub, pb_ub, 0.0)
                        for jj, d in T.Parallel(BC, K):
                            pb_ub[jj, d] = T.exp(pb_ub[jj, d])
                        T.tile.broadcast(ob_ub, qb_ub[rr, 0:K], axis=0)
                        for jj, d in T.Parallel(BC, K):
                            pb_ub[jj, d] = pb_ub[jj, d] * ob_ub[jj, d]
                        for jj, d in T.Parallel(BC, K):
                            pb_ub[jj, d] = pb_ub[jj, d] * k_ub[a0 + jj, d]
                        T.reduce_sum(pb_ub, red_ub, dim=-1)
                        for jj in T.Parallel(BC):
                            red_ub[jj] = red_ub[jj] * mblk_ub[rr, jj]
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


@tilelang.jit(out_idx=[-1], workspace_idx=[-6, -5, -4, -3, -2], pass_configs=pass_configs)
def chunk_o_ker_varlen(B, SEQ, H, HV, K, V, C, scale, NT_TOTAL, BC=16, dtype="float16", accum_dtype="float"):
    """O = (scale.Q . e^G) S + Aqk V', varlen.  Twin of chunk_o_ker above.

    See kda_chunk_cumsum.cumsum_ker_varlen for why the two are separate
    @tilelang.jit builders rather than one with a flag.

    Four things are specific to this stage:

      * **`tok()` clamps to this chunk's last valid row, not to SEQ.**  Under
        varlen `SEQ - 1` is the last row of the LAST sequence in the flattened
        batch, so every ragged chunk of every earlier sequence would take its
        anchor gate from a token in a different sequence.  `t0 + rows - 1` is
        both the semantically right row and in range.  Clamping (rather than
        shortening a loop, as stage 2 must) is still safe here for the original
        reason: what a clamped row produces lands in UB or in ws_* rows the
        bounded output store never reaches.
      * **The BC-block guard is `row < rows`, not a distance to SEQ.**  A block
        that starts past this sequence's eos still has data under it -- the next
        sequence -- so `t0 + row < SEQ` is true and would never fire.
      * **A row-validity mask goes into the qf exponent BEFORE exp().**  This is
        a latent overflow in the fixed-length kernel that varlen turns from rare
        into routine: in a partially valid BC block the pad rows hold G = 0, so
        `G - G_anchor` is `-G_anchor`, which is POSITIVE (G <= 0) and overflows
        to +inf under a strong gate.  qb_ub is 0 on those rows, so the next
        multiply is 0 * inf = NaN, and it reaches the cube through ws_qf and
        poisons every valid row of the strip matmul.  The file already folds a
        mask into this exponent on the COLUMN axis for exactly this reason
        (phase 2, and the diagonal patch); this adds the row axis.
      * **NB / NBV / NAB stay compile-time.**  They depend only on C, BC and
        VEC_NUM, never on `rows`, so the six cross-core flags stay outside every
        run-time branch.  Predicate loop BODIES, never trip counts.
    """

    assert C % (BC * VEC_NUM) == 0, f"need C % {BC * VEC_NUM} == 0, got C={C}"
    assert HV % H == 0, "HV must be divisible by H (GVA)"

    # Every token index below is clamped to this chunk's last valid row.
    # Clamping is SAFE in this stage and would NOT be in stage 2: what these
    # rows produce lands in ah_ub (UB) or in ws_* rows the bounded output store
    # never reaches, so nothing is written to GM at a clamped index.
    CV = C // VEC_NUM  # rows per vector core
    NB = C // BC  # anchor blocks per chunk
    NBV = NB // VEC_NUM  # anchor blocks built by each vector core
    NAB = CV // BC  # anchor blocks inside one core's rows
    GRP = HV // H  # value heads sharing one qk head

    def tok(idx, t0, rows):
        """Clamp a single-row token index to this chunk's last valid row.

        Single-row reads have region extents [1, 1, 1, *]; find_active_dim_indices
        keeps only the last two *active* dims, so the token axis is folded into
        the base address and is never bounds-checked.

        The bound is `t0 + rows - 1`, NOT `eos - 1` and NOT `SEQ - 1`.  `SEQ - 1`
        is the last token of the whole flattened batch and would reach into a
        different sequence entirely; `eos - 1` would be correct for a ragged
        chunk but wrong for a full one, where it names a token in a LATER chunk.
        """
        return T.if_then_else(idx < t0 + rows, idx, t0 + rows - 1)

    @T.prim_func
    def main(
        Q: T.Tensor([B, SEQ, H, K], dtype),  # type: ignore
        Kt: T.Tensor([B, SEQ, H, K], dtype),  # type: ignore
        Vt: T.Tensor([B, SEQ, HV, V], dtype),  # type: ignore  pseudo-values V'
        S: T.Tensor([B, HV, NT_TOTAL, K, V], dtype),  # type: ignore  entry states, chunk axis is the whole batch
        G: T.Tensor([B, SEQ, HV, K], accum_dtype),  # type: ignore
        MskInc: T.Tensor([C, C], accum_dtype),  # type: ignore  i >= j
        MskStr: T.Tensor([C, C], accum_dtype),  # type: ignore  i >  j
        # Meta goes before the workspaces: the decorator addresses them and
        # the output with NEGATIVE indices (workspace_idx=[-6..-2], out_idx=[-1]),
        # so appending after them would shift all six and the framework would
        # allocate real tensors as scratch.
        Meta: T.Tensor([NT_TOTAL, _VL.META_COLS], "int32"),  # type: ignore
        ws_qg: T.Tensor([HV * NT_TOTAL, C, K], dtype),  # type: ignore
        ws_qf: T.Tensor([HV * NT_TOTAL, C, K], dtype),  # type: ignore
        ws_kf: T.Tensor([HV * NT_TOTAL, NB, C, K], dtype),  # type: ignore
        ws_ao: T.Tensor([HV * NT_TOTAL, C, C], dtype),  # type: ignore  strip matmuls
        ws_aq: T.Tensor([HV * NT_TOTAL, C, C], dtype),  # type: ignore  full Aqk
        O: T.Tensor([B, SEQ, HV, V], dtype),  # type: ignore
    ):
        with T.Kernel(HV * NT_TOTAL, is_npu=True) as (cid, vid):
            ic = cid % NT_TOTAL  # flat chunk slot over the whole batch
            hv = cid // NT_TOTAL  # value head
            hq = hv // GRP  # qk head (GVA)

            # ic IS the slot stage 5 wrote this chunk's entry state into: the
            # metadata rows are built in (sequence, chunk) order, so row index
            # equals chunk_off[i_n] + i_t by construction.  That is why S is
            # indexed by ic below and no separate offset is needed.
            t0 = Meta[ic, _VL.META_T0]  # first token of this chunk, absolute
            rows = Meta[ic, _VL.META_ROWS]  # 1..C
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
            ob_half = T.alloc_ub([BC, K], dtype)

            ah_ub = T.alloc_ub([CV, C], dtype)

            mcol_ub = T.alloc_ub([C], accum_dtype)
            # The diagonal block's inclusive mask, read once as a block.
            # BC * BC fp32 = 1 KB.
            mblk_ub = T.alloc_ub([BC, BC], accum_dtype)
            # One materialised-broadcast tile for the kf build's column mask.
            # The diagonal patch does NOT need one -- its two masks became a
            # clamp and a post-reduction [BC] multiply, both free.
            mcol_b = T.alloc_ub([C, K], accum_dtype)
            vrow_ub = T.alloc_ub([BC], accum_dtype)  # row-validity mask for the qf build
            red_ub = T.alloc_ub([BC], accum_dtype)
            ob_ub = T.alloc_ub([BC, K], accum_dtype)

            with T.Scope("V"):
                # ============ phase 1: build the three cube operands ==========
                # Chunk-resident G and K.  [C, *] over the token axis is a
                # strided move: the row length is K, the row pitch is HV*K
                # (H*K for the qk-head tensors).
                # Unconditional: raggedness is a run-time property now.  The
                # bounded loads gap-fill the unused rows with 0 anyway, so this
                # is belt and braces -- but zero is also the right filler
                # semantically: g = 0 leaves the decay at 1, k = 0 contributes
                # nothing.
                T.tile.fill(g_ub, 0.0)
                T.tile.fill(kh_ub, 0)

                T.copy(G[0, t0 : t0 + rows, hv, 0:K], g_ub)
                T.copy(Kt[0, t0 : t0 + rows, hq, 0:K], kh_ub)
                T.copy(kh_ub, k_ub)

                for ab in range(NAB):
                    row = r0 + ab * BC  # chunk-local first row of the block

                    # Ragged tail: this BC-row block can start past SEQ, which
                    # would clamp validRow to 0, i.e. DataCopyExtParams with
                    # blockCount 0 -- outside the documented [1, 4095].  Fill
                    # first, then issue the DMA only if the block has any real
                    # row; the tile is exponentiated below, so a stale row
                    # would become inf.
                    # brows: how many of this BC block's rows exist.  The old
                    # guard tested `t0 + row < SEQ`, which under varlen is true
                    # for nearly every block -- the next sequence's data sits
                    # right there -- so it would never fire and the DMA would
                    # read the wrong sequence at full BC width.
                    bleft = T.if_then_else(rows > row, rows - row, 0)
                    brows = T.if_then_else(bleft < BC, bleft, BC)

                    T.tile.fill(qb_half, 0)
                    if brows > 0:
                        T.copy(Q[0, t0 + row : t0 + row + brows, hq, 0:K], qb_half)
                    T.copy(qb_half, qb_ub)
                    for i, d in T.Parallel(BC, K):
                        qb_ub[i, d] = qb_ub[i, d] * scale

                    # inter-chunk operand qg = (scale q) . e^{G}, exponent <= 0.
                    # Read straight out of the resident g_ub at a run-time row
                    # offset instead of re-reading the block from GM.  An older
                    # note here warned that UB->UB at a run-time row offset is
                    # unreliable -- that is about T.copy; reading at a run-time row
                    # offset *inside* T.Parallel (`g_ub[row + i, d]`) is the form
                    # phase 3 of this file has always used.
                    # It also drops the T.tile.fill: g_ub is already zeroed
                    # wholesale on the ragged branch, so a tail row reads 0 and
                    # exp(0) = 1 -- bit-identical to the old fill-then-exp.
                    for i, d in T.Parallel(BC, K):
                        pb_ub[i, d] = T.exp(g_ub[row + i, d])
                    for i, d in T.Parallel(BC, K):
                        ob_ub[i, d] = qb_ub[i, d] * pb_ub[i, d]
                    T.copy(ob_ub, ob_half)
                    T.copy(ob_half, ws_qg[cid, row, 0])

                    # strip operand qf = (scale q) . e^{G - G_anchor}.  Rows of
                    # the block are at or below the anchor, so the exponent is
                    # <= 0; qg cannot be reused for this (dividing it by
                    # e^{G_anchor} is exactly the overflow we are avoiding).
                    # Both the anchor row and the block come from the resident
                    # g_ub, saving one per-row GM read, one whole-block GM re-read
                    # and one fill.
                    T.tile.broadcast(ob_ub, g_ub[row, 0:K], axis=0)
                    for i, d in T.Parallel(BC, K):
                        pb_ub[i, d] = g_ub[row + i, d] - ob_ub[i, d]
                    # ★ Row-validity mask, folded in BEFORE exp().  A pad row of
                    # a partially valid block holds G = 0, so the line above
                    # leaves it at -G_anchor, which is POSITIVE (G <= 0) and
                    # overflows to +inf under a strong gate; qb_ub is 0 on that
                    # row, so the multiply two lines down would be 0 * inf = NaN
                    # and would reach the cube through ws_qf.  MskInc[i, j] is
                    # 1 iff i >= j, so MskInc[rows - 1, row + jj] is 1 exactly
                    # when row + jj < rows -- the row-validity indicator.  Row
                    # `row` is a multiple of BC, so this [BC] fp32 slice is
                    # 64B-aligned.
                    T.copy(MskInc[rows - 1, row], vrow_ub)
                    # The only OUTER-variable broadcast in this file, and the
                    # worst form there is: the generated AscendC is an
                    # outer_broadcast_idx loop with a GetValue and a narrow Muls
                    # per iteration.  Materialised into ob_ub it is one wide
                    # instruction.  The values do not change: on a full block vrow
                    # is all ones and this step is already an identity.
                    T.tile.broadcast(ob_ub, vrow_ub, axis=1)
                    for i, d in T.Parallel(BC, K):
                        pb_ub[i, d] = pb_ub[i, d] * ob_ub[i, d]
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

                    # strictly-lower row ar is the indicator of j < ar; for
                    # ar == 0 it is all zeros, so that block's matmul yields
                    # exact zeros instead of reading dirty workspace memory.
                    T.copy(MskStr[ar, 0], mcol_ub)
                    # Materialised broadcast.  An operand indexed by an INNER
                    # variable alone lowers to one narrow instruction per row in
                    # this dialect: 64 x Sub(128) instead of a single Sub(8192).
                    # The destination is fold_ub itself -- already this line's
                    # target, so no extra UB (only 10,368 B are spare here and a
                    # fresh [C, K] fp32 tile would need 32,768 B).
                    T.tile.broadcast(fold_ub, g_ub[ar, 0:K], axis=0)
                    for j, d in T.Parallel(C, K):
                        fold_ub[j, d] = fold_ub[j, d] - g_ub[j, d]
                    # ★ Overflow guard as a clamp (free), same argument as the
                    # diagonal patch: for j < ar the exponent is already <= 0
                    # because G is non-increasing, and for j >= ar it is >= 0,
                    # where the clamp gives exactly the 0 the mask gave.
                    T.tile.min(fold_ub, fold_ub, 0.0)
                    for j, d in T.Parallel(C, K):
                        fold_ub[j, d] = T.exp(fold_ub[j, d])
                    for j, d in T.Parallel(C, K):
                        fold_ub[j, d] = fold_ub[j, d] * k_ub[j, d]
                    # ★ The discarding mask.  Unlike the diagonal patch there is
                    # NO reduction after this -- fold_ub goes straight to the
                    # cube as a [C, K] operand -- so it cannot be moved past one.
                    # Materialise the broadcast instead: mcol_ub is [C] indexed
                    # by the OUTER variable, so as written it was one narrow op
                    # per row with a PipeBarrier<PIPE_ALL> and a GetValue each.
                    T.tile.broadcast(mcol_b, mcol_ub, axis=1)
                    for j, d in T.Parallel(C, K):
                        fold_ub[j, d] = fold_ub[j, d] * mcol_b[j, d]
                    T.copy(fold_ub, kh_ub)
                    T.copy(kh_ub, ws_kf[cid, a, 0, 0])

                T.set_cross_flag("MTE3", 0)

                # ============ phase 3: patch the diagonal blocks =============
                T.wait_cross_flag(1)
                T.copy(ws_ao[cid, r0, 0], ah_ub)
                # MskInc[i_loc, a0] is just "jj <= rr" -- independent of a0 and of
                # vid -- so it is identically the [BC, BC] top-left corner of
                # MskInc, a compile-time constant.  Read it once and index
                # mblk_ub[rr, jj] afterwards: rr comes from range(BC) and is a
                # compile-time constant while jj is the inner variable, so there is
                # neither a broadcast nor any scalar traffic.  This used to be one
                # narrow GM read per row, 32 per block.
                T.copy(MskInc[0, 0], mblk_ub)

                for ab in range(NAB):
                    a0 = r0 + ab * BC  # anchor row of this block

                    # One block read of Q.  The guard is word for word the one
                    # phase 1 of this builder uses: under varlen `t0 + row < SEQ`
                    # is almost always true (the next sequence's data sits right
                    # there), so the width has to be bounded by the row count.
                    ableft = T.if_then_else(rows > a0, rows - a0, 0)
                    abrows = T.if_then_else(ableft < BC, ableft, BC)
                    T.tile.fill(qb_half, 0)
                    if abrows > 0:
                        T.copy(Q[0, t0 + a0 : t0 + a0 + abrows, hq, 0:K], qb_half)
                    T.copy(qb_half, qb_ub)
                    for i, d in T.Parallel(BC, K):
                        qb_ub[i, d] = qb_ub[i, d] * scale

                    for rr in range(BC):
                        i_loc = a0 + rr  # chunk-local row being fixed

                        # Two UB->UB row copies used to sit here purely to give
                        # the broadcasts a 1-D source.  Feeding the row slice to
                        # the broadcast directly removes both of them (32 loop
                        # iterations x 2 per block).
                        #
                        # Materialised broadcast, straight into pb_ub (which is
                        # rewritten in full).  This site is in the diagonal patch
                        # and runs BC rows x NAB blocks = 32 times per block, an
                        # order of magnitude more often than the kf build's.
                        T.tile.broadcast(pb_ub, g_ub[i_loc, 0:K], axis=0)
                        for jj, d in T.Parallel(BC, K):
                            pb_ub[jj, d] = pb_ub[jj, d] - g_ub[a0 + jj, d]
                        T.tile.min(pb_ub, pb_ub, 0.0)
                        for jj, d in T.Parallel(BC, K):
                            pb_ub[jj, d] = T.exp(pb_ub[jj, d])
                        T.tile.broadcast(ob_ub, qb_ub[rr, 0:K], axis=0)
                        for jj, d in T.Parallel(BC, K):
                            pb_ub[jj, d] = pb_ub[jj, d] * ob_ub[jj, d]
                        for jj, d in T.Parallel(BC, K):
                            pb_ub[jj, d] = pb_ub[jj, d] * k_ub[a0 + jj, d]
                        T.reduce_sum(pb_ub, red_ub, dim=-1)
                        for jj in T.Parallel(BC):
                            red_ub[jj] = red_ub[jj] * mblk_ub[rr, jj]
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
                # ic is the flat chunk slot, which is exactly what stage 5
                # wrote (chunk_off[i_n] + i_t).  Both wrappers assert the shape.
                T.copy(S[0, hv, ic, 0, 0], s_l1)
                T.gemm_v0(qg_l1, s_l1, o_l0, init=True)

                T.copy(ws_aq[cid, 0, 0], aq_l1)
                # Bounded: the rows past eos belong to the next sequence and
                # are a torch.empty output the owning block may not have written
                # yet, so they can hold any bit pattern including inf/NaN.
                T.copy(Vt[0, t0 : t0 + rows, hv, 0:V], v_l1)
                T.gemm_v0(aq_l1, v_l1, o_l0, init=False)

                # fixpipe straight into O: a [C, V] tile with row pitch HV*V
                # Bounded.  This is the only GM store in the stage, and left
                # at C it writes up to C-1 rows of O straight onto the next
                # sequence.  `rows` is 1..C and never 0 -- 0 would be read as
                # "no tail, write the whole tile" (common.h:174, measured in
                # PROBES/probe_varlen5.log).
                T.copy(o_l0, O[0, t0 : t0 + rows, hv, 0:V])

    return main


_MSK_CACHE = {}


def _causal_masks(C, device):
    """The inclusive (i >= j) and strict (i > j) [C, C] indicators, built once.

    These are compile-time constants of the operator that used to be rebuilt on
    every call.  This was the single largest host-side cost in the pipeline: the
    2026-08-21 full-pipeline profile shows five host ops between chunk_h and
    chunk_o -- Range, GreaterEqual, Cast, Greater, Cast -- totalling 41.52 us,
    which is 94% of the 44.18 us gap there.  The gap had been attributed to
    waiting for chunk_h's `states` to land in GM; it was not, it was this.

    Safe to share across calls: both are read-only kernel inputs (neither is in
    out_idx or workspace_idx).  Same pattern as _IDT_CACHE in kda_solve_tril.py.
    """
    key = (C, str(device))
    m = _MSK_CACHE.get(key)
    if m is None:
        idx = torch.arange(C, device=device)
        m = ((idx[:, None] >= idx[None, :]).float(), (idx[:, None] > idx[None, :]).float())
        _MSK_CACHE[key] = m
    return m


def chunk_o(q, k, vnew, states, G, C=64, BC=16, scale=None, cu_seqlens=None):
    """Host wrapper.  O = (scale.Q . e^G) S + Aqk V', all external layout.

    q, k    [B, SEQ, H,  K]   dtype
    vnew    [B, SEQ, HV, V]   dtype   V' from stage 5
    states  [B, HV, N, K, V]  dtype   per-chunk entry states from stage 5
    G       [B, SEQ, HV, K]   fp32    stage 1 output

    The host only builds the two constant masks and looks the dtype up; no
    transposes, no reshapes, no state shuffling.

    With ``cu_seqlens`` the inputs are a flattened varlen batch (B == 1) and
    ``states`` is [1, HV, NT_TOTAL, K, V] -- the chunk axis spans the whole
    batch, with sequence n's chunk i at slot ``chunk_off[n] + i``.  That is the
    layout stage 5 writes, and the assert below pins it.
    """
    B, SEQ, H, K = q.shape
    HV, V = vnew.shape[2], vnew.shape[-1]
    assert C % (BC * VEC_NUM) == 0, f"need C % {BC * VEC_NUM} == 0, got C={C}"
    assert K % 16 == 0 and V % 16 == 0, "K and V must be multiples of 16"
    # The dtype is threaded into the kernel as a compile-time constant, so all
    # four Cube operands have to agree on it.  Kept on one line deliberately:
    # ruff 0.6.5 (requirements-lint.txt) and current ruff (what the CI action
    # installs) wrap a split assert in opposite directions, so a single line
    # under the 140-column limit is the only form both leave alone.
    assert k.dtype == q.dtype and vnew.dtype == q.dtype and states.dtype == q.dtype, "Q / Kt / V' / S must share one dtype"
    assert G.shape == (B, SEQ, HV, K), f"G must be [B, SEQ, HV, K], got {tuple(G.shape)}"
    # The chunk count must match stage 5's exactly, or the entry state read
    # here comes from the wrong chunk.  The slot index is folded into a base
    # address with no bounds check, so an off-by-one is silent.
    if cu_seqlens is None:
        _N = -(-SEQ // C)  # ceil: must match chunk_h's N_CHUNK exactly
        assert states.shape == (B, HV, _N, K, V), f"states must be [B, HV, ceil(SEQ/C), K, V], got {tuple(states.shape)}"

    elem = torch.finfo(q.dtype).bits // 8
    need = ub_bytes(C, K, V, BC, elem)
    assert need <= UB_LIMIT, f"UB budget {need} > {UB_LIMIT} for C={C} K={K}; lower C"

    # SEQ == 0 slips past the assert above (0 % C == 0) and would launch a
    # zero-block grid over an unwritten output.  A zero-length sequence is legal
    # input; there is no query to read out, so O is empty along the token axis.
    #
    # Under varlen this fires only for a wholly empty batch; a single empty
    # sequence contributes zero chunks and so gets no block.
    if SEQ == 0:
        return torch.empty((B, 0, HV, V), device=q.device, dtype=q.dtype)

    if scale is None:
        scale = K**-0.5

    msk_inc, msk_str = _causal_masks(C, q.device)

    dt = {torch.float16: "float16", torch.bfloat16: "bfloat16"}[q.dtype]
    if cu_seqlens is None:
        ker = chunk_o_ker(B, SEQ, H, HV, K, V, C, float(scale), BC=BC, dtype=dt)
        return ker(q, k, vnew, states, G, msk_inc, msk_str)

    bounds = _VL.varlen_bounds(cu_seqlens, q=q, k=k, v=vnew, g=G)
    meta = _VL.chunk_meta(bounds, C, q.device)
    nt_total = meta.shape[0]
    assert nt_total > 0, "a non-empty batch must produce at least one chunk"
    assert int(meta[:, _VL.META_ROWS].sum()) == SEQ, "chunk metadata does not cover every token exactly once"
    assert int(meta[:, _VL.META_ROWS].min()) >= 1, "a chunk with zero valid rows must not exist"
    assert states.shape == (B, HV, nt_total, K, V), f"states must be [1, HV, NT_TOTAL, K, V], got {tuple(states.shape)}"
    ker = chunk_o_ker_varlen(B, SEQ, H, HV, K, V, C, float(scale), nt_total, BC=BC, dtype=dt)
    return ker(q, k, vnew, states, G, msk_inc, msk_str, meta)


# ----------------------------------------------------------------- test
def _relerr(x, r):
    r = r.float()
    return (x.float() - r).abs().max().item() / max(r.abs().max().item(), 1e-9)


def _case(B, SEQ, H, HV, K, V, C, gate, dtype=torch.float16, BC=16):
    q, k, v, g, beta, _ = kda_chunk_ref.make_inputs(B, SEQ, H, HV, K, V, dtype=dtype, gate=gate)
    st = kda_chunk_ref.stage_tensors(q, k, v, g, beta, C=C, BC=BC)

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


def _vcase(seqlens, H, HV, K, V, C, gate, dtype=torch.float16, BC=16, note=""):
    """One varlen batch against the stage-6 golden, over the WHOLE flat token axis.

    Whole-axis on purpose: the single L0C -> GM store is what would corrupt a
    neighbour, and it writes finite, plausible values, so a per-sequence
    comparison would pass while the batch is wrong.  A non-finite result would
    instead point at the qf exponent, whose pad rows overflow to +inf and then
    to 0 * inf = NaN unless the row-validity mask is folded in before exp().
    """
    q, k, v, g, beta, _, cu = kda_chunk_ref.make_varlen_inputs(seqlens, H, HV, K, V, dtype=dtype, gate=gate)
    st = kda_chunk_ref.stage_tensors(q.cpu(), k.cpu(), v.cpu(), g.cpu(), beta.cpu(), C=C, BC=BC, cu_seqlens=cu.cpu())

    Vn = st["Vt"].contiguous().to(dtype).npu()
    S = st["states"].contiguous().to(dtype).npu()
    G = st["G"].contiguous().npu()

    got = chunk_o(q, k, Vn, S, G, C=C, BC=BC, cu_seqlens=cu)

    err = _relerr(got.cpu(), st["o"])
    finite = bool(torch.isfinite(got.float()).all())
    shape_ok = tuple(got.shape) == tuple(st["o"].shape)
    tol = 5e-3 if dtype == torch.float16 else 3e-2
    ok = finite and shape_ok and err < tol
    tag = "fp16" if dtype == torch.float16 else "bf16"
    print(
        f"  {str(seqlens):24s} HV{HV} K{K:<4d} C{C:<2d} {tag} {gate:8s} rel={err:.2e} finite={'Y' if finite else 'N'}  {'ok' if ok else 'FAIL'}  {note}"
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

    print("== ragged tail (SEQ % C != 0) ==")
    ok &= _case(2, 70, 1, 2, 64, 64, 64, "normal")  # R=6, one anchor block partly valid
    ok &= _case(1, 33, 1, 1, 64, 64, 32, "forget")  # R=1
    ok &= _case(1, 65, 1, 1, 128, 128, 64, "forget")  # K3 dim, R=1
    ok &= _case(2, 100, 2, 4, 64, 64, 32, "extreme")  # GVA + extreme gate on the tail
    ok &= _case(1, 96, 1, 1, 64, 64, 64, "normal")  # R=32, exact core boundary

    print("== gates that underflow inside a chunk (the NaN trap) ==")
    ok &= _case(1, 128, 1, 2, 64, 64, 64, "extreme")
    ok &= _case(1, 128, 1, 1, 64, 64, 32, "keep")

    print("== bf16 (dtype is threaded through from the inputs) ==")
    ok &= _case(2, 128, 2, 4, 64, 64, 64, "normal", dtype=torch.bfloat16)
    ok &= _case(1, 128, 1, 1, 128, 128, 64, "forget", dtype=torch.bfloat16)

    print("== varlen (cu_seqlens) ==")
    ok &= _vcase([64, 64, 64], 1, 2, 64, 64, 64, "normal", note="equal, chunk-aligned")
    ok &= _vcase([70, 33, 129], 1, 2, 64, 64, 64, "normal", note="every sequence ragged -- interior tails")
    ok &= _vcase([70, 0, 129], 1, 2, 64, 64, 64, "forget", note="empty sequence in the middle")
    ok &= _vcase([0, 70], 1, 2, 64, 64, 64, "normal", note="empty sequence first")
    ok &= _vcase([70, 0], 1, 2, 64, 64, 64, "normal", note="empty sequence last")
    ok &= _vcase([1, 200], 1, 2, 64, 64, 64, "forget", note="one token -- BC block partially valid")
    ok &= _vcase([20, 20], 1, 2, 64, 64, 64, "normal", note="both shorter than C/2")
    ok &= _vcase([70, 33], 1, 2, 64, 64, 64, "extreme", note="extreme gate on a partial BC block (the qf overflow)")
    ok &= _vcase([65, 65], 1, 1, 128, 128, 64, "forget", note="K3 dim, one valid tail row each")
    ok &= _vcase([100, 28], 2, 4, 64, 64, 32, "extreme", note="GVA + extreme gate, C = 32")
    ok &= _vcase([5], 1, 1, 64, 64, 64, "extreme", note="N = 1, shorter than one BC block")
    ok &= _vcase([70, 33], 2, 4, 64, 64, 64, "forget", dtype=torch.bfloat16, note="bf16 + GVA")

    if ok:
        print("Kernel Output Match!")
    else:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
