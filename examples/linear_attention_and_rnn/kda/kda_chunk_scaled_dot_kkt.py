"""KDA L1 stage 2: the decayed Gram matrix, at the frozen L0 interface.

    L[i, j] = beta_i * sum_d k[i, d] k[j, d] exp(G[i, d] - G[j, d]),   j < i

This is the stage where KDA stops being GDN.  GDN's decay factor exp(g_i - g_j)
is one scalar per (i, j), so its kernel lets the cube compute a plain K K^T and
rescales the result on the vector unit afterwards.  Per channel the factor is
exp(G[i, d] - G[j, d]): it sits *inside* the sum over d and cannot be hoisted
out of the matmul.

Why not the one-shot fold
-------------------------
Algebraically the factor splits as e^{G_i} . e^{-G_j} and can be folded into the
two operands, which keeps the whole thing a single matmul.  Numerically that is
a trap: G is a cumulative sum of non-positive gates, so e^{-G_j} grows without
bound down the chunk.  Measured on CPU with the reference layer, the naive fold
produces 2624 non-finite entries at the "forget" gate (min Gamma_C = -209) and
3840 at "extreme" (-841).  ``kda_chunk_ref.test_naive_fold_blows_up`` demonstrates
it rather than merely asserting it.

What this kernel does instead
-----------------------------
It evaluates the contraction directly, one output row at a time, with the causal
mask folded into the exponent *before* exp():

    e      = (G[i, :] - G[j, :]) * mask[i, j]   -> <= 0 for j < i, exactly 0 else
    L[i, j] = mask[i, j] * beta_i * sum_d k[i, d] k[j, d] e^{e}

Two properties make this safe by construction, and both are load-bearing:

  * Only differences of G are ever exponentiated, never a ratio of two exps.
    Deep inside a chunk e^{G_i} and e^{G_j} both underflow to 0 and 0/0 is NaN.
  * The mask multiplies the *exponent*, not the result.  G is non-increasing in
    i, so the j < i half of the exponent is non-positive for free, and the j > i
    half is crushed to exactly 0 before exp() can overflow it to inf.  Masking
    after exp() lets a single steep channel turn the discarded half into inf and
    then 0 * inf = NaN poisons the half that is kept.

This is strictly safer than the BC=16 anchored blocking used by
``kda_chunk_ref._decayed_dot`` (and by fla): the anchored form still bounds both
folded factors by 1, but this form never forms a positive exponent at all.  The
two agree to fp32 rounding, so the anchored golden validates this kernel
directly.  The reason to prefer anchoring is *speed*, not numerics -- it is what
lets the off-diagonal strips go to the cube.  See "known limitations".

Frozen interface (FLA contract)
-------------------------------
    Kt   [B, SEQ, H,  K]   dtype (fp16 / bf16)      read with hq = hv // GRP
    G    [B, SEQ, HV, K]   fp32, chunk-local cumsum of the log gate, <= 0
    Beta [B, SEQ, HV, 8]   fp32, only [..., 0] is used (32B alignment padding)
    Msk  [C, C]            fp32, strictly lower triangular
    L    [B, SEQ, HV, C]   dtype -- consumed as-is by stage 3 solve_tril

Note the token axis and the head-dim axis have the head axis between them, so a
[C, K] tile of one head is a *strided* read: C rows of K contiguous elements,
H * K apart in Kt and HV * K apart in G.  ``T.copy`` infers its region from the
*trailing* dims of the source, so the bare ``T.copy(Kt[bz, base, hq, 0], tile)``
that worked for the old [B, H, L, DK] layout would now read C *heads* of one
token, silently.  The token range is therefore written out as a slice --
``Kt[bz, base:base + C, hq, 0:K]``, region [1, C, 1, K].  A unit extent to the
left of the innermost one gets folded into the source row stride
(``compute_strideN`` in src/op/ascend.cc), so this lowers to a single
``DataCopyPad`` of C blocks with srcStride = (H - 1) * K elements.  Same index
shape as the GM->L1 reads in examples/deepseek_v4/lightning_indexer.py.
Single rows (query row, gate row, beta, output row) stay plain BufferLoads: a
one-row region needs no stride and is the pattern the L0 kernel already uses.

Parallel decomposition
----------------------
grid = B * HV * chunk_num, one block per (batch, value head, chunk).  The two
vector cores split the *rows* of the C x C output between them; both need every
key of the chunk, so both load the whole resident tile.  Nothing is contracted
across cores, so there is no cross-core communication and no workspace.

Ragged tail
-----------
``SEQ % C != 0`` is supported.  Two things are needed and they differ in kind:

  * The two resident [C, K] tiles are loaded by a clamped copy, so their rows
    R..C-1 hold stale UB.  Those rows are not inert -- ``gfull_ub`` is fed to
    ``exp()``, and the mask multiply that should discard the result computes
    0 * inf = NaN instead, landing in the reduction of a *valid* row.  Both
    tiles are therefore zeroed first on the ragged chunk.
  * The per-row store is a SINGLE-ROW copy, region [1, 1, 1, C].
    ``find_active_dim_indices`` keeps only the last two *active* dims, so the
    token axis is folded into the base address and never bounds-checked.
    Clamping the row index would be worse than useless -- it would write every
    pad row on top of ``L[SEQ-1]``.  The fix is to shorten the loop's trip count
    so no pad row is produced at all.  The two vector cores may then run
    different counts, which is safe here specifically because this kernel has no
    cross-core flag to deadlock on.

The mask-before-exp discipline that exists for the numerical-range reason above
also makes every pad *column* exactly 0 for free: a pad column j satisfies
j >= R > i for every valid row i, so the causal mask already kills it.

Known limitations
-----------------
  * DONE (L3 #14): the anchored BC decomposition described here is what BOTH
    builders now do.  The off-diagonal strips go to ``T.gemm_v0`` and only the
    diagonal blocks stay element-wise.  The two builders' arithmetic is
    identical operation for operation ON PURPOSE -- kda_full's varlen gate is
    bit-identity between them, not a tolerance.
  * Both vector cores load the full [C, K] key/gate tiles although core 0 only
    ever reads columns j < C/2.  Harmless duplication, kept for index clarity.
  * C must be even (two vector cores split C rows) and K % 16 == 0 / C % 16 == 0
    so every row copy starts on a 32B boundary.
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

# Only AUTO_SYNC, matching all six GDN kernels.  MEMORY_PLANNING is deliberately
# left off: on the backward bwd_dot it aliased a reduction target with a scratch
# tile -- the registers held the right values and only the store wrote zeros.
pass_configs = {tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True}

UB_LIMIT = 196352

# ------------------------------------------------------------------ route B
# The diagonal blocks can join the strips on the cube if the gate is folded into
# the two operands first, turning A[i,j] = sum_d X[i,d] Y[j,d] exp(G[i,d]-G[j,d])
# into a plain matmul of X.exp(G-Ga) against Y.exp(Ga-G).  The strip already does
# exactly that; the only thing stopping the SAME matmul from producing the
# diagonal columns as well is the clamp below, which pins the column exponent at
# 0 because the diagonal columns are the ones where it is positive and unbounded.
#
# Raising the clamp is all the arithmetic needs.  What bounds how far it can be
# raised is the operand dtype: the cube takes fp16 or bf16, and fp16 tops out at
# e^11.09 while the forget gate alone spans 70 nats inside one block.  bf16
# reaches e^88, so the operands become bf16 and 80 leaves room for the |k| factor
# and the K-long accumulation on top of exp (e^88 times a k of 4 already
# overflows bf16, and an inf here becomes a NaN when the plane mask multiplies
# it by zero).
#
# Accuracy was settled before any of this was written, on the CPU reference over
# all six stages (PERF/probes/probe_routeB_e2e.py): route B on L costs 1.4e-4
# end to end against a 5e-3 tolerance, it does not amplify through solve_tril /
# wy_fast / chunk_h, and it does not accumulate with chunk count.  Route B on
# chunk_o's Aqk costs 2.35e-3 -- sixteen times more, because L is damped by the
# triangular solve and Aqk is not -- which is why only this stage takes it.
# The upper bound on the column exponent, in nats.  It is the one number in this
# file that is a real approximation, so here is where it comes from.
#
# The operand is k * exp(clamp) and it is bf16; the cube then accumulates K of
# those against row operands, which are bounded by |k| because their own exponent
# is <= 0, into an fp32 accumulator.  Two conditions:
#
#     operand    |k| * e^clamp          <  3.39e38  (bf16 max)
#     accumulate K * |k|^2 * e^clamp    <  3.40e38  (fp32 max)
#
# The second binds.  At clamp = 80, e^80 = 5.54e34, so with K = 128 the
# accumulation is safe up to |k| = 6.9 per channel -- against the 0.09 an L2
# normalised key actually carries, a margin of 77x.  Callers feeding keys two
# orders of magnitude larger than normalised would need this lowered.
#
# The official operator's constant is NOT reusable here even though the geometry
# looks identical.  Its KDA_EXP2_CLAMP = 80 is base TWO, i.e. 55.45 nats, which
# was tried: it is safe for any |k| at all, and it FAILS this stage's own
# acceptance gate, because the forget gate spans about 70 nats inside one block
# and saturating that corrupts L to 6.3e-3 against a 5e-3 tolerance (bf16 data
# reached 1.25e-1 against 3e-2).  The end-to-end error stays inside tolerance --
# solve_tril damps stage 2 by an order of magnitude -- which is presumably why
# the bound works for an operator measured end to end.  We are measured at
# stage 2 as well, so we need the wider bound and the |k| assumption that comes
# with it.
ROUTE_B_CLAMP = 80.0
VEC_NUM = 2  # two vector cores split the C rows
BETA_PAD = 8  # width of the beta staging tile; CV*8*4 is a multiple of 32 B


def _ub_bytes(C, K, BC, elem):
    """Byte budget of the allocations below.

    The anchored form has the same footprint shape as chunk_o's, because it is
    now the same kind of kernel: five [C, K] planes dominate, everything else is
    a [BC, K] block or a vector.  No worst-case scratch term is carried here --
    every T.Parallel below is a single operation, so the lowering allocates no
    tmp_ub tile.  (The pre-anchor version budgeted one defensively and still fit;
    this one has 22 KB of headroom without the allowance.)
    """
    CV = C // VEC_NUM
    planes = 4 * C * K * 4  # g_ub, k_ub, fold_ub, mcol_b
    planes_low = C * K * elem  # kh_ub
    blk = BC * K * 4 + BC * K * elem  # pb_ub, ob_half (ob_ub was eliminated)
    out = CV * C * 4 + CV * C * elem  # ah_ub (fp32) + ahh_ub (dtype)
    small = C * 4 + BC * 4 + BC * BC * 4  # mcol_ub, red_ub, mblk_ub
    # Batched beta: the [CV, 8] staging tile, the [CV] vector, and the [CV, C]
    # materialised broadcast.
    #
    # Note that this function does NOT account for the compiler's own implicit
    # scratch, which grows with instruction width.  Do not widen an instruction on
    # the strength of this number alone -- that mistake has been made twice here
    # already (chunk_o's [32, 128] tile, and widening kkt's reduce_sum to [C, K]).
    beta = CV * BETA_PAD * 4 + CV * 4 + CV * C * 4
    return planes + planes_low + blk + out + small + beta


@tilelang.jit(out_idx=[-1], workspace_idx=[-4, -3, -2], pass_configs=pass_configs)
def kkt_ker(B, SEQ, H, HV, K, C, BC=16, dtype="float16", accum_dtype="float", route_b=False):
    """Anchored form: off-diagonal strips on the cube, diagonal blocks on vector.

    This replaces the row-at-a-time evaluation that ran entirely on the vector
    unit.  The maths is the anchored decomposition that ``kda_chunk_ref``
    already uses as its golden (``_decayed_dot``), so nothing about the result
    is new -- what is new is that the strips are now a matmul.

    Per row block ``a`` (rows ar .. ar+BC-1, anchored at its FIRST row ar):

        row operand     kr[i, d] = beta_i * k[i, d] * exp(G[i, d] - G[ar, d])
        column operand  kf[a][j, d] = k[j, d] * exp(G[ar, d] - G[j, d]) * [j < ar]
        strip           L[i, j] = sum_d kr[i, d] * kf[a][j, d]

    The anchor cancels in the product, so this is an identity, not an
    approximation.  It is also *bounded*, which the one-shot fold is not: G is
    non-increasing along the token axis, so i >= ar gives G[i] - G[ar] <= 0 and
    j < ar gives G[ar] - G[j] <= 0.  Both folded factors are therefore <= 1 in
    magnitude and neither can overflow -- the failure the one-shot fold shows
    (2624 non-finite entries at the "forget" gate, 3840 at "extreme"; see
    ``kda_chunk_ref.test_naive_fold_blows_up``) cannot occur here.

    Beta is folded into the row operand rather than applied afterwards.  It is a
    per-row scalar and kr is a per-row-block tile, so it costs nothing there,
    and it keeps the strip path to a SINGLE fp16 rounding: cube -> ws_ls -> L,
    with no post-multiply in between.  Identical to the golden, which has
    ``A = _dot(...) * beta`` -- scaling row i of X scales row i of the product.

    The strip needs no output mask.  Every column it produces satisfies
    j < ar <= i, which is exactly the strictly-lower condition, and columns
    j >= ar were already zeroed in kf[a] by the mask on the operand.

    The diagonal blocks stay element-wise, because no anchor bounds both sides
    inside a block: for i and j in the same block, either i - anchor or
    anchor - j has the wrong sign.  That path is the old kernel's inner loop
    with its [C, K] tiles narrowed to [BC, K] -- a quarter of the work at
    BC=16 -- and it keeps both L2 transformations (clamp instead of a pre-exp
    mask, and the discarding mask moved past the reduction).

    Phases, mirroring kda_chunk_o exactly:

        V  build kr (this core's rows) and kf[a] (alternating anchors)   flag 0
        C  NB strip matmuls kr[a] @ kf[a]^T                     -> ws_ls, flag 1
        V  patch the diagonal blocks over the strips, store L

    ONE THING THAT CHANGED AND IS EASY TO MISS
    ------------------------------------------
    The pre-anchor kernel let the two vector cores run DIFFERENT trip counts on
    a ragged chunk, and its comment justified that with "safe because this
    kernel has no cross-core flag to deadlock on (verified: zero
    set_cross_flag / wait_cross_flag)".  That justification is now void -- there
    are two flags.  Every loop below therefore has a COMPILE-TIME trip count and
    the ragged handling moved into per-iteration guards, which is what chunk_o
    does for the same reason.  A runtime-zero-trip T.serial would additionally
    break AUTO_SYNC, which cost a day in the varlen round.
    """
    assert HV % H == 0, "HV must be divisible by H (GVA)"
    assert C % (BC * VEC_NUM) == 0, f"need C % {BC * VEC_NUM} == 0, got C={C}"
    assert K % 16 == 0, "K must be a multiple of 16 for the cube operands"

    chunk_num = -(-SEQ // C)  # ceil: the last chunk may be ragged
    R = SEQ % C
    RAGGED = R != 0
    CV = C // VEC_NUM
    # The cube operand dtype.  Route B is the only reason it can differ from
    # the data dtype: its column operand carries exp of the intra-block gate
    # span, which fp16 cannot hold (e^11.09 against a forget gate spanning 70).
    wdt = "bfloat16" if route_b else dtype

    NB = C // BC  # anchor blocks per chunk
    NBV = NB // VEC_NUM  # anchor blocks each vector core builds
    NAB = CV // BC  # diagonal blocks each vector core patches
    GRP = HV // H
    NBLK = B * HV * chunk_num

    @T.prim_func
    def main(
        Kt: T.Tensor([B, SEQ, H, K], dtype),  # type: ignore
        G: T.Tensor([B, SEQ, HV, K], accum_dtype),  # type: ignore
        Beta: T.Tensor([B, SEQ, HV, 1], accum_dtype),  # type: ignore
        Msk: T.Tensor([2 * C, C], accum_dtype),  # type: ignore  strictly lower, i > j
        ws_kr: T.Tensor([NBLK, C, K], wdt),  # type: ignore   row operands
        ws_kf: T.Tensor([NBLK, NB, C, K], wdt),  # type: ignore  column operands
        ws_ls: T.Tensor([NBLK, C, C], accum_dtype),  # type: ignore  strip matmuls, unscaled
        L: T.Tensor([B, SEQ, HV, C], dtype),  # type: ignore
    ):
        with T.Kernel(NBLK, is_npu=True) as (cid, vid):
            bx = cid % chunk_num
            hv = (cid // chunk_num) % HV
            bz = (cid // chunk_num) // HV
            hq = hv // GRP
            base = bx * C
            r0 = vid * CV  # first chunk-local row this core owns

            kr_l1 = T.alloc_L1([BC, K], wdt)
            kf_l1 = T.alloc_L1([C, K], wdt)
            strip_l0 = T.alloc_L0C([BC, C], accum_dtype)

            g_ub = T.alloc_ub([C, K], accum_dtype)
            k_ub = T.alloc_ub([C, K], accum_dtype)
            kh_ub = T.alloc_ub([C, K], dtype)
            fold_ub = T.alloc_ub([C, K], accum_dtype)
            mcol_b = T.alloc_ub([C, K], accum_dtype)

            pb_ub = T.alloc_ub([BC, K], accum_dtype)
            ob_half = T.alloc_ub([BC, K], wdt)

            # fp32, not dtype: the strips land here unscaled and beta is
            # applied to the assembled row below, so this plane has to survive a
            # multiply before it is rounded.  One rounding for the whole row.
            ah_ub = T.alloc_ub([CV, C], accum_dtype)
            ahh_ub = T.alloc_ub([CV, C], dtype)  # the fp16 face of ah_ub, stored once

            # beta for this core's CV rows in one read: a 1-wide GM region lands
            # in a [CV, 8] tile (the DMA pre-fills with pad 0 and writes only
            # column 0), so the row sum *is* the column-0 extract -- and it ends up
            # in a 1-D buffer, which is the only shape the row broadcast accepts.
            # Same construct as wy_fast, already proven there.
            beta8_ub = T.alloc_ub([CV, BETA_PAD], accum_dtype)
            betav_ub = T.alloc_ub([CV], accum_dtype)  # CV*4 >= 32B
            # The row pitch must be exactly C.  Inside T.Parallel(CV, C) the
            # compiler addresses buf[i, j] densely as i*C + j and ignores the
            # buffer's declared pitch.  Borrowing mcol_b[0:CV, :] (pitch K) was
            # tried: the K == C cases passed and every K > C case was wrong.
            betab_ub = T.alloc_ub([CV, C], accum_dtype)
            mcol_ub = T.alloc_ub([C], accum_dtype)
            # The diagonal block's strictly-lower mask, read once as a block.
            # BC * BC fp32 = 1 KB.
            mblk_ub = T.alloc_ub([BC, BC], accum_dtype)
            red_ub = T.alloc_ub([BC], accum_dtype)

            with T.Scope("V"):
                # ---- resident tiles: every key and gate of the chunk --------
                # The explicit token slice is load-bearing; see the module
                # docstring.  On a ragged chunk rows R..C-1 are zeroed first:
                # a garbage gate row exponentiates to +inf and 0 * inf = NaN
                # then lands in a VALID row's reduction.  Zero is also right
                # semantically (g = 0 is alpha = 1, k = 0 contributes nothing),
                # and it makes the pad rows of kr and kf[a] exactly zero, so the
                # cube produces exact zeros for them with no guard of its own.
                if RAGGED and bx == chunk_num - 1:
                    T.tile.fill(kh_ub, 0)
                    T.tile.fill(g_ub, 0.0)

                T.copy(Kt[bz, base : base + C, hq, 0:K], kh_ub)
                T.copy(G[bz, base : base + C, hv, 0:K], g_ub)
                T.copy(kh_ub, k_ub)  # cast dtype -> fp32

                # ---- phase 1a: row operands for this core's NAB blocks ------
                for ab in range(NAB):
                    ar = r0 + ab * BC  # anchor row of this block

                    # g_ub is sliced at a RUNTIME row offset here, which is safe
                    # inside T.Parallel and only inside it: PROBE-A(b) measured
                    # k_ub[a0+jj, d] and found it already lowers to one full
                    # width instruction.  It is the UB->UB *copy* at a runtime
                    # row offset that needed proving separately (PROBE-A(a)).
                    # Materialised broadcast, straight into pb_ub with the
                    # subtraction written the other way round.  The old form
                    # borrowed a separate ob_ub only to have a pure destination,
                    # but pb_ub is already this line's destination -- and the
                    # [BC, K] fp32 tile that frees (8 KB) is exactly what the
                    # [CV, C] beta broadcast target below needs.
                    T.tile.broadcast(pb_ub, g_ub[ar, 0:K], axis=0)
                    for i, d in T.Parallel(BC, K):
                        pb_ub[i, d] = g_ub[ar + i, d] - pb_ub[i, d]
                    # Rows of the block sit at or below the anchor, so this is
                    # already <= 0 and the clamp is a no-op.  It is kept because
                    # it costs one wide instruction and removes the dependence
                    # on G's monotonicity for SAFETY rather than for accuracy.
                    T.tile.min(pb_ub, pb_ub, 0.0)
                    for i, d in T.Parallel(BC, K):
                        pb_ub[i, d] = T.exp(pb_ub[i, d])
                    for i, d in T.Parallel(BC, K):
                        pb_ub[i, d] = pb_ub[i, d] * k_ub[ar + i, d]

                    # No beta here.  It is a per-row scalar and it is applied to
                    # the finished row in phase 3, which costs one read per row
                    # instead of BC reads per block AND gives the strip and the
                    # diagonal a single shared fp32 -> fp16 rounding.

                    T.copy(pb_ub, ob_half)
                    T.copy(ob_half, ws_kr[cid, ar, 0])

                # ---- phase 1b: column operands, anchors split across cores --
                for ai in range(NBV):
                    a = vid + ai * VEC_NUM
                    ar = a * BC

                    # Row ar of the strictly-lower mask IS the indicator of
                    # j < ar.  For ar == 0 it is all zeros, so block 0's matmul
                    # yields exact zeros rather than reading dirty workspace.

                    # Materialised broadcast.  An operand indexed by an INNER
                    # variable alone lowers to one narrow instruction per row in
                    # this dialect: 64 x Sub(128) instead of a single Sub(8192).
                    # The destination is fold_ub itself -- already this line's
                    # target, so no extra UB (chunk_o has only 10,368 B spare and a
                    # fresh [C, K] fp32 tile would need 32,768 B).
                    T.tile.broadcast(fold_ub, g_ub[ar, 0:K], axis=0)
                    for j, d in T.Parallel(C, K):
                        fold_ub[j, d] = fold_ub[j, d] - g_ub[j, d]
                    # Overflow guard as a clamp, free: for j < ar the exponent
                    # is already <= 0, and for j >= ar it is >= 0 where the
                    # clamp gives exactly the 0 the mask below gives.
                    T.tile.min(fold_ub, fold_ub, ROUTE_B_CLAMP if route_b else 0.0)
                    # NO lower clamp here, deliberately.  The official operator
                    # clamps this quantity on both sides and copying that was
                    # measured to be actively harmful: the true exponent
                    # G_i - G_j is <= 0 inside the causal block, so the factored
                    # pair exp(G_i - G_a) * exp(G_a - G_j) is a small factor times
                    # a large one, and a lower clamp turns a factor that should
                    # underflow to exactly zero into ~1e-39.  Multiplied by its
                    # upper-clamped partner that yields O(0.1) where the true
                    # value is zero -- the extreme gate went from 3.0e-3 to
                    # 9.0e-2 end to end, eighteen times over tolerance
                    # (PERF/probes/probe_routeB_e2e.py).  The upper clamp alone is
                    # what prevents the overflow, and 0 * saturated is still 0.
                    for j, d in T.Parallel(C, K):
                        fold_ub[j, d] = T.exp(fold_ub[j, d])
                    for j, d in T.Parallel(C, K):
                        fold_ub[j, d] = fold_ub[j, d] * k_ub[j, d]
                    if route_b:
                        # The clamp above is what used to keep every column
                        # finite.  Raised to ROUTE_B_CLAMP it no longer does, and
                        # the columns past this block -- which the plane mask
                        # discards at the end of phase 3 -- would reach exp(80)
                        # and drive the accumulator toward inf, which the plane
                        # mask then turns into NaN rather than zero.  So the
                        # column mask that route A was able to delete comes back,
                        # at the boundary ar + BC instead of ar: columns inside
                        # this block are exactly the ones route B now needs.
                        T.copy(Msk[ar + BC, 0], mcol_ub)
                        T.tile.broadcast(mcol_b, mcol_ub, axis=1)
                        for j, d in T.Parallel(C, K):
                            fold_ub[j, d] = fold_ub[j, d] * mcol_b[j, d]
                    # The discarding mask.  Unlike the diagonal path there is no
                    # reduction after this -- fold_ub goes straight to the cube
                    # as a [C, K] operand -- so it cannot be moved past one.
                    # Materialise the broadcast instead.
                    # The column mask is gone entirely, saving one [C, K]
                    # broadcast, one [C, K] multiply and one [C] GM read per
                    # anchor block.  It used to do two things:
                    #   (a) stop exp(g_ar - g_j) overflowing for j >= ar, and
                    #   (b) zero those columns.
                    # (a) is already done by the clamp above (before the exp), and
                    # (b) has been taken over by the plane-wide strictly-lower mask
                    # at the end of phase 3: columns j in [ar, ar+BC) are
                    # overwritten wholesale by the diagonal patch, and columns
                    # j >= ar+BC always have j > i, so the plane mask zeroes them.
                    # Those columns therefore only need to be *finite*; their
                    # values do not matter.

                    # kh_ub cannot carry this any more: it holds Kt in the
                    # data dtype and the operand is now in wdt.  A second [C, K]
                    # tile does not fit, so ob_half -- already the operand dtype,
                    # and dead between phase 1a and here -- carries it in NB
                    # slices at compile-time offsets.
                    for t in range(C // BC):
                        T.copy(fold_ub[t * BC : (t + 1) * BC, :], ob_half)
                        T.copy(ob_half, ws_kf[cid, a, t * BC, 0])

                T.set_cross_flag("MTE3", 0)

                # ---- phase 3: patch the diagonal blocks, then store ---------
                # Still the SAME T.Scope("V") as phase 1.  chunk_o runs this
                # exact three-phase protocol out of exactly TWO scope blocks and
                # lets the cross flags carry the ordering.  Splitting it into a
                # second V block compiles and runs and gives strips that are
                # IDENTICALLY ZERO -- the cube executes, but nothing it writes
                # is ever observed.  Measured, not guessed:
                # PERF/probes/probe_kkt_regions.py reported diag=2e-04 (correct)
                # alongside strip=9.6e-01, with row 63 carrying exactly its 15
                # diagonal columns and nothing else.
                T.wait_cross_flag(1)
                T.copy(ws_ls[cid, r0, 0], ah_ub)
                # Msk[a0+rr, a0+jj] is just "rr > jj" -- independent of a0 and of
                # vid -- so it is identically the [BC, BC] top-left corner of Msk,
                # a compile-time constant.  Read it once and index mblk_ub[rr, jj]
                # afterwards: rr comes from range(BC) and is a compile-time
                # constant while jj is the inner variable, so there is neither a
                # broadcast nor any scalar traffic.  This used to be one narrow GM
                # read per row, 32 per block -- chunk_o had already moved to a
                # resident block and kkt had been missed.
                # Under route B the strip matmul already produced these
                # columns correctly, so the whole per-row patch below is dead --
                # which is the entire point: it was 62% of this kernel.
                if not route_b:
                    T.copy(Msk[0, 0], mblk_ub)

                for ab in range(0 if route_b else NAB):
                    a0 = r0 + ab * BC
                    for rr in range(BC):
                        i_loc = a0 + rr  # chunk-local row being finished

                        # No per-row DMA for k, G or beta any more -- all three
                        # are already resident.
                        #
                        # A note here used to claim that "a resident tile indexed
                        # at a runtime row INSIDE T.Parallel is one full-width
                        # instruction (PROBE-A(b) measured exactly this shape)".
                        # That is wrong.  Dumping the generated AscendC
                        # (PERF/probes/probe_kkt_asm.py) shows g_ub[i_loc, d]
                        # compiling to 16 Sub(128) instructions inside an
                        # outer_broadcast_idx loop.  It is g_ub[a0+jj, d], which
                        # uses both indices, that becomes a single Sub(2048) -- most
                        # likely what that probe actually measured.  Read the
                        # generated source to judge instruction shape; do not trust
                        # a comment.
                        # The pad rows need no guard either: k_ub and g_ub
                        # were zero-filled above, so a pad row contributes 0 and
                        # the store below is what is guarded.
                        #
                        # Msk is the one that stays a per-row read.  Msk[0:BC,
                        # 0:BC] does not work -- slicing the last two dims of a
                        # 2-D tensor lands on the wrong address (probe_beta_plane
                        # measured 0.88 max abs diff) -- and a resident [C, C]
                        # plane would cost 16 KB.

                        # Materialised broadcast.  g_ub[i_loc, d] is missing the
                        # jj index, i.e. it broadcasts along jj, and in the
                        # generated source that is 16 Sub(128) instructions inside
                        # an outer_broadcast_idx loop with a PipeBarrier<PIPE_V>
                        # each -- not the single full-width instruction an older
                        # comment claimed.  With the 32-iteration loop around it,
                        # these two sites alone cost 1024 barriers per block.
                        # The destination is pb_ub itself (already this line's
                        # target, so no extra UB), and the row slice feeds the
                        # broadcast directly, saving a UB->UB row copy.
                        T.tile.broadcast(pb_ub, g_ub[i_loc, 0:K], axis=0)
                        for jj, d in T.Parallel(BC, K):
                            pb_ub[jj, d] = pb_ub[jj, d] - g_ub[a0 + jj, d]
                        T.tile.min(pb_ub, pb_ub, 0.0)
                        for jj, d in T.Parallel(BC, K):
                            pb_ub[jj, d] = T.exp(pb_ub[jj, d])
                        # Same again.  This destination borrows the first BC rows
                        # of mcol_b, which is used only by the kf build and is dead
                        # during the diagonal phase; a slice as a broadcast target
                        # was verified exact by probe_slice_dst2.py.
                        T.tile.broadcast(mcol_b[0:BC, :], k_ub[i_loc, 0:K], axis=0)
                        for jj, d in T.Parallel(BC, K):
                            pb_ub[jj, d] = pb_ub[jj, d] * mcol_b[jj, d]
                        for jj, d in T.Parallel(BC, K):
                            pb_ub[jj, d] = pb_ub[jj, d] * k_ub[a0 + jj, d]

                        T.reduce_sum(pb_ub, red_ub, dim=-1)
                        # No mask here any more -- see the plane-wide mask at the
                        # end of phase 3 below.
                        T.copy(red_ub, ah_ub[ab * BC + rr, a0])

                # beta on the finished rows, IN PLACE, then ONE store.
                #
                # Separated from the patch loop above so that every column of a
                # row -- strip and diagonal alike -- is final before it is
                # scaled, and so the scale can be a plain read-modify-write of
                # the plane rather than an extract / scale / cast / store tail.
                #
                # That tail was three DMAs per output row.  With the diagonal
                # arithmetic now a quarter of what it was, per-row DMAs ARE the
                # kernel: ~3550 cycles a row against ~128 cycles of real work.
                # Two constructs make the batched form possible, and both were
                # measured rather than assumed (PERF/probes/probe_batch_store.py):
                #   in-place scale of a resident tile at a RUNTIME row  -- exact
                #   multi-row strided store into [B, SEQ, HV, C]        -- exact,
                #     and it touched no token outside the chunk and no other head
                # A third, broadcasting from a 2-D [C, 1] beta plane, does NOT
                # work, which is why beta is still read one token at a time.
                # Both of the tails below are done once for the whole plane
                # rather than row by row.  Per row they used to be "one 1-element
                # DMA + a PipeBarrier<PIPE_ALL> + a GetValue + a Muls(64)", 32
                # times per block -- PIPE_ALL is the most expensive barrier there
                # is, and the single GetValue / single PIPE_ALL that
                # probe_asm_audit reported were exactly here.
                #
                # First the mask: one Mul(2048) replaces 32 row-wise Mul(16).
                # This is an identity, not an approximation.  L is strictly lower
                # triangular, and across this plane of ah_ub:
                #   * diagonal-block columns -- exactly what the row-wise mask hit;
                #   * strip columns (j < the diagonal block) -- the mask is 1
                #     there, so multiplying changes nothing;
                #   * upper-triangle columns -- always j > i, so the plane mask
                #     zeroes them, which is what the (now removed) kf column mask
                #     used to do.
                # betab_ub is borrowed as the mask plane; the beta broadcast
                # overwrites it a few lines further down.
                T.copy(Msk[r0, 0], betab_ub)
                for i, j in T.Parallel(CV, C):
                    ah_ub[i, j] = ah_ub[i, j] * betab_ub[i, j]

                if RAGGED and bx == chunk_num - 1:
                    T.tile.fill(beta8_ub, 0.0)
                if (not RAGGED) or base + r0 < SEQ:
                    T.copy(Beta[bz, base + r0 : base + r0 + CV, hv, 0:1], beta8_ub, pad_value=0)
                T.reduce_sum(beta8_ub, betav_ub, dim=-1)
                # Then beta, materialised into betab_ub.  A dedicated [CV, C]
                # tile on top of everything else was tried and did not fit: the
                # extra 8 KB took the headroom from 15.2 KB down to 5.9 KB and the
                # kernel died with an aicore exception, because kkt's existing
                # 8192-wide instructions need that space for the compiler's
                # implicit scratch.
                T.tile.broadcast(betab_ub, betav_ub, axis=1)
                for i, j in T.Parallel(CV, C):
                    ah_ub[i, j] = ah_ub[i, j] * betab_ub[i, j]

                T.copy(ah_ub, ahh_ub)
                # One strided write of CV token rows, region [1, CV, 1, C].  The
                # guard is on the whole plane rather than per row: with at least
                # one valid row the token extent is clamped to the valid count,
                # and with none the extent would be <= 0, which Vector and Cube
                # read with OPPOSITE meanings -- the trap the varlen round hit.
                if (not RAGGED) or base + r0 < SEQ:
                    T.copy(ahh_ub, L[bz, base + r0 : base + r0 + CV, hv, 0:C])

            with T.Scope("C"):
                # ---- phase 2: NB strip matmuls -----------------------------
                # Textual position does not express execution order -- the two
                # cross flags do.  This sits after the V block for the same
                # reason chunk_o's does.
                T.wait_cross_flag(0)
                for a in range(NB):
                    T.copy(ws_kr[cid, a * BC, 0], kr_l1)
                    T.copy(ws_kf[cid, a, 0, 0], kf_l1)
                    T.gemm_v0(kr_l1, kf_l1, strip_l0, transpose_B=True, init=True)
                    T.copy(strip_l0, ws_ls[cid, a * BC, 0])
                T.set_cross_flag("FIX", 1)

    return main


@tilelang.jit(out_idx=[-1], workspace_idx=[-4, -3, -2], pass_configs=pass_configs)
def kkt_ker_varlen(B, SEQ, H, HV, K, C, NT_TOTAL, BC=16, dtype="float16", accum_dtype="float", route_b=False):
    """Anchored form, varlen.  Twin of kkt_ker above.

    See kda_chunk_cumsum.cumsum_ker_varlen for why the two are separate
    @tilelang.jit builders rather than one with a flag.

    THE ARITHMETIC IS DELIBERATELY IDENTICAL, OPERATION FOR OPERATION, to the
    fixed-length builder.  That is not tidiness: kda_full's varlen gate is
    bit-identity (``d_o == 0.0``), and it compares a varlen batch against the
    same sequences run one at a time through the FIXED-LENGTH kernel.  Two
    formulations that agree only to fp16 rounding fail that gate -- which is
    exactly what happened when only the fixed-length builder was anchored
    (|dO| = 1.5e-05).  Any future change to one builder's arithmetic has to be
    mirrored here in the same order.

    Only the addressing differs, and in three ways:

      * ``base`` and ``rows`` come from the per-chunk metadata.  Every place the
        fixed-length kernel bounds itself with ``SEQ - base``, this one uses
        ``rows``: under varlen ``SEQ`` is the length of the whole flattened
        batch, so ``SEQ - base`` is no bound at all for a chunk in the middle.
      * the resident tiles are zeroed unconditionally.  ANY chunk can be short
        here, not only the last one.
      * the store stays PER ROW, guarded by ``rows``.  The fixed builder's
        single [CV, C] strided store would run past this sequence's last token
        and into the next sequence -- the framework clamps against SEQ, which
        under varlen does not protect a sequence boundary.  Values are
        unaffected, so bit-identity survives.

    The cross-core flags are new here too, and they cost the old kernel's
    freedom: its docstring justified letting the two vector cores run different
    trip counts with "this kernel has none".  It has two now, so every loop
    below has a COMPILE-TIME trip count and the row count is applied as a guard.
    """
    assert HV % H == 0, "HV must be divisible by H (GVA)"
    assert C % (BC * VEC_NUM) == 0, f"need C % {BC * VEC_NUM} == 0, got C={C}"
    assert K % 16 == 0, "K must be a multiple of 16 for the cube operands"

    # See the fixed-length builder: route B is the only reason the cube operand
    # dtype can differ from the data dtype.
    wdt = "bfloat16" if route_b else dtype

    CV = C // VEC_NUM
    NB = C // BC
    NBV = NB // VEC_NUM
    NAB = CV // BC
    GRP = HV // H
    NBLK = HV * NT_TOTAL

    def tok(idx, t0, rows):
        """Clamp a single-row token index to this chunk's last valid row.

        Single-row reads have region extents [1, 1, 1, *]; find_active_dim_indices
        keeps only the last two *active* dims, so the token axis is folded into
        the base address and is never bounds-checked.  The bound is
        ``t0 + rows - 1`` -- not ``SEQ - 1``, which would happily read another
        sequence's tokens and produce a plausible wrong answer.
        """
        return T.if_then_else(idx < t0 + rows, idx, t0 + rows - 1)

    @T.prim_func
    def main(
        Kt: T.Tensor([B, SEQ, H, K], dtype),  # type: ignore
        G: T.Tensor([B, SEQ, HV, K], accum_dtype),  # type: ignore
        Beta: T.Tensor([B, SEQ, HV, 1], accum_dtype),  # type: ignore
        Msk: T.Tensor([2 * C, C], accum_dtype),  # type: ignore  strictly lower, i > j
        Meta: T.Tensor([NT_TOTAL, _VL.META_COLS], "int32"),  # type: ignore
        ws_kr: T.Tensor([NBLK, C, K], wdt),  # type: ignore   row operands
        ws_kf: T.Tensor([NBLK, NB, C, K], wdt),  # type: ignore  column operands
        ws_ls: T.Tensor([NBLK, C, C], accum_dtype),  # type: ignore  strip matmuls
        L: T.Tensor([B, SEQ, HV, C], dtype),  # type: ignore
    ):
        with T.Kernel(NBLK, is_npu=True) as (cid, vid):
            ic = cid % NT_TOTAL  # flat chunk index over the whole batch
            hv = cid // NT_TOTAL  # value head
            hq = hv // GRP  # qk head behind this hv

            base = Meta[ic, _VL.META_T0]  # first token of the chunk, absolute
            rows = Meta[ic, _VL.META_ROWS]  # 1..C
            r0 = vid * CV  # first chunk-local row this core owns

            kr_l1 = T.alloc_L1([BC, K], wdt)
            kf_l1 = T.alloc_L1([C, K], wdt)
            strip_l0 = T.alloc_L0C([BC, C], accum_dtype)

            g_ub = T.alloc_ub([C, K], accum_dtype)
            k_ub = T.alloc_ub([C, K], accum_dtype)
            kh_ub = T.alloc_ub([C, K], dtype)
            fold_ub = T.alloc_ub([C, K], accum_dtype)
            mcol_b = T.alloc_ub([C, K], accum_dtype)

            pb_ub = T.alloc_ub([BC, K], accum_dtype)
            ob_half = T.alloc_ub([BC, K], wdt)

            ah_ub = T.alloc_ub([CV, C], accum_dtype)
            ahh_ub = T.alloc_ub([CV, C], dtype)
            arow_half = T.alloc_ub([C], dtype)

            mcol_ub = T.alloc_ub([C], accum_dtype)
            # The diagonal block's strictly-lower mask, read once as a block.
            # BC * BC fp32 = 1 KB.
            mblk_ub = T.alloc_ub([BC, BC], accum_dtype)
            red_ub = T.alloc_ub([BC], accum_dtype)
            # beta for this core's CV rows in one read: a 1-wide GM region lands
            # in a [CV, 8] tile (the DMA pre-fills with pad 0 and writes only
            # column 0), so the row sum *is* the column-0 extract -- and it ends up
            # in a 1-D buffer, which is the only shape the row broadcast accepts.
            # Same construct as wy_fast, already proven there.
            beta8_ub = T.alloc_ub([CV, BETA_PAD], accum_dtype)
            betav_ub = T.alloc_ub([CV], accum_dtype)  # CV*4 >= 32B
            # The row pitch must be exactly C.  Inside T.Parallel(CV, C) the
            # compiler addresses buf[i, j] densely as i*C + j and ignores the
            # buffer's declared pitch.  Borrowing mcol_b[0:CV, :] (pitch K) was
            # tried: the K == C cases passed and every K > C case was wrong.
            betab_ub = T.alloc_ub([CV, C], accum_dtype)

            with T.Scope("V"):
                # Unconditional, unlike the fixed-length builder: any chunk here
                # can be short.  A stale gate row exponentiates to +inf and the
                # 0 * inf that follows is NaN in a VALID row's reduction.
                T.tile.fill(kh_ub, 0)
                T.tile.fill(g_ub, 0.0)

                T.copy(Kt[0, base : base + rows, hq, 0:K], kh_ub)
                T.copy(G[0, base : base + rows, hv, 0:K], g_ub)
                T.copy(kh_ub, k_ub)  # cast dtype -> fp32

                # ---- phase 1a: row operands for this core's NAB blocks ------
                for ab in range(NAB):
                    ar = r0 + ab * BC  # anchor row of this block

                    # Materialised broadcast, straight into pb_ub with the
                    # subtraction written the other way round.  The old form
                    # borrowed a separate ob_ub only to have a pure destination,
                    # but pb_ub is already this line's destination -- and the
                    # [BC, K] fp32 tile that frees (8 KB) is exactly what the
                    # [CV, C] beta broadcast target below needs.
                    T.tile.broadcast(pb_ub, g_ub[ar, 0:K], axis=0)
                    for i, d in T.Parallel(BC, K):
                        pb_ub[i, d] = g_ub[ar + i, d] - pb_ub[i, d]
                    T.tile.min(pb_ub, pb_ub, 0.0)
                    for i, d in T.Parallel(BC, K):
                        pb_ub[i, d] = T.exp(pb_ub[i, d])
                    for i, d in T.Parallel(BC, K):
                        pb_ub[i, d] = pb_ub[i, d] * k_ub[ar + i, d]

                    T.copy(pb_ub, ob_half)
                    T.copy(ob_half, ws_kr[cid, ar, 0])

                # ---- phase 1b: column operands, anchors split across cores --
                for ai in range(NBV):
                    a = vid + ai * VEC_NUM
                    ar = a * BC

                    # Materialised broadcast.  An operand indexed by an INNER
                    # variable alone lowers to one narrow instruction per row in
                    # this dialect: 64 x Sub(128) instead of a single Sub(8192).
                    # The destination is fold_ub itself -- already this line's
                    # target, so no extra UB (chunk_o has only 10,368 B spare and a
                    # fresh [C, K] fp32 tile would need 32,768 B).
                    T.tile.broadcast(fold_ub, g_ub[ar, 0:K], axis=0)
                    for j, d in T.Parallel(C, K):
                        fold_ub[j, d] = fold_ub[j, d] - g_ub[j, d]
                    T.tile.min(fold_ub, fold_ub, ROUTE_B_CLAMP if route_b else 0.0)
                    # NO lower clamp here, deliberately.  The official operator
                    # clamps this quantity on both sides and copying that was
                    # measured to be actively harmful: the true exponent
                    # G_i - G_j is <= 0 inside the causal block, so the factored
                    # pair exp(G_i - G_a) * exp(G_a - G_j) is a small factor times
                    # a large one, and a lower clamp turns a factor that should
                    # underflow to exactly zero into ~1e-39.  Multiplied by its
                    # upper-clamped partner that yields O(0.1) where the true
                    # value is zero -- the extreme gate went from 3.0e-3 to
                    # 9.0e-2 end to end, eighteen times over tolerance
                    # (PERF/probes/probe_routeB_e2e.py).  The upper clamp alone is
                    # what prevents the overflow, and 0 * saturated is still 0.
                    for j, d in T.Parallel(C, K):
                        fold_ub[j, d] = T.exp(fold_ub[j, d])
                    for j, d in T.Parallel(C, K):
                        fold_ub[j, d] = fold_ub[j, d] * k_ub[j, d]
                    if route_b:
                        # The clamp above is what used to keep every column
                        # finite.  Raised to ROUTE_B_CLAMP it no longer does, and
                        # the columns past this block -- which the plane mask
                        # discards at the end of phase 3 -- would reach exp(80)
                        # and drive the accumulator toward inf, which the plane
                        # mask then turns into NaN rather than zero.  So the
                        # column mask that route A was able to delete comes back,
                        # at the boundary ar + BC instead of ar: columns inside
                        # this block are exactly the ones route B now needs.
                        T.copy(Msk[ar + BC, 0], mcol_ub)
                        T.tile.broadcast(mcol_b, mcol_ub, axis=1)
                        for j, d in T.Parallel(C, K):
                            fold_ub[j, d] = fold_ub[j, d] * mcol_b[j, d]
                    # The column mask is gone entirely, saving one [C, K]
                    # broadcast, one [C, K] multiply and one [C] GM read per
                    # anchor block.  It used to do two things:
                    #   (a) stop exp(g_ar - g_j) overflowing for j >= ar, and
                    #   (b) zero those columns.
                    # (a) is already done by the clamp above (before the exp), and
                    # (b) has been taken over by the plane-wide strictly-lower mask
                    # at the end of phase 3: columns j in [ar, ar+BC) are
                    # overwritten wholesale by the diagonal patch, and columns
                    # j >= ar+BC always have j > i, so the plane mask zeroes them.
                    # Those columns therefore only need to be *finite*; their
                    # values do not matter.

                    # kh_ub cannot carry this any more: it holds Kt in the
                    # data dtype and the operand is now in wdt.  A second [C, K]
                    # tile does not fit, so ob_half -- already the operand dtype,
                    # and dead between phase 1a and here -- carries it in NB
                    # slices at compile-time offsets.
                    for t in range(C // BC):
                        T.copy(fold_ub[t * BC : (t + 1) * BC, :], ob_half)
                        T.copy(ob_half, ws_kf[cid, a, t * BC, 0])

                T.set_cross_flag("MTE3", 0)

                # ---- phase 3: patch the diagonal blocks ---------------------
                T.wait_cross_flag(1)
                T.copy(ws_ls[cid, r0, 0], ah_ub)
                # Msk[a0+rr, a0+jj] is just "rr > jj" -- independent of a0 and of
                # vid -- so it is identically the [BC, BC] top-left corner of Msk,
                # a compile-time constant.  Read it once and index mblk_ub[rr, jj]
                # afterwards: rr comes from range(BC) and is a compile-time
                # constant while jj is the inner variable, so there is neither a
                # broadcast nor any scalar traffic.  This used to be one narrow GM
                # read per row, 32 per block -- chunk_o had already moved to a
                # resident block and kkt had been missed.
                # Under route B the strip matmul already produced these
                # columns correctly, so the whole per-row patch below is dead --
                # which is the entire point: it was 62% of this kernel.
                if not route_b:
                    T.copy(Msk[0, 0], mblk_ub)

                for ab in range(0 if route_b else NAB):
                    a0 = r0 + ab * BC
                    for rr in range(BC):
                        i_loc = a0 + rr

                        # Materialised broadcast.  g_ub[i_loc, d] is missing the
                        # jj index, i.e. it broadcasts along jj, and in the
                        # generated source that is 16 Sub(128) instructions inside
                        # an outer_broadcast_idx loop with a PipeBarrier<PIPE_V>
                        # each -- not the single full-width instruction an older
                        # comment claimed.  With the 32-iteration loop around it,
                        # these two sites alone cost 1024 barriers per block.
                        # The destination is pb_ub itself (already this line's
                        # target, so no extra UB), and the row slice feeds the
                        # broadcast directly, saving a UB->UB row copy.
                        T.tile.broadcast(pb_ub, g_ub[i_loc, 0:K], axis=0)
                        for jj, d in T.Parallel(BC, K):
                            pb_ub[jj, d] = pb_ub[jj, d] - g_ub[a0 + jj, d]
                        T.tile.min(pb_ub, pb_ub, 0.0)
                        for jj, d in T.Parallel(BC, K):
                            pb_ub[jj, d] = T.exp(pb_ub[jj, d])
                        # Same again.  This destination borrows the first BC rows
                        # of mcol_b, which is used only by the kf build and is dead
                        # during the diagonal phase; a slice as a broadcast target
                        # was verified exact by probe_slice_dst2.py.
                        T.tile.broadcast(mcol_b[0:BC, :], k_ub[i_loc, 0:K], axis=0)
                        for jj, d in T.Parallel(BC, K):
                            pb_ub[jj, d] = pb_ub[jj, d] * mcol_b[jj, d]
                        for jj, d in T.Parallel(BC, K):
                            pb_ub[jj, d] = pb_ub[jj, d] * k_ub[a0 + jj, d]

                        T.reduce_sum(pb_ub, red_ub, dim=-1)
                        # No mask here any more -- see the plane-wide mask at the
                        # end of phase 3 below.
                        T.copy(red_ub, ah_ub[ab * BC + rr, a0])

                # ---- beta, then a PER-ROW store bounded by `rows` -----------
                # Same plane-wide form as the fixed-length builder.  beta can be
                # read as a whole block here without a per-row guard: over-reading
                # into the next sequence's beta is harmless, because those rows
                # have i_loc >= rows and the per-row store loop below never writes
                # them out.  The fill first is what gives the rows the framework
                # clamps away a 0 rather than a stale UB value.
                T.copy(Msk[r0, 0], betab_ub)
                for i, j in T.Parallel(CV, C):
                    ah_ub[i, j] = ah_ub[i, j] * betab_ub[i, j]

                T.tile.fill(beta8_ub, 0.0)
                if r0 < rows:
                    T.copy(Beta[0, base + r0 : base + r0 + CV, hv, 0:1], beta8_ub, pad_value=0)
                T.reduce_sum(beta8_ub, betav_ub, dim=-1)
                # Then beta, materialised into betab_ub.  A dedicated [CV, C]
                # tile on top of everything else was tried and did not fit: the
                # extra 8 KB took the headroom from 15.2 KB down to 5.9 KB and the
                # kernel died with an aicore exception, because kkt's existing
                # 8192-wide instructions need that space for the compiler's
                # implicit scratch.
                T.tile.broadcast(betab_ub, betav_ub, axis=1)
                for i, j in T.Parallel(CV, C):
                    ah_ub[i, j] = ah_ub[i, j] * betab_ub[i, j]

                T.copy(ah_ub, ahh_ub)

                # Per row, not the fixed builder's single strided store: that
                # one would write CV tokens unconditionally and the framework
                # clamps against SEQ -- the END OF THE BATCH -- so a short chunk
                # would land its pad rows on the next sequence's first tokens.
                for ab in range(NAB):
                    for rr in range(BC):
                        i_loc = ab * BC + rr
                        if r0 + i_loc < rows:
                            T.copy(ahh_ub[i_loc, 0:C], arow_half)
                            T.copy(arow_half, L[0, base + r0 + i_loc, hv, 0])

            with T.Scope("C"):
                # ---- phase 2: NB strip matmuls -----------------------------
                T.wait_cross_flag(0)
                for a in range(NB):
                    T.copy(ws_kr[cid, a * BC, 0], kr_l1)
                    T.copy(ws_kf[cid, a, 0, 0], kf_l1)
                    T.gemm_v0(kr_l1, kf_l1, strip_l0, transpose_B=True, init=True)
                    T.copy(strip_l0, ws_ls[cid, a * BC, 0])
                T.set_cross_flag("FIX", 1)

    return main


_MSK_CACHE = {}


def _strict_lower(C, device):
    """The strictly-lower [C, C] indicator, built once per (C, device).

    It is a compile-time constant of the operator, but it used to be rebuilt on
    every call.  Measured on the 2026-08-21 full-pipeline profile, the two host
    ops behind it (OnesLike + Tril) cost 7.50 us per call, and they sit in the gap
    BEFORE this kernel, so `msprof op` on the kernel cannot see them at all -- the
    only way to observe the saving is the inter-kernel gap in a full-pipeline
    profile.

    Safe to share across calls: the kernel takes this as a read-only input (it is
    not in out_idx and not in workspace_idx; the kernel only copies it out).
    Same pattern as _IDT_CACHE in kda_solve_tril.py.
    """
    key = (C, str(device))
    m = _MSK_CACHE.get(key)
    if m is None:
        # 2C rows, not C.  Route B masks columns at the boundary ar + BC, and for
        # the last anchor block that boundary is row C itself.  Rows C..2C-1 are
        # all ones, which is what "every column is below the boundary" means.
        m = torch.tril(torch.ones((2 * C, C), device=device, dtype=torch.float), diagonal=-1)
        _MSK_CACHE[key] = m
    return m


def _pick_route(route_b):
    """Resolve the route.  A plain parameter, never a look at the data.

    An earlier version decided this by reducing over G and comparing the widest
    intra-block gate span against the clamp.  That is wrong in a way no test here
    happened to catch, and it is worth recording why rather than just deleting it:

      * The reduction is over the WHOLE input, so slicing the input can change
        the answer.  kda_full asserts that a sequence split at a chunk boundary
        and run in two halves is BIT-IDENTICAL to running it whole; if the whole
        exceeds the threshold and a half does not, the two take different routes
        and that assertion breaks.  It passes today only because no test case
        straddles the threshold.
      * Reading the result back to decide forces a host-device sync every call,
        which stalls the pipeline and prevents graph capture.
      * It is all-or-nothing: one outlier block anywhere makes the whole call
        take the slow route, so latency is bimodal in a way the caller cannot
        predict or control.

    The official operator has no such guard -- `safeGate` is an attribute the
    caller passes (chunk_kda_fwd_tiling.cpp:182) -- and that is the right shape.
    The caller knows its gate distribution; the operator does not, and paying a
    reduction over G every call to guess at it buys nothing.

    The default is off.  Route B is an approximation -- it saturates a gate that
    spans more than the clamp -- and an extreme gate saturates hard enough to put
    L at 6.88e-01 against this stage's 5e-3 tolerance.  A caller whose gate is
    ordinary gets 2.3x on this stage by asking for it; a caller who does not ask
    keeps exactly the numerics this operator shipped with.

    KDA_ROUTE_B=0 / =1 overrides the argument, for A/B measurement.
    """
    env = os.environ.get("KDA_ROUTE_B")
    if env is not None:
        return env not in ("0", "", "false", "False")
    return route_b


def chunk_scaled_dot_kkt(k, G, beta, C=64, BC=16, cu_seqlens=None, route_b=False):
    """Host wrapper.  Returns L [B, SEQ, HV, C] in the dtype of ``k``.

    Arguments are the frozen external tensors:
        k     [B, SEQ, H,  K]  fp16 / bf16
        G     [B, SEQ, HV, K]  fp32, chunk-local cumsum of the log gate (stage 1)
        beta  [B, SEQ, HV] or an already padded [B, SEQ, HV, 8] fp32

    Host does no transposing, reshaping or state movement -- the kernel indexes
    [B, SEQ, HV, *] directly.  The only host work is the beta 32B padding, the
    constant C x C mask and a dtype lookup.

    With ``cu_seqlens`` the inputs are a flattened varlen batch (B == 1).  L
    keeps its layout -- one C-wide row per token, in flattened order -- and only
    the block-to-chunk mapping changes.
    """
    B, SEQ, H, K = k.shape
    HV = G.shape[2]
    assert G.shape == (B, SEQ, HV, K), f"G must be [B, SEQ, HV, K], got {tuple(G.shape)}"
    assert HV % H == 0, "HV must be divisible by H (GVA)"
    assert C % 2 == 0 and C % 16 == 0, "C must be even and 32B-aligned in dtype"
    assert K % 16 == 0, "K must be 32B-aligned in dtype"

    # Route B needs the kernel's block partition to match what the clamp was
    # reasoned about, which holds while chunks start at multiples of C.  Under
    # varlen a chunk starts at a sequence boundary instead, so the anchor row of
    # a block is not where this file assumes; that path keeps route A until it is
    # worked through.
    route_b = _pick_route(route_b) and cu_seqlens is None

    elem = torch.finfo(k.dtype).bits // 8
    need = _ub_bytes(C, K, BC, elem)
    assert need <= UB_LIMIT, f"UB budget {need} > {UB_LIMIT} for C={C} K={K}; lower C"

    # SEQ == 0 slips past the assert above (0 % C == 0) and would launch a
    # zero-block grid over an unwritten output.  A zero-length sequence is legal
    # input; there are no token pairs, so L is empty along the token axis.
    #
    # Under varlen this fires only for a wholly empty batch.  A single empty
    # sequence inside a non-empty one contributes zero chunks, so no block is
    # ever created for it and nothing special is needed.
    if SEQ == 0:
        return torch.empty((B, 0, HV, C), device=k.device, dtype=k.dtype)

    # beta: one 32B slot per token, only lane 0 carries a value.  Padding the
    # last axis to 8 rather than the head axis to HV + 8 is deliberate: the
    # latter starts head hv at byte 4 * hv, which is not a multiple of 32.
    assert beta.shape == (B, SEQ, HV), f"beta must be the contract shape [B, SEQ, HV], got {tuple(beta.shape)}"
    beta_p = beta.float().unsqueeze(-1)  # a view: no allocation, no copy

    msk = _strict_lower(C, k.device)

    # Only the cube operands change dtype under route B, never k or L: handing
    # L back in bf16 would silently convert the five stages downstream.
    dt = {torch.float16: "float16", torch.bfloat16: "bfloat16"}[k.dtype]
    if cu_seqlens is None:
        return kkt_ker(B, SEQ, H, HV, K, C, BC=BC, dtype=dt, route_b=route_b)(k, G.float(), beta_p, msk)

    bounds = _VL.varlen_bounds(cu_seqlens, q=k, g=G, beta=beta)
    meta = _VL.chunk_meta(bounds, C, k.device)
    nt_total = meta.shape[0]
    # Every token row of L is written by exactly one block, or the caller gets
    # dirty memory back: the output buffer is torch.empty, not zeros.
    assert nt_total > 0, "a non-empty batch must produce at least one chunk"
    assert int(meta[:, _VL.META_ROWS].sum()) == SEQ, "chunk metadata does not cover every token exactly once"
    return kkt_ker_varlen(B, SEQ, H, HV, K, C, nt_total, BC=BC, dtype=dt, route_b=route_b)(k, G.float(), beta_p, msk, meta)


# Alias matching the stage name used by the GDN pipeline.
kkt = chunk_scaled_dot_kkt


# ----------------------------------------------------------------- test
def _relerr(got, ref):
    got = got.float().cpu()
    ref = ref.float().cpu()
    return (got - ref).abs().max().item() / max(ref.abs().max().item(), 1e-9)


def _case(B, SEQ, H, HV, K, V, C, gate, dtype):
    # Inputs and golden are built on CPU in fp32: stage_tensors on the NPU
    # dispatches einsum to matmul with reduced-precision accumulation and drifts
    # two identical fp32 references by ~3e-4, which would blur the comparison.
    # The tensors handed to the kernel are bit-identical copies of these.
    q, k, v, g, beta, _ = kda_chunk_ref.make_inputs(B, SEQ, H, HV, K, V, device="cpu", dtype=dtype, gate=gate)
    st = kda_chunk_ref.stage_tensors(q, k, v, g, beta, C=C)
    G, ref = st["G"], st["L"]

    got = chunk_scaled_dot_kkt(k.npu(), G.npu(), beta.npu(), C=C)

    err = _relerr(got, ref)
    finite = bool(torch.isfinite(got.float()).all())
    tol = 5e-3 if dtype == torch.float16 else 3e-2  # bf16 has 8 mantissa bits
    ok = finite and err < tol
    tag = "bf16" if dtype == torch.bfloat16 else "fp16"
    print(
        f"  B{B} T{SEQ:<4d} H{H} HV{HV} K{K:<3d} C{C:<2d} {tag} {gate:8s} "
        f"relerr={err:.2e} finite={'Y' if finite else 'N'}  "
        f"{'ok' if ok else 'FAIL'}"
    )
    return ok


def _vcase(seqlens, H, HV, K, V, C, gate, dtype, note=""):
    """One varlen batch against the stage-2 golden, over the WHOLE flat token axis.

    Whole-axis, not per sequence, on purpose.  The failure this stage is most
    exposed to is a single-row store that walks past its own eos: the region is
    [1, 1, 1, C], so find_active_dim_indices keeps only the HV and C axes and the
    token index is folded into the base address with no bounds check at all.
    Such a store lands a fully-formed, finite, plausible L row on the NEXT
    sequence.  Sequence i's own rows stay correct, so a per-sequence comparison
    would pass while the batch is quietly wrong.
    """
    q, k, v, g, beta, _, cu = kda_chunk_ref.make_varlen_inputs(seqlens, H, HV, K, V, device="cpu", dtype=dtype, gate=gate)
    st = kda_chunk_ref.stage_tensors(q, k, v, g, beta, C=C, cu_seqlens=cu)
    G, ref = st["G"], st["L"]

    got = chunk_scaled_dot_kkt(k.npu(), G.npu(), beta.npu(), C=C, cu_seqlens=cu.npu())

    err = _relerr(got, ref)
    finite = bool(torch.isfinite(got.float()).all())
    tol = 5e-3 if dtype == torch.float16 else 3e-2
    ok = finite and err < tol and tuple(got.shape) == tuple(ref.shape)
    tag = "fp16" if dtype == torch.float16 else "bf16"
    print(f"  {str(seqlens):24s} HV{HV} K{K:<4d} C{C:<2d} {tag} {gate:8s} relerr={err:.2e}  {'ok' if ok else 'FAIL'}  {note}")
    return ok


def main():
    tilelang.disable_cache()
    torch.manual_seed(0)

    ok = True
    print("== C=32 / C=64, HV == H, two gate settings ==")
    for gate in ("normal", "forget"):
        ok &= _case(2, 256, 4, 4, 64, 64, 32, gate, torch.float16)
        ok &= _case(2, 256, 4, 4, 64, 64, 64, gate, torch.float16)

    print("== GVA: HV == 2H ==")
    for gate in ("normal", "forget"):
        ok &= _case(2, 128, 2, 4, 64, 64, 32, gate, torch.float16)
        ok &= _case(2, 128, 2, 4, 64, 64, 64, gate, torch.float16)

    print("== single head, and the gate extremes ==")
    ok &= _case(1, 64, 1, 1, 64, 64, 64, "normal", torch.float16)
    for gate in ("keep", "extreme"):
        ok &= _case(2, 128, 2, 2, 64, 64, 64, gate, torch.float16)

    print("== ragged tail (SEQ % C != 0) ==")
    ok &= _case(2, 70, 1, 2, 64, 64, 64, "normal", torch.float16)  # 70 = 64 + 6, tail rows in core 0 only
    ok &= _case(1, 33, 1, 1, 64, 64, 32, "forget", torch.float16)  # 33 = 32 + 1
    ok &= _case(1, 65, 1, 1, 128, 128, 64, "forget", torch.float16)  # one valid tail row; core 1 gets zero rows
    ok &= _case(2, 100, 2, 4, 64, 64, 32, "extreme", torch.float16)  # GVA + extreme gate on the tail

    print("== K3 spec: K = V = 128 ==")
    ok &= _case(1, 256, 2, 2, 128, 128, 64, "forget", torch.float16)
    ok &= _case(1, 256, 1, 2, 128, 128, 64, "normal", torch.float16)  # + GVA
    ok &= _case(1, 128, 1, 1, 128, 128, 32, "forget", torch.float16)

    print("== bf16 dtype passthrough ==")
    for gate in ("normal", "forget"):
        ok &= _case(2, 128, 2, 4, 64, 64, 64, gate, torch.bfloat16)

    print("== varlen (cu_seqlens) ==")
    ok &= _vcase([64, 64, 64], 1, 2, 64, 64, 64, "normal", torch.float16, "equal, chunk-aligned")
    ok &= _vcase([70, 33, 129], 1, 2, 64, 64, 64, "normal", torch.float16, "every sequence ragged -- interior tails")
    ok &= _vcase([70, 0, 129], 1, 2, 64, 64, 64, "forget", torch.float16, "empty sequence in the middle")
    ok &= _vcase([0, 70], 1, 2, 64, 64, 64, "normal", torch.float16, "empty sequence first")
    ok &= _vcase([70, 0], 1, 2, 64, 64, 64, "normal", torch.float16, "empty sequence last")
    ok &= _vcase([1, 200], 1, 2, 64, 64, 64, "forget", torch.float16, "one token, then a long sequence")
    ok &= _vcase([33, 33], 1, 1, 64, 64, 32, "forget", torch.float16, "33 = 32 + 1 twice")
    ok &= _vcase([65, 65], 1, 1, 128, 128, 64, "forget", torch.float16, "K3 dim, core 1 gets zero rows")
    ok &= _vcase([100, 28], 2, 4, 64, 64, 32, "extreme", torch.float16, "GVA + extreme gate, C = 32")
    ok &= _vcase([5], 1, 1, 64, 64, 64, "normal", torch.float16, "N = 1, shorter than a chunk")
    ok &= _vcase([70, 33], 2, 4, 64, 64, 64, "forget", torch.bfloat16, "bf16 passthrough + GVA")

    if ok:
        print("Kernel Output Match!")
    else:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
