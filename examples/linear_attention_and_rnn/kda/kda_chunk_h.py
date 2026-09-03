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
kda_chunk_ref.ref_wy_fast) and scale rides on q, which chunk_o consumes.

Ragged tail
-----------
``SEQ % C != 0`` is supported, and this is the stage where it costs real thought
rather than a clamp.

Everything on the token axis is a multi-row copy and is clamped for free: the
K / G / U loads shrink to the R valid rows, the ``Vt`` store never writes the pad
rows, and the cube's ``Vt`` read is both clamped and zero-initialised by
``copy_gm_to_l1`` -- which is exactly what makes ``kg^T V'`` sum over the R real
tokens with no change to the math.

Two things are NOT free:

  * **The chunk decay index.** ``G_C`` must be read at the last token that
    exists, ``min(t0 + C, SEQ) - 1``.  The old ``t0 + C - 1`` is out of bounds on
    a ragged chunk *and* semantically wrong, and nothing catches it: single-row
    reads put the token axis on a unit-extent dim, which
    ``find_active_dim_indices`` never bounds-checks.  A wrong index here does not
    show up as a small error -- the final state comes out scaled by a spurious
    e^Gamma, which under the `forget` gate is orders of magnitude.
  * **Stale UB feeding the cube.** ``g_ub`` is exponentiated and ``k_ub`` is
    published to ``ws_kg`` as a full [C, K] operand, so the tail rows are
    zero-filled before the clamped loads overwrite the valid part.

★ Nothing here base-clamps a tile index.  Unlike stage 6, clamping a *base* in
this stage would pull the previous chunk's real V' rows into ``kv_l0`` and
corrupt the state carry.  Only the single-row G_C index is clamped, and only to
the last valid token.

★ This kernel has four cross-core flag ids and is the easiest in the pipeline to
deadlock.  Tail handling fills buffers; it never skips a ``set_cross_flag`` or a
``wait_cross_flag``.

Known limitations (first pass)
------------------------------
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

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import kda_chunk_ref  # noqa: E402
import kda_varlen as _VL  # noqa: E402

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
    # ceil, not floor: the last chunk may be ragged.  SEQ is a Python int at
    # trace time, so this stays a compile-time constant, and the `states`
    # tensor simply gains one chunk -- which stage 6 must agree on.
    N_CHUNK = -(-SEQ // C)
    R = SEQ % C  # 0 when aligned; else the valid row count of the last chunk
    RAGGED = R != 0
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
                    # ★ The one genuine off-by-one of the whole tail-block change.
                    #
                    # G_C is the chunk's cumulative gate, and it has to be read
                    # at the LAST TOKEN THAT EXISTS.  On a ragged chunk row
                    # t0 + C - 1 is (a) past SEQ, and (b) not the right token
                    # even if it were in range.  Neither is caught for us: these
                    # are single-row reads, so the region extents are
                    # [1, 1, 1, *], find_active_dim_indices keeps only the last
                    # two *active* dims, and the token axis is folded into the
                    # base address without a bounds check.
                    #
                    # Getting this wrong does not produce a tolerance failure --
                    # the final state comes out scaled by a spurious e^Gamma,
                    # which under the `forget` gate is orders of magnitude.
                    #
                    # Runtime address, compile-time extent: exactly the pattern
                    # the varlen kernels in this repo already rely on.
                    glast_row = T.if_then_else(t0 + C <= SEQ, t0 + C - 1, SEQ - 1)

                    if RAGGED and ci == N_CHUNK - 1:
                        # The two tile loads below are clamped, so their tail
                        # rows keep stale UB.  g_ub feeds exp() and k_ub feeds
                        # the cube through ws_kg as a full [C, K] operand, so a
                        # garbage row becomes inf and then NaN.  Zero is also
                        # the right filler: k = 0 contributes nothing to
                        # kg^T V', and g = 0 leaves the coefficient at e^{G_C}.
                        T.tile.fill(k_half, 0)
                        T.tile.fill(g_ub, 0.0)

                    T.copy(Kt[bz, tv : tv + CV, hq, :], k_half)
                    T.copy(k_half, k_ub)
                    T.copy(G[bz, tv : tv + CV, hv, :], g_ub)
                    T.copy(G[bz, glast_row, hv, 0], glast_ub)  # G_C, all K
                    T.copy(G[bz, glast_row, hv, ko], gexp_ub)  # G_C, my rows
                    # Materialised broadcast.  glast_ub indexed by an INNER
                    # variable alone lowers to one narrow instruction per row in
                    # this dialect; spreading it into a full tile first turns the
                    # subtraction into a single wide tile-to-tile instruction.
                    # The destination is coeff_ub itself -- already this line's
                    # target, so no extra UB.
                    T.tile.broadcast(coeff_ub, glast_ub, axis=0)
                    for a, b in T.Parallel(CV, K):
                        coeff_ub[a, b] = coeff_ub[a, b] - g_ub[a, b]  # <= 0
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
                    #
                    # A note here used to claim that an outer-variable broadcast
                    # has to be done in place, because writing to a second buffer
                    # fails the UB alignment check.  That predates L2 and has been
                    # disproved: probe_chunkh_bcast.py runs T.tile.broadcast at
                    # chunk_h's real shape ([64] -> [64, 32]) and it compiles,
                    # passes the alignment check and is bit-identical.
                    #
                    # Slicing gexp out of glast_ub (it is a sub-range of the same
                    # row) to save this second per-row GM read was also tried:
                    # chunk_h went 269 -> 292 us, because a 1-D read at a run-time
                    # offset drops off the vector path.  Keep the separate read.
                    for i in T.Parallel(KV):
                        gexp_ub[i] = T.exp(gexp_ub[i])
                    # Materialised broadcast.  gexp_ub is indexed by the OUTER
                    # variable, the worst form in this dialect: the generated
                    # AscendC is a 64-iteration for loop with a GetValue and a
                    # Muls(32) each.  The compiler names that loop variable
                    # outer_broadcast_idx -- it knows this is a broadcast, it just
                    # cannot do the substitution for you.  Spread out it is a
                    # single Mul(2048); an isolated micro-benchmark measured
                    # 1866.80 -> 122.46 us, bit-identical
                    # (PERF/probes/probe_chunkh_bcast.py).
                    # kv_ub is borrowed as the target: it is not written until
                    # after wait_cross_flag(2) below, so it is dead here and this
                    # costs no extra UB.
                    T.tile.broadcast(kv_ub, gexp_ub, axis=1)
                    for i, j in T.Parallel(KV, BV):
                        s_ub[i, j] = s_ub[i, j] * kv_ub[i, j]

                    T.wait_cross_flag(2)
                    T.copy(ws_kv[cid, ko, 0], kv_ub)
                    for i, j in T.Parallel(KV, BV):
                        s_ub[i, j] = s_ub[i, j] + kv_ub[i, j]

                T.copy(s_ub, SF[bz, hv, ko, vo])

    return main


@tilelang.jit(out_idx=[-3, -2, -1], workspace_idx=[-7, -6, -5, -4], pass_configs=pass_configs)
def chunk_h_ker_varlen(B, SEQ, H, HV, K, V, C, BV, N_SEQ, NT_TOTAL, dtype="float16", accum_dtype="float"):
    """Per-chunk entry states, V' = U - W S, and the final state.  Varlen twin.

    This is the only chunk-SERIAL stage of the six, and that decides its varlen
    shape.  The other five keep a per-chunk grid and look their chunk up in a
    flat table; this one cannot, because S is carried from one chunk to the
    next.  So its grid stays per (sequence, value head, value block) and the
    chunk loop stays inside, with a RUN-TIME trip count taken from that
    sequence's own length.  It is the shape the shipped GDN varlen example uses
    (examples_experiment/chunk_gated_delta_rule/chunk_gated_delta_rule.py:98).

    Four things are specific to this stage:

      * **The empty sequence costs nothing.**  T_i == 0 gives NT_i == 0, the
        loop body never runs, and the pre-loop `T.copy(S0[i_n, ...], s_ub)` and
        post-loop `T.copy(s_ub, SF[i_n, ...])` deliver SF[i] == S0[i] on their
        own.  There is no special case anywhere, which is the point.
      * **NT_i is computed by the same textual expression in both scopes and
        never depends on vid.**  All eight cross-core flags live inside these
        two loops, and the AIC/AIV sync is a hardware counter: if the Cube ran a
        different number of iterations from the Vector cores, the kernel would
        deadlock rather than produce a wrong answer.
      * **G_C is read at the chunk's last VALID token.**  This was already the
        one genuine off-by-one of the ragged-tail round; varlen makes the old
        expression silently wrong rather than loudly out of range, because
        `t0 + C <= SEQ` is TRUE for every interior chunk when SEQ is the whole
        flattened batch.  See the comment at the site.
      * **`states` is NT_TOTAL-major.**  Sequence n's chunk i lives at slot
        `chunk_off[n] + i`, the same addressing FLA's prepare_chunk_offsets
        produces.  Stage 6 has to agree, and both wrappers assert the shape.
    """

    BV_NUM = V // BV
    VEC_NUM = 2  # 910B: one Cube core, two Vector cores
    CV = C // VEC_NUM  # token rows per vector core
    KV = K // VEC_NUM  # state rows per vector core
    GRP = HV // H  # GVA: GRP value heads per qk head
    NBLK = N_SEQ * HV * BV_NUM  # one block per (sequence, value head, value block)

    @T.prim_func
    def main(
        Kt: T.Tensor([B, SEQ, H, K], dtype),  # type: ignore
        W: T.Tensor([B, SEQ, HV, K], dtype),  # type: ignore
        U: T.Tensor([B, SEQ, HV, V], dtype),  # type: ignore
        G: T.Tensor([B, SEQ, HV, K], accum_dtype),  # type: ignore
        S0: T.Tensor([N_SEQ, HV, K, V], accum_dtype),  # type: ignore  one per SEQUENCE, not per batch
        # SeqMeta goes HERE, before the workspaces, and the position is not
        # free.  The decorator addresses the workspaces and the outputs with
        # NEGATIVE indices (workspace_idx=[-7, -6, -5, -4], out_idx=[-3, -2, -1]).
        # Appending a tensor after them would shift all seven, so the framework
        # would allocate real outputs as scratch and hand back uninitialised
        # memory -- a kernel that compiles, runs, and returns garbage.
        SeqMeta: T.Tensor([N_SEQ, _VL.SEQ_META_COLS], "int32"),  # type: ignore  (bos, eos, chunk_off)
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
        states: T.Tensor([B, HV, NT_TOTAL, K, V], dtype),  # type: ignore  chunk axis is the whole batch
        Vt: T.Tensor([B, SEQ, HV, V], dtype),  # type: ignore
        SF: T.Tensor([N_SEQ, HV, K, V], accum_dtype),  # type: ignore  one per SEQUENCE
    ):
        with T.Kernel(NBLK, is_npu=True) as (cid, vid):
            bx = cid % BV_NUM  # value block
            hv = (cid // BV_NUM) % HV  # value head
            i_n = cid // (BV_NUM * HV)  # sequence
            hq = hv // GRP  # qk head feeding this value head
            vo = bx * BV  # value offset of this block
            ko = vid * KV  # state rows of this vector core
            co = vid * CV  # token rows of this vector core

            # This sequence's span and where its chunks live in the flat run.
            bos = SeqMeta[i_n, _VL.SEQ_BOS]
            eos = SeqMeta[i_n, _VL.SEQ_EOS]
            c0 = SeqMeta[i_n, _VL.SEQ_CHUNK_OFF]
            # ★ Both scopes below derive their trip count from THIS line's
            # expression, verbatim, and it does not mention vid.  Eight
            # cross-core flags live inside those loops and the AIC/AIV sync is a
            # hardware counter, so a disagreement deadlocks rather than
            # miscomputes.  Zero for an empty sequence, which is exactly right.
            NT_i = T.ceildiv(eos - bos, C)

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
                for ci in T.serial(NT_i):
                    t0 = bos + ci * C
                    # 1..C, never 0: a chunk only exists for a non-empty
                    # sequence.  That matters because copy_gm_to_l1 reads a row
                    # count of 0 as "no tail, use the full tile" (common.h:60).
                    rows = T.min(C, eos - t0)

                    T.wait_cross_flag(3)
                    T.copy(ws_st[cid, 0, 0], s_l1)
                    # Bounded.  Left at C this reads the next sequence's W rows
                    # into L1, and copy_gm_to_l1 only zero-inits when the row
                    # count differs from the tile height -- which mid-tensor it
                    # would not.
                    T.copy(W[0, t0 : t0 + rows, hv, :], w_l1)
                    T.gemm_v0(w_l1, s_l1, wS_l0, init=True)  # W S
                    T.copy(wS_l0, ws_wS[cid, 0, 0])
                    T.set_cross_flag("FIX", 0)

                    T.wait_cross_flag(1)
                    T.copy(ws_kg[cid, 0, 0], k_l1)
                    # Bounded for the same reason, and here the rows past eos
                    # are worse than stale: Vt is a torch.empty output whose
                    # owning block may not have run yet, so they can be any
                    # bit pattern at all, inf and NaN included.
                    T.copy(Vt[0, t0 : t0 + rows, hv, vo : vo + BV], v_l1)
                    T.gemm_v0(k_l1, v_l1, kv_l0, transpose_A=True, init=True)
                    T.copy(kv_l0, ws_kv[cid, 0, 0])  # kg^T V'
                    T.set_cross_flag("FIX", 2)

            # -------------------------------------------------------- Vector
            with T.Scope("V"):
                T.copy(S0[i_n, hv, ko, vo], s_ub)  # fp32 state, stays resident
                # Explicit MTE2 -> MTE3 barrier, and it is NOT redundant.
                #
                # An empty sequence gives NT_i == 0, so the chunk loop below
                # never runs and this load is followed immediately by the
                # s_ub -> SF store at the bottom.  AUTO_SYNC places the barrier
                # for that dependency INSIDE the loop -- with a run-time trip
                # count it cannot know the loop may execute zero times -- so on
                # an empty sequence the store races the load and SF comes back
                # part S0, part stale UB.  It is not a clean failure either:
                # measured, the result tracked S0's magnitude while differing
                # element by element, so a magnitude check would have passed it.
                T.set_flag("mte2", "mte3", 0)
                T.wait_flag("mte2", "mte3", 0)

                for ci in T.serial(NT_i):
                    t0 = bos + ci * C
                    rows = T.min(C, eos - t0)  # same expression as the Cube scope
                    tv = t0 + co  # this core's first token

                    # This core's share of the chunk's valid rows.  CAN be zero
                    # -- any sequence whose last chunk holds at most C/2 rows
                    # leaves the second core with nothing -- so it guards
                    # branches and is never passed as a copy extent.
                    left = T.if_then_else(rows > co, rows - co, 0)
                    vrows = T.if_then_else(left < CV, left, CV)

                    # entry state of this chunk: one rounded tile, published
                    # twice -- to chunk_o as `states`, and to the Cube as the
                    # right operand of W S.
                    T.copy(s_ub, s_half)
                    # NT_TOTAL-major: this sequence's chunk ci sits at slot
                    # c0 + ci.  The slot index is folded into the base address
                    # with no bounds check, so a wrong chunk_off writes into
                    # another sequence's states silently -- which is why the
                    # host asserts the offsets before the launch.
                    T.copy(s_half, states[0, hv, c0 + ci, ko, vo])
                    T.copy(s_half, ws_st[cid, ko, 0])
                    T.set_cross_flag("MTE3", 3)

                    # kg = K . e^{G_C - G}.  GDN broadcasts one scalar per row
                    # here; KDA needs the whole [CV, K] tile.
                    # ★ The one genuine off-by-one of the whole tail-block change.
                    #
                    # G_C is the chunk's cumulative gate, and it has to be read
                    # at the LAST TOKEN THAT EXISTS.  On a ragged chunk row
                    # t0 + C - 1 is (a) past SEQ, and (b) not the right token
                    # even if it were in range.  Neither is caught for us: these
                    # are single-row reads, so the region extents are
                    # [1, 1, 1, *], find_active_dim_indices keeps only the last
                    # two *active* dims, and the token axis is folded into the
                    # base address without a bounds check.
                    #
                    # Getting this wrong does not produce a tolerance failure --
                    # the final state comes out scaled by a spurious e^Gamma,
                    # which under the `forget` gate is orders of magnitude.
                    #
                    # Runtime address, compile-time extent: exactly the pattern
                    # the varlen kernels in this repo already rely on.
                    # ★ Under varlen the old expression is not merely
                    # out of range, it is SILENTLY WRONG.  It read
                    #     t0 + C <= SEQ  ?  t0 + C - 1  :  SEQ - 1
                    # and with SEQ meaning the whole flattened batch the first
                    # branch is taken for every interior ragged chunk, so G_C
                    # would be read at a token belonging to the NEXT sequence.
                    # The failure is not a tolerance miss: final_state comes out
                    # scaled by a spurious e^Gamma, orders of magnitude under
                    # the `forget` gate.
                    glast_row = t0 + rows - 1

                    # Unconditional.  Raggedness is a run-time property under
                    # varlen, and when vrows == 0 the loads below are skipped
                    # entirely, so these fills are the only thing defining the
                    # tiles.  Zero is also the right filler: k = 0 contributes
                    # nothing to kg^T V', and g = 0 leaves the coefficient at
                    # e^{G_C}.
                    T.tile.fill(k_half, 0)
                    T.tile.fill(g_ub, 0.0)

                    if vrows > 0:
                        T.copy(Kt[0, tv : tv + vrows, hq, :], k_half)
                        T.copy(G[0, tv : tv + vrows, hv, :], g_ub)
                    T.copy(k_half, k_ub)
                    T.copy(G[0, glast_row, hv, 0], glast_ub)  # G_C, all K
                    T.copy(G[0, glast_row, hv, ko], gexp_ub)  # G_C, my rows
                    # Materialised broadcast.  glast_ub indexed by an INNER
                    # variable alone lowers to one narrow instruction per row in
                    # this dialect; spreading it into a full tile first turns the
                    # subtraction into a single wide tile-to-tile instruction.
                    # The destination is coeff_ub itself -- already this line's
                    # target, so no extra UB.
                    T.tile.broadcast(coeff_ub, glast_ub, axis=0)
                    for a, b in T.Parallel(CV, K):
                        coeff_ub[a, b] = coeff_ub[a, b] - g_ub[a, b]  # <= 0
                    for a, b in T.Parallel(CV, K):
                        coeff_ub[a, b] = T.exp(coeff_ub[a, b])
                    for a, b in T.Parallel(CV, K):
                        k_ub[a, b] = k_ub[a, b] * coeff_ub[a, b]
                    T.copy(k_ub, k_half)

                    # V' = U - W S
                    T.tile.fill(u_half, 0)
                    if vrows > 0:
                        T.copy(U[0, tv : tv + vrows, hv, vo : vo + BV], u_half)
                    T.copy(u_half, u_ub)
                    T.wait_cross_flag(0)
                    T.copy(ws_wS[cid, co, 0], ws_ub)
                    for a, b in T.Parallel(CV, BV):
                        u_ub[a, b] = u_ub[a, b] - ws_ub[a, b]
                    T.copy(u_ub, u_half)
                    # Bounded: the rows past this core's share belong to the
                    # next sequence and are written by that sequence's own
                    # block.  Guarded as well -- a zero row count is safe on the
                    # UB -> GM path (measured, PROBES/probe_varlen5.log) but it
                    # is an undocumented blockCount, and the guard costs nothing.
                    if vrows > 0:
                        T.copy(u_half, Vt[0, tv : tv + vrows, hv, vo : vo + BV])
                    # Full CV rows, deliberately NOT bounded and NOT guarded:
                    # the Cube reads ws_kg as a complete [C, K] operand, so the
                    # rows this core does not own have to be written zeros
                    # rather than left dirty.  The flag that follows is outside
                    # every branch for the same reason -- one core skipping it
                    # deadlocks the Cube forever.
                    T.copy(k_half, ws_kg[cid, co, 0])
                    T.set_cross_flag("MTE3", 1)

                    # per-channel state decay: row d of S carries e^{G_C[d]}.
                    #
                    # A note here used to claim that an outer-variable broadcast
                    # has to be done in place, because writing to a second buffer
                    # fails the UB alignment check.  That predates L2 and has been
                    # disproved: probe_chunkh_bcast.py runs T.tile.broadcast at
                    # chunk_h's real shape ([64] -> [64, 32]) and it compiles,
                    # passes the alignment check and is bit-identical.
                    #
                    # Slicing gexp out of glast_ub (it is a sub-range of the same
                    # row) to save this second per-row GM read was also tried:
                    # chunk_h went 269 -> 292 us, because a 1-D read at a run-time
                    # offset drops off the vector path.  Keep the separate read.
                    for i in T.Parallel(KV):
                        gexp_ub[i] = T.exp(gexp_ub[i])
                    # Materialised broadcast.  gexp_ub is indexed by the OUTER
                    # variable, the worst form in this dialect: the generated
                    # AscendC is a 64-iteration for loop with a GetValue and a
                    # Muls(32) each.  The compiler names that loop variable
                    # outer_broadcast_idx -- it knows this is a broadcast, it just
                    # cannot do the substitution for you.  Spread out it is a
                    # single Mul(2048); an isolated micro-benchmark measured
                    # 1866.80 -> 122.46 us, bit-identical
                    # (PERF/probes/probe_chunkh_bcast.py).
                    # kv_ub is borrowed as the target: it is not written until
                    # after wait_cross_flag(2) below, so it is dead here and this
                    # costs no extra UB.
                    T.tile.broadcast(kv_ub, gexp_ub, axis=1)
                    for i, j in T.Parallel(KV, BV):
                        s_ub[i, j] = s_ub[i, j] * kv_ub[i, j]

                    T.wait_cross_flag(2)
                    T.copy(ws_kv[cid, ko, 0], kv_ub)
                    for i, j in T.Parallel(KV, BV):
                        s_ub[i, j] = s_ub[i, j] + kv_ub[i, j]

                # Reached with NT_i == 0 too, and that is the whole of the
                # empty-sequence contract: s_ub still holds S0, so SF == S0.
                T.copy(s_ub, SF[i_n, hv, ko, vo])

    return main


# ----------------------------------------------------------------- host side
_DTYPE_STR = {torch.float16: "float16", torch.bfloat16: "bfloat16"}


_ZERO_S0_CACHE = {}


def _zero_state(n_lead, HV, K, V, device):
    """The all-zero entry state used when the caller passes no initial_state.

    Rebuilt on every call before this cache; the 2026-08-21 full-pipeline profile
    attributes 7.30 us per call to the ZerosLike sitting in the gap before this
    kernel.

    Safe to share across calls, and this one deserves the argument spelled out
    because it is the least obviously safe of the three: S0 is a read-only kernel
    input -- it is not in out_idx=[-3,-2,-1] nor in workspace_idx=[-7,-6,-5,-4],
    and the only thing the kernel does with it is `T.copy(S0[...], s_ub)` at the
    top of the vector scope.  Nothing writes it.  The SEQ == 0 early return hands
    back `s0.clone()`, not s0 itself, so a caller relaying the final state cannot
    reach this buffer either.
    """
    key = (n_lead, HV, K, V, str(device))
    z = _ZERO_S0_CACHE.get(key)
    if z is None:
        z = torch.zeros((n_lead, HV, K, V), device=device, dtype=torch.float32)
        _ZERO_S0_CACHE[key] = z
    return z


def chunk_h(kt, w, u, g_cumsum, C=64, BV=None, initial_state=None, cu_seqlens=None):
    """Host wrapper.  Semantics match kda_chunk_ref.ref_chunk_h.

    Only zero-filling the absent initial state, a dtype table lookup and the
    block-size choice happen here.  No transpose, no reshape, no staging copy:
    the kernel indexes the external [B, SEQ, HV, *] layout directly, which is
    the task's gate against hiding kernel cost on the host.

    With ``cu_seqlens`` the inputs are a flattened varlen batch (B == 1) and two
    shapes change, because they are the two that are not indexed by token:

        initial_state / final_state   [N, HV, K, V]     one per SEQUENCE
        states                        [1, HV, NT_TOTAL, K, V]

    where ``NT_TOTAL = sum_i ceil(T_i / C)`` and sequence n's chunk i lives at
    slot ``chunk_off[n] + i``.  Stage 6 reads ``states`` with the same
    addressing and asserts the same shape.
    """
    B, SEQ, H, K = kt.shape
    HV, V = u.shape[2], u.shape[-1]

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
    # unconditionally; the same choice as the L0 kernel.  Under varlen the
    # leading axis is the SEQUENCE COUNT, not the batch.
    n_lead = B if cu_seqlens is None else (cu_seqlens.numel() - 1)
    s0 = initial_state.float() if initial_state is not None else _zero_state(n_lead, HV, K, V, kt.device)
    assert s0.shape == (n_lead, HV, K, V) and s0.is_contiguous(), "S0 layout"

    # SEQ == 0 slips past the assert above (0 % C == 0) and would launch a
    # zero-block grid over unwritten outputs.  A zero-length sequence is legal
    # input.  This is the one stage where the answer is not simply "empty": no
    # token is consumed, so the state must pass through untouched and the final
    # state IS the initial state.  Cloned rather than aliased so a caller that
    # relays it into the next segment cannot mutate its own input.
    #
    # Under varlen this fires only when the WHOLE batch is empty.  A single
    # empty sequence inside a non-empty one is handled entirely in the kernel:
    # its NT_i is 0, the chunk loop never runs, and SF falls out equal to S0.
    if SEQ == 0:
        return (
            # [B, HV, 0, K, V] is right for both paths: fixed-length has no
            # chunks, and varlen's NT_TOTAL is 0 when every sequence is empty.
            torch.empty((B, HV, 0, K, V), device=kt.device, dtype=kt.dtype),
            torch.empty((B, 0, HV, V), device=kt.device, dtype=kt.dtype),
            s0.clone(),
        )

    if cu_seqlens is None:
        ker = chunk_h_ker(B, SEQ, H, HV, K, V, C, BV, dtype=_DTYPE_STR[kt.dtype])
        states, vt, sf = ker(kt, w, u, g_cumsum, s0)
        return states, vt, sf

    bounds = _VL.varlen_bounds(cu_seqlens, q=kt, v=u, g=g_cumsum, initial_state=s0)
    offsets, nt_total = _VL.chunk_layout(bounds, C)
    smeta = _VL.seq_meta(bounds, C, kt.device)
    n_seq = len(bounds)
    # The slot index into `states` is folded into a base address with no bounds
    # check, so a chunk_off built with a different C, or with the ceil applied
    # to the cumulative sum instead of per sequence, would write into another
    # sequence's states silently.  Assert the layout instead of trusting it.
    assert nt_total > 0, "a non-empty batch must produce at least one chunk"
    assert offsets[0] == 0, "the first sequence must start at chunk slot 0"
    assert all(offsets[i] <= offsets[i + 1] for i in range(n_seq - 1)), "chunk offsets must be non-decreasing"
    ker = chunk_h_ker_varlen(B, SEQ, H, HV, K, V, C, BV, n_seq, nt_total, dtype=_DTYPE_STR[kt.dtype])
    states, vt, sf = ker(kt, w, u, g_cumsum, s0, smeta)
    assert states.shape == (B, HV, nt_total, K, V), f"states must be [1, HV, NT_TOTAL, K, V], got {tuple(states.shape)}"
    return states, vt, sf


# ---------------------------------------------------------------------- test
def _relerr(x, r):
    r = r.float()
    # An all-empty varlen batch legitimately produces zero-element states and
    # Vt; .max() on those raises rather than returning 0.
    if x.numel() == 0 and r.numel() == 0:
        return 0.0
    return (x.float() - r).abs().max().item() / max(r.abs().max().item(), 1e-9)


def _case(B, SEQ, H, HV, K, V, C, gate, with_state, dtype):
    q, k, v, g, beta, s0 = kda_chunk_ref.make_inputs(B, SEQ, H, HV, K, V, dtype=dtype, gate=gate, with_state=with_state)

    gold = kda_chunk_ref.stage_tensors(q, k, v, g, beta, C=C, initial_state=s0)
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


def _vcase(seqlens, H, HV, K, V, C, gate, with_state, dtype, note=""):
    """One varlen batch against the stage-5 golden, over the WHOLE flat token axis.

    Three separate things are checked, because this stage can fail in three
    unrelated ways:

      * `Vt` over every token -- catches a store that walks past its own eos
        into the next sequence, which stays finite and plausible.
      * `states` over every slot -- catches a wrong chunk_off, which silently
        writes into another sequence's states.
      * `SF` per sequence -- catches the G_C off-by-one, whose signature is a
        final state scaled by a spurious e^Gamma rather than a small drift.  An
        empty sequence additionally has to pass its state through bit for bit.
    """
    q, k, v, g, beta, s0, cu = kda_chunk_ref.make_varlen_inputs(seqlens, H, HV, K, V, dtype=dtype, gate=gate, with_state=with_state)
    gold = kda_chunk_ref.stage_tensors(
        q.cpu(), k.cpu(), v.cpu(), g.cpu(), beta.cpu(), C=C, cu_seqlens=cu.cpu(), initial_state=None if s0 is None else s0.cpu()
    )
    W = gold["W"].contiguous().to(dtype).npu()
    U = gold["U"].contiguous().to(dtype).npu()
    G = gold["G"].contiguous().npu()

    states, vt, sf = chunk_h(k, W, U, G, C=C, initial_state=s0, cu_seqlens=cu)

    eS = _relerr(states.cpu(), gold["states"])
    eV = _relerr(vt.cpu(), gold["Vt"])
    eF = _relerr(sf.cpu(), gold["SF"])
    tol = 5e-3 if dtype == torch.float16 else 3e-2
    finite = bool(torch.isfinite(sf.float()).all()) and bool(torch.isfinite(vt.float()).all())
    shape_ok = tuple(states.shape) == tuple(gold["states"].shape) and tuple(sf.shape) == tuple(gold["SF"].shape)

    # An empty sequence consumes no token, so its final state must be its
    # initial state -- bit for bit, not merely within tolerance.
    passthru = True
    if with_state:
        for i, n in enumerate(seqlens):
            if n == 0:
                passthru &= bool(torch.equal(sf[i].cpu(), s0[i].float().cpu()))

    ok = finite and shape_ok and passthru and eS < tol and eV < tol and eF < tol
    tag = "fp16" if dtype == torch.float16 else "bf16"
    print(
        f"  {str(seqlens):22s} HV{HV} C{C:<2d} {tag} {gate:8s} S0={'Y' if with_state else 'N'} "
        f"st={eS:.2e} Vt={eV:.2e} SF={eF:.2e} pass={'Y' if passthru else 'N'}  {'ok' if ok else 'FAIL'}  {note}"
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

    print("== ragged tail (SEQ % C != 0) ==")
    for B, SEQ, H, HV, K, V, C, gate, ws in [
        (2, 70, 1, 2, 64, 64, 64, "normal", False),  # R=6
        (1, 33, 1, 1, 64, 64, 32, "forget", False),  # R=1, core 1 empty
        (1, 65, 1, 1, 128, 128, 64, "forget", True),  # K3 dim, R=1, with initial state
        (2, 100, 2, 4, 64, 64, 32, "extreme", True),  # GVA + extreme gate + state
        (1, 96, 1, 1, 64, 64, 64, "normal", False),  # R=32 == CV, exact core boundary
    ]:
        ok &= _case(B, SEQ, H, HV, K, V, C, gate, ws, torch.float16)

    print("== dtype passthrough (bf16) ==")
    for B, SEQ, H, HV, K, V, C, gate, ws in [
        (2, 128, 2, 4, 64, 64, 64, "normal", True),
        (1, 128, 1, 1, 128, 128, 64, "forget", True),
    ]:
        ok &= _case(B, SEQ, H, HV, K, V, C, gate, ws, torch.bfloat16)

    print("== varlen (cu_seqlens) ==")
    ok &= _vcase([64, 64, 64], 1, 2, 64, 64, 64, "normal", False, torch.float16, "equal, chunk-aligned")
    ok &= _vcase([70, 33, 129], 1, 2, 64, 64, 64, "normal", False, torch.float16, "every sequence ragged")
    ok &= _vcase([70, 33, 129], 1, 2, 64, 64, 64, "forget", True, torch.float16, "ragged + per-sequence S0")
    ok &= _vcase([70, 0, 129], 1, 2, 64, 64, 64, "forget", True, torch.float16, "empty in the middle, S0 passthrough")
    ok &= _vcase([0, 70], 1, 2, 64, 64, 64, "normal", True, torch.float16, "empty first")
    ok &= _vcase([70, 0], 1, 2, 64, 64, 64, "normal", True, torch.float16, "empty last")
    ok &= _vcase([0, 0], 1, 2, 64, 64, 64, "normal", True, torch.float16, "every sequence empty")
    ok &= _vcase([1, 200], 1, 2, 64, 64, 64, "forget", False, torch.float16, "one token -- core 1 gets vrows = 0")
    ok &= _vcase([20, 20], 1, 2, 64, 64, 64, "normal", False, torch.float16, "both shorter than C/2")
    ok &= _vcase([65, 65], 1, 1, 128, 128, 64, "forget", True, torch.float16, "K3 dim, one valid tail row each")
    ok &= _vcase([100, 28], 2, 4, 64, 64, 32, "extreme", False, torch.float16, "GVA + extreme gate, C = 32")
    ok &= _vcase([70, 33], 2, 4, 64, 64, 64, "forget", True, torch.bfloat16, "bf16 + GVA")

    if ok:
        print("Kernel Output Match!")
    else:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
