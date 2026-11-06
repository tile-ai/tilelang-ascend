"""Sparse MLA topk ReduceSum — head reduction of topk attention mass (DSA, Ascend).

For each packed token row s, kv group g and topk-selected position
idx = Indices[s_row, g, j], computes score = exp((Q @ KV_main + Q_tail @ KV_tail)
* sm_scale - Lse) with the causal mask built in (idx <= s and idx != -1, else
score = 0), then reduces over heads into ReduceSum[s, g, r, j]. The REPLICATE_H
merge and the final normalization belong to the caller — the delivery boundary
of this operator ends at ReduceSum.

Programming model: Expert mode.
  - Explicit buffers: T.alloc_L1 / T.alloc_L0C on the Cube side, T.alloc_ub on
    the Vector side, plus GM workspaces (ws_kv / ws_acc) for cross-core staging.
  - Explicit dual streams: T.Scope("C") (single fused gemm_v0) and T.Scope("V")
    (gather / mask / exp / reduce).
  - Manual synchronization: T.set_cross_flag / T.wait_cross_flag (mode-2 quorum).
  - No pass_configs (fully manual).

Cross-flag protocol (strict 1:1 pairing on the flat block stream
n = task*NI + t; every V-side set/wait stays inside a valid block's program
range and the C-side loop bound is exactly ntask, so no flag can dangle):
  F0 = 0..DEPTH-1        V->C, set on MTE3: the ws_kv[cid] slot block has been
                         fully gathered (mode-2: C proceeds only after both
                         AIVs have written their halves).
  F1 = DEPTH..2*DEPTH-1  C->V, set on FIX: the ws_acc[cid] slot block scores
                         are ready (L0C -> GM has landed).

Implementation notes:
  1. Row-addressing identity: s_row = bos + s_i == s_row_i (guaranteed by the
     packed layout). All Q/Lse/Indices/ReduceSum row indexing therefore uses
     s_row_i directly (pure pid arithmetic); GM scalar reads (TokenIndices /
     Offsets) are kept only where the V stream needs them: the gather base bos
     and the causal-mask scalar s_i.
  2. Gather addressing safety: KV row ids are clamped to [0, seq_len_kv - 1],
     so -1 / out-of-range indices always read legal memory; their contribution
     is zeroed by the causal mask, matching the reference semantics.
  3. ws_kv uses the merged D + D_tail layout [grid_size, BI, DDT] (one DMA
     instead of two); the Cube stream column-slices it into the two L1 tiles.
  4. Mask arithmetic: m1 = clamp((s_i + 1) - idx, 0, 1) = 1{idx <= s} and
     m2 = clamp(idx + 1, 0, 1) = 1{idx >= 0}, mask = m1 * m2, broadcast to
     [H, HALF]. `T.tile.sub` does not accept a scalar left operand, so
     (s_i + 1) - idx is computed as mul(-1) + add(scalar); s_i is a GM scalar
     combined into a PrimExpr scalar through T.float32(s_i + 1).

Generic path: a single code path serves any legal topk_indices (values in
[-1, seq_len_kv), any distribution — row-independent / partially shared /
all-identical / non-contiguous / shuffled; no index-content assumptions).

Supported domain (compile-time asserts):
  Fixed dtype combination Q/KV=bfloat16, Lse/ReduceSum=float32,
  Indices/Offsets/TokenIndices=int32;
  dim / tail_dim are powers of two; topk % block_I == 0; block_I is even
  and >= 32;
  head_kv = heads // kv_group satisfies (16 <= head_kv <= 64 and a power of
  two) or (head_kv % 64 == 0); padded_H != head_kv requires kv_group == 1.
  Total token counts and batch are runtime-dynamic (T.symbolic dims).
"""

import tilelang
from tilelang import language as T

__all__ = [
    "tl_sparse_mla_topk_reducesum",
    "sparse_mla_topk_reducesum",
    "prepare_token_indices",
]


# ============================================================================
# Kernel builder
# ============================================================================
def _reducesum_generic_body(
    heads,
    dim,
    tail_dim,
    topk,
    kv_group,
    sm_scale,
    block_I,
    grid_size,
):
    """Build the generic-path kernel body (the single builder).

    One builder, no guards, no variant parameters: a single code path traced
    for every legal index distribution.
    """
    # ---- Compile-time asserts (supported domain) ----
    assert dim == tilelang.math.next_power_of_2(dim), f"dim must be a power of 2, dim={dim}"
    assert tail_dim == tilelang.math.next_power_of_2(tail_dim), f"tail_dim must be a power of 2, tail_dim={tail_dim}"
    assert topk % block_I == 0, "otherwise will load some index=0 thus causing wrong kv to be loaded"
    assert block_I % 2 == 0 and block_I >= 32, f"block_I must be even and >= 32 (AIV half-split), got {block_I}"
    assert kv_group >= 1 and heads % kv_group == 0, f"heads={heads} must be divisible by kv_group={kv_group}"
    head_kv = heads // kv_group
    assert (16 <= head_kv <= 64 and head_kv == tilelang.math.next_power_of_2(head_kv)) or head_kv % 64 == 0, (
        f"head_kv={head_kv} out of supported domain: need (16<=head_kv<=64 and power of 2) or head_kv % 64 == 0"
    )
    if sm_scale is None:
        sm_scale = (1.0 / (dim + tail_dim)) ** 0.5

    # ---- Symbolic dynamic dims (one compile covers any batch / seq lens) ----
    seq_len = T.symbolic("seq_len")  # total packed query rows
    seq_len_kv = T.symbolic("seq_len_kv")  # total packed kv rows
    batch_plus_one = T.symbolic("batch_plus_one")  # batch + 1 (Offsets dim)

    dtype = "bfloat16"
    indices_dtype = "int32"
    accum_dtype = "float32"

    # ---- Derived compile-time constants ----
    padded_H = max(tilelang.math.next_power_of_2(head_kv), 16)
    if padded_H != head_kv:
        assert kv_group == 1, (
            "H padding path only supports kv_group == 1 (inherited from the CUDA original: "
            "otherwise Q/Output head slices need explicit masks)"
        )
    if head_kv > 64:
        assert head_kv % 64 == 0, "head_kv should be a multiple of 64"
        REPLICATE_H = head_kv // 64
    else:
        REPLICATE_H = 1
    H_per_block = padded_H if REPLICATE_H == 1 else 64

    BI = block_I
    HALF = BI // 2  # AIV column half (the topk direction is split across 2 AIVs)
    NI = topk // BI  # topk blocks per token (compile-time constant)
    DDT = dim + tail_dim
    # Head shard offset: 0 when REPLICATE_H == 1, else 64 heads per shard
    R_HEAD_OFFSET = 0 if REPLICATE_H == 1 else 64

    # Cross-flag ids (DEPTH slots in a ping-pong ring: slot s -> F0 id s % DEPTH
    # and F1 id DEPTH + s % DEPTH). DEPTH bounds both the V-side prefetch depth
    # and the slot reuse period; flag ids are a scarce hardware resource, so
    # only 2 * DEPTH ids are allocated.
    DEPTH = 4  # number of ws_kv / ws_acc slots (= V-side prefetch depth + 1)
    F0_READY_BASE = 0  # V->C: ws_kv slot gathered (set on MTE3), ids 0..DEPTH-1
    F1_ACC_BASE = DEPTH  # C->V: ws_acc slot scores ready (set on FIX), ids DEPTH..2*DEPTH-1
    # Cross-task look-ahead distance (compile-time constant). The V stream
    # issues the gather of block gn = n + LEAD from body(n), so the Cube side
    # always has DEPTH-bounded work queued behind the flag handshake instead of
    # locking step with the Vector side block by block.
    #   NI >= 2: LEAD <= NI, so the look-ahead reaches at most into task k+1
    #     and two scalar sets per body suffice;
    #   NI == 1 (BI == topk, one block per task): LEAD=2, and the look-ahead
    #     task offset becomes OFFSET_N = (NI - 1 + LEAD) // NI (1 for NI >= 2,
    #     2 for NI == 1), i.e. body(k) holds scalars of tasks k and k+OFFSET_N.
    LEAD = min(DEPTH - 1, NI) if NI > 1 else min(DEPTH - 1, 2)
    OFFSET_N = (NI - 1 + LEAD) // NI  # look-ahead task offset (1 when NI >= 2)
    # kv_gather_ub staging depth: side = gn % SIDES rotates the staging slots so
    # the row reads of one body (MTE2 writes) overlap with the bulk store-out of
    # an earlier body (MTE3 reads). Side-slot reuse safety has two tiers:
    #   SIDES >= LEAD+1 (UB-comfortable configs): the previous reader of the
    #     slot, bulk(gn-SIDES), has drained — post(gn-SIDES) already passed
    #     waitF1(gn-SIDES), F1(m) implies the Cube side consumed F0(m), and
    #     that implies both AIVs' bulk(m) stores had landed;
    #   SIDES == LEAD (UB-tight configs, e.g. BI=128 @ DDT=576): the chain
    #     above breaks (the bulk store and its own post segment share a body),
    #     and safety relies on the natural drain window instead: bulk(gn-SIDES)
    #     is issued from body(gn-SIDES-LEAD) while the row read reusing the
    #     slot is issued from body(gn-LEAD) — two bodies apart, each containing
    #     a post-segment F1 scalar stall, which leaves the MTE3 store ample
    #     time to drain (>= 4x margin between the drain window and the
    #     worst-case store latency). Never pair non-adjacent set/wait flags to
    #     implement this protection: set/wait flags are fire-before-arm (a miss
    #     leaves the wait hanging forever).
    _kv_gather_budget = 170 * 1024  # UB 192KB minus other buffers and alignment slack
    SIDES = (LEAD + 1) if (LEAD + 1) * (block_I // 2) * (dim + tail_dim) * 2 <= _kv_gather_budget else 2
    # Scalar-window invariants:
    #  (i)  the look-ahead gather block gn = k*NI+t+LEAD must belong to task k
    #       or task k+OFFSET_N — the body holds exactly those two scalar sets;
    #  (ii) prologue first-use side slots are pairwise distinct (LEAD <= SIDES
    #       makes side = s % SIDES == s for s < LEAD, so no protection needed).
    assert all(((t + LEAD) // NI) in (0, OFFSET_N) for t in range(NI)), (
        f"#18 scalar-window invariant violated: LEAD={LEAD} NI={NI} OFFSET_N={OFFSET_N}"
    )
    assert LEAD <= SIDES, f"#18 prologue side-collision: LEAD={LEAD} > SIDES={SIDES}"
    # indices_ub slot count (cross-body index prefetch): block m's indices are
    # prefetched into slot m % IDXSLOTS one body ahead of use; the last V-side
    # reader of that slot lives in body(m - IDXSLOTS), and IDXSLOTS >= LEAD + 2
    # keeps it strictly earlier than the issuing body, so an adjacent v->mte2
    # pair at the issuing point is sufficient protection.
    IDXSLOTS = LEAD + 2

    q_shape = [seq_len, heads, DDT]
    kv_shape = [seq_len_kv, kv_group, DDT]
    indices_shape = [seq_len, kv_group, topk]
    lse_shape = [seq_len, heads]
    offsets_shape = [batch_plus_one]
    token_indices_shape = [seq_len, 2]
    reducesum_shape = [seq_len, kv_group, REPLICATE_H, topk]

    @T.prim_func
    def main(
        Q: T.Tensor(q_shape, dtype),  # type: ignore  # [total_S, heads, DDT] bf16
        KV: T.Tensor(kv_shape, dtype),  # type: ignore  # [total_S_kv, G, DDT] bf16
        Indices: T.Tensor(indices_shape, indices_dtype),  # type: ignore  # [total_S, G, topk] int32
        Lse: T.Tensor(lse_shape, accum_dtype),  # type: ignore  # [total_S, heads] fp32
        Offsets: T.Tensor(offsets_shape, indices_dtype),  # type: ignore  # [B+1] int32
        TokenIndices: T.Tensor(token_indices_shape, indices_dtype),  # type: ignore  # [total_S, 2] int32
        ReduceSum: T.Tensor(reducesum_shape, accum_dtype),  # type: ignore  # out
        ws_kv: T.Tensor([grid_size, DEPTH * BI, DDT], dtype),  # V->C gather staging (DEPTH slots per core)
        ws_acc: T.Tensor([grid_size, DEPTH * H_per_block, BI], accum_dtype),  # C->V score staging (DEPTH slots)
    ):
        with T.Kernel(grid_size, is_npu=True) as (cid, vid):
            # ---- Cube side: L1 / L0C (gemm_v0 manages L0A/L0B internally) ----
            # Merged D+D_tail layout (DDT): a single gemm_v0 with K=DDT replaces
            # a two-segment GEMM split; the fp32 accumulation order over K is
            # unchanged and one gemm call's fixed overhead is saved per block.
            q_l1 = T.alloc_L1([H_per_block, DDT], dtype)
            kv_l1 = T.alloc_L1([BI, DDT], dtype)
            acc_s_l0c = T.alloc_L0C([H_per_block, BI], accum_dtype)

            # ---- Vector side: UB buffers ----
            # indices ring buffer (slot dim is addressable through the Var):
            # IDXSLOTS = LEAD + 2 slots support the cross-body prefetch (see
            # the IDXSLOTS derivation above).
            indices_ub = T.alloc_ub([IDXSLOTS, BI], indices_dtype)
            # Bulk gather staging buffer: [SIDES, HALF, DDT], side = gn % SIDES
            # rotates; row reads (MTE2 writes) overlap with the previous bulk
            # store-out (MTE3 reads) across bodies (side-reuse safety: see the
            # SIDES two-tier argument above).
            kv_gather_ub = T.alloc_ub([SIDES, HALF, DDT], dtype)
            # Pre-clamped KV row addresses (int32): one vectorized add + clamp
            # per gather segment, then the row loop is down to a single scalar
            # read + DataCopyPad per row.
            rows_ub = T.alloc_ub([BI], indices_dtype)
            lse_ub = T.alloc_ub([H_per_block], accum_dtype)
            lse2d_ub = T.alloc_ub([H_per_block, HALF], accum_dtype)  # broadcast once per task
            idx_f_ub = T.alloc_ub([HALF], accum_dtype)
            t_ub = T.alloc_ub([HALF], accum_dtype)
            m1_ub = T.alloc_ub([HALF], accum_dtype)
            m2_ub = T.alloc_ub([HALF], accum_dtype)
            mask_ub = T.alloc_ub([HALF], accum_dtype)
            mask2d_ub = T.alloc_ub([H_per_block, HALF], accum_dtype)
            acc_ub = T.alloc_ub([H_per_block, HALF], accum_dtype)
            rs_ub = T.alloc_ub([HALF], accum_dtype)

            # ---- Persistent grid over the flat block stream n = task*NI + t ----
            # Tasks are assigned round-robin: ntask = #{k : k*grid + cid <
            # total}. Both scopes advance the same flat block order, so every
            # F0/F1 set has a matching wait (count nblk = ntask*NI, strict 1:1);
            # the loop bound ntask itself acts as the existence guard, so no
            # flag can dangle. The V-stream body(k, t) = [look-ahead gather of
            # block k*NI+t+LEAD (may reach into task k+1); post of block
            # k*NI+t]: the next task's first gathers are issued mid-task here,
            # which removes the task-boundary drain/refill bubble.
            total_tasks = seq_len * REPLICATE_H * kv_group
            ntask = T.max(0, T.ceildiv(total_tasks - cid, grid_size))  # valid tasks on this core
            nblk = ntask * NI  # total flat blocks on this core

            with T.Scope("C"):
                for k in T.serial(ntask):
                    pid = k * grid_size + cid
                    b_s_i = pid % (seq_len * REPLICATE_H)
                    g_i = pid // (seq_len * REPLICATE_H)
                    s_row_i = b_s_i // REPLICATE_H  # packed global row (== bos + s_i)
                    r_i = b_s_i % REPLICATE_H  # REPLICATE shard id
                    H0 = g_i * padded_H + r_i * R_HEAD_OFFSET
                    H1 = H0 + H_per_block
                    # Q load (once per token; a contiguous [H, DDT] row slice,
                    # no gather). Cross-task q_l1/kv_l1 reuse is protected by
                    # the tail MTE1->MTE2 flag drained inside gemm_v0; the
                    # ws_kv GM->L1 copy of this iteration is protected by the
                    # head MTE2->MTE1 flag of gemm_v0.
                    T.copy(Q[s_row_i, H0:H1, :], q_l1)
                    for i in T.serial(NI):
                        n = k * NI + i  # flat block id (same domain as the V-side n/gn)
                        slot = n % DEPTH  # ping-pong slot id
                        T.wait_cross_flag(F0_READY_BASE + slot)
                        # ws_kv slot rows -> L1 in one contiguous copy (BI x DDT)
                        T.copy(ws_kv[cid, slot * BI : (slot + 1) * BI, :], kv_l1)
                        # Single GEMM with K=DDT (fp32 accumulation into L0C;
                        # the K order matches the original two-GEMM split).
                        T.gemm_v0(q_l1, kv_l1, acc_s_l0c, transpose_B=True, init=True)
                        # L0C -> GM (FIX); the tail M->FIX flag inside gemm_v0
                        # already guarantees the Mmad results have landed.
                        T.copy(acc_s_l0c, ws_acc[cid, slot * H_per_block : (slot + 1) * H_per_block, :])
                        T.set_cross_flag("FIX", F1_ACC_BASE + slot)

            with T.Scope("V"):
                # GM scalar reads live only in the V stream (the C stream never
                # consumes b_i/s_i/bos — it uses the row-addressing identity).
                # The look-ahead gather block gn = k*NI+t+LEAD belongs to task k
                # or task k+OFFSET_N (compile-time invariant), so the two scalar
                # sets held by each body (its own + the safely clamped
                # k+OFFSET_N preload) cover every look-ahead access.
                if ntask > 0:
                    # ---- preamble: task-0 scalars + lse2d(0) (ntask > 0 => in bounds) ----
                    b_s_0 = cid % (seq_len * REPLICATE_H)
                    g_0 = cid // (seq_len * REPLICATE_H)
                    s_row_0 = b_s_0 // REPLICATE_H
                    r_0 = b_s_0 % REPLICATE_H
                    H0_0 = g_0 * padded_H + r_0 * R_HEAD_OFFSET
                    b_0 = TokenIndices[s_row_0, 0]
                    s_i_0 = TokenIndices[s_row_0, 1]  # noqa: F841 - kept for parity with the task-0 scalar set
                    bos_0 = Offsets[b_0]
                    # lse_ub may only be rewritten after the broadcast has read it out
                    T.set_flag("v", "mte2", 0)
                    T.wait_flag("v", "mte2", 0)
                    T.copy(Lse[s_row_0, H0_0 : H0_0 + H_per_block], lse_ub)
                    T.set_flag("mte2", "v", 0)
                    T.wait_flag("mte2", "v", 0)
                    T.tile.broadcast(lse2d_ub, lse_ub, axis=1)  # once per task

                    # ---- prologue: gather blocks 0..LEAD-1 (compile-time unrolled) ----
                    # NI >= 2: LEAD <= NI, so every block belongs to task 0
                    # (preamble scalars, no guard). NI == 1 (LEAD = 2): block
                    # s >= 1 belongs to task s, which may not exist — all GM
                    # scalar reads go through clamped legal coordinates (the
                    # values are garbage but memory-safe and never consumed),
                    # and the bulk store / F0 are skipped inside the `s < nblk`
                    # existence guard, so phantom blocks produce no F0 and the
                    # nblk 1:1 pairing holds. side = s % SIDES is collision-free
                    # on first use (LEAD <= SIDES assert). Branch bodies are
                    # self-contained (TVMScript IfFrame scoping: bindings do
                    # not escape the frame).
                    for s in range(LEAD):
                        tau_s = s // NI  # owning task of block s (compile-time constant)
                        nslot = s % DEPTH
                        side = s % SIDES
                        islot = s % IDXSLOTS  # s < LEAD < IDXSLOTS => islot == s
                        if tau_s == 0:
                            # indices_ub[islot] rewrite guard: earlier Cast (V) reads drained
                            T.set_flag("v", "mte2", 0)
                            T.wait_flag("v", "mte2", 0)
                            # rows_ub rewrite guard: previous gather-segment scalar reads drained
                            T.set_flag("s", "v", 0)
                            T.wait_flag("s", "v", 0)
                            T.copy(Indices[s_row_0, g_0, (s % NI) * BI : (s % NI + 1) * BI], indices_ub[islot, :])
                            T.set_flag("mte2", "v", 0)
                            T.wait_flag("mte2", "v", 0)
                            # Vectorized row addresses (BI elements in 2 V ops):
                            # rows = clamp(idx + bos, 0, seq_len_kv - 1); the
                            # mask zeroes -1 / out-of-range contributions, the
                            # clamp only protects addressing.
                            T.tile.add(rows_ub, indices_ub[islot, :], bos_0)
                            T.tile.clamp(rows_ub, rows_ub, 0, seq_len_kv - 1, BI)
                            T.set_flag("v", "s", 0)
                            T.wait_flag("v", "s", 0)
                            for j in range(HALF):  # HALF row reads queued back-to-back, no sync
                                row_j = vid * HALF + j
                                kv_row = rows_ub[row_j]  # one scalar UB read per row
                                T.copy(KV[kv_row, g_0, :], kv_gather_ub[side, j, :])
                            T.set_flag("mte2", "mte3", 0)
                            T.wait_flag("mte2", "mte3", 0)
                            # rs_ub rewrite guard (adjacent pair, position
                            # matters): placed after the row reads drain and
                            # before this block's bulk store is issued, so it
                            # only captures the previous block's bulk / rs
                            # store (both already drained, wait ~= free); the
                            # reduce_sum of block s runs after this point in
                            # program order, so the rs_ub protection holds.
                            T.set_flag("mte3", "v", 0)
                            T.wait_flag("mte3", "v", 0)
                            # Bulk store-out of the whole ws_kv slot row range
                            # (one multi-block MTE3 copy)
                            T.copy(
                                kv_gather_ub[side, :, :],
                                ws_kv[cid, nslot * BI + vid * HALF : nslot * BI + (vid + 1) * HALF, :],
                            )
                            T.set_cross_flag("MTE3", F0_READY_BASE + nslot)
                        else:
                            # ---- NI == 1, s >= 1: block s belongs to task s (may not exist -> clamp) ----
                            pid_s = tau_s * grid_size + cid
                            b_s_s = pid_s % (seq_len * REPLICATE_H)
                            s_row_s = T.min(b_s_s // REPLICATE_H, seq_len - 1)
                            g_s = T.min(pid_s // (seq_len * REPLICATE_H), kv_group - 1)
                            b_sp = TokenIndices[s_row_s, 0]
                            bos_s = Offsets[b_sp]
                            T.set_flag("v", "mte2", 0)
                            T.wait_flag("v", "mte2", 0)
                            T.set_flag("s", "v", 0)
                            T.wait_flag("s", "v", 0)
                            T.copy(Indices[s_row_s, g_s, 0:BI], indices_ub[islot, :])
                            T.set_flag("mte2", "v", 0)
                            T.wait_flag("mte2", "v", 0)
                            T.tile.add(rows_ub, indices_ub[islot, :], bos_s)
                            T.tile.clamp(rows_ub, rows_ub, 0, seq_len_kv - 1, BI)
                            T.set_flag("v", "s", 0)
                            T.wait_flag("v", "s", 0)
                            for j in range(HALF):
                                row_j = vid * HALF + j
                                kv_row = rows_ub[row_j]
                                T.copy(KV[kv_row, g_s, :], kv_gather_ub[side, j, :])
                            T.set_flag("mte2", "mte3", 0)
                            T.wait_flag("mte2", "mte3", 0)
                            T.set_flag("mte3", "v", 0)
                            T.wait_flag("mte3", "v", 0)
                            # Existence guard (tau_s > 0 => NI == 1 => s < nblk
                            # iff task s is valid): the phantom block's row
                            # reads already landed in the side slot (never
                            # consumed; later side reuse is covered by the
                            # two-tier safety argument), while the bulk store /
                            # F0 are skipped, keeping the flag count at nblk
                            # with a strict 1:1 pairing.
                            if s < nblk:
                                T.copy(
                                    kv_gather_ub[side, :, :],
                                    ws_kv[cid, nslot * BI + vid * HALF : nslot * BI + (vid + 1) * HALF, :],
                                )
                                T.set_cross_flag("MTE3", F0_READY_BASE + nslot)

                    # ---- Pre-launch of indices for block LEAD (the main loop's
                    # first gather consumes it). The owning task of block LEAD
                    # is the compile-time constant LEAD // NI: LEAD < NI uses
                    # the task-0 scalars; LEAD >= NI derives the row from pure
                    # pid arithmetic, and the surrounding nblk > LEAD guard
                    # guarantees that pid is a valid task row (no clamp
                    # needed).
                    if LEAD < nblk:  # noqa: SIM300 - reads as "block LEAD exists"
                        if LEAD < NI:
                            T.set_flag("v", "mte2", 0)
                            T.wait_flag("v", "mte2", 0)
                            T.copy(Indices[s_row_0, g_0, LEAD * BI : (LEAD + 1) * BI], indices_ub[LEAD % IDXSLOTS, :])
                            T.set_flag("mte2", "v", 0)
                            T.wait_flag("mte2", "v", 0)
                        else:
                            pid_l = (LEAD // NI) * grid_size + cid
                            b_s_l = pid_l % (seq_len * REPLICATE_H)
                            s_row_l = b_s_l // REPLICATE_H
                            # The group id must be derived from the global task
                            # pid (monotonic across groups), not from the
                            # in-row number b_s_l (which stays <
                            # seq_len*REPLICATE_H and would always yield
                            # group 0, reading the wrong group's Indices when
                            # kv_group > 1 with NI == 1).
                            g_l = pid_l // (seq_len * REPLICATE_H)
                            T.set_flag("v", "mte2", 0)
                            T.wait_flag("v", "mte2", 0)
                            T.copy(Indices[s_row_l, g_l, 0:BI], indices_ub[LEAD % IDXSLOTS, :])
                            T.set_flag("mte2", "v", 0)
                            T.wait_flag("mte2", "v", 0)

                    for k in T.serial(ntask):
                        # ---- Task k's own scalars (k < ntask => always valid; used by post/select) ----
                        pid = k * grid_size + cid
                        b_s_i = pid % (seq_len * REPLICATE_H)
                        g_i = pid // (seq_len * REPLICATE_H)
                        s_row_i = b_s_i // REPLICATE_H
                        r_i = b_s_i % REPLICATE_H
                        H0 = g_i * padded_H + r_i * R_HEAD_OFFSET
                        b_i = TokenIndices[s_row_i, 0]
                        s_i = TokenIndices[s_row_i, 1]
                        bos = Offsets[b_i]
                        # ---- Look-ahead scalars of task k+OFFSET_N (the
                        # far-window gather blocks of this body). k+OFFSET_N
                        # may exceed ntask, so s_row is clamped to a legal row
                        # before reading: b_n is always a legal batch id, so
                        # Offsets[b_n] is in bounds; its value is only consumed
                        # inside the gn < nblk guard. ----
                        pid_n = pid + OFFSET_N * grid_size
                        b_s_n = pid_n % (seq_len * REPLICATE_H)
                        g_n = pid_n // (seq_len * REPLICATE_H)
                        s_row_n = b_s_n // REPLICATE_H
                        r_n = b_s_n % REPLICATE_H
                        s_row_safe = T.min(s_row_n, seq_len - 1)
                        b_n = TokenIndices[s_row_safe, 0]
                        bos_n = Offsets[b_n]

                        # ---- Slot reuse at depth DEPTH without free flags ----
                        #   * bulk(gn) rewrites ws_kv slot gn % DEPTH from
                        #     body(gn-LEAD), which is program-ordered after
                        #     body(gn-DEPTH) (LEAD <= DEPTH-1) — and the post
                        #     of body(gn-DEPTH) has already passed
                        #     waitF1(gn-DEPTH), so the Cube read of that slot
                        #     has drained (the head flag inside gemm_v0 covers
                        #     the ws_kv read);
                        #   * the Cube rewrite of ws_acc slot n % DEPTH is
                        #     released by F0(n), whose MTE3 store was issued
                        #     after the mte2->mte3 set that captured the
                        #     earlier post's acc read of slot n-DEPTH, so that
                        #     read has drained;
                        #   * task boundaries need no special case: the flat
                        #     block stream reuses slots with period DEPTH.
                        # The runtime `if gn < nblk` branches identically on
                        # both AIVs (same k/t), so cross-flag set/wait stay a
                        # strict 1:1 quorum, and every flag sits inside a
                        # valid block's program range (no dangling flags).
                        for t in T.serial(NI):
                            gn = k * NI + t + LEAD
                            if gn < nblk:
                                # ---- Look-ahead gather(gn) -> slot gn%DEPTH, side gn%SIDES, islot gn%IDXSLOTS ----
                                tg = gn % NI  # block index of gn inside its owning task
                                # tg belongs to task k+OFFSET_N iff t+LEAD >= NI
                                # (the window invariant guarantees no third task,
                                # so selecting between the two scalar sets is exact)
                                nxt = t + LEAD >= NI
                                nslot = gn % DEPTH
                                side = gn % SIDES
                                islot = gn % IDXSLOTS
                                s_row_g = T.if_then_else(nxt, s_row_n, s_row_i)
                                g_g = T.if_then_else(nxt, g_n, g_i)
                                # ---- indices(gn) landing path ----
                                # gn == LEAD: prefetched by the prologue above;
                                # gn > LEAD: prefetched by the hoist segment of
                                # body(gn-LEAD-1) (NI >= 2 guards t'+LEAD+1 <
                                # 2*NI; NI == 1 always issues via the k+3
                                # arithmetic window). The only hoist-coverage
                                # gap is LEAD >= NI with NI > 1 and t == 0,
                                # handled by the head fallback below; LEAD < NI
                                # never needs the fallback.
                                if LEAD >= NI and NI > 1:  # noqa: SIM102 - nesting mirrors the case analysis
                                    if t == 0:  # noqa: SIM102
                                        if gn > LEAD:
                                            T.set_flag("v", "mte2", 0)
                                            T.wait_flag("v", "mte2", 0)
                                            T.copy(
                                                Indices[s_row_g, g_g, tg * BI : (tg + 1) * BI],
                                                indices_ub[islot, :],
                                            )
                                            T.set_flag("mte2", "v", 0)
                                            T.wait_flag("mte2", "v", 0)
                                # rows_ub rewrite guard: g(gn-1)'s scalar reads drained
                                T.set_flag("s", "v", 0)
                                T.wait_flag("s", "v", 0)
                                # kv_gather_ub[side] reuse protection — the two
                                # tiers are documented at the SIDES definition;
                                # no flag is needed here in either tier.
                                # Vectorized row addresses (same as the
                                # prologue; the clamp only protects addressing).
                                # bos is passed branch-wise as a GM scalar —
                                # wrapping it in a select triggers an access_ptr
                                # parse failure in the Ascend codegen
                                # (AddsAndMulsOpCodegen).
                                if nxt:
                                    T.tile.add(rows_ub, indices_ub[islot, :], bos_n)
                                else:
                                    T.tile.add(rows_ub, indices_ub[islot, :], bos)
                                T.tile.clamp(rows_ub, rows_ub, 0, seq_len_kv - 1, BI)
                                T.set_flag("v", "s", 0)
                                T.wait_flag("v", "s", 0)
                                for j in range(HALF):
                                    row_j = vid * HALF + j
                                    kv_row = rows_ub[row_j]
                                    T.copy(KV[kv_row, g_g, :], kv_gather_ub[side, j, :])
                                # ---- indices(gn+1) prefetch: issued right after
                                # the row reads so it lands under this segment's
                                # mte2->mte3 drain marker (the landing wait is
                                # free). The v->mte2 pair before it drains the
                                # last Cast reader of the target slot
                                # (IDXSLOTS >= LEAD + 2 derivation above).
                                # NI >= 2: skipped when block gn+1 leaves the
                                # {k, k+1} scalar window — the consuming gather
                                # re-issues it at its head;
                                # NI == 1: block gn+1 lives in the k+3 pure-
                                # arithmetic window (indices addressing needs
                                # no GM scalars), guarded by gn+1 < nblk.
                                if gn + 1 < nblk:
                                    if NI == 1:
                                        pid_h3 = pid + 3 * grid_size
                                        b_s_h3 = pid_h3 % (seq_len * REPLICATE_H)
                                        s_row_h3 = b_s_h3 // REPLICATE_H
                                        g_h3 = pid_h3 // (seq_len * REPLICATE_H)
                                        T.set_flag("v", "mte2", 0)
                                        T.wait_flag("v", "mte2", 0)
                                        T.copy(Indices[s_row_h3, g_h3, 0:BI], indices_ub[(gn + 1) % IDXSLOTS, :])
                                    else:
                                        if t + LEAD + 1 < 2 * NI:
                                            T.set_flag("v", "mte2", 0)
                                            T.wait_flag("v", "mte2", 0)
                                            # Block gn+1 belongs to task k or
                                            # k+1 (the issue guard guarantees
                                            # t+LEAD+1 < 2*NI, so both scalar
                                            # sets are inside the window)
                                            nh = t + LEAD + 1 >= NI
                                            s_row_h = T.if_then_else(nh, s_row_n, s_row_i)
                                            g_h = T.if_then_else(nh, g_n, g_i)
                                            T.copy(
                                                Indices[s_row_h, g_h, ((gn + 1) % NI) * BI : ((gn + 1) % NI + 1) * BI],
                                                indices_ub[(gn + 1) % IDXSLOTS, :],
                                            )
                                T.set_flag("mte2", "mte3", 0)
                                T.wait_flag("mte2", "mte3", 0)
                                # rs_ub rewrite guard (adjacent pair, position
                                # matters, same as the prologue): placed after
                                # the row reads drain and before bulk(gn) is
                                # issued, it only captures the previous block's
                                # bulk / rs store (both already drained, wait
                                # ~= free), never this block's own just-issued
                                # bulk.
                                T.set_flag("mte3", "v", 0)
                                T.wait_flag("mte3", "v", 0)
                                T.copy(
                                    kv_gather_ub[side, :, :],
                                    ws_kv[cid, nslot * BI + vid * HALF : nslot * BI + (vid + 1) * HALF, :],
                                )
                                T.set_cross_flag("MTE3", F0_READY_BASE + nslot)

                            # ---- post(n = k*NI + t): wait F1(slot) -> load acc
                            # -> mask / exp / reduce -> store ----
                            # Tail-body rs_ub guard: when the gather above was
                            # skipped (gn >= nblk), the rs_ub protection pair
                            # that normally lives inside the gather segment is
                            # missing, and this post's reduce_sum (V writes
                            # rs_ub) could race with the previous post's rs
                            # store (an async MTE3 read) — the store would then
                            # flush mixed data. The tail body has no fresh bulk
                            # in flight, so the wait is nearly free; both AIV
                            # branches are identical (same k/t/nblk), the pair
                            # is adjacent, and the flag-protocol count is not
                            # affected.
                            if gn >= nblk:
                                T.set_flag("mte3", "v", 0)
                                T.wait_flag("mte3", "v", 0)
                            n = k * NI + t
                            slot = n % DEPTH
                            T.wait_cross_flag(F1_ACC_BASE + slot)
                            # acc_ub rewrite guard: p(n-1)'s vector-chain reads drained
                            T.set_flag("v", "mte2", 0)
                            T.wait_flag("v", "mte2", 0)
                            T.copy(
                                ws_acc[cid, slot * H_per_block : (slot + 1) * H_per_block, vid * HALF : (vid + 1) * HALF],
                                acc_ub,
                            )
                            # The acc-landing wait is sunk to just before the
                            # first acc read: the mask chain below (~8 V ops,
                            # depending only on the long-landed indices)
                            # overlaps with the acc MTE2 copy, so only the
                            # excess of the acc latency over the mask chain is
                            # exposed. The post segment carries no MTE3 wait:
                            # the rs_ub protection is handled by the adjacent
                            # mte3->v pair in the gather segment (or the tail
                            # guard above).

                            # ---- Mask arithmetic chain (same V pipe, zero barriers) ----
                            T.copy(
                                indices_ub[n % IDXSLOTS, vid * HALF : (vid + 1) * HALF], idx_f_ub
                            )  # int32 -> fp32 cast (indices landed in an earlier body)
                            T.tile.add(t_ub, idx_f_ub, T.float32(1.0))  # t = idx + 1
                            T.tile.mul(m1_ub, idx_f_ub, T.float32(-1.0))  # m1 = -idx
                            T.tile.add(m1_ub, m1_ub, T.float32(s_i + 1))  # m1 = (s+1) - idx
                            T.tile.clamp(m1_ub, m1_ub, 0.0, 1.0, HALF)  # 1{idx <= s}
                            T.tile.clamp(m2_ub, t_ub, 0.0, 1.0, HALF)  # 1{idx >= 0}
                            T.tile.mul(mask_ub, m1_ub, m2_ub)  # mask in {0, 1}
                            T.tile.broadcast(mask2d_ub, mask_ub, axis=0)  # [HALF] -> [H, HALF]

                            # ---- score -> p -> reduce (the lse subtraction is
                            # vectorized; lse2d_ub was preloaded at the tail of
                            # task k-1 — or by the preamble for k == 0) ----
                            # acc MTE2 landing proof (the marker also covers
                            # this body's row reads and the prefetched indices)
                            T.set_flag("mte2", "v", 0)
                            T.wait_flag("mte2", "v", 0)
                            T.tile.mul(acc_ub, acc_ub, sm_scale)
                            T.tile.sub(acc_ub, acc_ub, lse2d_ub)  # one 2D subtract (replaces a per-h loop)
                            T.tile.exp(acc_ub, acc_ub)
                            T.tile.mul(acc_ub, acc_ub, mask2d_ub)  # masked -> 0 (equivalent to fill(-inf) + exp)
                            T.reduce_sum(acc_ub, rs_ub, dim=0, clear=True)
                            T.set_flag("v", "mte3", 0)
                            T.wait_flag("v", "mte3", 0)
                            T.copy(rs_ub, ReduceSum[s_row_i, g_i, r_i, t * BI + vid * HALF : t * BI + (vid + 1) * HALF])

                        # ---- Task tail: preload lse2d of task k+1 (the guard
                        # guarantees the k+1 row is valid). When OFFSET_N > 1,
                        # the look-ahead scalar set belongs to task k+OFFSET_N,
                        # so the lse preload derives task k+1 from pure pid
                        # arithmetic — self-contained per TVMScript IfFrame
                        # scoping. Timing: within the V-pipe order, post(k, .)
                        # subtracts before broadcast(k+1); the lse_ub rewrite is
                        # guarded so the previous broadcast has read it out.
                        if k + 1 < ntask:
                            if OFFSET_N > 1:
                                pid_1 = pid + grid_size
                                b_s_1 = pid_1 % (seq_len * REPLICATE_H)
                                g_1 = pid_1 // (seq_len * REPLICATE_H)
                                s_row_1 = b_s_1 // REPLICATE_H
                                r_1 = b_s_1 % REPLICATE_H
                                H0_1 = g_1 * padded_H + r_1 * R_HEAD_OFFSET
                                T.set_flag("v", "mte2", 0)
                                T.wait_flag("v", "mte2", 0)
                                T.copy(Lse[s_row_1, H0_1 : H0_1 + H_per_block], lse_ub)
                                T.set_flag("mte2", "v", 0)
                                T.wait_flag("mte2", "v", 0)
                                T.tile.broadcast(lse2d_ub, lse_ub, axis=1)
                            else:
                                H0_n = g_n * padded_H + r_n * R_HEAD_OFFSET
                                T.set_flag("v", "mte2", 0)
                                T.wait_flag("v", "mte2", 0)
                                T.copy(Lse[s_row_n, H0_n : H0_n + H_per_block], lse_ub)
                                T.set_flag("mte2", "v", 0)
                                T.wait_flag("mte2", "v", 0)
                                T.tile.broadcast(lse2d_ub, lse_ub, axis=1)

    return main


@tilelang.jit(
    out_idx=[6],  # ReduceSum is framework-allocated and returned
    workspace_idx=[7, 8],  # ws_kv, ws_acc (static [grid_size, ...] shapes)
)
def tl_sparse_mla_topk_reducesum(
    heads,
    dim,
    tail_dim,
    topk,
    kv_group=1,
    sm_scale=None,
    block_I=32,
    grid_size=20,
):
    """Build the sparse_mla_topk_reducesum kernel (Expert mode, generic path).

    Args:
        heads: total number of heads (head_kv = heads // kv_group).
        dim: main QK dim D (power of two).
        tail_dim: tail QK dim D_tail (power of two).
        topk: selected positions per row (topk % block_I == 0).
        kv_group: number of kv groups G.
        sm_scale: defaults to (dim + tail_dim) ** -0.5.
        block_I: topk-direction block BI (even and >= 32; the AIV column half
            is BI // 2).
        grid_size: persistent grid size (compile-time constant; the host
            queries cube_core_num, which is 20 on 910B3).

    Returns:
        Callable kernel: func(Q, KV, Indices, Lse, Offsets, TokenIndices)
        -> ReduceSum [total_S, kv_group, REPLICATE_H, topk] float32.
    """
    return _reducesum_generic_body(heads, dim, tail_dim, topk, kv_group, sm_scale, block_I, grid_size)


# ============================================================================
# Host side: input preparation + call entry.
# Only builds the auxiliary TokenIndices tensor (constructed on CPU, one H2D
# copy) and compiles / invokes the kernel. User input buffers are never
# touched; the interface-level normalization (sum over REPLICATE_H + division)
# is outside the delivery boundary and left to the caller.
# ============================================================================
_BUILD_CACHE: dict = {}


def get_cube_core_num(default=20):
    """Query the device cube core count (source of the grid_size constant; 20 on 910B3)."""
    try:
        import torch

        if torch.npu.is_available():
            n = torch.npu.get_device_properties(0).cube_core_num
            if isinstance(n, int) and n > 0:
                return n
    except Exception:  # pragma: no cover - fall back to the default when probing fails
        pass
    return default


def prepare_token_indices(offsets):
    """Build TokenIndices [total_S, 2] int32 (batch_id, position-in-batch) from
    Offsets (cu_seqlens). Constructed on CPU, then a single H2D copy."""
    import torch

    off = offsets.detach().cpu().tolist()
    batch_ids = []
    positions = []
    for b in range(len(off) - 1):
        n = int(off[b + 1]) - int(off[b])
        batch_ids.append(torch.full((n,), b, dtype=torch.int32))
        positions.append(torch.arange(n, dtype=torch.int32))
    if batch_ids:
        ti = torch.stack([torch.cat(batch_ids), torch.cat(positions)], dim=1)
    else:
        ti = torch.zeros((0, 2), dtype=torch.int32)
    return ti.to(offsets.device)


def sparse_mla_topk_reducesum(
    q,
    kv,
    topk_indices,
    lse,
    offsets,
    dim_v=512,
    kv_group=None,
    sm_scale=None,
    block_I=32,
    grid_size=None,
):
    """Host call entry (same semantics as the original repository's
    sparse_mla_topk_reducesum_interface, with the delivery boundary ending at
    ReduceSum — the REPLICATE_H merge and normalization belong to the caller).

    Args:
        q: [total_S, heads, dim+tail_dim] bfloat16 (NPU)
        kv: [total_S_kv, kv_group, dim+tail_dim] bfloat16 (NPU)
        topk_indices: [total_S, kv_group, topk] int32 (NPU), values in
            [-1, total_S_kv) with any distribution (row-independent /
            partially shared / all-identical / non-contiguous / shuffled all
            take the same generic path)
        lse: [total_S, heads] float32 (NPU)
        offsets: [batch+1] int32 cu_seqlens
        dim_v: main dim D (tail_dim = last dim - dim_v)
        kv_group: defaults to kv.shape[1]
        sm_scale: defaults to (dim + tail_dim) ** -0.5
        block_I: topk-direction block (even and >= 32)
        grid_size: defaults to the device cube_core_num

    Returns:
        ReduceSum [total_S, kv_group, REPLICATE_H, topk] float32 (NPU)

    Performance note: block_I=128 is typically the fastest configuration when
    topk % 128 == 0; block_I=64 is the runner-up when topk % 64 == 0 but
    topk % 128 != 0. The default stays 32 to cover the widest topk support
    domain (128 only applies when topk % 128 == 0).
    """
    seq_len, heads, ddt = q.shape
    tail_dim = ddt - dim_v
    kv_group = kv.shape[1] if kv_group is None else kv_group
    topk = topk_indices.shape[-1]
    if grid_size is None:
        grid_size = get_cube_core_num()

    key = (heads, dim_v, tail_dim, topk, kv_group, sm_scale, block_I, grid_size)
    func = _BUILD_CACHE.get(key)
    if func is None:
        func = tl_sparse_mla_topk_reducesum(
            heads=heads,
            dim=dim_v,
            tail_dim=tail_dim,
            topk=topk,
            kv_group=kv_group,
            sm_scale=sm_scale,
            block_I=block_I,
            grid_size=grid_size,
        )
        _BUILD_CACHE[key] = func

    token_indices = prepare_token_indices(offsets)  # auxiliary tensor (CPU -> one H2D copy)
    return func(q, kv, topk_indices, lse, offsets, token_indices)
