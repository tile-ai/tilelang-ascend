import torch

import tilelang
from tilelang import language as T
from tilelang.intrinsics import make_zn_layout, make_nz_layout


# ===========================================================================
# Mask precomputation (host-side, shared across batch/head)
# ===========================================================================


def build_causal_mask(seq_q, seq_kv, window_size, device, block_M=128, block_N=128):
    """Build causal + optional sliding window mask on the host.

    Returns: [padded_seq_q, padded_seq_kv] float32 (1.0=visible, 0.0=masked).
    Padded to block_M/block_N alignment; out-of-bounds Q/K positions = 0.0 (masked).
    Shared across all batches and heads (no padding in this op).
    Right-aligned causal: query i sees key j iff j <= i + (seq_kv - seq_q).
    """
    padded_q = ((seq_q + block_M - 1) // block_M) * block_M
    padded_kv = ((seq_kv + block_N - 1) // block_N) * block_N
    offset = seq_kv - seq_q
    q_idx = torch.arange(padded_q, device=device).view(-1, 1)  # [padded_q, 1]
    k_idx = torch.arange(padded_kv, device=device).view(1, -1)  # [1, padded_kv]
    # Causal (right-aligned) + within-bounds
    visible = (k_idx <= q_idx + offset) & (q_idx < seq_q) & (k_idx < seq_kv)
    if window_size is not None:
        visible = visible & (k_idx >= q_idx + offset - window_size + 1)
    return visible.float()


# ===========================================================================
# JIT kernel (Expert mode, CV fusion, BHSD layout, mask-after-exp + sink)
# Structure ported from examples/gqa_fwd_varlen/gqa_fwd_varlen.py
# ===========================================================================

PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: False,  # manual CV separation
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: False,  # manual inter-core sync
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: False,  # manual intra-core sync
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: False,  # manual memory planning
}

NUM_CORES = 24  # 910B has 24 AI Cores (static task distribution)


@tilelang.jit(out_idx=[5], workspace_idx=[6, 7, 8], pass_configs=PASS_CONFIGS)
def flashattn(
    batch,
    heads,
    seq_q,
    seq_kv,
    dim,
    block_M=128,
    block_N=128,
    num_stages=8,
    cross_interval=1,
    has_window=False,
    real_seq_q=None,
    real_seq_kv=None,
):
    """Attention Sink Flash Attention forward kernel (BHSD, Expert mode).

    Expert CV pipeline directly ported from gqa_fwd_varlen (verified 17.78ms).
    Differences: MHA (no GQA), 2D mask (no batch dim), attention sink integration.

    Args:
        batch: batch size (compile-time).
        heads: number of attention heads (MHA: heads == head_kv, no GQA).
        seq_q: padded query sequence length (compile-time, must be divisible by block_M).
        seq_kv: padded key/value sequence length (compile-time, divisible by block_N).
        dim: head dimension (fixed 128).
        block_M: Q block size.
        block_N: K/V block size.
        num_stages: CV pipeline depth (batch KV iterations per outer loop).
        cross_interval: cross-core sync interval (sync every N iterations).
        has_window: whether sliding window mask is active (compile-time). When True,
            mask load+mul is always applied (window may affect any block). When False
            (causal-only), fully-visible blocks skip mask load+mul (Stage 3 iter2).
        real_seq_q: original (unpadded) query length, for padding-aware mask skip.
            Defaults to seq_q (no padding).
        real_seq_kv: original (unpadded) key/value length, for padding-aware mask
            skip. Defaults to seq_kv (no padding).

    Kernel inputs (9 tensors):
        Q:      [batch, heads, seq_q, dim]      float16  (idx 0)
        K:      [batch, heads, seq_kv, dim]      float16  (idx 1)
        V:      [batch, heads, seq_kv, dim]      float16  (idx 2)
        Sinks:  [heads, seq_q]                   float16  (idx 3, per-head sink,
                                                          pre-broadcast on host)
        Mask:   [seq_q, seq_kv]                  float32  (idx 4, precomputed causal+window,
                                                          2D shared across batch/head)
        Output: [batch, heads, seq_q, dim]       float16  (idx 5, out_idx)
        workspace_1: [NUM_CORES, num_stages, block_M, block_N] fp16  (idx 6, QK scores)
        workspace_2: [NUM_CORES, num_stages, block_M, block_N] fp16  (idx 7, softmax P)
        workspace_3: [NUM_CORES, num_stages, block_M, dim]    fp16  (idx 8, PV output)
    """
    sm_scale = (1.0 / dim) ** 0.5  # natural exp, no log2(e) factor
    dtype = "float16"
    accum_dtype = "float"

    q_shape = [batch, heads, seq_q, dim]
    kv_shape = [batch, heads, seq_kv, dim]
    mask_shape = [seq_q, seq_kv]
    o_shape = [batch, heads, seq_q, dim]

    assert seq_q % block_M == 0, f"seq_q ({seq_q}) must be divisible by block_M ({block_M})"
    assert seq_kv % block_N == 0, f"seq_kv ({seq_kv}) must be divisible by block_N ({block_N})"
    assert num_stages % 2 == 0, "num_stages must be even for double buffering"

    num_q_blocks = seq_q // block_M
    block_num = num_q_blocks * heads * batch
    # Causal right-aligned offset: query i sees key j iff j <= i + offset.
    # Used for per-q_block KV iteration cropping (Stage 3 iter1 optimization):
    # q_block bx only needs to iterate visible KV blocks, skipping fully-masked
    # tail blocks (which contribute 0 via mask-after-exp). For causal attention
    # with seq_q==seq_kv (offset=0), visible blocks for q_block bx = bx+1
    # (decreasing), saving ~48% of total KV work (sum 1..32=528 vs 32*32=1024).
    # Flag safety: each task consumes 3 "next-batch" cross-flags at batch 0 and
    # produces 3 at its last batch (variable eff_num_outer) — identical accounting
    # to the fixed-num_outer case (leftovers carry to next task / init / destroy).
    # Note: compile-time `num_outer = ceildiv(max_kv_iters, num_stages)` (and the
    # `max_kv_iters = seq_kv // block_N` it derived from) was replaced by runtime
    # `eff_kv_iters` / `eff_num_outer` in Stage 3 iter2 (see loop body below).
    causal_offset = seq_kv - seq_q

    # Stage 3 iter2: real (unpadded) dimensions for mask skip optimization.
    # When has_window=False, fully-visible causal blocks (mask all 1.0) skip the
    # mask load+mul. The skip condition needs the REAL causal offset (not padded)
    # and the real seq_kv (to detect padded key positions with mask=0).
    if real_seq_q is None:
        real_seq_q = seq_q
    if real_seq_kv is None:
        real_seq_kv = seq_kv
    real_causal_offset = real_seq_kv - real_seq_q

    q_tasks = block_num // NUM_CORES
    r_tasks = block_num % NUM_CORES

    # Cross-core semaphore IDs (Cube <-> Vector) — identical to gqa_fwd_varlen
    SEM_WS1_C2V = 0  # workspace_1 (QK^T) ready: Cube -> Vector
    SEM_WS1_V2C = 1  # workspace_1 consumed: Vector -> Cube
    SEM_WS2_V2C = 2  # workspace_2 (softmax P) ready: Vector -> Cube
    SEM_WS2_C2V = 3  # workspace_2 consumed: Cube -> Vector
    SEM_WS3_C2V = 4  # workspace_3 (PV output) ready: Cube -> Vector
    SEM_WS3_V2C = 5  # workspace_3 consumed: Vector -> Cube

    # Intra-core signal IDs (C Scope) — identical to gqa_fwd_varlen
    SIG_K_L1 = 0
    SIG_P_L1 = 1
    SIG_V_L1 = 2
    SIG_L0AB = 3  # double-buffer base: slot 0 = SIG_L0AB, slot 1 = SIG_L0AB + 1
    SIG_L0C = 5  # double-buffer base: slot 0 = SIG_L0C,  slot 1 = SIG_L0C + 1

    # Intra-core signal IDs (V Scope) — identical to gqa_fwd_varlen + sink sync
    SIG_IO_UB = 0  # io_buf double-buffer: slot 0 = SIG_IO_UB, slot 1 = SIG_IO_UB + 1
    SIG_S_HALF = 2
    SIG_MASK_FREE = 3  # V -> MTE2: buf_2d released after exp (mask can overwrite)
    SIG_MASK_READY = 4  # MTE2 -> V: mask loaded into buf_2d (mul can proceed)
    SIG_SINK_READY = 5  # MTE2 -> V: sinks_half_ub loaded from GM (sink load sync)

    @T.prim_func
    def main(
        Q: T.Tensor(q_shape, dtype),  # type: ignore
        K: T.Tensor(kv_shape, dtype),  # type: ignore
        V: T.Tensor(kv_shape, dtype),  # type: ignore
        Sinks: T.Tensor([heads, seq_q], dtype),  # type: ignore
        Mask: T.Tensor(mask_shape, accum_dtype),  # type: ignore
        Output: T.Tensor(o_shape, dtype),  # type: ignore
        workspace_1: T.Tensor([NUM_CORES, num_stages, block_M, block_N], dtype),  # type: ignore
        workspace_2: T.Tensor([NUM_CORES, num_stages, block_M, block_N], dtype),  # type: ignore
        workspace_3: T.Tensor([NUM_CORES, num_stages, block_M, dim], dtype),  # type: ignore
    ):
        with T.Kernel(NUM_CORES, is_npu=True) as (cid, vid):
            # ---- Buffer allocation (Expert: explicit memory hierarchy) ----
            # L1 buffers (Cube core)
            q_l1 = T.alloc_L1([block_M, dim], dtype)
            k_l1 = T.alloc_L1([block_N, dim], dtype)
            v_l1 = T.alloc_L1([block_N, dim], dtype)
            p_l1 = T.alloc_L1([block_M, block_N], dtype)

            # L1 layout optimization (ZN for Q/P/V, NZ for K with transpose)
            T.annotate_layout(
                {
                    q_l1: make_zn_layout(q_l1),
                    k_l1: make_nz_layout(k_l1),
                    p_l1: make_zn_layout(p_l1),
                    v_l1: make_zn_layout(v_l1),
                }
            )

            # L0 double buffering (2 slots for pipeline parallelism)
            l0a = T.alloc_L0A([2, block_M, dim], dtype)
            l0b = T.alloc_L0B([2, dim, block_N], dtype)
            l0c = T.alloc_L0C([2, block_M, block_N], accum_dtype)

            # UB buffers (vid split: block_M//2 per vid, Vector core)
            acc_o = T.alloc_ub([block_M // 2, dim], accum_dtype)

            # Batch softmax buffers (num_stages slots for deferred rescale)
            r_factors = T.alloc_ub([num_stages, block_M // 2, 1], accum_dtype)
            sumexp_is = T.alloc_ub([num_stages, block_M // 2, 1], accum_dtype)

            sumexp = T.alloc_ub([block_M // 2, 1], accum_dtype)
            neg_sm = T.alloc_ub([2, block_M // 2, 1], accum_dtype)  # double-buffered -m_i

            # IO and work buffers (reused across phases)
            io_buf = T.alloc_ub([2, block_M // 2, block_N], dtype)  # GM <-> UB (fp16, double-buffered)
            acc_s_half = T.alloc_ub([block_M // 2, block_N], dtype)  # fp16 softmax output

            work_ub = T.alloc_ub([block_M // 2, block_N], accum_dtype)  # main compute buffer (fp32)
            buf_2d = T.alloc_ub([block_M // 2, block_N], accum_dtype)  # broadcast+mask buffer (fp32)

            # ---- Sink UB buffers (this op unique, ~896B total) ----
            sinks_half_ub = T.alloc_ub([block_M // 2], dtype)  # 1D fp16 load target
            sinks_fp32_1d = T.alloc_ub([block_M // 2], accum_dtype)  # 1D fp32 convert
            sinks_ub = T.alloc_ub([block_M // 2, 1], accum_dtype)  # 2D fp32 broadcast (常驻)
            sink_exp_ub = T.alloc_ub([block_M // 2, 1], accum_dtype)  # 2D fp32 temp

            half_M = block_M // 2  # TIR variable for slice expressions

            # ---- Static task distribution (NUM_CORES=24) ----
            my_start = cid * q_tasks + T.if_then_else(cid < r_tasks, cid, r_tasks)
            my_count = q_tasks + T.if_then_else(cid < r_tasks, 1, 0)

            # ==================== Cube core (vid=0) ====================
            with T.Scope("C"):
                # init: pretend consumer already released
                T.set_cross_flag("MTE2", SEM_WS2_C2V)
                T.set_flag("MTE1", "MTE2", SIG_K_L1)
                T.set_flag("MTE1", "MTE2", SIG_P_L1)
                T.set_flag("MTE1", "MTE2", SIG_V_L1)
                T.set_flag("M", "MTE1", SIG_L0AB)
                T.set_flag("M", "MTE1", SIG_L0AB + 1)
                T.set_flag("FIX", "M", SIG_L0C)
                T.set_flag("FIX", "M", SIG_L0C + 1)

                for t in T.serial(my_count):
                    task_id = my_start + t
                    bx = task_id % num_q_blocks
                    by = (task_id // num_q_blocks) % heads  # MHA: by is head directly (no GQA)
                    bz = task_id // (num_q_blocks * heads)

                    # Causal KV iteration cropping (Stage 3 iter1): only iterate
                    # visible KV blocks for this q_block. Right-aligned causal:
                    # query i sees key j iff j <= i + causal_offset. Last query row
                    # of q_block bx is bx*block_M + block_M - 1; last visible key =
                    # min(seq_kv - 1, that + causal_offset). Visible KV block count =
                    # last_visible_key // block_N + 1 (>= 1 always). Fully-masked
                    # tail blocks contribute 0 (mask-after-exp) -> safe to skip.
                    # eff_num_outer (runtime TIR) replaces compile-time num_outer as
                    # the outer loop bound; both Cube and Vector compute the same
                    # value from the same bx, so flag pairing stays balanced.
                    _raw_last = bx * block_M + block_M - 1 + causal_offset
                    _last_vis = T.if_then_else(_raw_last < seq_kv, _raw_last, seq_kv - 1)
                    eff_kv_iters = T.floordiv(_last_vis, block_N) + 1
                    eff_num_outer = T.ceildiv(eff_kv_iters, num_stages)

                    T.copy(Q[bz, by, bx * block_M : (bx + 1) * block_M, :], q_l1)
                    T.barrier_all()

                    for k in T.serial(eff_num_outer):
                        _remaining = eff_kv_iters - k * num_stages
                        batch_iters = T.if_then_else(_remaining < num_stages, _remaining, num_stages)

                        # --- GEMM1: produce QK^T scores into workspace_1 ---
                        T.wait_cross_flag(SEM_WS1_V2C)
                        for i in T.serial(batch_iters):
                            side = i % 2
                            idx = k * num_stages + i

                            # K: GM -> L1 (MTE2 -> MTE1 flag). MHA: by is head index.
                            T.wait_flag("MTE1", "MTE2", SIG_K_L1)
                            T.copy(K[bz, by, idx * block_N : (idx + 1) * block_N, :], k_l1)
                            T.set_flag("MTE2", "MTE1", SIG_K_L1)

                            # Q: L1 -> L0A (only first 2 iterations, then reused)
                            T.wait_flag("M", "MTE1", SIG_L0AB + side)
                            if i < 2:
                                T.copy(q_l1, l0a[side, :, :])

                            # K: L1 -> L0B with transpose
                            T.wait_flag("MTE2", "MTE1", SIG_K_L1)
                            T.copy(k_l1, l0b[side, :, :], transpose=True)
                            T.set_flag("MTE1", "MTE2", SIG_K_L1)
                            T.set_flag("MTE1", "M", SIG_L0AB + side)

                            # MMA: QK^T -> L0C
                            T.wait_flag("MTE1", "M", SIG_L0AB + side)
                            T.wait_flag("FIX", "M", SIG_L0C + side)
                            T.mma(l0a[side, :, :], l0b[side, :, :], l0c[side, :, :], init=True)
                            T.set_flag("M", "MTE1", SIG_L0AB + side)
                            T.set_flag("M", "FIX", SIG_L0C + side)

                            # L0C -> workspace_1 (FIX pipeline)
                            T.wait_flag("M", "FIX", SIG_L0C + side)
                            T.copy(l0c[side, :, :], workspace_1[cid, i, :, :])
                            T.set_flag("FIX", "M", SIG_L0C + side)
                            if (i + 1) % cross_interval == 0 or i == batch_iters - 1:
                                T.set_cross_flag("FIX", SEM_WS1_C2V)

                        # --- GEMM2: consume P from workspace_2, produce PV into workspace_3 ---
                        T.wait_cross_flag(SEM_WS3_V2C)
                        for i in T.serial(batch_iters):
                            side = i % 2
                            idx = k * num_stages + i

                            # V: GM -> L1. MHA: by is head index.
                            T.wait_flag("MTE1", "MTE2", SIG_V_L1)
                            T.copy(V[bz, by, idx * block_N : (idx + 1) * block_N, :], v_l1)
                            T.set_flag("MTE2", "MTE1", SIG_V_L1)

                            # P: workspace_2 -> L1
                            T.wait_flag("MTE1", "MTE2", SIG_P_L1)
                            if i % cross_interval == 0:
                                T.wait_cross_flag(SEM_WS2_V2C)
                            T.copy(workspace_2[cid, i, :, :], p_l1)
                            T.set_flag("MTE2", "MTE1", SIG_P_L1)

                            # V: L1 -> L0B (no transpose for PV)
                            T.wait_flag("MTE2", "MTE1", SIG_V_L1)
                            T.wait_flag("M", "MTE1", SIG_L0AB + side)
                            T.copy(v_l1, l0b[side, :, :])
                            T.set_flag("MTE1", "MTE2", SIG_V_L1)

                            # P: L1 -> L0A (no transpose)
                            T.wait_flag("MTE2", "MTE1", SIG_P_L1)
                            T.copy(p_l1, l0a[side, :, :])
                            T.set_flag("MTE1", "MTE2", SIG_P_L1)
                            T.set_flag("MTE1", "M", SIG_L0AB + side)

                            # MMA: PV -> L0C
                            T.wait_flag("MTE1", "M", SIG_L0AB + side)
                            T.wait_flag("FIX", "M", SIG_L0C + side)
                            T.mma(l0a[side, :, :], l0b[side, :, :], l0c[side, :, :], init=True)
                            T.set_flag("M", "MTE1", SIG_L0AB + side)
                            T.set_flag("M", "FIX", SIG_L0C + side)

                            # L0C -> workspace_3
                            T.wait_flag("M", "FIX", SIG_L0C + side)
                            T.copy(l0c[side, :, :], workspace_3[cid, i, :, :])
                            T.set_flag("FIX", "M", SIG_L0C + side)
                            if (i + 1) % cross_interval == 0 or i == batch_iters - 1:
                                T.set_cross_flag("FIX", SEM_WS3_C2V)

                        T.set_cross_flag("MTE2", SEM_WS2_C2V)

                # destroy: consume outstanding init-direction flags
                T.wait_flag("MTE1", "MTE2", SIG_K_L1)
                T.wait_flag("MTE1", "MTE2", SIG_P_L1)
                T.wait_flag("MTE1", "MTE2", SIG_V_L1)
                T.wait_flag("M", "MTE1", SIG_L0AB)
                T.wait_flag("M", "MTE1", SIG_L0AB + 1)
                T.wait_flag("FIX", "M", SIG_L0C)
                T.wait_flag("FIX", "M", SIG_L0C + 1)

            # ==================== Vector core (vid=1) ====================
            with T.Scope("V"):
                # init: pretend workspaces already released by Cube
                T.set_cross_flag("MTE2", SEM_WS1_V2C)
                T.set_cross_flag("MTE2", SEM_WS3_V2C)
                # init: pretend both io_buf slots are free (consumer already released)
                T.set_flag("V", "MTE2", SIG_IO_UB)
                T.set_flag("V", "MTE2", SIG_IO_UB + 1)
                T.set_flag("MTE3", "V", SIG_S_HALF)

                for t in T.serial(my_count):
                    task_id = my_start + t
                    bx = task_id % num_q_blocks
                    by = (task_id // num_q_blocks) % heads  # MHA: by is head directly
                    bz = task_id // (num_q_blocks * heads)

                    # Causal KV iteration cropping (Stage 3 iter1) — MUST match Cube
                    # scope computation exactly (same bx/causal_offset/seq_kv/block_N/
                    # num_stages) so both sides agree on batch count and flag pairing.
                    _raw_last = bx * block_M + block_M - 1 + causal_offset
                    _last_vis = T.if_then_else(_raw_last < seq_kv, _raw_last, seq_kv - 1)
                    eff_kv_iters = T.floordiv(_last_vis, block_N) + 1
                    eff_num_outer = T.ceildiv(eff_kv_iters, num_stages)

                    T.tile.fill(acc_o, 0.0)
                    T.tile.fill(sumexp, 0.0)
                    T.tile.fill(neg_sm, 2**30)  # large positive = -inf max

                    # ---- Load sink value for this head (host pre-broadcast [heads, seq_q]) ----
                    # 3-step load (proven in Developer version + DESIGN §14.1):
                    #   1D fp16 slice load -> 1D fp32 convert -> 2D fp32 broadcast
                    # T.Parallel scalar broadcast does NOT work on Ascend (compiles to
                    # sequential copy, not broadcast) — host pre-broadcast is mandatory.
                    #
                    # CRITICAL: MTE2→V sync required. The GM→UB copy (Sinks→sinks_half_ub)
                    # runs on the MTE2 pipeline, but the subsequent UB→UB copy and broadcast
                    # run on the V pipeline. In Expert mode (AUTO_SYNC=False), this cross-
                    # pipeline dependency MUST be explicitly synced via set_flag/wait_flag,
                    # otherwise V may read sinks_half_ub before MTE2 finishes loading
                    # (race condition → non-deterministic output). Same pattern as gqa's
                    # mask load (SIG_MASK_READY). wait_flag(src="MTE2", dst="V") = V waits.
                    T.copy(
                        Sinks[by, bx * block_M + vid * half_M : bx * block_M + vid * half_M + half_M],
                        sinks_half_ub,
                    )
                    T.set_flag("MTE2", "V", SIG_SINK_READY)  # MTE2: signal load done
                    T.wait_flag("MTE2", "V", SIG_SINK_READY)  # V: wait for MTE2 load
                    T.copy(sinks_half_ub, sinks_fp32_1d)  # fp16 -> fp32 (1D, V pipeline)
                    # broadcast 1D [half_M] -> 2D [half_M, 1] (ascend_tile.py:2066-2073, axis=1)
                    T.tile.broadcast(sinks_ub, sinks_fp32_1d)

                    for k in T.serial(eff_num_outer):
                        _remaining = eff_kv_iters - k * num_stages
                        batch_iters = T.if_then_else(_remaining < num_stages, _remaining, num_stages)

                        # --- Phase 1: Softmax batch (compute exp, write workspace_2) ---
                        T.wait_cross_flag(SEM_WS2_C2V)
                        for i in T.serial(batch_iters):
                            cur = i % 2
                            prv = 1 - cur
                            idx = k * num_stages + i
                            io_side = i % 2  # io_buf double-buffer slot

                            # Read QK scores from workspace_1 (vid half)
                            T.wait_flag("V", "MTE2", SIG_IO_UB + io_side)
                            if i % cross_interval == 0:
                                T.wait_cross_flag(SEM_WS1_C2V)
                            T.copy(
                                workspace_1[cid, i, vid * half_M : vid * half_M + half_M, :],
                                io_buf[io_side, :, :],
                            )
                            T.set_flag("MTE2", "V", SIG_IO_UB + io_side)

                            T.wait_flag("MTE2", "V", SIG_IO_UB + io_side)
                            T.copy(io_buf[io_side, :, :], work_ub)  # fp16 -> fp32
                            T.set_flag("V", "MTE2", SIG_IO_UB + io_side)

                            # Online softmax: max (on raw scores, scale via axpy)
                            T.reduce_max(work_ub, neg_sm[cur, :, :], dim=-1)
                            T.tile.mul(neg_sm[cur, :, :], neg_sm[cur, :, :], -sm_scale)
                            T.tile.min(neg_sm[cur, :, :], neg_sm[cur, :, :], neg_sm[prv, :, :])

                            # Vectorized sub + scale: buf_2d = (work_ub - new_max) * sm_scale
                            T.tile.broadcast(buf_2d, neg_sm[cur, :, :])
                            T.tile.axpy(buf_2d, work_ub, sm_scale)
                            T.tile.exp(work_ub, buf_2d)

                            # === MASK AFTER EXP (2D mask tensor from GM, no batch dim) ===
                            # Stage 3 iter2: skip mask load+mul for fully-visible blocks.
                            # When has_window=False (causal-only), blocks where ALL keys are
                            # visible to ALL queries (mask all 1.0) skip the GM->UB mask load
                            # and the T.tile.mul (mul by 1.0 is a no-op). For the perf target
                            # (4096x4096 causal), ~94% of cropped iters are fully-visible.
                            #
                            # Flag safety (SIG_MASK_FREE/READY are intra-core semaphores):
                            # Each "apply" iter does balanced set/wait (V set FREE -> MTE2 wait
                            # FREE; MTE2 set READY -> V wait READY), leaving both at 0. "Skip"
                            # iters don't touch either flag -> stay at 0. No cross-iter dep.
                            # First iter: if skip, flags start at 0 (default); next apply sets
                            # FREE (0->1), MTE2 waits (1->0) — clean. No init needed.
                            #
                            # Skip condition (only when has_window=False):
                            #   causal_full: (idx+1)*block_N - 1 <= bx*block_M + real_causal_offset
                            #     (last key in block visible to first query in q_block)
                            #   no_pad_keys: (idx+1)*block_N <= real_seq_kv
                            #     (block has no padded key positions, which have mask=0)
                            if has_window:
                                T.set_flag("V", "MTE2", SIG_MASK_FREE)
                                T.wait_flag("V", "MTE2", SIG_MASK_FREE)
                                T.copy(
                                    Mask[
                                        bx * block_M + vid * half_M : bx * block_M + vid * half_M + half_M,
                                        idx * block_N : (idx + 1) * block_N,
                                    ],
                                    buf_2d,
                                )
                                T.set_flag("MTE2", "V", SIG_MASK_READY)
                                T.wait_flag("MTE2", "V", SIG_MASK_READY)
                                T.tile.mul(work_ub, work_ub, buf_2d)
                            else:
                                if (idx + 1) * block_N - 1 > bx * block_M + real_causal_offset or (idx + 1) * block_N > real_seq_kv:
                                    T.set_flag("V", "MTE2", SIG_MASK_FREE)
                                    T.wait_flag("V", "MTE2", SIG_MASK_FREE)
                                    T.copy(
                                        Mask[
                                            bx * block_M + vid * half_M : bx * block_M + vid * half_M + half_M,
                                            idx * block_N : (idx + 1) * block_N,
                                        ],
                                        buf_2d,
                                    )
                                    T.set_flag("MTE2", "V", SIG_MASK_READY)
                                    T.wait_flag("MTE2", "V", SIG_MASK_READY)
                                    T.tile.mul(work_ub, work_ub, buf_2d)

                            # Write masked softmax P to workspace_2 (via acc_s_half)
                            T.wait_flag("MTE3", "V", SIG_S_HALF)
                            T.copy(work_ub, acc_s_half)  # fp32 -> fp16
                            T.set_flag("V", "MTE3", SIG_S_HALF)

                            T.wait_flag("V", "MTE3", SIG_S_HALF)
                            T.copy(
                                acc_s_half,
                                workspace_2[cid, i, vid * half_M : vid * half_M + half_M, :],
                            )
                            T.set_flag("MTE3", "V", SIG_S_HALF)
                            if (i + 1) % cross_interval == 0 or i == batch_iters - 1:
                                T.set_cross_flag("MTE3", SEM_WS2_V2C)

                            # Precompute r_factors and sumexp_is for phase 2
                            T.reduce_sum(work_ub, sumexp_is[i, :, :], dim=-1)
                            T.tile.sub(r_factors[i, :, :], neg_sm[cur, :, :], neg_sm[prv, :, :])

                        T.set_cross_flag("MTE2", SEM_WS1_V2C)

                        # --- Phase 2: O accumulation batch (rescale + accumulate PV) ---
                        for i in T.serial(batch_iters):
                            # Deferred rescale: exp(old_max - new_max)
                            T.tile.exp(r_factors[i, :, :], r_factors[i, :, :])
                            T.tile.mul(sumexp, sumexp, r_factors[i, :, :])
                            T.tile.add(sumexp, sumexp, sumexp_is[i, :, :])
                            T.tile.broadcast(buf_2d, r_factors[i, :, :])
                            T.tile.mul(acc_o, acc_o, buf_2d)

                            # Read PV output from workspace_3 (vid half)
                            io_side = i % 2  # io_buf double-buffer slot
                            T.wait_flag("V", "MTE2", SIG_IO_UB + io_side)
                            if i % cross_interval == 0:
                                T.wait_cross_flag(SEM_WS3_C2V)
                            T.copy(
                                workspace_3[cid, i, vid * half_M : vid * half_M + half_M, :],
                                io_buf[io_side, :, :],
                            )
                            T.set_flag("MTE2", "V", SIG_IO_UB + io_side)

                            T.wait_flag("MTE2", "V", SIG_IO_UB + io_side)
                            T.copy(io_buf[io_side, :, :], work_ub)  # fp16 -> fp32
                            T.set_flag("V", "MTE2", SIG_IO_UB + io_side)

                            T.tile.add(acc_o, acc_o, work_ub)

                        T.set_cross_flag("MTE2", SEM_WS3_V2C)

                    # ============ Attention Sink (this op unique, stabilized) ============
                    # sumexp += exp(sinks - m*), where m* = max(sinks, m_i_final).
                    #
                    # Why m* (not m_i_final): the kernel uses mask-after-exp, so m_i_final
                    # = max(QK scores over iterated blocks). With causal KV cropping
                    # (Stage 3 iter1), fully-masked blocks are skipped, so m_i_final =
                    # max(visible scores only) — smaller than the nocrop m_i which
                    # included masked scores. When sinks > m_i_final, exp(sinks -
                    # m_i_final) overflows / dominates sumexp -> output collapses to 0
                    # (PRECISION FAIL, max_diff ~1.2 on l1_multi_irregular). The golden
                    # (ref_program) uses m* = max(sinks, logits_max) as stabilization, so
                    # exp(sinks - m*) <= 1. Fix: rescale sumexp/acc_o from m_i_final to
                    # m* = max(sinks, m_i_final), then add exp(sinks - m*) <= 1.
                    # Mathematically equivalent to exp(sinks - m_i_final) (DESIGN §2.5:
                    # "分子分母同乘 exp(m_i - sinks)"), but numerically stable. When
                    # sinks <= m_i_final (common case), m* = m_i_final, rescale = 1, no
                    # change — so cases that passed before are unaffected.
                    #
                    # m_i_final recovered via neg_sm monotonicity (DESIGN §14.3):
                    #   neg_sm = -m_i is monotonically non-increasing, so
                    #   min(neg_sm[0], neg_sm[1]) = neg_sm[cur_last] = -m_i_final
                    # r_factors[0..2] are free after Phase 2 (used as scratch, num_stages>=3).
                    # Step 1: neg_m = -m_i_final = min(neg_sm[0], neg_sm[1])
                    T.tile.min(r_factors[0, :, :], neg_sm[0, :, :], neg_sm[1, :, :])
                    # Step 2: neg_sinks = -sinks (scratch)
                    T.tile.mul(r_factors[1, :, :], sinks_ub, -1.0)
                    # Step 3: neg_m_star = min(neg_m, neg_sinks) = -max(m_i_final, sinks) = -m*
                    T.tile.min(sink_exp_ub, r_factors[0, :, :], r_factors[1, :, :])
                    # Step 4: rescale = m_i_final - m* = neg_m_star - neg_m (<= 0 -> exp <= 1)
                    T.tile.sub(r_factors[2, :, :], sink_exp_ub, r_factors[0, :, :])
                    T.tile.exp(r_factors[2, :, :], r_factors[2, :, :])  # exp(m_i_final - m*)
                    # Step 5: rescale sumexp and acc_o from m_i_final to m*
                    T.tile.mul(sumexp, sumexp, r_factors[2, :, :])  # [M,1] *= [M,1]
                    T.tile.broadcast(buf_2d, r_factors[2, :, :])  # [M,1] -> [M,N]
                    T.tile.mul(acc_o, acc_o, buf_2d)  # [M,N] *= [M,N]
                    # Step 6: sink term = exp(sinks - m*) = exp(sinks + neg_m_star) (<= 1)
                    T.tile.add(sink_exp_ub, sinks_ub, sink_exp_ub)
                    T.tile.exp(sink_exp_ub, sink_exp_ub)
                    # Step 7: sumexp += exp(sinks - m*)
                    T.tile.add(sumexp, sumexp, sink_exp_ub)

                    # ---- Final normalize: acc_o /= sumexp ----
                    T.tile.broadcast(buf_2d, sumexp)
                    T.tile.div(acc_o, acc_o, buf_2d)

                    # Write back (vid half)
                    T.copy(acc_o, acc_s_half)  # fp32 -> fp16
                    T.barrier_all()
                    T.copy(
                        acc_s_half,
                        Output[
                            bz,
                            by,
                            bx * block_M + vid * half_M : bx * block_M + vid * half_M + half_M,
                            :,
                        ],
                    )

                # destroy: consume outstanding init-direction flags
                T.wait_flag("V", "MTE2", SIG_IO_UB)
                T.wait_flag("V", "MTE2", SIG_IO_UB + 1)
                T.wait_flag("MTE3", "V", SIG_S_HALF)

    return main


# ===========================================================================
# Smoke test entry (CI compatibility)
#
# Repository CI (examples/bench_test.sh) marks a script PASSED only if its
# stdout contains "Test Passed!" or "Kernel Output Match!". This __main__
# runs the minimal L0 shape (l0_min_causal config) and validates against
# ref_program (imported from test_mha_sink_fwd_bhsd.py) so the main file is
# independently runnable in CI. The @tilelang.jit flashattn kernel above is
# unchanged.
# ===========================================================================


if __name__ == "__main__":
    import os
    import sys

    # Import golden from sibling test module (same dir on sys.path)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from test_mha_sink_fwd_bhsd import ref_program  # noqa: E402

    tilelang.disable_cache()
    torch.set_default_device("npu")
    torch.manual_seed(0)

    # Minimal smoke test — matches L0 "l0_min_causal" config:
    # batch=1, heads=1, seq_q=seq_kv=128, dim=128, causal (window=None).
    batch, heads = 1, 1
    seq_q, seq_kv, dim = 128, 128, 128
    window_size = None  # causal only
    block_M, block_N = 128, 128
    atol = 1e-2

    torch.manual_seed(0)
    q = torch.randn(batch, heads, seq_q, dim, dtype=torch.float16, device="npu")
    k = torch.randn(batch, heads, seq_kv, dim, dtype=torch.float16, device="npu")
    v = torch.randn(batch, heads, seq_kv, dim, dtype=torch.float16, device="npu")
    sinks = torch.randn(heads, dtype=torch.float16, device="npu")

    # Pre-broadcast sinks [heads] -> [heads, seq_q] (avoid T.Parallel broadcast bug)
    sinks_broad = sinks.unsqueeze(1).expand(-1, seq_q).contiguous()

    # Build causal mask (right-aligned, 2D shared across batch/head)
    mask = build_causal_mask(seq_q, seq_kv, window_size, "npu", block_M, block_N)

    kernel = flashattn(
        batch,
        heads,
        seq_q,
        seq_kv,
        dim,
        block_M=block_M,
        block_N=block_N,
        has_window=(window_size is not None),
        real_seq_q=seq_q,
        real_seq_kv=seq_kv,
    )

    out = kernel(q, k, v, sinks_broad, mask)
    torch.npu.synchronize()

    # Golden (device-agnostic, runs on NPU; takes original [heads] sinks)
    ref_out = ref_program(q, k, v, sinks, sliding_window=window_size, dtype=torch.float16)
    torch.npu.synchronize()

    max_diff = (out.float() - ref_out.float()).abs().max().item()
    print(f"max_diff: {max_diff:.6e}")
    assert max_diff < atol, f"Precision check failed: max_diff={max_diff} >= atol={atol}"
    print("Test Passed!")
