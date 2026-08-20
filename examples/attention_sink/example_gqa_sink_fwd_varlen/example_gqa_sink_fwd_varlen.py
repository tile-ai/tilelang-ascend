"""
GQA + Attention Sink Forward (Variable-length) for Ascend NPU — Expert Mode.

Layout: 4D padded [B, H, S, D] (varlen inputs are padded on host side).
Supports: GQA (grouped-query), causal mask, attention sink, fp16.

Based on fa_opt multi-buffer pipeline port (flash_attn_bhsd_expert_h16_d128.py):
  - Fixed Core + workspace [core_num, num_stages, ...] (L2 cache residency)
  - Multi-buffer pipeline with 6-flag batched cross-core sync
  - T.mma + L0A/L0B/L0C double-buffer + ZN/NZ layout
  - Fine-grained intra-core flags with complete buffer ownership

Performance (B=8, H=64, G=16, Sq=Sk=2048, D=128, causal, fp16, 910B3):
  ~12 ms (1.06x GPU baseline 11.4 ms)
"""

import tilelang
from tilelang import language as T
from tilelang.intrinsics import make_zn_layout, make_nz_layout
import torch
from typing import Optional

# ========== Kernel Implementation ==========

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: False,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: False,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: False,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: False,
}


@tilelang.jit(out_idx=[8], workspace_idx=[9, 10, 11], pass_configs=pass_configs)
def gqa_sink_fwd(
    batch_size,
    groups,
    heads,
    dim,
    seq_len_q,
    seq_len_kv,
    is_causal,
    mask_tiles=None,
    window_size=None,
    sm_scale=None,
    block_M=64,
    block_N=128,
    num_stages=14,  # fa_opt default: multi-buffer pipeline depth
    core_num=20,  # 910B3 cube_core_num (Fixed Core launch)
):
    # Block sizes as Python int literals for buffer shape compatibility.
    _BM = 64
    _BN = 128
    _half_M = _BM // 2  # = 32

    # Bug fix: alignment & compatibility assertions (were present in Developer
    # version but lost during fa_opt port). Without these, non-aligned seqlen
    # causes silent out-of-bounds writes; dim != _BN causes L0C reuse overflow.
    assert seq_len_q % _BM == 0, f"seq_len_q ({seq_len_q}) must be divisible by block_M ({_BM})"
    assert seq_len_kv % _BN == 0, f"seq_len_kv ({seq_len_kv}) must be divisible by block_N ({_BN})"
    assert dim == _BN, (
        f"dim ({dim}) must equal block_N ({_BN}) for L0C buffer reuse (GEMM1 output [_BM, _BN] and GEMM2 output [_BM, dim] share l0c)"
    )
    assert window_size is None, "window_size (sliding window) is not yet implemented in the kernel; only the golden reference supports it"
    assert num_stages >= 2, f"num_stages ({num_stages}) must be >= 2 for L0 double buffering"

    if sm_scale is None:
        sm_scale = (1.0 / dim) ** 0.5

    head_kv = heads // groups
    dtype = "float16"
    accum_dtype = "float"

    m_blocks = (seq_len_q + _BM - 1) // _BM
    n_blocks = (seq_len_kv + _BN - 1) // _BN
    block_num = m_blocks * heads * batch_size

    if mask_tiles is None:
        mask_tiles = batch_size * m_blocks * n_blocks

    # Static task distribution: evenly split block_num across core_num.
    # Cores 0..r-1 get (q+1) tasks; cores r..core_num-1 get q tasks.
    q_tasks = block_num // core_num
    r_tasks = block_num % core_num

    def task_range(cid_val):
        start = cid_val * q_tasks + T.if_then_else(cid_val < r_tasks, cid_val, r_tasks)
        count = q_tasks + T.if_then_else(cid_val < r_tasks, 1, 0)
        return start, count

    # --- Step 2: multi-buffer pipeline parameters (fa_opt port) ---
    # num_outer is the MAX outer-iteration count (from max KV seq len). Per-tile
    # causal tiles run fewer inner iters via dynamic batch_iters (guarded >= 0).
    num_iters_max = (seq_len_kv + _BN - 1) // _BN  # Python int
    num_outer = (num_iters_max + num_stages - 1) // num_stages
    cross_interval = 2  # fa_opt default: batch cross-core sync every 2 iters

    # 6 cross-core flag IDs — one per (workspace, direction). Avoids the flag
    # reuse deadlock of a 4-flag scheme (where one flag would serve both
    # "P ready" and "ws1 free", making init ambiguous). Matches fa_opt SEM_*.
    FLAG_S_READY = 0  # Cube GEMM1 -> Vector softmax : S written to ws1
    FLAG_WS1_FREE = 1  # Vector -> Cube GEMM1         : ws1 consumed (free)
    FLAG_P_READY = 2  # Vector softmax -> Cube GEMM2 : P written to ws2
    FLAG_WS2_FREE = 3  # Cube GEMM2 -> Vector         : ws2 consumed (free)
    FLAG_O_READY = 4  # Cube GEMM2 -> Vector O-accum : O written to ws3
    FLAG_WS3_FREE = 5  # Vector O-accum -> Cube GEMM2 : ws3 consumed (free)

    # Local event IDs are allocated per directed pipe pair. Different meanings keep
    # distinct names even when their directed pairs allow the same numeric ID.
    # MTE2 <-> MTE1
    SIG_K_L1 = 0  # MTE2 writes k_l1, MTE1 reads it
    SIG_P_L1 = 1  # MTE2 writes acc_s_l1 (P), MTE1 reads it
    SIG_V_L1 = 2  # MTE2 writes v_l1, MTE1 reads it
    SIG_Q_L1 = 3

    # MTE1 <-> M
    SIG_L0AB = 0  # double-buffer slots 0 and 1

    # M <-> FIX
    SIG_L0C = 0  # double-buffer slots 0 and 1

    # MTE2 <-> V
    SIG_IO_UB = 0  # MTE2 writes acc_s_ub_ (ws1->UB), V reads it (axpy)
    SIG_MASK_UB = 1  # MTE2 writes mask_half (Mask->UB), V reads it (cast)
    SIG_O_UB = 2  # MTE2 writes acc_o_ub (ws3->UB), V reads it (add)

    # V <-> MTE3
    SIG_S_HALF = 0  # V writes acc_s_half (cast P), MTE3 reads it (->ws2)
    SIG_O_HALF = 1  # V writes acc_o_half (cast O), MTE3 reads it (->Output)

    @T.prim_func
    def main(
        Q: T.Tensor([batch_size, heads, seq_len_q, dim], dtype),  # type: ignore
        K: T.Tensor([batch_size, head_kv, seq_len_kv, dim], dtype),  # type: ignore
        V: T.Tensor([batch_size, head_kv, seq_len_kv, dim], dtype),  # type: ignore
        Sinks: T.Tensor([heads], dtype),  # type: ignore
        q_seqlens: T.Tensor([batch_size], "int32"),  # type: ignore
        kv_seqlens: T.Tensor([batch_size], "int32"),  # type: ignore
        max_seqlen_k: T.int32,  # unused — kernel uses kv_seqlens[bz] instead; kept for backward compat
        Mask: T.Tensor([mask_tiles, _BM, _BN], dtype),  # type: ignore  # ③ fp16
        Output: T.Tensor([batch_size, heads, seq_len_q, dim], dtype),  # type: ignore
        workspace_1: T.Tensor([core_num, num_stages, _BM, _BN], accum_dtype),  # type: ignore
        workspace_2: T.Tensor([core_num, num_stages, _BM, _BN], dtype),  # type: ignore
        workspace_3: T.Tensor([core_num, num_stages, _BM, dim], accum_dtype),  # type: ignore
    ):
        with T.Kernel(core_num, is_npu=True) as (cid, vid):
            # --- Buffer allocation (Expert: explicit memory hierarchy) ---
            # Allocated once outside the grid-stride loop (Fixed Core reuse).
            # L1 buffers (Cube scope, full _BM for GEMM)
            q_l1 = T.alloc_L1([_BM, dim], dtype)
            k_l1 = T.alloc_L1([_BN, dim], dtype)
            v_l1 = T.alloc_L1([_BN, dim], dtype)
            acc_s_l1 = T.alloc_L1([_BM, _BN], dtype)

            # Step 3: ZN/NZ layout for L1 buffers (fa_opt L105-112).
            # q_l1/acc_s_l1/v_l1 ZN, k_l1 NZ (adapts to transpose in L0B load).
            T.annotate_layout(
                {
                    q_l1: make_zn_layout(q_l1),
                    k_l1: make_nz_layout(k_l1),
                    acc_s_l1: make_zn_layout(acc_s_l1),
                    v_l1: make_zn_layout(v_l1),
                }
            )

            # Step 3: L0 double-buffered buffers (fa_opt L114-116).
            # l0a/l0b shared between GEMM1 (Q/K) and GEMM2 (P/V) — different
            # time. l0c shared between GEMM1 (S) and GEMM2 (O) since
            # _BN == dim == 128 (same output shape).
            l0a = T.alloc_L0A([2, _BM, dim], dtype)
            l0b = T.alloc_L0B([2, dim, _BN], dtype)
            l0c = T.alloc_L0C([2, _BM, _BN], accum_dtype)

            # UB buffers (Vector scope, _half_M rows per vid)
            acc_o = T.alloc_ub([_half_M, dim], accum_dtype)
            logsum = T.alloc_ub([_half_M, 1], accum_dtype)
            scores_max = T.alloc_ub([_half_M, 1], accum_dtype)
            scores_max_prev = T.alloc_ub([_half_M, 1], accum_dtype)
            scores_sum = T.alloc_ub([_half_M, 1], accum_dtype)

            acc_s_ub = T.alloc_ub([_half_M, _BN], accum_dtype)
            acc_s_ub_ = T.alloc_ub([_half_M, _BN], accum_dtype)
            acc_s_half = T.alloc_ub([_half_M, _BN], dtype)
            # mask_half has independent UB address (Step 4: not shared with acc_s_half)
            mask_half = T.alloc_ub([_half_M, _BN], dtype)
            acc_o_ub = T.alloc_ub([_half_M, dim], accum_dtype)
            acc_o_half = T.alloc_ub([_half_M, dim], dtype)

            # Broadcast buffers for vectorized row-wise ops
            brd_bn = T.alloc_ub([_half_M, _BN], accum_dtype)
            brd_dim = T.alloc_ub([_half_M, dim], accum_dtype)

            # Step 2: multi-buffer pipeline scratch (precomputed per batch slot).
            # Written in softmax batch, read in O-accum batch (same num_outer).
            r_factors = T.alloc_ub([num_stages, _half_M, 1], accum_dtype)
            sumexp_is = T.alloc_ub([num_stages, _half_M, 1], accum_dtype)

            # --- Expert mode: explicit address annotation ---
            # Step 3: acc_s_l1 gets independent L1 address (NOT shared with
            # k_l1). In pipeline mode, GEMM1's last-iter MTE2 (write k_l1) and
            # GEMM2's first-iter MTE2 (write acc_s_l1) overlap; sharing an
            # address causes data races. L1 capacity 512KB >> 98KB used.
            T.annotate_address(
                {
                    # L1 addresses (fp16: 2 bytes; all independent)
                    # q_l1: 0, k_l1: BM*d*2, v_l1: (BM+BN)*d*2, acc_s_l1: (BM+2*BN)*d*2
                    # acc_s_l1 independent (NOT shared with k_l1) to avoid pipeline race
                    q_l1: 0,
                    k_l1: _BM * dim * 2,
                    v_l1: (_BM + _BN) * dim * 2,
                    acc_s_l1: (_BM + 2 * _BN) * dim * 2,
                    # L0A/L0B/L0C: independent physical stores, each starts at 0.
                    # L0A: 2*_BM*dim*2 = 32KB < 64KB
                    # L0B: 2*dim*_BN*2 = 64KB = 64KB (full)
                    # L0C: 2*_BM*_BN*4 = 64KB < 128KB (shared GEMM1/GEMM2)
                    l0a: 0,
                    l0b: 0,
                    l0c: 0,
                    # UB addresses (fp32: 4 bytes; dim == _BN == 128)
                    # h = _half_M, d = dim = _BN = 128
                    # Layout: acc_o | logsum | scores_max | scores_max_prev | scores_sum
                    #       | acc_s_ub | acc_s_ub_ | acc_s_half | mask_half
                    #       | acc_o_ub | acc_o_half | brd_bn(=brd_dim) | r_factors | sumexp_is
                    # All independent (Step 4: removed barrier_all). UB total ~108KB < 196KB.
                    # Base = h*(3d+4)*4 = 49664 for all buffers after acc_s_half.
                    # Offset N (in h*d units): 2→4→8→10→14, each N = prev_N + prev_buf_size.
                    acc_o: 0,
                    logsum: _half_M * dim * 4,
                    scores_max: _half_M * (dim + 1) * 4,
                    scores_max_prev: _half_M * (dim + 2) * 4,
                    scores_sum: _half_M * (dim + 3) * 4,
                    acc_s_ub: _half_M * (dim + 4) * 4,
                    acc_s_ub_: _half_M * (2 * dim + 4) * 4,
                    acc_s_half: _half_M * (3 * dim + 4) * 4,
                    mask_half: _half_M * (3 * dim + 4) * 4 + _half_M * dim * 2,  # +h*d*2 = 57856
                    acc_o_ub: _half_M * (3 * dim + 4) * 4 + _half_M * dim * 4,  # +h*d*4 = 66048
                    acc_o_half: _half_M * (3 * dim + 4) * 4 + _half_M * dim * 8,  # +h*d*8 = 82432
                    brd_bn: _half_M * (3 * dim + 4) * 4 + _half_M * dim * 10,  # +h*d*10 = 90624
                    brd_dim: _half_M * (3 * dim + 4) * 4 + _half_M * dim * 10,  # share with brd_bn
                    r_factors: _half_M * (3 * dim + 4) * 4 + _half_M * dim * 14,  # +h*d*14 = 107008
                    sumexp_is: _half_M * (3 * dim + 4) * 4 + _half_M * dim * 14 + num_stages * _half_M * 4,  # +n*h*4 = 108800
                }
            )

            my_start, my_count = task_range(cid)

            # ============================================================
            # Cube Scope — GEMM + data搬运
            # ============================================================
            with T.Scope("C"):
                # Step 2 init: pretend ws2 is free so the first Vector softmax
                # batch (k=0) can start writing P before Cube GEMM2 consumes it.
                T.set_cross_flag("MTE2", FLAG_WS2_FREE)
                # Step 3 init: consumers pre-release the Cube-side buffers.
                T.set_flag("MTE1", "MTE2", SIG_K_L1)
                T.set_flag("MTE1", "MTE2", SIG_P_L1)
                T.set_flag("MTE1", "MTE2", SIG_V_L1)
                T.set_flag("MTE1", "MTE2", SIG_Q_L1)
                T.set_flag("M", "MTE1", SIG_L0AB)
                T.set_flag("M", "MTE1", SIG_L0AB + 1)
                T.set_flag("FIX", "M", SIG_L0C)
                T.set_flag("FIX", "M", SIG_L0C + 1)
                for t in T.serial(my_count):
                    task_id = my_start + t
                    bx = task_id % m_blocks
                    by = task_id // m_blocks % heads
                    bz = task_id // m_blocks // heads % batch_size

                    # --- Varlen indexing (per-tile, inside loop) ---
                    q_seqlen = q_seqlens[bz]
                    kv_seqlen = kv_seqlens[bz]
                    kv_head_idx = by // groups

                    # Load Q tile and keep MTE1 ownership across all KV batches.
                    T.wait_flag("MTE1", "MTE2", SIG_Q_L1)
                    T.copy(Q[bz, by, bx * _BM : (bx + 1) * _BM, :], q_l1)
                    T.set_flag("MTE2", "MTE1", SIG_Q_L1)
                    T.wait_flag("MTE2", "MTE1", SIG_Q_L1)

                    # Causal loop range: skip KV blocks entirely masked by causal
                    # NOTE: T.if_then_else used because Python if/else doesn't
                    # scope variables properly inside @T.prim_func
                    # T.max(0, ...) clamps negative max_visible (when q_seqlen >> kv_seqlen)
                    offset = kv_seqlen - q_seqlen
                    max_visible = offset + (bx + 1) * _BM
                    full_iters = T.ceildiv(kv_seqlen, _BN)
                    causal_iters = T.min(T.ceildiv(T.max(0, max_visible), _BN), full_iters)
                    loop_iters = T.if_then_else(is_causal, causal_iters, full_iters)

                    # Step 2: outer loop over num_outer batches; inner batch_iters
                    # is dynamic and clamped >= 0 so causal tiles with fewer KV
                    # iters run no-op tail batches (flag counts stay balanced).
                    for k in T.serial(num_outer):
                        _remaining = loop_iters - k * num_stages
                        batch_iters = T.if_then_else(_remaining < 0, 0, T.if_then_else(_remaining < num_stages, _remaining, num_stages))

                        # --- GEMM1 batch: produce S into ws1 (slot i) ---
                        # Step 3: T.mma + L0 double-buffer + flag sync (fa_opt L163-190)
                        T.wait_cross_flag(FLAG_WS1_FREE)
                        for i in T.serial(batch_iters):
                            side = i % 2
                            idx = k * num_stages + i

                            # MTE2: Load K to L1
                            T.wait_flag("MTE1", "MTE2", SIG_K_L1)
                            T.copy(K[bz, kv_head_idx, idx * _BN : (idx + 1) * _BN, :], k_l1)
                            T.set_flag("MTE2", "MTE1", SIG_K_L1)

                            # MTE1: Load Q to L0A (only first 2 iters; reuse)
                            T.wait_flag("M", "MTE1", SIG_L0AB + side)
                            if i < 2:
                                T.copy(q_l1, l0a[side, :, :])

                            # MTE1: Load K^T to L0B (transpose for Q@K^T)
                            T.wait_flag("MTE2", "MTE1", SIG_K_L1)
                            T.copy(k_l1, l0b[side, :, :], transpose=True)
                            T.set_flag("MTE1", "MTE2", SIG_K_L1)
                            T.set_flag("MTE1", "M", SIG_L0AB + side)

                            # M: MMA Q @ K^T -> S (L0C)
                            T.wait_flag("MTE1", "M", SIG_L0AB + side)
                            T.wait_flag("FIX", "M", SIG_L0C + side)
                            T.mma(l0a[side, :, :], l0b[side, :, :], l0c[side, :, :], init=True)
                            T.set_flag("M", "MTE1", SIG_L0AB + side)
                            T.set_flag("M", "FIX", SIG_L0C + side)

                            # FIX: L0C -> workspace_1
                            T.wait_flag("M", "FIX", SIG_L0C + side)
                            T.copy(l0c[side, :, :], workspace_1[cid, i, :, :])
                            T.set_flag("FIX", "M", SIG_L0C + side)
                            if (i + 1) % cross_interval == 0 or i == batch_iters - 1:
                                T.set_cross_flag("FIX", FLAG_S_READY)

                        # --- GEMM2 batch: consume P from ws2, produce O into ws3 ---
                        # Step 3: T.mma + L0 double-buffer + flag sync (fa_opt L194-228)
                        T.wait_cross_flag(FLAG_WS3_FREE)
                        for i in T.serial(batch_iters):
                            side = i % 2
                            idx = k * num_stages + i

                            # MTE2: Load V to L1
                            T.wait_flag("MTE1", "MTE2", SIG_V_L1)
                            T.copy(V[bz, kv_head_idx, idx * _BN : (idx + 1) * _BN, :], v_l1)
                            T.set_flag("MTE2", "MTE1", SIG_V_L1)

                            # MTE2: Load P from ws2 to L1 (acc_s_l1)
                            T.wait_flag("MTE1", "MTE2", SIG_P_L1)
                            if i % cross_interval == 0:
                                T.wait_cross_flag(FLAG_P_READY)
                            T.copy(workspace_2[cid, i, :, :], acc_s_l1)
                            T.set_flag("MTE2", "MTE1", SIG_P_L1)

                            # MTE1: Load V to L0B
                            T.wait_flag("MTE2", "MTE1", SIG_V_L1)
                            T.wait_flag("M", "MTE1", SIG_L0AB + side)
                            T.copy(v_l1, l0b[side, :, :])
                            T.set_flag("MTE1", "MTE2", SIG_V_L1)

                            # MTE1: Load P to L0A
                            T.wait_flag("MTE2", "MTE1", SIG_P_L1)
                            T.copy(acc_s_l1, l0a[side, :, :])
                            T.set_flag("MTE1", "MTE2", SIG_P_L1)
                            T.set_flag("MTE1", "M", SIG_L0AB + side)

                            # M: MMA P @ V -> O (L0C)
                            T.wait_flag("MTE1", "M", SIG_L0AB + side)
                            T.wait_flag("FIX", "M", SIG_L0C + side)
                            T.mma(l0a[side, :, :], l0b[side, :, :], l0c[side, :, :], init=True)
                            T.set_flag("M", "MTE1", SIG_L0AB + side)
                            T.set_flag("M", "FIX", SIG_L0C + side)

                            # FIX: L0C -> workspace_3
                            T.wait_flag("M", "FIX", SIG_L0C + side)
                            T.copy(l0c[side, :, :], workspace_3[cid, i, :, :])
                            T.set_flag("FIX", "M", SIG_L0C + side)
                            if (i + 1) % cross_interval == 0 or i == batch_iters - 1:
                                T.set_cross_flag("FIX", FLAG_O_READY)

                        # ws2 consumed by this GEMM2 batch → free for next Vector
                        T.set_cross_flag("MTE2", FLAG_WS2_FREE)

                    # MTE1 no longer reads q_l1; return it before the next task reloads Q.
                    T.set_flag("MTE1", "MTE2", SIG_Q_L1)

                # Step 3 destroy: consume outstanding return-direction flags.
                T.wait_flag("MTE1", "MTE2", SIG_K_L1)
                T.wait_flag("MTE1", "MTE2", SIG_P_L1)
                T.wait_flag("MTE1", "MTE2", SIG_V_L1)
                T.wait_flag("MTE1", "MTE2", SIG_Q_L1)
                T.wait_flag("M", "MTE1", SIG_L0AB)
                T.wait_flag("M", "MTE1", SIG_L0AB + 1)
                T.wait_flag("FIX", "M", SIG_L0C)
                T.wait_flag("FIX", "M", SIG_L0C + 1)

            # ============================================================
            # Vector Scope — Softmax + Mask + Accumulation
            # ============================================================
            with T.Scope("V"):
                # Step 2 init: pretend ws1, ws3 are free so the first Cube GEMM
                # batches (k=0) can start writing before Vector consumes.
                T.set_cross_flag("MTE2", FLAG_WS1_FREE)
                T.set_cross_flag("MTE2", FLAG_WS3_FREE)
                # Step 4 init: pretend consumer already released V-scope UB
                # buffers (fa_opt L244-246) — enables pipeline start at i=0.
                # MTE2-written buffers (acc_s_ub_/mask_half/acc_o_ub): pretend
                # V already released so first MTE2 wait("V","MTE2") passes.
                # MTE3-read buffers (acc_s_half/acc_o_half): pretend MTE3
                # already released so first V wait("MTE3","V") passes.
                T.set_flag("V", "MTE2", SIG_IO_UB)
                T.set_flag("V", "MTE2", SIG_MASK_UB)
                T.set_flag("MTE3", "V", SIG_S_HALF)
                T.set_flag("V", "MTE2", SIG_O_UB)
                T.set_flag("MTE3", "V", SIG_O_HALF)
                for t in T.serial(my_count):
                    task_id = my_start + t
                    bx = task_id % m_blocks
                    by = task_id // m_blocks % heads
                    bz = task_id // m_blocks // heads % batch_size

                    # --- Varlen indexing (per-tile, must match Cube) ---
                    q_seqlen = q_seqlens[bz]
                    kv_seqlen = kv_seqlens[bz]

                    # Initialize accumulators (moved into loop: buffers reused)
                    T.tile.fill(acc_o, 0.0)
                    T.tile.fill(logsum, 0.0)
                    T.tile.fill(scores_max, -(2**30))
                    # Step 4: deleted barrier_all here — fill is a V-pipeline
                    # op and the following copy(scores_max->scores_max_prev)
                    # is also V; V pipeline executes in order, no sync needed.

                    # Causal loop range: must match Cube scope exactly
                    offset = kv_seqlen - q_seqlen
                    max_visible = offset + (bx + 1) * _BM
                    full_iters = T.ceildiv(kv_seqlen, _BN)
                    causal_iters = T.min(T.ceildiv(T.max(0, max_visible), _BN), full_iters)
                    loop_iters = T.if_then_else(is_causal, causal_iters, full_iters)

                    # Step 2: outer loop over num_outer batches (matches Cube).
                    for k in T.serial(num_outer):
                        _remaining = loop_iters - k * num_stages
                        batch_iters = T.if_then_else(_remaining < 0, 0, T.if_then_else(_remaining < num_stages, _remaining, num_stages))

                        # --- Softmax batch: produce P into ws2 (slot i),
                        #     precompute r_factors[i] / sumexp_is[i] ---
                        T.wait_cross_flag(FLAG_WS2_FREE)
                        for i in T.serial(batch_iters):
                            # Save old running max
                            T.copy(scores_max, scores_max_prev)

                            if i % cross_interval == 0:
                                T.wait_cross_flag(FLAG_S_READY)

                            idx = k * num_stages + i
                            # S from ws1 → UB (MTE2 GM->UB read). Step 4: use
                            # SIG_IO_UB flag to order MTE2 write vs V read of
                            # acc_s_ub_ (replaces barrier_all). fa_opt L268-272.
                            T.wait_flag("V", "MTE2", SIG_IO_UB)
                            T.copy(
                                workspace_1[cid, i, vid * _half_M : vid * _half_M + _half_M, :],
                                acc_s_ub_,
                            )
                            T.set_flag("MTE2", "V", SIG_IO_UB)
                            T.wait_flag("MTE2", "V", SIG_IO_UB)

                            # ④ Scale: acc_s_ub = acc_s_ub_ * sm_scale
                            T.tile.fill(acc_s_ub, 0.0)
                            T.tile.axpy(acc_s_ub, acc_s_ub_, sm_scale)

                            # ③ Load fp16 mask into mask_half (MTE2 GM->UB read).
                            # Step 4: SIG_MASK_UB orders MTE2 write vs V read
                            # of mask_half (replaces barrier_all).
                            tile_idx = bz * m_blocks * n_blocks + bx * n_blocks + idx
                            T.wait_flag("V", "MTE2", SIG_MASK_UB)
                            T.copy(
                                Mask[tile_idx, vid * _half_M : vid * _half_M + _half_M, :],
                                mask_half,
                            )
                            T.set_flag("MTE2", "V", SIG_MASK_UB)
                            T.wait_flag("MTE2", "V", SIG_MASK_UB)

                            # ③ Cast mask fp16→fp32, then apply
                            T.tile.cast(acc_s_ub_, mask_half, "CAST_NONE", _half_M * _BN)
                            T.tile.add(acc_s_ub, acc_s_ub, acc_s_ub_)
                            # Step 4: V done reading acc_s_ub_ (axpy + add) ->
                            # release to MTE2 for next iter's ws1 load.
                            T.set_flag("V", "MTE2", SIG_IO_UB)
                            # Step 4: V done reading mask_half (cast) -> release
                            # to MTE2 for next iter's mask load. mask_half now
                            # has independent UB address (not shared with
                            # acc_s_half), so no need to wait for MTE3.
                            T.set_flag("V", "MTE2", SIG_MASK_UB)

                            # --- Online softmax ---
                            T.reduce_max(acc_s_ub, scores_max, dim=-1)
                            T.tile.max(scores_max, scores_max, scores_max_prev)

                            # r_factors[i] = exp(old_max - new_max) (precomputed)
                            T.tile.sub(r_factors[i, :, :], scores_max_prev, scores_max)
                            T.tile.exp(r_factors[i, :, :], r_factors[i, :, :])

                            # Subtract max and exp
                            T.tile.broadcast(brd_bn, scores_max)
                            T.tile.sub(acc_s_ub, acc_s_ub, brd_bn)
                            T.tile.exp(acc_s_ub, acc_s_ub)

                            # sumexp_is[i] = sum(exp(S - max))
                            T.reduce_sum(acc_s_ub, sumexp_is[i, :, :], dim=-1)

                            # Cast P to fp16 and write to ws2 (slot i).
                            # Step 4: SIG_S_HALF orders V write (cast) vs MTE3
                            # read (->ws2) of acc_s_half (replaces 2 barrier).
                            # fa_opt L285-291. V must wait for MTE3 to release
                            # the previous iter's acc_s_half before overwriting.
                            T.wait_flag("MTE3", "V", SIG_S_HALF)
                            T.tile.cast(acc_s_half, acc_s_ub, "CAST_NONE", _half_M * _BN)
                            T.set_flag("V", "MTE3", SIG_S_HALF)
                            T.wait_flag("V", "MTE3", SIG_S_HALF)
                            T.copy(
                                acc_s_half,
                                workspace_2[cid, i, vid * _half_M : vid * _half_M + _half_M, :],
                            )
                            T.set_flag("MTE3", "V", SIG_S_HALF)
                            if (i + 1) % cross_interval == 0 or i == batch_iters - 1:
                                T.set_cross_flag("MTE3", FLAG_P_READY)

                        # ws1 consumed by this softmax batch → free for next Cube
                        T.set_cross_flag("MTE2", FLAG_WS1_FREE)

                        # --- O accumulation batch: rescale by r_factors[i],
                        #     accumulate O_i from ws3 (slot i) ---
                        for i in T.serial(batch_iters):
                            if i % cross_interval == 0:
                                T.wait_cross_flag(FLAG_O_READY)

                            # Apply r_factors[i] (rescale logsum and acc_o)
                            T.tile.mul(logsum, logsum, r_factors[i, :, :])
                            T.tile.add(logsum, logsum, sumexp_is[i, :, :])
                            T.tile.broadcast(brd_dim, r_factors[i, :, :])
                            T.tile.mul(acc_o, acc_o, brd_dim)

                            # Read O_i from ws3 → UB and accumulate.
                            # Step 4: SIG_O_UB orders MTE2 write vs V read of
                            # acc_o_ub (replaces barrier_all). acc_o_ub has an
                            # independent UB address (no longer shares with
                            # acc_s_half), so no cross-batch hazard.
                            T.wait_flag("V", "MTE2", SIG_O_UB)
                            T.copy(
                                workspace_3[cid, i, vid * _half_M : vid * _half_M + _half_M, :],
                                acc_o_ub,
                            )
                            T.set_flag("MTE2", "V", SIG_O_UB)
                            T.wait_flag("MTE2", "V", SIG_O_UB)
                            T.tile.add(acc_o, acc_o, acc_o_ub)
                            # Step 4: V done reading acc_o_ub -> release to MTE2
                            T.set_flag("V", "MTE2", SIG_O_UB)

                        # ws3 consumed by this O-accum batch → free for next Cube
                        T.set_cross_flag("MTE2", FLAG_WS3_FREE)

                    # --- Attention Sink (② removed all V→V barriers) ---
                    # NOTE: sink_val hoisted out of for-loop. When `by` carries
                    # a T.if_then_else condval (from task_range), codegen would
                    # otherwise inline the condval declaration into the
                    # scores_sum.SetValue() parameter position (invalid C++).
                    sink_val = Sinks[by]
                    for h_i in range(_half_M):
                        scores_sum[h_i, 0] = sink_val

                    T.copy(scores_max, scores_max_prev)

                    T.tile.max(scores_max, scores_max, scores_sum)

                    T.tile.sub(scores_max_prev, scores_max_prev, scores_max)
                    T.tile.exp(scores_max_prev, scores_max_prev)

                    T.tile.mul(logsum, logsum, scores_max_prev)
                    T.tile.broadcast(brd_dim, scores_max_prev)
                    T.tile.mul(acc_o, acc_o, brd_dim)

                    T.tile.sub(scores_sum, scores_sum, scores_max)
                    T.tile.exp(scores_sum, scores_sum)
                    T.tile.add(logsum, logsum, scores_sum)

                    # --- Zero OOB rows (this vid's _half_M rows) ---
                    oob_start = T.max(0, q_seqlen - bx * _BM)
                    for h_i in range(_half_M):
                        if h_i + vid * _half_M >= oob_start:
                            T.tile.fill(acc_o[h_i, :], 0.0)
                    # Step 4: deleted barrier_all — fill is V op, broadcast
                    # below is also V; V pipeline is in-order, no sync needed.

                    # --- Normalize ---
                    T.tile.broadcast(brd_dim, logsum)
                    # Step 4: deleted barrier_all — broadcast & div both V ops
                    T.tile.div(acc_o, acc_o, brd_dim)
                    # Step 4: deleted barrier_all — div & cast both V ops

                    # --- Write output (this vid's _half_M rows) ---
                    # Step 4: SIG_O_HALF orders V write (cast) vs MTE3 read
                    # (->Output) of acc_o_half (replaces barrier_all). acc_o_half
                    # has independent UB address, safe across tiles. fa_opt
                    # keeps a barrier here (L327); we go further with flag sync.
                    T.wait_flag("MTE3", "V", SIG_O_HALF)
                    T.tile.cast(acc_o_half, acc_o, "CAST_NONE", _half_M * dim)
                    T.set_flag("V", "MTE3", SIG_O_HALF)
                    T.wait_flag("V", "MTE3", SIG_O_HALF)
                    T.copy(
                        acc_o_half,
                        Output[bz, by, bx * _BM + vid * _half_M : bx * _BM + vid * _half_M + _half_M, :],
                    )
                    T.set_flag("MTE3", "V", SIG_O_HALF)

                # Step 4 destroy: consume outstanding init-direction flags
                # (fa_opt L332-334) — balances the init set_flags at scope
                # start. Each flag: init set 1 + per-iter set/wait pairs;
                # destroy wait 1 consumes the last per-iter set (or the init
                # set when my_count=0, keeping counts balanced & no deadlock).
                T.wait_flag("V", "MTE2", SIG_IO_UB)
                T.wait_flag("V", "MTE2", SIG_MASK_UB)
                T.wait_flag("MTE3", "V", SIG_S_HALF)
                T.wait_flag("V", "MTE2", SIG_O_UB)
                T.wait_flag("MTE3", "V", SIG_O_HALF)

    return main


# ========== Golden Reference ==========


def ref_program(
    q_unpad: torch.Tensor,
    k_unpad: torch.Tensor,
    v_unpad: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    sinks: torch.Tensor,
    batch_size: int,
    is_causal: bool,
    sliding_window: Optional[int] = None,
    groups: int = 1,
) -> torch.Tensor:
    """Reference implementation for varlen GQA attention with sinks."""
    total_q, num_heads, head_dim = q_unpad.shape
    _, num_key_value_heads, _ = k_unpad.shape

    sm_scale = 1.0 / head_dim**0.5
    output = torch.zeros_like(q_unpad)

    for b in range(batch_size):
        q_start = cu_seqlens_q[b].item()
        q_end = cu_seqlens_q[b + 1].item()
        k_start = cu_seqlens_k[b].item()
        k_end = cu_seqlens_k[b + 1].item()

        q_len = q_end - q_start
        k_len = k_end - k_start

        if q_len == 0:
            continue

        q_seq = q_unpad[q_start:q_end]
        k_seq = k_unpad[k_start:k_end]
        v_seq = v_unpad[k_start:k_end]

        q_seq = q_seq.view(q_len, num_key_value_heads, groups, head_dim)
        sinks_expanded = sinks.view(num_key_value_heads, groups, 1, 1).float()

        k_seq = k_seq.unsqueeze(2)
        v_seq = v_seq.unsqueeze(2)

        logits = torch.einsum("qhgd,khgd->hgqk", q_seq.float(), k_seq.float()) * sm_scale

        start_q = k_len - q_len
        pos_keys = torch.arange(k_len, device=q_unpad.device)
        pos_queries = torch.arange(q_len, device=q_unpad.device) + start_q

        if is_causal:
            mask = pos_keys[None, :] > pos_queries[:, None]
            mask = mask.float().masked_fill(mask, float("-inf"))
        else:
            mask = torch.zeros(q_len, k_len, device=q_unpad.device)

        if sliding_window is not None:
            too_old = pos_keys[None, :] < (pos_queries[:, None] - sliding_window + 1)
            mask.masked_fill_(too_old, float("-inf"))

        logits = logits + mask[None, None, :, :]

        logits_max = torch.max(logits, dim=-1, keepdim=True).values
        logits_or_sinks_max = torch.maximum(sinks_expanded, logits_max)
        sinks_exp = torch.exp(sinks_expanded - logits_or_sinks_max)
        unnormalized_scores = torch.exp(logits - logits_or_sinks_max)
        normalizer = unnormalized_scores.sum(dim=-1, keepdim=True) + sinks_exp
        scores = unnormalized_scores / normalizer

        out = torch.einsum("hgqk,khgd->qhgd", scores, v_seq.float())
        out = out.reshape(q_len, num_heads, head_dim).to(q_unpad.dtype)
        output[q_start:q_end] = out

    return output


# ========== Mask Construction ==========


def make_attention_mask(
    batch_size,
    max_seqlen_q,
    max_seqlen_k,
    q_seqlens,
    kv_seqlens,
    is_causal,
    block_M,
    block_N,
    device,
):
    """Pre-compute tiled attention mask [total_tiles, block_M, block_N] in float16."""
    # ③ Output dtype changed from float32 to float16
    NEG_INF = -(2**30)
    m_blocks = (max_seqlen_q + block_M - 1) // block_M
    n_blocks = (max_seqlen_k + block_N - 1) // block_N
    total_tiles = batch_size * m_blocks * n_blocks

    mask_full = torch.zeros(batch_size, max_seqlen_q, max_seqlen_k, dtype=torch.float32, device=device)
    i_idx = torch.arange(max_seqlen_q, device=device).unsqueeze(1)
    j_idx = torch.arange(max_seqlen_k, device=device).unsqueeze(0)

    for b in range(batch_size):
        q_len = q_seqlens[b].item()
        kv_len = kv_seqlens[b].item()
        offset = kv_len - q_len

        invalid = (i_idx >= q_len) | (j_idx >= kv_len)

        if is_causal:
            invalid = invalid | (j_idx > (i_idx + offset))

        mask_full[b] = invalid.float() * NEG_INF

    # Build in fp32 first, then convert to fp16
    mask_tiled_fp32 = torch.full((total_tiles, block_M, block_N), NEG_INF, dtype=torch.float32, device=device)
    for b in range(batch_size):
        for mb in range(m_blocks):
            for nb in range(n_blocks):
                tile_idx = b * m_blocks * n_blocks + mb * n_blocks + nb
                i_start = mb * block_M
                j_start = nb * block_N
                i_end = min(i_start + block_M, max_seqlen_q)
                j_end = min(j_start + block_N, max_seqlen_k)
                if i_end > i_start and j_end > j_start:
                    mask_tiled_fp32[tile_idx, : i_end - i_start, : j_end - j_start] = mask_full[b, i_start:i_end, j_start:j_end]

    # ③ Convert to fp16: -(2**30) saturates to -inf in fp16
    # This is correct for attention masking: exp(-inf) = 0 after softmax
    mask_tiled = mask_tiled_fp32.to(torch.float16)

    return mask_tiled, total_tiles


# ========== Test Helpers ==========


def make_varlen_data(batch, q_seqlen, k_seqlen, heads, head_kv, dim, dtype, device="npu"):
    """Construct uniform varlen data (all batches same length)."""
    UQ = batch * q_seqlen
    UKV = batch * k_seqlen
    Q = torch.randn(UQ, heads, dim, dtype=dtype, device=device)
    K = torch.randn(UKV, head_kv, dim, dtype=dtype, device=device)
    V = torch.randn(UKV, head_kv, dim, dtype=dtype, device=device)
    cu_seqlens_q = torch.arange(0, UQ + 1, q_seqlen, dtype=torch.int32, device=device)
    cu_seqlens_k = torch.arange(0, UKV + 1, k_seqlen, dtype=torch.int32, device=device)
    sinks = torch.randn(heads, dtype=dtype, device=device)
    return Q, K, V, cu_seqlens_q, cu_seqlens_k, sinks


def varlen_to_padded(q_unpad, cu_seqlens, max_seqlen, heads, dim):
    """Convert varlen [UQ, H, D] to padded [B, H, max_seqlen, D]."""
    batch = len(cu_seqlens) - 1
    padded = torch.zeros(batch, heads, max_seqlen, dim, dtype=q_unpad.dtype, device=q_unpad.device)
    for b in range(batch):
        s = cu_seqlens[b].item()
        e = cu_seqlens[b + 1].item()
        length = e - s
        padded[b, :, :length, :] = q_unpad[s:e].permute(1, 0, 2)
    return padded


def padded_to_varlen(padded, cu_seqlens, heads, dim):
    """Convert padded [B, H, max_seqlen, D] back to varlen [UQ, H, D]."""
    batch = padded.shape[0]
    parts = []
    for b in range(batch):
        s = cu_seqlens[b].item()
        e = cu_seqlens[b + 1].item()
        length = e - s
        parts.append(padded[b, :, :length, :].permute(1, 0, 2))
    return torch.cat(parts, dim=0)


# ========== Smoke Test (CI entry — prints "Test Passed!") ==========


if __name__ == "__main__":
    tilelang.disable_cache()
    torch.set_default_device("npu")
    torch.manual_seed(0)

    # Minimal L0 config (block-aligned, small)
    batch, heads, groups, q_seqlen, k_seqlen, dim = 2, 4, 4, 128, 128, 128
    is_causal = True
    block_M, block_N = 64, 128
    core_num, num_stages = 20, 14
    dtype = torch.float16
    head_kv = heads // groups

    Q_3d, K_3d, V_3d, cu_q, cu_k, sinks = make_varlen_data(batch, q_seqlen, k_seqlen, heads, head_kv, dim, dtype)
    Q_4d = varlen_to_padded(Q_3d, cu_q, q_seqlen, heads, dim)
    K_4d = varlen_to_padded(K_3d, cu_k, k_seqlen, head_kv, dim)
    V_4d = varlen_to_padded(V_3d, cu_k, k_seqlen, head_kv, dim)

    q_seqlens = torch.full([batch], q_seqlen, dtype=torch.int32, device="npu")
    kv_seqlens = torch.full([batch], k_seqlen, dtype=torch.int32, device="npu")

    mask_tiled, total_tiles = make_attention_mask(
        batch,
        q_seqlen,
        k_seqlen,
        q_seqlens,
        kv_seqlens,
        is_causal,
        block_M,
        block_N,
        "npu",
    )

    kernel = gqa_sink_fwd(
        batch,
        groups,
        heads,
        dim,
        q_seqlen,
        k_seqlen,
        is_causal,
        mask_tiles=total_tiles,
        block_M=block_M,
        block_N=block_N,
        num_stages=num_stages,
        core_num=core_num,
    )

    ws1 = torch.zeros(core_num, num_stages, block_M, block_N, dtype=torch.float32, device="npu")
    ws2 = torch.zeros(core_num, num_stages, block_M, block_N, dtype=torch.float16, device="npu")
    ws3 = torch.zeros(core_num, num_stages, block_M, dim, dtype=torch.float32, device="npu")

    out_4d = kernel(
        Q_4d,
        K_4d,
        V_4d,
        sinks,
        q_seqlens,
        kv_seqlens,
        k_seqlen,
        mask_tiled,
        None,
        ws1,
        ws2,
        ws3,
    )
    out_3d = padded_to_varlen(out_4d, cu_q, heads, dim)

    ref_out = ref_program(
        Q_3d,
        K_3d,
        V_3d,
        cu_q,
        cu_k,
        q_seqlen,
        k_seqlen,
        sinks,
        batch,
        is_causal,
        groups=groups,
    )

    max_diff = (out_3d.cpu().float() - ref_out.cpu().float()).abs().max().item()
    atol, rtol = 1e-2, 1e-2
    torch.testing.assert_close(out_3d.cpu().float(), ref_out.cpu().float(), rtol=rtol, atol=atol)
    print(f"max_diff={max_diff:.6f}")
    print("Test Passed!")
