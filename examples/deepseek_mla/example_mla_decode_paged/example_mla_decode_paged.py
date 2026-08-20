"""MLA Decode Paged Attention kernel for Ascend NPU.

DeepSeek Multi-head Latent Attention decode stage with paged KV cache support.
Fuses two attention score paths (Q·KV^T + Q_pe·K_pe^T), online softmax, then P·KV.

Programming mode: Developer (no T.Scope, no cross_flag, no barrier_all).
    pass_configs: AUTO_CV_COMBINE/SYNC/AUTO_SYNC/MEMORY_PLANNING all True.
    combineCV pass auto-separates Cube/Vector code and auto-inserts sync.

Key optimizations (cumulative -68.3% from 16528us baseline):
    1. U-2 grid: 2 hids per tile (hid_tiles=2), halves KV GM traffic and grid dispatch.
       acc_o[2,16,512]fp32=64KB fits in UB (196KB) — no GM swap (key advantage over U-1).
    2. Multi-buffer pipeline (num_stages=4, batch GEMM1 + batch GEMM3 for C/V overlap).
    3. Direct paged GM read in Cube (block_table scalar read + KV[kv_start:] copy).
    4. block_N=256 + kL0Size=64, block_H=32, Q pre-multiply scale, workspace_3 fp16.
    5. ZN/NZ layout for L1 buffers, head-major persistent grid.

Performance: ~5240 us (gap 1.73x vs GPU target 3036 us).
Precision: fp16 mixed tolerance (atol=2^-14, rtol=2^-9, max_abs=0.1, ratio>=0.99).

For the layered test suite (L0/L1/L2/Boundary + bench + msprof), see
``test_mla_decode_paged.py`` in the same directory.
"""

import torch
import tilelang
from tilelang import language as T
from tilelang.intrinsics import make_zn_layout, make_nz_layout

# Developer pass_configs: combineCV auto-separates C/V, AUTO_CV_SYNC auto-inserts sync.
# No manual T.Scope, no manual cross_flag — framework handles Cube/Vector division.
_developer_pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}


@tilelang.jit(out_idx=[5], workspace_idx=[6, 7, 8], pass_configs=_developer_pass_configs)
def mla_decode_tilelang(
    batch,
    h_q,
    h_kv,
    max_seqlen_pad,
    dv,
    dpe,
    block_N,
    block_H,
    block_size,
    cache_seqlen,
    core_num,
    softmax_scale=None,
):
    """MLA decode paged attention kernel (Developer mode, U-2 grid).

    Args:
        batch:          batch size
        h_q:            number of query heads
        h_kv:           number of KV heads (must be 1)
        max_seqlen_pad: padded max sequence length (multiple of block_N)
        dv:             value head dim (nope part)
        dpe:            RoPE head dim (pe part)
        block_N:        KV block size for attention computation
        block_H:        query head block size (32 to fit L0C)
        block_size:     paged KV block size (>= block_N, multiple of block_N)
        cache_seqlen:   KV cache sequence length (compile-time param)
        core_num:       number of AI Cube cores (20 for Ascend 910B3)
        softmax_scale:  d^-0.5 (applied at host side via Q pre-multiply)

    Kernel structure:
        Cube code: batch GEMM1 (Q@KV^T + Q_pe@K_pe^T → ws1) + batch GEMM3 (P@KV → ws3).
            KV read directly from GM via paged block_table addressing.
        Vector code: online softmax (ws1 → ws2) + O accumulate (ws3 → Output).
        AUTO_CV_SYNC auto-inserts cross-core sync at workspace read/write points.
    """
    if softmax_scale is None:
        softmax_scale = (dv + dpe) ** -0.5
    dtype = "float16"
    accum_dtype = "float"
    kv_group_num = h_q // h_kv
    VALID_BLOCK_H = min(block_H, kv_group_num)
    num_h_blocks = h_q // VALID_BLOCK_H
    num_blocks_per_batch = max_seqlen_pad // block_size
    # Static loop bound: ceildiv to include the partial tail block.
    loop_range = (cache_seqlen + block_N - 1) // block_N
    needs_tail_mask = (cache_seqlen % block_N) != 0
    # tail_valid: valid token count in the tail block; only consumed inside the
    # `if needs_tail_mask` branch (range(tail_valid, block_N) mask loop).
    # Kept as a compile-time constant so TIR codegen can specialize the mask range.
    tail_valid = cache_seqlen - (loop_range - 1) * block_N if loop_range > 0 else 0
    # NEG_INF for tail masking: -(2**30) is a finite fp32 value that acts as
    # -inf for softmax (exp(-1e9 - m_i) = 0) while avoiding NaN in arithmetic
    # (e.g. r_factors = m_i_prev - m_i: -(2**30) - (-(2**30)) = 0, not NaN).
    # Using float("-inf") causes TIR codegen failure for tail-masked shapes.
    NEG_INF = -(2.0**30)
    assert h_kv == 1, "h_kv must be 1"
    assert block_size >= block_N and block_size % block_N == 0
    assert max_seqlen_pad % block_N == 0
    assert cache_seqlen >= 1, "cache_seqlen must be >= 1"
    assert cache_seqlen <= max_seqlen_pad, f"cache_seqlen ({cache_seqlen}) must <= max_seqlen_pad ({max_seqlen_pad})"

    # U-2 grid: process 2 hid_blocks per tile (hid loop inside tile).
    # Reduces total_tiles 512→256, waves 26→13. KV loaded once per i-iter, reused for 2 hids.
    # acc_o[2,16,512]fp32=64KB fits in UB (196KB) — no GM swap (unlike U-1's 128KB overflow).
    hid_tiles = min(2, num_h_blocks)
    assert num_h_blocks % hid_tiles == 0, f"num_h_blocks ({num_h_blocks}) must be divisible by hid_tiles ({hid_tiles})"
    total_tiles = batch * (num_h_blocks // hid_tiles)
    waves = T.ceildiv(total_tiles, core_num)
    hm = block_H // 2  # V core processes half rows (vid split)

    # Multi-buffer pipeline: num_stages=4 is optimal (tested 3/5, both worse).
    num_stages = 4
    num_outer = T.ceildiv(loop_range, num_stages)
    ws_slots = num_stages * hid_tiles  # combined slot index: i * hid_tiles + hid_local

    @T.prim_func
    def main_no_split(
        Q: T.Tensor([batch, h_q, dv], dtype),
        Q_pe: T.Tensor([batch, h_q, dpe], dtype),
        KV: T.Tensor([batch * max_seqlen_pad, h_kv, dv], dtype),
        K_pe: T.Tensor([batch * max_seqlen_pad, h_kv, dpe], dtype),
        block_table: T.Tensor([batch, num_blocks_per_batch], "int32"),
        Output: T.Tensor([batch, h_q, dv], dtype),
        # workspace_1: S scores (fp32 — D-VALRANGE-L test produces ~241M, overflows fp16).
        workspace_1: T.Tensor([core_num, ws_slots, block_H, block_N], accum_dtype),
        # workspace_2: P matrix (fp16 — softmax output, read by Cube GEMM3).
        workspace_2: T.Tensor([core_num, ws_slots, block_H, block_N], dtype),
        # workspace_3: O partial (fp16 — P@V single-iter, halves GM traffic vs fp32).
        workspace_3: T.Tensor([core_num, ws_slots, block_H, dv], dtype),
    ):
        with T.Kernel(core_num, is_npu=True) as (cid, vid):
            # ---- L1 buffers (Cube core, persistent across waves) ----
            # U-2: 3D Q buffers for 2 hids (loaded once per tile, reused across k_outer).
            # KV/K_pe: single (h_kv=1, shared across hids — loaded once per i iter).
            q_all_l1 = T.alloc_L1([hid_tiles, block_H, dv], dtype)  # 64KB
            q_pe_all_l1 = T.alloc_L1([hid_tiles, block_H, dpe], dtype)  # 8KB
            kv_l1 = T.alloc_L1([block_N, dv], dtype)  # 256KB
            k_pe_l1 = T.alloc_L1([block_N, dpe], dtype)  # 32KB
            acc_s_l1 = T.alloc_L1([block_H, block_N], dtype)  # 16KB (reused per hid)
            acc_s_l0c = T.alloc_L0C([block_H, block_N], accum_dtype)
            acc_o_l0c = T.alloc_L0C([block_H, dv], accum_dtype)
            # L1 total: 376KB < 512KB

            # ZN/NZ layout (kv_l1 excluded: shared by GEMM1 NZ and GEMM3 ZN).
            T.annotate_layout(
                {
                    q_all_l1: make_zn_layout(q_all_l1),
                    q_pe_all_l1: make_zn_layout(q_pe_all_l1),
                    k_pe_l1: make_nz_layout(k_pe_l1),
                    acc_s_l1: make_zn_layout(acc_s_l1),
                }
            )

            # ---- UB buffers (Vector core, hm rows per vid) ----
            # U-2: per-hid acc_o state in UB (NO GM swap — key advantage over U-1).
            acc_o_all = T.alloc_ub([hid_tiles, hm, dv], accum_dtype)  # 64KB
            acc_o = T.alloc_ub([hm, dv], accum_dtype)  # 32KB (2D temp for T.tile ops)
            logsum = T.alloc_ub([hm], accum_dtype)  # 1D temp
            m_i = T.alloc_ub([hm], accum_dtype)  # 1D temp
            m_i_prev = T.alloc_ub([hm], accum_dtype)  # 1D temp
            acc_s_ub = T.alloc_ub([hm, block_N], accum_dtype)  # 16KB
            acc_s_ub_ = T.alloc_ub([hm, block_N], accum_dtype)  # 16KB (broadcast target)
            acc_s_half = T.alloc_ub([hm, block_N], dtype)  # 8KB
            acc_o_ub = T.alloc_ub([hm, dv], accum_dtype)  # 32KB (broadcast target)
            acc_o_half = T.alloc_ub([hm, dv], dtype)  # 16KB (ws3 staging + output cast)

            # Per-hid softmax state (UB arrays — copy to 1D temp before T.tile ops).
            m_i_all = T.alloc_ub([hid_tiles, hm], accum_dtype)  # 128B
            logsum_all = T.alloc_ub([hid_tiles, hm], accum_dtype)  # 128B

            # Multi-buffer pipeline buffers (per-iter, per-hid).
            r_factors = T.alloc_ub([num_stages, hid_tiles, hm], accum_dtype)  # 512B
            sumexp_is = T.alloc_ub([num_stages, hid_tiles, hm], accum_dtype)  # 512B
            # UB total: ~184KB < 196KB

            v_row = vid * hm

            # ================================================================
            # Cube code: batch GEMM1 + batch GEMM3 (combineCV auto-separates).
            # Direct paged GM read: block_table[bid, bt_idx] (GetValue) +
            # T.copy(KV[kv_start:...], kv_l1) paged GM→L1 copy.
            # ================================================================
            for w in T.serial(waves):
                tile_id = core_num * w + cid
                bid = tile_id // (num_h_blocks // hid_tiles)  # batch
                hid_block = tile_id % (num_h_blocks // hid_tiles)  # 0 or 1

                if bid < batch:
                    cur_kv_head = 0  # h_kv=1: all heads share the same KV head

                    # Load Q, Q_pe for 2 hids (3D L1, persistent across k_outer).
                    for hid_local in T.serial(hid_tiles):
                        h_start = (hid_block * hid_tiles + hid_local) * VALID_BLOCK_H
                        T.copy(Q[bid, h_start : h_start + block_H, :], q_all_l1[hid_local, :, :])
                        T.copy(Q_pe[bid, h_start : h_start + block_H, :], q_pe_all_l1[hid_local, :, :])

                    for k_outer in T.serial(num_outer):
                        _remaining = loop_range - k_outer * num_stages
                        batch_iters = T.if_then_else(_remaining < num_stages, _remaining, num_stages)

                        # --- GEMM1 batch: produce S scores → ws1 ---
                        # hid loop inside i loop: KV loaded once per i, reused for 2 hids.
                        for i in T.serial(batch_iters):
                            k_idx = k_outer * num_stages + i
                            bt_idx = (k_idx * block_N) // block_size
                            kv_start = block_table[bid, bt_idx] * block_size + (k_idx * block_N) % block_size

                            # Load KV, K_pe for this batch iter (paged GM → L1).
                            T.copy(KV[kv_start : kv_start + block_N, cur_kv_head, :], kv_l1)
                            T.copy(K_pe[kv_start : kv_start + block_N, cur_kv_head, :], k_pe_l1)

                            slot_base = i * hid_tiles
                            for hid_local in T.serial(hid_tiles):
                                slot = slot_base + hid_local
                                # GEMM1a (init=True) + GEMM1b (init=False, accumulate).
                                T.gemm_v0(
                                    q_all_l1[hid_local, :, :],
                                    kv_l1,
                                    acc_s_l0c,
                                    transpose_B=True,
                                    init=True,
                                    kL0Size=64,
                                )
                                T.gemm_v0(
                                    q_pe_all_l1[hid_local, :, :],
                                    k_pe_l1,
                                    acc_s_l0c,
                                    transpose_B=True,
                                    init=False,
                                )
                                # L0C → workspace_1 (FIX pipe, direct L0C→GM).
                                T.copy(acc_s_l0c, workspace_1[cid, slot, :, :])

                        # --- GEMM3 batch: consume P from ws2 → produce O → ws3 ---
                        # RELOAD kv_l1 (overwritten in GEMM1 batch — L2 cache absorbs).
                        for i in T.serial(batch_iters):
                            k_idx = k_outer * num_stages + i
                            bt_idx = (k_idx * block_N) // block_size
                            kv_start = block_table[bid, bt_idx] * block_size + (k_idx * block_N) % block_size
                            T.copy(KV[kv_start : kv_start + block_N, cur_kv_head, :], kv_l1)

                            slot_base = i * hid_tiles
                            for hid_local in T.serial(hid_tiles):
                                slot = slot_base + hid_local
                                T.copy(workspace_2[cid, slot, :, :], acc_s_l1)
                                T.gemm_v0(acc_s_l1, kv_l1, acc_o_l0c, init=True, kL0Size=64)
                                # L0C(fp32) → workspace_3(fp16) (FIX pipe, auto cast).
                                T.copy(acc_o_l0c, workspace_3[cid, slot, :, :])

            # ================================================================
            # Vector code: online softmax + O accumulate (combineCV auto-separates).
            # AUTO_CV_SYNC waits for Cube's workspace_1/3 to be filled.
            # ================================================================
            for w in T.serial(waves):
                tile_id = core_num * w + cid
                bid = tile_id // (num_h_blocks // hid_tiles)
                hid_block = tile_id % (num_h_blocks // hid_tiles)

                if bid < batch:
                    # Init per-hid softmax state + acc_o_all.
                    for hid_local in T.serial(hid_tiles):
                        T.tile.fill(acc_o, 0.0)
                        T.copy(acc_o, acc_o_all[hid_local, :, :])
                        T.tile.fill(m_i, -(2.0**30))
                        T.copy(m_i, m_i_all[hid_local, :])
                        T.tile.fill(logsum, 0.0)
                        T.copy(logsum, logsum_all[hid_local, :])

                    for k_outer in T.serial(num_outer):
                        _remaining = loop_range - k_outer * num_stages
                        batch_iters = T.if_then_else(_remaining < num_stages, _remaining, num_stages)

                        # --- softmax batch: read S → produce P → ws2 ---
                        for i in T.serial(batch_iters):
                            slot_base = i * hid_tiles
                            for hid_local in T.serial(hid_tiles):
                                slot = slot_base + hid_local

                                # Load per-hid state to 1D temps.
                                T.copy(m_i_all[hid_local, :], m_i)
                                T.copy(m_i, m_i_prev)

                                # Read S[i, hid] — Q pre-mul scale: scale in Q, no axpy.
                                T.copy(workspace_1[cid, slot, v_row : v_row + hm, :], acc_s_ub)

                                # Tail block masking (compile-time gated).
                                if needs_tail_mask:
                                    k_idx = k_outer * num_stages + i
                                    if k_idx * block_N + block_N > cache_seqlen:
                                        for h_i in range(hm):
                                            for j in range(tail_valid, block_N):
                                                acc_s_ub[h_i, j] = NEG_INF

                                # Online softmax: max → rescale → exp → sum.
                                T.reduce_max(acc_s_ub, m_i, dim=-1)
                                T.tile.max(m_i, m_i, m_i_prev)
                                T.tile.sub(m_i_prev, m_i_prev, m_i)
                                T.copy(m_i_prev, r_factors[i, hid_local, :])

                                # P = exp(S - m_new).
                                T.tile.broadcast(acc_s_ub_, m_i, axis=1)
                                T.tile.sub(acc_s_ub, acc_s_ub, acc_s_ub_)
                                T.tile.exp(acc_s_ub, acc_s_ub)

                                # sumexp_is[i, hid] = sum(P[i, hid]).
                                T.reduce_sum(acc_s_ub, m_i_prev, dim=-1)
                                T.copy(m_i_prev, sumexp_is[i, hid_local, :])

                                # P → workspace_2 (fp16 cast for Cube GEMM3).
                                T.copy(acc_s_ub, acc_s_half)
                                T.copy(acc_s_half, workspace_2[cid, slot, v_row : v_row + hm, :])

                                # Write back m_i (updated by reduce_max).
                                T.copy(m_i, m_i_all[hid_local, :])

                        # --- O accumulate batch: apply r_factors, read O_partial → acc_o ---
                        for i in T.serial(batch_iters):
                            slot_base = i * hid_tiles
                            for hid_local in T.serial(hid_tiles):
                                slot = slot_base + hid_local

                                # Load acc_o and logsum from per-hid UB state.
                                T.copy(acc_o_all[hid_local, :, :], acc_o)
                                T.copy(logsum_all[hid_local, :], logsum)

                                # Apply r_factors[i, hid]: acc_o *= exp(r), logsum *= exp(r).
                                T.copy(r_factors[i, hid_local, :], m_i_prev)
                                T.tile.exp(m_i_prev, m_i_prev)
                                T.tile.mul(logsum, logsum, m_i_prev)
                                T.tile.broadcast(acc_o_ub, m_i_prev, axis=1)
                                T.tile.mul(acc_o, acc_o, acc_o_ub)

                                # logsum += sumexp_is[i, hid].
                                T.copy(sumexp_is[i, hid_local, :], m_i_prev)
                                T.tile.add(logsum, logsum, m_i_prev)

                                # Read O_partial[i, hid], accumulate.
                                # workspace_3 fp16 → UB fp16 (acc_o_half) → UB fp32 (acc_o_ub).
                                T.copy(workspace_3[cid, slot, v_row : v_row + hm, :], acc_o_half)
                                T.copy(acc_o_half, acc_o_ub)
                                T.tile.add(acc_o, acc_o, acc_o_ub)

                                # Store acc_o back to per-hid UB state.
                                T.copy(acc_o, acc_o_all[hid_local, :, :])
                                T.copy(logsum, logsum_all[hid_local, :])

                    # ---- Post-loop: per hid normalize + write Output (fp16) ----
                    for hid_local in T.serial(hid_tiles):
                        global_hid = hid_block * hid_tiles + hid_local
                        h_start = global_hid * VALID_BLOCK_H

                        T.copy(acc_o_all[hid_local, :, :], acc_o)
                        T.copy(logsum_all[hid_local, :], logsum)

                        # Normalize: acc_o /= logsum.
                        # Clamp logsum to minimum 1e-30 to prevent division by zero
                        # (defensive: logsum > 0 in practice since cache_seqlen >= 1
                        # guarantees at least 1 finite score, but clamp protects
                        # against edge cases with all-masked rows).
                        T.tile.max(logsum, logsum, 1e-30)
                        T.tile.broadcast(acc_o_ub, logsum, axis=1)
                        T.tile.div(acc_o, acc_o, acc_o_ub)

                        # Cast fp32 → fp16 and write Output.
                        T.copy(acc_o, acc_o_half)
                        T.copy(
                            acc_o_half,
                            Output[bid, h_start + v_row : h_start + v_row + hm, :],
                        )

    return main_no_split


# ---------------------------------------------------------------------------
# Golden reference (CPU, fp32 computation)
# ---------------------------------------------------------------------------


def golden_mla_decode_paged(
    Q,
    Q_pe,
    KV,
    K_pe,
    block_table,
    cache_seqlens,
    batch,
    h_q,
    h_kv,
    dv,
    dpe,
    block_size,
    max_seqlen_pad,
    softmax_scale=None,
):
    """PyTorch reference implementation (CPU, fp32).

    Gathers paged KV per batch, computes scaled dot-product attention with
    Q_nope @ KV^T + Q_pe @ K_pe^T, softmax, then P @ KV.
    """
    if softmax_scale is None:
        scale = (dv + dpe) ** -0.5
    else:
        scale = float(softmax_scale)
    output = torch.zeros(batch, h_q, dv, dtype=torch.float32)

    for b in range(batch):
        seq_len = int(cache_seqlens[b])
        num_blocks = (seq_len + block_size - 1) // block_size

        # Gather KV blocks for this batch's sequence.
        kv_seq = torch.zeros(seq_len, dv, dtype=torch.float32)
        kpe_seq = torch.zeros(seq_len, dpe, dtype=torch.float32)
        for blk in range(num_blocks):
            phys_block = int(block_table[b, blk])
            kv_start = phys_block * block_size
            valid_n = min(block_size, seq_len - blk * block_size)
            kv_seq[blk * block_size : blk * block_size + valid_n] = KV[kv_start : kv_start + valid_n, 0, :].float()
            kpe_seq[blk * block_size : blk * block_size + valid_n] = K_pe[kv_start : kv_start + valid_n, 0, :].float()

        for h in range(h_q):
            q_nope = Q[b, h, :].float()
            q_pe = Q_pe[b, h, :].float()

            # Attention scores: Q @ KV^T + Q_pe @ K_pe^T.
            s = q_nope @ kv_seq.T + q_pe @ kpe_seq.T
            s = s * scale
            p = torch.softmax(s, dim=-1)
            output[b, h, :] = p @ kv_seq

    return output.to(torch.float16)


# ---------------------------------------------------------------------------
# Smoke test (CI entry — prints "Test Passed!")
# ---------------------------------------------------------------------------


def _smoke_test():
    """Minimal L0 smoke test for CI bench_test.sh entry.

    Runs the kernel on a small aligned configuration and checks precision
    against the PyTorch golden. Prints "Test Passed!" on success.
    """
    BLOCK_N = 256
    BLOCK_H = 32
    CORE_NUM = 20

    batch, h_q, h_kv = 1, 128, 1
    cache_seqlen, d, dv, block_size = 256, 576, 512, 256
    dpe = d - dv

    torch.manual_seed(42)
    Q_full = torch.randn(batch, h_q, d, dtype=torch.float16)
    Q = Q_full[..., :dv].contiguous()
    Q_pe = Q_full[..., dv:].contiguous()

    # Pre-multiply Q by softmax_scale (fuses scale into Q at host side).
    pre_scale = d**-0.5
    Q = (Q * pre_scale).contiguous()
    Q_pe = (Q_pe * pre_scale).contiguous()

    max_seqlen_pad = cache_seqlen
    num_blocks_per_batch = max_seqlen_pad // block_size
    blocked_k = torch.randn(batch * num_blocks_per_batch, block_size, h_kv, d, dtype=torch.float16)
    KV = blocked_k[..., :dv].reshape(-1, h_kv, dv).contiguous()
    K_pe = blocked_k[..., dv:].reshape(-1, h_kv, dpe).contiguous()
    block_table = torch.arange(batch * num_blocks_per_batch, dtype=torch.int32).reshape(batch, num_blocks_per_batch)

    # Defensive: validate block_table values are in valid range (host-side only).
    assert block_table.min() >= 0, "block_table contains negative index"
    assert block_table.max() < batch * num_blocks_per_batch, (
        f"block_table max {int(block_table.max())} exceeds KV pool size {batch * num_blocks_per_batch} blocks"
    )

    # Golden (CPU, fp32) — Q is pre-scaled, so golden must NOT apply scale again.
    cache_seqlens = torch.tensor([cache_seqlen] * batch, dtype=torch.int32)
    golden_out = golden_mla_decode_paged(
        Q,
        Q_pe,
        KV,
        K_pe,
        block_table,
        cache_seqlens,
        batch,
        h_q,
        h_kv,
        dv,
        dpe,
        block_size,
        max_seqlen_pad,
        softmax_scale=1.0,
    )

    # Move to NPU and run kernel.
    Q_npu = Q.npu()
    Q_pe_npu = Q_pe.npu()
    KV_npu = KV.npu()
    K_pe_npu = K_pe.npu()
    block_table_npu = block_table.npu()

    softmax_scale = d**-0.5
    kernel = mla_decode_tilelang(
        batch,
        h_q,
        h_kv,
        max_seqlen_pad,
        dv,
        dpe,
        BLOCK_N,
        BLOCK_H,
        block_size,
        cache_seqlen,
        CORE_NUM,
        softmax_scale,
    )
    out = kernel(Q_npu, Q_pe_npu, KV_npu, K_pe_npu, block_table_npu)
    out_cpu = out.cpu()

    # Precision check (fp16 mixed tolerance: atol=2^-14, rtol=2^-9, max_abs=0.1, ratio>=0.99).
    atol = 2.0**-14
    rtol = 2.0**-9
    max_abs_limit = 1e-1
    required_ratio = 0.99

    diff = (out_cpu.float() - golden_out.float()).abs()
    tolerance = atol + rtol * golden_out.float().abs()
    matched_ratio = (diff <= tolerance).float().mean().item()
    max_abs_error = diff.max().item()

    passed = matched_ratio >= required_ratio and max_abs_error <= max_abs_limit
    if passed:
        print(f"[PRECISION_PASS] smoke test matched_ratio={matched_ratio:.6f} max_abs_error={max_abs_error:.6e}")
        print("Test Passed!")
    else:
        print(f"[PRECISION_FAIL] smoke test matched_ratio={matched_ratio:.6f} max_abs_error={max_abs_error:.6e}")
        raise AssertionError(
            f"Smoke test failed: matched_ratio={matched_ratio:.6f} < {required_ratio} "
            f"or max_abs_error={max_abs_error:.6e} > {max_abs_limit}"
        )


if __name__ == "__main__":
    _smoke_test()
