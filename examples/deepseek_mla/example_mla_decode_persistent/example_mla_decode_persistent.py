"""DeepSeek MLA Decode Persistent Attention for Ascend NPU.

Implements Multi-head Latent Attention decode for DeepSeek-V2/V3 on Ascend NPU
using TileLang Developer mode (AUTO_CV_COMBINE + AUTO_CV_SYNC).

Programming mode: Developer (zero T.Scope, zero manual cross_flag, zero barrier_all).
  - combineCV pass auto-separates C/V code
  - framework auto-inserts cross-core sync

Key parameters:
  - q:    [batch, heads, dim]           fp16
  - q_pe: [batch, heads, pe_dim]        fp16  (rotary position embedding)
  - kv:   [batch, seqlen_kv, 1, dim]    fp16  (kv_head_num=1, MQA)
  - k_pe: [batch, seqlen_kv, 1, pe_dim] fp16
  - Output: [batch, heads, dim]         fp16

Golden config: B=128, H=128, S=8192, dim=512, pe_dim=64, block_N=128, block_H=64.

Single unified kernel (flashattn_mla_decode):
  - num_split=1 single-pass (no two-phase fallback)
  - workspace_3 dtype: fp32 (precision-safe for all dim/block_N configs)
  - KV reuse: num_stages=1, kv_l1 loaded in GEMM1 is retained for GEMM3
  - acc_o_half: [hm, dim] fp16, post-loop fp16 output cast only

References:
  - GPU source: tilelang/examples/deepseek_mla/example_mla_decode_persistent.py
  - Multi-buffer pipeline: examples/flash_attention/fa_opt/flash_attn_bhsd_expert_h16_d128.py
  - gemm_v0 init=False accumulate: examples/linear_attention_and_rnn/linear_attention_causal.py
"""

import tilelang
import torch
import torch.nn.functional as F
from tilelang import language as T
from tilelang.intrinsics import make_zn_layout, make_nz_layout

try:
    from einops import rearrange, einsum
except ImportError:
    rearrange = None
    einsum = None

# ============================================================================
# pass_configs — Developer mode (4 keys)
# ============================================================================

# AUTO_CV_COMBINE: combineCV pass auto-separates C/V code (no manual T.Scope).
# AUTO_CV_SYNC: framework auto-inserts cross-core sync (no manual cross_flag).
# AUTO_SYNC: gemm_v0 internal MTE1->M->FIX pipeline sync.
# MEMORY_PLANNING: gemm_v0 internal L0 address assignment.
_pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}

# ============================================================================
# Unified MLA Decode Kernel (Developer mode, persistent kernel)
# ============================================================================


@tilelang.jit(out_idx=[4], workspace_idx=[5, 6, 7, 8], pass_configs=_pass_configs)
def flashattn_mla_decode(batch, heads, seqlen_kv, dim, pe_dim, block_N, block_H, core_num):
    """MLA Decode Persistent Attention (single unified kernel).

    Single-pass flash attention with KV reuse and fp32 workspace_3.

    Key design decisions:
    - workspace_3 dtype: fp32 (precision-safe for all dim/block_N configs, no host
      dispatch needed). L0C(fp32) -> GM(fp32) direct copy (no auto-cast).
    - num_stages: 1 (no multi-buffer pipeline; batch_iters=1 always). This enables
      KV reuse: kv_l1 loaded in GEMM1 is retained for GEMM3 (no reload).
    - workspace slots: num_stages * hid_tiles (= 1 * 2 = 2 for golden config).
    - acc_o_half: [hm, dim] fp16 (32KB), used for post-loop fp32->fp16 output cast
      only. V scope ws3 read goes directly GM(fp32) -> UB(fp32 acc_o_ub).

    Safety: num_stages=1 means batch_iters=1 always, so kv_l1 is NOT overwritten
    between GEMM1 and GEMM3 within the same outer iter. KV reuse is safe.

    KV persistent: tile alloc is bid only (hid loops inside tile), KV loaded once
    per k iter, reused across hid_tiles heads. acc_o GM swap for per-hid state
    persistence.

    Developer mode: AUTO_CV_COMBINE + AUTO_CV_SYNC. No T.Scope, no manual
    cross_flag. ZN/NZ layout, V scope broadcast, axpy fusion.

    Outputs: Output [batch, heads, dim] fp16 (direct, no glse/Output_partial).
    """
    scale = (1.0 / (dim + pe_dim)) ** 0.5
    dtype = "float16"
    accum_dtype = "float"
    VALID_BLOCK_H = min(block_H, heads)
    hid_tiles = heads // VALID_BLOCK_H  # 128/64=2 (golden), 64/64=1 (small configs)
    total_tiles = batch  # no hid decomposition (hid loops inside tile)
    waves = T.ceildiv(total_tiles, core_num)
    hm = block_H // 2  # V core processes half rows

    # num_stages=1 (no multi-buffer). KV reuse: kv_l1 loaded in GEMM1 is retained for GEMM3.
    num_stages = 1
    loop_range = seqlen_kv // block_N
    num_outer = T.ceildiv(loop_range, num_stages)
    ws_slots = num_stages * hid_tiles  # 2 — combined slot index for workspace

    @T.prim_func
    def main(
        Q: T.Tensor([batch, heads, dim], dtype),
        Q_pe: T.Tensor([batch, heads, pe_dim], dtype),
        KV: T.Tensor([batch, seqlen_kv, dim], dtype),
        K_pe: T.Tensor([batch, seqlen_kv, pe_dim], dtype),
        Output: T.Tensor([batch, heads, dim], dtype),  # fp16 direct output
        workspace_1: T.Tensor([core_num, ws_slots, block_H, block_N], accum_dtype),
        workspace_2: T.Tensor([core_num, ws_slots, block_H, block_N], dtype),
        workspace_3: T.Tensor([core_num, ws_slots, block_H, dim], accum_dtype),  # fp32 (precision-safe)
        acc_o_gm: T.Tensor([core_num, hid_tiles, block_H, dim], accum_dtype),
    ):
        with T.Kernel(core_num, is_npu=True) as (cid, vid):
            # ---- L1 buffers (Cube core) — 3D Q for multi-hid, single KV ----
            q_all_l1 = T.alloc_L1([hid_tiles, block_H, dim], dtype)
            q_pe_all_l1 = T.alloc_L1([hid_tiles, block_H, pe_dim], dtype)
            kv_l1 = T.alloc_L1([block_N, dim], dtype)  # single, reused
            k_pe_l1 = T.alloc_L1([block_N, pe_dim], dtype)
            acc_s_l1 = T.alloc_L1([block_H, block_N], dtype)
            acc_s_l0c = T.alloc_L0C([block_H, block_N], accum_dtype)
            acc_o_l0c = T.alloc_L0C([block_H, dim], accum_dtype)

            # ZN/NZ layout for Cube GEMM efficiency
            T.annotate_layout(
                {
                    q_all_l1: make_zn_layout(q_all_l1),
                    q_pe_all_l1: make_zn_layout(q_pe_all_l1),
                    k_pe_l1: make_nz_layout(k_pe_l1),
                    acc_s_l1: make_zn_layout(acc_s_l1),
                }
            )

            # ---- UB buffers (Vector core, hm rows) ----
            acc_o = T.alloc_ub([hm, dim], accum_dtype)  # single, GM swap per hid
            logsum = T.alloc_ub([hm], accum_dtype)  # 1D temp
            m_i = T.alloc_ub([hm], accum_dtype)  # 1D temp
            m_i_prev = T.alloc_ub([hm], accum_dtype)  # 1D temp
            acc_s_ub = T.alloc_ub([hm, block_N], accum_dtype)
            acc_s_ub_ = T.alloc_ub([hm, block_N], accum_dtype)  # dual-use: S read + broadcast target
            acc_s_half = T.alloc_ub([hm, block_N], dtype)
            acc_o_ub = T.alloc_ub([hm, dim], accum_dtype)

            # acc_o_half for post-loop fp16 output cast only.
            # [hm, dim] fp16 (32KB). Liveness: post-loop only (ws3 read goes directly to acc_o_ub).
            acc_o_half = T.alloc_ub([hm, dim], dtype)

            # Per-hid softmax state (UB arrays — small)
            m_i_all = T.alloc_ub([hid_tiles, hm], accum_dtype)
            logsum_all = T.alloc_ub([hid_tiles, hm], accum_dtype)

            # Multi-buffer pipeline buffers — [1, hid_tiles, hm]
            r_factors = T.alloc_ub([num_stages, hid_tiles, hm], accum_dtype)
            sumexp_is = T.alloc_ub([num_stages, hid_tiles, hm], accum_dtype)

            v_row = vid * hm

            # ---- C scope (Cube core): GEMM batch + L0C->GM writes + GM->L1 reads ----
            # CV sync is automatic (AUTO_CV_SYNC=True inserts cross-core flags).
            # workspace_3: L0C(fp32)->GM(fp32) direct copy (no auto-cast).
            # KV reuse: num_stages=1, kv_l1 loaded in GEMM1 is retained for GEMM3 (no reload).

            for w in T.serial(waves):
                tile_id = core_num * w + cid
                bid = tile_id  # no hid decomposition

                if bid < batch:
                    # Load ALL hid Q (3D L1, persistent across k_outer)
                    for hid in T.serial(hid_tiles):
                        h_start = hid * VALID_BLOCK_H
                        T.copy(Q[bid, h_start : h_start + block_H, :], q_all_l1[hid, :, :])
                        T.copy(Q_pe[bid, h_start : h_start + block_H, :], q_pe_all_l1[hid, :, :])

                    for k_outer in T.serial(num_outer):
                        _remaining = loop_range - k_outer * num_stages
                        batch_iters = T.if_then_else(_remaining < num_stages, _remaining, num_stages)

                        # --- GEMM1 batch: produce S -> ws1 ---
                        # num_stages=1 -> batch_iters=1 always.
                        # kv_l1 loaded here is RETAINED for GEMM3 batch (KV reuse).
                        for i in T.serial(batch_iters):
                            k_idx = k_outer * num_stages + i
                            kv_start = k_idx * block_N

                            T.copy(KV[bid, kv_start : kv_start + block_N, :], kv_l1)
                            T.copy(K_pe[bid, kv_start : kv_start + block_N, :], k_pe_l1)

                            slot_base = i * hid_tiles
                            for hid in T.serial(hid_tiles):
                                slot = slot_base + hid
                                T.gemm_v0(q_all_l1[hid, :, :], kv_l1, acc_s_l0c, transpose_B=True, init=True)
                                T.gemm_v0(q_pe_all_l1[hid, :, :], k_pe_l1, acc_s_l0c, transpose_B=True, init=False)
                                T.copy(acc_s_l0c, workspace_1[cid, slot, :, :])

                        # --- GEMM3 batch: consume P from ws2 -> produce O -> ws3 ---
                        # KV REUSE: kv_l1 retains GEMM1 loaded KV (no reload).
                        for i in T.serial(batch_iters):
                            slot_base = i * hid_tiles
                            for hid in T.serial(hid_tiles):
                                slot = slot_base + hid
                                T.copy(workspace_2[cid, slot, :, :], acc_s_l1)
                                T.gemm_v0(acc_s_l1, kv_l1, acc_o_l0c, init=True)
                                # L0C(fp32)->GM(fp32) direct copy
                                T.copy(acc_o_l0c, workspace_3[cid, slot, :, :])

            # ---- V scope (Vector core): softmax + O accumulate + direct Output write ----
            # workspace_3 read: direct GM(fp32)->UB(fp32 acc_o_ub) (no fp16 staging).
            # CV sync is automatic (AUTO_CV_SYNC=True inserts cross-core flags).

            for w in T.serial(waves):
                tile_id = core_num * w + cid
                bid = tile_id  # no hid decomposition

                if bid < batch:
                    # Init per-hid softmax state + acc_o_gm
                    for hid in T.serial(hid_tiles):
                        T.tile.fill(m_i, -(2**30))
                        T.copy(m_i, m_i_all[hid, :])
                        T.tile.fill(logsum, 0.0)
                        T.copy(logsum, logsum_all[hid, :])
                        T.tile.fill(acc_o, 0.0)
                        T.copy(acc_o, acc_o_gm[cid, hid, v_row : v_row + hm, :])

                    for k_outer in T.serial(num_outer):
                        _remaining = loop_range - k_outer * num_stages
                        batch_iters = T.if_then_else(_remaining < num_stages, _remaining, num_stages)

                        # --- softmax batch: read S -> produce P -> ws2 ---
                        for i in T.serial(batch_iters):
                            slot_base = i * hid_tiles
                            for hid in T.serial(hid_tiles):
                                slot = slot_base + hid

                                T.copy(m_i_all[hid, :], m_i)
                                T.copy(m_i, m_i_prev)

                                # Read S[i, hid], scale
                                T.tile.fill(acc_s_ub, 0.0)
                                T.copy(workspace_1[cid, slot, v_row : v_row + hm, :], acc_s_ub_)
                                T.tile.axpy(acc_s_ub, acc_s_ub_, scale)

                                # Online softmax: max -> rescale -> exp -> sum
                                T.reduce_max(acc_s_ub, m_i, dim=-1)
                                T.tile.max(m_i, m_i, m_i_prev)
                                T.tile.sub(m_i_prev, m_i_prev, m_i)
                                T.copy(m_i_prev, r_factors[i, hid, :])

                                # P = exp(S*scale - m_new)
                                T.tile.broadcast(acc_s_ub_, m_i, axis=1)
                                T.tile.sub(acc_s_ub, acc_s_ub, acc_s_ub_)
                                T.tile.exp(acc_s_ub, acc_s_ub)

                                # sumexp_is[i, hid] = sum(P[i, hid])
                                T.reduce_sum(acc_s_ub, m_i_prev, dim=-1)
                                T.copy(m_i_prev, sumexp_is[i, hid, :])

                                # P -> workspace_2[slot] (fp16 cast for Cube GEMM)
                                T.copy(acc_s_ub, acc_s_half)
                                T.copy(acc_s_half, workspace_2[cid, slot, v_row : v_row + hm, :])

                                # Write back m_i to m_i_all (updated by reduce_max)
                                T.copy(m_i, m_i_all[hid, :])

                        # --- O accumulate batch: apply r_factors, read O_partial -> acc_o ---
                        for i in T.serial(batch_iters):
                            slot_base = i * hid_tiles
                            for hid in T.serial(hid_tiles):
                                slot = slot_base + hid

                                # Swap acc_o from GM (per-hid state)
                                T.copy(acc_o_gm[cid, hid, v_row : v_row + hm, :], acc_o)

                                # Load per-hid logsum to 1D temp
                                T.copy(logsum_all[hid, :], logsum)

                                # Apply r_factors[i, hid]: acc_o *= exp(r), logsum *= exp(r)
                                T.copy(r_factors[i, hid, :], m_i_prev)
                                T.tile.exp(m_i_prev, m_i_prev)
                                T.tile.mul(logsum, logsum, m_i_prev)
                                T.tile.broadcast(acc_o_ub, m_i_prev, axis=1)
                                T.tile.mul(acc_o, acc_o, acc_o_ub)

                                # logsum += sumexp_is[i, hid]
                                T.copy(sumexp_is[i, hid, :], m_i_prev)
                                T.tile.add(logsum, logsum, m_i_prev)

                                # workspace_3 fp32: direct GM(fp32)->UB(fp32 acc_o_ub).
                                T.copy(workspace_3[cid, slot, v_row : v_row + hm, :], acc_o_ub)

                                T.tile.add(acc_o, acc_o, acc_o_ub)

                                # Swap acc_o to GM (preserve per-hid state)
                                T.copy(acc_o, acc_o_gm[cid, hid, v_row : v_row + hm, :])

                                # Write back logsum to logsum_all (updated by mul+add)
                                T.copy(logsum, logsum_all[hid, :])

                    # ---- Post-loop: per hid normalize + write Output (fp16) ----
                    for hid in T.serial(hid_tiles):
                        h_start = hid * VALID_BLOCK_H

                        # Swap acc_o from GM (final state for this hid)
                        T.copy(acc_o_gm[cid, hid, v_row : v_row + hm, :], acc_o)
                        # Load per-hid logsum to 1D temp
                        T.copy(logsum_all[hid, :], logsum)

                        # Normalize: acc_o /= logsum
                        T.tile.broadcast(acc_o_ub, logsum, axis=1)
                        T.tile.div(acc_o, acc_o, acc_o_ub)

                        # Write Output (acc_o_half is [hm, dim] fp16 = 32KB, full size)
                        T.copy(acc_o, acc_o_half)  # UB fp32 -> UB fp16 cast
                        T.copy(
                            acc_o_half,
                            Output[bid, h_start + v_row : h_start + v_row + hm, :],
                        )

    return main


# ============================================================================
# Golden Reference (PyTorch, CPU, standard F.softmax — natural exp/log)
# ============================================================================


def ref_mla_decode(q, q_pe, kv, k_pe):
    """PyTorch reference implementation (CPU, standard softmax).

    Uses einops if available; falls back to native torch.bmm otherwise.

    Args:
        q:    [batch, heads, dim]         fp16
        q_pe: [batch, heads, pe_dim]      fp16
        kv:   [batch, seqlen_kv, 1, dim]  fp16 (kv_head_num=1)
        k_pe: [batch, seqlen_kv, 1, pe_dim] fp16
    Returns:
        out:  [batch, heads, dim]         fp16
    """
    dim = q.shape[-1]
    pe_dim = q_pe.shape[-1]
    # NOTE: kernel uses scale = 1/sqrt(dim+pe_dim) with T.tile.axpy (multiplication).
    # Golden uses scale = sqrt(dim+pe_dim) with division. Mathematically equivalent:
    # s * (1/sqrt(d)) == s / sqrt(d). Kept as division for readability (NEW-4).
    scale = (dim + pe_dim) ** 0.5  # sqrt(dim+pe_dim), scores / scale

    if rearrange is not None and einsum is not None:
        kv_head_num = kv.shape[2] if kv.dim() == 4 else 1
        assert kv_head_num == 1, "kv_head_num must be 1"
        num_head_groups = q.shape[1] // kv_head_num

        q_r = rearrange(q, "b (h g) d -> b g h d", g=num_head_groups)
        q_pe_r = rearrange(q_pe, "b (h g) d -> b g h d", g=num_head_groups)
        kv_r = rearrange(kv, "b n h d -> b h n d")
        k_pe_r = rearrange(k_pe, "b n h d -> b h n d")

        query = torch.concat([q_r, q_pe_r], dim=-1)  # [b, g, h, dim+pe_dim]
        key = torch.concat([kv_r, k_pe_r], dim=-1)  # [b, h, s, dim+pe_dim]

        # einops 路径：显式转 fp32 保证与 native 路径精度一致（NEW-1 fix）
        scores = einsum(query.float(), key.float(), "b g h d, b h s d -> b g h s")
        attention = F.softmax(scores / scale, dim=-1)
        out = einsum(attention.float(), kv_r.float(), "b g h s, b h s d -> b g h d")
        out = rearrange(out, "b g h d -> b (h g) d")
        return out.half()

    # Native fallback (no einops dependency)
    B = q.shape[0]
    S = kv.shape[1]
    kv_3d = kv.reshape(B, S, dim)
    k_pe_3d = k_pe.reshape(B, S, pe_dim)

    query = torch.cat([q, q_pe], dim=-1)  # [B, H, dim+pe_dim]
    key = torch.cat([kv_3d, k_pe_3d], dim=-1)  # [B, S, dim+pe_dim]

    scores = torch.bmm(query.float(), key.float().transpose(1, 2))
    attention = F.softmax(scores / scale, dim=-1)
    out = torch.bmm(attention, kv_3d.float())  # [B, H, dim]
    return out.half()


# ============================================================================
# Host-side wrapper: kernel launch
# ============================================================================


def run_mla_decode(q, q_pe, kv, k_pe, block_N, block_H, core_num):
    """Run MLA decode. Host-side: metadata-only view + kernel launch.

    Single unified kernel path (num_split=1, fp32 workspace_3). No host dispatch
    needed — fp32 workspace_3 is precision-safe for all dim/block_N configs.

    Args:
        q:    [batch, heads, dim]         fp16 NPU
        q_pe: [batch, heads, pe_dim]      fp16 NPU
        kv:   [batch, seqlen_kv, 1, dim]  fp16 NPU (kv_head_num=1)
        k_pe: [batch, seqlen_kv, 1, pe_dim] fp16 NPU
        block_N: KV block size (must divide seqlen_kv)
        block_H: head block size (must divide heads)
        core_num: NPU cube core count
    Returns:
        Output: [batch, heads, dim] fp16 NPU
    """
    # ---- P0 input validation (early-fail before any kernel launch) ----
    # C1: dtype check (all tensors must be fp16)
    expected_dtype = torch.float16
    assert q.dtype == expected_dtype, f"q.dtype must be fp16, got {q.dtype}"
    assert q_pe.dtype == expected_dtype, f"q_pe.dtype must be fp16, got {q_pe.dtype}"
    assert kv.dtype == expected_dtype, f"kv.dtype must be fp16, got {kv.dtype}"
    assert k_pe.dtype == expected_dtype, f"k_pe.dtype must be fp16, got {k_pe.dtype}"

    # C2: device check (all tensors must be on NPU)
    assert q.device.type == "npu", f"q must be on NPU, got {q.device}"
    assert q_pe.device.type == "npu", f"q_pe must be on NPU, got {q_pe.device}"
    assert kv.device.type == "npu", f"kv must be on NPU, got {kv.device}"
    assert k_pe.device.type == "npu", f"k_pe must be on NPU, got {k_pe.device}"

    # C3: ndim + shape consistency checks (before batch/heads/etc. are derived)
    # ndim 校验
    assert q.ndim == 3, f"q.ndim must be 3 [batch, heads, dim], got {q.ndim}"
    assert q_pe.ndim == 3, f"q_pe.ndim must be 3 [batch, heads, pe_dim], got {q_pe.ndim}"
    assert kv.ndim == 4, f"kv.ndim must be 4 [batch, seqlen_kv, 1, dim], got {kv.ndim}"
    assert k_pe.ndim == 4, f"k_pe.ndim must be 4 [batch, seqlen_kv, 1, pe_dim], got {k_pe.ndim}"

    # batch 一致
    assert q.shape[0] == q_pe.shape[0] == kv.shape[0] == k_pe.shape[0], (
        f"batch mismatch: q={q.shape[0]}, q_pe={q_pe.shape[0]}, kv={kv.shape[0]}, k_pe={k_pe.shape[0]}"
    )

    # heads 一致
    assert q.shape[1] == q_pe.shape[1], f"heads mismatch: q={q.shape[1]}, q_pe={q_pe.shape[1]}"

    # seqlen_kv 一致
    assert kv.shape[1] == k_pe.shape[1], f"seqlen_kv mismatch: kv={kv.shape[1]}, k_pe={k_pe.shape[1]}"

    # dim 一致（kv dim == q dim, k_pe dim == q_pe dim）
    assert kv.shape[3] == q.shape[2], f"dim mismatch: kv.shape[3]={kv.shape[3]}, q.shape[2]={q.shape[2]}"
    assert k_pe.shape[3] == q_pe.shape[2], f"pe_dim mismatch: k_pe.shape[3]={k_pe.shape[3]}, q_pe.shape[2]={q_pe.shape[2]}"

    # ---- derive shape metadata (only after C1-C3 validated) ----
    batch = q.shape[0]
    heads = q.shape[1]
    seqlen_kv = kv.shape[1]
    dim = q.shape[2]
    pe_dim = q_pe.shape[2]

    # W1: positive value range checks
    assert batch > 0, f"batch must be > 0, got {batch}"
    assert heads > 0, f"heads must be > 0, got {heads}"
    assert seqlen_kv > 0, f"seqlen_kv must be > 0, got {seqlen_kv}"
    assert dim > 0, f"dim must be > 0, got {dim}"
    assert pe_dim > 0, f"pe_dim must be > 0, got {pe_dim}"
    assert block_N > 0, f"block_N must be > 0, got {block_N}"
    assert block_H > 0, f"block_H must be > 0, got {block_H}"
    assert core_num > 0, f"core_num must be > 0, got {core_num}"

    # ---- existing kernel-launch constraints (unchanged) ----
    assert kv.shape[2] == 1, "kv_head_num must be 1"
    assert seqlen_kv >= block_N, f"seqlen_kv ({seqlen_kv}) must be >= block_N ({block_N})"
    assert seqlen_kv % block_N == 0, f"seqlen_kv ({seqlen_kv}) must be divisible by block_N ({block_N})"
    assert heads % block_H == 0, f"heads ({heads}) must be divisible by block_H ({block_H})"
    assert dim % 2 == 0, f"dim ({dim}) must be divisible by 2 (for V core dim split)"

    # NEW-7: Fractal 16-byte alignment (gemm_v0 uses roundUp16 internally, source:
    # src/tl_templates/ascend/common.h:1243). Non-16-aligned M/K causes silent padding
    # that misaligns L0C output with the allocated [block_H, block_N] / [block_H, dim] tiles.
    _FRACTAL_ALIGN = 16
    assert dim % _FRACTAL_ALIGN == 0, f"dim ({dim}) must be aligned to {_FRACTAL_ALIGN} (fractal granularity for gemm_v0)"
    assert pe_dim % _FRACTAL_ALIGN == 0, f"pe_dim ({pe_dim}) must be aligned to {_FRACTAL_ALIGN} (fractal granularity for gemm_v0)"
    assert block_N % _FRACTAL_ALIGN == 0, f"block_N ({block_N}) must be aligned to {_FRACTAL_ALIGN} (fractal granularity for gemm_v0)"
    assert block_H % _FRACTAL_ALIGN == 0, f"block_H ({block_H}) must be aligned to {_FRACTAL_ALIGN} (fractal granularity for gemm_v0)"

    # W2: block_H must be even (V core processes block_H // 2 rows)
    assert block_H % 2 == 0, f"block_H ({block_H}) must be even (V core processes block_H // 2 rows)"

    # W4: L0B/L0A ping-pong slot budget for gemm_v0 transpose_B path (GEMM1a: Q@K^T).
    # Source: src/tl_templates/ascend/common.h:1226 — kL0Budget = 64KB (single K-tile)
    # or 32KB (multi K-tile ping-pong). kL0Size=128 (gemm_v0 default).
    # When dim > 128: kL0split > 1, kL0Budget = 32KB -> block_N <= 128, block_H <= 128.
    # When dim <= 128: kL0split = 1, kL0Budget = 64KB -> block_N <= 256, block_H <= 256.
    _GEMM_KL0SIZE = 128
    _L0_PINGPONG_BUDGET = (32 * 1024) if dim > _GEMM_KL0SIZE else (64 * 1024)
    assert block_N * _GEMM_KL0SIZE * 2 <= _L0_PINGPONG_BUDGET, (
        f"block_N ({block_N}) exceeds L0B budget for dim={dim}: "
        f"block_N*128*2 ({block_N * _GEMM_KL0SIZE * 2}) > {_L0_PINGPONG_BUDGET} bytes; "
        f"reduce block_N (max allowed: {_L0_PINGPONG_BUDGET // (_GEMM_KL0SIZE * 2)})"
    )
    assert block_H * _GEMM_KL0SIZE * 2 <= _L0_PINGPONG_BUDGET, (
        f"block_H ({block_H}) exceeds L0A budget for dim={dim}: "
        f"block_H*128*2 ({block_H * _GEMM_KL0SIZE * 2}) > {_L0_PINGPONG_BUDGET} bytes; "
        f"reduce block_H (max allowed: {_L0_PINGPONG_BUDGET // (_GEMM_KL0SIZE * 2)})"
    )

    # Host-side metadata-only view (squeeze kv_head_num=1, zero-copy)
    # W11: assert contiguous — view() is zero-copy, requires contiguous memory.
    assert kv.is_contiguous(), f"kv must be contiguous for zero-copy view, got stride={kv.stride()}"
    assert k_pe.is_contiguous(), f"k_pe must be contiguous for zero-copy view, got stride={k_pe.stride()}"
    KV_3d = kv.view(batch, seqlen_kv, dim)
    K_pe_3d = k_pe.view(batch, seqlen_kv, pe_dim)

    # Single unified kernel: fp32 workspace_3, num_stages=1, KV reuse.
    phase1_mod = flashattn_mla_decode(batch, heads, seqlen_kv, dim, pe_dim, block_N, block_H, core_num)
    output = phase1_mod(q, q_pe, KV_3d, K_pe_3d)
    return output


if __name__ == "__main__":
    tilelang.disable_cache()
    torch.set_default_device("npu")
    torch.manual_seed(0)

    B, H, S, D, PE = 1, 64, 128, 512, 64
    q = torch.randn(B, H, D, dtype=torch.float16, device="npu")
    q_pe = torch.randn(B, H, PE, dtype=torch.float16, device="npu")
    kv = torch.randn(B, S, 1, D, dtype=torch.float16, device="npu")
    k_pe = torch.randn(B, S, 1, PE, dtype=torch.float16, device="npu")

    # Dynamic core_num detection (NEW-3 fix: was hardcoded 20, breaks on A5/other NPUs)
    core_num = int(torch.npu.get_device_properties("npu").cube_core_num)
    output = run_mla_decode(q, q_pe, kv, k_pe, block_N=128, block_H=64, core_num=core_num)
    torch.npu.synchronize()

    golden = ref_mla_decode(q.cpu(), q_pe.cpu(), kv.cpu(), k_pe.cpu())

    # Precision check: fp16 mixed tolerance dual-threshold
    # atol=2^-14=6.10e-5, rtol=2^-9=1.95e-3, max_abs_limit=1e-1, required_ratio=0.99
    atol, rtol, max_abs_limit, required_ratio = (2**-14, 2**-9, 1e-1, 0.99)
    a = output.detach().cpu().float()
    g = golden.detach().cpu().float()
    abs_err = (a - g).abs()
    matched_ratio = (abs_err <= (atol + rtol * g.abs())).float().mean().item()
    max_abs = abs_err.max().item()
    passed = matched_ratio >= required_ratio and max_abs <= max_abs_limit

    print(f"[PRECISION_{'PASS' if passed else 'FAIL'}] ratio={matched_ratio:.4f} max_abs={max_abs:.6e}")
    assert passed, f"Precision check failed: ratio={matched_ratio:.4f}, max_abs={max_abs:.6e}"
    print("Test Passed!")
