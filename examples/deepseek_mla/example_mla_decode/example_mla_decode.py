"""MLA Decode (DeepSeek Multi-head Latent Attention Decode) for Ascend NPU.

Dual-loop structure + persistent grid in Developer mode (pass_configs 4x True).

Key design:
  1. Persistent grid: T.Kernel(core_num=20) + T.serial(waves) outer loop.
     Each core processes its tiles independently — no cross-block C/V dependency
     (unlike grid-stride, which Bug 7.2.2 blocks). workspace 46MB → 7.2MB (L2-resident).
  2. Dual-loop: Cube waves loop (batch GEMM1 + GEMM2) + Vector waves loop
     (batch softmax + O accumulate). combineCV auto-separates C/V, no T.Scope needed.
  3. Manual multi-buffer (T.serial + batch_iters) replaces T.Pipelined.
     num_stages=2 (Bug 7.2.4 limits T.Pipelined to ≤2 with online softmax).
  4. kv_l1 [num_stages, BLOCK_N, dim] double-buffer: GEMM1 loads slot i,
     GEMM2 reuses slot i — eliminates KV reload between batches.

Constraints: Developer mode, zero T.barrier_all / T.sync_grid / manual sync primitives.
Tail mask: col_indices + T.tile.compare + T.tile.select for non-aligned seqlen_kv.

Input value constraints:
  - Input magnitude should satisfy |x| <= 100 for fp16 stability.
  - Larger magnitudes (e.g. ±65504 fp16 max) cause Q@KV^T intermediate to overflow
    fp16 range (65504^2 ≈ 4.3e9 >> 65504), leading to precision degradation
    (matched_ratio can drop to <10%). This is an inherent fp16 attention limitation,
    not a kernel bug.
  - For extreme input ranges, consider pre-scaling Q/K_pe by 1/sqrt(D+Dpe) before
    calling this kernel.
"""

import tilelang
import torch
import torch.nn.functional as F
from tilelang import language as T

tilelang.disable_cache()

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
}

BLOCK_H = 64
BLOCK_N = 128
CORE_NUM = 20  # Ascend910B3 cube core count


@tilelang.jit(out_idx=[5], workspace_idx=[6, 7, 8], pass_configs=pass_configs)
def mla_decode(batch, heads, kv_head_num, seqlen_kv, dim, pe_dim, actual_seqlen_kv=-1, core_num=CORE_NUM):
    """MLA Decode kernel (dual-loop + persistent grid, Developer mode).

    Args:
        batch: batch size.
        heads: query head number (must be multiple of BLOCK_H=64).
        kv_head_num: KV head number (must be 1).
        seqlen_kv: KV sequence length (padded to BLOCK_N=128 multiple).
        dim: latent dimension (512).
        pe_dim: position encoding dimension (64).
        actual_seqlen_kv: actual KV length for tail masking; -1 = no mask.
        core_num: number of AI Cube cores (20 for Ascend910B3).

    Returns:
        prim_func main(Q, Q_pe, KV, K_pe, col_indices, Output, ws1, ws2, ws3)
    """
    # P1-1: batch=0 leads to block_num=head_blocks*batch=0, so the Cube loop runs
    # 0 times and L1 buffers (q_l1 etc.) get no alloc node, triggering a
    # compile-time InternalError ("Cannot find pre-allocated address for buffer").
    # Reject early with a clear message.
    assert batch >= 1, f"batch ({batch}) must be >= 1"
    assert kv_head_num == 1, "kv_head_num must be 1"
    assert heads % BLOCK_H == 0, f"heads ({heads}) must be a multiple of {BLOCK_H}"
    assert heads >= BLOCK_H, f"heads ({heads}) must be >= {BLOCK_H}"

    dtype = "float16"
    accum_dtype = "float"
    sm_scale = (1.0 / (dim + pe_dim)) ** 0.5

    # P2-1: Only -1 is the valid sentinel for "no mask"; other negative values
    # (e.g. -100) are ambiguous and almost certainly caller bugs. Previously all
    # negatives silently fell through to the default path. Reject explicitly.
    if actual_seqlen_kv < 0:
        assert actual_seqlen_kv == -1, f"actual_seqlen_kv must be -1 (no mask) or >= 1, got {actual_seqlen_kv}"
        actual_seqlen_kv = seqlen_kv
    assert seqlen_kv % BLOCK_N == 0, "seqlen_kv must be padded to BLOCK_N multiple"
    assert actual_seqlen_kv <= seqlen_kv, "actual_seqlen_kv must be <= padded seqlen_kv"
    assert actual_seqlen_kv >= 1, "actual_seqlen_kv must be >= 1 (all-masked yields NaN)"

    head_blocks = heads // BLOCK_H
    block_num = head_blocks * batch
    half_M = BLOCK_H // 2
    waves = T.ceildiv(block_num, core_num)

    # num_stages: 2 for aligned (double buffer), 1 for tail-mask (Bug 7.2.4: ≤2)
    num_stages = 1 if actual_seqlen_kv < seqlen_kv else 2
    loop_range = T.ceildiv(seqlen_kv, BLOCK_N)
    num_outer = T.ceildiv(loop_range, num_stages)

    @T.prim_func
    def main(
        Q: T.Tensor([batch, heads, dim], dtype),  # type: ignore
        Q_pe: T.Tensor([batch, heads, pe_dim], dtype),  # type: ignore
        KV: T.Tensor([batch, seqlen_kv, kv_head_num, dim], dtype),  # type: ignore
        K_pe: T.Tensor([batch, seqlen_kv, kv_head_num, pe_dim], dtype),  # type: ignore
        col_indices: T.Tensor([BLOCK_N], accum_dtype),  # type: ignore
        Output: T.Tensor([batch, heads, dim], dtype),  # type: ignore
        # ws1: S scores (Q@KV^T + Q_pe@K_pe^T, pre-softmax), fp32 for precision
        workspace_1: T.Tensor([core_num, num_stages, BLOCK_H, BLOCK_N], accum_dtype),
        # ws2: P scores (post-softmax), fp16 for GEMM2 input
        workspace_2: T.Tensor([core_num, num_stages, BLOCK_H, BLOCK_N], dtype),
        # ws3: O partial (P@V per-iter), fp32 (GM↔UB cross-dtype not supported, FB5)
        workspace_3: T.Tensor([core_num, num_stages, BLOCK_H, dim], accum_dtype),
    ):
        with T.Kernel(core_num, is_npu=True) as (cid, vid):
            # ---- Cube L1 buffers (persistent across waves) ----
            q_l1 = T.alloc_L1([BLOCK_H, dim], dtype)
            q_pe_l1 = T.alloc_L1([BLOCK_H, pe_dim], dtype)
            # kv_l1 [num_stages, ...] double-buffer: GEMM1 and GEMM2 share same slot,
            # eliminating KV reload between batches.
            # k_pe_l1 [num_stages, ...] double-buffer: lets GEMM1[i+1] K_pe load overlap
            # with GEMM1[i] GEMM1b (MTE1/Cube pipeline), +16KB over single buffer.
            # L1 total: 376KB < 512KB (q 64 + q_pe 8 + kv 256 + k_pe 32 + acc_s 16)
            kv_l1 = T.alloc_L1([num_stages, BLOCK_N, dim], dtype)
            k_pe_l1 = T.alloc_L1([num_stages, BLOCK_N, pe_dim], dtype)
            acc_s_l1 = T.alloc_L1([BLOCK_H, BLOCK_N], dtype)
            # L0C: single-buffer (MEMORY_PLANNING reuses acc_s_l0c/acc_o_l0c addresses)
            acc_s_l0c = T.alloc_L0C([BLOCK_H, BLOCK_N], accum_dtype)
            acc_o_l0c = T.alloc_L0C([BLOCK_H, dim], accum_dtype)

            # ---- Vector UB buffers (persistent across waves) ----
            acc_o = T.alloc_ub([half_M, dim], accum_dtype)
            sumexp = T.alloc_ub([half_M, 1], accum_dtype)
            m_i = T.alloc_ub([half_M, 1], accum_dtype)
            m_i_prev = T.alloc_ub([half_M, 1], accum_dtype)
            m_i_2d = T.alloc_ub([half_M, BLOCK_N], accum_dtype)
            acc_s_ub = T.alloc_ub([half_M, BLOCK_N], accum_dtype)
            sumexp_i_ub = T.alloc_ub([half_M, 1], accum_dtype)
            acc_s_half = T.alloc_ub([half_M, BLOCK_N], dtype)
            acc_o_ub = T.alloc_ub([half_M, dim], accum_dtype)
            acc_o_half = T.alloc_ub([half_M, dim], dtype)
            col_indices_ub = T.alloc_ub([BLOCK_N], accum_dtype)
            mask_ub = T.alloc_ub([BLOCK_N], accum_dtype)
            # Multi-buffer pipeline state (cross-batch online softmax)
            r_factors = T.alloc_ub([num_stages, half_M, 1], accum_dtype)
            sumexp_is = T.alloc_ub([num_stages, half_M, 1], accum_dtype)

            v_row = vid * half_M

            # Load col_indices once per core (persistent)
            T.copy(col_indices[:], col_indices_ub)

            # ================================================================
            # Cube waves loop: batch GEMM1 (S = Q@KV^T + Q_pe@K_pe^T) + batch GEMM2 (O = P@KV)
            # combineCV auto-separates L1/GEMM ops to Cube core.
            # AUTO_CV_SYNC auto-inserts sync at workspace read/write points.
            # ================================================================
            for w in T.serial(waves):
                tile_id = core_num * w + cid
                if tile_id < block_num:
                    hid = tile_id % head_blocks
                    bid = tile_id // head_blocks

                    T.copy(Q[bid, hid * BLOCK_H : (hid + 1) * BLOCK_H, :], q_l1)
                    T.copy(Q_pe[bid, hid * BLOCK_H : (hid + 1) * BLOCK_H, :], q_pe_l1)

                    for k_outer in T.serial(num_outer):
                        _remaining = loop_range - k_outer * num_stages
                        batch_iters = T.if_then_else(_remaining < num_stages, _remaining, num_stages)

                        # GEMM1 batch: load KV + K_pe, compute S = Q@KV^T + Q_pe@K_pe^T → ws1
                        # K_pe load hoisted before GEMM1a so both MTE3 loads issue consecutively
                        for i in T.serial(batch_iters):
                            kv_start = (k_outer * num_stages + i) * BLOCK_N
                            T.copy(KV[bid, kv_start : kv_start + BLOCK_N, 0, :], kv_l1[i, :, :])
                            T.copy(K_pe[bid, kv_start : kv_start + BLOCK_N, 0, :], k_pe_l1[i, :, :])
                            T.gemm_v0(q_l1, kv_l1[i, :, :], acc_s_l0c, transpose_B=True, init=True)
                            T.gemm_v0(q_pe_l1, k_pe_l1[i, :, :], acc_s_l0c, transpose_B=True)
                            T.copy(acc_s_l0c, workspace_1[cid, i, :, :])

                        # GEMM2 batch: O = P@KV → ws3 (reuse kv_l1[i], no reload)
                        for i in T.serial(batch_iters):
                            T.copy(workspace_2[cid, i, :, :], acc_s_l1)
                            T.gemm_v0(acc_s_l1, kv_l1[i, :, :], acc_o_l0c, init=True, kL0Size=16)
                            T.copy(acc_o_l0c, workspace_3[cid, i, :, :])

            # ================================================================
            # Vector waves loop: online softmax + O accumulate
            # combineCV auto-separates UB/tile ops to Vector core.
            # ================================================================
            for w in T.serial(waves):
                tile_id = core_num * w + cid
                if tile_id < block_num:
                    hid = tile_id % head_blocks
                    bid = tile_id // head_blocks

                    # Init online softmax state (per tile)
                    T.tile.fill(acc_o, 0.0)
                    T.tile.fill(sumexp, 0.0)
                    T.tile.fill(m_i, 2**30)  # positive init for negative-domain min-merge

                    for k_outer in T.serial(num_outer):
                        _remaining = loop_range - k_outer * num_stages
                        batch_iters = T.if_then_else(_remaining < num_stages, _remaining, num_stages)

                        # Softmax batch: S → P → ws2
                        for i in T.serial(batch_iters):
                            kv_start = (k_outer * num_stages + i) * BLOCK_N
                            T.copy(m_i, m_i_prev)
                            T.copy(workspace_1[cid, i, v_row : v_row + half_M, :], acc_s_ub)

                            # Tail mask for non-aligned seqlen_kv
                            if actual_seqlen_kv < seqlen_kv:
                                remaining = actual_seqlen_kv - kv_start
                                T.tile.compare(mask_ub, col_indices_ub, remaining, "LT")
                                for h_i in T.serial(half_M):
                                    T.tile.select(
                                        acc_s_ub[h_i, :],
                                        mask_ub,
                                        acc_s_ub[h_i, :],
                                        -T.infinity(accum_dtype),
                                        "VSEL_TENSOR_SCALAR_MODE",
                                    )

                            # Online softmax (negative-domain min-merge, ref flash_attn h16_d128)
                            # m_i = -scale*max (negative), min-merge with prev
                            T.reduce_max(acc_s_ub, m_i, dim=-1)
                            T.tile.mul(m_i, m_i, -sm_scale)
                            T.tile.min(m_i, m_i, m_i_prev)
                            # r_factor = cur - prev (sub swap: 1 step replaces sub+mul)
                            T.tile.sub(m_i_prev, m_i, m_i_prev)
                            T.copy(m_i_prev, r_factors[i, :, :])

                            # axpy: acc_s = scale*acc_s + (-scale*max) = scale*(acc_s - max)
                            T.tile.broadcast(m_i_2d, m_i)
                            T.tile.axpy(m_i_2d, acc_s_ub, sm_scale)
                            T.tile.exp(acc_s_ub, m_i_2d)

                            T.reduce_sum(acc_s_ub, sumexp_i_ub, dim=-1)
                            T.copy(sumexp_i_ub, sumexp_is[i, :, :])
                            T.copy(acc_s_ub, acc_s_half)
                            T.copy(acc_s_half, workspace_2[cid, i, v_row : v_row + half_M, :])

                        # O accumulate batch: rescale acc_o by r, add O_partial
                        for i in T.serial(batch_iters):
                            T.copy(r_factors[i, :, :], m_i_prev)
                            T.tile.exp(m_i_prev, m_i_prev)
                            T.tile.mul(sumexp, sumexp, m_i_prev)
                            T.copy(sumexp_is[i, :, :], sumexp_i_ub)
                            T.tile.add(sumexp, sumexp, sumexp_i_ub)
                            T.tile.broadcast(acc_o_ub, m_i_prev)
                            T.tile.mul(acc_o, acc_o, acc_o_ub)
                            T.copy(workspace_3[cid, i, v_row : v_row + half_M, :], acc_o_ub)
                            T.tile.add(acc_o, acc_o, acc_o_ub)

                    # Final normalize: acc_o /= sumexp, write Output (fp16)
                    T.tile.broadcast(acc_o_ub, sumexp)
                    T.tile.div(acc_o, acc_o, acc_o_ub)
                    T.copy(acc_o, acc_o_half)
                    T.copy(
                        acc_o_half,
                        Output[bid, hid * BLOCK_H + v_row : hid * BLOCK_H + v_row + half_M, :],
                    )

    return main


def ref_mla_decode(q, q_pe, kv, k_pe):
    """PyTorch golden reference (kv_head_num=1).

    Inputs: q [B,H,D], q_pe [B,H,pe], kv [B,N,1,D], k_pe [B,N,1,pe]
    Output: [B, H, D]
    """
    assert kv.shape[2] == 1, f"golden expects kv_head_num=1, got kv.shape={kv.shape}"
    # P2-2: Detect dim/shape mismatch between tensors early. Without these, a
    # caller passing dim=256 Q into a dim=512 kernel silently produces corrupted
    # output (tilelang JIT does not cross-check tensor shapes against the scalar
    # `dim`/`pe_dim` kernel args).
    assert q.shape[-1] == kv.shape[-1], f"dim mismatch: q.shape[-1]={q.shape[-1]} != kv.shape[-1]={kv.shape[-1]}"
    assert q_pe.shape[-1] == k_pe.shape[-1], f"pe_dim mismatch: q_pe.shape[-1]={q_pe.shape[-1]} != k_pe.shape[-1]={k_pe.shape[-1]}"
    assert q.shape[0] == kv.shape[0], f"batch mismatch: q.shape[0]={q.shape[0]} != kv.shape[0]={kv.shape[0]}"
    assert q.shape[1] == q_pe.shape[1], f"heads mismatch: q.shape[1]={q.shape[1]} != q_pe.shape[1]={q_pe.shape[1]}"
    dim = q.shape[-1]
    pe_dim = q_pe.shape[-1]
    scale = (dim + pe_dim) ** 0.5

    kv_2d = kv.squeeze(2)
    k_pe_2d = k_pe.squeeze(2)
    scores = torch.matmul(q.float(), kv_2d.float().transpose(1, 2))
    scores = scores + torch.matmul(q_pe.float(), k_pe_2d.float().transpose(1, 2))
    scores = scores / scale
    attention = F.softmax(scores, dim=-1)
    out = torch.matmul(attention, kv_2d.float())
    return out.to(q.dtype)


if __name__ == "__main__":
    # Smoke test (CI entry, prints "Test Passed!")
    torch.set_default_device("npu")
    torch.manual_seed(0)

    B, H, N, D, Dpe = 1, 64, 128, 512, 64
    q = torch.randn(B, H, D, dtype=torch.float16)
    q_pe = torch.randn(B, H, Dpe, dtype=torch.float16)
    kv = torch.randn(B, N, 1, D, dtype=torch.float16)
    k_pe = torch.randn(B, N, 1, Dpe, dtype=torch.float16)
    col_indices = torch.arange(BLOCK_N, dtype=torch.float32)
    # P1-2: col_indices length must match BLOCK_N. tilelang JIT does NOT always
    # validate the runtime tensor shape against the declared T.Tensor([BLOCK_N]),
    # so an undersized col_indices (e.g. arange(64) for BLOCK_N=128) silently
    # produces wrong output (observed max_diff up to 0.707). Guard at the caller.
    assert col_indices.shape[0] == BLOCK_N, f"col_indices length must be {BLOCK_N}, got {col_indices.shape[0]}"

    kernel = mla_decode(B, H, 1, N, D, Dpe)
    output = kernel(q, q_pe, kv, k_pe, col_indices)
    torch.npu.synchronize()

    ref = ref_mla_decode(q, q_pe, kv, k_pe)
    max_diff = (output.float() - ref.float()).abs().max().item()
    assert max_diff < 1e-1, f"Precision check failed: max_diff={max_diff}"
    print("Test Passed!")
