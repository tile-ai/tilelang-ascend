"""Native Sparse Attention Decode (NSA Decode) — TileLang Ascend kernel.

Developer mode (hybrid): pass_configs four-True (combineCV + auto_cv_sync +
auto_sync + memory_planning), with explicit L1/L0C/UB allocation and workspace
GM relay. No T.barrier_all, no manual T.Scope, no manual set_flag/wait_flag.
"""

import tilelang
import torch
from tilelang import language as T

# Developer mode pass_configs: sync & C/V split fully auto-managed by passes.
PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}


@tilelang.jit(out_idx=[5], workspace_idx=[6, 7, 8, 9, 10], pass_configs=PASS_CONFIGS)
def nsa_decode(batch, seq_len, query_heads, kv_head_num, dim, selected_blocks, block_size):
    """NSA Decode kernel (Developer mode).

    For each (b, h), gathers S selected KV blocks (each BS tokens) via
    host-expanded RowIndices, then runs online softmax attention with
    neg-scaled max + axpy fusion (ref: example_mla_decode.py).

    Args (JIT compile-time constants):
        batch, seq_len: grid dims (block_num = batch * kv_head_num).
        query_heads: HQ (Q head count, HQ = kv_head_num * G).
        kv_head_num: H (must be 1).
        dim: D (head dimension).
        selected_blocks: S (number of selected KV blocks).
        block_size: BS (tokens per block).

    Tensor args (prim_func):
        Q:           [B, 1, HQ, D]            float16
        K:           [B, seq_len, H, D]       float16
        V:           [B, seq_len, H, D]       float16
        RowIndices:  [B, 1, H, S*BS]          int32   (host: block_idx*BS + [0..BS-1])
        BlockCounts: [B, 1, H]                float32 (valid block count)
        Output:      [B, 1, HQ, D]            float16 (out_idx=5)
        workspace_0: [block_num, S*BS, D]     float16 (gathered K, V→C)
        workspace_1: [block_num, G, S*BS]     float32 (GEMM1 acc_s, C→V)
        workspace_2: [block_num, G, S*BS]     float16 (softmax output, V→C)
        workspace_3: [block_num, G, D]        float32 (GEMM2 acc_o, C→V)
        workspace_4: [block_num, S*BS, D]     float16 (gathered V, V→C)
    """
    assert kv_head_num == 1, "kv_head_num must be 1"
    G = query_heads // kv_head_num
    assert G >= 16, f"G={G} must be >= 16 for L0C fractal"
    assert G % 2 == 0, f"G={G} must be even for vid split"

    dtype = "float16"
    accum_dtype = "float32"
    BS = block_size
    D = dim
    S = selected_blocks
    sm_scale = (1.0 / D) ** 0.5  # D^-0.5 (no log2(e), Bug 7.1.6: ascend_exp2 unavailable)

    head_kv = kv_head_num
    block_num = batch * head_kv
    total_rows = S * BS

    @T.prim_func
    def main(
        Q: T.Tensor([batch, 1, query_heads, dim], dtype),  # type: ignore
        K: T.Tensor([batch, seq_len, kv_head_num, dim], dtype),  # type: ignore
        V: T.Tensor([batch, seq_len, kv_head_num, dim], dtype),  # type: ignore
        RowIndices: T.Tensor([batch, 1, kv_head_num, total_rows], "int32"),  # type: ignore
        BlockCounts: T.Tensor([batch, 1, kv_head_num], accum_dtype),  # type: ignore
        Output: T.Tensor([batch, 1, query_heads, dim], dtype),  # type: ignore
        workspace_0: T.Tensor([block_num, S * BS, D], dtype),  # type: ignore
        workspace_1: T.Tensor([block_num, G, S * BS], accum_dtype),  # type: ignore
        workspace_2: T.Tensor([block_num, G, S * BS], dtype),  # type: ignore
        workspace_3: T.Tensor([block_num, G, D], accum_dtype),  # type: ignore
        workspace_4: T.Tensor([block_num, S * BS, D], dtype),  # type: ignore
    ):
        with T.Kernel(block_num, is_npu=True) as (cid, vid):
            i_b = cid // head_kv
            i_h = cid % head_kv
            g_start = i_h * G

            # Cube: L1 + L0C buffers
            q_l1 = T.alloc_L1([G, D], dtype)
            k_l1 = T.alloc_L1([BS, D], dtype)
            v_l1 = T.alloc_L1([BS, D], dtype)
            acc_s_l1 = T.alloc_L1([G, BS], dtype)
            acc_s_l0c = T.alloc_L0C([G, BS], accum_dtype)
            acc_o_l0c = T.alloc_L0C([G, D], accum_dtype)

            # Vector: UB buffers (vid split: G//2 rows per vid)
            acc_o = T.alloc_ub([G // 2, D], accum_dtype)
            sumexp = T.alloc_ub([G // 2, 1], accum_dtype)
            m_i = T.alloc_ub([G // 2, 1], accum_dtype)
            m_i_2d = T.alloc_ub([G // 2, BS], accum_dtype)
            acc_s_ub = T.alloc_ub([G // 2, BS], accum_dtype)
            m_i_prev = T.alloc_ub([G // 2, 1], accum_dtype)
            sumexp_i_ub = T.alloc_ub([G // 2, 1], accum_dtype)
            acc_s_half = T.alloc_ub([G // 2, BS], dtype)
            acc_o_ub = T.alloc_ub([G // 2, D], accum_dtype)
            acc_o_half = T.alloc_ub([G // 2, D], dtype)
            row_indices_ub = T.alloc_ub([total_rows], "int32")

            # batch gather buffers (accumulate BS//2 rows in UB before writing GM)
            k_rows_ub = T.alloc_ub([BS // 2, D], dtype)
            v_rows_ub = T.alloc_ub([BS // 2, D], dtype)

            # block_counts mask buffers (2D for 256-byte alignment)
            # mask_2d_ub uses uint8 packed bitmask (1 bit per element) per AscendC
            # CompareScalar/Select selMask requirement: uint8_t, not float.
            # Shape [G//2, BS//8] = [8, 4] = 32 bytes (32B aligned).
            block_counts_scalar_ub = T.alloc_ub([1], accum_dtype)
            block_counts_1d_ub = T.alloc_ub([BS], accum_dtype)
            block_counts_2d_ub = T.alloc_ub([G // 2, BS], accum_dtype)
            mask_2d_ub = T.alloc_ub([G // 2, BS // 8], "uint8")
            # Guard mask for block_counts=0 check on acc_o [G//2, D=16]:
            # packed = G//2 * D // 8 = 8 * 2 = 16 bytes, padded to 32B alignment.
            guard_mask_ub = T.alloc_ub([32], "uint8")

            # Q load + init (loop-external)
            T.copy(Q[i_b, 0, g_start : g_start + G, :], q_l1)
            T.tile.fill(acc_o, 0.0)
            T.tile.fill(sumexp, 0.0)
            T.tile.fill(m_i, 2**30)  # neg-scaled -inf (MLA-style min-merge)

            # Main loop: per-block [gather K/V → GEMM1 → softmax → GEMM2 → accumulate]
            for i in T.serial(S):
                # Vector: read block row indices + gather K/V
                T.copy(RowIndices[i_b, 0, i_h, i * BS : i * BS + BS], row_indices_ub)
                g_off = vid * (G // 2)
                bs_off = i * BS + vid * (BS // 2)

                block_start = row_indices_ub[0]
                vid_start = block_start + vid * (BS // 2)
                T.copy(K[i_b, vid_start : vid_start + BS // 2, i_h, :], k_rows_ub)
                T.copy(k_rows_ub, workspace_0[cid, bs_off : bs_off + BS // 2, :])
                T.copy(V[i_b, vid_start : vid_start + BS // 2, i_h, :], v_rows_ub)
                T.copy(v_rows_ub, workspace_4[cid, bs_off : bs_off + BS // 2, :])

                # Cube: GEMM1 Q × K^T → acc_s
                T.copy(workspace_0[cid, i * BS : i * BS + BS, :], k_l1)
                T.gemm_v0(q_l1, k_l1, acc_s_l0c, transpose_B=True, init=True)
                T.copy(acc_s_l0c, workspace_1[cid, :, i * BS : i * BS + BS])

                # Vector: online softmax
                T.copy(workspace_1[cid, g_off : g_off + G // 2, i * BS : i * BS + BS], acc_s_ub)

                # block_counts mask: invalid block (i >= block_counts) → acc_s = -inf
                T.copy(BlockCounts[i_b, 0, i_h : i_h + 1], block_counts_scalar_ub)
                T.tile.broadcast(block_counts_1d_ub, block_counts_scalar_ub)
                T.tile.broadcast(block_counts_2d_ub, block_counts_1d_ub)
                T.tile.sub(block_counts_2d_ub, block_counts_2d_ub, i)
                T.tile.compare(mask_2d_ub, block_counts_2d_ub, 0.0, "GT")
                T.tile.select(acc_s_ub, mask_2d_ub, acc_s_ub, -T.infinity(accum_dtype), "VSEL_TENSOR_SCALAR_MODE")

                # Online softmax (neg-scaled max + axpy fusion)
                T.copy(m_i, m_i_prev)
                T.reduce_max(acc_s_ub, m_i, dim=-1)
                T.tile.mul(m_i, m_i, -sm_scale)
                T.tile.min(m_i, m_i, m_i_prev)
                T.tile.sub(m_i_prev, m_i, m_i_prev)  # r = exp(m_i - m_i_old)
                T.tile.exp(m_i_prev, m_i_prev)
                T.tile.broadcast(m_i_2d, m_i)
                # axpy: dst = scalar * src + dst → m_i_2d = scale*(acc_s - max)
                T.tile.axpy(m_i_2d, acc_s_ub, sm_scale)
                T.tile.exp(acc_s_ub, m_i_2d)  # softmax numerator
                T.reduce_sum(acc_s_ub, sumexp_i_ub, dim=-1)
                T.tile.mul(sumexp, sumexp, m_i_prev)  # sumexp *= r
                T.tile.add(sumexp, sumexp, sumexp_i_ub)  # sumexp += sum(softmax_i)

                # Write softmax result to workspace_2 for Cube GEMM2
                T.copy(acc_s_ub, acc_s_half)  # fp32 → fp16 cast
                T.copy(acc_s_half, workspace_2[cid, g_off : g_off + G // 2, i * BS : i * BS + BS])

                # Vector: rescale acc_o by r (first iter: r≈0, no-op)
                T.tile.broadcast(acc_o_ub, m_i_prev)
                T.tile.mul(acc_o, acc_o, acc_o_ub)

                # Cube: GEMM2 softmax(acc_s) × V → acc_o
                T.copy(workspace_2[cid, :, i * BS : i * BS + BS], acc_s_l1)
                T.copy(workspace_4[cid, i * BS : i * BS + BS, :], v_l1)
                T.gemm_v0(acc_s_l1, v_l1, acc_o_l0c, init=True)
                T.copy(acc_o_l0c, workspace_3[cid, :, :])

                # Vector: accumulate acc_o
                T.copy(workspace_3[cid, g_off : g_off + G // 2, :], acc_o_ub)
                T.tile.add(acc_o, acc_o, acc_o_ub)

            # Final normalization: acc_o /= sumexp
            T.tile.broadcast(acc_o_ub, sumexp)
            T.tile.div(acc_o, acc_o, acc_o_ub)

            # block_counts=0 guard: bc=0 → output 0 (not NaN)
            # Use block_counts > 0 (not sumexp > 0) to avoid swallowing inf/nan NaN
            T.tile.broadcast(sumexp, block_counts_scalar_ub)
            T.tile.broadcast(acc_o_ub, sumexp)
            T.tile.compare(guard_mask_ub, acc_o_ub, 0.0, "GT")
            T.tile.select(acc_o, guard_mask_ub, acc_o, 0.0, "VSEL_TENSOR_SCALAR_MODE")

            # Output: fp32 → fp16 → GM
            T.copy(acc_o, acc_o_half)
            T.copy(
                acc_o_half,
                Output[i_b, 0, g_start + vid * G // 2 : g_start + vid * G // 2 + G // 2, :],
            )

    return main


# =============================================================================
# Host-side helpers
# =============================================================================
def expand_block_indices_to_rows(block_indices, block_size, seq_len):
    """Expand BlockIndices [B,1,H,S] to RowIndices [B,1,H,S*BS] int32.

    Complete blocks keep real indices; incomplete/padding blocks set to 0
    (kernel masks via block_counts).
    """
    row_indices = block_indices.unsqueeze(-1) * block_size + torch.arange(block_size)
    row_indices = row_indices.view(*block_indices.shape[:-1], block_indices.shape[-1] * block_size).to(torch.int32)

    block_complete = (block_indices * block_size + block_size) <= seq_len
    complete_mask = block_complete.repeat_interleave(block_size, dim=-1)
    return torch.where(complete_mask, row_indices, torch.zeros_like(row_indices))


def validate_block_indices(block_indices, block_counts, block_size, seq_len):
    """Validate that valid blocks (within block_counts) are complete."""
    B, _, H, S = block_indices.shape
    bc = block_counts.to(torch.long).view(B, H)
    for b in range(B):
        for h in range(H):
            for s in range(bc[b, h].item()):
                idx = block_indices[b, 0, h, s].item()
                assert idx * block_size + block_size <= seq_len, (
                    f"Valid block (b={b}, h={h}, s={s}) idx={idx} is incomplete: "
                    f"idx*BS+BS={idx * block_size + block_size} > seq_len={seq_len}."
                )


def run_kernel(Q, K, V, block_indices, block_counts, block_size, seq_len):
    """Compile kernel + expand RowIndices + run (workspace auto-allocated by framework)."""
    B = Q.shape[0]
    HQ = Q.shape[2]
    H = K.shape[2]
    D = K.shape[-1]
    S = block_indices.shape[-1]

    row_indices = expand_block_indices_to_rows(block_indices, block_size, seq_len)
    block_counts_f32 = block_counts.to(torch.float32)
    validate_block_indices(block_indices, block_counts, block_size, seq_len)

    kernel = nsa_decode(B, seq_len, HQ, H, D, S, block_size)
    q_npu, k_npu, v_npu = Q.npu(), K.npu(), V.npu()
    ri_npu = row_indices.npu()
    bc_npu = block_counts_f32.npu()

    # workspace_idx=[6,7,8,9,10] in @tilelang.jit: framework auto-allocates 5 workspace tensors.
    output = kernel(q_npu, k_npu, v_npu, ri_npu, bc_npu)
    torch.npu.synchronize()
    return output


# =============================================================================
# Smoke test entry (CI compatibility, outputs "Test Passed!")
# =============================================================================
if __name__ == "__main__":
    import sys

    tilelang.disable_cache()
    torch.manual_seed(0)

    # Minimal L0 smoke config.
    # NOTE: use SEQ_LEN (not T) to avoid shadowing tilelang.language as T above.
    B, SEQ_LEN, H, HQ, D, S, BS = 2, 64, 1, 16, 16, 1, 32
    DTYPE = torch.float16
    G = HQ // H

    Q = torch.randn((B, 1, HQ, D), dtype=DTYPE)
    K = torch.randn((B, SEQ_LEN, H, D), dtype=DTYPE)
    V = torch.randn((B, SEQ_LEN, H, D), dtype=DTYPE)

    block_indices = torch.zeros((B, 1, H, S), dtype=torch.long)
    block_counts = torch.full((B, 1, H), S, dtype=torch.long)

    # Inline PyTorch CPU golden (self-contained, no import from test file).
    # scale = D^-0.5 (matches kernel, no log2(e) factor).
    scale = D**-0.5
    k_e = K.repeat_interleave(G, dim=2).float()  # GQA expand: H -> HQ
    v_e = V.repeat_interleave(G, dim=2).float()
    bi_e = block_indices.repeat_interleave(G, dim=2)
    bc_e = block_counts.repeat_interleave(G, dim=2)
    q_f = Q.float()
    # Expand block_idx -> rows [B, 1, HQ, S*BS]
    row_idx = (bi_e.unsqueeze(-1) * BS + torch.arange(BS)).reshape(B, 1, HQ, S * BS)
    row_idx = row_idx.clamp(0, k_e.shape[1] - 1)
    blk_id = torch.arange(S).repeat_interleave(BS)  # Block id per row for masking
    ref_out = torch.zeros_like(q_f)
    for b in range(B):
        q_b = q_f[b, 0] * scale  # [HQ, D]
        for h in range(HQ):
            bc = bc_e[b, 0, h].item()
            if bc <= 0:
                continue
            ri = row_idx[b, 0, h]  # [S*BS]
            k_g = k_e[b, ri, h, :]  # [S*BS, D]
            v_g = v_e[b, ri, h, :]
            attn = q_b[h] @ k_g.T  # [S*BS]
            attn = attn.masked_fill(blk_id >= bc, float("-inf"))
            attn = torch.softmax(attn, dim=0)
            ref_out[b, 0, h] = attn @ v_g
    ref_out = ref_out.to(Q.dtype)

    out = run_kernel(Q, K, V, block_indices, block_counts, BS, SEQ_LEN)

    # Precision check: mixed tolerance dual-gate (precision-standard.md §4.1).
    atol, rtol, max_abs_limit, required_ratio = 2**-14, 2**-9, 1e-1, 0.99
    a = out.detach().cpu().float()
    g = ref_out.detach().float()
    # inf/nan structural compare
    special = ~torch.isfinite(g)
    if special.any() and (
        not torch.equal(torch.isnan(a[special]), torch.isnan(g[special]))
        or not torch.equal(torch.isinf(a[special]), torch.isinf(g[special]))
    ):
        print("inf/nan position mismatch")
        sys.exit(1)
    m = torch.isfinite(g)
    if m.sum().item() == 0:
        ratio, max_abs = 1.0, 0.0
    else:
        abs_err = (a[m] - g[m]).abs()
        ratio = (abs_err <= (atol + rtol * g[m].abs())).float().mean().item()
        max_abs = abs_err.max().item()
    print(f"matched_ratio={ratio:.4f} max_abs={max_abs:.3e}")
    assert ratio >= required_ratio and max_abs <= max_abs_limit
    print("Test Passed!")
