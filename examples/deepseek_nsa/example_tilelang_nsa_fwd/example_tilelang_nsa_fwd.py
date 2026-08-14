"""Native Sparse Attention Forward (NSA Forward) — TileLang Ascend kernel.

Developer mode (hybrid): pass_configs four-True (combineCV + auto_cv_sync +
auto_sync + memory_planning), with explicit L1/L0C/UB allocation, explicit
workspace_idx, T.Pipelined(num_stages=2) and L0C double buffer.
Overlaps Cube/Vector across iterations via auto-inserted CrossCoreFlag (no
T.barrier_all, no manual T.Scope, no manual set_flag/wait_flag).
"""

import tilelang
from tilelang import language as T

# Developer mode pass_configs: sync & C/V split fully auto-managed by passes.
PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,  # auto C/V split
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,  # auto cross-core sync
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,  # auto intra-core sync
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,  # auto L0/L1/UB address
}


@tilelang.jit(out_idx=[4], workspace_idx=[5, 6, 7], pass_configs=PASS_CONFIGS)
def native_sparse_attention(
    batch,
    seq_len,
    head_kv,
    heads,
    dim,
    selected_blocks,
    bs_pad,
    scale,
):
    """NSA Forward kernel (Developer mode hybrid).

    For each (b, t, h), gathers S selected KV blocks (each BS tokens) and runs
    single-pass softmax attention over the selected tokens only.

    Args (JIT compile-time constants):
        batch, seq_len: grid dims (block_num = batch * seq_len * head_kv).
        head_kv: H (KV head count).
        heads: HQ (query head count, HQ = head_kv * G, G must be even for vid split).
        dim: D (head dimension).
        selected_blocks: S (number of selected KV blocks).
        bs_pad: internal tile size (padded from block_size for 256-byte alignment).
        scale: softmax scale factor (None → 1/sqrt(dim); NPU uses T.exp, so scale
            is NOT multiplied by log2(e), unlike GPU T.exp2 path).

    Tensor args (prim_func):
        Q:          [batch, seq_len, heads, dim]                  float16
        K_selected: [batch*seq_len*head_kv, S*bs_pad, dim]        float16 (host pre-gathered)
        V_selected: [batch*seq_len*head_kv, S*bs_pad, dim]        float16 (host pre-gathered)
        CausalMask: [batch*seq_len*head_kv, G, S*bs_pad]          float32 (host precomputed additive: 0.0 visible / -2^30 masked)
        Output:     [batch, seq_len, heads, dim]                  float16 (out_idx=4)
        workspace_1: [core_num, G, KV_LEN]  float32 (acc_s Cube→Vector, workspace_idx=5)
        workspace_2: [core_num, G, KV_LEN]  float16 (acc_s_half Vector→Cube, workspace_idx=6)
        workspace_3: [core_num, G, D]       float32 (acc_o Cube→Vector, workspace_idx=7)
    """
    # scale=None → default to 1/sqrt(dim) (proto.yaml declares scale default=null).
    if scale is None:
        scale = (1.0 / dim) ** 0.5

    G = heads // head_kv
    # vid split: vid=0 covers G heads [0:G//2], vid=1 covers [G//2:G].
    # odd G is rejected (vid split requires even G to cover all heads without gap).
    assert G % 2 == 0, f"G={G} (heads//head_kv) must be even for vid split"
    D = dim
    S = selected_blocks
    KV_LEN = S * bs_pad
    GH = G // 2  # heads per vid

    q_shape = [batch, seq_len, heads, D]
    kv_sel_shape = [batch * seq_len * head_kv, KV_LEN, D]
    mask_shape = [batch * seq_len * head_kv, G, KV_LEN]
    dtype = "float16"
    accum_dtype = "float32"

    # 1D Kernel: each block handles one (i_t, i_b, i_h) combination.
    block_num = seq_len * batch * head_kv
    # Largest divisor of block_num <= 20 (exact division; T.Pipelined needs uniform
    # iteration count across cores). 20 is the physical core count upper bound.
    core_num = min(block_num, 20)
    while core_num > 1 and block_num % core_num != 0:
        core_num -= 1
    if core_num < 4:
        core_num = 1  # single-core fallback for small block_nums (incl. prime and <4)
    single_core_load = block_num // core_num

    # Workspace shapes: [core_num, G, ...] — per-cid; vid splits G dimension.
    ws1_shape = [core_num, G, KV_LEN]
    ws2_shape = [core_num, G, KV_LEN]
    ws3_shape = [core_num, G, D]

    @T.prim_func
    def main(
        Q: T.Tensor(q_shape, dtype),  # type: ignore
        K_selected: T.Tensor(kv_sel_shape, dtype),  # type: ignore
        V_selected: T.Tensor(kv_sel_shape, dtype),  # type: ignore
        CausalMask: T.Tensor(mask_shape, accum_dtype),  # type: ignore
        Output: T.Tensor(q_shape, dtype),  # type: ignore
        workspace_1: T.Tensor(ws1_shape, accum_dtype),  # type: ignore
        workspace_2: T.Tensor(ws2_shape, dtype),  # type: ignore
        workspace_3: T.Tensor(ws3_shape, accum_dtype),  # type: ignore
    ):
        with T.Kernel(core_num, is_npu=True) as (cid, vid):
            # === Cube side L1 buffers (per-cid, reused across iterations) ===
            Q_shared = T.alloc_L1([G, D], dtype)
            K_shared = T.alloc_L1([KV_LEN, D], dtype)
            V_shared = T.alloc_L1([KV_LEN, D], dtype)
            acc_s_l1 = T.alloc_L1([G, KV_LEN], dtype)

            # === Cube side L0C buffers (double-buffered: GEMM[k]→L0C[side_k] overlaps
            # Fixpipe[k-1]→L0C[1-side_k]). init=True on each GEMM below re-initializes
            # L0C (no cross-iteration accumulation), so side reuse (inner%2) is safe. ===
            acc_s_l0c = T.alloc_L0C([2, G, KV_LEN], accum_dtype)
            acc_o_l0c = T.alloc_L0C([2, G, D], accum_dtype)

            # === Vector side UB buffers (per-vid, GH=G//2 heads) ===
            acc_s_ub = T.alloc_ub([GH, KV_LEN], accum_dtype)  # scaled+masked scores
            acc_s_half = T.alloc_ub([GH, KV_LEN], dtype)  # fp16 cast for GEMM2
            acc_o_ub = T.alloc_ub([GH, D], accum_dtype)  # GEMM2 output (fp32)
            acc_o_half = T.alloc_ub([GH, D], dtype)  # fp16 cast for Output
            scores_max = T.alloc_ub([GH], accum_dtype)  # softmax max per head
            scores_sum = T.alloc_ub([GH], accum_dtype)  # softmax sum per head
            mask_ub = T.alloc_ub([GH, KV_LEN], accum_dtype)  # additive causal mask
            scores_max_2d = T.alloc_ub([GH, KV_LEN], accum_dtype)  # broadcast of scores_max
            scores_sum_2d = T.alloc_ub([GH, D], accum_dtype)  # broadcast of scores_sum

            # T.Pipelined(num_stages=2): overlap Cube[k] with Vector[k-1] via L0C
            # double buffer. cross_interval=1 (default): sync every iteration via
            # auto-inserted CrossCoreFlag (no manual barrier_all).
            for inner in T.Pipelined(single_core_load, num_stages=2):
                side = inner % 2
                block_idx = cid * single_core_load + inner
                # Decompose block_idx → (i_b, i_t, i_h). Layout matches K_sel_3d
                # first dim (b * (T*H) + t * H + h) so Q/K/V indexing is consistent.
                i_h = block_idx % head_kv
                i_t = (block_idx // head_kv) % seq_len
                i_b = block_idx // (head_kv * seq_len)

                # --- Cube: Load Q/K/V from GM to L1 ---
                T.copy(Q[i_b, i_t, i_h * G : (i_h + 1) * G, :], Q_shared)
                T.copy(K_selected[block_idx, :, :], K_shared)
                T.copy(V_selected[block_idx, :, :], V_shared)

                # --- Cube: GEMM1 Q @ K^T → L0C[side] (scores) ---
                T.gemm_v0(Q_shared, K_shared, acc_s_l0c[side, :, :], transpose_B=True, init=True)
                # Fixpipe L0C[side] → workspace_1[cid] (Cube→Vector via GM)
                T.copy(acc_s_l0c[side, :, :], workspace_1[cid, :, :])

                # --- Vector: Load mask + read workspace_1 (vid split: GH heads) ---
                T.copy(CausalMask[block_idx, vid * GH : vid * GH + GH, :], mask_ub)
                T.copy(workspace_1[cid, vid * GH : vid * GH + GH, :], acc_s_ub)

                # --- Vector: Scale + mask + single-pass softmax (on [GH, KV_LEN]) ---
                T.tile.mul(acc_s_ub, acc_s_ub, scale)  # scores *= scale
                T.tile.add(acc_s_ub, acc_s_ub, mask_ub)  # scores += mask (-2^30 → ~-inf)
                T.reduce_max(acc_s_ub, scores_max, dim=-1)  # max over KV_LEN
                T.tile.broadcast(scores_max_2d, scores_max, axis=1)  # [GH] → [GH, KV_LEN]
                T.tile.sub(acc_s_ub, acc_s_ub, scores_max_2d)  # scores -= max (numerically stable)
                T.tile.exp(acc_s_ub, acc_s_ub)  # exp(scores - max)
                T.reduce_sum(acc_s_ub, scores_sum, dim=-1)  # sum over KV_LEN (for normalize)

                # --- Vector: Cast fp32→fp16 + write workspace_2 (Vector→Cube via GM) ---
                T.copy(acc_s_ub, acc_s_half)
                T.copy(acc_s_half, workspace_2[cid, vid * GH : vid * GH + GH, :])

                # --- Cube: Read workspace_2 (full G) → GEMM2 attn @ V → L0C[side] ---
                T.copy(workspace_2[cid, :, :], acc_s_l1)
                T.gemm_v0(acc_s_l1, V_shared, acc_o_l0c[side, :, :], init=True)
                # Fixpipe L0C[side] → workspace_3[cid] (Cube→Vector via GM)
                T.copy(acc_o_l0c[side, :, :], workspace_3[cid, :, :])

                # --- Vector: Read workspace_3 (vid split) + normalize + write Output ---
                T.copy(workspace_3[cid, vid * GH : vid * GH + GH, :], acc_o_ub)
                T.tile.broadcast(scores_sum_2d, scores_sum, axis=1)  # [GH] → [GH, D]
                T.tile.div(acc_o_ub, acc_o_ub, scores_sum_2d)  # acc_o /= sum (normalize)
                T.copy(acc_o_ub, acc_o_half)  # fp32 → fp16
                T.copy(
                    acc_o_half,
                    Output[
                        i_b,
                        i_t,
                        i_h * G + vid * GH : i_h * G + vid * GH + GH,
                        :,
                    ],
                )

    return main


# =============================================================================
# Smoke test entry (CI compatibility)
#
# Repository CI (bench_test.sh) runs `python example_tilelang_nsa_fwd.py` and
# marks PASSED only if stdout contains "Test Passed!". This __main__ runs the
# minimal L0 shape and validates against an embedded PyTorch golden so the main
# file is independently runnable in CI.
# =============================================================================
if __name__ == "__main__":
    import torch

    tilelang.disable_cache()
    torch.manual_seed(0)

    # Minimal L0 smoke config (matches test_nsa_l0 l0_nsa_basic).
    # NOTE: use SEQ_LEN (not T) to avoid shadowing tilelang.language as T above.
    B, SEQ_LEN, H, HQ, D, S, BS, BS_PAD, SCALE = 2, 64, 1, 16, 32, 1, 32, 64, 0.1
    G = HQ // H
    KV_LEN = S * BS_PAD
    DTYPE = torch.float16
    NEG_INF = -(2.0**30)

    gen = torch.Generator().manual_seed(0)
    Q = torch.randn((B, SEQ_LEN, HQ, D), dtype=DTYPE, generator=gen)
    K = torch.randn((B, SEQ_LEN, H, D), dtype=DTYPE, generator=gen)
    V = torch.randn((B, SEQ_LEN, H, D), dtype=DTYPE, generator=gen)

    # block_indices / block_counts (causal, 1 selected block per query).
    # bi filled with SEQ_LEN (sentinel = invalid); bc counts valid blocks.
    bi = torch.full((B, SEQ_LEN, H, S), SEQ_LEN, dtype=torch.long)
    bc = torch.zeros((B, SEQ_LEN, H), dtype=torch.long)
    for b in range(B):
        for t in range(SEQ_LEN):
            for h in range(H):
                picks = torch.randperm(max(1, t // BS), generator=gen)[:S]
                bi[b, t, h, : len(picks)] = picks
                bc[b, t, h] = (bi[b, t, h] != SEQ_LEN).sum().item()
    bi = bi.sort(-1)[0]  # sort for deterministic gather order

    # Pre-gather K_selected / V_selected / CausalMask on CPU (host-side preprocessing).
    # K_sel/V_sel: real tokens + padding zeros. mask: 0.0 visible / -2^30 masked.
    K_sel = torch.zeros(B, SEQ_LEN, H, KV_LEN, D, dtype=DTYPE)
    V_sel = torch.zeros(B, SEQ_LEN, H, KV_LEN, D, dtype=DTYPE)
    mask = torch.full((B, SEQ_LEN, H, KV_LEN), NEG_INF, dtype=torch.float32)
    for b in range(B):
        for t in range(SEQ_LEN):
            for h in range(H):
                bc_val = bc[b, t, h].item()
                for s in range(S):
                    if s >= bc_val:
                        continue
                    block_idx = bi[b, t, h, s].item()
                    for j in range(BS):
                        pos = block_idx * BS + j
                        if 0 <= pos < SEQ_LEN and pos <= t:  # causal: pos <= t
                            K_sel[b, t, h, s * BS + j, :] = K[b, pos, h, :]
                            V_sel[b, t, h, s * BS + j, :] = V[b, pos, h, :]
                            mask[b, t, h, s * BS + j] = 0.0

    # Reshape to 3D kernel inputs: [B*T*H, KV_LEN, D] and mask [B*T*H, G, KV_LEN].
    K_sel_3d = K_sel.reshape(B * SEQ_LEN * H, KV_LEN, D)
    V_sel_3d = V_sel.reshape(B * SEQ_LEN * H, KV_LEN, D)
    mask_3d = mask.reshape(B * SEQ_LEN * H, KV_LEN).unsqueeze(1).expand(-1, G, -1).contiguous()

    # Compile + run kernel.
    kernel = native_sparse_attention(
        batch=B,
        seq_len=SEQ_LEN,
        head_kv=H,
        heads=HQ,
        dim=D,
        selected_blocks=S,
        bs_pad=BS_PAD,
        scale=SCALE,
    )
    out = kernel(Q.npu(), K_sel_3d.npu(), V_sel_3d.npu(), mask_3d.npu())
    torch.npu.synchronize()

    # Golden: PyTorch CPU reference (single-pass softmax, matches kernel semantics).
    Q_f, K_f, V_f = Q.float(), K.float(), V.float()
    ref = torch.zeros(B, SEQ_LEN, HQ, D, dtype=torch.float32)
    for b in range(B):
        for t in range(SEQ_LEN):
            for h in range(H):
                positions = []
                bc_val = bc[b, t, h].item()
                for s in range(S):
                    if s >= bc_val:
                        continue
                    block_idx = bi[b, t, h, s].item()
                    for j in range(BS):
                        pos = block_idx * BS + j
                        if pos <= t and 0 <= pos < SEQ_LEN:
                            positions.append(pos)
                if not positions:
                    continue
                k_g = K_f[b, positions, h, :]
                v_g = V_f[b, positions, h, :]
                for gi in range(G):
                    q = Q_f[b, t, h * G + gi, :]
                    scores = torch.matmul(q, k_g.T) * SCALE
                    attn = torch.softmax(scores, dim=0)
                    ref[b, t, h * G + gi, :] = torch.matmul(attn, v_g)
    ref = ref.to(DTYPE)

    # Precision check: mixed tolerance dual-gate (precision-standard.md §4.1).
    # Float16 thresholds: atol=2^-14, rtol=2^-9, max_abs_limit=1e-1, required_ratio=0.99.
    # NOTE: hardcoded for float16 (the only dtype this kernel supports); if dtype
    # is extended, sync with precision-standard.md §二 table or import get_precision.
    atol, rtol, max_abs_limit, required_ratio = 2**-14, 2**-9, 1e-1, 0.99
    a = out.detach().cpu().float()
    g = ref.detach().float()
    # inf/nan structural compare (precision-standard.md §3.1): positions must match,
    # not counted in numeric tolerance.
    special = ~torch.isfinite(g)
    if special.any() and (
        not torch.equal(torch.isnan(a[special]), torch.isnan(g[special]))
        or not torch.equal(torch.isinf(a[special]), torch.isinf(g[special]))
    ):
        raise AssertionError("inf/nan position mismatch between actual and golden")
    m = torch.isfinite(g)  # golden finite positions: full numeric compare
    if m.sum().item() == 0:
        ratio, max_abs = 1.0, 0.0
    else:
        abs_err = (a[m] - g[m]).abs()
        ratio = (abs_err <= (atol + rtol * g[m].abs())).float().mean().item()
        max_abs = abs_err.max().item()
    print(f"matched_ratio={ratio:.4f} max_abs={max_abs:.3e}")
    assert ratio >= required_ratio and max_abs <= max_abs_limit
    print("Test Passed!")
