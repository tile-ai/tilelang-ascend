"""Native Sparse Attention Forward (NSA Forward) — TileLang Ascend kernel.

Developer mode (hybrid): pass_configs four-True (combineCV + auto_cv_sync +
auto_sync + memory_planning), with alloc_shared/fragment (compiler-mapped
L1/UB/L0C via InferAllocScope), T.Pipelined(num_stages=2) and L0C double buffer.
Overlaps Cube/Vector across iterations via auto-inserted CrossCoreFlag (no
T.barrier_all, no manual T.Scope, no manual set_flag/wait_flag).
All Cube↔Vector data handoffs are on-chip direct T.copy (L0C→UB, UB→L1) —
no GM workspace buffers (eliminates 256KB GM traffic per iteration).
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


@tilelang.jit(out_idx=[4], pass_configs=PASS_CONFIGS)
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
    """NSA Forward kernel (Developer mode hybrid, on-chip direct).

    For each (b, t, h), gathers S selected KV blocks (each BS tokens) and runs
    single-pass softmax attention over the selected tokens only.

    Args (JIT compile-time constants):
        batch, seq_len: grid dims (block_num = batch * seq_len * head_kv).
        head_kv: H (KV head count).
        heads: HQ (query head count, HQ = head_kv * G).
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
    """
    # scale=None → default to 1/sqrt(dim) (proto.yaml declares scale default=null).
    if scale is None:
        scale = (1.0 / dim) ** 0.5

    G = heads // head_kv
    D = dim
    S = selected_blocks
    KV_LEN = S * bs_pad

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

    @T.prim_func
    def main(
        Q: T.Tensor(q_shape, dtype),  # type: ignore
        K_selected: T.Tensor(kv_sel_shape, dtype),  # type: ignore
        V_selected: T.Tensor(kv_sel_shape, dtype),  # type: ignore
        CausalMask: T.Tensor(mask_shape, accum_dtype),  # type: ignore
        Output: T.Tensor(q_shape, dtype),  # type: ignore
    ):
        with T.Kernel(core_num, threads=1, is_npu=True) as (cid):
            # === Cube side L1 buffers (alloc_shared, compiler maps to L1) ===
            Q_shared = T.alloc_shared([G, D], dtype)
            K_shared = T.alloc_shared([KV_LEN, D], dtype)
            V_shared = T.alloc_shared([KV_LEN, D], dtype)
            acc_s_l1 = T.alloc_shared([G, KV_LEN], dtype)

            # === Cube side L0C buffers (alloc_fragment, compiler maps to L0C;
            # double-buffered: GEMM[k]→L0C[side_k] overlaps Fixpipe[k-1]→L0C[1-side_k]).
            # init=True on each GEMM below re-initializes L0C (no cross-iteration
            # accumulation), so side reuse (inner%2) is safe. ===
            acc_s_l0c = T.alloc_fragment([2, G, KV_LEN], accum_dtype)
            acc_o_l0c = T.alloc_fragment([2, G, D], accum_dtype)

            # === Vector side UB buffers (alloc_shared, compiler maps to UB) ===
            acc_s_ub = T.alloc_shared([G, KV_LEN], accum_dtype)  # scaled+masked scores
            acc_s_half = T.alloc_shared([G, KV_LEN], dtype)  # fp16 cast for GEMM2
            acc_o_ub = T.alloc_shared([G, D], accum_dtype)  # GEMM2 output (fp32)
            acc_o_half = T.alloc_shared([G, D], dtype)  # fp16 cast for Output
            scores_max = T.alloc_shared([G], accum_dtype)  # softmax max per head
            scores_sum = T.alloc_shared([G], accum_dtype)  # softmax sum per head
            mask_ub = T.alloc_shared([G, KV_LEN], accum_dtype)  # additive causal mask
            scores_max_2d = T.alloc_shared([G, KV_LEN], accum_dtype)  # broadcast of scores_max
            scores_sum_2d = T.alloc_shared([G, D], accum_dtype)  # broadcast of scores_sum

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
                # C→V handoff #1: L0C[side] → UB direct (on-chip, no GM workspace)
                T.copy(acc_s_l0c[side, :, :], acc_s_ub)

                # --- Vector: Load mask (full G, no vid split) ---
                T.copy(CausalMask[block_idx, :, :], mask_ub)

                # --- Vector: Scale + mask + single-pass softmax (on [G, KV_LEN]) ---
                T.tile.mul(acc_s_ub, acc_s_ub, scale)  # scores *= scale
                T.tile.add(acc_s_ub, acc_s_ub, mask_ub)  # scores += mask (-2^30 → ~-inf)
                T.reduce_max(acc_s_ub, scores_max, dim=-1)  # max over KV_LEN
                T.tile.broadcast(scores_max_2d, scores_max, axis=1)  # [G] → [G, KV_LEN]
                T.tile.sub(acc_s_ub, acc_s_ub, scores_max_2d)  # scores -= max (numerically stable)
                T.tile.exp(acc_s_ub, acc_s_ub)  # exp(scores - max)
                T.reduce_sum(acc_s_ub, scores_sum, dim=-1)  # sum over KV_LEN (for normalize)

                # --- Vector: Cast fp32→fp16 + V→C handoff #2: UB→L1 direct ---
                T.copy(acc_s_ub, acc_s_half)
                T.copy(acc_s_half, acc_s_l1)

                # --- Cube: GEMM2 attn @ V → L0C[side] ---
                T.gemm_v0(acc_s_l1, V_shared, acc_o_l0c[side, :, :], init=True)
                # C→V handoff #3: L0C[side] → UB direct (on-chip, no GM workspace)
                T.copy(acc_o_l0c[side, :, :], acc_o_ub)

                # --- Vector: Normalize + write Output (full G, no vid split) ---
                T.tile.broadcast(scores_sum_2d, scores_sum, axis=1)  # [G] → [G, D]
                T.tile.div(acc_o_ub, acc_o_ub, scores_sum_2d)  # acc_o /= sum (normalize)
                T.copy(acc_o_ub, acc_o_half)  # fp32 → fp16
                T.copy(
                    acc_o_half,
                    Output[i_b, i_t, i_h * G : (i_h + 1) * G, :],
                )

    return main


# =============================================================================
# Smoke test entry (CI compatibility, FULLY SELF-CONTAINED)
#
# Repository CI (bench_test.sh) runs `python example_tilelang_nsa_fwd.py` and
# marks PASSED only if stdout contains "Test Passed!". This __main__ block is
# intentionally self-contained: it does NOT import from the sibling test module
# (test_example_tilelang_nsa_fwd.py) so the example file can be smoke-tested
# in isolation. All helper logic (input generation, pre-gather, reshape,
# golden reference, precision check) is inlined below for the single L0 case.
#
# `import torch` is deferred to inside __main__ to avoid making torch a
# module-level dependency of the example file (keeps tilelang imports clean).
# =============================================================================
if __name__ == "__main__":
    import torch  # noqa: E402

    tilelang.disable_cache()
    torch.manual_seed(0)

    # Single L0 smoke config (matches test_nsa_l0 l0_nsa_basic).
    # NOTE: use SEQ_LEN (not T) to avoid shadowing tilelang.language as T above.
    B, SEQ_LEN, H, HQ, D, S, BS, BS_PAD, SCALE = 2, 64, 1, 16, 32, 1, 32, 64, 0.1
    G = HQ // H  # 16 query heads per KV head (must be even for vid split)
    KV_LEN = S * BS_PAD  # 64
    DTYPE = torch.float16
    NEG_INF = -(2.0**30)  # additive mask sentinel for "masked" positions

    # --- Generate inputs on CPU (inline: randn + randperm for block selection). ---
    # Matches gen_test_inputs: sentinel SEQ_LEN = invalid; sorted for deterministic order.
    gen = torch.Generator().manual_seed(0)
    Q = torch.randn((B, SEQ_LEN, HQ, D), dtype=DTYPE, generator=gen)
    K = torch.randn((B, SEQ_LEN, H, D), dtype=DTYPE, generator=gen)
    V = torch.randn((B, SEQ_LEN, H, D), dtype=DTYPE, generator=gen)
    bi = torch.full((B, SEQ_LEN, H, S), SEQ_LEN, dtype=torch.long)
    bc = torch.zeros((B, SEQ_LEN, H), dtype=torch.long)
    for b in range(B):
        for t in range(SEQ_LEN):
            for h in range(H):
                picks = torch.randperm(max(1, t // BS), generator=gen)[:S]
                bi[b, t, h, : len(picks)] = picks
                bc[b, t, h] = (bi[b, t, h] != SEQ_LEN).sum().item()
    bi = bi.sort(-1)[0]

    # --- Pre-gather K_selected / V_selected / CausalMask on CPU (inline for single case). ---
    # K_sel/V_sel: real tokens + padding zeros. mask: 0.0 visible / -2^30 masked.
    # Causal rule: only positions pos <= t are visible (is_causal=True for L0 basic).
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
                        if 0 <= pos < SEQ_LEN and pos <= t:
                            K_sel[b, t, h, s * BS + j, :] = K[b, pos, h, :]
                            V_sel[b, t, h, s * BS + j, :] = V[b, pos, h, :]
                            mask[b, t, h, s * BS + j] = 0.0

    # --- Reshape to 3D kernel inputs (inline, no _to_3d_inputs helper). ---
    # K_sel/V_sel: [B, T, H, KV_LEN, D] -> [B*T*H, KV_LEN, D].
    # mask: [B, T, H, KV_LEN] -> [B*T*H, G, KV_LEN] (broadcast to all G query heads).
    K_sel_3d = K_sel.reshape(B * SEQ_LEN * H, KV_LEN, D)
    V_sel_3d = V_sel.reshape(B * SEQ_LEN * H, KV_LEN, D)
    mask_3d = mask.reshape(B * SEQ_LEN * H, KV_LEN).unsqueeze(1).expand(-1, G, -1).contiguous()

    # --- Compile + run kernel. ---
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

    # --- Golden: PyTorch CPU reference (single-pass softmax, matches kernel semantics). ---
    # scale is NOT multiplied by log2(e): NPU uses T.exp, GPU uses T.exp2 * log2(e)
    # (mathematically equivalent). Uses original Q/K/V/bi directly (no pre-gather).
    Q_f, K_f, V_f = Q.float(), K.float(), V.float()
    ref = torch.zeros(B, SEQ_LEN, HQ, D, dtype=torch.float32)
    for b in range(B):
        for t in range(SEQ_LEN):
            for h in range(H):
                # Collect valid (causal) positions for this (b, t, h).
                positions = []
                bc_val = bc[b, t, h].item()
                for s in range(S):
                    if s >= bc_val:
                        continue
                    block_idx = bi[b, t, h, s].item()
                    for j in range(BS):
                        pos = block_idx * BS + j
                        if 0 <= pos < SEQ_LEN and pos <= t:
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

    # --- Precision check: mixed tolerance dual-gate (precision-standard.md §4.1). ---
    # Float16 thresholds: atol=2^-14, rtol=2^-9, max_abs_limit=1e-1, required_ratio=0.99.
    # Pass condition: matched_ratio >= required_ratio AND max_abs <= max_abs_limit.
    # inf/nan positions: structural compare (not counted in numeric tolerance).
    atol, rtol, max_abs_limit, required_ratio = 2**-14, 2**-9, 1e-1, 0.99
    a = out.detach().cpu().float()
    g = ref.detach().cpu().float()
    # inf/nan structural compare (precision-standard.md §3.1).
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
    assert ratio >= required_ratio and max_abs <= max_abs_limit, f"precision check failed: ratio={ratio:.4f} max_abs={max_abs:.3e}"
    print("Test Passed!")
