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
    block_size,
    bs_pad,
    scale,
    is_causal=True,
    core_num=None,
):
    """NSA Forward kernel (Developer mode hybrid, on-chip direct, kernel-internal mask).

    For each (b, t, h), gathers S selected KV blocks (each BS tokens) and runs
    single-pass softmax attention over the selected tokens only.

    Args (JIT compile-time constants):
        batch, seq_len: grid dims (block_num = batch * seq_len * head_kv).
        head_kv: H (KV head count).
        heads: HQ (query head count, HQ = head_kv * G).
        dim: D (head dimension).
        selected_blocks: S (number of selected KV blocks).
        block_size: BS (actual block size, tokens per selected block).
        bs_pad: internal tile size (padded from block_size for 256-byte alignment).
        scale: softmax scale factor (None → 1/sqrt(dim); NPU uses T.exp, so scale
            is NOT multiplied by log2(e), unlike GPU T.exp2 path).
        is_causal: if True, mask = (token_pos <= i_t); if False, mask = (token_pos < seq_len).
        core_num: physical AI Cube core count. If None, auto-detect via
            torch.npu.get_device_properties().cube_core_num (e.g. 20 for Ascend 910B3).

    Tensor args (prim_func):
        Q:           [batch, seq_len, heads, dim]                  float16
        K_selected:  [batch*seq_len*head_kv, S*bs_pad, dim]        float16 (host pre-gathered, packed at s*BS+j)
        V_selected:  [batch*seq_len*head_kv, S*bs_pad, dim]        float16 (host pre-gathered, packed at s*BS+j)
        BlockStarts: [batch*seq_len*head_kv, S]                    int32 (block_indices*BS, host pre-multiplied)
        Output:      [batch, seq_len, heads, dim]                  float16 (out_idx=4)

    Kernel-internal causal mask: computed from BlockStarts + token position via
    arith_progression + compare + select (no host precomputed CausalMask tensor).
    Invalid blocks (sentinel block_indices=seq_len → BlockStarts=seq_len*BS) are
    masked automatically (token_pos >> i_t). Padding positions (S*BS..KV_LEN-1)
    are filled with -inf.
    """
    # scale=None → default to 1/sqrt(dim) (proto.yaml declares scale default=null).
    if scale is None:
        scale = (1.0 / dim) ** 0.5

    # Auto-detect physical AI Cube core count if not provided.
    # Ascend 910B3: cube_core_num=20, vector_core_num=40.
    if core_num is None:
        import torch
        import torch_npu  # noqa: F401 (registers npu backend)

        props = torch.npu.get_device_properties(torch.npu.current_device())
        core_num = props.cube_core_num

    G = heads // head_kv
    D = dim
    S = selected_blocks
    BS = block_size
    KV_LEN = S * bs_pad

    q_shape = [batch, seq_len, heads, D]
    kv_sel_shape = [batch * seq_len * head_kv, KV_LEN, D]
    block_starts_shape = [batch * seq_len * head_kv, S]
    dtype = "float16"
    accum_dtype = "float32"

    # 1D Kernel: each block handles one (i_t, i_b, i_h) combination.
    block_num = seq_len * batch * head_kv
    # Largest divisor of block_num <= core_num (exact division; T.Pipelined needs
    # uniform iteration count across cores). core_num is the physical Cube core
    # count (auto-detected via torch.npu.get_device_properties).
    core_num = min(block_num, core_num)
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
        BlockStarts: T.Tensor(block_starts_shape, "int32"),  # type: ignore
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
            scores_max_2d = T.alloc_shared([G, KV_LEN], accum_dtype)  # broadcast of scores_max
            scores_sum_2d = T.alloc_shared([G, D], accum_dtype)  # broadcast of scores_sum

            # === Kernel-internal causal mask buffers ===
            # mask_ub: full [G, KV_LEN] additive mask (0.0=visible, NEG_INF=masked).
            #   Built per-block via row-by-row T.copy (1D contiguous; 2D non-contiguous
            #   T.copy flattens to 1D, scrambling data). Applied with T.tile.add (same
            #   pattern as old host-precomputed mask, proven to avoid NaN).
            # col_pos: 1D [BS] for arith_progression (j = 0..BS-1).
            # cond_s_2d: 2D [G, BS] per-block token_pos, then overwritten with additive mask.
            # compare_result: 1D [G*BS//8] uint8 packed bitmask for compare output (selMask).
            #   AscendC CompareScalar requires uint8_t output (1 bit per element, packed 8/byte).
            #   Select selMask also requires uint8. G*BS is always divisible by 8 (G>=16, BS>=32).
            #   (select with selMask==dst causes read-after-write hazard; separate buffer avoids it.)
            # zero_s: 2D [G, BS] filled with 0.0 (src0 for select: visible→0.0).
            #   (compare/select require 256-byte alignment; G>=16 ensures [G,BS] is aligned.)
            mask_ub = T.alloc_shared([G, KV_LEN], accum_dtype)  # full additive mask
            col_pos = T.alloc_shared([BS], accum_dtype)  # j = 0..BS-1 progression
            cond_s_2d = T.alloc_shared([G, BS], accum_dtype)  # per-block work buffer
            compare_result = T.alloc_shared([G * BS // 8], "uint8")  # packed bitmask (selMask)
            zero_s = T.alloc_shared([G, BS], accum_dtype)  # constant 0.0 for select src0

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

                # --- Vector: Scale + kernel-internal causal mask + single-pass softmax ---
                T.tile.mul(acc_s_ub, acc_s_ub, scale)  # scores *= scale

                # Kernel-internal causal mask: compute additive mask inline from BlockStarts.
                # For each block s in [0, S), token_pos = BlockStarts[block_idx, s] + j
                # (j = 0..BS-1, BlockStarts = block_indices*BS pre-multiplied on host).
                # mask_ub: 0.0=visible, NEG_INF=masked. Built per-block via row-by-row
                #   T.copy (1D contiguous slices). Applied with T.tile.add (same as old
                #   host-precomputed mask; NEG_INF is finite → no NaN when all masked).
                # Conversion: compare → uint8 packed bitmask (1=visible, 0=masked), then
                #   select(zero_s, NEG_INF) → 0.0/NEG_INF.
                #   (Arithmetic mul+sub fails: 0.0*NEG_INF=NaN in IEEE 754.)
                #   (select with selMask==dst causes RAW hazard; separate compare_result buffer.)
                #   (AscendC CompareScalar/Select require uint8_t mask, not float — per review.)
                # Invalid blocks (sentinel → BlockStarts=seq_len*BS → token_pos >> i_t → masked).
                # Padding positions (S*BS..KV_LEN-1) remain NEG_INF from fill.
                # Loop is Python-unrolled: s, g are compile-time constants.
                NEG_INF = -(2.0**30)
                T.tile.fill(mask_ub, NEG_INF)  # default: all masked (incl. padding)
                T.tile.fill(zero_s, 0.0)  # constant 0.0 for select src0

                for s in range(S):
                    bs_start = BlockStarts[block_idx, s]  # GM scalar read
                    T.tile.arith_progression(col_pos, 0, 1, BS)  # col_pos = [0,1,...,BS-1]
                    T.tile.broadcast(cond_s_2d, col_pos)  # [BS] → [G, BS]
                    T.tile.add(cond_s_2d, cond_s_2d, bs_start)  # token_pos = bs_start + j
                    if is_causal:
                        T.tile.compare(compare_result, cond_s_2d, i_t, "LE")  # uint8 bitmask
                    else:
                        T.tile.compare(compare_result, cond_s_2d, float(seq_len - 1), "LE")
                    # Convert bitmask → 0.0/NEG_INF: visible→zero_s(0.0), masked→NEG_INF(scalar)
                    T.tile.select(cond_s_2d, compare_result, zero_s, NEG_INF, "VSEL_TENSOR_SCALAR_MODE")
                    # Row-by-row 1D copy to mask_ub slice (contiguous in row-major layout).
                    for g in range(G):
                        T.copy(cond_s_2d[g, :], mask_ub[g, s * BS : (s + 1) * BS])

                T.tile.add(acc_s_ub, acc_s_ub, mask_ub)  # scores += mask

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

                # --- Vector: Normalize + write Output ---
                T.tile.broadcast(scores_sum_2d, scores_sum, axis=1)  # [G] → [G, D]
                T.tile.div(acc_o_ub, acc_o_ub, scores_sum_2d)  # acc_o /= sum (normalize)
                T.copy(acc_o_ub, acc_o_half)  # fp32 → fp16
                T.copy(
                    acc_o_half,
                    Output[i_b, i_t, i_h * G : (i_h + 1) * G, :],
                )

    return main


# =============================================================================
# Golden reference (CPU, based on naive_nsa; independent of prepare_inputs)
#
# This is the SINGLE golden implementation. The test file imports it:
#   from example_tilelang_nsa_fwd import golden_nsa_fwd
# =============================================================================
def golden_nsa_fwd(Q, K, V, block_indices, block_counts, block_size=32, scale=0.1, is_causal=True):
    """PyTorch CPU reference based on naive_nsa (g_slc=g_swa=ones, window_size=0).

    Uses original Q/K/V/block_indices directly (no pre-gather). Single-pass softmax
    matches the NPU kernel. scale is NOT multiplied by log2(e) — NPU uses T.exp,
    GPU uses T.exp2 * log2(e), which are mathematically equivalent.
    """
    import torch  # deferred: keeps tilelang imports clean at module level

    B, T, HQ, D = Q.shape
    H = K.shape[2]
    G = HQ // H
    S = block_indices.shape[-1]
    BS = block_size

    dtype = Q.dtype
    Q_f, K_f, V_f = Q.float(), K.float(), V.float()
    Output = torch.zeros(B, T, HQ, D, dtype=torch.float32)

    for b in range(B):
        for t in range(T):
            for h in range(H):
                # Collect valid positions for this (b, t, h).
                positions = []
                bc_val = block_counts[b, t, h].item() if isinstance(block_counts, torch.Tensor) else block_counts
                for s in range(S):
                    if s >= bc_val:
                        continue
                    bi = block_indices[b, t, h, s].item()
                    for j in range(BS):
                        pos = bi * BS + j
                        if is_causal and pos > t:
                            continue
                        if pos < 0 or pos >= T:
                            continue
                        positions.append(pos)

                if len(positions) == 0:
                    continue

                k_gathered = K_f[b, positions, h, :]
                v_gathered = V_f[b, positions, h, :]

                # Per query head in the group: standard attention.
                for g in range(G):
                    hq = h * G + g
                    q = Q_f[b, t, hq, :]
                    scores = torch.matmul(q, k_gathered.T) * scale
                    attn = torch.softmax(scores, dim=0)
                    Output[b, t, hq, :] = torch.matmul(attn, v_gathered)

    return Output.to(dtype)


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
    G = HQ // H  # query heads per KV head (GQA group size)
    KV_LEN = S * BS_PAD  # 64
    DTYPE = torch.float16

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

    # --- Pre-gather K_selected / V_selected on CPU (inline for single case). ---
    # K_sel/V_sel: real tokens + padding zeros. Causal mask is computed in-kernel.
    # Tokens packed at positions s*BS+j for block s, j in [0, BS).
    K_sel = torch.zeros(B, SEQ_LEN, H, KV_LEN, D, dtype=DTYPE)
    V_sel = torch.zeros(B, SEQ_LEN, H, KV_LEN, D, dtype=DTYPE)
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
                        if 0 <= pos < SEQ_LEN:
                            K_sel[b, t, h, s * BS + j, :] = K[b, pos, h, :]
                            V_sel[b, t, h, s * BS + j, :] = V[b, pos, h, :]

    # --- Reshape to 3D kernel inputs (inline, no _to_3d_inputs helper). ---
    # K_sel/V_sel: [B, T, H, KV_LEN, D] -> [B*T*H, KV_LEN, D].
    K_sel_3d = K_sel.reshape(B * SEQ_LEN * H, KV_LEN, D)
    V_sel_3d = V_sel.reshape(B * SEQ_LEN * H, KV_LEN, D)
    # BlockStarts: [B, T, H, S] -> [B*T*H, S] int32 (block_indices * BS, pre-multiplied).
    block_starts = (bi.to(torch.int32) * BS).reshape(B * SEQ_LEN * H, S)

    # --- Compile + run kernel. ---
    kernel = native_sparse_attention(
        batch=B,
        seq_len=SEQ_LEN,
        head_kv=H,
        heads=HQ,
        dim=D,
        selected_blocks=S,
        block_size=BS,
        bs_pad=BS_PAD,
        scale=SCALE,
        is_causal=True,
    )
    out = kernel(Q.npu(), K_sel_3d.npu(), V_sel_3d.npu(), block_starts.npu())
    torch.npu.synchronize()

    # --- Golden: PyTorch CPU reference (single-pass softmax, matches kernel semantics). ---
    # Calls the module-level golden_nsa_fwd (same function the test file imports).
    # scale is NOT multiplied by log2(e): NPU uses T.exp, GPU uses T.exp2 * log2(e)
    # (mathematically equivalent). Uses original Q/K/V/bi directly (no pre-gather).
    ref = golden_nsa_fwd(Q, K, V, bi, bc, block_size=BS, scale=SCALE, is_causal=True)

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
