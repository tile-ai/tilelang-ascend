"""NSA Forward VarLen -- TileLang Ascend kernel.

Native Sparse Attention forward pass with variable-length sequence support.
Computes Q@K^T -> softmax -> P@V with online softmax, causal masking, and gate
multiplication, specialized for S=1 (single selected block per head).

Developer mode (hybrid): pass_configs four-True with explicit L1/L0C/UB
allocation. No T.Scope or manual barriers -- AUTO_CV_COMBINE separates
Cube/Vector cores, AUTO_CV_SYNC handles inter-core sync.

Key design:
- Persistent grid: T.Kernel(core_num=20), waves=ceildiv(total_tiles, core_num)
- Multi-buffer pipeline: num_stages=4 workspace slots per core
- Dual-loop: Cube waves (GEMM1+GEMM2) then Vector waves (softmax+accumulate)
- Q pre-multiply scale on host (eliminates per-iter axpy in Vector softmax)
- S=1 specialization: scores_scale=0, eliminates online softmax scale chain
- Host-precomputed varlen indices: breaks dependent GM scalar read chain
- Kernel-internal causal mask: computed from BosPerToken/IsSafe + token
  position via arith_progression + compare + select (no host pre-compute)
"""

import tilelang
from tilelang import language as T
from tilelang.intrinsics import make_nz_layout, make_zn_layout

# Developer mode pass_configs: sync & C/V split fully auto-managed by passes.
PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,  # auto C/V split
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,  # auto cross-core sync
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,  # auto intra-core sync
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,  # auto L0/L1/UB address
}


@tilelang.jit(out_idx=[3], workspace_idx=[8, 9, 10], pass_configs=PASS_CONFIGS)
def native_sparse_attention_varlen(
    batch,
    c_seq_len,
    heads,
    dim,
    is_causal=True,
    scale=None,
    block_size=32,
    groups=16,
    selected_blocks=1,
    dtype="float16",
    core_num=20,
):
    """NSA Forward VarLen kernel (Developer mode hybrid, multi-buffer pipeline).

    Args (JIT compile-time constants):
        batch, c_seq_len, heads, dim: tensor shape parameters.
        is_causal: accepted for API compat (causal mask computed in kernel).
        scale: accepted for API compat (Q is pre-scaled on host).
        block_size, groups, selected_blocks: NSA hyperparameters (S must be 1).
        dtype: output dtype ("float16"/"bfloat16"/"float32").
        core_num: physical AI Cube cores (default 20 for Ascend 910B3).

    Q/K/V use gemm_dtype (fp32 input is pre-cast to fp16 on host -- Cube doesn't
    support fp32 GEMM). O_slc uses dtype (fp32 output preserved for fp32 input).
    """
    # S=1 specialization: scores_scale=0, eliminates online softmax scale chain.
    assert selected_blocks == 1, (
        f"optD4 optimization requires selected_blocks=1 (S=1 specialization), got selected_blocks={selected_blocks}."
    )

    # Cube GEMM only supports fp16/bf16; fp32 input is pre-cast to fp16 on host.
    # Reject unsupported dtypes early to avoid cryptic codegen errors (e.g. float64
    # triggers "Divide by zero" in make_zn_layout fractal computation).
    assert dtype in ("float16", "bfloat16", "float32"), (
        f"Unsupported dtype: {dtype}. Supported: float16, bfloat16, float32 "
        f"(fp32 is pre-cast to fp16 on host since Cube does not support fp32 GEMM)."
    )

    if scale is None:
        scale = (1.0 / dim) ** 0.5

    head_kv = heads // groups
    G = groups
    BS = block_size
    BK = BV = dim
    accum_dtype = "float"
    gemm_dtype = "float16" if dtype == "float32" else dtype
    NEG_INF = -(2**30)
    half_G = G // 2  # per-AIV groups (vid split: G=16 -> half_G=8)

    # Persistent grid: launch core_num blocks, each processes waves tiles.
    total_tiles = c_seq_len * batch * head_kv
    waves = T.ceildiv(total_tiles, core_num)

    # Multi-buffer pipeline: num_stages workspace slots per cid.
    num_stages = 4
    num_waves_outer = T.ceildiv(waves, num_stages)

    @T.prim_func
    def main(
        Q: T.Tensor([c_seq_len, heads, dim], gemm_dtype),  # type: ignore
        K: T.Tensor([c_seq_len, head_kv, dim], gemm_dtype),  # type: ignore
        V: T.Tensor([c_seq_len, head_kv, dim], gemm_dtype),  # type: ignore
        # O_slc: final output = (acc_o / logsum) * g_slc (gate multiplication fused in kernel).
        # Named O_slc for API compat with NSA paper; semantically equals golden's `o`
        # (NOT golden's intermediate `o_slc` which excludes gate multiplication).
        O_slc: T.Tensor([c_seq_len, heads, dim], dtype),  # type: ignore
        BosPerToken: T.Tensor([c_seq_len], "int32"),  # type: ignore
        IsSafe: T.Tensor([c_seq_len, head_kv], "int32"),  # type: ignore
        BlockCounts: T.Tensor([c_seq_len, head_kv], "int32"),  # type: ignore
        g_slc: T.Tensor([c_seq_len, heads], accum_dtype),  # type: ignore
        workspace_1: T.Tensor([core_num, num_stages, G, BS], accum_dtype),  # type: ignore
        workspace_2: T.Tensor([core_num, num_stages, G, BS], gemm_dtype),  # type: ignore
        workspace_3: T.Tensor([core_num, num_stages, G, BV], accum_dtype),  # type: ignore
    ):
        with T.Kernel(core_num, is_npu=True) as (cid, vid):
            v_offset = vid * half_G  # per-AIV group offset

            # ===== Cube L1 buffers (GEMM inputs, full G for GEMM) =====
            q_l1 = T.alloc_L1([G, BK], gemm_dtype)
            k_l1 = T.alloc_L1([BS, BK], gemm_dtype)
            v_l1 = T.alloc_L1([BS, BV], gemm_dtype)
            p_l1 = T.alloc_L1([G, BS], gemm_dtype)

            # ===== Cube L0C (GEMM outputs, fp32 accumulation) =====
            acc_s = T.alloc_L0C([G, BS], accum_dtype)
            acc_o_tmp = T.alloc_L0C([G, BV], accum_dtype)

            # ZN/NZ layout annotation for L1 buffers (fractal layout for Cube GEMM).
            # q_l1/p_l1: A input -> ZN; k_l1/v_l1: B input -> NZ.
            T.annotate_layout(
                {
                    q_l1: make_zn_layout(q_l1),
                    k_l1: make_nz_layout(k_l1),
                    p_l1: make_zn_layout(p_l1),
                    v_l1: make_nz_layout(v_l1),
                }
            )

            # ===== Vector UB buffers (half_G per AIV) =====
            acc_s_ub = T.alloc_ub([half_G, BS], accum_dtype)
            # Kernel-internal causal mask buffers (computed from BosPerToken/IsSafe).
            col_pos = T.alloc_ub([BS], accum_dtype)  # 1D [0, 1, ..., BS-1]
            mask_ub = T.alloc_ub([half_G, BS], accum_dtype)  # 2D broadcast mask
            acc_s_cast_ub = T.alloc_ub([half_G, BS], gemm_dtype)
            acc_o = T.alloc_ub([half_G, BV], accum_dtype)
            acc_o_ub = T.alloc_ub([half_G, BV], accum_dtype)
            acc_o_half = T.alloc_ub([half_G, BV], dtype)

            scores_max = T.alloc_ub([half_G], accum_dtype)
            scores_sum = T.alloc_ub([half_G], accum_dtype)
            logsum = T.alloc_ub([half_G], accum_dtype)
            g_slc_ub = T.alloc_ub([half_G], accum_dtype)

            scores_max_brd = T.alloc_ub([half_G, BS], accum_dtype)
            g_slc_brd = T.alloc_ub([half_G, BV], accum_dtype)

            # Save per-tile logsum between softmax/accumulate batches.
            logsum_slots = T.alloc_ub([num_stages, half_G], accum_dtype)

            # ===== Cube code: batch GEMM1 + batch GEMM2 =====
            # No T.Scope -- combineCV auto-separates based on L1/GEMM operations.
            # AUTO_CV_SYNC auto-inserts sync at workspace read/write points.
            for w_outer in T.serial(num_waves_outer):
                _remaining = waves - w_outer * num_stages
                batch_iters = T.if_then_else(_remaining < num_stages, _remaining, num_stages)

                # GEMM1 batch: Q @ K^T -> ws1[cid, i]
                for i in T.serial(batch_iters):
                    w = w_outer * num_stages + i
                    tile_id = core_num * w + cid

                    if tile_id < total_tiles:
                        bx = tile_id % c_seq_len
                        bz = tile_id // c_seq_len
                        i_h = bz % head_kv

                        bos = BosPerToken[bx]
                        i_s = IsSafe[bx, i_h]

                        T.copy(Q[bx, i_h * G : (i_h + 1) * G, :], q_l1)
                        T.copy(K[bos + i_s : bos + i_s + BS, i_h, :], k_l1)
                        T.gemm_v0(q_l1, k_l1, acc_s, transpose_B=True, init=True)
                        T.copy(acc_s, workspace_1[cid, i, :, :])

                # GEMM2 batch: P @ V -> ws3[cid, i]
                for i in T.serial(batch_iters):
                    w = w_outer * num_stages + i
                    tile_id = core_num * w + cid

                    if tile_id < total_tiles:
                        bx = tile_id % c_seq_len
                        bz = tile_id // c_seq_len
                        i_h = bz % head_kv
                        bos = BosPerToken[bx]
                        i_s = IsSafe[bx, i_h]

                        T.copy(workspace_2[cid, i, :, :], p_l1)
                        T.copy(V[bos + i_s : bos + i_s + BS, i_h, :], v_l1)
                        T.gemm_v0(p_l1, v_l1, acc_o_tmp, init=True)
                        T.copy(acc_o_tmp, workspace_3[cid, i, :, :])

            # ===== Vector code: batch softmax + batch accumulate =====
            # No T.Scope -- combineCV auto-separates based on UB/tile operations.
            for w_outer in T.serial(num_waves_outer):
                _remaining = waves - w_outer * num_stages
                batch_iters = T.if_then_else(_remaining < num_stages, _remaining, num_stages)

                # Softmax batch: ws1[cid, i] -> ws2[cid, i], save logsum_slots[i]
                for i in T.serial(batch_iters):
                    w = w_outer * num_stages + i
                    tile_id = core_num * w + cid

                    if tile_id < total_tiles:
                        bx = tile_id % c_seq_len
                        bz = tile_id // c_seq_len
                        i_h = bz % head_kv
                        NS = BlockCounts[bx, i_h]

                        T.tile.fill(acc_o, 0.0)
                        T.tile.fill(logsum, 0.0)
                        T.tile.fill(scores_max, NEG_INF)

                        # Q pre-scaled on host: ws1 = Q_scaled @ K^T = scale * (Q @ K^T).
                        T.copy(workspace_1[cid, i, v_offset : v_offset + half_G, :], acc_s_ub)

                        # Kernel-internal causal mask: mask[j] = 0 if (i_s+j <= i_t) else -inf.
                        # i_t = bx - BosPerToken[bx] (token position in segment),
                        # i_s = IsSafe[bx, i_h] (start index of selected KV block).
                        bos = BosPerToken[bx]
                        i_t = bx - bos
                        i_s = IsSafe[bx, i_h]
                        T.tile.arith_progression(col_pos, 0, 1, BS)
                        # mask_ub = (col_pos + i_s) <= i_t ? 1.0 : 0.0, then select -inf.
                        T.tile.broadcast(mask_ub, col_pos)
                        T.tile.add(mask_ub, mask_ub, i_s)
                        T.tile.compare(mask_ub, mask_ub, i_t, "LE")
                        T.tile.select(
                            acc_s_ub,
                            mask_ub,
                            acc_s_ub,
                            -T.infinity(accum_dtype),
                            "VSEL_TENSOR_SCALAR_MODE",
                        )
                        T.reduce_max(acc_s_ub, scores_max, dim=-1, clear=True)
                        T.tile.broadcast(scores_max_brd, scores_max)
                        T.tile.sub(acc_s_ub, acc_s_ub, scores_max_brd)
                        T.tile.exp(acc_s_ub, acc_s_ub)
                        # NS=0 guard: zero P when no selected blocks.
                        if NS <= 0:
                            T.tile.fill(acc_s_ub, 0.0)
                        T.reduce_sum(acc_s_ub, scores_sum, dim=-1, clear=True)
                        T.tile.add(logsum, logsum, scores_sum)
                        T.copy(acc_s_ub, acc_s_cast_ub)
                        T.copy(acc_s_cast_ub, workspace_2[cid, i, v_offset : v_offset + half_G, :])

                        T.copy(logsum, logsum_slots[i, :])

                # Accumulate batch: ws3[cid, i] -> O_slc
                for i in T.serial(batch_iters):
                    w = w_outer * num_stages + i
                    tile_id = core_num * w + cid

                    if tile_id < total_tiles:
                        bx = tile_id % c_seq_len
                        bz = tile_id // c_seq_len
                        i_h = bz % head_kv
                        NS = BlockCounts[bx, i_h]

                        T.copy(logsum_slots[i, :], logsum)

                        # scores_scale=0 (S=1) + acc_o fill(0) -> mul redundant -> single copy.
                        T.copy(workspace_3[cid, i, v_offset : v_offset + half_G, :], acc_o_ub)
                        T.copy(acc_o_ub, acc_o)
                        T.copy(g_slc[bx, i_h * G + v_offset : i_h * G + v_offset + half_G], g_slc_ub)
                        # NS=0: logsum=0 -> div by 0 -> inf -> 0*inf=nan.
                        # Skip div/broadcast/mul and fill 0 directly to avoid nan propagation.
                        if NS <= 0:
                            T.tile.fill(acc_o, 0.0)
                        else:
                            T.tile.div(g_slc_ub, g_slc_ub, logsum)
                            T.tile.broadcast(g_slc_brd, g_slc_ub)
                            T.tile.mul(acc_o, acc_o, g_slc_brd)
                        T.copy(acc_o, acc_o_half)
                        T.copy(
                            acc_o_half,
                            O_slc[bx, i_h * G + v_offset : i_h * G + v_offset + half_G, :],
                        )

    return main


def _prepare_token_indices(offsets):
    """Build token_indices [C_SEQ_LEN, 2] = (batch_idx, token_idx_in_seq) from offsets [N+1].

    CPU-only construction (no NPU aclnn). Returns int32 tensor.
    """
    n = len(offsets) - 1
    total = int(offsets[-1].item())
    token_indices = torch.zeros(total, 2, dtype=torch.int32)
    idx = 0
    for batch_idx in range(n):
        seg_len = int(offsets[batch_idx + 1].item()) - int(offsets[batch_idx].item())
        for t in range(seg_len):
            token_indices[idx, 0] = batch_idx
            token_indices[idx, 1] = t
            idx += 1
    return token_indices


def _build_smoke_inputs(n, c_seq_len, h, hq, d, s, bs, dtype):
    """Build self-contained smoke-test inputs for the L0 config (no golden, no test-file import).

    Constructs deterministic Q/K/V/g_slc + varlen indices on CPU,
    then moves to NPU. Used only by `__main__` for CI smoke test.
    Causal mask is computed in-kernel (no host pre-compute).
    """
    torch.manual_seed(42)

    # offsets: split c_seq_len into n BS-aligned segments.
    if n == 1:
        offsets = torch.tensor([0, c_seq_len], dtype=torch.int32)
    else:
        n_slots = c_seq_len // bs
        slot_splits = torch.randperm(n_slots - 1)[: n - 1].sort().values + 1
        split_pts = (slot_splits * bs).to(torch.int32)
        offsets = torch.cat(
            [
                torch.tensor([0], dtype=torch.int32),
                split_pts,
                torch.tensor([c_seq_len], dtype=torch.int32),
            ]
        )

    token_indices = _prepare_token_indices(offsets)

    # block_indices: each token selects up to s candidate blocks within segment & causal.
    block_indices = torch.zeros((1, c_seq_len, h, s), dtype=torch.int64)
    block_counts = torch.zeros((1, c_seq_len, h), dtype=torch.int64)
    for c in range(c_seq_len):
        i_n = int(token_indices[c, 0].item())
        t = int(token_indices[c, 1].item())
        bos = int(offsets[i_n].item())
        seg_len = int(offsets[i_n + 1].item()) - bos
        max_safe = (seg_len - bs) // bs
        max_causal = t // bs
        max_block = min(max_safe, max_causal)
        n_candidates = max_block + 1
        if n_candidates <= 0:
            continue
        for head_idx in range(h):
            n_sel = min(s, n_candidates)
            i_i = torch.randperm(n_candidates)[:n_sel].sort().values
            block_indices[0, c, head_idx, :n_sel] = i_i
            block_counts[0, c, head_idx] = n_sel

    def _make_kv(seq_len, n_expand, kv_dtype):
        perm = torch.randperm(seq_len)
        return torch.linspace(0, 1, steps=seq_len, dtype=kv_dtype)[perm].view(1, seq_len, 1, 1).expand(1, seq_len, n_expand, d).contiguous()

    q = _make_kv(c_seq_len, hq, dtype)
    k = _make_kv(c_seq_len, h, dtype)
    v = _make_kv(c_seq_len, h, dtype)
    g_slc = torch.rand((1, c_seq_len, hq), dtype=dtype)
    scale = d**-0.5

    # Causal mask is computed in-kernel from BosPerToken/IsSafe + token position.
    # No host-side mask pre-computation needed.

    return q, k, v, block_indices, block_counts, offsets, token_indices, g_slc, scale


def _run_kernel_smoke(q, k, v, block_indices, block_counts, offsets, token_indices, g_slc, scale, dtype_str):
    """Host wrapper: H2D-safe preprocessing + kernel call (self-contained, no test-file import).

    - fp32 input is pre-cast to fp16 on host (Cube doesn't support fp32 GEMM).
    - Q is pre-multiplied with scale (fuse scale into Q, eliminates per-iter axpy).
    - All int64 -> int32 conversions done on CPU before H2D.
    - Causal mask computed in-kernel (no host pre-compute, no mask tensor passed).
    """
    B, C_SEQ_LEN, H, K_dim = k.shape
    _, _, HQ, V_dim = q.shape
    _, _, _, S = block_indices.shape
    G = HQ // H
    BS = (offsets[1] - offsets[0]).item() if len(offsets) > 1 else 32  # infer BS from offsets
    batch = len(offsets) - 1

    gemm_dtype_str = "float16" if dtype_str == "float32" else dtype_str
    gemm_dtype = getattr(torch, gemm_dtype_str)

    # Q pre-multiply scale (fuse scale into Q at host side).
    q = (q * scale).contiguous()

    # Host-side pre-cast fp32 -> fp16.
    if dtype_str != gemm_dtype_str:
        q = q.to(gemm_dtype)
        k = k.to(gemm_dtype)
        v = v.to(gemm_dtype)

    # GM workspace auto-allocated by framework via workspace_idx (no host allocation).
    core_num = 20

    # Host-side precompute varlen indices to break scalar GM read chain.
    bos_per_token = offsets[token_indices[:, 0]].to(torch.int32)
    bi_2d = block_indices.view(C_SEQ_LEN, H, S)
    bc_2d = block_counts.view(C_SEQ_LEN, H)
    i_s_safe = torch.where(
        bc_2d > 0,
        bi_2d[:, :, 0].to(torch.int32) * BS,
        torch.zeros(C_SEQ_LEN, H, dtype=torch.int32),
    )

    o_slc = native_sparse_attention_varlen(
        batch=batch,
        c_seq_len=C_SEQ_LEN,
        heads=HQ,
        dim=K_dim,
        is_causal=True,
        scale=scale,
        block_size=BS,
        groups=G,
        selected_blocks=S,
        dtype=dtype_str,
        core_num=core_num,
    )(
        q.view(C_SEQ_LEN, HQ, K_dim).npu(),
        k.view(C_SEQ_LEN, H, K_dim).npu(),
        v.view(C_SEQ_LEN, H, K_dim).npu(),
        bos_per_token.npu(),
        i_s_safe.npu(),
        bc_2d.to(torch.int32).npu(),
        g_slc.view(C_SEQ_LEN, HQ).float().npu(),
    )
    return o_slc.view(B, C_SEQ_LEN, HQ, V_dim)


if __name__ == "__main__":
    """CI smoke test: single L0 case, self-contained (no test-file import, no golden).

    Runs N=2, C_SEQ=64, HQ=16, H=1, D=64, S=1, BS=32, fp16 and verifies:
    1. Kernel compiles and runs without error.
    2. Output shape matches expected [B, C_SEQ_LEN, HQ, D].
    3. Output has no NaN / Inf (structural sanity check).

    Precision comparison against PyTorch golden is done in
    test_example_tilelang_nsa_fwd_varlen.py (precision test suite).
    Outputs "Test Passed!" for CI bench_test.sh.
    """
    import torch

    import tilelang

    tilelang.disable_cache()

    # L0 config (single representative case for CI).
    N, C_SEQ_LEN, H, HQ, D, S, BS = 2, 64, 1, 16, 64, 1, 32
    DTYPE = torch.float16
    DTYPE_STR = "float16"

    # === Build self-contained smoke-test inputs (no test-file import) ===
    q, k, v, bi, bc, off, ti, g, scale = _build_smoke_inputs(N, C_SEQ_LEN, H, HQ, D, S, BS, DTYPE)

    # === Run kernel on NPU ===
    out = _run_kernel_smoke(q, k, v, bi, bc, off, ti, g, scale, DTYPE_STR)

    # === Structural sanity check (no golden, no precision comparison) ===
    out_cpu = out.detach().cpu()

    # 1. Shape check: output must match [B, C_SEQ_LEN, HQ, D].
    expected_shape = (1, C_SEQ_LEN, HQ, D)
    assert out_cpu.shape == expected_shape, f"Output shape mismatch: got {tuple(out_cpu.shape)}, expected {expected_shape}"

    # 2. NaN / Inf check: output must not contain NaN or Inf.
    has_nan = torch.isnan(out_cpu).any().item()
    has_inf = torch.isinf(out_cpu).any().item()
    assert not has_nan, f"Output contains NaN (shape={tuple(out_cpu.shape)})"
    assert not has_inf, f"Output contains Inf (shape={tuple(out_cpu.shape)})"

    # 3. Value range sanity: output magnitude should be within a reasonable bound.
    #    NSA forward output = softmax(QK^T) @ V * g_slc, with Q/K/V in [0,1] and g_slc in [0,1).
    #    Expected magnitude: |out| <= D * |V_max| * |g_slc_max| ~ 64 * 1 * 1 = 64.
    out_abs_max = out_cpu.abs().max().item()
    assert out_abs_max < 100.0, f"Output magnitude unreasonable: max_abs={out_abs_max}"

    print(
        f"[SMOKE_PASS] l0 N={N} C_SEQ={C_SEQ_LEN} D={D} S={S} BS={BS} dtype={DTYPE_STR} "
        f"shape={tuple(out_cpu.shape)} max_abs={out_abs_max:.3e}"
    )
    print("Test Passed!")
