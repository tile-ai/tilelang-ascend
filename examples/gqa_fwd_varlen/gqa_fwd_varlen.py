import torch

import tilelang
from tilelang import language as T
from tilelang.intrinsics import make_zn_layout, make_nz_layout


def generate_random_padding_mask(max_seqlen, batch_size, device, mode="random"):
    """Generate a random padding mask.

    Args:
        max_seqlen: maximum possible sequence length (= padded seqlen).
        batch_size: number of sequences.
        device: torch device (e.g. "npu").
        mode: "full" (all max), "random" ([max-20, max]), "third" ([max//3, max]).

    Returns:
        padding_mask: [batch_size, max_seqlen] bool tensor.
    """
    assert mode in ["full", "random", "third"]
    if mode == "full":
        lengths = torch.full((batch_size, 1), max_seqlen, device=device, dtype=torch.int32)
    elif mode == "random":
        lengths = torch.randint(max(1, max_seqlen - 20), max_seqlen + 1, (batch_size, 1), device=device)
    elif mode == "third":
        lengths = torch.randint(max_seqlen // 3, max_seqlen + 1, (batch_size, 1), device=device)
    padding_mask = torch.arange(max_seqlen, device=device).unsqueeze(0) < lengths
    return padding_mask


def mask_to_cu_seqlens(padding_mask):
    """Convert a [batch, seqlen] bool mask to cu_seqlens [batch+1] int32."""
    lengths = padding_mask.sum(dim=1).to(torch.int32)
    cu_seqlens = torch.zeros(padding_mask.shape[0] + 1, dtype=torch.int32, device=padding_mask.device)
    cu_seqlens[1:] = torch.cumsum(lengths, dim=0)
    return cu_seqlens


def build_attention_mask(
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    is_causal,
    device,
):
    """Build the attention mask tensor on the host.

    Returns:
        mask: [batch, max_seqlen_q, max_seqlen_k] float32.
              1.0 = visible, 0.0 = masked (padding or causal).
    """
    batch = int(cu_seqlens_q.shape[0]) - 1
    q_idx = torch.arange(max_seqlen_q, device=device).view(-1, 1)  # [M, 1]
    k_idx = torch.arange(max_seqlen_k, device=device).view(1, -1)  # [1, N]
    mask = torch.zeros(batch, max_seqlen_q, max_seqlen_k, dtype=torch.float32, device=device)
    for b in range(batch):
        q_len = int(cu_seqlens_q[b + 1].item()) - int(cu_seqlens_q[b].item())
        kv_len = int(cu_seqlens_k[b + 1].item()) - int(cu_seqlens_k[b].item())
        offset = kv_len - q_len
        pad_mask = (q_idx < q_len) & (k_idx < kv_len)  # [M, N]
        if is_causal:
            causal_mask = k_idx <= q_idx + offset  # True = visible
            visible = pad_mask & causal_mask
        else:
            visible = pad_mask
        mask[b] = visible.float()
    return mask


# ===========================================================================
# JIT kernel (Expert mode, CV fusion, 4D padded layout, mask tensor)
# Structure follows flash_attn_bhsd.py; additions: GQA, varlen, causal, mask.
# ===========================================================================

PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: False,  # manual CV separation
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: False,  # manual inter-core sync
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: False,  # manual intra-core sync
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: False,  # manual memory planning
}

NUM_CORES = 24  # 910B has 24 AI Cores (static task distribution)


@tilelang.jit(out_idx=[4], workspace_idx=[5, 6, 7], pass_configs=PASS_CONFIGS)
def flashattn(
    batch_size,
    groups,
    heads,
    dim,
    max_seqlen_q,
    max_seqlen_k,
    is_causal,
    block_M=128,
    block_N=128,
    num_stages=8,
    cross_interval=1,
    apply_mask=True,
):
    """GQA varlen Flash Attention forward kernel (Expert mode, pipelined).

    Iter 2: CV pipeline rewrite following fa_opt/flash_attn_bhsd_expert_h16_d128.py.
    - num_stages=14 multi-stage pipeline (batch KV iterations)
    - T.mma + L0A/L0B/L0C double buffering (replaces T.gemm_v0)
    - T.set_flag/wait_flag fine-grained intra-core sync (replaces T.barrier_all)
    - T.tile.broadcast + T.tile.axpy vectorized softmax (replaces per-row loop)
    - T.annotate_layout ZN/NZ layout optimization
    - Mask integration: mask applied after exp via T.tile.mul, synced with barrier_all

    Iter 3: 2-flag mask sync + param tuning (21.26 ms, 51.72 TFlops).
    - 2-flag mask sync: SIG_MASK_FREE (V→MTE2) + SIG_MASK_READY (MTE2→V) replaces
      2 T.barrier_all() per KV iter. No init needed (first iter V exp runs first).
    - num_stages=14→8 (bench_mark confirmed optimal: 8+8 two full batches vs 14+2)
    - cross_interval=2→1 (more frequent sync lets Vector start earlier)
    - Standalone 2-flag gain only 1.4% (noise range), but param tuning gave 9.9% total

    Iter 4: Mask skip + double-buffered io_buf (target: ≤15 ms).
    - apply_mask compile-time flag: when False (non-causal + full padding), skips
      mask GM load (MTE2), mask mul (V), and 2-flag sync entirely. This eliminates
      MTE2 serialization (was 2 loads/iter: QK 16KB + mask 32KB; now 1 load/iter:
      QK 16KB only). MTE2 pipeline becomes continuous QK loads, fully hidden behind
      V compute. Root cause: mask load on same MTE2 unit blocked next QK load.
    - Double-buffered io_buf [2, half_M, block_N]: MTE2 loads QK[i+1] into io_buf[1]
      while V processes QK[i] from io_buf[0]. Removes io_buf release wait, enabling
      full MTE2/V pipeline overlap. UB: +16KB (132.8→148.8 KB < 192 KB limit).
    - block_N=256 NOT feasible: L0B [2,128,256] fp16=128KB > 64KB L0B limit;
      GEMM2 P matrix [128,256] in L0A also overflows. Confirmed via capacity analysis.

    Args:
        batch_size: number of sequences (compile-time).
        groups: GQA group size (heads // head_kv).
        heads: number of Q heads.
        dim: head dimension (fixed 128 for L0).
        max_seqlen_q: padded max Q sequence length (compile-time).
        max_seqlen_k: padded max K sequence length (compile-time).
        is_causal: whether to apply causal mask (compile-time, documentation only).
        block_M: Q block size.
        block_N: K/V block size.
        num_stages: pipeline depth (batch KV iterations per outer loop).
        cross_interval: cross-core sync interval (sync every N iterations).
        apply_mask: whether to apply attention mask (compile-time). False skips
            mask load+mul for non-causal full-padding case (mask is all 1.0).

    Kernel inputs (4D padded + mask tensor + 3 GM workspaces):
        Q: [batch, heads, max_seqlen_q, dim]                 # 0
        K: [batch, head_kv, max_seqlen_k, dim]               # 1
        V: [batch, head_kv, max_seqlen_k, dim]               # 2
        Mask: [batch, max_seqlen_q, max_seqlen_k] float32    # 3
        Output: [batch, heads, max_seqlen_q, dim]             # 4 (out_idx)
        workspace_1: [NUM_CORES, num_stages, block_M, block_N] fp16  # 5 (QK scores)
        workspace_2: [NUM_CORES, num_stages, block_M, block_N] fp16  # 6 (softmax P)
        workspace_3: [NUM_CORES, num_stages, block_M, dim]    fp16  # 7 (PV output)
    """
    head_kv = heads // groups
    sm_scale = (1.0 / dim) ** 0.5  # natural exp, no log2(e) factor
    dtype = "float16"
    accum_dtype = "float"

    q_shape = [batch_size, heads, max_seqlen_q, dim]
    kv_shape = [batch_size, head_kv, max_seqlen_k, dim]
    mask_shape = [batch_size, max_seqlen_q, max_seqlen_k]
    o_shape = [batch_size, heads, max_seqlen_q, dim]

    assert max_seqlen_q % block_M == 0, f"max_seqlen_q ({max_seqlen_q}) must be divisible by block_M ({block_M})"
    assert max_seqlen_k % block_N == 0, f"max_seqlen_k ({max_seqlen_k}) must be divisible by block_N ({block_N})"
    assert num_stages % 2 == 0, "num_stages must be even for double buffering"

    num_q_blocks = max_seqlen_q // block_M
    max_kv_iters = max_seqlen_k // block_N
    block_num = num_q_blocks * heads * batch_size
    num_outer = T.ceildiv(max_kv_iters, num_stages)

    q_tasks = block_num // NUM_CORES
    r_tasks = block_num % NUM_CORES

    # Cross-core semaphore IDs (Cube <-> Vector)
    SEM_WS1_C2V = 0  # workspace_1 (QK^T) ready: Cube -> Vector
    SEM_WS1_V2C = 1  # workspace_1 consumed: Vector -> Cube
    SEM_WS2_V2C = 2  # workspace_2 (softmax P) ready: Vector -> Cube
    SEM_WS2_C2V = 3  # workspace_2 consumed: Cube -> Vector
    SEM_WS3_C2V = 4  # workspace_3 (PV output) ready: Cube -> Vector
    SEM_WS3_V2C = 5  # workspace_3 consumed: Vector -> Cube

    # Intra-core signal IDs (C Scope)
    SIG_K_L1 = 0
    SIG_P_L1 = 1
    SIG_V_L1 = 2
    SIG_L0AB = 3  # double-buffer base: slot 0 = SIG_L0AB, slot 1 = SIG_L0AB + 1
    SIG_L0C = 5  # double-buffer base: slot 0 = SIG_L0C,  slot 1 = SIG_L0C + 1

    # Intra-core signal IDs (V Scope)
    # io_buf double-buffered: slot 0 = SIG_IO_UB, slot 1 = SIG_IO_UB + 1
    SIG_IO_UB = 0
    SIG_S_HALF = 2
    SIG_MASK_FREE = 3  # V -> MTE2: buf_2d released after exp (mask can overwrite)
    SIG_MASK_READY = 4  # MTE2 -> V: mask loaded into buf_2d (mul can proceed)

    @T.prim_func
    def main(
        Q: T.Tensor(q_shape, dtype),  # type: ignore
        K: T.Tensor(kv_shape, dtype),  # type: ignore
        V: T.Tensor(kv_shape, dtype),  # type: ignore
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
            # NOTE: use block_M // 2 (Python int) in alloc shapes, not half_M (TIR var)
            acc_o = T.alloc_ub([block_M // 2, dim], accum_dtype)

            # Batch softmax buffers (num_stages slots for deferred rescale)
            r_factors = T.alloc_ub([num_stages, block_M // 2, 1], accum_dtype)
            sumexp_is = T.alloc_ub([num_stages, block_M // 2, 1], accum_dtype)

            sumexp = T.alloc_ub([block_M // 2, 1], accum_dtype)
            neg_sm = T.alloc_ub([2, block_M // 2, 1], accum_dtype)  # double-buffered max

            # IO and work buffers (reused across phases)
            # Double-buffered io_buf: MTE2 loads QK[i+1] while V processes QK[i]
            io_buf = T.alloc_ub([2, block_M // 2, block_N], dtype)  # GM <-> UB transfer (fp16)
            acc_s_half = T.alloc_ub([block_M // 2, block_N], dtype)  # fp16 softmax output

            work_ub = T.alloc_ub([block_M // 2, block_N], accum_dtype)  # main compute buffer (fp32)
            buf_2d = T.alloc_ub([block_M // 2, block_N], accum_dtype)  # broadcast+mask buffer (fp32)
            # NOTE: mask reuses buf_2d after exp consumes it (saves 32KB UB)

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
                    by = (task_id // num_q_blocks) % heads
                    bz = task_id // (num_q_blocks * heads)
                    kv_head_idx = by // groups  # GQA

                    T.copy(Q[bz, by, bx * block_M : (bx + 1) * block_M, :], q_l1)
                    T.barrier_all()

                    for k in T.serial(num_outer):
                        _remaining = max_kv_iters - k * num_stages
                        batch_iters = T.if_then_else(_remaining < num_stages, _remaining, num_stages)

                        # --- GEMM1: produce QK^T scores into workspace_1 ---
                        T.wait_cross_flag(SEM_WS1_V2C)
                        for i in T.serial(batch_iters):
                            side = i % 2
                            idx = k * num_stages + i

                            # K: GM -> L1 (MTE2 -> MTE1 flag)
                            T.wait_flag("MTE1", "MTE2", SIG_K_L1)
                            T.copy(K[bz, kv_head_idx, idx * block_N : (idx + 1) * block_N, :], k_l1)
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

                            # V: GM -> L1
                            T.wait_flag("MTE1", "MTE2", SIG_V_L1)
                            T.copy(V[bz, kv_head_idx, idx * block_N : (idx + 1) * block_N, :], v_l1)
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
                    by = (task_id // num_q_blocks) % heads
                    bz = task_id // (num_q_blocks * heads)

                    T.tile.fill(acc_o, 0.0)
                    T.tile.fill(sumexp, 0.0)
                    T.tile.fill(neg_sm, 2**30)  # large positive = -inf max

                    for k in T.serial(num_outer):
                        _remaining = max_kv_iters - k * num_stages
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

                            if apply_mask:
                                # === MASK AFTER EXP (mask tensor from GM) ===
                                # buf_2d is free after exp consumed it.
                                # 2-flag sync (replaces barrier_all): V releases buf_2d
                                # after exp, MTE2 loads mask, MTE2 signals V for mul.
                                # No init needed: first iter V runs exp first, then sets
                                # SIG_MASK_FREE; MTE2 waits and unblocks after that.
                                T.set_flag("V", "MTE2", SIG_MASK_FREE)
                                T.wait_flag("V", "MTE2", SIG_MASK_FREE)
                                T.copy(
                                    Mask[
                                        bz,
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

                    # Final normalize: acc_o /= sumexp
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
# runs the minimal L0 shape (l0_min_full_nc config) and validates against
# F.scaled_dot_product_attention so the main file is independently runnable
# in CI. The @tilelang.jit flashattn kernel above is unchanged.
# ===========================================================================


if __name__ == "__main__":
    import torch.nn.functional as F

    tilelang.disable_cache()
    torch.set_default_device("npu")
    torch.manual_seed(0)

    # Minimal smoke test — matches L0 "l0_min_full_nc" config:
    # batch=1, heads=4, groups=2, sq=skv=128, dim=128, non-causal, full padding.
    batch, heads, groups = 1, 4, 2
    sq, skv, dim = 128, 128, 128
    is_causal = False
    padding_mode = "full"
    block_M, block_N = 128, 128
    head_kv = heads // groups
    atol = 1e-2

    torch.manual_seed(0)
    q = torch.randn(batch, heads, sq, dim, dtype=torch.float16, device="npu")
    k = torch.randn(batch, head_kv, skv, dim, dtype=torch.float16, device="npu")
    v = torch.randn(batch, head_kv, skv, dim, dtype=torch.float16, device="npu")

    # Full padding -> all-visible mask (1.0). cu_seqlens consumed on host only.
    q_mask = generate_random_padding_mask(sq, batch, "npu", mode=padding_mode)
    k_mask = generate_random_padding_mask(skv, batch, "npu", mode=padding_mode)
    cu_seqlens_q = mask_to_cu_seqlens(q_mask)
    cu_seqlens_k = mask_to_cu_seqlens(k_mask)
    attn_mask = build_attention_mask(cu_seqlens_q, cu_seqlens_k, sq, skv, is_causal, "npu")

    # Non-causal + full padding + block-aligned -> mask is all 1.0 -> apply_mask=False.
    apply_mask = is_causal or padding_mode != "full" or (sq % block_M != 0) or (skv % block_N != 0)
    kernel = flashattn(
        batch,
        groups,
        heads,
        dim,
        sq,
        skv,
        is_causal,
        block_M=block_M,
        block_N=block_N,
        apply_mask=apply_mask,
    )

    out = kernel(q, k, v, attn_mask)
    torch.npu.synchronize()

    # Golden: SDPA with GQA repeat (NPU equivalent of flash_attn_varlen_func).
    # SDPA scale defaults to 1/sqrt(dim), matching the kernel's sm_scale.
    k_rep = k.repeat_interleave(groups, dim=1)  # [batch, heads, skv, dim]
    v_rep = v.repeat_interleave(groups, dim=1)
    ref_out = F.scaled_dot_product_attention(q, k_rep, v_rep, is_causal=False)
    torch.npu.synchronize()

    max_diff = (out.float() - ref_out.float()).abs().max().item()
    print(f"max_diff: {max_diff:.6e}")
    assert max_diff < atol, f"Precision check failed: max_diff={max_diff} >= atol={atol}"
    print("Test Passed!")
