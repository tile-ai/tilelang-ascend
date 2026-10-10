"""Sparse MLA Forward Pipelined (DeepSeek V3.2 Multi-head Latent Attention).

Developer mode + combineCV. Single kernel, persistent grid.
Computes O = softmax(Q @ KV^T / sqrt(D)) @ KV[:dim] with sparse KV gather.

Design:
  - Cube: GEMM1 (Q @ KV^T, K=576) + GEMM2 (P @ KV[:dim])
  - Vector: sparse KV gather + online softmax + output accumulate
  - Workspace GM relay (4 tensors) for Cube <-> Vector data exchange
  - Causal mask via T.tile.compare + T.tile.select (uint8 packed bitmask)

All synchronization handled by auto_sync + auto_cv_sync + auto_cv_combine.
"""

import tilelang
import torch
from tilelang import language as T
from tilelang.intrinsics import make_zn_layout, make_nz_layout

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
}


@tilelang.jit(out_idx=[4], workspace_idx=[5, 6, 7, 8], pass_configs=pass_configs)
def sparse_mla_fwd_pipelined(
    heads,
    dim,
    tail_dim,
    topk,
    kv_stride,
    kv_group=1,
    sm_scale=None,
    is_causal=True,
    CP0=True,
    block_I=128,
    core_num=20,
):
    """Build the sparse MLA forward kernel.

    Args:
        heads: number of query heads. head_kv = heads // kv_group must be a
            power of two in [16, 64] (16/32/64) or a multiple of 64 (> 64;
            with kv_group > 1 it must additionally be a power of two).
            Other values are rejected at entry (OOB Q/Output access risk).
        dim: V dimension (512).
        tail_dim: rope/PE dimension (64).
        topk: number of sparse indices (must be a multiple of block_I and
            >= 2 * block_I, i.e. NI >= 2: NI=1 breaks Ascend codegen L1 address
            planning for T.Pipelined(1, num_stages=2)).
        kv_stride: KV stride.
        kv_group: KV group count (1 for MLA).
        sm_scale: softmax scale (None => 1/sqrt(dim+tail_dim)).
        is_causal: must be True.
        CP0: deprecated, no effect (kept for API compatibility; grid no longer
            special-cases q_start==0).
        block_I: indices block size. Must be 128 (the only verified value):
            block_I % 64 == 0 is structurally required (T.tile.compare on
            [block_I] fp32 needs 256B alignment, subsuming the BI//8 packed
            mask and BI//2 vid split), and other multiples of 64 are broken
            (64 -> AscendMemoryPlanning failure, 192 -> compile segfault).
        core_num: persistent grid size.

    Returns:
        prim_func sparse_mla_fwd_kernel(Q, KV, Indices, q_start_index_s, Output, ws1/ws3/ws4/ws5)

    Note: Lse is not output (Ascend codegen does not support scalar T.log).
    """
    assert dim == tilelang.math.next_power_of_2(dim), f"dim must be power of 2, got {dim}"
    assert tail_dim == tilelang.math.next_power_of_2(tail_dim), f"tail_dim must be power of 2, got {tail_dim}"
    assert is_causal, "non-causal is not supported"
    assert topk % block_I == 0, "topk must be a multiple of block_I"

    # Constraints documented in the assert message and docstring; see also
    # codegen_ascend.cc CompareScalarCodegen (256B alignment) and debug_log.
    assert block_I == 128, (
        f"block_I must be 128 (only verified value): block_I % 64 == 0 is required "
        f"by T.tile.compare 256B alignment on [block_I] fp32, and other multiples "
        f"of 64 (64 -> AscendMemoryPlanning failure, 192 -> compile segfault) are "
        f"broken; got block_I={block_I}"
    )

    # NI >= 2 required: topk == block_I (NI=1) breaks Ascend codegen L1 address
    # planning for T.Pipelined(1, num_stages=2) ("Cannot find pre-allocated
    # address for buffer: kv_full_l1", codegen_ascend.cc:830). Framework issue
    # pending; constrained at entry.
    assert topk >= 2 * block_I, (
        f"topk must be >= 2 * block_I (NI >= 2): NI=1 fails Ascend codegen L1 "
        f"address planning for T.Pipelined(1, num_stages=2); got topk={topk}, "
        f"block_I={block_I}"
    )

    # sm_scale has no log2(e) premultiplication (unlike the GPU exp2 version):
    #     Ascend has no exp2 hardware support, the kernel uses plain T.tile.exp.
    sm_scale = (1.0 / (dim + tail_dim)) ** 0.5 if sm_scale is None else sm_scale

    batch = T.symbolic("batch")
    seq_len = T.symbolic("seq_len")
    seq_len_kv = T.symbolic("seq_len_kv")

    head_kv = heads // kv_group
    q_shape = [batch, seq_len, heads, dim + tail_dim]
    kv_shape = [batch, seq_len_kv, kv_group, dim + tail_dim]
    o_shape = [batch, seq_len, heads, dim]
    indices_shape = [batch, seq_len, kv_group, topk]

    indices_dtype = "int32"
    dtype = "bfloat16"
    accum_dtype = "float"

    BI = block_I
    NI = tilelang.cdiv(topk, block_I)
    D = dim
    D_tail = tail_dim
    KV_stride = kv_stride

    padded_H = max(tilelang.math.next_power_of_2(head_kv), 16)

    # Valid head_kv domain documented in the asserts below and the docstring.
    if head_kv > 64:
        assert head_kv % 64 == 0, f"head_kv should be a multiple of 64, got {head_kv}"
        if kv_group > 1:
            assert head_kv == tilelang.math.next_power_of_2(head_kv), (
                f"kv_group > 1 requires power-of-two head_kv (group base offset uses padded_H), got head_kv={head_kv}"
            )
        REPLICATE_H = head_kv // 64
    else:
        assert head_kv in (16, 32, 64), (
            f"head_kv (heads//kv_group) must be one of 16/32/64 when <= 64 "
            f"(power of two, >= 16); got {head_kv}. Other values cause OOB "
            f"Q/Output access via padded_H"
        )
        REPLICATE_H = 1

    H_per_block = padded_H if REPLICATE_H == 1 else 64
    v_block = H_per_block // 2

    # All seq_len query rows are always scheduled: rows with q_i < kv_stride - 1
    # attend kv0 only (force-valid), implemented by max_kv_i = max(raw - 1, 0).
    # Do NOT reintroduce row-skipping grid shortcuts (caused OOB, see debug_log
    # Attempt 4).
    total_tiles_expr = seq_len * REPLICATE_H * batch * kv_group

    @T.prim_func
    def sparse_mla_fwd_kernel(
        Q: T.Tensor(q_shape, dtype),  # type: ignore
        KV: T.Tensor(kv_shape, dtype),  # type: ignore
        Indices: T.Tensor(indices_shape, indices_dtype),  # type: ignore
        q_start_index_s: T.Tensor([1], indices_dtype),  # type: ignore
        Output: T.Tensor(o_shape, dtype),  # type: ignore
        workspace_1: T.Tensor([core_num, BI, D + D_tail], dtype),
        workspace_3: T.Tensor([core_num, H_per_block, BI], dtype),
        workspace_4: T.Tensor([core_num, H_per_block, BI], dtype),
        workspace_5: T.Tensor([core_num, H_per_block, D], dtype),
    ):
        with T.Kernel(core_num, is_npu=True) as (cid, vid):
            # ===== Cube L1 / L0C buffers (persistent across waves) =====
            q_full_l1 = T.alloc_L1([H_per_block, D + D_tail], dtype)
            kv_full_l1 = T.alloc_L1([BI, D + D_tail], dtype)
            acc_s_l1 = T.alloc_L1([H_per_block, BI], dtype)
            acc_s_l0c = T.alloc_L0C([H_per_block, BI], accum_dtype)
            acc_o_l0c = T.alloc_L0C([H_per_block, D], accum_dtype)

            # L1 layout: ZN for A matrices, NZ for B matrix (transpose_B=True).
            T.annotate_layout(
                {
                    q_full_l1: make_zn_layout(q_full_l1),
                    kv_full_l1: make_nz_layout(kv_full_l1),
                    acc_s_l1: make_zn_layout(acc_s_l1),
                }
            )

            # ===== Vector UB buffers (persistent across waves) =====
            acc_o = T.alloc_ub([v_block, D], accum_dtype)
            sumexp = T.alloc_ub([v_block], accum_dtype)
            m_i = T.alloc_ub([v_block], accum_dtype)
            m_i_prev = T.alloc_ub([v_block], accum_dtype)
            indices_ub_ = T.alloc_ub([BI], indices_dtype)
            indices_ub_float = T.alloc_ub([BI], accum_dtype)
            kv_full_ub = T.alloc_ub([BI // 2, D + D_tail], dtype)
            acc_s_ub = T.alloc_ub([v_block, BI], accum_dtype)
            acc_s_ub_ = T.alloc_ub([v_block, BI], accum_dtype)
            sumexp_i_ub = T.alloc_ub([v_block], accum_dtype)
            acc_s_half = T.alloc_ub([v_block, BI], dtype)
            acc_o_ub = T.alloc_ub([v_block, D], accum_dtype)
            acc_o_half = T.alloc_ub([v_block, D], dtype)
            mask_ub = T.alloc_ub([BI // 8], "uint8")

            # ===== Persistent grid outer loop =====
            for core_index in T.serial(T.ceildiv(total_tiles_expr, core_num)):
                pid = core_index * core_num + cid
                if pid < total_tiles_expr:
                    # Decode tile coordinates
                    bx = pid % (seq_len * REPLICATE_H)
                    by = pid // (seq_len * REPLICATE_H) % batch
                    bz = pid // (seq_len * REPLICATE_H) // batch % kv_group

                    b_i = by
                    g_i = bz
                    s_i = bx // REPLICATE_H
                    H0 = g_i * padded_H + (bx % REPLICATE_H) * H_per_block

                    # max_kv_i: causal mask boundary.
                    q_i = q_start_index_s[0] + s_i
                    raw = (q_i + 1) // KV_stride
                    max_kv_i = T.max(raw - 1, 0)

                    # ===== Load Q =====
                    T.copy(Q[b_i, s_i, H0 : H0 + H_per_block, :], q_full_l1)

                    # ===== Init online softmax state =====
                    T.tile.fill(acc_o, 0.0)
                    T.tile.fill(sumexp, 0.0)
                    T.tile.fill(m_i, -(2.0**30))

                    # NI loop: GEMM1 -> gather -> softmax -> GEMM2 -> accumulate.
                    # Cross-queue ordering over ws1/ws3/ws4/ws5 is enforced by
                    # auto_cv_sync (combineCV pass).
                    for i_i in T.Pipelined(NI, num_stages=2):
                        # --- Cube: GEMM1 ---
                        T.copy(workspace_1[cid, 0:BI, 0 : D + D_tail], kv_full_l1)
                        T.gemm_v0(q_full_l1, kv_full_l1, acc_s_l0c, transpose_B=True, init=True)
                        T.copy(acc_s_l0c, workspace_3[cid, 0:H_per_block, 0:BI])

                        # --- Vector: gather KV by sparse indices ---
                        T.copy(Indices[b_i, s_i, g_i, i_i * BI : i_i * BI + BI], indices_ub_)
                        T.copy(indices_ub_, indices_ub_float)
                        T.tile.compare(mask_ub, indices_ub_float, T.float32(max_kv_i), "LE")
                        for bi_i in range(BI // 2):
                            idx = indices_ub_[bi_i + vid * BI // 2]
                            T.copy(KV[b_i, idx, g_i, :], kv_full_ub[bi_i, :])
                        T.copy(kv_full_ub, workspace_1[cid, vid * BI // 2 : (vid + 1) * BI // 2, :])

                        # --- Vector: online softmax ---
                        T.tile.fill(acc_s_ub_, 0.0)
                        for h_i in range(v_block):
                            T.tile.select(
                                acc_s_ub[h_i, :],
                                mask_ub,
                                acc_s_ub_[h_i, :],
                                -T.infinity(accum_dtype),
                                "VSEL_TENSOR_SCALAR_MODE",
                            )
                        T.copy(m_i, m_i_prev)
                        # Relay ws3 (bf16) through acc_s_half: GM bf16 -> UB bf16 -> UB fp32.
                        T.copy(
                            workspace_3[cid, vid * v_block : vid * v_block + v_block, :],
                            acc_s_half,
                        )
                        T.copy(acc_s_half, acc_s_ub_)
                        # Fuse add+mul: acc_s_ub = sm_scale * S + acc_s_ub (0 or -inf).
                        T.tile.axpy(acc_s_ub, acc_s_ub_, sm_scale)
                        T.reduce_max(acc_s_ub, m_i, dim=-1)
                        T.tile.max(m_i, m_i, m_i_prev)
                        T.tile.sub(m_i_prev, m_i_prev, m_i)
                        T.tile.exp(m_i_prev, m_i_prev)
                        T.tile.broadcast(acc_s_ub_, m_i)
                        T.tile.sub(acc_s_ub, acc_s_ub, acc_s_ub_)
                        T.tile.exp(acc_s_ub, acc_s_ub)
                        T.reduce_sum(acc_s_ub, sumexp_i_ub, dim=-1)
                        T.tile.mul(sumexp, sumexp, m_i_prev)
                        T.tile.add(sumexp, sumexp, sumexp_i_ub)
                        T.copy(acc_s_ub, acc_s_half)
                        T.copy(
                            acc_s_half,
                            workspace_4[cid, vid * v_block : vid * v_block + v_block, :],
                        )

                        # --- Cube: GEMM2 ---
                        T.copy(workspace_4[cid, 0:H_per_block, 0:BI], acc_s_l1)
                        T.gemm_v0(acc_s_l1, kv_full_l1[:, :D], acc_o_l0c, init=True)
                        T.copy(acc_o_l0c, workspace_5[cid, 0:H_per_block, 0:D])

                        # --- Vector: O accumulate ---
                        T.tile.broadcast(acc_o_ub, m_i_prev)
                        T.tile.mul(acc_o, acc_o, acc_o_ub)
                        # Relay ws5 (bf16): GM bf16 -> UB bf16 -> UB fp32.
                        T.copy(
                            workspace_5[cid, vid * v_block : vid * v_block + v_block, :],
                            acc_o_half,
                        )
                        T.copy(acc_o_half, acc_o_ub)
                        T.tile.add(acc_o, acc_o, acc_o_ub)

                    # ===== Final rescale + output =====
                    T.tile.broadcast(acc_o_ub, sumexp)
                    T.tile.div(acc_o, acc_o, acc_o_ub)
                    T.copy(acc_o, acc_o_half)
                    T.copy(
                        acc_o_half,
                        Output[b_i, s_i, H0 + vid * v_block : H0 + (vid + 1) * v_block, :],
                    )

    return sparse_mla_fwd_kernel


# ============================================================================
# Smoke test: compile + run + shape/dtype verification
# ============================================================================


if __name__ == "__main__":
    tilelang.disable_cache()
    torch.set_default_device("npu")
    torch.manual_seed(0)

    # Minimal smoke shape (fast compile + verify)
    B, S, SKV, H, HKV, DQK, DV, topk = 1, 1024, 8192, 128, 1, 576, 512, 2048
    dtype = torch.bfloat16
    q_start_s_index = 1024
    KV_stride = 1

    print(f"Smoke test: B={B}, S={S}, SKV={SKV}, H={H}, DQK={DQK}, DV={DV}, topk={topk}")

    q = torch.randn((B, S, H, DQK), dtype=dtype, device="npu") / 10
    kv = torch.randn((B, SKV, HKV, DQK), dtype=dtype, device="npu") / 10
    q.clamp_(-10, 10)
    kv.clamp_(-10, 10)
    q_start_t = torch.tensor([q_start_s_index], dtype=torch.int32, device="npu")

    indices = torch.full((B, S, HKV, topk), SKV - 1, dtype=torch.int32, device="npu")
    for b in range(B):
        for t in range(S):
            for h in range(HKV):
                avail = min(max(1, ((t + q_start_s_index) // KV_stride)), SKV)
                i_i = torch.randperm(avail)[:topk]
                indices[b, t, h, : len(i_i)] = i_i

    kernel = sparse_mla_fwd_pipelined(
        heads=H,
        dim=DV,
        tail_dim=DQK - DV,
        topk=topk,
        kv_stride=KV_stride,
        kv_group=HKV,
        sm_scale=None,
        is_causal=True,
    )
    out = kernel(q, kv, indices, q_start_t)
    torch.npu.synchronize()

    # Shape + dtype verification
    assert out.shape == (B, S, H, DV), f"shape mismatch: {out.shape} != {(B, S, H, DV)}"
    assert out.dtype == dtype, f"dtype mismatch: {out.dtype} != {dtype}"
    assert torch.isfinite(out).all(), "output contains non-finite values"

    print(f"Output shape={tuple(out.shape)}, dtype={out.dtype}, finite=True")
    print("Test Passed!")
