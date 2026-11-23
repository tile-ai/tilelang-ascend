"""Sparse MLA Forward Pipelined (DeepSeek V3.2 Multi-head Latent Attention).

Computes O = softmax(Q @ KV^T * sm_scale + causal_mask) @ KV[:dim], attending
only the top-K KV rows selected per query block.

Single kernel on a persistent grid, written in Developer mode: the Cube/Vector
split and all synchronization are handled by the compiler, so the kernel body
contains no explicit scope or barrier calls.

Design:
  - Cube:   GEMM1 (Q @ KV^T, K = dim + tail_dim) and GEMM2 (P @ KV[:dim])
  - Vector: sparse KV gather, online softmax, output accumulation
  - Four GM workspace tensors relay data between Cube and Vector
  - Causal mask built with T.tile.compare into a packed uint8 bitmask, then
    applied with T.tile.select
"""

import tilelang
import torch
from numbers import Integral

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
    block_I=128,
    core_num=20,
):
    """Build the sparse MLA forward kernel.

    Args:
        heads: number of query heads. Must be divisible by kv_group, and
            head_kv = heads // kv_group must be 16, 32 or 64, or a multiple of
            64 (with kv_group > 1, a power of two).
        dim: V dimension (512).
        tail_dim: rope/PE dimension (64).
        topk: number of sparse indices. Must be a multiple of block_I and
            at least 2 * block_I.
        kv_stride: KV stride, >= 1.
        kv_group: KV group count (1 for MLA). Must divide heads exactly.
        sm_scale: softmax scale (None => 1 / sqrt(dim + tail_dim)).
        is_causal: must be True.
        block_I: indices block size. Must be 128.
        core_num: persistent grid size.

    Returns:
        prim_func sparse_mla_fwd_kernel(Q, KV, Indices, q_start_index_s, Output,
        workspace_1/3/4/5)

    Input contract:
        For every query row t < kv_stride - 1 and every (b, g),
        Indices[b, t, g, :] must contain 0. Those rows have an empty causal
        window, so the kernel can only select index 0; if it is absent the whole
        row is masked to -inf and the output becomes NaN. Call
        check_indices_kv0_contract() to validate inputs before launching.

    Note:
        Lse is not produced: Ascend codegen does not support a scalar T.log.
    """
    # Accept any integer type (numpy scalars included) and normalize to int:
    # next_power_of_2 needs int.bit_length(), and a numpy dtype reaching
    # T.Pipelined(NI) cannot be handled by TVM.
    for _name, _val in (
        ("heads", heads),
        ("dim", dim),
        ("tail_dim", tail_dim),
        ("topk", topk),
        ("kv_stride", kv_stride),
        ("kv_group", kv_group),
        ("block_I", block_I),
        ("core_num", core_num),
    ):
        assert isinstance(_val, Integral), f"{_name} must be an integer, got {type(_val).__name__}: {_val!r}"
    heads, dim, tail_dim, topk = int(heads), int(dim), int(tail_dim), int(topk)
    kv_stride, kv_group, block_I, core_num = int(kv_stride), int(kv_group), int(block_I), int(core_num)

    assert dim == tilelang.math.next_power_of_2(dim), f"dim must be a power of 2, got {dim}"
    assert tail_dim == tilelang.math.next_power_of_2(tail_dim), f"tail_dim must be a power of 2, got {tail_dim}"
    assert is_causal, "non-causal is not supported"

    # T.tile.compare over a [block_I] fp32 operand requires 256B alignment, which
    # also covers the BI//8 packed mask and the BI//2 vector split.
    assert block_I == 128, f"block_I must be 128, got {block_I}"
    assert topk % block_I == 0, f"topk must be a multiple of block_I, got topk={topk}, block_I={block_I}"
    # NI >= 2: with a single iteration the pipelined KV buffer gets no L1 address plan.
    assert topk >= 2 * block_I, f"topk must be >= 2 * block_I, got topk={topk}, block_I={block_I}"

    assert heads >= 1, f"heads must be >= 1, got {heads}"
    assert kv_group >= 1, f"kv_group must be >= 1, got {kv_group}"
    # head_kv sizes padded_H, H_per_block and the grid, so a non-divisible pair
    # would truncate it and leave the trailing heads unwritten in Output.
    assert heads % kv_group == 0, f"heads must be divisible by kv_group, got heads={heads}, kv_group={kv_group}"

    # kv_stride is the device-side divisor of the causal boundary.
    assert kv_stride >= 1, f"kv_stride must be >= 1, got {kv_stride}"
    assert core_num >= 1, f"core_num must be >= 1, got {core_num}"

    # Ascend has no exp2 hardware support, so the kernel uses plain T.tile.exp
    # and sm_scale carries no log2(e) factor.
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

    if head_kv > 64:
        assert head_kv % 64 == 0, f"head_kv must be a multiple of 64 when > 64, got {head_kv}"
        if kv_group > 1:
            # The group base offset is computed from padded_H, so head_kv must
            # not be padded when several groups share the KV.
            assert head_kv == tilelang.math.next_power_of_2(head_kv), f"kv_group > 1 requires a power-of-two head_kv, got head_kv={head_kv}"
        REPLICATE_H = head_kv // 64
    else:
        # padded_H would round other values up to 16 and read Q / write Output
        # out of bounds.
        assert head_kv in (16, 32, 64), f"head_kv must be 16, 32 or 64 when <= 64, got {head_kv}"
        REPLICATE_H = 1

    H_per_block = padded_H if REPLICATE_H == 1 else 64
    v_block = H_per_block // 2

    # Every query row is scheduled, including rows whose causal window is empty.
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

                    # Causal boundary: the highest KV position this row may attend.
                    # Clamped at 0 so that rows with an empty causal window still
                    # get a well-formed mask; those rows can only select index 0
                    # (see the input contract in the factory docstring).
                    q_i = q_start_index_s[0] + s_i
                    raw = (q_i + 1) // KV_stride
                    max_kv_i = T.max(raw - 1, 0)

                    # ===== Load Q =====
                    T.copy(Q[b_i, s_i, H0 : H0 + H_per_block, :], q_full_l1)

                    # ===== Init online softmax state =====
                    T.tile.fill(acc_o, 0.0)
                    T.tile.fill(sumexp, 0.0)
                    T.tile.fill(m_i, -(2.0**30))

                    # Per index block: GEMM1 -> gather -> softmax -> GEMM2 ->
                    # accumulate. Ordering between the Cube and Vector queues over
                    # the four workspaces is inserted by the compiler.
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
# Input-contract validation
# ============================================================================


def check_indices_kv0_contract(indices, kv_stride, q_start_index_s=0, raise_on_violation=True):
    """Validate the early-row KV0 input contract required by the kernel.

    Rows with ``t < kv_stride - 1`` have an empty causal window, so the kernel's
    mask (``indices <= max_kv_i``, clamped to 0) can only select index 0. If 0 is
    absent from such a row, every score in it is masked to -inf and the output
    becomes NaN. This check verifies that 0 is present.

    Args:
        indices: int32 tensor of shape [B, S, kv_group, topk].
        kv_stride: KV stride passed to the kernel factory.
        q_start_index_s: scalar or 1-element int32 tensor. Used only to report
            the absolute row id in the error message.
        raise_on_violation: raise ValueError on the first violating row
            (default), or return the list of violating (b, t, g) tuples.

    Returns:
        Empty list if the contract holds, otherwise the violating (b, t, g)
        tuples (only when raise_on_violation is False).

    Raises:
        ValueError: if the contract is violated and raise_on_violation is True.
    """
    idx = indices.detach().cpu()
    if idx.dim() != 4:
        raise ValueError(f"indices must be 4-D [B, S, kv_group, topk], got shape {tuple(idx.shape)}")
    B, S, G, _topk = idx.shape

    n_early = max(0, min(int(kv_stride) - 1, S))
    if n_early == 0:
        return []  # kv_stride <= 1: no row has an empty causal window

    if torch.is_tensor(q_start_index_s):
        q_start = int(q_start_index_s.detach().cpu().reshape(-1)[0])
    else:
        q_start = int(q_start_index_s or 0)

    has_zero = (idx[:, :n_early] == 0).any(dim=-1)  # [B, n_early, G]
    bad = (~has_zero).nonzero().tolist()
    if not bad:
        return []

    violations = [(int(b), int(t), int(g)) for b, t, g in bad]
    if raise_on_violation:
        b, t, g = violations[0]
        raise ValueError(
            f"Indices violates the early-row KV0 contract at (b={b}, t={t}, g={g}): "
            f"row t < kv_stride - 1 (kv_stride={kv_stride}) has an empty causal window, "
            f"so the kernel can only select index 0, but 0 is absent from "
            f"Indices[{b}, {t}, {g}, :] (absolute q_i={q_start + t}). Every score in that "
            f"row would be masked to -inf and the output would become NaN. Inject index 0 "
            f"into the leading kv_stride - 1 rows. Total violating rows: {len(violations)}."
        )
    return violations


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

    # With kv_stride=1 there are no early rows, so this is a no-op; it shows
    # where a caller with kv_stride > 1 must validate.
    check_indices_kv0_contract(indices, KV_stride, q_start_s_index)

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
