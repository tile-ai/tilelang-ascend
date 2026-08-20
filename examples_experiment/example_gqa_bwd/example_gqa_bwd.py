"""
GQA Flash Attention Forward + Backward for Ascend NPU (Expert Mode).
Layout: BHSD (Batch, Heads, SeqLen, Dim).
Supports D_qk != D_v (separate dimensions for Q/K and V).
"""

import tilelang
from tilelang import DataType, language as T
from tilelang.intrinsics import make_zn_layout, make_nz_layout
import torch
import torch.nn.functional as F

# ============================================================================
# Common pass_configs for Expert mode
# ============================================================================

_expert_pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: False,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: False,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: False,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: False,
}

NUM_CORES = 24

# Signal IDs for pipeline backward C scope sync
_SIG_K = 0
_SIG_MN = 1
_SIG_V = 2
_SIG_K5 = 3
_SIG_L0C_MN = 0
_SIG_L0C_ND = 1

# ============================================================================
# Forward (GQA + LSE + causal mask, supports D_qk != D_v)
# ============================================================================


@tilelang.jit(out_idx=[3, 4], workspace_idx=[5, 6, 7], pass_configs=_expert_pass_configs)
def flashattn_fwd(batch, heads, seq_len, dim_qk, dim_v, is_causal, block_M, block_N, groups=1):
    """Forward: produces O [B,H,N,dim_v] fp16 and lse [B,H,N] fp32."""
    assert seq_len % block_M == 0, f"seq_len ({seq_len}) must be divisible by block_M ({block_M})"
    assert seq_len % block_N == 0, f"seq_len ({seq_len}) must be divisible by block_N ({block_N})"
    sm_scale = (1.0 / dim_qk) ** 0.5
    head_kv = heads // groups
    dtype = "float16"
    accum_dtype = "float"

    q_shape = [batch, heads, seq_len, dim_qk]
    k_shape = [batch, head_kv, seq_len, dim_qk]
    v_shape = [batch, head_kv, seq_len, dim_v]
    o_shape = [batch, heads, seq_len, dim_v]
    lse_shape = [batch, heads, seq_len]
    block_num = (seq_len // block_M) * heads * batch

    @T.prim_func
    def main(
        Q: T.Tensor(q_shape, dtype),
        K: T.Tensor(k_shape, dtype),
        V: T.Tensor(v_shape, dtype),
        Output: T.Tensor(o_shape, dtype),
        lse: T.Tensor(lse_shape, accum_dtype),
        workspace_1: T.Tensor([block_num, block_M, block_N], accum_dtype),
        workspace_2: T.Tensor([block_num, block_M, block_N], dtype),
        workspace_3: T.Tensor([block_num, block_M, dim_v], accum_dtype),
    ):
        with T.Kernel(block_num, is_npu=True) as (cid, vid):
            bx = cid % (seq_len // block_M)
            by = cid // (seq_len // block_M) % heads
            bz = cid // (seq_len // block_M) // heads % batch
            kv_by = by // groups

            q_l1 = T.alloc_L1([block_M, dim_qk], dtype)
            k_l1 = T.alloc_L1([block_N, dim_qk], dtype)
            v_l1 = T.alloc_L1([block_N, dim_v], dtype)
            acc_s_l1 = T.alloc_L1([block_M, block_N], dtype)
            acc_s_l0c = T.alloc_L0C([block_M, block_N], accum_dtype)
            acc_o_l0c = T.alloc_L0C([block_M, dim_v], accum_dtype)

            acc_o = T.alloc_ub([block_M // 2, dim_v], accum_dtype)
            sumexp = T.alloc_ub([block_M // 2], accum_dtype)
            m_i = T.alloc_ub([block_M // 2], accum_dtype)
            acc_s_ub = T.alloc_ub([block_M // 2, block_N], accum_dtype)
            m_i_prev = T.alloc_ub([block_M // 2], accum_dtype)
            acc_s_ub_ = T.alloc_ub([block_M // 2, block_N], accum_dtype)
            sumexp_i_ub = T.alloc_ub([block_M // 2], accum_dtype)
            acc_s_half = T.alloc_ub([block_M // 2, block_N], dtype)
            acc_o_ub = T.alloc_ub([block_M // 2, dim_v], accum_dtype)
            acc_o_half = T.alloc_ub([block_M // 2, dim_v], dtype)
            col_pos = T.alloc_ub([block_N], accum_dtype)
            cmp_mask = T.alloc_ub([block_N], accum_dtype)
            m_i_2d = T.alloc_ub([block_M // 2, block_N], accum_dtype)
            m_prev_2d = T.alloc_ub([block_M // 2, dim_v], accum_dtype)
            sumexp_2d = T.alloc_ub([block_M // 2, dim_v], accum_dtype)

            T.annotate_address(
                {
                    q_l1: 0,
                    k_l1: block_M * dim_qk * DataType(dtype).bits // 8,
                    acc_s_l1: block_M * dim_qk * DataType(dtype).bits // 8,
                    v_l1: block_M * (block_N + dim_qk) * DataType(dtype).bits // 8,
                    acc_s_l0c: 0,
                    acc_o_l0c: 0,
                    acc_o: 0,
                    sumexp: (block_M // 2) * dim_v * 4,
                    m_i: (block_M // 2) * dim_v * 4 + (block_M // 2) * 4,
                    acc_s_ub: (block_M // 2) * dim_v * 4 + (block_M // 2) * 4 + (block_M // 2) * 4,
                    m_i_prev: (block_M // 2) * dim_v * 4 + (block_M // 2) * 4 + (block_M // 2) * 4 + (block_M // 2) * block_N * 4,
                    acc_s_ub_: (block_M // 2) * dim_v * 4
                    + (block_M // 2) * 4
                    + (block_M // 2) * 4
                    + (block_M // 2) * block_N * 4
                    + (block_M // 2) * 4,
                    sumexp_i_ub: (block_M // 2) * dim_v * 4
                    + (block_M // 2) * 4
                    + (block_M // 2) * 4
                    + (block_M // 2) * block_N * 4
                    + (block_M // 2) * 4
                    + (block_M // 2) * block_N * 4,
                    acc_s_half: (block_M // 2) * dim_v * 4
                    + (block_M // 2) * 4
                    + (block_M // 2) * 4
                    + (block_M // 2) * block_N * 4
                    + (block_M // 2) * 4
                    + (block_M // 2) * block_N * 4
                    + (block_M // 2) * 4,
                    acc_o_ub: (block_M // 2) * dim_v * 4
                    + (block_M // 2) * 4
                    + (block_M // 2) * 4
                    + (block_M // 2) * block_N * 4
                    + (block_M // 2) * 4
                    + (block_M // 2) * block_N * 4
                    + (block_M // 2) * 4
                    + (block_M // 2) * block_N * 2,
                    acc_o_half: (block_M // 2) * dim_v * 4
                    + (block_M // 2) * 4
                    + (block_M // 2) * 4
                    + (block_M // 2) * block_N * 4
                    + (block_M // 2) * 4
                    + (block_M // 2) * block_N * 4
                    + (block_M // 2) * 4
                    + (block_M // 2) * block_N * 2
                    + (block_M // 2) * dim_v * 4,
                    m_i_2d: (block_M // 2) * dim_v * 4
                    + (block_M // 2) * 4
                    + (block_M // 2) * 4
                    + (block_M // 2) * block_N * 4
                    + (block_M // 2) * 4,
                    m_prev_2d: (block_M // 2) * dim_v * 4
                    + (block_M // 2) * 4
                    + (block_M // 2) * 4
                    + (block_M // 2) * block_N * 4
                    + (block_M // 2) * 4
                    + (block_M // 2) * block_N * 4
                    + (block_M // 2) * 4
                    + (block_M // 2) * block_N * 2,
                    sumexp_2d: (block_M // 2) * dim_v * 4
                    + (block_M // 2) * 4
                    + (block_M // 2) * 4
                    + (block_M // 2) * block_N * 4
                    + (block_M // 2) * 4
                    + (block_M // 2) * block_N * 4
                    + (block_M // 2) * 4
                    + (block_M // 2) * block_N * 2,
                }
            )

            with T.Scope("C"):
                T.copy(Q[bz, by, bx * block_M : (bx + 1) * block_M, :], q_l1)
                T.barrier_all()
                for k in T.serial(T.ceildiv((bx + 1) * block_M, block_N) if is_causal else T.ceildiv(seq_len, block_N)):
                    T.copy(K[bz, kv_by, k * block_N : (k + 1) * block_N, :], k_l1)
                    T.barrier_all()
                    T.gemm_v0(q_l1, k_l1, acc_s_l0c, transpose_B=True, init=True)
                    T.barrier_all()
                    T.copy(acc_s_l0c, workspace_1[cid, :, :])
                    T.barrier_all()
                    T.set_cross_flag("FIX", 0)
                    T.wait_cross_flag(1)
                    T.barrier_all()
                    T.copy(workspace_2[cid, :, :], acc_s_l1)
                    T.copy(V[bz, kv_by, k * block_N : (k + 1) * block_N, :], v_l1)
                    T.barrier_all()
                    T.gemm_v0(acc_s_l1, v_l1, acc_o_l0c, init=True)
                    T.barrier_all()
                    T.copy(acc_o_l0c, workspace_3[cid, :, :])
                    T.barrier_all()
                    T.set_cross_flag("FIX", 2)
                    T.wait_cross_flag(3)

            with T.Scope("V"):
                T.tile.fill(acc_o, 0.0)
                T.tile.fill(sumexp, 0.0)
                T.tile.fill(m_i, -(2**30))
                T.barrier_all()
                for _k in T.serial(T.ceildiv((bx + 1) * block_M, block_N) if is_causal else T.ceildiv(seq_len, block_N)):
                    T.tile.fill(acc_s_ub, 0.0)
                    T.copy(m_i, m_i_prev)
                    T.barrier_all()
                    T.wait_cross_flag(0)
                    T.copy(workspace_1[cid, vid * block_M // 2 : vid * block_M // 2 + block_M // 2, :], acc_s_ub_)
                    T.barrier_all()
                    T.tile.add(acc_s_ub, acc_s_ub, acc_s_ub_)
                    T.tile.mul(acc_s_ub, acc_s_ub, sm_scale)

                    if is_causal:
                        T.tile.arith_progression(col_pos, _k * block_N, 1, block_N)
                        for h_i in range(block_M // 2):
                            row_pos_val = (bx * block_M + vid * block_M // 2 + h_i) * 1.0
                            T.tile.compare(cmp_mask, col_pos, row_pos_val, "LE")
                            T.tile.select(acc_s_ub[h_i, :], cmp_mask, acc_s_ub[h_i, :], -T.infinity(accum_dtype), "VSEL_TENSOR_SCALAR_MODE")

                    T.reduce_max(acc_s_ub, m_i, dim=-1)
                    T.tile.max(m_i, m_i, m_i_prev)
                    T.tile.sub(m_i_prev, m_i_prev, m_i)
                    T.tile.exp(m_i_prev, m_i_prev)
                    T.tile.broadcast(m_i_2d, m_i, axis=1)
                    T.tile.sub(acc_s_ub, acc_s_ub, m_i_2d)
                    T.tile.exp(acc_s_ub, acc_s_ub)
                    T.reduce_sum(acc_s_ub, sumexp_i_ub, dim=-1)
                    T.tile.mul(sumexp, sumexp, m_i_prev)
                    T.tile.add(sumexp, sumexp, sumexp_i_ub)
                    T.tile.broadcast(m_prev_2d, m_i_prev, axis=1)
                    T.tile.mul(acc_o, acc_o, m_prev_2d)
                    T.copy(acc_s_ub, acc_s_half)
                    T.barrier_all()
                    T.copy(acc_s_half, workspace_2[cid, vid * block_M // 2 : vid * block_M // 2 + block_M // 2, :])
                    T.barrier_all()
                    T.set_cross_flag("MTE3", 1)
                    T.wait_cross_flag(2)
                    T.barrier_all()
                    T.copy(workspace_3[cid, vid * block_M // 2 : vid * block_M // 2 + block_M // 2, :], acc_o_ub)
                    T.barrier_all()
                    T.tile.add(acc_o, acc_o, acc_o_ub)
                    T.barrier_all()
                    T.set_cross_flag("V", 3)
                    T.barrier_all()
                T.tile.broadcast(sumexp_2d, sumexp, axis=1)
                T.tile.div(acc_o, acc_o, sumexp_2d)
                T.copy(acc_o, acc_o_half)
                T.barrier_all()
                T.copy(acc_o_half, Output[bz, by, bx * block_M + vid * block_M // 2 : bx * block_M + vid * block_M // 2 + block_M // 2, :])
                T.barrier_all()
                T.tile.ln(sumexp, sumexp)
                T.tile.add(sumexp, sumexp, m_i)
                T.barrier_all()
                T.copy(sumexp, lse[bz, by, bx * block_M + vid * block_M // 2 : bx * block_M + vid * block_M // 2 + block_M // 2])
                T.barrier_all()

    return main


# ============================================================================
# Forward v4: L0 double buffer + Fixed Core + batched softmax + fine-grained sync
# ============================================================================


@tilelang.jit(out_idx=[3, 4], workspace_idx=[5, 6, 7], pass_configs=_expert_pass_configs)
def flashattn_fwd_v4(
    batch,
    heads,
    seq_len,
    dim_qk,
    dim_v,
    is_causal,
    block_M=32,
    block_N=64,
    groups=1,
    num_stages=8,
    cross_interval=2,
):
    """Forward v4: block_M=32 enables L0 double buffering.

    Key changes from v3:
    - block_M=32 (was 64) → L0 fits double buffer for both GEMMs
    - L0 double buffering with [2, ...] syntax for GEMM1 and GEMM2
    - Batched iterations (num_stages per batch)
    - Fine-grained flag sync (set_flag/wait_flag) within C and V scopes
    - Persistent q_l1 ownership spans all KV batches in one logical task
    - Batched softmax (r_factors + sumexp_is, decoupled from O accumulation)
    """
    assert heads % groups == 0
    assert num_stages % 2 == 0, "num_stages must be even for double buffering"
    assert seq_len % block_M == 0, f"seq_len ({seq_len}) must be divisible by block_M ({block_M})"
    assert seq_len % block_N == 0, f"seq_len ({seq_len}) must be divisible by block_N ({block_N})"

    head_kv = heads // groups
    dtype = "float16"
    accum_dtype = "float"
    sm_scale = (1.0 / dim_qk) ** 0.5

    q_shape = [batch, heads, seq_len, dim_qk]
    k_shape = [batch, head_kv, seq_len, dim_qk]
    v_shape = [batch, head_kv, seq_len, dim_v]
    o_shape = [batch, heads, seq_len, dim_v]
    lse_shape = [batch, heads, seq_len]

    num_seq_blocks = seq_len // block_M
    block_num = num_seq_blocks * heads * batch
    num_iters = T.ceildiv(seq_len, block_N)
    half_M = block_M // 2

    q_tasks = block_num // NUM_CORES
    r_tasks = block_num % NUM_CORES

    # Cross-core semaphore IDs
    SEM_WS1_C2V = 0
    SEM_WS1_V2C = 1
    SEM_WS2_V2C = 2
    SEM_WS2_C2V = 3
    SEM_WS3_C2V = 4
    SEM_WS3_V2C = 5

    # Local event IDs are allocated per directed pipe pair. Different meanings keep
    # distinct names even when their directed pairs allow the same numeric ID.
    # MTE2 <-> MTE1
    SIG_K_L1 = 0
    SIG_P_L1 = 1
    SIG_V_L1 = 2
    SIG_Q_L1 = 3

    # MTE1 <-> M
    SIG_L0AB = 0  # double-buffer slots 0 and 1

    # M <-> FIX
    SIG_L0C = 0  # double-buffer slots 0 and 1

    # MTE2 <-> V
    SIG_IO_UB = 0

    # V <-> MTE3
    SIG_S_HALF = 0

    def task_range(cid_val):
        start = cid_val * q_tasks + T.if_then_else(cid_val < r_tasks, cid_val, r_tasks)
        count = q_tasks + T.if_then_else(cid_val < r_tasks, 1, 0)
        return start, count

    @T.prim_func
    def main(
        Q: T.Tensor(q_shape, dtype),
        K: T.Tensor(k_shape, dtype),
        V: T.Tensor(v_shape, dtype),
        Output: T.Tensor(o_shape, dtype),
        lse: T.Tensor(lse_shape, accum_dtype),
        workspace_1: T.Tensor([NUM_CORES, num_stages, block_M, block_N], accum_dtype),
        workspace_2: T.Tensor([NUM_CORES, num_stages, block_M, block_N], dtype),
        workspace_3: T.Tensor([NUM_CORES, num_stages, block_M, dim_v], accum_dtype),
    ):
        with T.Kernel(NUM_CORES, is_npu=True) as (cid, vid):
            # --- L1 buffers ---
            q_l1 = T.alloc_L1([block_M, dim_qk], dtype)
            k_l1 = T.alloc_L1([block_N, dim_qk], dtype)
            v_l1 = T.alloc_L1([block_N, dim_v], dtype)
            p_l1 = T.alloc_L1([block_M, block_N], dtype)

            T.annotate_layout(
                {
                    q_l1: make_zn_layout(q_l1),
                    k_l1: make_nz_layout(k_l1),
                    p_l1: make_zn_layout(p_l1),
                    v_l1: make_zn_layout(v_l1),
                }
            )

            # --- GEMM1 L0 double buffer (Q @ K^T) ---
            g1_l0a = T.alloc_L0A([2, block_M, dim_qk], dtype)
            g1_l0b = T.alloc_L0B([2, dim_qk, block_N], dtype)
            g1_l0c = T.alloc_L0C([2, block_M, block_N], accum_dtype)

            # --- GEMM2 L0 double buffer (P @ V) ---
            g2_l0a = T.alloc_L0A([2, block_M, block_N], dtype)
            g2_l0b = T.alloc_L0B([2, block_N, dim_v], dtype)
            g2_l0c = T.alloc_L0C([2, block_M, dim_v], accum_dtype)

            # Separate L0A and L0C to avoid double-buffer read/write conflicts
            # L0B can overlap (80KB > 64KB limit, but GEMM1/GEMM2 use L0B serially)
            T.annotate_address(
                {
                    g1_l0a: 0,
                    g2_l0a: 24576,  # 24KB offset, no overlap with g1_l0a
                    g1_l0b: 0,
                    g2_l0b: 0,  # overlaps g1_l0b (L0B total 80KB > 64KB)
                    g1_l0c: 0,
                    g2_l0c: 16384,  # 16KB offset, no overlap with g1_l0c
                }
            )

            # --- UB buffers ---
            # I/O intermediate buffers (for two-step workspace loads, fp32 to match workspace)
            io_buf = T.alloc_ub([half_M, block_N], accum_dtype)  # for S loads (fp32)
            io_buf_o = T.alloc_ub([half_M, dim_v], accum_dtype)  # for O loads (fp32)

            # Softmax computation
            work_ub = T.alloc_ub([half_M, block_N], accum_dtype)
            buf_2d = T.alloc_ub([half_M, block_N], accum_dtype)
            acc_s_half = T.alloc_ub([half_M, block_N], dtype)

            # O accumulation
            acc_o = T.alloc_ub([half_M, dim_v], accum_dtype)
            acc_o_ub = T.alloc_ub([half_M, dim_v], accum_dtype)
            broadcast_buf = T.alloc_ub([half_M, dim_v], accum_dtype)
            acc_o_half = T.alloc_ub([half_M, dim_v], dtype)

            # Batched softmax state
            r_factors = T.alloc_ub([num_stages, half_M, 1], accum_dtype)
            sumexp_is = T.alloc_ub([num_stages, half_M, 1], accum_dtype)
            sumexp = T.alloc_ub([half_M, 1], accum_dtype)
            neg_sm = T.alloc_ub([2, half_M, 1], accum_dtype)

            # LSE
            lse_buf = T.alloc_ub([half_M, 1], accum_dtype)

            # Causal mask (defined unconditionally so V scope can access them;
            # v1 defines them outside `if is_causal:` too — see line 90-91)
            col_pos = T.alloc_ub([block_N], accum_dtype)
            cmp_mask = T.alloc_ub([block_N], accum_dtype)

            my_start, my_count = task_range(cid)

            # ================================================================
            # C Scope — GEMM1 + GEMM2 with L0 double buffering
            # ================================================================
            with T.Scope("C"):
                # Init flags — pretend all buffers are free
                T.set_cross_flag("MTE2", SEM_WS2_C2V)
                T.set_flag("MTE1", "MTE2", SIG_K_L1)
                T.set_flag("MTE1", "MTE2", SIG_P_L1)
                T.set_flag("MTE1", "MTE2", SIG_V_L1)
                T.set_flag("MTE1", "MTE2", SIG_Q_L1)
                T.set_flag("M", "MTE1", SIG_L0AB)
                T.set_flag("M", "MTE1", SIG_L0AB + 1)
                T.set_flag("FIX", "M", SIG_L0C)
                T.set_flag("FIX", "M", SIG_L0C + 1)

                for t in T.serial(my_count):
                    task_id = my_start + t
                    bx = task_id % num_seq_blocks
                    by = (task_id // num_seq_blocks) % heads
                    bz = task_id // (num_seq_blocks * heads)
                    kv_by = by // groups

                    T.wait_flag("MTE1", "MTE2", SIG_Q_L1)
                    T.copy(Q[bz, by, bx * block_M : (bx + 1) * block_M, :], q_l1)
                    T.set_flag("MTE2", "MTE1", SIG_Q_L1)
                    T.wait_flag("MTE2", "MTE1", SIG_Q_L1)

                    _task_iters = T.ceildiv((bx + 1) * block_M, block_N) if is_causal else num_iters
                    _task_outer = T.ceildiv(_task_iters, num_stages)
                    for k in T.serial(_task_outer):
                        _remaining = _task_iters - k * num_stages
                        batch_iters = T.if_then_else(_remaining < num_stages, _remaining, num_stages)

                        # --- GEMM1 batch: Q @ K^T → workspace_1 ---
                        T.wait_cross_flag(SEM_WS1_V2C)
                        for i in T.serial(batch_iters):
                            side = i % 2
                            idx = k * num_stages + i

                            # Load K to L1
                            T.wait_flag("MTE1", "MTE2", SIG_K_L1)
                            T.copy(K[bz, kv_by, idx * block_N : (idx + 1) * block_N, :], k_l1)
                            T.set_flag("MTE2", "MTE1", SIG_K_L1)

                            # Copy Q to L0A (only first 2 iters — Q is constant)
                            T.wait_flag("M", "MTE1", SIG_L0AB + side)
                            if i < 2:
                                T.copy(q_l1, g1_l0a[side, :, :])

                            # Copy K to L0B (transposed)
                            T.wait_flag("MTE2", "MTE1", SIG_K_L1)
                            T.copy(k_l1, g1_l0b[side, :, :], transpose=True)
                            T.set_flag("MTE1", "MTE2", SIG_K_L1)
                            T.set_flag("MTE1", "M", SIG_L0AB + side)

                            # MMA
                            T.wait_flag("MTE1", "M", SIG_L0AB + side)
                            T.wait_flag("FIX", "M", SIG_L0C + side)
                            T.mma(g1_l0a[side, :, :], g1_l0b[side, :, :], g1_l0c[side, :, :], init=True)
                            T.set_flag("M", "MTE1", SIG_L0AB + side)
                            T.set_flag("M", "FIX", SIG_L0C + side)

                            # Copy result to workspace
                            T.wait_flag("M", "FIX", SIG_L0C + side)
                            T.copy(g1_l0c[side, :, :], workspace_1[cid, i, :, :])
                            T.set_flag("FIX", "M", SIG_L0C + side)
                            if (i + 1) % cross_interval == 0 or i == batch_iters - 1:
                                T.set_cross_flag("FIX", SEM_WS1_C2V)

                        # --- GEMM2 batch: P @ V → workspace_3 ---
                        T.wait_cross_flag(SEM_WS3_V2C)
                        for i in T.serial(batch_iters):
                            side = i % 2
                            idx = k * num_stages + i

                            # Load V to L1
                            T.wait_flag("MTE1", "MTE2", SIG_V_L1)
                            T.copy(V[bz, kv_by, idx * block_N : (idx + 1) * block_N, :], v_l1)
                            T.set_flag("MTE2", "MTE1", SIG_V_L1)

                            # Load P from workspace_2 to L1
                            T.wait_flag("MTE1", "MTE2", SIG_P_L1)
                            if i % cross_interval == 0:
                                T.wait_cross_flag(SEM_WS2_V2C)
                            T.copy(workspace_2[cid, i, :, :], p_l1)
                            T.set_flag("MTE2", "MTE1", SIG_P_L1)

                            # Copy V to L0B
                            T.wait_flag("MTE2", "MTE1", SIG_V_L1)
                            T.wait_flag("M", "MTE1", SIG_L0AB + side)
                            T.copy(v_l1, g2_l0b[side, :, :])
                            T.set_flag("MTE1", "MTE2", SIG_V_L1)

                            # Copy P to L0A
                            T.wait_flag("MTE2", "MTE1", SIG_P_L1)
                            T.copy(p_l1, g2_l0a[side, :, :])
                            T.set_flag("MTE1", "MTE2", SIG_P_L1)
                            T.set_flag("MTE1", "M", SIG_L0AB + side)

                            # MMA
                            T.wait_flag("MTE1", "M", SIG_L0AB + side)
                            T.wait_flag("FIX", "M", SIG_L0C + side)
                            T.mma(g2_l0a[side, :, :], g2_l0b[side, :, :], g2_l0c[side, :, :], init=True)
                            T.set_flag("M", "MTE1", SIG_L0AB + side)
                            T.set_flag("M", "FIX", SIG_L0C + side)

                            # Copy result to workspace
                            T.wait_flag("M", "FIX", SIG_L0C + side)
                            T.copy(g2_l0c[side, :, :], workspace_3[cid, i, :, :])
                            T.set_flag("FIX", "M", SIG_L0C + side)
                            if (i + 1) % cross_interval == 0 or i == batch_iters - 1:
                                T.set_cross_flag("FIX", SEM_WS3_C2V)

                        T.set_cross_flag("MTE2", SEM_WS2_C2V)

                    # MTE1 no longer reads q_l1; return it before the next task reloads Q.
                    T.set_flag("MTE1", "MTE2", SIG_Q_L1)

                # Destroy: consume outstanding init-direction flags
                T.wait_flag("MTE1", "MTE2", SIG_K_L1)
                T.wait_flag("MTE1", "MTE2", SIG_P_L1)
                T.wait_flag("MTE1", "MTE2", SIG_V_L1)
                T.wait_flag("MTE1", "MTE2", SIG_Q_L1)
                T.wait_flag("M", "MTE1", SIG_L0AB)
                T.wait_flag("M", "MTE1", SIG_L0AB + 1)
                T.wait_flag("FIX", "M", SIG_L0C)
                T.wait_flag("FIX", "M", SIG_L0C + 1)

            # ================================================================
            # V Scope — Batched softmax + O accumulation
            # ================================================================
            with T.Scope("V"):
                # Init flags
                T.set_cross_flag("MTE2", SEM_WS1_V2C)
                T.set_cross_flag("MTE2", SEM_WS3_V2C)
                T.set_flag("V", "MTE2", SIG_IO_UB)
                T.set_flag("MTE3", "V", SIG_S_HALF)

                for t in T.serial(my_count):
                    task_id = my_start + t
                    bx = task_id % num_seq_blocks
                    by = (task_id // num_seq_blocks) % heads
                    bz = task_id // (num_seq_blocks * heads)

                    T.tile.fill(acc_o, 0.0)
                    T.tile.fill(sumexp, 0.0)
                    T.tile.fill(neg_sm, 2**30)

                    _task_iters = T.ceildiv((bx + 1) * block_M, block_N) if is_causal else num_iters
                    _task_outer = T.ceildiv(_task_iters, num_stages)
                    for k in T.serial(_task_outer):
                        _remaining = _task_iters - k * num_stages
                        batch_iters = T.if_then_else(_remaining < num_stages, _remaining, num_stages)

                        # --- Softmax batch (reference two-step pattern) ---
                        T.wait_cross_flag(SEM_WS2_C2V)
                        for i in T.serial(batch_iters):
                            cur = i % 2
                            prv = 1 - cur

                            # Step 1: MTE2 loads S from workspace_1 → io_buf
                            T.wait_flag("V", "MTE2", SIG_IO_UB)
                            if i % cross_interval == 0:
                                T.wait_cross_flag(SEM_WS1_C2V)
                            T.copy(workspace_1[cid, i, vid * half_M : vid * half_M + half_M, :], io_buf)
                            T.set_flag("MTE2", "V", SIG_IO_UB)

                            # Step 2: V copies io_buf → work_ub (fp16→fp32)
                            T.wait_flag("MTE2", "V", SIG_IO_UB)
                            T.copy(io_buf, work_ub)
                            T.set_flag("V", "MTE2", SIG_IO_UB)

                            # Causal mask (V unit, operates on work_ub)
                            if is_causal:
                                idx = k * num_stages + i
                                T.tile.arith_progression(col_pos, idx * block_N, 1, block_N)
                                for h_i in range(half_M):
                                    row_pos_val = (bx * block_M + vid * half_M + h_i) * 1.0
                                    T.tile.compare(cmp_mask, col_pos, row_pos_val, "LE")
                                    T.tile.select(
                                        work_ub[h_i, :], cmp_mask, work_ub[h_i, :], -T.infinity(accum_dtype), "VSEL_TENSOR_SCALAR_MODE"
                                    )

                            # Batched softmax computation (V unit, on work_ub)
                            T.reduce_max(work_ub, neg_sm[cur, :, :], dim=-1)
                            T.tile.mul(neg_sm[cur, :, :], neg_sm[cur, :, :], -sm_scale)
                            T.tile.min(neg_sm[cur, :, :], neg_sm[cur, :, :], neg_sm[prv, :, :])
                            T.tile.broadcast(buf_2d, neg_sm[cur, :, :])
                            T.tile.axpy(buf_2d, work_ub, sm_scale)
                            T.tile.exp(work_ub, buf_2d)

                            # Store P: work_ub (fp32) → acc_s_half (fp16)
                            T.wait_flag("MTE3", "V", SIG_S_HALF)
                            T.copy(work_ub, acc_s_half)
                            T.set_flag("V", "MTE3", SIG_S_HALF)

                            # MTE3: acc_s_half → workspace_2 (GM)
                            T.wait_flag("V", "MTE3", SIG_S_HALF)
                            T.copy(acc_s_half, workspace_2[cid, i, vid * half_M : vid * half_M + half_M, :])
                            T.set_flag("MTE3", "V", SIG_S_HALF)
                            if (i + 1) % cross_interval == 0 or i == batch_iters - 1:
                                T.set_cross_flag("MTE3", SEM_WS2_V2C)

                            # sumexp_i and r_factor (V unit, reads work_ub)
                            T.reduce_sum(work_ub, sumexp_is[i, :, :], dim=-1)
                            T.tile.sub(r_factors[i, :, :], neg_sm[cur, :, :], neg_sm[prv, :, :])

                        T.set_cross_flag("MTE2", SEM_WS1_V2C)

                        # --- O accumulation batch (reference pattern) ---
                        for i in T.serial(batch_iters):
                            # Rescale acc_o with r_factor
                            T.tile.exp(r_factors[i, :, :], r_factors[i, :, :])
                            T.tile.mul(sumexp, sumexp, r_factors[i, :, :])
                            T.tile.add(sumexp, sumexp, sumexp_is[i, :, :])
                            T.tile.broadcast(broadcast_buf, r_factors[i, :, :])
                            T.tile.mul(acc_o, acc_o, broadcast_buf)

                            # Step 1: MTE2 loads O from workspace_3 → io_buf_o
                            T.wait_flag("V", "MTE2", SIG_IO_UB)
                            if i % cross_interval == 0:
                                T.wait_cross_flag(SEM_WS3_C2V)
                            T.copy(workspace_3[cid, i, vid * half_M : vid * half_M + half_M, :], io_buf_o)
                            T.set_flag("MTE2", "V", SIG_IO_UB)

                            # Step 2: V copies io_buf_o → acc_o_ub (fp16→fp32)
                            T.wait_flag("MTE2", "V", SIG_IO_UB)
                            T.copy(io_buf_o, acc_o_ub)
                            T.set_flag("V", "MTE2", SIG_IO_UB)

                            T.tile.add(acc_o, acc_o, acc_o_ub)

                        T.set_cross_flag("MTE2", SEM_WS3_V2C)

                    # Final normalization: acc_o /= sumexp
                    T.tile.broadcast(broadcast_buf, sumexp)
                    T.tile.div(acc_o, acc_o, broadcast_buf)

                    # Write output (fp32 → fp16 → GM)
                    T.copy(acc_o, acc_o_half)
                    T.barrier_all()
                    T.copy(acc_o_half, Output[bz, by, bx * block_M + vid * half_M : bx * block_M + vid * half_M + half_M, :])

                    # LSE = ln(sumexp) - neg_sm[0]
                    # neg_sm[0] = -sm_scale * global_max after min merge
                    T.tile.min(neg_sm[0, :, :], neg_sm[0, :, :], neg_sm[1, :, :])
                    T.tile.ln(lse_buf, sumexp)
                    T.tile.sub(lse_buf, lse_buf, neg_sm[0, :, :])
                    T.barrier_all()
                    T.copy(lse_buf, lse[bz, by, bx * block_M + vid * half_M : bx * block_M + vid * half_M + half_M])
                    T.barrier_all()

                # Destroy: consume outstanding init-direction flags
                T.wait_flag("V", "MTE2", SIG_IO_UB)
                T.wait_flag("MTE3", "V", SIG_S_HALF)

    return main


# ============================================================================
# Backward Preprocess — Delta = sum(O * dO, dim=-1)
# ============================================================================


@tilelang.jit(out_idx=[2], pass_configs=_expert_pass_configs)
def flashattn_bwd_preprocess(batch, heads, seq_len, dim_v, blk=32):
    dtype = "float16"
    accum_dtype = "float"
    shape = [batch, heads, seq_len, dim_v]
    block_num = heads * (seq_len // blk) * batch

    @T.prim_func
    def main(
        O: T.Tensor(shape, dtype),
        dO: T.Tensor(shape, dtype),
        Delta: T.Tensor([batch, heads, seq_len], accum_dtype),
    ):
        with T.Kernel(block_num, is_npu=True) as (cid, vid):
            by = cid % (seq_len // blk)
            bx = cid // (seq_len // blk) % heads
            bz = cid // (seq_len // blk) // heads % batch

            o_ub = T.alloc_ub([blk // 2, dim_v], dtype)
            do_ub = T.alloc_ub([blk // 2, dim_v], dtype)
            sum_ub = T.alloc_ub([blk // 2, dim_v], accum_dtype)
            prod_ub = T.alloc_ub([blk // 2, dim_v], accum_dtype)
            do_fp32 = T.alloc_ub([blk // 2, dim_v], accum_dtype)
            delta_ub = T.alloc_ub([blk // 2], accum_dtype)

            T.annotate_address(
                {
                    o_ub: 0,
                    do_ub: blk // 2 * dim_v * DataType(dtype).bits // 8,
                    sum_ub: blk // 2 * dim_v * 2 * DataType(dtype).bits // 8,
                    prod_ub: blk // 2 * dim_v * 2 * DataType(dtype).bits // 8 + blk // 2 * dim_v * DataType(accum_dtype).bits // 8,
                    do_fp32: blk // 2 * dim_v * 2 * DataType(dtype).bits // 8 + 2 * blk // 2 * dim_v * DataType(accum_dtype).bits // 8,
                    delta_ub: blk // 2 * dim_v * 2 * DataType(dtype).bits // 8 + 3 * blk // 2 * dim_v * DataType(accum_dtype).bits // 8,
                }
            )

            with T.Scope("V"):
                T.tile.fill(sum_ub, 0.0)
                T.barrier_all()
                for _k in T.serial(T.ceildiv(dim_v, dim_v)):
                    T.copy(O[bz, bx, by * blk + vid * blk // 2 : by * blk + vid * blk // 2 + blk // 2, :], o_ub)
                    T.copy(dO[bz, bx, by * blk + vid * blk // 2 : by * blk + vid * blk // 2 + blk // 2, :], do_ub)
                    T.barrier_all()
                    T.copy(o_ub, prod_ub)
                    T.copy(do_ub, do_fp32)
                    T.barrier_all()
                    T.tile.mul(prod_ub, prod_ub, do_fp32)
                    T.barrier_all()
                    T.tile.add(sum_ub, sum_ub, prod_ub)
                    T.barrier_all()
                T.reduce_sum(sum_ub, delta_ub, dim=-1)
                T.barrier_all()
                T.copy(delta_ub, Delta[bz, bx, by * blk + vid * blk // 2 : by * blk + vid * blk // 2 + blk // 2])
                T.barrier_all()

    return main


# ============================================================================
# Backward Postprocess — dQ fp32 -> fp16
# ============================================================================


@tilelang.jit(out_idx=[1], pass_configs=_expert_pass_configs)
def flashattn_bwd_postprocess(batch, heads, seq_len, dim_qk, blk=64):
    dtype = "float16"
    accum_dtype = "float"
    shape = [batch, heads, seq_len, dim_qk]
    block_num = (seq_len // blk) * heads * batch

    @T.prim_func
    def main(
        dQ: T.Tensor(shape, accum_dtype),
        dQ_out: T.Tensor(shape, dtype),
    ):
        with T.Kernel(block_num, is_npu=True) as (cid, vid):
            bx = cid % (seq_len // blk)
            by = cid // (seq_len // blk) % heads
            bz = cid // (seq_len // blk) // heads % batch

            dq_ub = T.alloc_ub([blk // 2, dim_qk], accum_dtype)
            dq_half = T.alloc_ub([blk // 2, dim_qk], dtype)

            T.annotate_address(
                {
                    dq_ub: 0,
                    dq_half: blk // 2 * dim_qk * DataType(accum_dtype).bits // 8,
                }
            )

            with T.Scope("V"):
                T.copy(dQ[bz, by, bx * blk + vid * blk // 2 : bx * blk + vid * blk // 2 + blk // 2, :], dq_ub)
                T.barrier_all()
                T.copy(dq_ub, dq_half)
                T.barrier_all()
                T.copy(dq_half, dQ_out[bz, by, bx * blk + vid * blk // 2 : bx * blk + vid * blk // 2 + blk // 2, :])
                T.barrier_all()

    return main


# ============================================================================
# Backward Pipeline: fine-grained set_flag/wait_flag sync in C scope
# ============================================================================


@tilelang.jit(pass_configs=_expert_pass_configs)
def flashattn_bwd_pipeline(
    batch,
    heads,
    seq_len,
    dim_qk,
    dim_v,
    is_causal,
    block_M,
    block_N,
    groups=1,
    num_stages=8,
):
    """Backward kernel with fine-grained pipeline sync in C scope.

    Replaces barrier_all() with set_flag/wait_flag within C scope phases
    to overlap MTE2 (GM->L1 data loading), M (MMA compute via gemm_v0),
    and FIX (L0C->GM result writing).

    Phase structure per batch:
      Phase 1 (C): GEMM1 for all iters -> ws_s_dp[cid, i, :, :]
      Phase 2 (V): softmax for all iters -> ws_p_ds[cid, i, :, :]
      Phase 3 (C): GEMM2+GEMM3 for all iters
      Phase 4 (V): atomic_add dV, compute dS -> ws_p_ds[cid, i, :, :]
      Phase 5 (C): GEMM4+GEMM5 for all iters

    cross_flag: kept between phases for C-V synchronization.
    set_flag/wait_flag: used within phases for C-internal pipeline overlap.
    """
    assert seq_len % block_M == 0, f"seq_len ({seq_len}) must be divisible by block_M ({block_M})"
    assert seq_len % block_N == 0, f"seq_len ({seq_len}) must be divisible by block_N ({block_N})"
    sm_scale = (1.0 / dim_qk) ** 0.5
    head_kv = heads // groups
    dtype = "float16"
    accum_dtype = "float"
    dim_qk_padded = ((dim_qk + 127) // 128) * 128

    q_shape = [batch, heads, seq_len, dim_qk_padded]
    k_shape = [batch, head_kv, seq_len, dim_qk_padded]
    v_shape = [batch, head_kv, seq_len, dim_v]
    do_shape = [batch, heads, seq_len, dim_v]
    dq_shape_padded = [batch, heads, seq_len, dim_qk_padded]
    dk_shape_padded = [batch, head_kv, seq_len, dim_qk_padded]
    bwd_block_num = (seq_len // block_M) * heads * batch

    num_iters = seq_len // block_N
    eff_stages = num_stages if num_stages <= num_iters else num_iters
    # Use ceildiv to ensure all iterations are processed
    num_outer = (num_iters + eff_stages - 1) // eff_stages

    @T.prim_func
    def main(
        Q: T.Tensor(q_shape, dtype),
        K: T.Tensor(k_shape, dtype),
        V: T.Tensor(v_shape, dtype),
        dO: T.Tensor(do_shape, dtype),
        lse: T.Tensor([batch, heads, seq_len], accum_dtype),
        Delta: T.Tensor([batch, heads, seq_len], accum_dtype),
        dQ: T.Tensor(dq_shape_padded, accum_dtype),
        dK: T.Tensor(dk_shape_padded, accum_dtype),
        dV: T.Tensor(v_shape, accum_dtype),
        ws_s_dp: T.Tensor([bwd_block_num, num_stages, block_M, block_N], accum_dtype),
        ws_p_ds: T.Tensor([bwd_block_num, num_stages, block_M, block_N], dtype),
        ws_dv_dk: T.Tensor([bwd_block_num, num_stages, block_N, max(dim_qk_padded, dim_v)], accum_dtype),
    ):
        with T.Kernel(bwd_block_num, is_npu=True) as (cid, vid):
            bx = cid % (seq_len // block_M)
            by = cid // (seq_len // block_M) % heads
            bz = cid // (seq_len // block_M) // heads % batch
            kv_by = by // groups

            q_l1 = T.alloc_L1([block_M, dim_qk_padded], dtype)
            do_l1 = T.alloc_L1([block_M, dim_v], dtype)
            k_l1 = T.alloc_L1([block_N, dim_qk_padded], dtype)
            v_l1 = T.alloc_L1([block_N, dim_v], dtype)
            mn_l1 = T.alloc_L1([block_M, block_N], dtype)
            k5_l1 = T.alloc_L1([block_N, dim_qk_padded], dtype)

            l0c_mn = T.alloc_L0C([block_M, block_N], accum_dtype)
            l0c_nd_v = T.alloc_L0C([block_N, dim_v], accum_dtype)
            l0c_nd_qk = T.alloc_L0C([block_N, dim_qk_padded], accum_dtype)
            l0c_dq = T.alloc_L0C([block_M, dim_qk_padded], accum_dtype)

            work_ub = T.alloc_ub([block_M // 2, block_N], accum_dtype)
            dp_ub = T.alloc_ub([block_M // 2, block_N], accum_dtype)
            p_half = T.alloc_ub([block_M // 2, block_N], dtype)
            lse_ub = T.alloc_ub([block_M // 2], accum_dtype)
            delta_ub = T.alloc_ub([block_M // 2], accum_dtype)
            dv_tmp = T.alloc_ub([block_N // 2, max(dim_qk_padded, dim_v)], accum_dtype)
            col_pos = T.alloc_ub([block_N], accum_dtype)
            cmp_mask = T.alloc_ub([block_N], accum_dtype)

            T.annotate_address(
                {
                    q_l1: 0,
                    do_l1: block_M * dim_qk_padded * DataType(dtype).bits // 8,
                    k_l1: (block_M * dim_qk_padded * DataType(dtype).bits // 8 + block_M * dim_v * DataType(dtype).bits // 8),
                    v_l1: (
                        block_M * dim_qk_padded * DataType(dtype).bits // 8
                        + block_M * dim_v * DataType(dtype).bits // 8
                        + block_N * dim_qk_padded * DataType(dtype).bits // 8
                    ),
                    mn_l1: (
                        block_M * dim_qk_padded * DataType(dtype).bits // 8
                        + block_M * dim_v * DataType(dtype).bits // 8
                        + block_N * dim_qk_padded * DataType(dtype).bits // 8
                        + block_N * dim_v * DataType(dtype).bits // 8
                    ),
                    k5_l1: (
                        block_M * dim_qk_padded * DataType(dtype).bits // 8
                        + block_M * dim_v * DataType(dtype).bits // 8
                        + block_N * dim_qk_padded * DataType(dtype).bits // 8
                        + block_N * dim_v * DataType(dtype).bits // 8
                        + block_M * block_N * DataType(dtype).bits // 8
                    ),
                    l0c_mn: 0,
                    l0c_nd_v: block_M * block_N * DataType(accum_dtype).bits // 8,
                    l0c_nd_qk: block_M * block_N * DataType(accum_dtype).bits // 8,
                    l0c_dq: (block_M * block_N + block_N * max(dim_v, dim_qk_padded)) * DataType(accum_dtype).bits // 8,
                    work_ub: 0,
                    dp_ub: block_M // 2 * block_N * DataType(accum_dtype).bits // 8,
                    p_half: 2 * block_M // 2 * block_N * DataType(accum_dtype).bits // 8,
                    lse_ub: (
                        2 * block_M // 2 * block_N * DataType(accum_dtype).bits // 8 + block_M // 2 * block_N * DataType(dtype).bits // 8
                    ),
                    delta_ub: (
                        2 * block_M // 2 * block_N * DataType(accum_dtype).bits // 8
                        + block_M // 2 * block_N * DataType(dtype).bits // 8
                        + block_M // 2 * DataType(accum_dtype).bits // 8
                    ),
                    dv_tmp: (
                        2 * block_M // 2 * block_N * DataType(accum_dtype).bits // 8
                        + block_M // 2 * block_N * DataType(dtype).bits // 8
                        + 2 * block_M // 2 * DataType(accum_dtype).bits // 8
                    ),
                    col_pos: (
                        2 * block_M // 2 * block_N * DataType(accum_dtype).bits // 8
                        + block_M // 2 * block_N * DataType(dtype).bits // 8
                        + 2 * block_M // 2 * DataType(accum_dtype).bits // 8
                        + block_N // 2 * max(dim_qk_padded, dim_v) * DataType(accum_dtype).bits // 8
                    ),
                    cmp_mask: (
                        2 * block_M // 2 * block_N * DataType(accum_dtype).bits // 8
                        + block_M // 2 * block_N * DataType(dtype).bits // 8
                        + 2 * block_M // 2 * DataType(accum_dtype).bits // 8
                        + block_N // 2 * max(dim_qk_padded, dim_v) * DataType(accum_dtype).bits // 8
                        + block_N * DataType(accum_dtype).bits // 8
                    ),
                }
            )

            with T.Scope("C"):
                T.copy(Q[bz, by, bx * block_M : (bx + 1) * block_M, :], q_l1)
                T.copy(dO[bz, by, bx * block_M : (bx + 1) * block_M, :], do_l1)
                T.barrier_all()

                T.set_flag("M", "MTE2", _SIG_K)
                T.set_flag("M", "MTE2", _SIG_MN)
                T.set_flag("M", "MTE2", _SIG_V)
                T.set_flag("M", "MTE2", _SIG_K5)
                T.set_flag("FIX", "M", _SIG_L0C_MN)
                T.set_flag("FIX", "M", _SIG_L0C_ND)

                for k_outer in range(num_outer):
                    batch_iters = eff_stages

                    for i in range(batch_iters):
                        idx = k_outer * num_stages + i

                        T.wait_flag("M", "MTE2", _SIG_K)
                        T.copy(K[bz, kv_by, idx * block_N : (idx + 1) * block_N, :], k_l1)
                        T.set_flag("MTE2", "M", _SIG_K)

                        T.wait_flag("MTE2", "M", _SIG_K)
                        T.wait_flag("FIX", "M", _SIG_L0C_MN)
                        T.gemm_v0(q_l1, k_l1, l0c_mn, transpose_B=True, init=True)
                        T.set_flag("M", "FIX", _SIG_L0C_MN)
                        T.set_flag("M", "MTE2", _SIG_K)

                        T.wait_flag("M", "FIX", _SIG_L0C_MN)
                        T.copy(l0c_mn, ws_s_dp[cid, i, :, :])
                        T.set_flag("FIX", "M", _SIG_L0C_MN)

                    T.set_cross_flag("FIX", 0)

                    for i in range(batch_iters):
                        idx = k_outer * num_stages + i

                        T.wait_cross_flag(1)
                        T.barrier_all()

                        T.wait_flag("M", "MTE2", _SIG_MN)
                        T.copy(ws_p_ds[cid, i, :, :], mn_l1)
                        T.set_flag("MTE2", "M", _SIG_MN)

                        T.wait_flag("MTE2", "M", _SIG_MN)
                        T.wait_flag("FIX", "M", _SIG_L0C_ND)
                        T.gemm_v0(mn_l1, do_l1, l0c_nd_v, transpose_A=True, init=True)
                        T.set_flag("M", "FIX", _SIG_L0C_ND)
                        T.set_flag("M", "MTE2", _SIG_MN)

                        T.wait_flag("M", "FIX", _SIG_L0C_ND)
                        T.copy(l0c_nd_v, ws_dv_dk[cid, i, :, :])
                        T.set_flag("FIX", "M", _SIG_L0C_ND)

                        T.wait_flag("M", "MTE2", _SIG_V)
                        T.copy(V[bz, kv_by, idx * block_N : (idx + 1) * block_N, :], v_l1)
                        T.set_flag("MTE2", "M", _SIG_V)

                        T.wait_flag("MTE2", "M", _SIG_V)
                        T.wait_flag("FIX", "M", _SIG_L0C_MN)
                        T.gemm_v0(do_l1, v_l1, l0c_mn, transpose_B=True, init=True)
                        T.set_flag("M", "FIX", _SIG_L0C_MN)
                        T.set_flag("M", "MTE2", _SIG_V)

                        T.wait_flag("M", "FIX", _SIG_L0C_MN)
                        T.copy(l0c_mn, ws_s_dp[cid, i, :, :])
                        T.set_flag("FIX", "M", _SIG_L0C_MN)

                    T.set_cross_flag("FIX", 2)

                    for i in range(batch_iters):
                        idx = k_outer * num_stages + i

                        T.wait_cross_flag(3)
                        T.barrier_all()

                        T.wait_flag("M", "MTE2", _SIG_MN)
                        T.copy(ws_p_ds[cid, i, :, :], mn_l1)
                        T.set_flag("MTE2", "M", _SIG_MN)

                        T.wait_flag("MTE2", "M", _SIG_MN)
                        T.wait_flag("FIX", "M", _SIG_L0C_ND)
                        T.gemm_v0(mn_l1, q_l1, l0c_nd_qk, transpose_A=True, init=True)
                        T.set_flag("M", "FIX", _SIG_L0C_ND)

                        T.wait_flag("M", "FIX", _SIG_L0C_ND)
                        T.copy(l0c_nd_qk, ws_dv_dk[cid, i, :, :])
                        T.set_flag("FIX", "M", _SIG_L0C_ND)

                        T.set_cross_flag("FIX", 4)
                        T.barrier_all()

                        T.wait_flag("M", "MTE2", _SIG_K5)
                        T.copy(K[bz, kv_by, idx * block_N : (idx + 1) * block_N, :], k5_l1)
                        T.set_flag("MTE2", "M", _SIG_K5)

                        T.wait_flag("MTE2", "M", _SIG_K5)
                        if k_outer == 0 and i == 0:
                            T.gemm_v0(mn_l1, k5_l1, l0c_dq, init=True)
                        else:
                            T.gemm_v0(mn_l1, k5_l1, l0c_dq, init=False)
                        T.set_flag("M", "MTE2", _SIG_MN)
                        T.set_flag("M", "MTE2", _SIG_K5)

                T.barrier_all()
                T.copy(l0c_dq, dQ[bz, by, bx * block_M : (bx + 1) * block_M, :])
                T.barrier_all()

                T.wait_flag("M", "MTE2", _SIG_K)
                T.wait_flag("M", "MTE2", _SIG_MN)
                T.wait_flag("M", "MTE2", _SIG_V)
                T.wait_flag("M", "MTE2", _SIG_K5)
                T.wait_flag("FIX", "M", _SIG_L0C_MN)
                T.wait_flag("FIX", "M", _SIG_L0C_ND)

            with T.Scope("V"):
                for _k_outer in range(num_outer):
                    batch_iters = eff_stages

                    T.wait_cross_flag(0)
                    T.barrier_all()

                    for i in range(batch_iters):
                        idx = _k_outer * num_stages + i

                        T.copy(ws_s_dp[cid, i, vid * block_M // 2 : vid * block_M // 2 + block_M // 2, :], work_ub)
                        T.barrier_all()

                        T.tile.mul(work_ub, work_ub, sm_scale)

                        T.copy(lse[bz, by, bx * block_M + vid * block_M // 2 : bx * block_M + vid * block_M // 2 + block_M // 2], lse_ub)
                        T.barrier_all()
                        for h_i in range(block_M // 2):
                            T.tile.sub(work_ub[h_i, :], work_ub[h_i, :], lse_ub[h_i])
                        T.tile.exp(work_ub, work_ub)

                        if is_causal and block_N >= 64:
                            T.tile.arith_progression(col_pos, idx * block_N, 1, block_N)
                            for h_i in range(block_M // 2):
                                row_pos_val = (bx * block_M + vid * block_M // 2 + h_i) * 1.0
                                T.tile.compare(cmp_mask, col_pos, row_pos_val, "LE")
                                T.tile.select(work_ub[h_i, :], cmp_mask, work_ub[h_i, :], 0.0, "VSEL_TENSOR_SCALAR_MODE")

                        T.copy(work_ub, p_half)
                        T.barrier_all()
                        T.copy(p_half, ws_p_ds[cid, i, vid * block_M // 2 : vid * block_M // 2 + block_M // 2, :])
                        T.barrier_all()
                        T.set_cross_flag("MTE3", 1)

                    T.wait_cross_flag(2)
                    T.barrier_all()

                    for i in range(batch_iters):
                        idx = _k_outer * num_stages + i

                        T.copy(ws_dv_dk[cid, i, vid * block_N // 2 : vid * block_N // 2 + block_N // 2, :dim_v], dv_tmp)
                        T.barrier_all()
                        T.tile.atomic_add(
                            dV[bz, kv_by, idx * block_N + vid * block_N // 2 : idx * block_N + vid * block_N // 2 + block_N // 2, :], dv_tmp
                        )

                        T.copy(ws_s_dp[cid, i, vid * block_M // 2 : vid * block_M // 2 + block_M // 2, :], dp_ub)
                        T.barrier_all()

                        T.copy(ws_p_ds[cid, i, vid * block_M // 2 : vid * block_M // 2 + block_M // 2, :], p_half)
                        T.barrier_all()
                        T.copy(p_half, work_ub)

                        T.copy(
                            Delta[bz, by, bx * block_M + vid * block_M // 2 : bx * block_M + vid * block_M // 2 + block_M // 2], delta_ub
                        )
                        T.barrier_all()

                        for h_i in range(block_M // 2):
                            T.tile.sub(dp_ub[h_i, :], dp_ub[h_i, :], delta_ub[h_i])
                        T.tile.mul(work_ub, work_ub, dp_ub)
                        T.tile.mul(work_ub, work_ub, sm_scale)

                        T.copy(work_ub, p_half)
                        T.barrier_all()
                        T.copy(p_half, ws_p_ds[cid, i, vid * block_M // 2 : vid * block_M // 2 + block_M // 2, :])
                        T.barrier_all()
                        T.set_cross_flag("V", 3)

                    for i in range(batch_iters):
                        idx = _k_outer * num_stages + i
                        T.wait_cross_flag(4)
                        T.barrier_all()
                        T.copy(ws_dv_dk[cid, i, vid * block_N // 2 : vid * block_N // 2 + block_N // 2, :dim_qk_padded], dv_tmp)
                        T.barrier_all()
                        T.tile.atomic_add(
                            dK[bz, kv_by, idx * block_N + vid * block_N // 2 : idx * block_N + vid * block_N // 2 + block_N // 2, :], dv_tmp
                        )
                        T.barrier_all()

    return main


# ============================================================================
# Golden Reference
# ============================================================================


def ref_program(Q, K, V, is_causal, groups=1):
    Q_f = Q.float()
    K_f = K.float().repeat_interleave(groups, dim=1) if groups > 1 else K.float()
    V_f = V.float().repeat_interleave(groups, dim=1) if groups > 1 else V.float()
    dim_qk = Q_f.shape[-1]
    scale = 1.0 / (dim_qk**0.5)
    scores = torch.matmul(Q_f, K_f.transpose(-2, -1)) * scale
    if is_causal:
        N = Q.shape[2]
        mask = torch.tril(torch.ones(N, N, device=scores.device, dtype=torch.bool))
        scores = scores.masked_fill(~mask.unsqueeze(0).unsqueeze(0), float("-inf"))
    P = F.softmax(scores, dim=-1)
    O = torch.matmul(P, V_f)
    return O.half()


def ref_bwd(Q, K, V, dO, is_causal, groups=1):
    Q_f = Q.float().requires_grad_(True)
    K_f = K.float().requires_grad_(True)
    V_f = V.float().requires_grad_(True)
    K_rep = K_f.repeat_interleave(groups, dim=1) if groups > 1 else K_f
    V_rep = V_f.repeat_interleave(groups, dim=1) if groups > 1 else V_f
    dim_qk = Q_f.shape[-1]
    scale = 1.0 / (dim_qk**0.5)
    scores = torch.matmul(Q_f, K_rep.transpose(-2, -1)) * scale
    if is_causal:
        N = Q.shape[2]
        mask = torch.tril(torch.ones(N, N, device=scores.device, dtype=torch.bool))
        scores = scores.masked_fill(~mask.unsqueeze(0).unsqueeze(0), float("-inf"))
    P = F.softmax(scores, dim=-1)
    O = torch.matmul(P, V_rep)
    O.backward(dO.float())
    return Q_f.grad.half(), K_f.grad.half(), V_f.grad.half()


# ============================================================================
# Autograd Function (BSHD <-> BHSD layout conversion)
# ============================================================================


class _attention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, causal, groups=1):
        B, N, H, D_qk = q.shape
        D_v = v.shape[-1]
        q_bhsd = q.permute(0, 2, 1, 3).contiguous()
        k_bhsd = k.permute(0, 2, 1, 3).contiguous()
        v_bhsd = v.permute(0, 2, 1, 3).contiguous()

        block_M = 64
        mod = flashattn_fwd(B, H, N, D_qk, D_v, causal, block_M, block_M, groups)
        o_bhsd, lse = mod(q_bhsd, k_bhsd, v_bhsd)

        o = o_bhsd.permute(0, 2, 1, 3).contiguous()

        ctx.save_for_backward(q_bhsd, k_bhsd, v_bhsd, o_bhsd, lse)
        ctx.causal = causal
        ctx.groups = groups
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, lse = ctx.saved_tensors
        B, H, N, D_qk = q.shape
        D_v = v.shape[-1]
        H_kv = k.shape[1]
        groups = ctx.groups

        do_bhsd = do.permute(0, 2, 1, 3).contiguous()

        prep_mod = flashattn_bwd_preprocess(B, H, N, D_v, blk=32)
        delta = prep_mod(o, do_bhsd)

        block_M, block_N = 64, 32
        if ctx.causal:
            block_N = 64

        dim_qk_padded = ((D_qk + 127) // 128) * 128
        num_stages = 8

        dQ = torch.zeros(B, H, N, dim_qk_padded, dtype=torch.float32, device="npu")
        dK = torch.zeros(B, H_kv, N, dim_qk_padded, dtype=torch.float32, device="npu")
        dV = torch.zeros(B, H_kv, N, D_v, dtype=torch.float32, device="npu")

        Q_padded = torch.zeros(B, H, N, dim_qk_padded, dtype=torch.float16, device="npu")
        Q_padded[:, :, :, :D_qk] = q
        K_padded = torch.zeros(B, H_kv, N, dim_qk_padded, dtype=torch.float16, device="npu")
        K_padded[:, :, :, :D_qk] = k

        bwd_block_num = (N // block_M) * H * B
        ws_s_dp = torch.empty(bwd_block_num, num_stages, block_M, block_N, dtype=torch.float32, device="npu")
        ws_p_ds = torch.empty(bwd_block_num, num_stages, block_M, block_N, dtype=torch.float16, device="npu")
        ws_dv_dk = torch.empty(bwd_block_num, num_stages, block_N, max(dim_qk_padded, D_v), dtype=torch.float32, device="npu")

        kernel = flashattn_bwd_pipeline(B, H, N, D_qk, D_v, ctx.causal, block_M, block_N, groups, num_stages)
        kernel(Q_padded, K_padded, v, do_bhsd, lse, delta, dQ, dK, dV, ws_s_dp, ws_p_ds, ws_dv_dk)

        dQ = dQ[:, :, :, :D_qk].half()
        dK = dK[:, :, :, :D_qk].half()
        dV = dV.half()

        dQ = dQ.permute(0, 2, 1, 3).contiguous()
        dK = dK.permute(0, 2, 1, 3).contiguous()
        dV = dV.permute(0, 2, 1, 3).contiguous()

        return dQ, dK, dV, None, None


attention = _attention.apply


# ===========================================================================
# __main__: minimal L0 smoke test (CI bench_test.sh runs this directly).
# stdout must contain "Test Passed!" for CI to mark the script PASSED.
# Matches L0 "FWD-GQA-causal + BWD-GQA-causal" minimal config:
#   B=1, H=2, H_kv=1, N=128, D_qk=D_v=64, groups=2, causal=True.
# ===========================================================================


if __name__ == "__main__":
    tilelang.disable_cache()
    torch.set_default_device("npu")
    torch.manual_seed(42)

    # Minimal L0 config (causal GQA, D_qk=D_v=64)
    B, H, N, D_qk, D_v, groups = 1, 2, 128, 64, 64, 2
    H_kv = H // groups
    causal = True
    atol = 1e-2

    Q = torch.randn(B, H, N, D_qk, dtype=torch.float16, device="npu")
    K = torch.randn(B, H_kv, N, D_qk, dtype=torch.float16, device="npu")
    V = torch.randn(B, H_kv, N, D_v, dtype=torch.float16, device="npu")
    dO = torch.randn(B, H, N, D_v, dtype=torch.float16, device="npu")

    # --- Forward smoke test (flashattn_fwd v1, causal) ---
    bM, bN = 64, 64
    fwd_mod = flashattn_fwd(B, H, N, D_qk, D_v, causal, bM, bN, groups)
    O_npu, lse_npu = fwd_mod(Q, K, V)
    torch.npu.synchronize()

    O_ref = ref_program(Q, K, V, causal, groups)
    fwd_max_diff = (O_npu.float() - O_ref.float()).abs().max().item()
    assert fwd_max_diff < atol, f"Forward precision check failed: max_diff={fwd_max_diff} >= atol={atol}"

    # --- Backward smoke test (flashattn_bwd_pipeline, causal) ---
    prep_mod = flashattn_bwd_preprocess(B, H, N, D_v, blk=32)
    Delta_npu = prep_mod(O_npu, dO)
    torch.npu.synchronize()

    D_qk_padded = ((D_qk + 127) // 128) * 128
    num_stages = 8
    dQ = torch.zeros(B, H, N, D_qk_padded, dtype=torch.float32, device="npu")
    dK = torch.zeros(B, H_kv, N, D_qk_padded, dtype=torch.float32, device="npu")
    dV = torch.zeros(B, H_kv, N, D_v, dtype=torch.float32, device="npu")

    Q_padded = torch.zeros(B, H, N, D_qk_padded, dtype=torch.float16, device="npu")
    Q_padded[:, :, :, :D_qk] = Q
    K_padded = torch.zeros(B, H_kv, N, D_qk_padded, dtype=torch.float16, device="npu")
    K_padded[:, :, :, :D_qk] = K

    bwd_block_num = (N // bM) * H * B
    ws_s_dp = torch.empty(bwd_block_num, num_stages, bM, bN, dtype=torch.float32, device="npu")
    ws_p_ds = torch.empty(bwd_block_num, num_stages, bM, bN, dtype=torch.float16, device="npu")
    ws_dv_dk = torch.empty(bwd_block_num, num_stages, bN, max(D_qk_padded, D_v), dtype=torch.float32, device="npu")

    bwd_mod = flashattn_bwd_pipeline(B, H, N, D_qk, D_v, causal, bM, bN, groups, num_stages)
    bwd_mod(Q_padded, K_padded, V, dO, lse_npu, Delta_npu, dQ, dK, dV, ws_s_dp, ws_p_ds, ws_dv_dk)
    torch.npu.synchronize()

    dQ_ref, dK_ref, dV_ref = ref_bwd(Q, K, V, dO, causal, groups)
    bwd_max_diff = max(
        (dV.half().float() - dV_ref.float()).abs().max().item(),
        (dK[:, :, :, :D_qk].half().float() - dK_ref.float()).abs().max().item(),
        (dQ[:, :, :, :D_qk].half().float() - dQ_ref.float()).abs().max().item(),
    )
    assert bwd_max_diff < atol, f"Backward precision check failed: max_diff={bwd_max_diff} >= atol={atol}"

    print(f"max_diff: fwd={fwd_max_diff:.6e} bwd={bwd_max_diff:.6e}")
    print("Test Passed!")
