"""MHA Form A kernel: prefill (S > 16) with D <= 128: a T.mma full pipeline.

One of the three mutually exclusive and exhaustive C-scope forms this operator
selects from (S, D). Build it through mha.py's mha_kernel(), which dispatches on
mha_common.select_form(); calling this factory directly is supported for IR
inspection and dumps, and it produces the same kernel.

The tiling tiers, the task split and the whole event-id budget come from
mha_common.derive_config(), and the V scope comes from mha_common.make_emit_v(),
so this file holds only its own C-scope body plus the V-side buffer declarations
TVMScript requires to be emitted inside the prim_func itself.

See mha.py for the operator-level contract (precision invariants, pipeline shape,
sync model, constraints) and DESIGN.md in this directory for the derivation.
"""

import tilelang
from tilelang import language as T
from tilelang.intrinsics import make_nz_layout, make_zn_layout

from mha_common import (
    CROSS_INTERVAL,
    NUM_STAGES,
    derive_config,
    make_emit_v,
    pass_configs,
)


@tilelang.jit(out_idx=[3], workspace_idx=[4, 5, 6], pass_configs=pass_configs)
def mha_form_a_kernel(
    B,
    S,
    S_kv,
    N,
    D,
    dtype="float16",
    scaleValue=-1.0,
    is_causal=False,
):
    """Build the Form A kernel for one (shape, dtype, attr) point."""
    cfg = derive_config(B, S, S_kv, N, D, dtype, scaleValue, is_causal)
    assert cfg.form_a, "mha_form_a_kernel builds Form A only; call mha_kernel() to dispatch"
    # Unpacked from cfg under the exact names the prim_func body reads: TVMScript
    # resolves a prim_func's free names against {**globals, **co_freevars}, so an
    # attribute access like cfg.block_M would be parsed as a TIR expression.
    SEM_WS1_C2V = cfg.SEM_WS1_C2V
    SEM_WS1_V2C = cfg.SEM_WS1_V2C
    SEM_WS2_C2V = cfg.SEM_WS2_C2V
    SEM_WS2_V2C = cfg.SEM_WS2_V2C
    SEM_WS3_C2V = cfg.SEM_WS3_C2V
    SEM_WS3_V2C = cfg.SEM_WS3_V2C
    SIG_K_L1 = cfg.SIG_K_L1
    SIG_L0AB = cfg.SIG_L0AB
    SIG_L0C = cfg.SIG_L0C
    SIG_P_L1 = cfg.SIG_P_L1
    SIG_Q_L1 = cfg.SIG_Q_L1
    SIG_V_L1 = cfg.SIG_V_L1
    accum_dtype = cfg.accum_dtype
    block_M = cfg.block_M
    block_N = cfg.block_N
    causal_offset = cfg.causal_offset
    core_num = cfg.core_num
    half_M = cfg.half_M
    io3_is_alias = cfg.io3_is_alias
    kv_iters = cfg.kv_iters
    q_tasks = cfg.q_tasks
    r_tasks = cfg.r_tasks
    s_blocks = cfg.s_blocks
    sig_io3 = cfg.sig_io3
    single_kv_block = cfg.single_kv_block
    # The shared V scope, bound to this same configuration.
    emit_v = make_emit_v(cfg)

    @T.prim_func
    def main(
        Q: T.Tensor([B, S, N, D], dtype),  # type: ignore
        K: T.Tensor([B, S_kv, N, D], dtype),  # type: ignore
        V: T.Tensor([B, S_kv, N, D], dtype),  # type: ignore
        Output: T.Tensor([B, S, N, D], dtype),  # type: ignore
        workspace_1: T.Tensor([core_num, NUM_STAGES, block_M, block_N], accum_dtype),  # type: ignore
        workspace_2: T.Tensor([core_num, NUM_STAGES, block_M, block_N], dtype),  # type: ignore
        workspace_3: T.Tensor([core_num, NUM_STAGES, block_M, D], accum_dtype),  # type: ignore
    ):
        with T.Kernel(core_num, is_npu=True) as (cid, vid):
            # ---- C-side L1, ZN/NZ annotated for the mma operands ----
            q_l1 = T.alloc_L1([block_M, D], dtype)
            k_l1 = T.alloc_L1([block_N, D], dtype)
            v_l1 = T.alloc_L1([block_N, D], dtype)
            p_l1 = T.alloc_L1([block_M, block_N], dtype)
            T.annotate_layout(
                {
                    q_l1: make_zn_layout(q_l1),
                    k_l1: make_nz_layout(k_l1),
                    p_l1: make_zn_layout(p_l1),
                    v_l1: make_zn_layout(v_l1),
                }
            )
            # ---- L0 double-slot pools: D == block_N shares one pool, D == 64 splits ----
            l0a_s = T.alloc_L0A([2, block_M, D], dtype)
            l0b_s = T.alloc_L0B([2, D, block_N], dtype)
            l0c_s = T.alloc_L0C([2, block_M, block_N], accum_dtype)
            l0a_o = l0a_s if block_N == D else T.alloc_L0A([2, block_M, block_N], dtype)
            l0b_o = l0b_s if block_N == D else T.alloc_L0B([2, block_N, D], dtype)
            l0c_o = l0c_s if block_N == D else T.alloc_L0C([2, block_M, D], accum_dtype)
            # ---- V-side UB (vid half rows) ----
            io_buf = T.alloc_ub([2, half_M, block_N], accum_dtype)
            io3 = io_buf if io3_is_alias else T.alloc_ub([2, half_M, D], accum_dtype)
            work_ub = T.alloc_ub([half_M, block_N], accum_dtype)
            buf_2d = T.alloc_ub([half_M, block_N], accum_dtype)
            rt_2d = T.alloc_ub([half_M, block_N], accum_dtype)
            r_2d = buf_2d if block_N == D else T.alloc_ub([half_M, D], accum_dtype)
            mask_2d = T.alloc_ub([half_M * block_N // 8], "uint8")
            col_pos = T.alloc_ub([block_N], accum_dtype)
            row_thresh = T.alloc_ub([half_M], accum_dtype)
            rr_ub = T.alloc_ub([half_M, block_N], dtype)
            acc_s_half = T.alloc_ub([half_M, block_N], dtype)
            acc_o = T.alloc_ub([half_M, D], accum_dtype)
            out_half = T.alloc_ub([half_M, D], dtype)
            m_i = T.alloc_ub([half_M, 1], accum_dtype)
            m_i_prev = T.alloc_ub([half_M, 1], accum_dtype)
            sumexp = T.alloc_ub([half_M, 1], accum_dtype)
            r_factors = T.alloc_ub([NUM_STAGES, half_M, 1], accum_dtype)
            sumexp_is = T.alloc_ub([NUM_STAGES, half_M, 1], accum_dtype)

            my_start = cid * q_tasks + T.if_then_else(cid < r_tasks, cid, r_tasks)
            my_count = q_tasks + T.if_then_else(cid < r_tasks, 1, 0)

            # ==================== Cube core (Form A: mma full pipeline) ====================
            with T.Scope("C"):
                # init: pre-set the tokens the first consumers would otherwise wait for
                T.set_cross_flag("MTE2", SEM_WS2_C2V)
                T.set_flag("MTE1", "MTE2", SIG_K_L1)
                T.set_flag("MTE1", "MTE2", SIG_P_L1)
                T.set_flag("MTE1", "MTE2", SIG_V_L1)
                T.set_flag("MTE1", "MTE2", SIG_Q_L1)
                T.set_flag("M", "MTE1", SIG_L0AB)
                T.set_flag("M", "MTE1", SIG_L0AB + 1)
                T.set_flag("FIX", "M", SIG_L0C)
                T.set_flag("FIX", "M", SIG_L0C + 1)

                if single_kv_block:
                    # ---- Task-batched single-KV-block C path (Form A) ----
                    # Mirror of the V-side batched branch, carrying the Form A
                    # mma pipeline body (GEMM1/GEMM2 batches) over verbatim
                    # from the generic per-task flow: slot i <-> task
                    # my_start + kb*NUM_STAGES + i. num_tb and batch_tasks come
                    # from the SAME formulas as the V mirror, because the two
                    # scopes advance their flag pairing in lockstep.
                    # Why batch tasks here: with kv_iters == 1 both
                    # eff_num_outer and batch_iters collapse to 1, so the
                    # NUM_STAGES pipeline and the io_buf[2] ping-pong it
                    # feeds never have a second slot in flight -- every task
                    # would run a strictly serial C-QK -> V-softmax -> C-PV
                    # -> V-epilogue chain with no C/V overlap. T.Pipelined
                    # is unavailable here, so batching the TASK dimension is
                    # how the overlap is expressed manually. Two Form-A-only
                    # deltas, both pure Q scheduling, no arithmetic change:
                    #   (a) the Q GM->L1 load moves INSIDE the slot loop,
                    #       because each task has its own s0, and
                    #   (b) q_l1 -> l0a_s[side] is issued for EVERY slot (the
                    #       generic body guards it with `if i < 2` because Q
                    #       is kv-invariant there; here Q changes per slot).
                    # SIG_Q_L1 token balance: per slot, wait(MTE1->MTE2) +
                    # set(MTE2->MTE1) + wait(MTE2->MTE1) + set(MTE1->MTE2)
                    # right after the single MTE1 read of q_l1 = net 0, the
                    # same accounting as the generic task level, so the
                    # task-end return at the bottom of the else branch is
                    # deliberately NOT emitted here.
                    num_tb = T.ceildiv(my_count, NUM_STAGES)
                    for kb in T.serial(num_tb):
                        _remaining_t = my_count - kb * NUM_STAGES
                        batch_tasks = T.if_then_else(_remaining_t < NUM_STAGES, _remaining_t, NUM_STAGES)
                        valid_n = min(block_N, S_kv)  # single block: idx == 0

                        # --- GEMM1 batch: per-task scores -> w1 slot i ---
                        T.wait_cross_flag(SEM_WS1_V2C)
                        for i in T.serial(batch_tasks):
                            side = i % 2
                            task = my_start + kb * NUM_STAGES + i
                            bx = task % s_blocks
                            by = task // s_blocks % N
                            bz = task // s_blocks // N % B
                            s0 = bx * block_M
                            valid_m = T.min(block_M, S - s0)
                            # Q: GM -> L1 (per-slot residency; BSHD row-stride)
                            T.wait_flag("MTE1", "MTE2", SIG_Q_L1)
                            T.copy(Q[bz, s0 : s0 + valid_m, by, :], q_l1)
                            T.set_flag("MTE2", "MTE1", SIG_Q_L1)
                            T.wait_flag("MTE2", "MTE1", SIG_Q_L1)
                            # K: GM -> L1 (single block => j0 == 0)
                            T.wait_flag("MTE1", "MTE2", SIG_K_L1)
                            T.copy(K[bz, 0:valid_n, by, :], k_l1)
                            T.set_flag("MTE2", "MTE1", SIG_K_L1)
                            # Q: L1 -> L0A (EVERY slot: task-dependent Q)
                            T.wait_flag("M", "MTE1", SIG_L0AB + side)
                            T.copy(q_l1, l0a_s[side, :, :])
                            T.set_flag("MTE1", "MTE2", SIG_Q_L1)
                            # K: L1 -> L0B transposed (NZ)
                            T.wait_flag("MTE2", "MTE1", SIG_K_L1)
                            T.copy(k_l1, l0b_s[side, :, :], transpose=True)
                            T.set_flag("MTE1", "MTE2", SIG_K_L1)
                            T.set_flag("MTE1", "M", SIG_L0AB + side)
                            # mma: Q K^T -> L0C
                            T.wait_flag("MTE1", "M", SIG_L0AB + side)
                            T.wait_flag("FIX", "M", SIG_L0C + side)
                            T.mma(l0a_s[side, :, :], l0b_s[side, :, :], l0c_s[side, :, :], init=True)
                            T.set_flag("M", "MTE1", SIG_L0AB + side)
                            T.set_flag("M", "FIX", SIG_L0C + side)
                            # L0C -> w1 (FIX)
                            T.wait_flag("M", "FIX", SIG_L0C + side)
                            T.copy(l0c_s[side, :, :], workspace_1[cid, i, :, :])
                            T.set_flag("FIX", "M", SIG_L0C + side)
                            if (i + 1) % CROSS_INTERVAL == 0 or i == batch_tasks - 1:
                                T.set_cross_flag("FIX", SEM_WS1_C2V)

                        # --- GEMM2 batch: per-task P (w2 slot i) @ V -> Output
                        # --- GEMM2 batch: per-task P (workspace_2 slot i) @ V,
                        # written straight to Output by the FIX pipe. Under the
                        # pre-normalization specialization O = P_hat @ V is
                        # already FINAL, so the fp32 workspace_3 GM round trip,
                        # the V-side phase-2 acc_o pass and the workspace_3
                        # semaphore pair are all unnecessary and are omitted.
                        # The fixpipe fp32 -> dtype RNE cast is bit-identical to
                        # the V-side cast it replaces, and the L0C -> BSHD 4D
                        # store honours a runtime row extent (including a tail
                        # valid_m < block_M, bf16 and D=128).
                        # The C side writes ALL valid_m rows at once; there is
                        # no vid row split on this path.
                        for i in T.serial(batch_tasks):
                            side = i % 2
                            task = my_start + kb * NUM_STAGES + i
                            bx = task % s_blocks
                            by = task // s_blocks % N
                            bz = task // s_blocks // N % B
                            s0 = bx * block_M
                            valid_m = T.min(block_M, S - s0)
                            # V: GM -> L1
                            T.wait_flag("MTE1", "MTE2", SIG_V_L1)
                            T.copy(V[bz, 0:valid_n, by, :], v_l1)
                            T.set_flag("MTE2", "MTE1", SIG_V_L1)
                            # P: w2 -> p_l1
                            T.wait_flag("MTE1", "MTE2", SIG_P_L1)
                            if i % CROSS_INTERVAL == 0:
                                T.wait_cross_flag(SEM_WS2_V2C)
                            T.copy(workspace_2[cid, i, :, :], p_l1)
                            T.set_flag("MTE2", "MTE1", SIG_P_L1)
                            # V: L1 -> L0B (no transpose for PV)
                            T.wait_flag("MTE2", "MTE1", SIG_V_L1)
                            T.wait_flag("M", "MTE1", SIG_L0AB + side)
                            T.copy(v_l1, l0b_o[side, :, :])
                            T.set_flag("MTE1", "MTE2", SIG_V_L1)
                            # P: L1 -> L0A
                            T.wait_flag("MTE2", "MTE1", SIG_P_L1)
                            T.copy(p_l1, l0a_o[side, :, :])
                            T.set_flag("MTE1", "MTE2", SIG_P_L1)
                            T.set_flag("MTE1", "M", SIG_L0AB + side)
                            # mma: P V -> L0C
                            T.wait_flag("MTE1", "M", SIG_L0AB + side)
                            T.wait_flag("FIX", "M", SIG_L0C + side)
                            T.mma(l0a_o[side, :, :], l0b_o[side, :, :], l0c_o[side, :, :], init=True)
                            T.set_flag("M", "MTE1", SIG_L0AB + side)
                            T.set_flag("M", "FIX", SIG_L0C + side)
                            # L0C -> Output (FIX direct write, runtime row
                            # extent; SIG_L0C slot ownership unchanged)
                            T.wait_flag("M", "FIX", SIG_L0C + side)
                            T.copy(l0c_o[side, :, :], Output[bz, s0 : s0 + valid_m, by, :])
                            T.set_flag("FIX", "M", SIG_L0C + side)

                        T.set_cross_flag("MTE2", SEM_WS2_C2V)

                else:
                    # ---- Flat (task, kv) continuous slot stream: the Form A
                    # generic multi-KV-block domain. Mirror of the Form B-mma
                    # flat branch, with TWO Form-A-specific deltas, both pure
                    # Q/l0a_s scheduling with no arithmetic change:
                    # (a) the l0a_s reload condition is `i < 2 or kv < 2`.
                    #     When D == block_N the shared pool means l0a_o aliases
                    #     l0a_s and every GEMM2 batch overwrites it, so
                    #     batch-head slots MUST reload Q (this preserves the
                    #     original `if i < 2` semantics); in addition a flat
                    #     mid-batch task switch needs the new task's first two
                    #     slots to reload their sides, since Q is task-invariant
                    #     and kv >= 2 slots reuse the resident copy.
                    # (b) the q_l1 return (SIG_Q_L1 MTE1->MTE2 set) moves to the
                    #     task's LAST slot (kv == eff-1), not kv == 1. Tasks that
                    #     span batch boundaries re-read q_l1 at later batch-head
                    #     slots (delta-a reloads); releasing earlier would let
                    #     the NEXT task's kv0 MTE2 load overwrite q_l1 while
                    #     those reads are still queued on MTE1. M-pipe/MTE1
                    #     in-order execution from the last slot covers every
                    #     q_l1 read of the task. Per-task accounting is
                    #     unchanged: one full SIG_Q_L1 round (kv == 0, with the
                    #     DMA and all its flags inside one condition) plus one
                    #     return (kv == eff-1) = net 0, init/destroy untouched.
                    # The walk state machine, the batch skeleton and the V
                    # mirror are shared with the Form B-mma flat branch, and
                    # C/V derive every bound from the same formulas. ----
                    total_flat = T.alloc_var("int32", init=0)
                    for t in T.serial(my_count):
                        _tw = my_start + t
                        _s0w = (_tw % s_blocks) * block_M
                        _effw = T.floordiv(T.min(_s0w + block_M - 1 + causal_offset, S_kv - 1), block_N) + 1 if is_causal else kv_iters
                        total_flat += _effw
                    num_fb = T.ceildiv(total_flat, NUM_STAGES)
                    tl_head = T.alloc_var("int32", init=0)  # persistent batch-head walk state
                    kv_head = T.alloc_var("int32", init=0)
                    for fb in T.serial(num_fb):
                        _rem_f = total_flat - fb * NUM_STAGES
                        batch_iters = T.if_then_else(_rem_f < NUM_STAGES, _rem_f, NUM_STAGES)

                        # --- GEMM1 batch: flat slots -> w1 ---
                        T.wait_cross_flag(SEM_WS1_V2C)
                        tl_g1 = T.alloc_var("int32", init=tl_head)
                        kv_g1 = T.alloc_var("int32", init=kv_head)
                        for i in T.serial(batch_iters):
                            side = i % 2
                            task = my_start + tl_g1
                            bx = task % s_blocks
                            by = task // s_blocks % N
                            bz = task // s_blocks // N % B
                            s0 = bx * block_M
                            valid_m = T.min(block_M, S - s0)
                            eff_kv_iters = (
                                T.floordiv(T.min(s0 + block_M - 1 + causal_offset, S_kv - 1), block_N) + 1 if is_causal else kv_iters
                            )
                            idx = kv_g1
                            j0 = idx * block_N
                            valid_n = T.min(block_N, S_kv - j0)
                            if kv_g1 == 0:
                                # ---- task switch: Q GM->L1 (full SIG_Q_L1
                                # MTE2 round inside the condition; the l0a_s
                                # reload rides this slot's L0AB round below) ----
                                T.wait_flag("MTE1", "MTE2", SIG_Q_L1)
                                T.copy(Q[bz, s0 : s0 + valid_m, by, :], q_l1)
                                T.set_flag("MTE2", "MTE1", SIG_Q_L1)
                                T.wait_flag("MTE2", "MTE1", SIG_Q_L1)
                            # K: GM -> L1 (BSHD row-stride, runtime extent)
                            T.wait_flag("MTE1", "MTE2", SIG_K_L1)
                            T.copy(K[bz, j0 : j0 + valid_n, by, :], k_l1)
                            T.set_flag("MTE2", "MTE1", SIG_K_L1)
                            # Q: L1 -> L0A (delta-a: batch-head OR task-head sides)
                            T.wait_flag("M", "MTE1", SIG_L0AB + side)
                            if i < 2 or kv_g1 < 2:
                                T.copy(q_l1, l0a_s[side, :, :])
                            # K: L1 -> L0B transposed (NZ)
                            T.wait_flag("MTE2", "MTE1", SIG_K_L1)
                            T.copy(k_l1, l0b_s[side, :, :], transpose=True)
                            T.set_flag("MTE1", "MTE2", SIG_K_L1)
                            if kv_g1 == eff_kv_iters - 1:
                                # ---- task-end q_l1 return (delta-b: LAST
                                # slot; MTE1 in-order covers every q_l1 read
                                # of the task incl. later batch-head reloads) ----
                                T.set_flag("MTE1", "MTE2", SIG_Q_L1)
                            T.set_flag("MTE1", "M", SIG_L0AB + side)
                            # mma: Q K^T -> L0C
                            T.wait_flag("MTE1", "M", SIG_L0AB + side)
                            T.wait_flag("FIX", "M", SIG_L0C + side)
                            T.mma(l0a_s[side, :, :], l0b_s[side, :, :], l0c_s[side, :, :], init=True)
                            T.set_flag("M", "MTE1", SIG_L0AB + side)
                            T.set_flag("M", "FIX", SIG_L0C + side)
                            # L0C -> w1 (FIX)
                            T.wait_flag("M", "FIX", SIG_L0C + side)
                            T.copy(l0c_s[side, :, :], workspace_1[cid, i, :, :])
                            T.set_flag("FIX", "M", SIG_L0C + side)
                            if (i + 1) % CROSS_INTERVAL == 0 or i == batch_iters - 1:
                                T.set_cross_flag("FIX", SEM_WS1_C2V)
                            # walk advance (pure scalar, no DMA/flags)
                            kv_g1 += 1
                            if kv_g1 == eff_kv_iters:
                                kv_g1 = 0
                                tl_g1 += 1

                        # --- GEMM2 batch: consume w2 (P), produce PV -> w3 ---
                        T.wait_cross_flag(SEM_WS3_V2C)
                        tl_g2 = T.alloc_var("int32", init=tl_head)
                        kv_g2 = T.alloc_var("int32", init=kv_head)
                        for i in T.serial(batch_iters):
                            side = i % 2
                            task = my_start + tl_g2
                            bx = task % s_blocks
                            by = task // s_blocks % N
                            bz = task // s_blocks // N % B
                            s0 = bx * block_M
                            eff_kv_iters = (
                                T.floordiv(T.min(s0 + block_M - 1 + causal_offset, S_kv - 1), block_N) + 1 if is_causal else kv_iters
                            )
                            idx = kv_g2
                            j0 = idx * block_N
                            valid_n = T.min(block_N, S_kv - j0)
                            # V: GM -> L1
                            T.wait_flag("MTE1", "MTE2", SIG_V_L1)
                            T.copy(V[bz, j0 : j0 + valid_n, by, :], v_l1)
                            T.set_flag("MTE2", "MTE1", SIG_V_L1)
                            # P: w2 -> p_l1
                            T.wait_flag("MTE1", "MTE2", SIG_P_L1)
                            if i % CROSS_INTERVAL == 0:
                                T.wait_cross_flag(SEM_WS2_V2C)
                            T.copy(workspace_2[cid, i, :, :], p_l1)
                            T.set_flag("MTE2", "MTE1", SIG_P_L1)
                            # V: L1 -> L0B (no transpose for PV)
                            T.wait_flag("MTE2", "MTE1", SIG_V_L1)
                            T.wait_flag("M", "MTE1", SIG_L0AB + side)
                            T.copy(v_l1, l0b_o[side, :, :])
                            T.set_flag("MTE1", "MTE2", SIG_V_L1)
                            # P: L1 -> L0A
                            T.wait_flag("MTE2", "MTE1", SIG_P_L1)
                            T.copy(p_l1, l0a_o[side, :, :])
                            T.set_flag("MTE1", "MTE2", SIG_P_L1)
                            T.set_flag("MTE1", "M", SIG_L0AB + side)
                            # mma: P V -> L0C
                            T.wait_flag("MTE1", "M", SIG_L0AB + side)
                            T.wait_flag("FIX", "M", SIG_L0C + side)
                            T.mma(l0a_o[side, :, :], l0b_o[side, :, :], l0c_o[side, :, :], init=True)
                            T.set_flag("M", "MTE1", SIG_L0AB + side)
                            T.set_flag("M", "FIX", SIG_L0C + side)
                            # L0C -> w3 (FIX)
                            T.wait_flag("M", "FIX", SIG_L0C + side)
                            T.copy(l0c_o[side, :, :], workspace_3[cid, i, :, :])
                            T.set_flag("FIX", "M", SIG_L0C + side)
                            if (i + 1) % CROSS_INTERVAL == 0 or i == batch_iters - 1:
                                T.set_cross_flag("FIX", SEM_WS3_C2V)
                            # walk advance (pure scalar, no DMA/flags)
                            kv_g2 += 1
                            if kv_g2 == eff_kv_iters:
                                kv_g2 = 0
                                tl_g2 += 1

                        T.set_cross_flag("MTE2", SEM_WS2_C2V)
                        # batch-head state advance (GEMM2 walk ended at the next head)
                        tl_head = tl_g2
                        kv_head = kv_g2

                # destroy: consume outstanding init-direction flags
                T.wait_flag("MTE1", "MTE2", SIG_K_L1)
                T.wait_flag("MTE1", "MTE2", SIG_P_L1)
                T.wait_flag("MTE1", "MTE2", SIG_V_L1)
                T.wait_flag("MTE1", "MTE2", SIG_Q_L1)
                T.wait_flag("M", "MTE1", SIG_L0AB)
                T.wait_flag("M", "MTE1", SIG_L0AB + 1)
                T.wait_flag("FIX", "M", SIG_L0C)
                T.wait_flag("FIX", "M", SIG_L0C + 1)

            # ==================== Vector core (shared emitter) ====================
            with T.Scope("V"):
                emit_v(
                    io_buf,
                    io3,
                    work_ub,
                    buf_2d,
                    rt_2d,
                    r_2d,
                    mask_2d,
                    col_pos,
                    row_thresh,
                    rr_ub,
                    acc_s_half,
                    acc_o,
                    out_half,
                    m_i,
                    m_i_prev,
                    sumexp,
                    r_factors,
                    sumexp_is,
                    Output,
                    workspace_1,
                    workspace_2,
                    workspace_3,
                    cid,
                    vid,
                    my_start,
                    my_count,
                    sig_io3,
                )

    return main


# This file is a library module (mha.py and test_mha.py import it), but the
# examples/bench_test.sh runner executes nearly every .py under examples/<op>/ as
# a standalone script and requires a pass token in its output. The __name__ guard
# below therefore keeps `import mha_form_a` side-effect free while giving
# `python mha_form_a.py` a real self-check: it builds THIS form's kernel for one
# in-form shape and validates it against a torch reference on the device.
# run_form_self_check() first asserts the shape dispatches to Form A, so this
# module's kernel is guaranteed to be the one exercised.
if __name__ == "__main__":
    from mha_common import FORM_A, run_form_self_check

    # Form A: S > 16 and D <= 128. Small, in-domain, structurally representative
    # (single KV block, so the FIX-direct-write epilogue runs).
    run_form_self_check(mha_form_a_kernel, form=FORM_A, B=2, S=128, S_kv=128, N=8, D=64)
