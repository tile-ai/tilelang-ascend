"""MHA Form B gemm_v0 kernel: decode (S <= 16), all D: the T.gemm_v0 fallback.

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

from mha_common import (
    CROSS_INTERVAL,
    NUM_STAGES,
    derive_config,
    make_emit_v,
    pass_configs,
)


@tilelang.jit(out_idx=[3], workspace_idx=[4, 5, 6], pass_configs=pass_configs)
def mha_form_b_gemm_v0_kernel(
    B,
    S,
    S_kv,
    N,
    D,
    dtype="float16",
    scaleValue=-1.0,
    is_causal=False,
):
    """Build the Form B gemm_v0 kernel for one (shape, dtype, attr) point."""
    cfg = derive_config(B, S, S_kv, N, D, dtype, scaleValue, is_causal)
    assert not (cfg.form_a or cfg.form_b_mma), "mha_form_b_gemm_v0_kernel builds the decode fallback only; call mha_kernel() to dispatch"
    # Unpacked from cfg under the exact names the prim_func body reads: TVMScript
    # resolves a prim_func's free names against {**globals, **co_freevars}, so an
    # attribute access like cfg.block_M would be parsed as a TIR expression.
    SEM_WS1_C2V = cfg.SEM_WS1_C2V
    SEM_WS1_V2C = cfg.SEM_WS1_V2C
    SEM_WS2_C2V = cfg.SEM_WS2_C2V
    SEM_WS2_V2C = cfg.SEM_WS2_V2C
    SEM_WS3_C2V = cfg.SEM_WS3_C2V
    SEM_WS3_V2C = cfg.SEM_WS3_V2C
    accum_dtype = cfg.accum_dtype
    block_M = cfg.block_M
    block_N = cfg.block_N
    causal_offset = cfg.causal_offset
    core_num = cfg.core_num
    gemm1_kL0 = cfg.gemm1_kL0
    half_M = cfg.half_M
    io3_is_alias = cfg.io3_is_alias
    kv_db = cfg.kv_db
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
            # ---- C-side L1 (no annotation: gemm_v0 handles L0 internally) ----
            q_l1 = T.alloc_L1([block_M, D], dtype)
            # K-side double slot, kv_db domain only. A 3D L1 buffer with no
            # layout annotation plus a runtime slice behaves exactly like a
            # serial sequence of 2D buffers, which is what double buffering
            # needs here.
            k_l1 = T.alloc_L1([2, block_N, D], dtype) if kv_db else T.alloc_L1([block_N, D], dtype)
            # V-side double slot, kv_db only; the same 3D no-annotation plus
            # runtime-slice mechanism as k_l1.
            v_l1 = T.alloc_L1([2, block_N, D], dtype) if kv_db else T.alloc_L1([block_N, D], dtype)
            p_l1 = T.alloc_L1([block_M, block_N], dtype)
            # ---- L0C single slots (gemm_v0 manages L0A/L0B by kL0Size) ----
            acc_s_l0c = T.alloc_L0C([block_M, block_N], accum_dtype)
            acc_o_l0c = T.alloc_L0C([block_M, D], accum_dtype)
            # ---- V-side UB: identical to Form A (emit_v is shared) ----
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

            # ================= Cube core (Form B: gemm_v0, barrier-free) =================
            with T.Scope("C"):
                # init: pretend the workspace_2 slots are already released by
                # the consumer, because the V side's first phase-1 batch waits
                # on SEM_WS2_C2V. No SIG_* tokens are pre-set: this form uses
                # no intra-core user flags at all.
                T.set_cross_flag("MTE2", SEM_WS2_C2V)
                if single_kv_block:
                    # ---- Task-batched single-KV-block C path ----
                    # One protocol round batches NUM_STAGES TASKS, since
                    # kv_iters == 1 means every task is exactly one KV block:
                    # slot i <-> task my_start + kb*NUM_STAGES + i. The per-slot
                    # ops are identical to the generic flow, and the V mirror
                    # derives num_tb/batch_tasks from the SAME my_start/my_count
                    # formulas so the pairing stays in lockstep.
                    num_tb = T.ceildiv(my_count, NUM_STAGES)
                    for kb in T.serial(num_tb):
                        _remaining_t = my_count - kb * NUM_STAGES
                        batch_tasks = T.if_then_else(_remaining_t < NUM_STAGES, _remaining_t, NUM_STAGES)
                        valid_n = min(block_N, S_kv)  # single block: idx == 0

                        # --- GEMM1 batch: per-task scores -> w1 slot i ---
                        T.wait_cross_flag(SEM_WS1_V2C)
                        for i in T.serial(batch_tasks):
                            task = my_start + kb * NUM_STAGES + i
                            bx = task % s_blocks
                            by = task // s_blocks % N
                            bz = task // s_blocks // N % B
                            s0 = bx * block_M
                            valid_m = T.min(block_M, S - s0)
                            # Barrier-free ordering. The gemm_v0 prologue
                            # orders prior MTE2 (Q/K loads) before its MTE1
                            # reads and prior FIX (workspace writes) before
                            # its mma; the epilogue orders MTE1 before later
                            # MTE2 (L1 slot reuse) and mma before later FIX
                            # (the L0C read). That covers every intra-core
                            # segment boundary, so no barrier_all is needed.
                            # Same-pipe ops are in-order by hardware; cross-
                            # core hazards stay on the semaphore protocol.
                            T.copy(Q[bz, s0 : s0 + valid_m, by, :], q_l1)
                            T.copy(K[bz, 0:valid_n, by, :], k_l1)
                            T.gemm_v0(q_l1, k_l1, acc_s_l0c, transpose_B=True, init=True, kL0Size=gemm1_kL0)
                            T.copy(acc_s_l0c, workspace_1[cid, i, :, :])
                            if (i + 1) % CROSS_INTERVAL == 0 or i == batch_tasks - 1:
                                T.set_cross_flag("FIX", SEM_WS1_C2V)

                        # --- GEMM2 batch: per-task P (w2 slot i) @ V -> w3 slot i ---
                        T.wait_cross_flag(SEM_WS3_V2C)
                        for i in T.serial(batch_tasks):
                            task = my_start + kb * NUM_STAGES + i
                            by = task // s_blocks % N
                            bz = task // s_blocks // N % B
                            T.copy(V[bz, 0:valid_n, by, :], v_l1)
                            if i % CROSS_INTERVAL == 0:
                                T.wait_cross_flag(SEM_WS2_V2C)
                            T.copy(workspace_2[cid, i, :, :], p_l1)
                            T.gemm_v0(p_l1, v_l1, acc_o_l0c, init=True)
                            T.copy(acc_o_l0c, workspace_3[cid, i, :, :])
                            if (i + 1) % CROSS_INTERVAL == 0 or i == batch_tasks - 1:
                                T.set_cross_flag("FIX", SEM_WS3_C2V)

                        T.set_cross_flag("MTE2", SEM_WS2_C2V)
                elif kv_db:
                    # ---- kv_db: dual-slot K/V prefetch, stock gemm_v0
                    # default-prologue sync. Every gemm_v0 call self-issues
                    # SetFlag/WaitFlag<MTE2_MTE1>(0): the wait drains the
                    # WHOLE MTE2 queue, so every operand enqueued before
                    # the call is L1-resident before the call's MTE1 reads.
                    # No user flag is needed, and none may be added here --
                    # a user SetFlag<MTE2_MTE1>(0) would leave a dangling
                    # token that a later default prologue wait could
                    # consume stale, starting the reads before the copies
                    # retired. The pre-call prefetch positions below are
                    # kept as-is; under the full drain a prefetch simply
                    # retires inside the consuming call's prologue (it
                    # joins that call's critical path instead of streaming
                    # through it).
                    #
                    # K slots use side = idx % 2 with idx = k*NUM_STAGES + i.
                    # NUM_STAGES is even, so a round/task head slot is always
                    # side 0. Issue sites, chosen so that
                    # #prefetch == #consumed == sum(eff_kv_iters) -- no slot
                    # is loaded twice or read before it is loaded:
                    #   task-0 head (t == 0): Q(0), then K_0 into slot 0.
                    #   K pre-call prefetch, iteration idx: K_{idx+1} into
                    #      slot (idx+1)%2. The guard idx+1 < eff_kv_iters is
                    #      GLOBAL, so it crosses round boundaries and no
                    #      separate round-end site is needed.
                    #   task-end priming (t+1 < my_count): Q(t+1) THEN K'_0,
                    #      in that order, so both operands of GEMM1'(0) sit
                    #      in the MTE2 queue ahead of the next task's first
                    #      call, whose drain makes them resident; slot 0.
                    # GEMM1 consumes k_l1[idx % 2, :, :] as a runtime slice
                    # at a single call site -- the same pattern GEMM2 uses
                    # for its V operand.
                    #
                    # V side: v_l1 has two slots, so V(idx+1) -- slot
                    # (idx+1)%2, last read by gemm2(idx-1) -- is issuable at
                    # iteration idx's pre-call position: after gemm2(idx-1)
                    # returned (its epilogue MTE1_MTE2 release frees the
                    # slot). Lookahead is exactly one slot; V(idx+2) would
                    # race gemm2(idx)'s own slot. V sites:
                    #   round-head V priming at the GEMM1-batch tail
                    #      (i == batch_iters-1): the round's head V into
                    #      STATIC slot 0 (NUM_STAGES even =>
                    #      (k*NUM_STAGES)%2 == 0), so needClear is true --
                    #      the base case of the finite-stale induction
                    #      below. Placed after ALL of this round's K loads,
                    #      keeping issue order == consume order.
                    #   V pre-call prefetch, GEMM2 iteration idx: V_{idx+1}
                    #      into slot (idx+1)%2. The runtime offset makes
                    #      needClear false, so tail rows may retain stale
                    #      data; that is safe by induction, because every
                    #      slot's FIRST load is a full tile (slot 0 via the
                    #      round-head clear, slot 1 via V(idx0+1) with
                    #      valid_n == block_N since S_kv > 256 here) and the
                    #      matching P columns are exactly 0, so
                    #      0 * finite = 0 with no NaN path. The guard
                    #      i+1 < batch_iters is LOCAL: round-head V comes
                    #      from the GEMM1-batch tail, never twice.
                    #   GEMM2 iteration idx: the SEM_WS2 wait and the P copy
                    #      keep their original positions; the call's default
                    #      drain covers V(idx) and P(idx), with
                    #      B = v_l1[idx%2,:,:] as a runtime slice.
                    # Slot release needs no extra flags: the gemm epilogue's
                    # SetFlag<MTE1_MTE2> orders every MTE2 issued after a
                    # call behind that call's MTE1 reads, so the K(idx+1)
                    # and V(idx+1) prefetches -- which overwrite the slot
                    # gemm(idx-1) read and are issued after that call
                    # returned -- are correctly ordered. For Q(t+1) at the
                    # task end, q_l1 was released by task t's last GEMM1
                    # epilogue, and the GEMM1 phase precedes the GEMM2
                    # phase. Semaphore protocol positions are unchanged.
                    # The task-end priming sits after the task's last
                    # set_cross_flag("MTE2", SEM_WS2_C2V), where the MTE2
                    # queue is empty (the last P copy is 8 KB and the
                    # round's K/V loads retired during the GEMM2 phase), so
                    # w2 release is not delayed.
                    for t in T.serial(my_count):
                        task = my_start + t
                        bx = task % s_blocks
                        by = task // s_blocks % N
                        bz = task // s_blocks // N % B
                        s0 = bx * block_M
                        valid_m = T.min(block_M, S - s0)
                        eff_kv_iters = T.floordiv(T.min(s0 + block_M - 1 + causal_offset, S_kv - 1), block_N) + 1 if is_causal else kv_iters
                        eff_num_outer = T.ceildiv(eff_kv_iters, NUM_STAGES)

                        if t == 0:
                            # Task-0 head priming: Q and K_0 enter the MTE2
                            # queue before the first GEMM1(0) call, whose
                            # default prologue drain makes both resident.
                            # For t > 0, Q(t) and K_0(t) were already primed
                            # at task t-1's end, hence the t == 0 guard: no
                            # double load.
                            hvalid = T.min(block_N, S_kv)
                            T.copy(Q[bz, s0 : s0 + valid_m, by, :], q_l1)
                            T.copy(K[bz, 0:hvalid, by, :], k_l1[0, :, :])

                        for k in T.serial(eff_num_outer):
                            _remaining = eff_kv_iters - k * NUM_STAGES
                            batch_iters = T.if_then_else(_remaining < NUM_STAGES, _remaining, NUM_STAGES)

                            # --- GEMM1 batch: scores -> w1 ---
                            T.wait_cross_flag(SEM_WS1_V2C)
                            for i in T.serial(batch_iters):
                                idx = k * NUM_STAGES + i
                                # PRE-CALL prefetch of K(idx+1). Position
                                # unchanged; under the default full drain
                                # the copy retires inside gemm1(idx)'s
                                # prologue, and gemm1(idx+1) finds its
                                # operand slot already resident.
                                if idx + 1 < eff_kv_iters:
                                    nidx = idx + 1
                                    nj0 = nidx * block_N
                                    nvalid = T.min(block_N, S_kv - nj0)
                                    T.copy(K[bz, nj0 : nj0 + nvalid, by, :], k_l1[nidx % 2, :, :])
                                # GEMM1 consume: single call site with the
                                # K slot as a runtime slice. Both operands
                                # Q and K(idx) were enqueued before this
                                # call, so its default prologue drain makes
                                # them resident.
                                T.gemm_v0(q_l1, k_l1[idx % 2, :, :], acc_s_l0c, transpose_B=True, init=True, kL0Size=gemm1_kL0)
                                T.copy(acc_s_l0c, workspace_1[cid, i, :, :])
                                if (i + 1) % CROSS_INTERVAL == 0 or i == batch_iters - 1:
                                    T.set_cross_flag("FIX", SEM_WS1_C2V)
                                if i == batch_iters - 1:
                                    # Round-head V priming at the GEMM1
                                    # batch tail. FIFO order puts it after
                                    # every K issue of this round, and
                                    # GEMM2 iteration 0 consumes it.
                                    # STATIC slot 0 (NUM_STAGES even =>
                                    # (k*NUM_STAGES)%2 == 0), so needClear
                                    # is true: the stale-free base case of
                                    # the induction.
                                    vj0 = k * NUM_STAGES * block_N
                                    vvalid = T.min(block_N, S_kv - vj0)
                                    T.copy(V[bz, vj0 : vj0 + vvalid, by, :], v_l1[0, :, :])

                            # --- GEMM2 batch: consume w2 (P), produce PV -> w3 ---
                            T.wait_cross_flag(SEM_WS3_V2C)
                            for i in T.serial(batch_iters):
                                idx = k * NUM_STAGES + i
                                if i % CROSS_INTERVAL == 0:
                                    T.wait_cross_flag(SEM_WS2_V2C)
                                # The P copy stays at its original
                                # position; the call's default drain covers
                                # both V(idx) and P(idx).
                                T.copy(workspace_2[cid, i, :, :], p_l1)
                                # PRE-CALL V(idx+1) prefetch. Dual slot
                                # (idx+1)%2 was released by gemm2(idx-1)'s
                                # return; under the default full drain the
                                # copy retires inside gemm2(idx)'s prologue.
                                # The guard is LOCAL because the round-head
                                # V comes from the GEMM1-batch tail and
                                # must not load twice.
                                if i + 1 < batch_iters:
                                    nvidx = idx + 1
                                    nvj0 = nvidx * block_N
                                    nvvalid = T.min(block_N, S_kv - nvj0)
                                    T.copy(V[bz, nvj0 : nvj0 + nvvalid, by, :], v_l1[nvidx % 2, :, :])
                                T.gemm_v0(p_l1, v_l1[idx % 2, :, :], acc_o_l0c, init=True)
                                T.copy(acc_o_l0c, workspace_3[cid, i, :, :])
                                if (i + 1) % CROSS_INTERVAL == 0 or i == batch_iters - 1:
                                    T.set_cross_flag("FIX", SEM_WS3_C2V)

                            T.set_cross_flag("MTE2", SEM_WS2_C2V)

                        if t + 1 < my_count:
                            # Task-end priming, placed after the task's last
                            # set of WS2_C2V. FIFO order mandates Q(t+1) BEFORE
                            # K'_0 so that both GEMM1'(0) operands sit in the
                            # MTE2 queue ahead of the next task's first call,
                            # whose default drain makes them resident. The
                            # bz'/by'/extent values are re-derived from
                            # task+1 by pure scalar arithmetic.
                            ntask = task + 1
                            nbx = ntask % s_blocks
                            nby = ntask // s_blocks % N
                            nbz = ntask // s_blocks // N % B
                            ns0 = nbx * block_M
                            nvalid_m = T.min(block_M, S - ns0)
                            tvalid = T.min(block_N, S_kv)
                            T.copy(Q[nbz, ns0 : ns0 + nvalid_m, nby, :], q_l1)
                            T.copy(K[nbz, 0:tvalid, nby, :], k_l1[0, :, :])
                else:
                    for t in T.serial(my_count):
                        task = my_start + t
                        bx = task % s_blocks
                        by = task // s_blocks % N
                        bz = task // s_blocks // N % B
                        s0 = bx * block_M
                        valid_m = T.min(block_M, S - s0)
                        eff_kv_iters = T.floordiv(T.min(s0 + block_M - 1 + causal_offset, S_kv - 1), block_N) + 1 if is_causal else kv_iters
                        eff_num_outer = T.ceildiv(eff_kv_iters, NUM_STAGES)

                        # Barrier-free ordering: the first gemm_v0 prologue of
                        # the task drains this MTE2 Q load before its own MTE1
                        # reads, and q_l1 reuse across tasks is ordered by the
                        # previous gemm's epilogue (MTE1 -> MTE2).
                        T.copy(Q[bz, s0 : s0 + valid_m, by, :], q_l1)

                        for k in T.serial(eff_num_outer):
                            _remaining = eff_kv_iters - k * NUM_STAGES
                            batch_iters = T.if_then_else(_remaining < NUM_STAGES, _remaining, NUM_STAGES)

                            # --- GEMM1 batch: scores -> w1 ---
                            T.wait_cross_flag(SEM_WS1_V2C)
                            for i in T.serial(batch_iters):
                                idx = k * NUM_STAGES + i
                                j0 = idx * block_N
                                valid_n = T.min(block_N, S_kv - j0)
                                T.copy(K[bz, j0 : j0 + valid_n, by, :], k_l1)
                                T.gemm_v0(q_l1, k_l1, acc_s_l0c, transpose_B=True, init=True, kL0Size=gemm1_kL0)
                                T.copy(acc_s_l0c, workspace_1[cid, i, :, :])
                                if (i + 1) % CROSS_INTERVAL == 0 or i == batch_iters - 1:
                                    T.set_cross_flag("FIX", SEM_WS1_C2V)

                            # --- GEMM2 batch: consume w2 (P), produce PV -> w3 ---
                            T.wait_cross_flag(SEM_WS3_V2C)
                            for i in T.serial(batch_iters):
                                idx = k * NUM_STAGES + i
                                j0 = idx * block_N
                                valid_n = T.min(block_N, S_kv - j0)
                                T.copy(V[bz, j0 : j0 + valid_n, by, :], v_l1)
                                if i % CROSS_INTERVAL == 0:
                                    T.wait_cross_flag(SEM_WS2_V2C)
                                T.copy(workspace_2[cid, i, :, :], p_l1)
                                T.gemm_v0(p_l1, v_l1, acc_o_l0c, init=True)
                                T.copy(acc_o_l0c, workspace_3[cid, i, :, :])
                                if (i + 1) % CROSS_INTERVAL == 0 or i == batch_iters - 1:
                                    T.set_cross_flag("FIX", SEM_WS3_C2V)

                            T.set_cross_flag("MTE2", SEM_WS2_C2V)

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
# below therefore keeps `import mha_form_b_gemm_v0` side-effect free while giving
# `python mha_form_b_gemm_v0.py` a real self-check: it builds THIS form's kernel
# for one in-form shape and validates it against a torch reference on the device.
# run_form_self_check() first asserts the shape dispatches to Form B gemm_v0, so
# this module's kernel is guaranteed to be the one exercised.
if __name__ == "__main__":
    from mha_common import FORM_B_GEMM_V0, run_form_self_check

    # Form B gemm_v0 decode fallback: S <= 16, all D. D == 256 with S_kv=512 >
    # 256 puts it in the kv_db domain (K/V dual-side L1 double buffer, stock
    # gemm_v0 default prologue), the decode path's most distinct structure.
    run_form_self_check(mha_form_b_gemm_v0_kernel, form=FORM_B_GEMM_V0, B=2, S=1, S_kv=512, N=8, D=256)
