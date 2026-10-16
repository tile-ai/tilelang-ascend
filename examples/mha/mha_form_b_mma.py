"""MHA Form B-mma kernel: prefill (S > 16) with D == 256: a manual K/N-split T.mma pipeline.

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
from tilelang.intrinsics import make_zn_layout

from mha_common import (
    CROSS_INTERVAL,
    NUM_STAGES,
    derive_config,
    make_emit_v,
    pass_configs,
)


@tilelang.jit(out_idx=[3], workspace_idx=[4, 5, 6], pass_configs=pass_configs)
def mha_form_b_mma_kernel(
    B,
    S,
    S_kv,
    N,
    D,
    dtype="float16",
    scaleValue=-1.0,
    is_causal=False,
):
    """Build the Form B-mma kernel for one (shape, dtype, attr) point."""
    cfg = derive_config(B, S, S_kv, N, D, dtype, scaleValue, is_causal)
    assert cfg.form_b_mma, "mha_form_b_mma_kernel builds Form B-mma only; call mha_kernel() to dispatch"
    # Unpacked from cfg under the exact names the prim_func body reads: TVMScript
    # resolves a prim_func's free names against {**globals, **co_freevars}, so an
    # attribute access like cfg.block_M would be parsed as a TIR expression.
    KC = cfg.KC
    NC = cfg.NC
    SEM_WS1_C2V = cfg.SEM_WS1_C2V
    SEM_WS1_V2C = cfg.SEM_WS1_V2C
    SEM_WS2_C2V = cfg.SEM_WS2_C2V
    SEM_WS2_V2C = cfg.SEM_WS2_V2C
    SEM_WS3_C2V = cfg.SEM_WS3_C2V
    SEM_WS3_V2C = cfg.SEM_WS3_V2C
    SIG_K_L1 = cfg.SIG_K_L1
    SIG_L0AO = cfg.SIG_L0AO
    SIG_L0AS = cfg.SIG_L0AS
    SIG_L0C = cfg.SIG_L0C
    SIG_L0KC = cfg.SIG_L0KC
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

    # ==================== Form B-mma (prefill S > 16 and D = 256): a manual
    # K/N-split T.mma pipeline. gemm_v0 drains the M pipe with a per-step
    # PipeBarrier<PIPE_M>, which at this width leaves the scalar pipe waiting
    # on MTE1 for most of the wall time. Chunk-level token flow instead
    # overlaps three pipes: MTE1(c+1) || M(c) || FIX(c-1).
    # The batch protocol is the one the other forms use too (task batch when
    # single_kv_block, flat stream when multi-block). GEMM2 uses a narrow
    # l0c_o pool plus a per-chunk column-segment FIX write, because an mma
    # targeting an L0C column sub-region is rejected by the cube unit
    # ("CUBE instruction parameter is invalid").
    # Token accounting: init pre-sets 13, destroy waits 12, every round is
    # self-paired, no token is left dangling and no DMA is conditional without
    # its flags. FORBIDDEN in this prim_func: T.gemm_v0 (it would collide on
    # internal event id 0), T.barrier_all, threads=, and T.Pipelined. ======

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
            # ---- C-side L1. k_l1 is ZN-annotated here, NOT NZ, and the
            # distinction is load-bearing. The copy_gm_to_l1 template always
            # writes canonical zN (LayoutL1 is hard-tagged layout::zN), while a
            # column-segment transpose read computes its source offset as
            # layout_map.Forward(region_min). Under an nZ annotation that yields
            # the nZ-formula offset (2048 for column 128 of a [128, 256] tile),
            # which does not match the physical zN fill; the correct base is
            # 16384, i.e. the zN formula, which is also the nZ(256, 128) tile-1
            # base. Full-buffer reads start at offset 0 and are layout-agnostic,
            # which is why Form A's NZ annotation on k_l1 is and stays correct
            # there. Annotating ZN explicitly gives the k column segments the
            # same Forward formula as the q/v column segments.
            q_l1 = T.alloc_L1([block_M, D], dtype)
            k_l1 = T.alloc_L1([block_N, D], dtype)
            v_l1 = T.alloc_L1([block_N, D], dtype)
            p_l1 = T.alloc_L1([block_M, block_N], dtype)
            T.annotate_layout(
                {
                    q_l1: make_zn_layout(q_l1),
                    k_l1: make_zn_layout(k_l1),
                    p_l1: make_zn_layout(p_l1),
                    v_l1: make_zn_layout(v_l1),
                }
            )
            # ---- Explicit L0 pools. Budget at KC=NC=128, block_M=32,
            # block_N=128: L0A 32 KB, L0B 64 KB (the shared pool sits exactly at
            # its ceiling), L0C 64 KB for the narrow PV pool. ----
            l0a_s = T.alloc_L0A([2, block_M, KC], dtype)  # GEMM1 A: Q k-chunks (task-resident)
            l0a_o = T.alloc_L0A([2, block_M, block_N], dtype)  # GEMM2 A: P, side ping-pong
            l0b_pool = T.alloc_L0B([2, KC, block_N], dtype)  # SHARED: GEMM1 K^T kc / GEMM2 V nc
            l0c_s = T.alloc_L0C([2, block_M, block_N], accum_dtype)  # scores: K-split chain target
            l0c_o = T.alloc_L0C([2, block_M, NC], accum_dtype)  # PV: narrow pool, slot=chunk
            # ---- V-side UB: identical in all three forms (emit_v is shared) ----
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

            # ==================== Cube core (Form B-mma) ====================
            with T.Scope("C"):
                # init: 13 tokens pre-set = 1 semaphore + 4 L1 + 2 L0KC +
                # 2 L0AO + 2 L0AS + 2 L0C; destroy waits 12 of them.
                T.set_cross_flag("MTE2", SEM_WS2_C2V)
                T.set_flag("MTE1", "MTE2", SIG_K_L1)
                T.set_flag("MTE1", "MTE2", SIG_P_L1)
                T.set_flag("MTE1", "MTE2", SIG_V_L1)
                T.set_flag("MTE1", "MTE2", SIG_Q_L1)
                T.set_flag("M", "MTE1", SIG_L0KC)
                T.set_flag("M", "MTE1", SIG_L0KC + 1)
                T.set_flag("M", "MTE1", SIG_L0AO)
                T.set_flag("M", "MTE1", SIG_L0AO + 1)
                T.set_flag("M", "MTE1", SIG_L0AS)
                T.set_flag("M", "MTE1", SIG_L0AS + 1)
                T.set_flag("FIX", "M", SIG_L0C)
                T.set_flag("FIX", "M", SIG_L0C + 1)

                if single_kv_block:
                    # ---- Task-batched single-KV-block C path: the same
                    # protocol mirror as the other forms, with this form's mma
                    # body. Slot i <-> task my_start + kb*NUM_STAGES + i;
                    # num_tb/batch_tasks come from the SAME formulas as the V
                    # mirror, so flag pairing stays in lockstep. ----
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
                            # Q: GM -> L1, per slot (Q is task-dependent here)
                            T.wait_flag("MTE1", "MTE2", SIG_Q_L1)
                            T.copy(Q[bz, s0 : s0 + valid_m, by, :], q_l1)
                            T.set_flag("MTE2", "MTE1", SIG_Q_L1)
                            T.wait_flag("MTE2", "MTE1", SIG_Q_L1)
                            # K: GM -> L1 (single block => j0 == 0)
                            T.wait_flag("MTE1", "MTE2", SIG_K_L1)
                            T.copy(K[bz, 0:valid_n, by, :], k_l1)
                            T.set_flag("MTE2", "MTE1", SIG_K_L1)
                            # Q: L1 column segments -> L0A, EVERY slot, since Q
                            # is task-dependent here. One SIG_L0AS round per
                            # slot: the trailing M waits move the M-pipe block
                            # to a point where no M work is in flight, so they
                            # cost nothing, and the task-internal mma reads of
                            # l0a_s are covered by M-pipe in-order execution and
                            # therefore carry no SIG_L0AS at all.
                            T.wait_flag("M", "MTE1", SIG_L0AS)
                            T.copy(q_l1[:, 0:KC], l0a_s[0, :, :])
                            T.wait_flag("M", "MTE1", SIG_L0AS + 1)
                            T.copy(q_l1[:, KC : 2 * KC], l0a_s[1, :, :])
                            T.set_flag("MTE1", "M", SIG_L0AS)
                            T.set_flag("MTE1", "M", SIG_L0AS + 1)
                            T.wait_flag("MTE1", "M", SIG_L0AS)
                            T.wait_flag("MTE1", "M", SIG_L0AS + 1)
                            T.set_flag("MTE1", "MTE2", SIG_Q_L1)  # q_l1 MTE1 read done
                            # K-split mma chain: chunk c=0 with init=True, then
                            # chunk c=1 with init=False so the fp32 L0C
                            # accumulation continues in ascending K order.
                            T.wait_flag("MTE2", "MTE1", SIG_K_L1)
                            # ---- k-chunk c=0 (K columns 0:KC) ----
                            T.wait_flag("M", "MTE1", SIG_L0KC)  # pool chunk 0 released
                            T.copy(k_l1[:, 0:KC], l0b_pool[0, :, :], transpose=True)
                            T.set_flag("MTE1", "M", SIG_L0KC)
                            T.wait_flag("MTE1", "M", SIG_L0KC)
                            T.wait_flag("FIX", "M", SIG_L0C + side)  # l0c_s slot writable
                            T.mma(l0a_s[0, :, :], l0b_pool[0, :, :], l0c_s[side, :, :], init=True)
                            T.set_flag("M", "MTE1", SIG_L0KC)  # pool chunk 0 released
                            # ---- k-chunk c=1 (K columns KC:2*KC, chained) ----
                            T.wait_flag("M", "MTE1", SIG_L0KC + 1)
                            T.copy(k_l1[:, KC : 2 * KC], l0b_pool[1, :, :], transpose=True)
                            T.set_flag("MTE1", "M", SIG_L0KC + 1)
                            T.wait_flag("MTE1", "M", SIG_L0KC + 1)
                            T.mma(l0a_s[1, :, :], l0b_pool[1, :, :], l0c_s[side, :, :], init=False)
                            T.set_flag("M", "MTE1", SIG_L0KC + 1)
                            T.set_flag("M", "FIX", SIG_L0C + side)  # scores full block ready
                            T.set_flag("MTE1", "MTE2", SIG_K_L1)  # k_l1 read done (after c1 load)
                            # L0C -> w1 (FIX full block)
                            T.wait_flag("M", "FIX", SIG_L0C + side)
                            T.copy(l0c_s[side, :, :], workspace_1[cid, i, :, :])
                            T.set_flag("FIX", "M", SIG_L0C + side)
                            # Slot-end l0a_s return. By M-pipe order this slot's
                            # last GEMM1 mma has already read l0a_s and GEMM2
                            # never reads it, so the next slot's head wait
                            # consumes the token: net 0 per slot.
                            T.set_flag("M", "MTE1", SIG_L0AS)
                            T.set_flag("M", "MTE1", SIG_L0AS + 1)
                            if (i + 1) % CROSS_INTERVAL == 0 or i == batch_tasks - 1:
                                T.set_cross_flag("FIX", SEM_WS1_C2V)

                        # --- GEMM2 batch: per-task P (workspace_2 slot i) @ V ->
                        # workspace_3 slot i. This form keeps workspace_3 and
                        # the V-side phase 2, matching the emit_v
                        # `if not form_a:` gate, so the V scope is shared
                        # unchanged with the gemm_v0 form. ---
                        T.wait_cross_flag(SEM_WS3_V2C)
                        for i in T.serial(batch_tasks):
                            side = i % 2
                            task = my_start + kb * NUM_STAGES + i
                            by = task // s_blocks % N
                            bz = task // s_blocks // N % B
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
                            # P: L1 -> L0A (side ping-pong)
                            T.wait_flag("M", "MTE1", SIG_L0AO + side)
                            T.wait_flag("MTE2", "MTE1", SIG_P_L1)
                            T.copy(p_l1, l0a_o[side, :, :])
                            T.set_flag("MTE1", "MTE2", SIG_P_L1)
                            T.set_flag("MTE1", "M", SIG_L0AO + side)
                            T.wait_flag("MTE2", "MTE1", SIG_V_L1)
                            # N-split mma plus a per-chunk column-segment FIX
                            # write. Each chunk accumulates into its own narrow
                            # l0c_o slot with init=True, so the chunks are
                            # column-independent and need no chaining.
                            # ---- n-chunk c=0 (V cols 0:NC -> w3 cols 0:NC) ----
                            T.wait_flag("M", "MTE1", SIG_L0KC)
                            T.copy(v_l1[:, 0:NC], l0b_pool[0, :, :])  # direct, no transpose
                            T.set_flag("MTE1", "M", SIG_L0KC)
                            T.wait_flag("MTE1", "M", SIG_L0KC)
                            T.wait_flag("MTE1", "M", SIG_L0AO + side)  # p ready
                            T.wait_flag("FIX", "M", SIG_L0C)  # narrow slot 0 writable
                            T.mma(l0a_o[side, :, :], l0b_pool[0, :, :], l0c_o[0, :, :], init=True)
                            T.set_flag("M", "MTE1", SIG_L0KC)
                            T.set_flag("M", "FIX", SIG_L0C)  # chunk 0 ready
                            T.wait_flag("M", "FIX", SIG_L0C)
                            T.copy(l0c_o[0, :, :], workspace_3[cid, i, :, 0:NC])  # colseg FIX
                            T.set_flag("FIX", "M", SIG_L0C)
                            # ---- n-chunk c=1 (V cols NC:2*NC -> w3 cols NC:2*NC) ----
                            T.wait_flag("M", "MTE1", SIG_L0KC + 1)
                            T.copy(v_l1[:, NC : 2 * NC], l0b_pool[1, :, :])
                            T.set_flag("MTE1", "M", SIG_L0KC + 1)
                            T.wait_flag("MTE1", "M", SIG_L0KC + 1)
                            T.wait_flag("FIX", "M", SIG_L0C + 1)  # narrow slot 1 writable
                            T.mma(l0a_o[side, :, :], l0b_pool[1, :, :], l0c_o[1, :, :], init=True)
                            T.set_flag("M", "MTE1", SIG_L0KC + 1)
                            T.set_flag("M", "FIX", SIG_L0C + 1)
                            T.set_flag("M", "MTE1", SIG_L0AO + side)  # l0a_o both chunks read
                            T.wait_flag("M", "FIX", SIG_L0C + 1)
                            T.copy(l0c_o[1, :, :], workspace_3[cid, i, :, NC : 2 * NC])
                            T.set_flag("FIX", "M", SIG_L0C + 1)
                            T.set_flag("MTE1", "MTE2", SIG_V_L1)  # v_l1 read done (both chunks)
                            if (i + 1) % CROSS_INTERVAL == 0 or i == batch_tasks - 1:
                                # after BOTH colseg FIX writes (FIX pipe is in-order)
                                T.set_cross_flag("FIX", SEM_WS3_C2V)

                        T.set_cross_flag("MTE2", SEM_WS2_C2V)

                else:
                    # ---- Flat (task, kv) continuous slot stream: this form's
                    # generic multi-KV-block domain. Replaces per-task nested
                    # [task][outer-batch] loops so a slot batch can cross task
                    # boundaries without draining. Structure mirrors the
                    # emit_v flat branch VERBATIM: a one-time O(my_count)
                    # scalar walk builds total_flat = sum(eff_kv(t)); slots
                    # are batched in groups of NUM_STAGES; a per-slot walk
                    # state (T.alloc_var) maps slot -> (task, kv). Causal
                    # masking makes eff_kv runtime-nonuniform per task, which
                    # is what forces the walk -- a compile-time-uniform mapping
                    # only works for decode.
                    # Task switch (kv == 0): Q GM->L1 plus the l0a_s preload.
                    # The FULL SIG_Q_L1 and SIG_L0AS rounds both live INSIDE
                    # the runtime `if`, so their waits and sets enter and exit
                    # together and the per-task net stays 0 -- the task-granular
                    # accounting moved to switch granularity. A conditional DMA
                    # must always carry all of its flags in the same condition.
                    # Task end (kv == eff-1): the L0AS return sets, placed by
                    # M-pipe order after this task's last GEMM1 mma read.
                    # The GEMM1 batch still fully precedes the GEMM2 batch,
                    # because l0b_pool is shared between the two phases and
                    # must not interleave at iteration level. Per-slot bodies
                    # are the generic ones verbatim. ----
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
                                # ---- Task switch: Q load (SIG_Q_L1 round) plus
                                # the l0a_s preload (SIG_L0AS round). Both rounds
                                # sit entirely inside this condition, so every
                                # wait is matched by its set on the same
                                # execution path. ----
                                T.wait_flag("MTE1", "MTE2", SIG_Q_L1)
                                T.copy(Q[bz, s0 : s0 + valid_m, by, :], q_l1)
                                T.set_flag("MTE2", "MTE1", SIG_Q_L1)
                                T.wait_flag("MTE2", "MTE1", SIG_Q_L1)
                                T.wait_flag("M", "MTE1", SIG_L0AS)
                                T.copy(q_l1[:, 0:KC], l0a_s[0, :, :])
                                T.wait_flag("M", "MTE1", SIG_L0AS + 1)
                                T.copy(q_l1[:, KC : 2 * KC], l0a_s[1, :, :])
                                T.set_flag("MTE1", "M", SIG_L0AS)
                                T.set_flag("MTE1", "M", SIG_L0AS + 1)
                                T.wait_flag("MTE1", "M", SIG_L0AS)
                                T.wait_flag("MTE1", "M", SIG_L0AS + 1)
                                T.set_flag("MTE1", "MTE2", SIG_Q_L1)  # q_l1 MTE1 read done
                            # K: GM -> L1 (BSHD row-stride, runtime extent)
                            T.wait_flag("MTE1", "MTE2", SIG_K_L1)
                            T.copy(K[bz, j0 : j0 + valid_n, by, :], k_l1)
                            T.set_flag("MTE2", "MTE1", SIG_K_L1)
                            T.wait_flag("MTE2", "MTE1", SIG_K_L1)
                            # K-split mma chain: c=0 init=True -> c=1
                            # init=False (ascending-K fp32 accumulation).
                            # ---- k-chunk c=0 (K columns 0:KC) ----
                            T.wait_flag("M", "MTE1", SIG_L0KC)  # pool chunk 0 released
                            T.copy(k_l1[:, 0:KC], l0b_pool[0, :, :], transpose=True)
                            T.set_flag("MTE1", "M", SIG_L0KC)
                            T.wait_flag("MTE1", "M", SIG_L0KC)
                            T.wait_flag("FIX", "M", SIG_L0C + side)  # l0c_s slot writable
                            T.mma(l0a_s[0, :, :], l0b_pool[0, :, :], l0c_s[side, :, :], init=True)
                            T.set_flag("M", "MTE1", SIG_L0KC)  # pool chunk 0 released
                            # ---- k-chunk c=1 (K columns KC:2*KC, chained) ----
                            T.wait_flag("M", "MTE1", SIG_L0KC + 1)
                            T.copy(k_l1[:, KC : 2 * KC], l0b_pool[1, :, :], transpose=True)
                            T.set_flag("MTE1", "M", SIG_L0KC + 1)
                            T.wait_flag("MTE1", "M", SIG_L0KC + 1)
                            T.mma(l0a_s[1, :, :], l0b_pool[1, :, :], l0c_s[side, :, :], init=False)
                            T.set_flag("M", "MTE1", SIG_L0KC + 1)
                            T.set_flag("M", "FIX", SIG_L0C + side)  # scores full block ready
                            T.set_flag("MTE1", "MTE2", SIG_K_L1)  # k_l1 read done
                            # L0C -> w1 (FIX full block)
                            T.wait_flag("M", "FIX", SIG_L0C + side)
                            T.copy(l0c_s[side, :, :], workspace_1[cid, i, :, :])
                            T.set_flag("FIX", "M", SIG_L0C + side)
                            if kv_g1 == eff_kv_iters - 1:
                                # ---- Task end: the l0a_s return. By M-pipe
                                # order this task's last GEMM1 mma has already
                                # read l0a_s and GEMM2 never reads it, so the
                                # next switch's head wait consumes the token:
                                # one round per task. ----
                                T.set_flag("M", "MTE1", SIG_L0AS)
                                T.set_flag("M", "MTE1", SIG_L0AS + 1)
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
                            # P: L1 -> L0A (side ping-pong)
                            T.wait_flag("M", "MTE1", SIG_L0AO + side)
                            T.wait_flag("MTE2", "MTE1", SIG_P_L1)
                            T.copy(p_l1, l0a_o[side, :, :])
                            T.set_flag("MTE1", "MTE2", SIG_P_L1)
                            T.set_flag("MTE1", "M", SIG_L0AO + side)
                            T.wait_flag("MTE2", "MTE1", SIG_V_L1)
                            # N-split mma plus a per-chunk column-segment FIX
                            # write. Each chunk accumulates into its own narrow
                            # l0c_o slot with init=True, so the chunks are
                            # column-independent and need no chaining.
                            # ---- n-chunk c=0 (V cols 0:NC -> w3 cols 0:NC) ----
                            T.wait_flag("M", "MTE1", SIG_L0KC)
                            T.copy(v_l1[:, 0:NC], l0b_pool[0, :, :])  # direct, no transpose
                            T.set_flag("MTE1", "M", SIG_L0KC)
                            T.wait_flag("MTE1", "M", SIG_L0KC)
                            T.wait_flag("MTE1", "M", SIG_L0AO + side)  # p ready
                            T.wait_flag("FIX", "M", SIG_L0C)  # narrow slot 0 writable
                            T.mma(l0a_o[side, :, :], l0b_pool[0, :, :], l0c_o[0, :, :], init=True)
                            T.set_flag("M", "MTE1", SIG_L0KC)
                            T.set_flag("M", "FIX", SIG_L0C)  # chunk 0 ready
                            T.wait_flag("M", "FIX", SIG_L0C)
                            T.copy(l0c_o[0, :, :], workspace_3[cid, i, :, 0:NC])  # colseg FIX
                            T.set_flag("FIX", "M", SIG_L0C)
                            # ---- n-chunk c=1 (V cols NC:2*NC -> w3 cols NC:2*NC) ----
                            T.wait_flag("M", "MTE1", SIG_L0KC + 1)
                            T.copy(v_l1[:, NC : 2 * NC], l0b_pool[1, :, :])
                            T.set_flag("MTE1", "M", SIG_L0KC + 1)
                            T.wait_flag("MTE1", "M", SIG_L0KC + 1)
                            T.wait_flag("FIX", "M", SIG_L0C + 1)  # narrow slot 1 writable
                            T.mma(l0a_o[side, :, :], l0b_pool[1, :, :], l0c_o[1, :, :], init=True)
                            T.set_flag("M", "MTE1", SIG_L0KC + 1)
                            T.set_flag("M", "FIX", SIG_L0C + 1)
                            T.set_flag("M", "MTE1", SIG_L0AO + side)  # l0a_o both chunks read
                            T.wait_flag("M", "FIX", SIG_L0C + 1)
                            T.copy(l0c_o[1, :, :], workspace_3[cid, i, :, NC : 2 * NC])
                            T.set_flag("FIX", "M", SIG_L0C + 1)
                            T.set_flag("MTE1", "MTE2", SIG_V_L1)  # v_l1 read done
                            if (i + 1) % CROSS_INTERVAL == 0 or i == batch_iters - 1:
                                # after BOTH colseg FIX writes (FIX pipe is in-order)
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

                # destroy: symmetric wait x12 for the tokens pre-set by init
                T.wait_flag("MTE1", "MTE2", SIG_K_L1)
                T.wait_flag("MTE1", "MTE2", SIG_P_L1)
                T.wait_flag("MTE1", "MTE2", SIG_V_L1)
                T.wait_flag("MTE1", "MTE2", SIG_Q_L1)
                T.wait_flag("M", "MTE1", SIG_L0KC)
                T.wait_flag("M", "MTE1", SIG_L0KC + 1)
                T.wait_flag("M", "MTE1", SIG_L0AO)
                T.wait_flag("M", "MTE1", SIG_L0AO + 1)
                T.wait_flag("M", "MTE1", SIG_L0AS)
                T.wait_flag("M", "MTE1", SIG_L0AS + 1)
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
# below therefore keeps `import mha_form_b_mma` side-effect free while giving
# `python mha_form_b_mma.py` a real self-check: it builds THIS form's kernel for
# one in-form shape and validates it against a torch reference on the device.
# run_form_self_check() first asserts the shape dispatches to Form B-mma, so this
# module's kernel is guaranteed to be the one exercised.
if __name__ == "__main__":
    from mha_common import FORM_B_MMA, run_form_self_check

    # Form B-mma: S > 16 and D == 256. Multi KV block (S_kv=512, block_N=128 ->
    # kv_iters=4), so both the manual K/N-split mma pipeline and the generic flat
    # (task, kv) slot stream run. Smallest S above the decode threshold that still
    # reaches the multi-block domain.
    run_form_self_check(mha_form_b_mma_kernel, form=FORM_B_MMA, B=1, S=32, S_kv=512, N=4, D=256)
