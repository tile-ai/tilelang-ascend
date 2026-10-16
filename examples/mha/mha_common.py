"""Shared host-side configuration and the single V-scope emitter for the MHA example.

This module is the one source of everything the three kernel forms must agree on:

* ``pass_configs`` / ``NUM_STAGES`` / ``CROSS_INTERVAL`` -- the Expert-mode
  compiler switches and the C/V pipeline geometry.
* ``derive_config`` -- the whole host-side tiling, dispatch-predicate, task-split
  and event-id budget derivation. It is pure Python over scalar arguments and
  touches no tensor data, so every trip count it produces is a compile-time
  constant. Keeping it in one place is what keeps the three forms from
  drifting apart on block_M/block_N/gemm1_kL0 or on the semaphore id space.
* ``select_form`` -- the three-way dispatch predicate, expressed through
  ``derive_config`` so that the wrapper, the tests and the kernel factories can
  never disagree about which form a shape belongs to.
* ``make_emit_v`` -- the factory for the shared V-scope ``@T.macro``.

Why emit_v is a macro and why it lives here. TVMScript resolves ``T.*`` frames at
parse time, so a plain Python emitter called from inside a prim_func has its ``T.*``
emissions silently dropped: the kernel still compiles, the generated body is
missing, and the aicore hangs on an orphaned wait. ``@T.macro`` survives
parse-time scoping, which is why the V scope can be shared at all. It must stay
exactly one definition -- all four precision invariants of this operator live
inside it, so a second copy would be a second thing to keep bit-identical.

Name resolution contract. ``tvm/script/parser/tir/entry.py::prim_func`` and
``core/parser.py::ScriptMacro.__call__`` both resolve free names against
``inspect_function_capture(func)`` == ``{**func.__globals__, **co_freevars}``.
Consequently every name a moved body reads from its enclosing scope has to be
bound here, or in the enclosing factory, under exactly the same name. Attribute
access such as ``cfg.block_M`` is NOT equivalent: TVMScript would parse it as a
TIR expression. That is why ``make_emit_v`` unpacks the config into plain locals
before the macro definition instead of passing the config object through.
"""

import dataclasses

import tilelang
from tilelang import language as T

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: False,  # manual T.Scope("C"/"V")
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: False,  # manual SEM 0-5
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: False,  # manual SIG flag matrix
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: False,  # explicit alloc_* declarations
}

NUM_STAGES = 8  # C/V pipeline depth in slots; must be even (double-slot rotation)
CROSS_INTERVAL = 1  # slots per cross-core semaphore Set/Wait pair


def _cube_core_num() -> int:
    """Physical cube-core count (host metadata query; touches no tensor data)."""
    try:
        import torch_npu

        return int(torch_npu.npu.get_device_properties(0).cube_core_num)
    except Exception:  # noqa: BLE001 - conservative 910B-class fallback
        return 20


# Form identifiers returned by select_form(). The three are mutually exclusive and
# exhaustive over (S, D); mha.py dispatches on them and nothing else.
FORM_A = "form_a"
FORM_B_MMA = "form_b_mma"
FORM_B_GEMM_V0 = "form_b_gemm_v0"


@dataclasses.dataclass(frozen=True)
class MhaConfig:
    """Every host-side constant one of the three kernel forms is built from.

    Produced by derive_config(); consumed by the three jit factories and by
    make_emit_v(). Frozen so a form cannot silently retile itself."""

    B: int
    S: int
    S_kv: int
    N: int
    D: int
    dtype: str
    is_causal: bool
    accum_dtype: str
    sm_scale: float
    block_M: int
    block_N: int
    gemm1_kL0: int
    form_a: bool
    form_b_mma: bool
    KC: int
    NC: int
    half_M: int
    causal_offset: int
    need_bound_mask: bool
    need_mask: bool
    offset_eff: int
    s_blocks: int
    kv_iters: int
    single_kv_block: bool
    flat_v: bool
    kv_db: bool
    block_num: int
    core_num: int
    q_tasks: int
    r_tasks: int
    SEM_WS1_C2V: int
    SEM_WS1_V2C: int
    SEM_WS2_V2C: int
    SEM_WS2_C2V: int
    SEM_WS3_C2V: int
    SEM_WS3_V2C: int
    SIG_K_L1: int
    SIG_P_L1: int
    SIG_V_L1: int
    SIG_Q_L1: int
    SIG_L0AB: int
    SIG_L0KC: int
    SIG_L0AO: int
    SIG_L0AS: int
    SIG_L0C: int
    SIG_IO_UB: int
    SIG_IO3: int
    SIG_S_HALF: int
    io3_is_alias: bool
    sig_io3: int


def derive_config(
    B,
    S,
    S_kv,
    N,
    D,
    dtype="float16",
    scaleValue=-1.0,
    is_causal=False,
):
    """Host-side configuration derivation for one (shape, dtype, attr) point.

    Pure Python over scalars: no tensor is touched, so every value returned is a
    compile-time constant as far as the generated kernel is concerned. The body is
    the single authoritative tiling / dispatch / event-id budget of this operator;
    the three form factories all read it through this one call."""
    accum_dtype = "float"
    assert NUM_STAGES % 2 == 0, "NUM_STAGES must be even: the io_buf/L0 slots rotate in pairs"
    # ---- Host-side config derivation: pure Python, touches no tensor data ----
    sm_scale = scaleValue if scaleValue > 0 else (1.0 / D) ** 0.5
    # block_M / block_N / gemm1_kL0 three-tier system. Every tier is the largest
    # value that still fits its on-chip budget (the 32 KB L0B ping-pong and the
    # L1 capacity); the predicates below encode those budget solutions.
    block_M = 16 if S <= 16 else (32 if D == 256 else 64)
    if S <= 16:
        if D == 256 and S_kv > 256:
            # kv_db dual-side double-buffer domain. bN=128 keeps L1 inside
            # q8 + k[2]x64 + v[2]x64 + p4 = 268 KB of the 511.75 KB budget, and
            # that headroom is what makes two k_l1 and two v_l1 slots possible.
            # The second V slot is in turn what allows V(idx+1) to be issued
            # BEFORE the gemm call instead of after it returns: with a single
            # v_l1 slot the epilogue release order forbids a pre-call issue at
            # bN=256. Only the multi-block domain (S_kv > 256) takes this tier;
            # the single-block domain keeps the tiers below.
            block_N = 128
        else:
            # bN=512 (D<=128, S_kv>256): GEMM1 transpose_B B-tile 512*kL0(32)*2B =
            # 32768 == the 4-step ping-pong L0B budget; D=256 excluded by L1 (536KB).
            block_N = 512 if (D <= 128 and S_kv > 256) else (256 if S_kv > 128 else 128)
    else:
        block_N = 128
    gemm1_kL0 = (
        32
        if (S <= 16 and D <= 128 and S_kv > 256)  # bN=512: 512*32*2 = 32768 <= 32KB
        # bN=256, D=256 single-block tier (128 < S_kv <= 256): 256*64*2 =
        # 32768 <= 32KB. The kv_db multi-block domain (S_kv > 256) uses bN=128
        # and falls through to the 128 tier: 128*128*2 = 32768 <= 32KB. Two
        # ping-pong steps are kept on purpose -- kL0=256 would collapse the
        # GEMM1 K loop to a single step, serialising MTE1 against M with no
        # overlap, while leaving the total k-step count per core unchanged.
        else (64 if (S <= 16 and D == 256 and S_kv > 128 and S_kv <= 256) else 128)
    )
    # ---- Expert-mode dispatch predicates ----
    form_a = (S > 16) and (D <= 128)  # Form A: T.mma full pipeline
    # ---- Three-way dispatch: mutually exclusive and exhaustive over (S, D) ----
    form_b_mma = (S > 16) and (D == 256)  # Form B-mma: manual K/N-split mma pipeline
    # else (S <= 16): Form B gemm_v0 decode fallback, all D
    KC = 128  # Form B-mma k-chunk width (compile-time constant; D=256 -> 2 chunks)
    NC = 128  # Form B-mma n-chunk width (compile-time constant; D=256 -> 2 chunks)
    if form_b_mma:
        assert D % KC == 0 and D % NC == 0, "K/N-split requires D divisible by the KC/NC chunk width"
    half_M = block_M // 2  # rows owned by one AIV (vid row split): 8/16/32
    causal_offset = S_kv - S  # bottom-right alignment (>= 0 when causal, asserted)
    need_bound_mask = S_kv % block_N != 0  # tail-column clamp (compile-time)
    need_mask = is_causal or need_bound_mask
    # Unified threshold offset: causal -> bottom-right offset; non-causal -> big
    # constant so T.min(row + offset_eff, S_kv-1) collapses to S_kv-1.
    offset_eff = causal_offset if is_causal else 2**30
    s_blocks = (S + block_M - 1) // block_M
    kv_iters = (S_kv + block_N - 1) // block_N  # compile-time constant
    single_kv_block = kv_iters == 1  # i.e. S_kv <= block_N
    # ---- Flat (task, kv) slot-stream domain: the generic multi-KV-block prefill
    # of both mma forms. Iterating slots over a flat (task, kv) index instead of
    # nested [task][kv-block] loops lets one slot batch cross task boundaries
    # without draining the pipeline. The single-KV-block branch (which batches
    # tasks instead) and the S <= 16 decode fallback use their own loops. This
    # is a parse-time closure flag: emit_v reads it to pick its V-side branch
    # and both C-scope generic branches read it too, so all three must agree.
    flat_v = (S > 16) and not single_kv_block
    # ---- Decode D=256 K/V dual-side L1 double-buffer domain ----
    # gemm_v0 fallback only (S <= 16), D == 256, more than one KV block. L1
    # footprint q8 + k[2]x64 + v[2]x64 + p4 = 268 KB of the 511.75 KB budget.
    # Sync = the stock gemm_v0 default prologue alone: every call self-issues
    # SetFlag/WaitFlag<MTE2_MTE1>(0), draining the whole MTE2 pipe, so all
    # operands enqueued before the call (Q, K(idx), the K(idx+1)/V(idx+1)
    # prefetches, V(idx), P) are L1-resident before its MTE1 reads, and slot
    # reuse is ordered by the epilogue's SetFlag<MTE1_MTE2>. This path must
    # therefore carry ZERO user MTE2_MTE1 flags: a dangling user token on id 0
    # could be consumed stale by a later default-prologue wait. C-side intra-
    # core scheduling only: the V scope, the semaphore matrix, the single-
    # block path and the D <= 128 domains are untouched.
    kv_db = (S <= 16) and (D == 256) and not single_kv_block
    block_num = B * N * s_blocks
    core_num = min(block_num, _cube_core_num())  # never launch more cores than tasks
    q_tasks = block_num // core_num  # base tasks per core
    r_tasks = block_num % core_num  # first r_tasks cores take one extra

    # ---- Cross-core semaphore ids (AIC <-> 2 x AIV group, mode=2) ----
    SEM_WS1_C2V = 0  # w1 scores ready: Cube -> Vector
    SEM_WS1_V2C = 1  # w1 consumed: Vector -> Cube
    SEM_WS2_V2C = 2  # w2 softmax P ready: Vector -> Cube
    SEM_WS2_C2V = 3  # w2 consumed: Cube -> Vector
    SEM_WS3_C2V = 4  # w3 PV ready: Cube -> Vector
    SEM_WS3_V2C = 5  # w3 consumed: Vector -> Cube

    # ---- C-side intra-core event ids, one space per directed pipe pair (Form A) ----
    # MTE2 <-> MTE1
    SIG_K_L1 = 0
    SIG_P_L1 = 1
    SIG_V_L1 = 2
    SIG_Q_L1 = 3
    # MTE1 <-> M
    SIG_L0AB = 0  # double slots 0/1 (Form A)
    # MTE1 <-> M, the Form B-mma fine-grained tokens. Ids 0-5 stay within the
    # 8-id hardware limit. This prim_func contains no T.gemm_v0 call, so these
    # ids cannot collide with gemm_v0's internal L0AB event id 0.
    SIG_L0KC = 0  # l0b_pool chunk 0/1; GEMM1 kc and GEMM2 nc share this stream
    SIG_L0AO = 2  # l0a_o side 0/1 (GEMM2 A ping-pong)
    SIG_L0AS = 4  # l0a_s chunk 0/1 (GEMM1 A, task/slot-level preload)
    # M <-> FIX
    SIG_L0C = 0  # double slots 0/1

    # ---- V-side intra-core event ids (per AIV) ----
    # MTE2 <-> V
    SIG_IO_UB = 0  # io_buf (w1) double slots 0/1
    SIG_IO3 = 2  # io_buf3 (w3) double slots 2/3 (only when D != block_N)
    # MTE3 <-> V
    SIG_S_HALF = 0  # acc_s_half / out_half staging ownership

    # workspace_3 staging and rescale scratch: a shape-forced buffer choice.
    # D == block_N -> alias io_buf / buf_2d (identical shapes, reused in disjoint
    # sequential V-pipe windows, token-balanced on shared flag ids);
    # D != block_N -> dedicated buffers, because the shapes then differ.
    io3_is_alias = block_N == D
    sig_io3 = SIG_IO_UB if io3_is_alias else SIG_IO3
    return MhaConfig(
        B=B,
        S=S,
        S_kv=S_kv,
        N=N,
        D=D,
        dtype=dtype,
        is_causal=is_causal,
        accum_dtype=accum_dtype,
        sm_scale=sm_scale,
        block_M=block_M,
        block_N=block_N,
        gemm1_kL0=gemm1_kL0,
        form_a=form_a,
        form_b_mma=form_b_mma,
        KC=KC,
        NC=NC,
        half_M=half_M,
        causal_offset=causal_offset,
        need_bound_mask=need_bound_mask,
        need_mask=need_mask,
        offset_eff=offset_eff,
        s_blocks=s_blocks,
        kv_iters=kv_iters,
        single_kv_block=single_kv_block,
        flat_v=flat_v,
        kv_db=kv_db,
        block_num=block_num,
        core_num=core_num,
        q_tasks=q_tasks,
        r_tasks=r_tasks,
        SEM_WS1_C2V=SEM_WS1_C2V,
        SEM_WS1_V2C=SEM_WS1_V2C,
        SEM_WS2_V2C=SEM_WS2_V2C,
        SEM_WS2_C2V=SEM_WS2_C2V,
        SEM_WS3_C2V=SEM_WS3_C2V,
        SEM_WS3_V2C=SEM_WS3_V2C,
        SIG_K_L1=SIG_K_L1,
        SIG_P_L1=SIG_P_L1,
        SIG_V_L1=SIG_V_L1,
        SIG_Q_L1=SIG_Q_L1,
        SIG_L0AB=SIG_L0AB,
        SIG_L0KC=SIG_L0KC,
        SIG_L0AO=SIG_L0AO,
        SIG_L0AS=SIG_L0AS,
        SIG_L0C=SIG_L0C,
        SIG_IO_UB=SIG_IO_UB,
        SIG_IO3=SIG_IO3,
        SIG_S_HALF=SIG_S_HALF,
        io3_is_alias=io3_is_alias,
        sig_io3=sig_io3,
    )


def select_form(
    B,
    S,
    S_kv,
    N,
    D,
    dtype="float16",
    scaleValue=-1.0,
    is_causal=False,
):
    """Name of the kernel form this shape dispatches to.

    One of FORM_A / FORM_B_MMA / FORM_B_GEMM_V0. Derived from derive_config()
    rather than from a restated predicate, so the wrapper, the three factories and
    the dispatch-guard test cannot disagree. B / S_kv / N / dtype / scaleValue /
    is_causal do not influence the choice -- the form depends on (S, D) alone --
    they are accepted so a caller can hand over the full argument list."""
    cfg = derive_config(B, S, S_kv, N, D, dtype, scaleValue, is_causal)
    if cfg.form_a:
        return FORM_A
    if cfg.form_b_mma:
        return FORM_B_MMA
    return FORM_B_GEMM_V0


# ==================== The Form B-mma GEMM1/GEMM2 bodies are INLINE-EXPANDED
# at their four call sites (single-block / generic x GEMM1 / GEMM2). Do not
# factor them into a plain-Python closure emitter: TVMScript resolves T.*
# frames at parse time, so such an emitter's emissions are silently dropped.
# The kernel still compiles, but the generated C scope then holds only the
# caller-level instructions and the aicore hangs on an orphaned
# WaitFlag<M_FIX>. emit_v is a @T.macro precisely because a macro survives
# parse-time scoping; an inline body needs no such wrapper. ==============
# ==================== V-side buffer allocation is shared by all forms but
# must be emitted inside each prim_func (TVMScript scoping). Form dispatch is
# a Python-level branch that builds one self-contained prim_func. ========


def make_emit_v(cfg):
    """Bind the shared V-scope @T.macro to one host configuration.

    Returns the emit_v macro object, to be stored in a local of the calling jit
    factory so that the prim_func captures it as a closure variable.

    Every name the macro body reads from its enclosing scope is unpacked below
    under exactly the name it has in the macro body: @T.macro is hygienic, so at
    expansion time TVMScript builds a fresh variable table holding only
    `{**emit_v.__globals__, **emit_v co_freevars}` plus the macro's own 28
    parameters. A name missing from this unpack would not raise -- it would simply
    fail to resolve, and the emission would vanish from the generated V scope.
    """
    B = cfg.B
    N = cfg.N
    S = cfg.S
    SEM_WS1_C2V = cfg.SEM_WS1_C2V
    SEM_WS1_V2C = cfg.SEM_WS1_V2C
    SEM_WS2_C2V = cfg.SEM_WS2_C2V
    SEM_WS2_V2C = cfg.SEM_WS2_V2C
    SEM_WS3_C2V = cfg.SEM_WS3_C2V
    SEM_WS3_V2C = cfg.SEM_WS3_V2C
    SIG_IO_UB = cfg.SIG_IO_UB
    SIG_S_HALF = cfg.SIG_S_HALF
    S_kv = cfg.S_kv
    accum_dtype = cfg.accum_dtype
    block_M = cfg.block_M
    block_N = cfg.block_N
    causal_offset = cfg.causal_offset
    flat_v = cfg.flat_v
    form_a = cfg.form_a
    half_M = cfg.half_M
    io3_is_alias = cfg.io3_is_alias
    is_causal = cfg.is_causal
    kv_iters = cfg.kv_iters
    need_mask = cfg.need_mask
    offset_eff = cfg.offset_eff
    s_blocks = cfg.s_blocks
    single_kv_block = cfg.single_kv_block
    sm_scale = cfg.sm_scale

    # ==================== Shared V-scope emitter (@T.macro) ====================
    @T.macro
    def emit_v(
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
        sig_io3_base,
    ):
        # init: pre-set the tokens whose consumers would otherwise wait first
        T.set_cross_flag("MTE2", SEM_WS1_V2C)
        T.set_cross_flag("MTE2", SEM_WS3_V2C)
        T.set_flag("V", "MTE2", SIG_IO_UB)
        T.set_flag("V", "MTE2", SIG_IO_UB + 1)
        if not io3_is_alias:
            T.set_flag("V", "MTE2", sig_io3_base)
            T.set_flag("V", "MTE2", sig_io3_base + 1)
        T.set_flag("MTE3", "V", SIG_S_HALF)

        if single_kv_block:
            # ---- Task-batched single-KV-block V path (both mma forms) ----
            # kv_iters == 1, so one protocol round batches NUM_STAGES *tasks*:
            # slot i <-> task my_start + kb*NUM_STAGES + i. The per-slot op
            # sequence is identical to the generic per-task flow. num_tb and
            # batch_tasks MUST be derived from exactly the same formulas as the
            # C-side mirror, because the two scopes advance their flag pairing in
            # lockstep and any divergence deadlocks.
            # The branch serves Form A and Form B alike: the Form A C scope has
            # its own matching `if single_kv_block:` task batch, so the V side
            # needs no form predicate here.
            num_tb = T.ceildiv(my_count, NUM_STAGES)
            for kb in T.serial(num_tb):
                _remaining_t = my_count - kb * NUM_STAGES
                batch_tasks = T.if_then_else(_remaining_t < NUM_STAGES, _remaining_t, NUM_STAGES)

                # ---- Phase 1 batch: per-task softmax -> w2 slots ----
                T.wait_cross_flag(SEM_WS2_C2V)
                for i in T.serial(batch_tasks):
                    task = my_start + kb * NUM_STAGES + i
                    bx = task % s_blocks
                    s0 = bx * block_M  # loop-body scope: TVMScript bans cross-block defs
                    io_side = i % 2
                    T.tile.fill(sumexp, 0.0)
                    T.tile.fill(m_i, -(2**30))
                    if need_mask:
                        T.tile.arith_progression(row_thresh, s0 + vid * half_M + offset_eff, 1, half_M)
                        T.tile.min(row_thresh, row_thresh, T.cast(S_kv - 1, accum_dtype))

                    # w1[cid, i, vid half] -> io_buf -> work_ub (fp32 mainline)
                    T.wait_flag("V", "MTE2", SIG_IO_UB + io_side)
                    if i % CROSS_INTERVAL == 0:
                        T.wait_cross_flag(SEM_WS1_C2V)
                    T.copy(
                        workspace_1[cid, i, vid * half_M : vid * half_M + half_M, :],
                        io_buf[io_side, :, :],
                    )
                    T.set_flag("MTE2", "V", SIG_IO_UB + io_side)
                    T.wait_flag("MTE2", "V", SIG_IO_UB + io_side)
                    T.copy(io_buf[io_side, :, :], work_ub)
                    T.set_flag("V", "MTE2", SIG_IO_UB + io_side)

                    T.copy(m_i, m_i_prev)  # snapshot the previous max BEFORE reduce
                    # Golden score grid (precision invariant 1): this exact op order
                    T.copy(work_ub, rr_ub)  # r16(qk)
                    T.copy(rr_ub, work_ub)
                    T.tile.mul(work_ub, work_ub, sm_scale)  # * s in fp32
                    T.copy(work_ub, rr_ub)  # r16(scores) = golden grid
                    T.copy(rr_ub, work_ub)

                    if need_mask:  # noqa: SIM102 - parse-time outer / runtime inner
                        # Single block => idx == 0. The condition is uniform over
                        # the s0 block, so both vids take the same branch: flag
                        # pairing stays balanced and this is a pure-V segment.
                        if block_N - 1 > T.min(s0 + offset_eff, S_kv - 1):
                            T.tile.arith_progression(col_pos, 0, 1, block_N)
                            T.tile.broadcast(buf_2d, col_pos, axis=0)  # [bN] -> [hM,bN]
                            T.tile.broadcast(rt_2d, row_thresh, axis=1)  # [hM] -> [hM,bN]
                            T.tile.compare(mask_2d, buf_2d, rt_2d, "LE")
                            T.tile.select(
                                work_ub,
                                mask_2d,
                                work_ub,
                                -T.infinity(accum_dtype),
                                "VSEL_TENSOR_SCALAR_MODE",
                            )

                    T.reduce_max(work_ub, m_i, dim=-1)
                    T.tile.max(m_i, m_i, m_i_prev)
                    T.tile.sub(r_factors[i, :, :], m_i_prev, m_i)
                    T.tile.broadcast(buf_2d, m_i, axis=1)
                    T.tile.sub(work_ub, work_ub, buf_2d)
                    T.tile.exp(work_ub, work_ub)  # P = exp(s - m_new) in [0, 1]
                    T.reduce_sum(work_ub, sumexp_is[i, :, :], dim=-1)

                    # Pre-normalization specialization (precision invariant 2):
                    # divide by sumexp BEFORE the dtype rounding, which is what
                    # reproduces the golden one-shot softmax grid.
                    T.tile.add(sumexp, sumexp_is[i, :, :], 1e-30)
                    T.tile.broadcast(buf_2d, sumexp, axis=1)
                    T.tile.div(work_ub, work_ub, buf_2d)

                    # P -> w2 slot i (SIG_S_HALF ownership closed loop)
                    T.wait_flag("MTE3", "V", SIG_S_HALF)
                    T.copy(work_ub, acc_s_half)  # fp32 -> dtype (P native grid)
                    T.set_flag("V", "MTE3", SIG_S_HALF)
                    T.wait_flag("V", "MTE3", SIG_S_HALF)
                    T.copy(
                        acc_s_half,
                        workspace_2[cid, i, vid * half_M : vid * half_M + half_M, :],
                    )
                    T.set_flag("MTE3", "V", SIG_S_HALF)
                    if (i + 1) % CROSS_INTERVAL == 0 or i == batch_tasks - 1:
                        T.set_cross_flag("MTE3", SEM_WS2_V2C)
                T.set_cross_flag("MTE2", SEM_WS1_V2C)

                # ---- Phase 2 batch: per-task workspace_3 slot -> acc_o -> writeback.
                # Form B only. Under the pre-normalization specialization
                # O = P_hat @ V is already final, so the Form A single-block C
                # scope writes Output directly from l0c_o and neither this phase
                # nor the workspace_3 semaphore round exists there.
                # `form_a` is a parse-time Python bool captured by the closure, so
                # the branch is resolved before lowering (same mechanism as
                # `if not io3_is_alias:` in the init block above).
                # The init pre-set of SEM_WS3_V2C is deliberately kept even on the
                # Form A path: it is a dangling-but-harmless token, exactly like
                # the by-design dangling SEM_WS1_V2C +1, because cross-core
                # semaphore state is per-launch and the C side never waits on it.
                if not form_a:
                    for i in T.serial(batch_tasks):
                        task = my_start + kb * NUM_STAGES + i
                        bx = task % s_blocks
                        by = task // s_blocks % N
                        bz = task // s_blocks // N % B
                        s0 = bx * block_M
                        valid_m = T.min(block_M, S - s0)
                        rows_vid = T.min(half_M, T.max(0, valid_m - vid * half_M))
                        io_side = i % 2
                        T.wait_flag("V", "MTE2", sig_io3_base + io_side)
                        if i % CROSS_INTERVAL == 0:
                            T.wait_cross_flag(SEM_WS3_C2V)
                        T.copy(
                            workspace_3[cid, i, vid * half_M : vid * half_M + half_M, :],
                            io3[io_side, :, :],
                        )
                        T.set_flag("MTE2", "V", sig_io3_base + io_side)
                        T.wait_flag("MTE2", "V", sig_io3_base + io_side)
                        T.tile.fill(acc_o, 0.0)
                        T.tile.add(acc_o, acc_o, io3[io_side, :, :])
                        T.set_flag("V", "MTE2", sig_io3_base + io_side)
                        # writeback inside the SIG_S_HALF closed loop + rows_vid guard
                        T.wait_flag("MTE3", "V", SIG_S_HALF)
                        T.copy(acc_o, out_half)  # fp32 -> dtype
                        T.set_flag("V", "MTE3", SIG_S_HALF)
                        T.wait_flag("V", "MTE3", SIG_S_HALF)
                        if rows_vid > 0:
                            T.copy(
                                out_half[0:rows_vid, :],
                                Output[bz, s0 + vid * half_M : s0 + vid * half_M + rows_vid, by, :],
                            )
                        T.set_flag("MTE3", "V", SIG_S_HALF)
                    T.set_cross_flag("MTE2", SEM_WS3_V2C)

        elif flat_v:
            # ---- Flat (task, kv) slot stream, V side: the generic multi-block
            # prefill of both mma forms.
            # Task-state resets MUST sit in phase 2's kv == 0 slot, not phase 1's:
            # within a round phase 1 runs to completion before phase 2 starts, so
            # resetting acc_o/sumexp in phase 1 would destroy the PREVIOUS task's
            # in-flight accumulation that is still waiting for its phase 2 tail.
            # m_i is the exception -- it is phase-1 exclusive, and reduce_max
            # overwrite semantics plus the m_i_prev chain make fill -> copy
            # equivalent to the generic task-head sequence.
            # Delta versus the single-block path: under causal masking
            # eff_kv_iters is runtime-nonuniform per task, so the flat index ->
            # (task, kv) mapping is a sequential walk state machine (T.alloc_var,
            # O(1) amortised per slot plus a one-time O(my_count) total_flat
            # walk). C and V compute it IDENTICALLY from the same
            # my_start/my_count and the same causal formula; it is never
            # transmitted through a workspace. The per-slot op sequence is the
            # generic one verbatim -- only batch membership changes, so task
            # boundaries do not drain the pipeline.
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

                # ---- Phase 1: batched online softmax (w1 -> P -> w2), flat slots ----
                T.wait_cross_flag(SEM_WS2_C2V)
                tl_p1 = T.alloc_var("int32", init=tl_head)
                kv_p1 = T.alloc_var("int32", init=kv_head)
                for i in T.serial(batch_iters):
                    io_side = i % 2
                    task = my_start + tl_p1
                    bx = task % s_blocks
                    s0 = bx * block_M
                    eff = T.floordiv(T.min(s0 + block_M - 1 + causal_offset, S_kv - 1), block_N) + 1 if is_causal else kv_iters
                    idx = kv_p1
                    # w1[cid, i, vid half] -> io_buf -> work_ub (fp32 mainline)
                    T.wait_flag("V", "MTE2", SIG_IO_UB + io_side)
                    if i % CROSS_INTERVAL == 0:
                        T.wait_cross_flag(SEM_WS1_C2V)
                    T.copy(
                        workspace_1[cid, i, vid * half_M : vid * half_M + half_M, :],
                        io_buf[io_side, :, :],
                    )
                    T.set_flag("MTE2", "V", SIG_IO_UB + io_side)
                    T.wait_flag("MTE2", "V", SIG_IO_UB + io_side)
                    T.copy(io_buf[io_side, :, :], work_ub)
                    T.set_flag("V", "MTE2", SIG_IO_UB + io_side)

                    if kv_p1 == 0:
                        # Task-start resets: phase-1 EXCLUSIVE state only (m_i, see
                        # the note above; acc_o/sumexp are reset in phase 2's
                        # kv == 0 slot) plus the row_thresh recompute. The
                        # thresholds are task-invariant and this is a pure-V
                        # segment carrying no flags.
                        T.tile.fill(m_i, -(2**30))
                        if need_mask:
                            T.tile.arith_progression(row_thresh, s0 + vid * half_M + offset_eff, 1, half_M)
                            T.tile.min(row_thresh, row_thresh, T.cast(S_kv - 1, accum_dtype))

                    T.copy(m_i, m_i_prev)  # snapshot the previous max BEFORE reduce
                    # Golden score grid (precision invariant 1): this exact op order
                    T.copy(work_ub, rr_ub)  # r16(qk)
                    T.copy(rr_ub, work_ub)
                    T.tile.mul(work_ub, work_ub, sm_scale)  # * s in fp32
                    T.copy(work_ub, rr_ub)  # r16(scores) = golden grid
                    T.copy(rr_ub, work_ub)

                    if need_mask:  # noqa: SIM102 - parse-time outer / runtime inner
                        # Runtime skip for fully visible blocks (precision
                        # invariant 4). The condition is uniform over the s0 block,
                        # so both vids take the same branch: flag pairing stays
                        # balanced and either way this is a pure-V segment.
                        if idx * block_N + block_N - 1 > T.min(s0 + offset_eff, S_kv - 1):
                            j0 = idx * block_N
                            T.tile.arith_progression(col_pos, j0, 1, block_N)
                            T.tile.broadcast(buf_2d, col_pos, axis=0)  # [bN] -> [hM,bN]
                            T.tile.broadcast(rt_2d, row_thresh, axis=1)  # [hM] -> [hM,bN]
                            T.tile.compare(mask_2d, buf_2d, rt_2d, "LE")
                            T.tile.select(
                                work_ub,
                                mask_2d,
                                work_ub,
                                -T.infinity(accum_dtype),
                                "VSEL_TENSOR_SCALAR_MODE",
                            )

                    T.reduce_max(work_ub, m_i, dim=-1)
                    T.tile.max(m_i, m_i, m_i_prev)
                    # Deferred rescale: store the max difference now and exp it in
                    # phase 2. Arithmetically identical to exp-ing per iteration,
                    # and matching the golden grid depends on that identity.
                    T.tile.sub(r_factors[i, :, :], m_i_prev, m_i)
                    T.tile.broadcast(buf_2d, m_i, axis=1)
                    T.tile.sub(work_ub, work_ub, buf_2d)
                    T.tile.exp(work_ub, work_ub)  # P = exp(s - m_new) in [0, 1]
                    T.reduce_sum(work_ub, sumexp_is[i, :, :], dim=-1)

                    # P -> workspace_2 (inside the SIG_S_HALF ownership closed loop)
                    T.wait_flag("MTE3", "V", SIG_S_HALF)
                    T.copy(work_ub, acc_s_half)  # fp32 -> dtype (P native grid)
                    T.set_flag("V", "MTE3", SIG_S_HALF)
                    T.wait_flag("V", "MTE3", SIG_S_HALF)
                    T.copy(
                        acc_s_half,
                        workspace_2[cid, i, vid * half_M : vid * half_M + half_M, :],
                    )
                    T.set_flag("MTE3", "V", SIG_S_HALF)
                    if (i + 1) % CROSS_INTERVAL == 0 or i == batch_iters - 1:
                        T.set_cross_flag("MTE3", SEM_WS2_V2C)
                    # walk advance (pure scalar; eff = THIS slot's task)
                    kv_p1 += 1
                    if kv_p1 == eff:
                        kv_p1 = 0
                        tl_p1 += 1
                T.set_cross_flag("MTE2", SEM_WS1_V2C)

                # ---- Phase 2: O accumulation batch (w3 -> acc_o), flat slots ----
                tl_p2 = T.alloc_var("int32", init=tl_head)
                kv_p2 = T.alloc_var("int32", init=kv_head)
                for i in T.serial(batch_iters):
                    io_side = i % 2
                    task = my_start + tl_p2
                    bx = task % s_blocks
                    by = task // s_blocks % N
                    bz = task // s_blocks // N % B
                    s0 = bx * block_M
                    valid_m = T.min(block_M, S - s0)
                    rows_vid = T.min(half_M, T.max(0, valid_m - vid * half_M))
                    eff = T.floordiv(T.min(s0 + block_M - 1 + causal_offset, S_kv - 1), block_N) + 1 if is_causal else kv_iters
                    if kv_p2 == 0:
                        # Task-start resets: the generic task-head fills migrated
                        # to the flat stream -- same first-consumption point, same
                        # fill values. This MUST be phase 2: a phase 1 reset would
                        # destroy the previous task's in-flight acc_o/sumexp, which
                        # the tail below still consumes.
                        T.tile.fill(acc_o, 0.0)
                        T.tile.fill(sumexp, 0.0)
                    T.tile.exp(r_factors[i, :, :], r_factors[i, :, :])  # r_i
                    T.tile.mul(sumexp, sumexp, r_factors[i, :, :])
                    T.tile.add(sumexp, sumexp, sumexp_is[i, :, :])
                    T.tile.broadcast(r_2d, r_factors[i, :, :], axis=1)  # [hM,1]->[hM,D]
                    T.tile.mul(acc_o, acc_o, r_2d)
                    T.wait_flag("V", "MTE2", sig_io3_base + io_side)
                    if i % CROSS_INTERVAL == 0:
                        T.wait_cross_flag(SEM_WS3_C2V)
                    T.copy(
                        workspace_3[cid, i, vid * half_M : vid * half_M + half_M, :],
                        io3[io_side, :, :],
                    )
                    T.set_flag("MTE2", "V", sig_io3_base + io_side)
                    T.wait_flag("MTE2", "V", sig_io3_base + io_side)
                    T.tile.add(acc_o, acc_o, io3[io_side, :, :])
                    T.set_flag("V", "MTE2", sig_io3_base + io_side)
                    if kv_p2 == eff - 1:
                        # ---- Task tail: final normalization and writeback ride the
                        # task's LAST flat slot. The whole SIG_S_HALF closed loop
                        # sits inside the runtime `if`, so its waits and sets enter
                        # and exit together. The condition is vid-independent --
                        # tl/kv/eff walk identically on both vids -- so both AIVs
                        # branch the same way. ----
                        T.tile.add(sumexp, sumexp, 1e-30)  # all-masked-row guard
                        T.tile.broadcast(r_2d, sumexp, axis=1)
                        T.tile.div(acc_o, acc_o, r_2d)
                        T.wait_flag("MTE3", "V", SIG_S_HALF)
                        T.copy(acc_o, out_half)  # fp32 -> dtype
                        T.set_flag("V", "MTE3", SIG_S_HALF)
                        T.wait_flag("V", "MTE3", SIG_S_HALF)
                        if rows_vid > 0:
                            T.copy(
                                out_half[0:rows_vid, :],
                                Output[bz, s0 + vid * half_M : s0 + vid * half_M + rows_vid, by, :],
                            )
                        T.set_flag("MTE3", "V", SIG_S_HALF)
                    # walk advance
                    kv_p2 += 1
                    if kv_p2 == eff:
                        kv_p2 = 0
                        tl_p2 += 1
                T.set_cross_flag("MTE2", SEM_WS3_V2C)
                # batch-head state advance (the phase 2 walk ended at the next head)
                tl_head = tl_p2
                kv_head = kv_p2

        else:
            for t in T.serial(my_count):
                task = my_start + t
                bx = task % s_blocks
                by = task // s_blocks % N
                bz = task // s_blocks // N % B
                s0 = bx * block_M
                valid_m = T.min(block_M, S - s0)
                # Causal KV runtime trim. MUST match the C-scope formula verbatim:
                # both scopes derive their trip counts from it and advance their
                # flag pairing in lockstep.
                eff_kv_iters = T.floordiv(T.min(s0 + block_M - 1 + causal_offset, S_kv - 1), block_N) + 1 if is_causal else kv_iters
                eff_num_outer = T.ceildiv(eff_kv_iters, NUM_STAGES)
                rows_vid = T.min(half_M, T.max(0, valid_m - vid * half_M))

                T.tile.fill(acc_o, 0.0)
                T.tile.fill(sumexp, 0.0)
                T.tile.fill(m_i, -(2**30))
                if need_mask:
                    # task-invariant row thresholds (hoisted; vid half-region start):
                    # thresh_r = min(s0 + vid*hM + r + offset_eff, S_kv-1)
                    T.tile.arith_progression(row_thresh, s0 + vid * half_M + offset_eff, 1, half_M)
                    T.tile.min(row_thresh, row_thresh, T.cast(S_kv - 1, accum_dtype))

                for k in T.serial(eff_num_outer):
                    _remaining = eff_kv_iters - k * NUM_STAGES
                    batch_iters = T.if_then_else(_remaining < NUM_STAGES, _remaining, NUM_STAGES)

                    # ---- Phase 1: batched online softmax (w1 -> P -> w2) ----
                    T.wait_cross_flag(SEM_WS2_C2V)
                    for i in T.serial(batch_iters):
                        io_side = i % 2
                        idx = k * NUM_STAGES + i
                        # w1[cid, i, vid half] -> io_buf -> work_ub (fp32 mainline)
                        T.wait_flag("V", "MTE2", SIG_IO_UB + io_side)
                        if i % CROSS_INTERVAL == 0:
                            T.wait_cross_flag(SEM_WS1_C2V)
                        T.copy(
                            workspace_1[cid, i, vid * half_M : vid * half_M + half_M, :],
                            io_buf[io_side, :, :],
                        )
                        T.set_flag("MTE2", "V", SIG_IO_UB + io_side)
                        T.wait_flag("MTE2", "V", SIG_IO_UB + io_side)
                        T.copy(io_buf[io_side, :, :], work_ub)
                        T.set_flag("V", "MTE2", SIG_IO_UB + io_side)

                        T.copy(m_i, m_i_prev)  # snapshot the previous max BEFORE reduce
                        # Golden score grid (precision invariant 1): round16(qk) ->
                        # * s in fp32 -> round16(scores). This chain order is part
                        # of the golden grid and must not be reordered.
                        # rr_ub is the dedicated round-trip scratch and must NEVER
                        # alias acc_s_half, which is the MTE3 cross-core copy source
                        # under the SIG_S_HALF ownership loop.
                        T.copy(work_ub, rr_ub)  # r16(qk)
                        T.copy(rr_ub, work_ub)
                        T.tile.mul(work_ub, work_ub, sm_scale)  # * s in fp32
                        T.copy(work_ub, rr_ub)  # r16(scores) = golden grid
                        T.copy(rr_ub, work_ub)

                        if need_mask:  # noqa: SIM102 - the two-level split is deliberate:
                            # the outer `if` is PARSE-TIME (a Python bool selecting the
                            # kernel variant) while the inner `if` is a RUNTIME TIR
                            # condition. Folding them into `need_mask and <TIR expr>`
                            # would hand a BoolOp AST node to the TVMScript parser.
                            # Runtime skip for fully visible blocks (precision
                            # invariant 4): the condition is uniform over the s0
                            # block, so both vids take the same branch and flag
                            # pairing stays balanced either way.
                            if idx * block_N + block_N - 1 > T.min(s0 + offset_eff, S_kv - 1):
                                j0 = idx * block_N
                                T.tile.arith_progression(col_pos, j0, 1, block_N)
                                T.tile.broadcast(buf_2d, col_pos, axis=0)  # [bN] -> [hM,bN]
                                T.tile.broadcast(rt_2d, row_thresh, axis=1)  # [hM] -> [hM,bN]
                                T.tile.compare(mask_2d, buf_2d, rt_2d, "LE")
                                T.tile.select(
                                    work_ub,
                                    mask_2d,
                                    work_ub,
                                    -T.infinity(accum_dtype),
                                    "VSEL_TENSOR_SCALAR_MODE",
                                )

                        T.reduce_max(work_ub, m_i, dim=-1)
                        T.tile.max(m_i, m_i, m_i_prev)
                        # Deferred rescale: store the max difference now and exp it
                        # in phase 2. Arithmetically identical to exp-ing per
                        # iteration, and the golden grid depends on that identity.
                        T.tile.sub(r_factors[i, :, :], m_i_prev, m_i)
                        T.tile.broadcast(buf_2d, m_i, axis=1)
                        T.tile.sub(work_ub, work_ub, buf_2d)
                        T.tile.exp(work_ub, work_ub)  # P = exp(s - m_new) in [0, 1]
                        T.reduce_sum(work_ub, sumexp_is[i, :, :], dim=-1)

                        if single_kv_block:
                            # Pre-normalization specialization (precision invariant
                            # 2): P_hat = P/(sumexp+1e-30) BEFORE the dtype
                            # rounding, which is the golden one-shot softmax grid.
                            # Phase 2 then skips the rescale and the final division.
                            T.tile.add(sumexp, sumexp_is[0, :, :], 1e-30)
                            T.tile.broadcast(buf_2d, sumexp, axis=1)
                            T.tile.div(work_ub, work_ub, buf_2d)

                        # P -> workspace_2 (SIG_S_HALF ownership closed loop)
                        T.wait_flag("MTE3", "V", SIG_S_HALF)
                        T.copy(work_ub, acc_s_half)  # fp32 -> dtype (P native grid)
                        T.set_flag("V", "MTE3", SIG_S_HALF)
                        T.wait_flag("V", "MTE3", SIG_S_HALF)
                        T.copy(
                            acc_s_half,
                            workspace_2[cid, i, vid * half_M : vid * half_M + half_M, :],
                        )
                        T.set_flag("MTE3", "V", SIG_S_HALF)
                        if (i + 1) % CROSS_INTERVAL == 0 or i == batch_iters - 1:
                            T.set_cross_flag("MTE3", SEM_WS2_V2C)
                    T.set_cross_flag("MTE2", SEM_WS1_V2C)

                    # ---- Phase 2: O accumulation batch (w3 -> acc_o) ----
                    for i in T.serial(batch_iters):
                        if not single_kv_block:
                            T.tile.exp(r_factors[i, :, :], r_factors[i, :, :])  # r_i
                            T.tile.mul(sumexp, sumexp, r_factors[i, :, :])
                            T.tile.add(sumexp, sumexp, sumexp_is[i, :, :])
                            T.tile.broadcast(r_2d, r_factors[i, :, :], axis=1)  # [hM,1]->[hM,D]
                            T.tile.mul(acc_o, acc_o, r_2d)
                        io_side = i % 2
                        T.wait_flag("V", "MTE2", sig_io3_base + io_side)
                        if i % CROSS_INTERVAL == 0:
                            T.wait_cross_flag(SEM_WS3_C2V)
                        T.copy(
                            workspace_3[cid, i, vid * half_M : vid * half_M + half_M, :],
                            io3[io_side, :, :],
                        )
                        T.set_flag("MTE2", "V", sig_io3_base + io_side)
                        T.wait_flag("MTE2", "V", sig_io3_base + io_side)
                        T.tile.add(acc_o, acc_o, io3[io_side, :, :])
                        T.set_flag("V", "MTE2", sig_io3_base + io_side)
                    T.set_cross_flag("MTE2", SEM_WS3_V2C)

                # ---- task tail: final normalization (generic path) ----
                if not single_kv_block:
                    T.tile.add(sumexp, sumexp, 1e-30)  # all-masked-row guard
                    T.tile.broadcast(r_2d, sumexp, axis=1)
                    T.tile.div(acc_o, acc_o, r_2d)
                # Writeback inside the SIG_S_HALF closed loop, under the rows_vid
                # runtime guard. The final store must stay inside the ownership
                # loop: out_half is released to MTE3 by the loop's closing set.
                T.wait_flag("MTE3", "V", SIG_S_HALF)
                T.copy(acc_o, out_half)  # fp32 -> dtype
                T.set_flag("V", "MTE3", SIG_S_HALF)
                T.wait_flag("V", "MTE3", SIG_S_HALF)
                if rows_vid > 0:
                    T.copy(
                        out_half[0:rows_vid, :],
                        Output[bz, s0 + vid * half_M : s0 + vid * half_M + rows_vid, by, :],
                    )
                T.set_flag("MTE3", "V", SIG_S_HALF)

        # destroy: consume exactly the tokens pre-set by the init block
        T.wait_flag("V", "MTE2", SIG_IO_UB)
        T.wait_flag("V", "MTE2", SIG_IO_UB + 1)
        if not io3_is_alias:
            T.wait_flag("V", "MTE2", sig_io3_base)
            T.wait_flag("V", "MTE2", sig_io3_base + 1)
        T.wait_flag("MTE3", "V", SIG_S_HALF)

    return emit_v


# ==================== Runnable self-check entries ====================
# examples/bench_test.sh's legacy runner executes nearly every .py under
# examples/<op>/ as a standalone script and only counts it as passed when the
# output contains a pass token ("Test Passed!" / "Kernel Output Match"). Every
# module of this operator therefore carries a real, __name__-guarded executable
# entry: the guard keeps `import <module>` side-effect free (mha.py, test_mha.py
# and the three form modules all import each other), while `python <module>.py`
# runs an actual check. run_form_self_check() below is the single on-device
# numerical check shared by the three form modules (DRY: one torch reference and
# one tolerance gate, not three copies); mha_common.py's own entry is the
# device-free host quick-check _host_self_check().


def run_form_self_check(
    kernel_factory,
    *,
    form,
    B,
    S,
    S_kv,
    N,
    D,
    dtype="float16",
    scaleValue=-1.0,
    is_causal=False,
):
    """On-device numerical self-check for one form module, shared by all three.

    Host-side orchestration only; the kernel it builds and runs is the calling
    module's own. Steps:

    1. Assert the shape really dispatches to ``form`` via select_form() -- the
       same single source mha_kernel() routes through. This is the key guard: a
       mistyped shape would otherwise let a form module's entry silently build
       and test a *different* form, leaving this module's kernel unexercised
       while CI still prints the pass token.
    2. Build the kernel by calling ``kernel_factory`` directly (not through
       mha.py's wrapper), so there is no second instance of the same kernel.
    3. Run deterministic inputs (torch.manual_seed(0)) on the npu.
    4. Compare against a torch reference on the kernel's own rounding grid.
    5. Apply the repo's mixed-tolerance gate (identical table to mha.py and
       test_mha.py) and print ``Test Passed!`` as the last line.

    All torch / torch_npu work happens inside the body, so importing this module
    stays free of device side effects.
    """
    import torch

    # (1) Dispatch guard: the shape must belong to THIS form.
    got = select_form(B, S, S_kv, N, D, dtype, scaleValue, is_causal)
    assert got == form, f"shape (B={B},S={S},S_kv={S_kv},N={N},D={D}) dispatches to {got}, not {form}"

    # (2) Build this module's kernel directly (the jit decorator is lazy, so
    #     importing the module compiled nothing; compilation happens here).
    kernel = kernel_factory(
        B=B,
        S=S,
        S_kv=S_kv,
        N=N,
        D=D,
        dtype=dtype,
        scaleValue=scaleValue,
        is_causal=is_causal,
    )

    # (3) Deterministic inputs, run on device.
    scale = scaleValue if scaleValue > 0 else 1.0 / D**0.5
    tdt = getattr(torch, dtype)
    torch.manual_seed(0)
    query = torch.randn(B, S, N, D, dtype=tdt)
    key = torch.randn(B, S_kv, N, D, dtype=tdt)
    value = torch.randn(B, S_kv, N, D, dtype=tdt)
    out = kernel(query.to("npu"), key.to("npu"), value.to("npu"))
    torch.npu.synchronize()

    # (4) Torch reference, semantics identical to mha.py's ref_mha: computed in
    #     the input dtype over [B, N, S, S_kv] scores as QK^T -> *scale ->
    #     softmax -> @V, with a bottom-right aligned causal mask (j > i +
    #     (S_kv - S)) and fully-masked rows forced to zero weights.
    def _ref(q, k, v, s, causal):
        Sq, Skv = q.shape[1], k.shape[1]
        scores = torch.matmul(q.transpose(1, 2), k.transpose(1, 2).transpose(-2, -1)) * s
        if causal:
            i = torch.arange(Sq).unsqueeze(-1)
            j = torch.arange(Skv).unsqueeze(0)
            scores = scores.masked_fill(j > i + (Skv - Sq), float("-inf"))
        row_max = scores.max(dim=-1, keepdim=True).values
        weights = torch.nn.functional.softmax(scores, dim=-1)
        weights = torch.where(torch.isinf(row_max) & (row_max < 0), torch.zeros_like(weights), weights)
        return torch.matmul(weights, v.transpose(1, 2)).transpose(1, 2).contiguous()

    ref = _ref(query, key, value, scale, is_causal)

    # (5) The mixed-tolerance gate test_mha.py uses: at least 99% of the elements
    #     inside atol + rtol*|ref|, and no element past the dtype's ceiling.
    atol, rtol, max_abs_limit = {"float16": (2**-14, 2**-9, 1e-1), "bfloat16": (2**-10, 2**-6, 1e0)}[dtype]
    out_f, ref_f = out.detach().cpu().float(), ref.float()
    abs_err = (out_f - ref_f).abs()
    matched = (abs_err <= atol + rtol * ref_f.abs()).float().mean().item()
    max_abs = abs_err.max().item()

    name = getattr(kernel_factory, "__name__", form)
    print(f"{name} self-check: B={B} S={S} S_kv={S_kv} N={N} D={D} dtype={dtype} causal={is_causal} scale={scale:.6f}")
    print(f"matched={matched:.5f} (gate >= 0.99)   max_abs_err={max_abs:.3e} (gate <= {max_abs_limit:g})")
    assert matched >= 0.99, f"only {matched:.5f} of elements are within tolerance (gate 0.99)"
    assert max_abs <= max_abs_limit, f"max_abs_err {max_abs:.3e} exceeds the {max_abs_limit:g} ceiling"
    print("Test Passed!")


def _host_self_check():
    """Device-free quick check that mha_common.py runs when executed directly.

    The legacy runner executes this shared module as a script too, so it needs
    its own pass token. It stays entirely on the host -- pure Python scalar
    assertions over derive_config / select_form / make_emit_v, no kernel
    compilation and no on-device compute (derive_config's only device touch is
    _cube_core_num(), a cached host metadata query with a stock fallback) -- so
    it is near-instant and immune to the runner's parallel JIT-cache lock
    contention. It asserts the three contracts the whole operator rests on:

    1. The three-way dispatch is mutually exclusive and exhaustive over (S, D),
       including both edges of the S > 16 threshold and every tiling tier.
    2. derive_config's tiling / iteration / task-split / event-id derivations
       satisfy their documented relations on every grid point.
    3. make_emit_v returns a hygienic emit_v TIRMacro whose captured closure
       covers every free name of the macro body with the config's own values.
    """
    forms = (FORM_A, FORM_B_MMA, FORM_B_GEMM_V0)
    assert NUM_STAGES % 2 == 0, "NUM_STAGES must be even: the io_buf/L0 slots rotate in pairs"
    assert len(set(forms)) == 3, "the three form identifiers must be pairwise distinct"

    n_pts = 0
    # Boundary-covering grid: both edges of S > 16, all three D, and single- vs
    # multi-KV-block S_kv straddling the 128 / 256 / 512 block_N tier edges.
    for S in (1, 2, 16, 17, 32, 128, 2048):
        for D in (64, 128, 256):
            for S_kv in (64, 128, 129, 256, 257, 512, 2048, 4096):
                for is_causal in (False, True):
                    if is_causal and S_kv < S:
                        continue  # out of domain: mha() requires S <= S_kv when causal
                    for scaleValue in (-1.0, 0.5):
                        n_pts += 1
                        cfg = derive_config(2, S, S_kv, 4, D, "float16", scaleValue, is_causal)
                        got = select_form(2, S, S_kv, 4, D, "float16", scaleValue, is_causal)
                        tag = f"(S={S},D={D},S_kv={S_kv},causal={is_causal},scale={scaleValue})"

                        # ---- (1) dispatch: closed, mutually exclusive, exhaustive ----
                        assert got in forms, f"{tag}: select_form returned {got} outside the three identifiers"
                        assert int(cfg.form_a) + int(cfg.form_b_mma) <= 1, f"{tag}: form_a and form_b_mma both set"
                        assert (got == FORM_A) == cfg.form_a, f"{tag}: select_form/derive_config disagree on form_a"
                        assert (got == FORM_B_MMA) == cfg.form_b_mma, f"{tag}: select_form/derive_config disagree on form_b_mma"
                        assert (got == FORM_B_GEMM_V0) == (not cfg.form_a and not cfg.form_b_mma), (
                            f"{tag}: gemm_v0 must be the exact complement of form_a|form_b_mma"
                        )
                        # The form depends on (S, D) alone; restate the documented predicate.
                        expect = FORM_A if (S > 16 and D <= 128) else (FORM_B_MMA if (S > 16 and D == 256) else FORM_B_GEMM_V0)
                        assert got == expect, f"{tag}: form {got} != documented predicate {expect}"

                        # ---- (2) config-derivation invariants ----
                        assert cfg.block_M == (16 if S <= 16 else (32 if D == 256 else 64)), f"{tag}: block_M {cfg.block_M}"
                        assert cfg.block_N in (128, 256, 512), f"{tag}: block_N {cfg.block_N} outside tiers"
                        assert cfg.gemm1_kL0 in (32, 64, 128), f"{tag}: gemm1_kL0 {cfg.gemm1_kL0} outside tiers"
                        if S > 16:
                            assert cfg.block_N == 128 and cfg.gemm1_kL0 == 128, f"{tag}: prefill tier must be bN=128,kL0=128"
                        assert cfg.half_M == cfg.block_M // 2, f"{tag}: half_M != block_M // 2"
                        assert cfg.s_blocks == -(-S // cfg.block_M), f"{tag}: s_blocks != ceil(S/block_M)"
                        assert cfg.kv_iters == -(-S_kv // cfg.block_N), f"{tag}: kv_iters != ceil(S_kv/block_N)"
                        assert cfg.single_kv_block == (cfg.kv_iters == 1) == (S_kv <= cfg.block_N), f"{tag}: single_kv_block"
                        assert cfg.flat_v == (S > 16 and not cfg.single_kv_block), f"{tag}: flat_v"
                        assert cfg.kv_db == (S <= 16 and D == 256 and not cfg.single_kv_block), f"{tag}: kv_db domain"
                        assert cfg.need_bound_mask == (S_kv % cfg.block_N != 0), f"{tag}: need_bound_mask"
                        assert cfg.need_mask == (is_causal or cfg.need_bound_mask), f"{tag}: need_mask"
                        assert cfg.causal_offset == S_kv - S, f"{tag}: causal_offset"
                        assert cfg.offset_eff == (S_kv - S if is_causal else 2**30), f"{tag}: offset_eff"
                        assert cfg.block_num == 2 * 4 * cfg.s_blocks, f"{tag}: block_num != B*N*s_blocks"
                        assert 1 <= cfg.core_num <= cfg.block_num, f"{tag}: core_num {cfg.core_num} out of [1, block_num]"
                        assert cfg.q_tasks * cfg.core_num + cfg.r_tasks == cfg.block_num, f"{tag}: task split not an exact cover"
                        assert 0 <= cfg.r_tasks < cfg.core_num, f"{tag}: r_tasks {cfg.r_tasks} out of [0, core_num)"
                        exp_scale = scaleValue if scaleValue > 0 else (1.0 / D) ** 0.5
                        assert cfg.sm_scale == exp_scale, f"{tag}: sm_scale {cfg.sm_scale} != {exp_scale}"
                        assert cfg.io3_is_alias == (cfg.block_N == D), f"{tag}: io3_is_alias"
                        assert cfg.sig_io3 == (cfg.SIG_IO_UB if cfg.io3_is_alias else cfg.SIG_IO3), f"{tag}: sig_io3"
                        sem_ids = {
                            cfg.SEM_WS1_C2V,
                            cfg.SEM_WS1_V2C,
                            cfg.SEM_WS2_V2C,
                            cfg.SEM_WS2_C2V,
                            cfg.SEM_WS3_C2V,
                            cfg.SEM_WS3_V2C,
                        }
                        assert sem_ids == set(range(6)), f"{tag}: cross-core SEM ids must be exactly 0..5, got {sorted(sem_ids)}"

    print(f"[HOST-SELF-CHECK] dispatch/config grid: {n_pts} points, all invariants held")

    # ---- (3) make_emit_v contract on representative configs (one per domain) ----
    cfg_fields = {f.name for f in dataclasses.fields(MhaConfig)}
    repr_points = [
        ("form_a_multi", (2, 128, 256, 8, 64), False),
        ("form_a_single", (2, 128, 128, 8, 64), True),
        ("form_b_mma_multi", (2, 32, 512, 4, 256), False),
        ("form_b_mma_single", (1, 64, 64, 2, 256), False),
        ("gemm_v0_kvdb", (2, 1, 512, 8, 256), False),
        ("gemm_v0_single", (1, 1, 128, 4, 256), False),
        ("gemm_v0_d64", (2, 1, 64, 2, 64), False),
    ]
    for name, (b, s, skv, n, d), causal in repr_points:
        cfg = derive_config(b, s, skv, n, d, "float16", -1.0, causal)
        ev = make_emit_v(cfg)
        assert type(ev).__name__ == "TIRMacro", f"{name}: make_emit_v returned {type(ev).__name__}, not a TIRMacro"
        assert ev.__name__ == "emit_v", f"{name}: macro __name__ is {ev.__name__}, not emit_v"
        assert ev.hygienic is True, f"{name}: emit_v must be hygienic (@T.macro default)"
        src = ev.source.source
        assert src.strip(), f"{name}: empty macro source"
        for marker in ("T.set_cross_flag", "T.wait_cross_flag", "T.set_flag", "T.wait_flag", "T.copy"):
            assert marker in src, f"{name}: marker {marker} missing from emit_v body"
        freevars = ev.func.__code__.co_freevars
        missing = [v for v in freevars if v not in ev.closure_vars]
        assert not missing, f"{name}: emit_v free names missing from closure_vars: {missing}"
        mismatched = [v for v in freevars if v in cfg_fields and ev.closure_vars[v] != getattr(cfg, v)]
        assert not mismatched, f"{name}: closure value diverges from cfg for {mismatched}"
        print(f"[HOST-SELF-CHECK] make_emit_v[{name}]: hygienic emit_v TIRMacro, {len(freevars)} free names captured with cfg values")

    print("Test Passed!")


if __name__ == "__main__":
    _host_self_check()
