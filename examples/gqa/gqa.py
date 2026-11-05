"""GQA (Grouped Query Attention) forward — single-file delivery.

Flash-attention style online-softmax GQA forward for Ascend NPU. All kernel
variants are merged into this module:

  - ``gqa_expert()`` — the dispatch factory: builds a compile-fallback chain
    of specialized kernels per shape (dense M-packing decode kernels with
    manual-flag deep pipelining, G-stacked prefill kernels, a per-head
    block_M ladder, and a last-resort g2 fallback) and returns the first
    factory that compiles.
  - Custom BSND<->BHSD transpose / pad kernels (``transpose_qkv`` /
    ``transpose_out`` / ``_pad_kv_npu``) — every fp16/bf16 device-side data
    movement is handled by tilelang kernels, because several aclnn binaries
    (Transpose / ZerosLike / FillScalar / ViewCopy on fp16/bf16) are missing
    from some CANN 9.x SoC packages (error 561103). See the notes at each
    call site for the exact avoided-op rationale.

Public API:
    gqa(query, key, value, scaleValue=-1.0, is_causal=False)
        -> output [B, S, N_q, D] in BSND layout (tensor interface).
    gqa_expert(B, S, S_kv, sm_scale, is_causal, ...)
        -> factory that compiles and returns the kernel callable. Layout
        contract: BSND direct supply for S <= 4 (decode), BHSD for S > 4
        (prefill, caller transposes).

Supported: fp16/bf16, D in {64..512}, arbitrary B/N_q/N_kv/S/S_kv —
non-block-aligned shapes are padded in-kernel (no torch device-side pad ops).

Note: the BSND adapter is implemented inline in ``gqa()`` so this file
stays a self-contained single-file example (no cross-module imports).
"""

import tilelang
from tilelang import language as T
from tilelang.intrinsics import make_zn_layout, make_nz_layout
import torch

# ─────────────────────────────────────────────────────────────────────
#  Expert-mode constants
# ─────────────────────────────────────────────────────────────────────

# Ascend910B3 fixed-core count (torch.npu.get_device_properties:
# cube_core_num=20, vector_core_num=40 — the 1:2 MIX_AIC_1_2 grouping).
P28_NUM_CORES = 20

# Expert-mode pass configs: all four AUTO passes OFF (manual C/V
# scopes + manual flag chains + manual memory planning — the flash_attn
# expert template configuration).
P28_EXPERT_PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: False,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: False,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: False,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: False,
}


@tilelang.jit(
    out_idx=[4],
    pass_configs={
        tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
        tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
        tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
        tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
    },
)
def _gqa_kernel_expert(
    B,
    S_padded,
    S_kv_padded,
    sm_scale,
    is_causal,
    N_q=8,
    N_kv=2,
    D=128,
    dtype="float16",
    block_M=32,
    skv_minus_s=0,
    use_mask=True,
    block_N=64,
):
    """GQA forward kernel — Expert mode with L0C Double Buffer.

    Key differences from developer-mode gqa.py:
    - Explicit alloc_L0C [2, ...] instead of alloc_fragment for GEMM output buffers
    - side = k % 2 alternation for double buffer
    - All auto passes ON: T.gemm_v0 internal template handles L0C sync

    Causal block skipping:
    For causal attention, Q rows i in block bx only attend to columns
    j <= i + (S_kv - S) <= (bx+1)*block_M - 1 + skv_minus_s.  A KV block k
    (covering columns [k*block_N, (k+1)*block_N)) is fully masked iff
    k*block_N >= (bx+1)*block_M + skv_minus_s.  Such blocks contribute zero
    to softmax/GEMM2 and are skipped entirely (no K/V/mask load, no GEMM).
    skv_minus_s MUST be the ORIGINAL S_kv - S (pre-padding); padded columns
    are additionally masked by the host mask, so skipping at the original
    boundary is always safe.

    block_N is a JIT parameter supplied by the dispatch factory — the
    gqa_expert factory in this file passes 128 for this kernel (the
    signature default 64 applies only to direct construction).  A larger
    block_N halves the KV iteration count and grows the GEMM1/GEMM2 tiles
    (e.g. [16,128]x[128,128] at block_N=128).
    """
    accum_dtype = "float"

    G = N_q // N_kv
    q_blocks = S_padded // block_M
    kv_iters = S_kv_padded // block_N
    block_num = B * N_q * q_blocks

    # KV loop pipelining. Two framework defects constrain where the
    # pipeline may be enabled:
    #   1. threads=2 + Pipelined races: the phase-grouped C/V loops +
    #      per-vid workspace half-writes lose the strict per-iteration flag
    #      ping-pong of the serial form -> non-deterministic corruption
    #      (NaN / ratio 0.09 / max_abs 1.5e3, varies per process).
    #   2. A causal-skip predicate that actually fires (runtime, cid-dependent
    #      `if k*block_N < (bx+1)*block_M + skip_limit` with skip_limit=0)
    #      SEGFAULTS the compiler when the IfThenElse survives inside a
    #      Pipelined loop. Non-causal (skip_limit = 1<<30) and provably
    #      always-true predicates are folded away and compile fine.
    # Guard: pipeline only NON-CAUSAL loops with kv_iters >= 2 AND
    # kv_iters % 2 == 0, at threads=1 (single-core C/V — the same
    # configuration proven by the dense kernel). Odd kv_iters corrupts: the
    # phase grouping emits k_outer < kv_iters/2 groups (integer division) and
    # silently drops the tail iteration (Skv=300 -> kv=3: ratio 0.003-0.048,
    # missing last KV block). Causal / odd / kv_iters<2 keep the serial
    # threads=2 form.
    # D > 128 is ALSO excluded: the
    # pipelined threads=1 form plans ALL buffers in ONE scope and D=256
    # exhausts the UB envelope (AscendMemoryPlanning "Memory allocation
    # failed for: acc_o_ub required: 65536, new memory available: 0").  The
    # serial
    # threads=2 form splits C/V scopes and compiles at D=256.
    use_pipeline = (not is_causal) and D <= 128 and kv_iters >= 2 and kv_iters % 2 == 0
    pipe_stages = 2 if use_pipeline else 0
    kernel_threads = 1 if use_pipeline else 2

    # running-max init floor.  A fixed small-magnitude init like -(1<<30)
    # (~-1.07e9) CLAMPS the first block's max
    # for rows whose (scaled) max
    # is below it — with full-range fp16 inputs (|scale*s| up to ~5e10)
    # a causal row with few visible columns can hit that, zeroing all its
    # P values (output 0 vs the reference's argmax one-hot).  sm_scale*_BIG_NEG
    # (~-8.8e28) is below any finite fp32 scaled score, so it acts as a pure
    # "no prior max" sentinel.  Bit-identical for normal ranges.
    m_init_floor = sm_scale * _BIG_NEG

    # Compile-time skip threshold: non-causal → never skip (huge limit);
    # causal → original S_kv - S boundary.
    if is_causal:
        skip_limit = skv_minus_s
    else:
        skip_limit = 1 << 30

    @T.prim_func
    def main(
        Q: T.Tensor([B, N_q, S_padded, D], dtype),  # type: ignore  BHSD layout
        K: T.Tensor([B, N_kv, S_kv_padded, D], dtype),  # type: ignore  BHSD layout
        V: T.Tensor([B, N_kv, S_kv_padded, D], dtype),  # type: ignore  BHSD layout
        Mask: T.Tensor([B, S_padded, S_kv_padded], accum_dtype),  # type: ignore
        Output: T.Tensor([B, N_q, S_padded, D], dtype),  # type: ignore  BHSD layout
    ):
        with T.Kernel(block_num, threads=kernel_threads, is_npu=True) as (cid):
            bx = cid % q_blocks
            hq = (cid // q_blocks) % N_q
            bz = cid // (q_blocks * N_q)
            g = hq // G

            # ============= L1 缓存（Q/K/V 输入）============
            q_l1 = T.alloc_shared([block_M, D], dtype)
            k_l1 = T.alloc_shared([block_N, D], dtype)
            v_l1 = T.alloc_shared([block_N, D], dtype)

            # ============= L0C 双缓冲（GEMM 输出）============
            # GEMM1 scores: [2, block_M, block_N], side=k%2 alternation, init=True
            acc_s_l0c = T.alloc_L0C([2, block_M, block_N], accum_dtype)
            # GEMM2 output: [2, block_M, D], side=k%2 alternation, init=True
            acc_o_l0c = T.alloc_L0C([2, block_M, D], accum_dtype)

            # ============= UB 计算缓冲 =============
            acc_s_ub = T.alloc_shared([block_M, block_N], accum_dtype)
            acc_s_half = T.alloc_shared([block_M, block_N], dtype)
            acc_s_l1 = T.alloc_shared([block_M, block_N], dtype)

            m_i = T.alloc_shared([block_M], accum_dtype)
            m_i_prev = T.alloc_shared([block_M], accum_dtype)
            sumexp = T.alloc_shared([block_M], accum_dtype)
            sumexp_i_ub = T.alloc_shared([block_M], accum_dtype)

            acc_o = T.alloc_shared([block_M, D], accum_dtype)
            acc_o_ub = T.alloc_shared([block_M, D], accum_dtype)
            acc_o_half = T.alloc_shared([block_M, D], dtype)

            row_bc_d = T.alloc_shared([block_M, D], accum_dtype)
            row_bc_n = T.alloc_shared([block_M, block_N], accum_dtype)

            mask_block = T.alloc_shared([block_M, block_N], accum_dtype)

            # ============= 初始化 =============
            T.tile.fill(acc_o, 0.0)
            T.tile.fill(sumexp, 0.0)
            # m_init_floor = sm_scale*_BIG_NEG (pure sentinel, see above)
            T.tile.fill(m_i, m_init_floor)

            # BSND strided reads are net-negative (256B rows starve MTE2 bursts) —
            # keep BHSD contiguous reads + host transpose.
            T.copy(Q[bz, hq, bx * block_M : (bx + 1) * block_M, :], q_l1)

            # ============= KV 迭代循环（非 causal 流水化；L0C 双缓冲 + causal 块级跳过）============
            # Skip KV blocks that are fully masked by the causal
            # boundary: block k is needed iff k*block_N < (bx + 1)*block_M + skip_limit.
            # Skipped blocks contribute zero (their softmax weights underflow to 0
            # after the _BIG_NEG (=-1e30) additive mask in the unskipped
            # reference path).
            # num_stages=pipe_stages: 2 for non-causal kv_iters>=2
            # (predicate folds away -> safe), 0 (plain serial) for causal — a
            # firing runtime skip predicate inside a Pipelined loop segfaults
            # the compiler (see kernel-level comment above).
            for k in T.Pipelined(kv_iters, num_stages=pipe_stages):
                if k * block_N < (bx + 1) * block_M + skip_limit:
                    side = k % 2

                    T.copy(K[bz, g, k * block_N : (k + 1) * block_N, :], k_l1)
                    if use_mask:
                        T.copy(
                            Mask[bz, bx * block_M : (bx + 1) * block_M, k * block_N : (k + 1) * block_N],
                            mask_block,
                        )

                    # GEMM1: Q @ K^T → scores (double buffer side, init=True)
                    T.gemm_v0(q_l1, k_l1, acc_s_l0c[side, :, :], transpose_B=True, init=True)

                    T.copy(acc_s_l0c[side, :, :], acc_s_ub)
                    T.tile.mul(acc_s_ub, acc_s_ub, sm_scale)

                    # Max-after-mask order: the additive mask is
                    # applied BEFORE the running max.  With the mask after
                    # the max, a masked column's huge RAW score (full-range
                    # fp16, |s| up to 5.5e11) lifts the max so high that
                    # every VISIBLE column's exp underflows -> output 0
                    # while the reference (mask applied before softmax)
                    # attends normally.  With _BIG_NEG = -1e30 a masked
                    # column can never win the max; the m_init_floor
                    # sentinel keeps
                    # fully-masked rows at P == 0 (the mask survives the
                    # shift: it lives in acc_s_ub, not in the max).
                    if use_mask:
                        T.tile.add(acc_s_ub, acc_s_ub, mask_block)

                    # Online softmax: update running max
                    # clear=True is used here: this kernel compiles fine
                    # with clear=True, and the clear=False lowering inserts
                    # scalar GetValue/SetValue merge loops (~1.8x kernel
                    # slowdown).  clear=False is retained ONLY in the
                    # last-resort g2 fallback kernel (see the note there).
                    T.copy(m_i, m_i_prev)
                    T.reduce_max(acc_s_ub, m_i, dim=-1)
                    T.tile.max(m_i, m_i, m_i_prev)
                    T.tile.sub(m_i_prev, m_i_prev, m_i)
                    T.tile.exp(m_i_prev, m_i_prev)

                    T.tile.broadcast(row_bc_n, m_i, axis=1)
                    T.tile.sub(acc_s_ub, acc_s_ub, row_bc_n)

                    # Exp
                    T.tile.exp(acc_s_ub, acc_s_ub)

                    T.reduce_sum(acc_s_ub, sumexp_i_ub, dim=-1)
                    T.tile.mul(sumexp, sumexp, m_i_prev)
                    T.tile.add(sumexp, sumexp, sumexp_i_ub)

                    T.copy(acc_s_ub, acc_s_half)
                    T.copy(acc_s_half, acc_s_l1)
                    T.copy(V[bz, g, k * block_N : (k + 1) * block_N, :], v_l1)

                    # GEMM2: scores @ V → output chunk (double buffer side, init=True)
                    T.gemm_v0(acc_s_l1, v_l1, acc_o_l0c[side, :, :], init=True)
                    T.copy(acc_o_l0c[side, :, :], acc_o_ub)

                    T.tile.broadcast(row_bc_d, m_i_prev, axis=1)
                    T.tile.mul(acc_o, acc_o, row_bc_d)
                    T.tile.add(acc_o, acc_o, acc_o_ub)

            # ============= Final normalization =============
            # fully-masked-row guard: causal-skip blocks
            # keep sumexp == 0 -> 0/0 = NaN (the reference outputs 0 for fully
            # masked rows).  Add a tiny constant so 0/1e-30 == 0; normal rows
            # have sumexp >= exp(0) = 1 (max-shifted), unchanged in fp32.
            T.tile.add(sumexp, sumexp, 1e-30)
            T.tile.broadcast(row_bc_d, sumexp, axis=1)
            T.tile.div(acc_o, acc_o, row_bc_d)

            T.copy(acc_o, acc_o_half)
            T.copy(
                acc_o_half,
                Output[bz, hq, bx * block_M : (bx + 1) * block_M, :],
            )

    return main


@tilelang.jit(
    out_idx=[4],
    pass_configs={
        tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
        tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
        tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
        tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
    },
)
def _gqa_kernel_prefill_g2(
    B,
    S_padded,
    S_kv_padded,
    sm_scale,
    is_causal,
    N_q=8,
    N_kv=2,
    D=128,
    dtype="float16",
    block_M=64,
    skv_minus_s=0,
    use_mask=True,
    block_N=64,
):
    """Prefill G_inner=2 M-stacking kernel — BHSD layout,
    segment-sequential form.

    One instance serves the 2 Q heads of one (bz, kv-group) at the same bx:
    K/V/mask blocks are loaded ONCE per iteration and shared by both heads'
    GEMM chains (KV GM traffic ÷2 and instance count ÷2 vs the per-head
    kernel — q_blocks/G_batches = (S/BM, ceil(G/2)) vs (S/BM, G)).

    Design (segment-sequential, driven by the memory budget):
      - NO fused [2*BM, ...] compute buffers.  The whole-tile fused form
        needs 3 concurrent 64KB UB buffers (acc_o + acc_s_ub + row_bc_n/mask
        = 196.6KB > the 196,352B shared.ub pool ceiling this kernel's UB
        allocations are planned against — a distinct quantity from the
        host-side _UB_BYTES=196608 transpose-tiling budget) and
        SEGFAULTS/aborts at M_tile=128.  The segment-sequential form instead runs each head
        segment through the EXACT
        per-head kernel chain ([block_M, ...] shapes, all whole-buffer ops,
        all proven GEMM shapes) with per-segment L0C/accumulator buffers —
        the two segments merely SHARE k_l1/v_l1/mask_block and the softmax
        temporaries (sequential lifetimes).
      - threads=1 ALWAYS: (a) the AscendVidReduction sub-tile scatter
        halving defect is structurally incompatible with segment
        scatter at threads=2; (b) threads=2 + Pipelined races.  threads=1
        also removes the cross-core flag chain of the serial threads=2
        ping-pong.
      - grid = B * N_kv * q_blocks * G_batches (G_batches = ceil(G/2) as an
        independent dim — MQA/MHA shapes keep per-head parallelism).
      - causal skip predicate shared by both segments (same bx -> same
        visible-column bound k*block_N < (bx+1)*block_M + skip_limit).
      - mask rows shared: Mask[bz, bx*BM:(bx+1)*BM, ...] loaded ONCE per
        iteration into mask_block (mask depends only on (s, j)) — mask GM
        traffic ÷2 as well.
      - L0C single-buffered per segment with init=True (the [2, ...]
        structural double buffer measured zero-benefit):
        4 x [BM, BN or D] f32 = 128KB at D=128 (exactly the 131,072B
        wmma.accumulator planning limit, boundary-inclusive) / 192KB at
        D=256 (2x[64,128] + 2x[64,256] — the same envelope the per-head
        D256 block_M=64 path runs in production).
      - Pipelined(num_stages=2) only when
        D<=128 AND non-causal AND NOT use_mask AND kv_iters>=2 AND even:
        mask_block x(num_stages+1)=3 versioning must fit the UB pool
        alongside the accumulators, a FIRING causal-skip predicate inside a
        Pipelined loop segfaults the compiler, and
        D256+pipelined is an unbudgeted combination (D256 prefill in
        practice is causal, which the guard already excludes).  Non-causal
        shapes with S_kv % 128 == 0 have use_mask=False and are the only
        pipeline-eligible combination; causal / colpad / odd-kv /
        D256 run serial (num_stages=0 — plain serial, where the runtime
        skip predicate is safe).

    Tail G_batch (G odd): segment 1's q_l1_1 is zero-filled; its softmax
    rows are independent (row-wise max/sum/rescale) and finite, and the
    output scatter is guarded by the LOCAL segment index (gb*2+1 < G —
    NOT hq < N_q: segments beyond G belong to the next kv group).
    """
    accum_dtype = "float"

    G = N_q // N_kv
    G_batches = (G + 1) // 2
    q_blocks = S_padded // block_M
    kv_iters = S_kv_padded // block_N
    block_num = B * N_kv * q_blocks * G_batches
    # tail G_batch exists only when G is odd (last gb serves 1 head); resolved
    # at trace time so even-G shapes emit only the unguarded two-segment path
    tail_possible = (G % 2) != 0

    # pipeline guard — see docstring.  The segment-sequential L1 budget
    # (q_l1 x2 + k/v_l1 x(num_stages+1) + acc_s_l1 x2) fits both D=128
    # (256KB) and D=256 (442KB) under the 524,032B shared.l1 pool, but D256
    # pipelined is an unbudgeted combination (D256 prefill in practice is
    # causal, which the guard already excludes): the budget table
    # ruled "D256 stays serial" (737KB whole-tile-form upper bound) — keep
    # D<=128 in the guard so only the proven combination pipelines.
    use_pipeline = D <= 128 and (not is_causal) and (not use_mask) and kv_iters >= 2 and kv_iters % 2 == 0
    pipe_stages = 2 if use_pipeline else 0

    if is_causal:
        skip_limit = skv_minus_s
    else:
        skip_limit = 1 << 30

    # running-max init floor.  This kernel applies
    # the UNSCALED additive mask (_BIG_NEG = -1e30) to the SCALED scores
    # before the running max (max-after-mask).  A masked column can then never win the
    # max over a visible one (|scale*s| <= ~5e10 << 1e30), and fully-masked
    # rows (mixed blocks) get their max FLOORED at mask/1000 so the mask
    # cannot cancel in the shift: floor = _BIG_NEG*1e-3 lies above the
    # fully-masked masked value (~_BIG_NEG) and below every visible scaled
    # max (>= -5e10).  Replaces the clamp-at-0, which broke full-range
    # rows whose max sat below 0 (all P -> exp(underflow) -> output 0).
    m_init_floor = _BIG_NEG * 1e-3

    @T.prim_func
    def main(
        Q: T.Tensor([B, N_q, S_padded, D], dtype),  # type: ignore  BHSD layout
        K: T.Tensor([B, N_kv, S_kv_padded, D], dtype),  # type: ignore  BHSD layout
        V: T.Tensor([B, N_kv, S_kv_padded, D], dtype),  # type: ignore  BHSD layout
        Mask: T.Tensor([B, S_padded, S_kv_padded], accum_dtype),  # type: ignore
        Output: T.Tensor([B, N_q, S_padded, D], dtype),  # type: ignore  BHSD layout
    ):
        with T.Kernel(block_num, threads=1, is_npu=True) as (cid):
            bx = cid % q_blocks
            gb = (cid // q_blocks) % G_batches
            kg = (cid // (q_blocks * G_batches)) % N_kv
            bz = cid // (q_blocks * G_batches * N_kv)

            # segment-0 head is always in range (gb < G_batches -> gb*2 < G)
            hq0 = kg * G + gb * 2

            # ============= L1 缓存（Q/K/V 输入，两段共享 k/v）============
            q_l1_0 = T.alloc_shared([block_M, D], dtype)
            q_l1_1 = T.alloc_shared([block_M, D], dtype)
            k_l1 = T.alloc_shared([block_N, D], dtype)
            v_l1 = T.alloc_shared([block_N, D], dtype)
            # GEMM2 A operands (per segment)
            acc_s_l1_0 = T.alloc_shared([block_M, block_N], dtype)
            acc_s_l1_1 = T.alloc_shared([block_M, block_N], dtype)

            # ============= L0C（GEMM 输出，分段单缓冲 init=True）============
            acc_s_l0c_0 = T.alloc_L0C([block_M, block_N], accum_dtype)
            acc_s_l0c_1 = T.alloc_L0C([block_M, block_N], accum_dtype)
            acc_o_l0c_0 = T.alloc_L0C([block_M, D], accum_dtype)
            acc_o_l0c_1 = T.alloc_L0C([block_M, D], accum_dtype)

            # ============= UB 计算缓冲（段间独立命名，顺序生存期由 MEMORY_PLANNING 复用）============
            # per-segment names are kept — sharing ANY buffer the
            # two GEMM chains touch (acc_s_ub / acc_s_half / acc_o_ub) breaks
            # CombineCV's workspace pairing ("Mismatch in sync points between
            # cube and vec"), and per-segment names alone raced via the
            # memory-planner's cross-name aliasing (see the manual barrier
            # below for the actual fix).
            acc_s_ub_0 = T.alloc_shared([block_M, block_N], accum_dtype)
            acc_s_ub_1 = T.alloc_shared([block_M, block_N], accum_dtype)
            acc_s_half_0 = T.alloc_shared([block_M, block_N], dtype)
            acc_s_half_1 = T.alloc_shared([block_M, block_N], dtype)

            m_i_0 = T.alloc_shared([block_M], accum_dtype)
            m_i_prev_0 = T.alloc_shared([block_M], accum_dtype)
            sumexp_0 = T.alloc_shared([block_M], accum_dtype)
            m_i_1 = T.alloc_shared([block_M], accum_dtype)
            m_i_prev_1 = T.alloc_shared([block_M], accum_dtype)
            sumexp_1 = T.alloc_shared([block_M], accum_dtype)
            sumexp_i_ub = T.alloc_shared([block_M], accum_dtype)

            acc_o_0 = T.alloc_shared([block_M, D], accum_dtype)
            acc_o_1 = T.alloc_shared([block_M, D], accum_dtype)
            acc_o_ub_0 = T.alloc_shared([block_M, D], accum_dtype)
            acc_o_ub_1 = T.alloc_shared([block_M, D], accum_dtype)
            acc_o_half = T.alloc_shared([block_M, D], dtype)

            row_bc_d_0 = T.alloc_shared([block_M, D], accum_dtype)
            row_bc_d_1 = T.alloc_shared([block_M, D], accum_dtype)
            row_bc_f = T.alloc_shared([block_M, D], accum_dtype)
            row_bc_n_0 = T.alloc_shared([block_M, block_N], accum_dtype)
            row_bc_n_1 = T.alloc_shared([block_M, block_N], accum_dtype)

            mask_block = T.alloc_shared([block_M, block_N], accum_dtype)

            # ============= 初始化 =============
            T.tile.fill(acc_o_0, 0.0)
            T.tile.fill(acc_o_1, 0.0)
            T.tile.fill(sumexp_0, 0.0)
            T.tile.fill(sumexp_1, 0.0)
            # m_init_floor: sentinel + fully-masked-row max floor (see
            # factory note) — replaces a fixed -(1<<30)-style init.
            T.tile.fill(m_i_0, m_init_floor)
            T.tile.fill(m_i_1, m_init_floor)

            # Q load, head-major: segment g holds Q[bz, hq0+g, bx*BM:(bx+1)*BM, :]
            # — each a contiguous 2D box in BHSD.  Tail G_batch (G odd):
            # segment 1 stays zero (finite softmax, never scattered back).
            if tail_possible:
                T.copy(
                    Q[bz, hq0, bx * block_M : (bx + 1) * block_M, :],
                    q_l1_0,
                )
                if gb * 2 + 1 < G:
                    T.copy(
                        Q[bz, hq0 + 1, bx * block_M : (bx + 1) * block_M, :],
                        q_l1_1,
                    )
                else:
                    T.tile.fill(q_l1_1, 0.0)
            else:
                T.copy(
                    Q[bz, hq0, bx * block_M : (bx + 1) * block_M, :],
                    q_l1_0,
                )
                T.copy(
                    Q[bz, hq0 + 1, bx * block_M : (bx + 1) * block_M, :],
                    q_l1_1,
                )

            # ============= KV 迭代循环（K/V/mask 装载一次，两 head 段共享；skip 谓词共享）============
            for k in T.Pipelined(kv_iters, num_stages=pipe_stages):
                if k * block_N < (bx + 1) * block_M + skip_limit:
                    # ---- shared loads (once per iteration, both segments) ----
                    T.copy(K[bz, kg, k * block_N : (k + 1) * block_N, :], k_l1)
                    T.copy(V[bz, kg, k * block_N : (k + 1) * block_N, :], v_l1)

                    # ================= segment 0 =================
                    # GEMM1: Q0 @ K^T -> scores
                    T.gemm_v0(q_l1_0, k_l1, acc_s_l0c_0, transpose_B=True, init=True)

                    T.copy(acc_s_l0c_0, acc_s_ub_0)
                    T.tile.mul(acc_s_ub_0, acc_s_ub_0, sm_scale)
                    # additive mask applied BEFORE the running-max
                    # update (max-after-mask — the reference semantics): masked
                    # columns can no longer leak a dominant raw score into
                    # m_i (fixes the pre-existing vrange8/bf16-causal 0/0
                    # underflow edge).  Together with the shared
                    # temporaries this keeps mask_block's live range
                    # [copy, seg-1 add] genuinely overlapping every shared
                    # temp's live range, so memory planning cannot alias
                    # them (the cross-pipe race).
                    if use_mask:
                        T.copy(
                            Mask[bz, bx * block_M : (bx + 1) * block_M, k * block_N : (k + 1) * block_N],
                            mask_block,
                        )
                        T.tile.add(acc_s_ub_0, acc_s_ub_0, mask_block)

                    # Online softmax: update running max (row-wise)
                    T.copy(m_i_0, m_i_prev_0)
                    # clear=False is used here (g2 only): the clear=True
                    # form of this last-resort kernel hit an
                    # undefined-CUDART_INF compile error on some tilelang
                    # builds; the accumulate-style clear=False lowering
                    # compiles everywhere, and the m_init_floor =
                    # _BIG_NEG*1e-3 sentinel makes it exact.  Perf is
                    # irrelevant on this fallback-only path.
                    T.reduce_max(acc_s_ub_0, m_i_0, dim=-1, clear=False)
                    T.tile.max(m_i_0, m_i_0, m_i_prev_0)
                    # No clamp-at-0: with the
                    # _BIG_NEG=-1e30 mask and the m_init_floor sentinel the
                    # fully-masked-row cancellation is handled by the init
                    # floor (m_i = max(masked_value, floor) = floor ->
                    # exp(masked - floor) underflows), and clamping at 0
                    # would BREAK full-range rows whose max sits below 0.
                    T.tile.sub(m_i_prev_0, m_i_prev_0, m_i_0)
                    T.tile.exp(m_i_prev_0, m_i_prev_0)

                    T.tile.broadcast(row_bc_n_0, m_i_0, axis=1)
                    T.tile.sub(acc_s_ub_0, acc_s_ub_0, row_bc_n_0)

                    T.tile.exp(acc_s_ub_0, acc_s_ub_0)

                    T.reduce_sum(acc_s_ub_0, sumexp_i_ub, dim=-1)
                    T.tile.mul(sumexp_0, sumexp_0, m_i_prev_0)
                    T.tile.add(sumexp_0, sumexp_0, sumexp_i_ub)

                    T.copy(acc_s_ub_0, acc_s_half_0)
                    T.copy(acc_s_half_0, acc_s_l1_0)

                    # GEMM2: probs0 @ V -> output chunk
                    T.gemm_v0(acc_s_l1_0, v_l1, acc_o_l0c_0, init=True)
                    T.copy(acc_o_l0c_0, acc_o_ub_0)

                    T.tile.broadcast(row_bc_d_0, m_i_prev_0, axis=1)
                    T.tile.mul(acc_o_0, acc_o_0, row_bc_d_0)
                    T.tile.add(acc_o_0, acc_o_0, acc_o_ub_0)

                    # ================= segment 1 =================
                    # GEMM1: Q1 @ K^T -> scores (k_l1 REUSED — KV ÷2)
                    T.gemm_v0(q_l1_1, k_l1, acc_s_l0c_1, transpose_B=True, init=True)

                    T.copy(acc_s_l0c_1, acc_s_ub_1)
                    T.tile.mul(acc_s_ub_1, acc_s_ub_1, sm_scale)

                    # max-after-mask for segment 1 as well (mask_block
                    # already loaded by segment 0's chain above)
                    if use_mask:
                        T.tile.add(acc_s_ub_1, acc_s_ub_1, mask_block)

                    T.copy(m_i_1, m_i_prev_1)
                    # clear=False kept (see the segment-0 note above)
                    T.reduce_max(acc_s_ub_1, m_i_1, dim=-1, clear=False)
                    T.tile.max(m_i_1, m_i_1, m_i_prev_1)
                    # No clamp-at-0 for segment 1 either
                    # (same rationale as segment 0 above).
                    T.tile.sub(m_i_prev_1, m_i_prev_1, m_i_1)
                    T.tile.exp(m_i_prev_1, m_i_prev_1)

                    T.tile.broadcast(row_bc_n_1, m_i_1, axis=1)
                    T.tile.sub(acc_s_ub_1, acc_s_ub_1, row_bc_n_1)

                    T.tile.exp(acc_s_ub_1, acc_s_ub_1)

                    T.reduce_sum(acc_s_ub_1, sumexp_i_ub, dim=-1)
                    T.tile.mul(sumexp_1, sumexp_1, m_i_prev_1)
                    T.tile.add(sumexp_1, sumexp_1, sumexp_i_ub)

                    T.copy(acc_s_ub_1, acc_s_half_1)
                    T.copy(acc_s_half_1, acc_s_l1_1)

                    # GEMM2: probs1 @ V -> output chunk (v_l1 REUSED)
                    T.gemm_v0(acc_s_l1_1, v_l1, acc_o_l0c_1, init=True)
                    T.copy(acc_o_l0c_1, acc_o_ub_1)

                    T.tile.broadcast(row_bc_d_1, m_i_prev_1, axis=1)
                    T.tile.mul(acc_o_1, acc_o_1, row_bc_d_1)
                    T.tile.add(acc_o_1, acc_o_1, acc_o_ub_1)

                    # manual V->MTE2 iteration barrier (mask path
                    # only), placed at the END of the loop body: emitted
                    # after the last V-side op it lands on the AIV side of
                    # the CombineCV split, and in issue order its
                    # WaitFlag precedes ALL of iteration k+1's AIV MTE2 ops
                    # (scores readback + mask copy) while its SetFlag
                    # follows ALL of iteration k's V-side writes — exactly
                    # the barrier the per-head kernel gets for free from
                    # its single acc_s_ub name (visible WAW).  Without it,
                    # memory planning's cross-name UB aliasing
                    # (acc_s_ub_1 == acc_s_ub_0 @66560, acc_s_half_1 ==
                    # mask_block @99328) races iteration k's V writes
                    # against iteration k+1's MTE2 writes (6/10 first-call
                    # corruption).  Event id 15 is
                    # outside auto-sync's 0-7 allocation; iteration 0 needs
                    # no barrier (only the Fill inits precede it, disjoint
                    # addresses).  The Pipelined (non-mask) path is
                    # untouched.
                    if use_mask:
                        T.set_flag("v", "mte2", 15)
                        T.wait_flag("v", "mte2", 15)

            # ============= Final normalization + scatter (per segment) =============
            # fully-masked-row guard: causal-skip blocks
            # keep sumexp == 0 -> 0/0 = NaN (the reference outputs 0).  With the
            # m_i clamp the mask path also drives fully-masked rows to
            # sumexp == 0, so both paths converge to 0 / 1e-30 == 0 here.
            T.tile.add(sumexp_0, sumexp_0, 1e-30)
            T.tile.broadcast(row_bc_f, sumexp_0, axis=1)
            T.tile.div(acc_o_0, acc_o_0, row_bc_f)
            T.copy(acc_o_0, acc_o_half)
            T.copy(
                acc_o_half,
                Output[bz, hq0, bx * block_M : (bx + 1) * block_M, :],
            )

            T.tile.add(sumexp_1, sumexp_1, 1e-30)
            T.tile.broadcast(row_bc_f, sumexp_1, axis=1)
            T.tile.div(acc_o_1, acc_o_1, row_bc_f)
            T.copy(acc_o_1, acc_o_half)
            if tail_possible:
                if gb * 2 + 1 < G:
                    T.copy(
                        acc_o_half,
                        Output[bz, hq0 + 1, bx * block_M : (bx + 1) * block_M, :],
                    )
            else:
                T.copy(
                    acc_o_half,
                    Output[bz, hq0 + 1, bx * block_M : (bx + 1) * block_M, :],
                )

    return main


def _dense_tile_params(G, S, D):
    """Host-side tiling params for the dense M-packing kernel.

    Returns (G_inner, G_batches, M_tile):
      - G_inner: Q heads packed into one GEMM M dim (KV loaded once per G_inner)
      - M_tile:  padded GEMM M rows, snapped to a power of two (16/32/64).
        M=16/64 are the shapes already exercised by the per-head kernel
        (decode block_M=16 / prefill block_M=64); M=48-style tiles are
        unverified in this repo, so G_inner is stepped down until the padded
        tile is a power of two.
      - M_tile_max: 64 for D<=128, 32 for D>128 (L0C/UB budget).
    """
    M_tile_max = 64 if D <= 128 else 32
    G_inner = max(1, min(G, M_tile_max // S))
    while G_inner > 1:
        raw = ((G_inner * S + 15) // 16) * 16
        if raw <= M_tile_max and (raw & (raw - 1)) == 0:
            break
        G_inner -= 1
    raw = ((G_inner * S + 15) // 16) * 16
    M_tile = max(16, raw)
    G_batches = (G + G_inner - 1) // G_inner
    return G_inner, G_batches, M_tile


@tilelang.jit(
    out_idx=[4],
    pass_configs={
        tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
        tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
        tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
        tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
    },
)
def _gqa_kernel_dense(
    B,
    S,
    S_kv_padded,
    sm_scale,
    is_causal,
    N_q=8,
    N_kv=2,
    D=128,
    dtype="float16",
    skv_minus_s=0,
    use_mask=True,
    block_N=64,
    M_tile=16,
    G_inner=4,
):
    """Decode dense M-packing kernel — BSND direct supply.

    One instance per (bz, kv-group).  G_inner Q heads sharing one KV head are
    DENSELY packed into the GEMM M dim so each K/V block is loaded ONCE per
    G_inner heads.

    The kernel consumes/produces the caller's BSND layout
    ([B, S, N_q, D]) directly — the four host-side BSND<->BHSD transposes of
    the decode path are eliminated.  Access patterns:
      - K/V ingress: 4D strided reads K[bz, k*BN:(k+1)*BN, kg, :] — rows of
        D fp16 (256B at D=128) with row stride N_kv*D.  READ direction is
        the framework-safe one (WRITE direction stays forbidden).
        The measured MTE2 penalty on the largest-KV shapes is
        offset by gains elsewhere; net win everywhere because
        the removed transposes dominated the decode elapsed time.
      - M-packing row order is s-major: packed row r = s*G_inner + g_i.
        Because BSND places the head dim immediately before D, a fixed
        (bz, s) slice over G_inner consecutive heads is a CONTIGUOUS
        [G_inner, D] box — both the Q load and the output scatter are
        contiguous 2D copies (S copies of [G_inner, D] each; tail G_batch
        falls back to guarded per-row copies).
      - Mask rows are host-replicated s-major: packed row s*G_inner+g_i
        uses mask[s] (the mask depends only on (s, j), shared across all
        G_inner head segments); in-kernel mask load stays a single
        [M_tile, block_N] 2D copy.

    Structure intentionally mirrors the proven per-head kernel
    (_gqa_kernel_expert): AUTO_CV_COMBINE + AUTO_SYNC, NO explicit GM
    workspace.  Differences vs the per-head kernel:
      - threads=1 (NOT 2): the AscendVidReduction pass (auto sub-block
        split, active only at threads=2) is structurally incompatible with
        sub-tile UB->GM segment scatters — it halves the scatter loop trip
        count whenever the loop var indexes the first dim of a vid-reduced
        UB buffer (offset/extent == loop_var), and halves per-copy row
        extents, silently dropping ~half the output segments/rows (verified
        on generated source: `for (g_i_1 < G_inner/2)` + maskShapeM
        `(vid==0 ? ceil(S/2) : 0)`). threads=1 deactivates the pass; the
        C/V pipes run sequentially on one core with intra-core sync, which
        also removes the cross-core flag chain (~0.3-0.5us x 6-10/iter).
        Measured: dense t=1 kernel is 2.2-7.9x faster than the per-head
        t=2 kernel on decode shapes (instance-count reduction dominates).
      - grid = B * N_kv * G_batches (G_batches independent dim: MQA/MHA
        shapes keep their per-head parallelism, no core starvation)
      - Q/Output tensors use the REAL S (no host row padding: no ConcatD
        pad op, no output crop)
      - L0C single-buffered with init=True (dropping the [2, ...]
        side-alternation measured zero-benefit) so M_tile=64
        fits the 128KB L0C budget
      - causal block skip bound: k*block_N < S + skv_minus_s (the max
        visible column over the packed rows is S-1 + (S_kv - S) = S_kv-1)

    Pad rows [G_inner*S, M_tile) of q_l1 are zero-filled: their softmax rows
    are independent (row-wise max/sum/rescale) and are simply not scattered
    back to Output.
    """
    accum_dtype = "float"

    G = N_q // N_kv
    G_batches = (G + G_inner - 1) // G_inner
    kv_iters = S_kv_padded // block_N
    block_num = B * N_kv * G_batches
    # tail G_batch exists only when G % G_inner != 0; resolved at trace time
    # so G-divisible shapes emit only the bulk path
    tail_possible = (G % G_inner) != 0

    # KV loop pipelining depth — same rationale as the per-head
    # kernel (see _gqa_kernel_expert). Static budget: worst dense case
    # D256/M_tile=32 needs 609KB of the 704KB
    # L1+UB with the x(num_stages+1)=3 versioning of k_l1/v_l1/mask_block.
    # kv_iters < 2 OR ODD kv_iters degenerates to num_stages=0 (plain
    # serial): odd trip counts corrupt (phase grouping drops the tail
    # iteration, e.g. Skv=300 -> kv=3) — same defect class as the per-head
    # kernel. num_stages=3 is ~4-8% faster on M_tile=16 shapes but
    # SEGFAULTS the compiler on M_tile=64 with no diagnostic and
    # shares the divisibility defect (kv=4, 4%3!=0) — rejected.
    pipe_stages = 2 if (kv_iters >= 2 and kv_iters % 2 == 0) else 0

    # Compile-time skip threshold: non-causal -> never skip; causal ->
    # original S_kv - S boundary (blocks beyond it are pure padding, fully
    # masked by the host mask anyway).
    if is_causal:
        skip_limit = skv_minus_s
    else:
        skip_limit = 1 << 30

    # running-max init floor — pure "no prior max" sentinel below any
    # finite fp32 scaled score (see _gqa_kernel_expert).  Bit-identical for
    # normal ranges; fixes full-range fp16 rows with few visible columns.
    m_init_floor = sm_scale * _BIG_NEG

    @T.prim_func
    def main(
        Q: T.Tensor([B, S, N_q, D], dtype),  # type: ignore  BSND, real S (no pad)
        K: T.Tensor([B, S_kv_padded, N_kv, D], dtype),  # type: ignore  BSND layout
        V: T.Tensor([B, S_kv_padded, N_kv, D], dtype),  # type: ignore  BSND layout
        Mask: T.Tensor([B, M_tile, S_kv_padded], accum_dtype),  # type: ignore  rows replicated (s-major)
        Output: T.Tensor([B, S, N_q, D], dtype),  # type: ignore  BSND, real S (no pad)
    ):
        with T.Kernel(block_num, threads=1, is_npu=True) as (cid):
            gb = cid % G_batches
            kg = (cid // G_batches) % N_kv
            bz = cid // (G_batches * N_kv)

            # ============= L1 缓存（Q/K/V 输入）============
            q_l1 = T.alloc_shared([M_tile, D], dtype)
            k_l1 = T.alloc_shared([block_N, D], dtype)
            v_l1 = T.alloc_shared([block_N, D], dtype)

            # ============= L0C（GEMM 输出，单缓冲 init=True）============
            acc_s_l0c = T.alloc_L0C([M_tile, block_N], accum_dtype)
            acc_o_l0c = T.alloc_L0C([M_tile, D], accum_dtype)

            # ============= UB 计算缓冲 =============
            acc_s_ub = T.alloc_shared([M_tile, block_N], accum_dtype)
            acc_s_half = T.alloc_shared([M_tile, block_N], dtype)
            acc_s_l1 = T.alloc_shared([M_tile, block_N], dtype)

            m_i = T.alloc_shared([M_tile], accum_dtype)
            m_i_prev = T.alloc_shared([M_tile], accum_dtype)
            sumexp = T.alloc_shared([M_tile], accum_dtype)
            sumexp_i_ub = T.alloc_shared([M_tile], accum_dtype)

            acc_o = T.alloc_shared([M_tile, D], accum_dtype)
            acc_o_ub = T.alloc_shared([M_tile, D], accum_dtype)
            acc_o_half = T.alloc_shared([M_tile, D], dtype)

            row_bc_f = T.alloc_shared([M_tile, D], accum_dtype)
            row_bc_n = T.alloc_shared([M_tile, block_N], accum_dtype)

            mask_block = T.alloc_shared([M_tile, block_N], accum_dtype)

            # ============= 初始化 =============
            T.tile.fill(acc_o, 0.0)
            T.tile.fill(sumexp, 0.0)
            T.tile.fill(m_i, m_init_floor)

            # Dense Q packing, s-major: packed row s*G_inner + g_i holds
            # Q[bz, s, kg*G + gb*G_inner + g_i, :].  BSND makes a fixed-(bz,s)
            # slice over G_inner consecutive heads a CONTIGUOUS [G_inner, D]
            # box.  The guard for tail batches is on the LOCAL segment index
            # (gb*G_inner + g_i < G), NOT on hq < N_q: in a tail G_batch the
            # local segments beyond G would leak into the NEXT kv group's
            # heads (hq still < N_q for kg < N_kv-1), racing them with
            # wrong-KV attention output.
            T.tile.fill(q_l1, 0.0)
            hq0 = kg * G + gb * G_inner
            if tail_possible:
                if gb * G_inner + G_inner <= G:
                    for s in range(S):
                        T.copy(
                            Q[bz, s, hq0 : hq0 + G_inner, :],
                            q_l1[s * G_inner : (s + 1) * G_inner, :],
                        )
                else:
                    for g_i in range(G_inner):
                        g_local = gb * G_inner + g_i
                        if g_local < G:
                            hq = kg * G + g_local
                            for s in range(S):
                                T.copy(
                                    Q[bz, s, hq, :],
                                    q_l1[s * G_inner + g_i, :],
                                )
            else:
                for s in range(S):
                    T.copy(
                        Q[bz, s, hq0 : hq0 + G_inner, :],
                        q_l1[s * G_inner : (s + 1) * G_inner, :],
                    )

            # ============= KV 迭代循环（流水化；K/V strided read 每实例仅载一次，服务 G_inner heads）============
            # Whole-body causal-skip IfThenElse is the pipeline-supported loop
            # shape (predicate re-applied per stage block with the skewed loop
            # var — see _gqa_kernel_expert comment / inject_pipeline.cc).
            # K/V ingress is a 4D strided read on the BSND
            # tensor (row stride N_kv*D) — read direction bit-exact.
            for k in T.Pipelined(kv_iters, num_stages=pipe_stages):
                if k * block_N < S + skip_limit:
                    T.copy(K[bz, k * block_N : (k + 1) * block_N, kg, :], k_l1)
                    if use_mask:
                        T.copy(
                            Mask[bz, 0:M_tile, k * block_N : (k + 1) * block_N],
                            mask_block,
                        )

                    # GEMM1: packed Q @ K^T -> scores
                    T.gemm_v0(q_l1, k_l1, acc_s_l0c, transpose_B=True, init=True)

                    T.copy(acc_s_l0c, acc_s_ub)
                    T.tile.mul(acc_s_ub, acc_s_ub, sm_scale)

                    # Max-after-mask order — mask BEFORE
                    # the running max (see _gqa_kernel_expert; a masked
                    # column's huge raw score must not lift the max).
                    if use_mask:
                        T.tile.add(acc_s_ub, acc_s_ub, mask_block)

                    # Online softmax: update running max (row-wise over packed rows)
                    T.copy(m_i, m_i_prev)
                    T.reduce_max(acc_s_ub, m_i, dim=-1)
                    T.tile.max(m_i, m_i, m_i_prev)
                    T.tile.sub(m_i_prev, m_i_prev, m_i)
                    T.tile.exp(m_i_prev, m_i_prev)

                    T.tile.broadcast(row_bc_n, m_i, axis=1)
                    T.tile.sub(acc_s_ub, acc_s_ub, row_bc_n)

                    T.tile.exp(acc_s_ub, acc_s_ub)

                    T.reduce_sum(acc_s_ub, sumexp_i_ub, dim=-1)
                    T.tile.mul(sumexp, sumexp, m_i_prev)
                    T.tile.add(sumexp, sumexp, sumexp_i_ub)

                    T.copy(acc_s_ub, acc_s_half)
                    T.copy(acc_s_half, acc_s_l1)
                    T.copy(V[bz, k * block_N : (k + 1) * block_N, kg, :], v_l1)

                    # GEMM2: scores @ V -> output chunk
                    T.gemm_v0(acc_s_l1, v_l1, acc_o_l0c, init=True)
                    T.copy(acc_o_l0c, acc_o_ub)

                    T.tile.broadcast(row_bc_f, m_i_prev, axis=1)
                    T.tile.mul(acc_o, acc_o, row_bc_f)
                    T.tile.add(acc_o, acc_o, acc_o_ub)

            # ============= Final normalization =============
            # fully-masked-row guard (see _gqa_kernel_expert).
            T.tile.add(sumexp, sumexp, 1e-30)
            T.tile.broadcast(row_bc_f, sumexp, axis=1)
            T.tile.div(acc_o, acc_o, row_bc_f)

            # Output scatter, s-major contiguous boxes (strided WRITE stays
            # forbidden — s-major packing makes the destination rows for a
            # fixed (bz, s) contiguous, so only contiguous copies are used).
            # Tail G_batch: guarded per-row copies (same local-segment guard
            # as the Q load — segments beyond G belong to the next kv group).
            T.copy(acc_o, acc_o_half)
            if tail_possible:
                if gb * G_inner + G_inner <= G:
                    for s in range(S):
                        T.copy(
                            acc_o_half[s * G_inner : (s + 1) * G_inner, :],
                            Output[bz, s, hq0 : hq0 + G_inner, :],
                        )
                else:
                    for g_i in range(G_inner):
                        g_local = gb * G_inner + g_i
                        if g_local < G:
                            hq = kg * G + g_local
                            for s in range(S):
                                T.copy(
                                    acc_o_half[s * G_inner + g_i, :],
                                    Output[bz, s, hq, :],
                                )
            else:
                for s in range(S):
                    T.copy(
                        acc_o_half[s * G_inner : (s + 1) * G_inner, :],
                        Output[bz, s, hq0 : hq0 + G_inner, :],
                    )

    return main


@tilelang.jit(out_idx=[4], workspace_idx=[5, 6, 7], pass_configs=P28_EXPERT_PASS_CONFIGS)
def _gqa_kernel_dense_expert(
    B,
    S,
    S_kv_padded,
    sm_scale,
    is_causal,
    N_q=8,
    N_kv=2,
    D=128,
    dtype="float16",
    use_mask=True,
    block_N=128,
    M_tile=16,
    G_inner=4,
    num_stages=8,
    cross_interval=1,
):
    """Expert manual-flag deep-pipeline dense decode kernel.

    Dispatch guard (wrapper): 1<=S<=4, fp16/bf16, D==128, D==block_N,
    kv_iters>=4 and even, G % G_inner == 0 (no tail G_batch — the scatter
    plan is factory-level static), block_num >= NUM_CORES (all cores busy).
    One instance per (bz, kg, gb) — same task space as the AUTO dense kernel —
    statically split over NUM_CORES cores (the guard requires instances >=
    cores; even split via q/r remainder assignment).

    Dataflow (per instance): Q packed s-major into q_l1 [M_tile, D] (pad rows
    zero-filled); KV loop batched by num_stages: GEMM1 batch (K load + mma ->
    ws_s) then GEMM2 batch (V load + ws_p -> p_l1 + mma -> ws_o); V-side runs
    phase-1 softmax batch + phase-2 O-accumulation batch concurrently with the
    C-side of the following batch (cross-core semaphores, cross_interval-grain).
    """
    assert block_N == D, "expert dense kernel requires D == block_N (shared l0a/l0b/l0c slots)"
    assert num_stages % 2 == 0, "num_stages must be even (neg_sm slot parity across batches)"
    accum_dtype = "float"

    # neg_sm init cap.  Since the max-after-mask rework the additive mask is
    # applied to the raw scores BEFORE the running max, so the cap is the
    # max-after-mask fully-masked-row floor at mask/2 (identical rationale as
    # _gqa_kernel_prefill_expert): a fully-masked row's neg_sm ~ +scale*1e30
    # is clamped to scale*5e29, keeping P = exp(cap + scale*(s-1e30)) at 0
    # instead of canceling the mask in the shift.  Normal rows keep the
    # exact max (real neg_sm <= ~5e10 << cap).
    neg_sm_cap = -sm_scale * _BIG_NEG * 0.5

    G = N_q // N_kv
    G_batches = (G + G_inner - 1) // G_inner
    kv_iters = S_kv_padded // block_N
    assert kv_iters >= 4 and kv_iters % 2 == 0, "expert path requires even kv_iters >= 4"
    assert (G % G_inner) == 0, "expert path requires G % G_inner == 0 (no tail G_batch — the scatter plan is factory-level static)"
    block_num = B * N_kv * G_batches
    assert block_num >= P28_NUM_CORES, "expert path requires block_num >= NUM_CORES (all cores busy)"
    tail_possible = (G % G_inner) != 0
    num_outer = T.ceildiv(kv_iters, num_stages)
    half = M_tile // 2

    NUM_CORES = P28_NUM_CORES
    q_tasks = block_num // NUM_CORES
    r_tasks = block_num % NUM_CORES

    # Cross-core semaphores (C <-> V)
    SEM_WS1_C2V = 0  # ws_s ready:      C(FIX) -> V
    SEM_WS1_V2C = 1  # ws_s consumed:   V(MTE2) -> C
    SEM_WS2_V2C = 2  # ws_p ready:      V(MTE3) -> C
    SEM_WS2_C2V = 3  # ws_p consumed:   C(MTE2) -> V
    SEM_WS3_C2V = 4  # ws_o ready:      C(FIX) -> V
    SEM_WS3_V2C = 5  # ws_o consumed:   V(MTE2) -> C

    # Local directed event ids (per pipe pair)
    SIG_K_L1 = 0  # MTE2 <-> MTE1 (k_l1[side] ownership, slots 0/1)
    SIG_V_L1 = 2  # MTE2 <-> MTE1 (v_l1[side] ownership, slots 0/1)
    SIG_P_L1 = 4  # MTE2 <-> MTE1 (p_l1 ownership)
    SIG_Q_L1 = 5  # MTE2 <-> MTE1 (q_l1 ownership)
    SIG_QFILL = 6  # V -> MTE2 (q_l1 fill ordering)
    SIG_L0AB = 0  # MTE1 <-> M (l0a/l0b slots 0/1)
    SIG_L0C = 0  # M <-> FIX (l0c slots 0/1)
    SIG_IO_UB = 0  # MTE2 <-> V (io_buf slots 0/1)
    SIG_MASK = 2  # MTE2 <-> V (mask_v ready/free)
    SIG_S_HALF = 0  # V <-> MTE3 (acc_s_half ownership)

    def task_range(cid_val):
        start = cid_val * q_tasks + T.if_then_else(cid_val < r_tasks, cid_val, r_tasks)
        count = q_tasks + T.if_then_else(cid_val < r_tasks, 1, 0)
        return start, count

    # ---- factory-level scatter plan (per vid): the (s, v) row intersections
    # are computed with plain Python ints; only `vid` stays runtime.
    # `for x in range()` does NOT unroll at trace time (TVMScript parses it as
    # a TIR loop), so each vid gets explicitly-duplicated STATIC first/last
    # segment slices plus fixed-slot static copies for the middle segments —
    # at most two middles, since len(segs) <= S <= 4 (assert below).  The
    # middle scatters MUST stay unconditional: runtime-conditional MTE3
    # scatters inside the vid branch break the SIG_S_HALF handshake and
    # HANG the launch; see the vid==0 scatter note.
    scatter_plan = {}
    for v in range(2):
        segs = []
        for s in range(S):
            lo = s * G_inner if s * G_inner > v * half else v * half
            hi = (s + 1) * G_inner if (s + 1) * G_inner < (v + 1) * half else (v + 1) * half
            if lo < hi:
                # (s, local_lo, local_hi, g_lo, g_hi): local offsets index
                # acc_s_half (vid-relative rows); g offsets index the head
                # segment within Output[bz, s, hq0 + g_lo : hq0 + g_hi, :].
                segs.append((s, lo - v * half, hi - v * half, lo - s * G_inner, hi - s * G_inner))
        scatter_plan[v] = segs
    # Two static middle slots cover len(segs) <= 4; guaranteed by the decode
    # dispatch guard (1 <= S <= 4, and len(segs) <= S).  Fail at compile time
    # (-> dense_auto fallback) instead of silently dropping rows if the guard
    # is ever relaxed.
    assert S <= 4, "static middle-segment scatter slots require S <= 4 (decode guard)"

    @T.prim_func
    def main(
        Q: T.Tensor([B, S, N_q, D], dtype),  # type: ignore  BSND, real S
        K: T.Tensor([B, S_kv_padded, N_kv, D], dtype),  # type: ignore  BSND
        V: T.Tensor([B, S_kv_padded, N_kv, D], dtype),  # type: ignore  BSND
        Mask: T.Tensor([B, M_tile, S_kv_padded], accum_dtype),  # type: ignore  rows replicated s-major
        Output: T.Tensor([B, S, N_q, D], dtype),  # type: ignore  BSND, real S
        ws_s: T.Tensor([NUM_CORES, num_stages, M_tile, block_N], accum_dtype),  # type: ignore
        ws_p: T.Tensor([NUM_CORES, num_stages, M_tile, block_N], dtype),  # type: ignore
        ws_o: T.Tensor([NUM_CORES, num_stages, M_tile, D], accum_dtype),  # type: ignore
    ):
        with T.Kernel(NUM_CORES, is_npu=True) as (cid, vid):
            # ===== C-side buffers =====
            q_l1 = T.alloc_L1([M_tile, D], dtype)
            # k/v double buffering: two separate 2D buffers per operand,
            # selected by a runtime `if` (whole-buffer copies only — sliced
            # copies of layout-annotated L1 buffers lower incorrectly)
            k_l1_0 = T.alloc_L1([block_N, D], dtype)
            k_l1_1 = T.alloc_L1([block_N, D], dtype)
            v_l1_0 = T.alloc_L1([block_N, D], dtype)
            v_l1_1 = T.alloc_L1([block_N, D], dtype)
            p_l1 = T.alloc_L1([M_tile, block_N], dtype)

            T.annotate_layout(
                {
                    q_l1: make_zn_layout(q_l1),
                    k_l1_0: make_nz_layout(k_l1_0),
                    k_l1_1: make_nz_layout(k_l1_1),
                    p_l1: make_zn_layout(p_l1),
                    v_l1_0: make_zn_layout(v_l1_0),
                    v_l1_1: make_zn_layout(v_l1_1),
                }
            )

            # shared GEMM1/GEMM2 slots (D == block_N makes the shapes coincide)
            l0a = T.alloc_L0A([2, M_tile, D], dtype)
            l0b = T.alloc_L0B([2, D, block_N], dtype)
            l0c = T.alloc_L0C([2, M_tile, block_N], accum_dtype)

            # ===== V-side buffers (per vid, half rows) =====
            io_buf = T.alloc_ub([2, half, block_N], accum_dtype)
            work_ub = T.alloc_ub([half, block_N], accum_dtype)
            buf_2d = T.alloc_ub([half, block_N], accum_dtype)
            acc_s_half = T.alloc_ub([half, block_N], dtype)
            acc_o = T.alloc_ub([half, D], accum_dtype)
            mask_v = T.alloc_ub([half, block_N], accum_dtype)
            neg_sm = T.alloc_ub([2, half, 1], accum_dtype)
            sumexp = T.alloc_ub([half, 1], accum_dtype)
            sumexp_is = T.alloc_ub([num_stages, half, 1], accum_dtype)
            r_factors = T.alloc_ub([num_stages, half, 1], accum_dtype)

            my_start, my_count = task_range(cid)

            with T.Scope("C"):
                # init: pretend consumers already released everything
                T.set_cross_flag("MTE2", SEM_WS2_C2V)
                T.set_flag("MTE1", "MTE2", SIG_K_L1)
                T.set_flag("MTE1", "MTE2", SIG_K_L1 + 1)
                T.set_flag("MTE1", "MTE2", SIG_P_L1)
                T.set_flag("MTE1", "MTE2", SIG_V_L1)
                T.set_flag("MTE1", "MTE2", SIG_V_L1 + 1)
                T.set_flag("MTE1", "MTE2", SIG_Q_L1)
                T.set_flag("M", "MTE1", SIG_L0AB)
                T.set_flag("M", "MTE1", SIG_L0AB + 1)
                T.set_flag("FIX", "M", SIG_L0C)
                T.set_flag("FIX", "M", SIG_L0C + 1)

                for t in T.serial(my_count):
                    task_id = my_start + t
                    gb = task_id % G_batches
                    kg = (task_id // G_batches) % N_kv
                    bz = task_id // (G_batches * N_kv)
                    hq0 = kg * G + gb * G_inner

                    # --- Q load: pad-row fill + s-major dense packing ---
                    T.wait_flag("MTE1", "MTE2", SIG_Q_L1)
                    T.tile.fill(q_l1, 0.0)
                    T.set_flag("V", "MTE2", SIG_QFILL)
                    T.wait_flag("V", "MTE2", SIG_QFILL)
                    if tail_possible:
                        if gb * G_inner + G_inner <= G:
                            for s in range(S):
                                T.copy(
                                    Q[bz, s, hq0 : hq0 + G_inner, :],
                                    q_l1[s * G_inner : (s + 1) * G_inner, :],
                                )
                        else:
                            for g_i in range(G_inner):
                                g_local = gb * G_inner + g_i
                                if g_local < G:
                                    hq = kg * G + g_local
                                    for s in range(S):
                                        T.copy(
                                            Q[bz, s, hq, :],
                                            q_l1[s * G_inner + g_i, :],
                                        )
                    else:
                        for s in range(S):
                            T.copy(
                                Q[bz, s, hq0 : hq0 + G_inner, :],
                                q_l1[s * G_inner : (s + 1) * G_inner, :],
                            )
                    T.set_flag("MTE2", "MTE1", SIG_Q_L1)
                    T.wait_flag("MTE2", "MTE1", SIG_Q_L1)

                    for k in T.serial(num_outer):
                        _remaining = kv_iters - k * num_stages
                        batch_iters = T.if_then_else(_remaining < num_stages, _remaining, num_stages)

                        # --- GEMM1 batch: K loads + mma(Q, K^T) -> ws_s ---
                        T.wait_cross_flag(SEM_WS1_V2C)
                        for i in T.serial(batch_iters):
                            side = i % 2
                            idx = k * num_stages + i

                            T.wait_flag("MTE1", "MTE2", SIG_K_L1 + side)
                            if i % 2 == 0:
                                T.copy(K[bz, idx * block_N : (idx + 1) * block_N, kg, :], k_l1_0)
                            else:
                                T.copy(K[bz, idx * block_N : (idx + 1) * block_N, kg, :], k_l1_1)
                            T.set_flag("MTE2", "MTE1", SIG_K_L1 + side)

                            T.wait_flag("M", "MTE1", SIG_L0AB + side)
                            if i < 2:
                                T.copy(q_l1, l0a[side, :, :])

                            T.wait_flag("MTE2", "MTE1", SIG_K_L1 + side)
                            if i % 2 == 0:
                                T.copy(k_l1_0, l0b[0, :, :], transpose=True)
                            else:
                                T.copy(k_l1_1, l0b[1, :, :], transpose=True)
                            T.set_flag("MTE1", "MTE2", SIG_K_L1 + side)
                            T.set_flag("MTE1", "M", SIG_L0AB + side)

                            T.wait_flag("MTE1", "M", SIG_L0AB + side)
                            T.wait_flag("FIX", "M", SIG_L0C + side)
                            T.mma(l0a[side, :, :], l0b[side, :, :], l0c[side, :, :], init=True)
                            T.set_flag("M", "MTE1", SIG_L0AB + side)
                            T.set_flag("M", "FIX", SIG_L0C + side)

                            T.wait_flag("M", "FIX", SIG_L0C + side)
                            T.copy(l0c[side, :, :], ws_s[cid, i, :, :])
                            T.set_flag("FIX", "M", SIG_L0C + side)
                            if (i + 1) % cross_interval == 0 or i == batch_iters - 1:
                                T.set_cross_flag("FIX", SEM_WS1_C2V)

                        # --- GEMM2 batch: V loads + ws_p -> p_l1 + mma(P, V) -> ws_o ---
                        T.wait_cross_flag(SEM_WS3_V2C)
                        for i in T.serial(batch_iters):
                            side = i % 2
                            idx = k * num_stages + i

                            T.wait_flag("MTE1", "MTE2", SIG_V_L1 + side)
                            if i % 2 == 0:
                                T.copy(V[bz, idx * block_N : (idx + 1) * block_N, kg, :], v_l1_0)
                            else:
                                T.copy(V[bz, idx * block_N : (idx + 1) * block_N, kg, :], v_l1_1)
                            T.set_flag("MTE2", "MTE1", SIG_V_L1 + side)

                            T.wait_flag("MTE1", "MTE2", SIG_P_L1)
                            if i % cross_interval == 0:
                                T.wait_cross_flag(SEM_WS2_V2C)
                            T.copy(ws_p[cid, i, :, :], p_l1)
                            T.set_flag("MTE2", "MTE1", SIG_P_L1)

                            T.wait_flag("MTE2", "MTE1", SIG_V_L1 + side)
                            T.wait_flag("M", "MTE1", SIG_L0AB + side)
                            if i % 2 == 0:
                                T.copy(v_l1_0, l0b[0, :, :])
                            else:
                                T.copy(v_l1_1, l0b[1, :, :])
                            T.set_flag("MTE1", "MTE2", SIG_V_L1 + side)

                            T.wait_flag("MTE2", "MTE1", SIG_P_L1)
                            T.copy(p_l1, l0a[side, :, :])
                            T.set_flag("MTE1", "MTE2", SIG_P_L1)
                            T.set_flag("MTE1", "M", SIG_L0AB + side)

                            T.wait_flag("MTE1", "M", SIG_L0AB + side)
                            T.wait_flag("FIX", "M", SIG_L0C + side)
                            T.mma(l0a[side, :, :], l0b[side, :, :], l0c[side, :, :], init=True)
                            T.set_flag("M", "MTE1", SIG_L0AB + side)
                            T.set_flag("M", "FIX", SIG_L0C + side)

                            T.wait_flag("M", "FIX", SIG_L0C + side)
                            T.copy(l0c[side, :, :], ws_o[cid, i, :, :])
                            T.set_flag("FIX", "M", SIG_L0C + side)
                            if (i + 1) % cross_interval == 0 or i == batch_iters - 1:
                                T.set_cross_flag("FIX", SEM_WS3_C2V)

                        T.set_cross_flag("MTE2", SEM_WS2_C2V)

                    # MTE1 no longer reads q_l1; return it before the next task reloads Q
                    T.set_flag("MTE1", "MTE2", SIG_Q_L1)

                # destroy: consume outstanding init-direction flags
                T.wait_flag("MTE1", "MTE2", SIG_K_L1)
                T.wait_flag("MTE1", "MTE2", SIG_K_L1 + 1)
                T.wait_flag("MTE1", "MTE2", SIG_P_L1)
                T.wait_flag("MTE1", "MTE2", SIG_V_L1)
                T.wait_flag("MTE1", "MTE2", SIG_V_L1 + 1)
                T.wait_flag("MTE1", "MTE2", SIG_Q_L1)
                T.wait_flag("M", "MTE1", SIG_L0AB)
                T.wait_flag("M", "MTE1", SIG_L0AB + 1)
                T.wait_flag("FIX", "M", SIG_L0C)
                T.wait_flag("FIX", "M", SIG_L0C + 1)

            with T.Scope("V"):
                # init: pretend producers already released everything
                T.set_cross_flag("MTE2", SEM_WS1_V2C)
                T.set_cross_flag("MTE2", SEM_WS3_V2C)
                T.set_flag("V", "MTE2", SIG_IO_UB)
                T.set_flag("V", "MTE2", SIG_IO_UB + 1)
                T.set_flag("MTE3", "V", SIG_S_HALF)

                for t in T.serial(my_count):
                    task_id = my_start + t
                    gb = task_id % G_batches
                    kg = (task_id // G_batches) % N_kv
                    bz = task_id // (G_batches * N_kv)
                    hq0 = kg * G + gb * G_inner

                    T.tile.fill(acc_o, 0.0)
                    T.tile.fill(sumexp, 0.0)
                    T.tile.fill(neg_sm, neg_sm_cap)

                    for k in T.serial(num_outer):
                        _remaining = kv_iters - k * num_stages
                        batch_iters = T.if_then_else(_remaining < num_stages, _remaining, num_stages)

                        # --- phase 1: softmax batch (read ws_s, write ws_p) ---
                        T.wait_cross_flag(SEM_WS2_C2V)
                        for i in T.serial(batch_iters):
                            cur = i % 2
                            prv = 1 - cur
                            idx = k * num_stages + i
                            io_side = i % 2

                            T.wait_flag("V", "MTE2", SIG_IO_UB + io_side)
                            if i % cross_interval == 0:
                                T.wait_cross_flag(SEM_WS1_C2V)
                            T.copy(ws_s[cid, i, vid * half : vid * half + half, :], io_buf[io_side, :, :])
                            T.set_flag("MTE2", "V", SIG_IO_UB + io_side)

                            if use_mask:
                                # 2-flag mask sync (varlen pattern): MTE2 loads
                                # mask_v only after the previous iteration's add
                                # consumed it (set+wait = V->MTE2 ordering point)
                                T.set_flag("V", "MTE2", SIG_MASK)
                                T.wait_flag("V", "MTE2", SIG_MASK)
                                T.copy(
                                    Mask[bz, vid * half : vid * half + half, idx * block_N : (idx + 1) * block_N],
                                    mask_v,
                                )
                                T.set_flag("MTE2", "V", SIG_MASK)

                            T.wait_flag("MTE2", "V", SIG_IO_UB + io_side)
                            T.copy(io_buf[io_side, :, :], work_ub)
                            T.set_flag("V", "MTE2", SIG_IO_UB + io_side)

                            # online softmax (batched 2-phase form)
                            # Max-after-mask order — the
                            # additive mask is applied to the raw scores
                            # BEFORE the running max (a masked column's huge
                            # raw score must not lift the max and underflow
                            # every visible weight; see _gqa_kernel_expert).
                            if use_mask:
                                T.wait_flag("MTE2", "V", SIG_MASK)
                                T.tile.add(work_ub, work_ub, mask_v)
                            # clear=True is used here: these
                            # kernels compile fine with clear=True,
                            # and the clear=False lowering inserts
                            # scalar merge loops (see the _gqa_kernel_expert
                            # clear=True note).
                            T.reduce_max(work_ub, neg_sm[cur, :, :], dim=-1)
                            T.tile.mul(neg_sm[cur, :, :], neg_sm[cur, :, :], -sm_scale)
                            T.tile.min(neg_sm[cur, :, :], neg_sm[cur, :, :], neg_sm[prv, :, :])

                            T.tile.broadcast(buf_2d, neg_sm[cur, :, :])
                            T.tile.axpy(buf_2d, work_ub, sm_scale)
                            T.tile.exp(work_ub, buf_2d)

                            T.reduce_sum(work_ub, sumexp_is[i, :, :], dim=-1)
                            T.tile.sub(r_factors[i, :, :], neg_sm[cur, :, :], neg_sm[prv, :, :])

                            # write probs -> ws_p (via acc_s_half)
                            T.wait_flag("MTE3", "V", SIG_S_HALF)
                            T.copy(work_ub, acc_s_half)
                            T.set_flag("V", "MTE3", SIG_S_HALF)

                            T.wait_flag("V", "MTE3", SIG_S_HALF)
                            T.copy(acc_s_half, ws_p[cid, i, vid * half : vid * half + half, :])
                            T.set_flag("MTE3", "V", SIG_S_HALF)
                            if (i + 1) % cross_interval == 0 or i == batch_iters - 1:
                                T.set_cross_flag("MTE3", SEM_WS2_V2C)

                        T.set_cross_flag("MTE2", SEM_WS1_V2C)

                        # --- phase 2: O accumulation batch (read ws_o) ---
                        for i in T.serial(batch_iters):
                            T.tile.exp(r_factors[i, :, :], r_factors[i, :, :])
                            T.tile.mul(sumexp, sumexp, r_factors[i, :, :])
                            T.tile.add(sumexp, sumexp, sumexp_is[i, :, :])
                            T.tile.broadcast(buf_2d, r_factors[i, :, :])
                            T.tile.mul(acc_o, acc_o, buf_2d)

                            io_side = i % 2
                            T.wait_flag("V", "MTE2", SIG_IO_UB + io_side)
                            if i % cross_interval == 0:
                                T.wait_cross_flag(SEM_WS3_C2V)
                            T.copy(ws_o[cid, i, vid * half : vid * half + half, :], io_buf[io_side, :, :])
                            T.set_flag("MTE2", "V", SIG_IO_UB + io_side)

                            T.wait_flag("MTE2", "V", SIG_IO_UB + io_side)
                            T.copy(io_buf[io_side, :, :], work_ub)
                            T.set_flag("V", "MTE2", SIG_IO_UB + io_side)

                            T.tile.add(acc_o, acc_o, work_ub)

                        T.set_cross_flag("MTE2", SEM_WS3_V2C)

                    # --- final normalize + s-major scatter (per vid) ---
                    # fully-masked-row guard (see
                    # _gqa_kernel_expert).
                    T.tile.add(sumexp, sumexp, 1e-30)
                    T.tile.broadcast(buf_2d, sumexp)
                    T.tile.div(acc_o, acc_o, buf_2d)
                    T.wait_flag("MTE3", "V", SIG_S_HALF)
                    T.copy(acc_o, acc_s_half)
                    T.set_flag("V", "MTE3", SIG_S_HALF)
                    T.wait_flag("V", "MTE3", SIG_S_HALF)
                    # NOTE: helper calls inside a runtime-if branch drop their
                    # emitted statements (TVM Script branch scoping) — the two
                    # vid blocks are inlined verbatim.
                    if vid == 0:
                        segs = scatter_plan[0]
                        if len(segs) >= 1:
                            s_f, l_f, h_f, g_f, ge_f = segs[0]
                            T.copy(acc_s_half[l_f:h_f, :], Output[bz, s_f, hq0 + g_f : hq0 + ge_f, :])
                        if len(segs) >= 2:
                            s_l, l_l, h_l, g_l, ge_l = segs[-1]
                            T.copy(acc_s_half[l_l:h_l, :], Output[bz, s_l, hq0 + g_l : hq0 + ge_l, :])
                            if len(segs) > 2:
                                # Middle segments segs[1:-1] — STATIC fixed-slot
                                # unroll.  These `if`s
                                # have Python-bool conditions: the TVMScript
                                # parser folds them at trace time (tir parser
                                # visit_if), so each taken slot emits ONE
                                # unconditional static-slice MTE3 scatter,
                                # structurally identical to the segs[0]/segs[-1]
                                # copies above — zero new runtime control flow,
                                # SIG_S_HALF handshake untouched.  Two slots are
                                # exhaustive: len(segs) <= S <= 4 (decode guard
                                # + factory assert) leaves at most two middles.
                                # The scatters MUST stay unconditional:
                                # `for s in range(S)` inside the prim_func is
                                # a TIR loop (`s` is a runtime Var, never
                                # unrolled at trace time), so an s-loop would
                                # make the MTE3 scatters runtime-conditional
                                # inside the vid branch, break the SIG_S_HALF
                                # handshake and HANG the launch.  Each slot
                                # also re-unpacks its own seg tuple locally:
                                # reading `s_f`/`h_f` from another branch
                                # frame fails with `Undefined variable`
                                # (TVMScript scopes branch-body assignments
                                # to the if's var frame).
                                s_m1, l_m1, h_m1, g_m1, ge_m1 = segs[1]
                                T.copy(acc_s_half[l_m1:h_m1, :], Output[bz, s_m1, hq0 + g_m1 : hq0 + ge_m1, :])
                                if len(segs) > 3:
                                    s_m2, l_m2, h_m2, g_m2, ge_m2 = segs[2]
                                    T.copy(acc_s_half[l_m2:h_m2, :], Output[bz, s_m2, hq0 + g_m2 : hq0 + ge_m2, :])
                    if vid == 1:
                        segs = scatter_plan[1]
                        if len(segs) >= 1:
                            s_f, l_f, h_f, g_f, ge_f = segs[0]
                            T.copy(acc_s_half[l_f:h_f, :], Output[bz, s_f, hq0 + g_f : hq0 + ge_f, :])
                        if len(segs) >= 2:
                            s_l, l_l, h_l, g_l, ge_l = segs[-1]
                            T.copy(acc_s_half[l_l:h_l, :], Output[bz, s_l, hq0 + g_l : hq0 + ge_l, :])
                            if len(segs) > 2:
                                # Middle segments segs[1:-1] — static fixed-slot
                                # unroll, identical to the vid==0 block above
                                # (see the note there for why the scatters must
                                # stay unconditional).
                                s_m1, l_m1, h_m1, g_m1, ge_m1 = segs[1]
                                T.copy(acc_s_half[l_m1:h_m1, :], Output[bz, s_m1, hq0 + g_m1 : hq0 + ge_m1, :])
                                if len(segs) > 3:
                                    s_m2, l_m2, h_m2, g_m2, ge_m2 = segs[2]
                                    T.copy(acc_s_half[l_m2:h_m2, :], Output[bz, s_m2, hq0 + g_m2 : hq0 + ge_m2, :])
                    T.set_flag("MTE3", "V", SIG_S_HALF)

                # destroy: consume outstanding init-direction flags
                T.wait_flag("V", "MTE2", SIG_IO_UB)
                T.wait_flag("V", "MTE2", SIG_IO_UB + 1)
                T.wait_flag("MTE3", "V", SIG_S_HALF)

    return main


@tilelang.jit(out_idx=[4], workspace_idx=[5, 6, 7], pass_configs=P28_EXPERT_PASS_CONFIGS)
def _gqa_kernel_dense_expert_d256(
    B,
    S,
    S_kv_padded,
    sm_scale,
    is_causal,
    N_q=8,
    N_kv=2,
    D=256,
    dtype="float16",
    use_mask=True,
    block_N=64,
    M_tile=16,
    G_inner=4,
    num_stages=16,
    cross_interval=4,
):
    """D256 dense decode expert — BN=64, per-phase single-slot l0b.

    Same task space / flag discipline / 2-phase batched online softmax as
    the D128 expert (_gqa_kernel_dense_expert).  The D128 template asserts
    D == block_N because its single l0b [2, D, BN] is shared between
    GEMM1 (K^T as [D, BN]) and GEMM2 (V as [BN, D]) — the shapes only
    coincide when D == BN.  At D=256 no BN fits both whole-D phase shapes
    into the 64KB L0B budget, so this variant keeps ONE whole-D slot per
    phase (l0b_k [256,64] + l0b_v [64,256] = 64KB exact, single-slot with
    per-iter flag cycles) and each phase runs a single whole-D mma:
      - K/V GM boxes stay [64, 256] — the 512B-row segment geometry runs
        at high GM bandwidth
      - l0a_q [2, M, D] / l0c_s [2, M, BN] / l0c_o [2, M, D] keep the
        D128 double-buffered slot discipline; l0a_p is single-slot
      - kv_iters doubles (S_kv_padded / 64; S_kv_padded is a 128 multiple
        so kv_iters is always even)
    V-side: phase-1 works on [half, BN] tiles (io_buf/work/buf_2d split
    from the phase-2 [half, D] tiles — D128 shared them via D == BN);
    acc_o/acc_o_half are [half, D].
    Guard (wrapper): 1<=S<=4, D==256, fp16/bf16, kv_iters(BN=64)>=4,
    G % G_inner == 0, block_num >= NUM_CORES.
    """
    assert D == 256 and block_N == 64, "dense_expert_d256 requires D=256, block_N=64"
    NUM_CORES = P28_NUM_CORES
    assert num_stages % 2 == 0, "num_stages must be even (neg_sm slot parity)"
    accum_dtype = "float"

    # neg_sm init cap — max-after-mask fully-masked-row floor at mask/2
    # (see _gqa_kernel_dense_expert).
    neg_sm_cap = -sm_scale * _BIG_NEG * 0.5

    G = N_q // N_kv
    G_batches = (G + G_inner - 1) // G_inner
    kv_iters = S_kv_padded // block_N
    assert kv_iters >= 2 and kv_iters % 2 == 0, "even kv_iters >= 2 required"
    assert (G % G_inner) == 0, "no tail G_batch (factory-static scatter plan)"
    block_num = B * N_kv * G_batches
    assert block_num >= NUM_CORES, "all cores busy"
    num_outer = T.ceildiv(kv_iters, num_stages)
    half = M_tile // 2

    q_tasks = block_num // NUM_CORES
    r_tasks = block_num % NUM_CORES

    # Cross-core semaphores (C <-> V)
    SEM_WS1_C2V = 0  # ws_s ready:      C(FIX) -> V
    SEM_WS1_V2C = 1  # ws_s consumed:   V(MTE2) -> C
    SEM_WS2_V2C = 2  # ws_p ready:      V(MTE3) -> C
    SEM_WS2_C2V = 3  # ws_p consumed:   C(MTE2) -> V
    SEM_WS3_C2V = 4  # ws_o ready:      C(FIX) -> V
    SEM_WS3_V2C = 5  # ws_o consumed:   V(MTE2) -> C

    # Local directed event ids (per pipe pair)
    SIG_K_L1 = 0  # MTE2 <-> MTE1 (k_l1[side] ownership)
    SIG_V_L1 = 2  # MTE2 <-> MTE1 (v_l1[side] ownership)
    SIG_P_L1 = 4  # MTE2 <-> MTE1 (p_l1 ownership)
    SIG_Q_L1 = 5  # MTE2 <-> MTE1 (q_l1 ownership)
    SIG_QFILL = 6  # V -> MTE2 (q_l1 fill ordering)
    SIG_L0AQ = 0  # MTE1 <-> M (l0a_q slots 0/1)
    SIG_L0AP = 2  # MTE1 <-> M (single l0a_p)
    SIG_L0BK = 3  # MTE1 <-> M (single l0b_k)
    SIG_L0BV = 4  # MTE1 <-> M (single l0b_v)
    SIG_L0CS = 0  # M <-> FIX (l0c_s slots 0/1)
    SIG_L0CO = 2  # M <-> FIX (l0c_o slots 0/1)
    SIG_IO_UB = 0  # MTE2 <-> V (io_buf / io_buf_o slots 0/1)
    SIG_MASK = 2  # MTE2 <-> V (mask_v ready/free)
    SIG_S_HALF = 0  # V <-> MTE3 (acc_p_half / acc_o_half staging)

    def task_range(cid_val):
        start = cid_val * q_tasks + T.if_then_else(cid_val < r_tasks, cid_val, r_tasks)
        count = q_tasks + T.if_then_else(cid_val < r_tasks, 1, 0)
        return start, count

    # factory-level scatter plan (per vid) — identical to the D128 expert
    scatter_plan = {}
    for v in range(2):
        segs = []
        for s in range(S):
            lo = s * G_inner if s * G_inner > v * half else v * half
            hi = (s + 1) * G_inner if (s + 1) * G_inner < (v + 1) * half else (v + 1) * half
            if lo < hi:
                segs.append((s, lo - v * half, hi - v * half, lo - s * G_inner, hi - s * G_inner))
        scatter_plan[v] = segs
    # Two static middle slots cover len(segs) <= 4 (see _gqa_kernel_dense_expert).
    assert S <= 4, "static middle-segment scatter slots require S <= 4 (decode guard)"

    @T.prim_func
    def main(
        Q: T.Tensor([B, S, N_q, D], dtype),  # type: ignore  BSND, real S
        K: T.Tensor([B, S_kv_padded, N_kv, D], dtype),  # type: ignore  BSND
        V: T.Tensor([B, S_kv_padded, N_kv, D], dtype),  # type: ignore  BSND
        Mask: T.Tensor([B, M_tile, S_kv_padded], accum_dtype),  # type: ignore
        Output: T.Tensor([B, S, N_q, D], dtype),  # type: ignore  BSND, real S
        ws_s: T.Tensor([NUM_CORES, num_stages, M_tile, block_N], accum_dtype),  # type: ignore
        ws_p: T.Tensor([NUM_CORES, num_stages, M_tile, block_N], dtype),  # type: ignore
        ws_o: T.Tensor([NUM_CORES, num_stages, M_tile, D], accum_dtype),  # type: ignore
    ):
        with T.Kernel(NUM_CORES, is_npu=True) as (cid, vid):
            # ===== C-side buffers =====
            q_l1 = T.alloc_L1([M_tile, D], dtype)
            k_l1_0 = T.alloc_L1([block_N, D], dtype)
            k_l1_1 = T.alloc_L1([block_N, D], dtype)
            v_l1_0 = T.alloc_L1([block_N, D], dtype)
            v_l1_1 = T.alloc_L1([block_N, D], dtype)
            p_l1 = T.alloc_L1([M_tile, block_N], dtype)

            T.annotate_layout(
                {
                    q_l1: make_zn_layout(q_l1),
                    k_l1_0: make_nz_layout(k_l1_0),
                    k_l1_1: make_nz_layout(k_l1_1),
                    p_l1: make_zn_layout(p_l1),
                    v_l1_0: make_zn_layout(v_l1_0),
                    v_l1_1: make_zn_layout(v_l1_1),
                }
            )

            # per-phase single-slot l0b (whole-D mma per phase) — 64KB exact
            l0a_q = T.alloc_L0A([2, M_tile, D], dtype)
            l0a_p = T.alloc_L0A([M_tile, block_N], dtype)
            l0b_k = T.alloc_L0B([D, block_N], dtype)
            l0b_v = T.alloc_L0B([block_N, D], dtype)
            l0c_s = T.alloc_L0C([2, M_tile, block_N], accum_dtype)
            l0c_o = T.alloc_L0C([2, M_tile, D], accum_dtype)

            # ===== V-side buffers (per vid, half rows) =====
            io_buf = T.alloc_ub([2, half, block_N], accum_dtype)
            io_buf_o = T.alloc_ub([2, half, D], accum_dtype)
            work_ub = T.alloc_ub([half, block_N], accum_dtype)
            work_o_ub = T.alloc_ub([half, D], accum_dtype)
            buf_2d = T.alloc_ub([half, block_N], accum_dtype)
            buf_2d_o = T.alloc_ub([half, D], accum_dtype)
            acc_p_half = T.alloc_ub([half, block_N], dtype)
            acc_o = T.alloc_ub([half, D], accum_dtype)
            acc_o_half = T.alloc_ub([half, D], dtype)
            mask_v = T.alloc_ub([half, block_N], accum_dtype)
            neg_sm = T.alloc_ub([2, half, 1], accum_dtype)
            sumexp = T.alloc_ub([half, 1], accum_dtype)
            sumexp_is = T.alloc_ub([num_stages, half, 1], accum_dtype)
            r_factors = T.alloc_ub([num_stages, half, 1], accum_dtype)

            my_start, my_count = task_range(cid)

            with T.Scope("C"):
                # init: pretend consumers already released everything
                T.set_cross_flag("MTE2", SEM_WS2_C2V)
                T.set_flag("MTE1", "MTE2", SIG_K_L1)
                T.set_flag("MTE1", "MTE2", SIG_K_L1 + 1)
                T.set_flag("MTE1", "MTE2", SIG_P_L1)
                T.set_flag("MTE1", "MTE2", SIG_V_L1)
                T.set_flag("MTE1", "MTE2", SIG_V_L1 + 1)
                T.set_flag("MTE1", "MTE2", SIG_Q_L1)
                T.set_flag("M", "MTE1", SIG_L0AQ)
                T.set_flag("M", "MTE1", SIG_L0AQ + 1)
                T.set_flag("M", "MTE1", SIG_L0AP)
                T.set_flag("M", "MTE1", SIG_L0BK)
                T.set_flag("M", "MTE1", SIG_L0BV)
                T.set_flag("FIX", "M", SIG_L0CS)
                T.set_flag("FIX", "M", SIG_L0CS + 1)
                T.set_flag("FIX", "M", SIG_L0CO)
                T.set_flag("FIX", "M", SIG_L0CO + 1)

                for t in T.serial(my_count):
                    task_id = my_start + t
                    gb = task_id % G_batches
                    kg = (task_id // G_batches) % N_kv
                    bz = task_id // (G_batches * N_kv)
                    hq0 = kg * G + gb * G_inner

                    # --- Q load: pad-row fill + s-major dense packing ---
                    T.wait_flag("MTE1", "MTE2", SIG_Q_L1)
                    T.tile.fill(q_l1, 0.0)
                    T.set_flag("V", "MTE2", SIG_QFILL)
                    T.wait_flag("V", "MTE2", SIG_QFILL)
                    for s in range(S):
                        T.copy(
                            Q[bz, s, hq0 : hq0 + G_inner, :],
                            q_l1[s * G_inner : (s + 1) * G_inner, :],
                        )
                    T.set_flag("MTE2", "MTE1", SIG_Q_L1)
                    T.wait_flag("MTE2", "MTE1", SIG_Q_L1)

                    for k in T.serial(num_outer):
                        _remaining = kv_iters - k * num_stages
                        batch_iters = T.if_then_else(_remaining < num_stages, _remaining, num_stages)

                        # --- GEMM1 batch: K loads + mma(Q, K^T) -> ws_s ---
                        T.wait_cross_flag(SEM_WS1_V2C)
                        for i in T.serial(batch_iters):
                            side = i % 2
                            idx = k * num_stages + i

                            T.wait_flag("MTE1", "MTE2", SIG_K_L1 + side)
                            if i % 2 == 0:
                                T.copy(K[bz, idx * block_N : (idx + 1) * block_N, kg, :], k_l1_0)
                            else:
                                T.copy(K[bz, idx * block_N : (idx + 1) * block_N, kg, :], k_l1_1)
                            T.set_flag("MTE2", "MTE1", SIG_K_L1 + side)

                            T.wait_flag("M", "MTE1", SIG_L0AQ + side)
                            if i < 2:
                                T.copy(q_l1, l0a_q[side, :, :])
                            T.wait_flag("MTE2", "MTE1", SIG_K_L1 + side)
                            T.wait_flag("M", "MTE1", SIG_L0BK)
                            if i % 2 == 0:
                                T.copy(k_l1_0, l0b_k, transpose=True)
                            else:
                                T.copy(k_l1_1, l0b_k, transpose=True)
                            T.set_flag("MTE1", "MTE2", SIG_K_L1 + side)
                            T.set_flag("MTE1", "M", SIG_L0AQ + side)
                            T.set_flag("MTE1", "M", SIG_L0BK)

                            T.wait_flag("MTE1", "M", SIG_L0AQ + side)
                            T.wait_flag("MTE1", "M", SIG_L0BK)
                            T.wait_flag("FIX", "M", SIG_L0CS + side)
                            T.mma(l0a_q[side, :, :], l0b_k, l0c_s[side, :, :], init=True)
                            T.set_flag("M", "MTE1", SIG_L0AQ + side)
                            T.set_flag("M", "MTE1", SIG_L0BK)
                            T.set_flag("M", "FIX", SIG_L0CS + side)

                            T.wait_flag("M", "FIX", SIG_L0CS + side)
                            T.copy(l0c_s[side, :, :], ws_s[cid, i, :, :])
                            T.set_flag("FIX", "M", SIG_L0CS + side)
                            if (i + 1) % cross_interval == 0 or i == batch_iters - 1:
                                T.set_cross_flag("FIX", SEM_WS1_C2V)

                        # --- GEMM2 batch: V loads + ws_p -> p_l1 + mma(P, V) -> ws_o ---
                        T.wait_cross_flag(SEM_WS3_V2C)
                        for i in T.serial(batch_iters):
                            side = i % 2
                            idx = k * num_stages + i

                            T.wait_flag("MTE1", "MTE2", SIG_V_L1 + side)
                            if i % 2 == 0:
                                T.copy(V[bz, idx * block_N : (idx + 1) * block_N, kg, :], v_l1_0)
                            else:
                                T.copy(V[bz, idx * block_N : (idx + 1) * block_N, kg, :], v_l1_1)
                            T.set_flag("MTE2", "MTE1", SIG_V_L1 + side)

                            T.wait_flag("MTE1", "MTE2", SIG_P_L1)
                            if i % cross_interval == 0:
                                T.wait_cross_flag(SEM_WS2_V2C)
                            T.copy(ws_p[cid, i, :, :], p_l1)
                            T.set_flag("MTE2", "MTE1", SIG_P_L1)

                            T.wait_flag("M", "MTE1", SIG_L0AP)
                            T.wait_flag("MTE2", "MTE1", SIG_P_L1)
                            T.copy(p_l1, l0a_p)
                            T.set_flag("MTE1", "MTE2", SIG_P_L1)
                            T.set_flag("MTE1", "M", SIG_L0AP)
                            T.wait_flag("MTE2", "MTE1", SIG_V_L1 + side)
                            T.wait_flag("M", "MTE1", SIG_L0BV)
                            if i % 2 == 0:
                                T.copy(v_l1_0, l0b_v)
                            else:
                                T.copy(v_l1_1, l0b_v)
                            T.set_flag("MTE1", "MTE2", SIG_V_L1 + side)
                            T.set_flag("MTE1", "M", SIG_L0BV)

                            T.wait_flag("MTE1", "M", SIG_L0AP)
                            T.wait_flag("MTE1", "M", SIG_L0BV)
                            T.wait_flag("FIX", "M", SIG_L0CO + side)
                            T.mma(l0a_p, l0b_v, l0c_o[side, :, :], init=True)
                            T.set_flag("M", "MTE1", SIG_L0AP)
                            T.set_flag("M", "MTE1", SIG_L0BV)
                            T.set_flag("M", "FIX", SIG_L0CO + side)

                            T.wait_flag("M", "FIX", SIG_L0CO + side)
                            T.copy(l0c_o[side, :, :], ws_o[cid, i, :, :])
                            T.set_flag("FIX", "M", SIG_L0CO + side)
                            if (i + 1) % cross_interval == 0 or i == batch_iters - 1:
                                T.set_cross_flag("FIX", SEM_WS3_C2V)

                        T.set_cross_flag("MTE2", SEM_WS2_C2V)

                    # MTE1 no longer reads q_l1; return it before the next task reloads Q
                    T.set_flag("MTE1", "MTE2", SIG_Q_L1)

                # destroy: consume outstanding init-direction flags
                T.wait_flag("MTE1", "MTE2", SIG_K_L1)
                T.wait_flag("MTE1", "MTE2", SIG_K_L1 + 1)
                T.wait_flag("MTE1", "MTE2", SIG_P_L1)
                T.wait_flag("MTE1", "MTE2", SIG_V_L1)
                T.wait_flag("MTE1", "MTE2", SIG_V_L1 + 1)
                T.wait_flag("MTE1", "MTE2", SIG_Q_L1)
                T.wait_flag("M", "MTE1", SIG_L0AQ)
                T.wait_flag("M", "MTE1", SIG_L0AQ + 1)
                T.wait_flag("M", "MTE1", SIG_L0AP)
                T.wait_flag("M", "MTE1", SIG_L0BK)
                T.wait_flag("M", "MTE1", SIG_L0BV)
                T.wait_flag("FIX", "M", SIG_L0CS)
                T.wait_flag("FIX", "M", SIG_L0CS + 1)
                T.wait_flag("FIX", "M", SIG_L0CO)
                T.wait_flag("FIX", "M", SIG_L0CO + 1)

            with T.Scope("V"):
                # init: pretend producers already released everything
                T.set_cross_flag("MTE2", SEM_WS1_V2C)
                T.set_cross_flag("MTE2", SEM_WS3_V2C)
                T.set_flag("V", "MTE2", SIG_IO_UB)
                T.set_flag("V", "MTE2", SIG_IO_UB + 1)
                T.set_flag("MTE3", "V", SIG_S_HALF)

                for t in T.serial(my_count):
                    task_id = my_start + t
                    gb = task_id % G_batches
                    kg = (task_id // G_batches) % N_kv
                    bz = task_id // (G_batches * N_kv)
                    hq0 = kg * G + gb * G_inner

                    T.tile.fill(acc_o, 0.0)
                    T.tile.fill(sumexp, 0.0)
                    T.tile.fill(neg_sm, neg_sm_cap)

                    for k in T.serial(num_outer):
                        _remaining = kv_iters - k * num_stages
                        batch_iters = T.if_then_else(_remaining < num_stages, _remaining, num_stages)

                        # --- phase 1: softmax batch (read ws_s, write ws_p) ---
                        T.wait_cross_flag(SEM_WS2_C2V)
                        for i in T.serial(batch_iters):
                            cur = i % 2
                            prv = 1 - cur
                            idx = k * num_stages + i
                            io_side = i % 2

                            T.wait_flag("V", "MTE2", SIG_IO_UB + io_side)
                            if i % cross_interval == 0:
                                T.wait_cross_flag(SEM_WS1_C2V)
                            T.copy(ws_s[cid, i, vid * half : vid * half + half, :], io_buf[io_side, :, :])
                            T.set_flag("MTE2", "V", SIG_IO_UB + io_side)

                            if use_mask:
                                # 2-flag mask sync (varlen pattern)
                                T.set_flag("V", "MTE2", SIG_MASK)
                                T.wait_flag("V", "MTE2", SIG_MASK)
                                T.copy(
                                    Mask[bz, vid * half : vid * half + half, idx * block_N : (idx + 1) * block_N],
                                    mask_v,
                                )
                                T.set_flag("MTE2", "V", SIG_MASK)

                            T.wait_flag("MTE2", "V", SIG_IO_UB + io_side)
                            T.copy(io_buf[io_side, :, :], work_ub)
                            T.set_flag("V", "MTE2", SIG_IO_UB + io_side)

                            # online softmax (batched 2-phase form)
                            # max-after-mask order — the additive mask is
                            # applied BEFORE the running max (see
                            # _gqa_kernel_dense_expert).
                            if use_mask:
                                T.wait_flag("MTE2", "V", SIG_MASK)
                                T.tile.add(work_ub, work_ub, mask_v)
                            # clear=True is kept: it compiles correctly on
                            # all builds and avoids the slow scalar merge
                            # loops of the clear=False lowering (see the
                            # _gqa_kernel_expert note).
                            T.reduce_max(work_ub, neg_sm[cur, :, :], dim=-1)
                            T.tile.mul(neg_sm[cur, :, :], neg_sm[cur, :, :], -sm_scale)
                            T.tile.min(neg_sm[cur, :, :], neg_sm[cur, :, :], neg_sm[prv, :, :])

                            T.tile.broadcast(buf_2d, neg_sm[cur, :, :])
                            T.tile.axpy(buf_2d, work_ub, sm_scale)
                            T.tile.exp(work_ub, buf_2d)

                            T.reduce_sum(work_ub, sumexp_is[i, :, :], dim=-1)
                            T.tile.sub(r_factors[i, :, :], neg_sm[cur, :, :], neg_sm[prv, :, :])

                            # write probs -> ws_p (via acc_p_half)
                            T.wait_flag("MTE3", "V", SIG_S_HALF)
                            T.copy(work_ub, acc_p_half)
                            T.set_flag("V", "MTE3", SIG_S_HALF)

                            T.wait_flag("V", "MTE3", SIG_S_HALF)
                            T.copy(acc_p_half, ws_p[cid, i, vid * half : vid * half + half, :])
                            T.set_flag("MTE3", "V", SIG_S_HALF)
                            if (i + 1) % cross_interval == 0 or i == batch_iters - 1:
                                T.set_cross_flag("MTE3", SEM_WS2_V2C)

                        T.set_cross_flag("MTE2", SEM_WS1_V2C)

                        # --- phase 2: O accumulation batch (read ws_o) ---
                        for i in T.serial(batch_iters):
                            T.tile.exp(r_factors[i, :, :], r_factors[i, :, :])
                            T.tile.mul(sumexp, sumexp, r_factors[i, :, :])
                            T.tile.add(sumexp, sumexp, sumexp_is[i, :, :])
                            T.tile.broadcast(buf_2d_o, r_factors[i, :, :])
                            T.tile.mul(acc_o, acc_o, buf_2d_o)

                            io_side = i % 2
                            T.wait_flag("V", "MTE2", SIG_IO_UB + io_side)
                            if i % cross_interval == 0:
                                T.wait_cross_flag(SEM_WS3_C2V)
                            T.copy(ws_o[cid, i, vid * half : vid * half + half, :], io_buf_o[io_side, :, :])
                            T.set_flag("MTE2", "V", SIG_IO_UB + io_side)

                            T.wait_flag("MTE2", "V", SIG_IO_UB + io_side)
                            T.copy(io_buf_o[io_side, :, :], work_o_ub)
                            T.set_flag("V", "MTE2", SIG_IO_UB + io_side)

                            T.tile.add(acc_o, acc_o, work_o_ub)

                        T.set_cross_flag("MTE2", SEM_WS3_V2C)

                    # --- final normalize + s-major scatter (per vid) ---
                    # fully-masked-row guard (see
                    # _gqa_kernel_expert).
                    T.tile.add(sumexp, sumexp, 1e-30)
                    T.tile.broadcast(buf_2d_o, sumexp)
                    T.tile.div(acc_o, acc_o, buf_2d_o)
                    T.wait_flag("MTE3", "V", SIG_S_HALF)
                    T.copy(acc_o, acc_o_half)
                    T.set_flag("V", "MTE3", SIG_S_HALF)
                    T.wait_flag("V", "MTE3", SIG_S_HALF)
                    if vid == 0:
                        segs = scatter_plan[0]
                        if len(segs) >= 1:
                            s_f, l_f, h_f, g_f, ge_f = segs[0]
                            T.copy(acc_o_half[l_f:h_f, :], Output[bz, s_f, hq0 + g_f : hq0 + ge_f, :])
                        if len(segs) >= 2:
                            s_l, l_l, h_l, g_l, ge_l = segs[-1]
                            T.copy(acc_o_half[l_l:h_l, :], Output[bz, s_l, hq0 + g_l : hq0 + ge_l, :])
                            if len(segs) > 2:
                                # Middle segments segs[1:-1] — static fixed-slot
                                # unroll; see the _gqa_kernel_dense_expert
                                # vid==0 note: the scatters must stay
                                # unconditional — a runtime `for s in range(S)`
                                # TIR loop inside the vid branch breaks the
                                # SIG_S_HALF handshake and hangs the launch;
                                # Python-bool frames fold at trace
                                # time and emit unconditional static scatters.
                                s_m1, l_m1, h_m1, g_m1, ge_m1 = segs[1]
                                T.copy(acc_o_half[l_m1:h_m1, :], Output[bz, s_m1, hq0 + g_m1 : hq0 + ge_m1, :])
                                if len(segs) > 3:
                                    s_m2, l_m2, h_m2, g_m2, ge_m2 = segs[2]
                                    T.copy(acc_o_half[l_m2:h_m2, :], Output[bz, s_m2, hq0 + g_m2 : hq0 + ge_m2, :])
                    if vid == 1:
                        segs = scatter_plan[1]
                        if len(segs) >= 1:
                            s_f, l_f, h_f, g_f, ge_f = segs[0]
                            T.copy(acc_o_half[l_f:h_f, :], Output[bz, s_f, hq0 + g_f : hq0 + ge_f, :])
                        if len(segs) >= 2:
                            s_l, l_l, h_l, g_l, ge_l = segs[-1]
                            T.copy(acc_o_half[l_l:h_l, :], Output[bz, s_l, hq0 + g_l : hq0 + ge_l, :])
                            if len(segs) > 2:
                                # Middle segments segs[1:-1] — static fixed-slot
                                # unroll; see the _gqa_kernel_dense_expert
                                # vid==0 note: the scatters must stay
                                # unconditional — a runtime `for s in range(S)`
                                # TIR loop inside the vid branch breaks the
                                # SIG_S_HALF handshake and hangs the launch;
                                # Python-bool frames fold at trace
                                # time and emit unconditional static scatters.
                                s_m1, l_m1, h_m1, g_m1, ge_m1 = segs[1]
                                T.copy(acc_o_half[l_m1:h_m1, :], Output[bz, s_m1, hq0 + g_m1 : hq0 + ge_m1, :])
                                if len(segs) > 3:
                                    s_m2, l_m2, h_m2, g_m2, ge_m2 = segs[2]
                                    T.copy(acc_o_half[l_m2:h_m2, :], Output[bz, s_m2, hq0 + g_m2 : hq0 + ge_m2, :])
                    T.set_flag("MTE3", "V", SIG_S_HALF)

                # destroy: consume outstanding init-direction flags
                T.wait_flag("V", "MTE2", SIG_IO_UB)
                T.wait_flag("V", "MTE2", SIG_IO_UB + 1)
                T.wait_flag("MTE3", "V", SIG_S_HALF)

    return main


@tilelang.jit(out_idx=[4], workspace_idx=[5, 6, 7], pass_configs=P28_EXPERT_PASS_CONFIGS)
def _gqa_kernel_prefill_expert(
    B,
    S_padded,
    S_kv_padded,
    sm_scale,
    is_causal,
    N_q=8,
    N_kv=2,
    D=128,
    dtype="float16",
    block_M=64,
    skv_minus_s=0,
    use_mask=True,
    block_N=128,
    num_stages=4,
    cross_interval=2,
):
    """Expert manual-flag deep-pipeline per-head prefill kernel.

    Dispatch guard (wrapper): S>4, D==128, kv_iters==1 or even kv_iters>=2,
    causal requires S_kv>=S, instance count B*N_q*q_blocks >= NUM_CORES.
    One task per (bz, hq, bx) Q-head block, statically split over NUM_CORES
    cores.

    kv_iters==1 (ns=1, odd) relaxation: with a single batch
    iteration the r_factors[0] = neg_sm[0] - neg_sm[1] path reads the
    neg_sm_cap init in slot 1 (both slots are filled with neg_sm_cap =
    -sm_scale*_BIG_NEG*0.5 before the batch), so r_factors[0] is hugely
    negative and exp(r_factors[0]) underflows to 0 — the rescale
    multiplies only the zero-initialized sumexp/acc_o, so the
    odd-slot parity hazard does not arise.  Odd ns>1 stays rejected: across
    multiple outer iterations the prv slot goes stale (running min lives in
    the cur slot of the previous, odd-length batch).

    Dataflow (per task): Q block loaded into q_l1 [block_M, D] (full BHSD
    box); KV loop batched by num_stages: GEMM1 batch (K load + mma -> ws_s)
    then GEMM2 batch (V load + ws_p -> p_l1 + mma -> ws_o); V-side (2 AIV,
    half rows each) runs phase-1 softmax batch + phase-2 O-accumulation batch
    concurrently with the C-side of the following batch (cross-core
    semaphores, cross_interval grain).

    Causal skip: KV blocks with idx*block_N >= (bx+1)*block_M + skv_minus_s
    are fully masked; the visible set is a block PREFIX, so the skip is
    expressed as a runtime loop bound kv_useful (identical semantics to the
    g2/per-head skip predicate, branch-free for flag pairing: both scopes
    compute the same kv_useful from the same task_id inputs, so the
    batch_iters sequence — and therefore every set/wait pair count — matches
    on the C and V sides).  Non-causal is subsumed: skip_limit = 1<<30 makes
    kv_u_raw huge, clamped to kv_iters.  NOTE: computed WITHOUT a trace-time
    `if` — TVM Script scopes branch-body assignments to the if's var frame
    (tir/parser.py visit_if), so an if/else form would leave batch_iters
    undefined after the branch.

    Softmax is max-after-mask (as in the g2/per-head prefill
    kernels: the additive mask is applied to the raw scores BEFORE the
    running-max update — avoids the max-before-mask bf16-causal underflow
    edge).  Output needs no scatter plan: BHSD makes each vid's [half, D]
    row segment one contiguous box (runtime-base/static-extent copy).
    """
    assert block_N == D, "expert prefill kernel requires D == block_N (square l0b tiles)"
    # ns==1 allowed (single-batch kv=1 form; the neg_sm odd-slot
    # parity hazard needs >=2 batch iterations to manifest).  Odd ns>1 is
    # still rejected: the prv slot goes stale across outer iterations.
    assert num_stages == 1 or num_stages % 2 == 0, "num_stages must be 1 or even (neg_sm slot parity across batches)"
    assert num_stages <= ((S_kv_padded + block_N - 1) // block_N), "num_stages <= kv_iters"
    accum_dtype = "float"

    # max-after-mask neg_sm init cap — sentinel + fully-masked-row max
    # floor at mask/2 (identical rationale as
    # _gqa_kernel_prefill_g2stack; see the note there).
    neg_sm_cap = -sm_scale * _BIG_NEG * 0.5

    G = N_q // N_kv
    q_blocks = S_padded // block_M
    kv_iters = S_kv_padded // block_N
    # kv_iters==1 allowed (ns=1 single-batch); odd kv_iters>1
    # (e.g. Skv=300 -> kv=3) still falls back to the g2/per-head paths
    assert kv_iters == 1 or (kv_iters >= 2 and kv_iters % 2 == 0), "expert prefill requires kv_iters == 1 or even kv_iters >= 2"
    assert (not is_causal) or skv_minus_s >= 0, "causal expert prefill requires S_kv >= S"
    block_num = B * N_q * q_blocks
    assert block_num >= P28_NUM_CORES, "expert prefill requires block_num >= NUM_CORES"
    num_outer = T.ceildiv(kv_iters, num_stages)
    half = block_M // 2
    # causal skip boundary offset (original, pre-padding — same as g2);
    # 1<<30 for non-causal subsumes the no-skip case into the clamp below
    skip_limit = skv_minus_s if is_causal else (1 << 30)

    q_tasks = block_num // P28_NUM_CORES
    r_tasks = block_num % P28_NUM_CORES

    # Cross-core semaphores (C <-> V) — verbatim dense-expert allocation
    SEM_WS1_C2V = 0  # ws_s ready:      C(FIX) -> V
    SEM_WS1_V2C = 1  # ws_s consumed:   V(MTE2) -> C
    SEM_WS2_V2C = 2  # ws_p ready:      V(MTE3) -> C
    SEM_WS2_C2V = 3  # ws_p consumed:   C(MTE2) -> V
    SEM_WS3_C2V = 4  # ws_o ready:      C(FIX) -> V
    SEM_WS3_V2C = 5  # ws_o consumed:   V(MTE2) -> C

    # Local directed event ids (per pipe pair) — verbatim dense expert
    # minus SIG_QFILL (no pad-row fill: S_padded is block aligned and the
    # wrapper zero-pads Q rows)
    SIG_K_L1 = 0  # MTE2 <-> MTE1 (k_l1[side] ownership, slots 0/1)
    SIG_V_L1 = 2  # MTE2 <-> MTE1 (v_l1[side] ownership, slots 0/1)
    SIG_P_L1 = 4  # MTE2 <-> MTE1 (p_l1 ownership)
    SIG_Q_L1 = 5  # MTE2 <-> MTE1 (q_l1 ownership)
    SIG_L0AB = 0  # MTE1 <-> M (l0a/l0b slots 0/1)
    SIG_L0C = 0  # M <-> FIX (l0c slots 0/1)
    SIG_IO_UB = 0  # MTE2 <-> V (io_buf slots 0/1)
    SIG_MASK = 2  # MTE2 <-> V (mask_v ready/free)
    SIG_S_HALF = 0  # V <-> MTE3 (acc_s_half ownership)

    def task_range(cid_val):
        start = cid_val * q_tasks + T.if_then_else(cid_val < r_tasks, cid_val, r_tasks)
        count = q_tasks + T.if_then_else(cid_val < r_tasks, 1, 0)
        return start, count

    @T.prim_func
    def main(
        Q: T.Tensor([B, N_q, S_padded, D], dtype),  # type: ignore  BHSD
        K: T.Tensor([B, N_kv, S_kv_padded, D], dtype),  # type: ignore  BHSD
        V: T.Tensor([B, N_kv, S_kv_padded, D], dtype),  # type: ignore  BHSD
        Mask: T.Tensor([B, S_padded, S_kv_padded], accum_dtype),  # type: ignore
        Output: T.Tensor([B, N_q, S_padded, D], dtype),  # type: ignore  BHSD
        ws_s: T.Tensor([P28_NUM_CORES, num_stages, block_M, block_N], accum_dtype),  # type: ignore
        ws_p: T.Tensor([P28_NUM_CORES, num_stages, block_M, block_N], dtype),  # type: ignore
        ws_o: T.Tensor([P28_NUM_CORES, num_stages, block_M, D], accum_dtype),  # type: ignore
    ):
        with T.Kernel(P28_NUM_CORES, is_npu=True) as (cid, vid):
            # ===== C-side buffers (verbatim dense expert) =====
            q_l1 = T.alloc_L1([block_M, D], dtype)
            # k/v double buffering: two separate 2D buffers per operand,
            # selected by a runtime `if` (whole-buffer copies only — sliced
            # copies of layout-annotated L1 buffers lower incorrectly)
            k_l1_0 = T.alloc_L1([block_N, D], dtype)
            k_l1_1 = T.alloc_L1([block_N, D], dtype)
            v_l1_0 = T.alloc_L1([block_N, D], dtype)
            v_l1_1 = T.alloc_L1([block_N, D], dtype)
            p_l1 = T.alloc_L1([block_M, block_N], dtype)

            T.annotate_layout(
                {
                    q_l1: make_zn_layout(q_l1),
                    k_l1_0: make_nz_layout(k_l1_0),
                    k_l1_1: make_nz_layout(k_l1_1),
                    p_l1: make_zn_layout(p_l1),
                    v_l1_0: make_zn_layout(v_l1_0),
                    v_l1_1: make_zn_layout(v_l1_1),
                }
            )

            # shared GEMM1/GEMM2 slots (D == block_N makes the shapes coincide)
            l0a = T.alloc_L0A([2, block_M, D], dtype)
            l0b = T.alloc_L0B([2, D, block_N], dtype)
            l0c = T.alloc_L0C([2, block_M, block_N], accum_dtype)

            # ===== V-side buffers (per vid, half rows) — verbatim dense expert =====
            io_buf = T.alloc_ub([2, half, block_N], accum_dtype)
            work_ub = T.alloc_ub([half, block_N], accum_dtype)
            buf_2d = T.alloc_ub([half, block_N], accum_dtype)
            acc_s_half = T.alloc_ub([half, block_N], dtype)
            acc_o = T.alloc_ub([half, D], accum_dtype)
            mask_v = T.alloc_ub([half, block_N], accum_dtype)
            neg_sm = T.alloc_ub([2, half, 1], accum_dtype)
            sumexp = T.alloc_ub([half, 1], accum_dtype)
            sumexp_is = T.alloc_ub([num_stages, half, 1], accum_dtype)
            r_factors = T.alloc_ub([num_stages, half, 1], accum_dtype)

            my_start, my_count = task_range(cid)

            with T.Scope("C"):
                # init: pretend consumers already released everything
                T.set_cross_flag("MTE2", SEM_WS2_C2V)
                T.set_flag("MTE1", "MTE2", SIG_K_L1)
                T.set_flag("MTE1", "MTE2", SIG_K_L1 + 1)
                T.set_flag("MTE1", "MTE2", SIG_P_L1)
                T.set_flag("MTE1", "MTE2", SIG_V_L1)
                T.set_flag("MTE1", "MTE2", SIG_V_L1 + 1)
                T.set_flag("MTE1", "MTE2", SIG_Q_L1)
                T.set_flag("M", "MTE1", SIG_L0AB)
                T.set_flag("M", "MTE1", SIG_L0AB + 1)
                T.set_flag("FIX", "M", SIG_L0C)
                T.set_flag("FIX", "M", SIG_L0C + 1)

                for t in T.serial(my_count):
                    task_id = my_start + t
                    bx = task_id % q_blocks
                    hq = (task_id // q_blocks) % N_q
                    bz = task_id // (q_blocks * N_q)
                    kg = hq // G

                    # --- Q load: full [block_M, D] box (BHSD contiguous) ---
                    T.wait_flag("MTE1", "MTE2", SIG_Q_L1)
                    T.copy(
                        Q[bz, hq, bx * block_M : (bx + 1) * block_M, :],
                        q_l1,
                    )
                    T.set_flag("MTE2", "MTE1", SIG_Q_L1)
                    T.wait_flag("MTE2", "MTE1", SIG_Q_L1)

                    for k in T.serial(num_outer):
                        # causal skip as a RUNTIME loop bound: visible KV
                        # blocks are the prefix [0, kv_useful) — the same set
                        # the g2 skip predicate k*BN < (bx+1)*BM + skip_limit
                        # admits.  Non-causal is subsumed: skip_limit = 1<<30
                        # makes kv_u_raw huge, clamped to kv_iters.
                        bound = (bx + 1) * block_M + skip_limit
                        kv_u_raw = (bound + block_N - 1) // block_N
                        kv_useful = T.if_then_else(kv_u_raw > kv_iters, kv_iters, kv_u_raw)
                        _remaining = kv_useful - k * num_stages
                        _rem_c = T.if_then_else(_remaining < 0, 0, _remaining)
                        batch_iters = T.if_then_else(_rem_c < num_stages, _rem_c, num_stages)

                        # --- GEMM1 batch: K loads + mma(Q, K^T) -> ws_s ---
                        T.wait_cross_flag(SEM_WS1_V2C)
                        for i in T.serial(batch_iters):
                            side = i % 2
                            idx = k * num_stages + i

                            T.wait_flag("MTE1", "MTE2", SIG_K_L1 + side)
                            if i % 2 == 0:
                                T.copy(
                                    K[bz, kg, idx * block_N : (idx + 1) * block_N, :],
                                    k_l1_0,
                                )
                            else:
                                T.copy(
                                    K[bz, kg, idx * block_N : (idx + 1) * block_N, :],
                                    k_l1_1,
                                )
                            T.set_flag("MTE2", "MTE1", SIG_K_L1 + side)

                            T.wait_flag("M", "MTE1", SIG_L0AB + side)
                            if i < 2:
                                T.copy(q_l1, l0a[side, :, :])

                            T.wait_flag("MTE2", "MTE1", SIG_K_L1 + side)
                            if i % 2 == 0:
                                T.copy(k_l1_0, l0b[0, :, :], transpose=True)
                            else:
                                T.copy(k_l1_1, l0b[1, :, :], transpose=True)
                            T.set_flag("MTE1", "MTE2", SIG_K_L1 + side)
                            T.set_flag("MTE1", "M", SIG_L0AB + side)

                            T.wait_flag("MTE1", "M", SIG_L0AB + side)
                            T.wait_flag("FIX", "M", SIG_L0C + side)
                            T.mma(l0a[side, :, :], l0b[side, :, :], l0c[side, :, :], init=True)
                            T.set_flag("M", "MTE1", SIG_L0AB + side)
                            T.set_flag("M", "FIX", SIG_L0C + side)

                            T.wait_flag("M", "FIX", SIG_L0C + side)
                            T.copy(l0c[side, :, :], ws_s[cid, i, :, :])
                            T.set_flag("FIX", "M", SIG_L0C + side)
                            if (i + 1) % cross_interval == 0 or i == batch_iters - 1:
                                T.set_cross_flag("FIX", SEM_WS1_C2V)

                        # --- GEMM2 batch: V loads + ws_p -> p_l1 + mma(P, V) -> ws_o ---
                        T.wait_cross_flag(SEM_WS3_V2C)
                        for i in T.serial(batch_iters):
                            side = i % 2
                            idx = k * num_stages + i

                            T.wait_flag("MTE1", "MTE2", SIG_V_L1 + side)
                            if i % 2 == 0:
                                T.copy(
                                    V[bz, kg, idx * block_N : (idx + 1) * block_N, :],
                                    v_l1_0,
                                )
                            else:
                                T.copy(
                                    V[bz, kg, idx * block_N : (idx + 1) * block_N, :],
                                    v_l1_1,
                                )
                            T.set_flag("MTE2", "MTE1", SIG_V_L1 + side)

                            T.wait_flag("MTE1", "MTE2", SIG_P_L1)
                            if i % cross_interval == 0:
                                T.wait_cross_flag(SEM_WS2_V2C)
                            T.copy(ws_p[cid, i, :, :], p_l1)
                            T.set_flag("MTE2", "MTE1", SIG_P_L1)

                            T.wait_flag("MTE2", "MTE1", SIG_V_L1 + side)
                            T.wait_flag("M", "MTE1", SIG_L0AB + side)
                            if i % 2 == 0:
                                T.copy(v_l1_0, l0b[0, :, :])
                            else:
                                T.copy(v_l1_1, l0b[1, :, :])
                            T.set_flag("MTE1", "MTE2", SIG_V_L1 + side)

                            T.wait_flag("MTE2", "MTE1", SIG_P_L1)
                            T.copy(p_l1, l0a[side, :, :])
                            T.set_flag("MTE1", "MTE2", SIG_P_L1)
                            T.set_flag("MTE1", "M", SIG_L0AB + side)

                            T.wait_flag("MTE1", "M", SIG_L0AB + side)
                            T.wait_flag("FIX", "M", SIG_L0C + side)
                            T.mma(l0a[side, :, :], l0b[side, :, :], l0c[side, :, :], init=True)
                            T.set_flag("M", "MTE1", SIG_L0AB + side)
                            T.set_flag("M", "FIX", SIG_L0C + side)

                            T.wait_flag("M", "FIX", SIG_L0C + side)
                            T.copy(l0c[side, :, :], ws_o[cid, i, :, :])
                            T.set_flag("FIX", "M", SIG_L0C + side)
                            if (i + 1) % cross_interval == 0 or i == batch_iters - 1:
                                T.set_cross_flag("FIX", SEM_WS3_C2V)

                        T.set_cross_flag("MTE2", SEM_WS2_C2V)

                    # MTE1 no longer reads q_l1; return it before the next task reloads Q
                    T.set_flag("MTE1", "MTE2", SIG_Q_L1)

                # destroy: consume outstanding init-direction flags
                T.wait_flag("MTE1", "MTE2", SIG_K_L1)
                T.wait_flag("MTE1", "MTE2", SIG_K_L1 + 1)
                T.wait_flag("MTE1", "MTE2", SIG_P_L1)
                T.wait_flag("MTE1", "MTE2", SIG_V_L1)
                T.wait_flag("MTE1", "MTE2", SIG_V_L1 + 1)
                T.wait_flag("MTE1", "MTE2", SIG_Q_L1)
                T.wait_flag("M", "MTE1", SIG_L0AB)
                T.wait_flag("M", "MTE1", SIG_L0AB + 1)
                T.wait_flag("FIX", "M", SIG_L0C)
                T.wait_flag("FIX", "M", SIG_L0C + 1)

            with T.Scope("V"):
                # init: pretend producers already released everything
                T.set_cross_flag("MTE2", SEM_WS1_V2C)
                T.set_cross_flag("MTE2", SEM_WS3_V2C)
                T.set_flag("V", "MTE2", SIG_IO_UB)
                T.set_flag("V", "MTE2", SIG_IO_UB + 1)
                T.set_flag("MTE3", "V", SIG_S_HALF)

                for t in T.serial(my_count):
                    task_id = my_start + t
                    bx = task_id % q_blocks
                    hq = (task_id // q_blocks) % N_q
                    bz = task_id // (q_blocks * N_q)

                    T.tile.fill(acc_o, 0.0)
                    T.tile.fill(sumexp, 0.0)
                    T.tile.fill(neg_sm, neg_sm_cap)

                    for k in T.serial(num_outer):
                        # same unified runtime kv_useful bound as the C scope
                        # (identical formula + inputs -> identical batch_iters
                        # sequence -> flag pairs stay balanced on both sides)
                        bound = (bx + 1) * block_M + skip_limit
                        kv_u_raw = (bound + block_N - 1) // block_N
                        kv_useful = T.if_then_else(kv_u_raw > kv_iters, kv_iters, kv_u_raw)
                        _remaining = kv_useful - k * num_stages
                        _rem_c = T.if_then_else(_remaining < 0, 0, _remaining)
                        batch_iters = T.if_then_else(_rem_c < num_stages, _rem_c, num_stages)

                        # --- phase 1: softmax batch (read ws_s, write ws_p) ---
                        T.wait_cross_flag(SEM_WS2_C2V)
                        for i in T.serial(batch_iters):
                            cur = i % 2
                            prv = 1 - cur
                            idx = k * num_stages + i
                            io_side = i % 2

                            T.wait_flag("V", "MTE2", SIG_IO_UB + io_side)
                            if i % cross_interval == 0:
                                T.wait_cross_flag(SEM_WS1_C2V)
                            T.copy(ws_s[cid, i, vid * half : vid * half + half, :], io_buf[io_side, :, :])
                            T.set_flag("MTE2", "V", SIG_IO_UB + io_side)

                            if use_mask:
                                # 2-flag mask sync (varlen pattern): MTE2 loads
                                # mask_v only after the previous iteration's add
                                # consumed it (set+wait = V->MTE2 ordering point)
                                T.set_flag("V", "MTE2", SIG_MASK)
                                T.wait_flag("V", "MTE2", SIG_MASK)
                                T.copy(
                                    Mask[
                                        bz,
                                        bx * block_M + vid * half : bx * block_M + vid * half + half,
                                        idx * block_N : (idx + 1) * block_N,
                                    ],
                                    mask_v,
                                )
                                T.set_flag("MTE2", "V", SIG_MASK)

                            T.wait_flag("MTE2", "V", SIG_IO_UB + io_side)
                            T.copy(io_buf[io_side, :, :], work_ub)
                            T.set_flag("V", "MTE2", SIG_IO_UB + io_side)

                            # online softmax (batched 2-phase form) —
                            # max-after-mask: the additive mask is applied
                            # to the raw scores BEFORE the running-max update
                            # (the reference semantics; matches the g2/per-head
                            # prefill kernels, avoids the max-before-mask
                            # bf16-causal underflow edge)
                            if use_mask:
                                T.wait_flag("MTE2", "V", SIG_MASK)
                                T.tile.add(work_ub, work_ub, mask_v)
                            # clear=True is kept: it compiles correctly on
                            # all builds and avoids the slow scalar merge
                            # loops of the clear=False lowering (see the
                            # _gqa_kernel_expert note).
                            T.reduce_max(work_ub, neg_sm[cur, :, :], dim=-1)
                            T.tile.mul(neg_sm[cur, :, :], neg_sm[cur, :, :], -sm_scale)
                            T.tile.min(neg_sm[cur, :, :], neg_sm[cur, :, :], neg_sm[prv, :, :])

                            T.tile.broadcast(buf_2d, neg_sm[cur, :, :])
                            T.tile.axpy(buf_2d, work_ub, sm_scale)
                            T.tile.exp(work_ub, buf_2d)

                            T.reduce_sum(work_ub, sumexp_is[i, :, :], dim=-1)
                            T.tile.sub(r_factors[i, :, :], neg_sm[cur, :, :], neg_sm[prv, :, :])

                            # write probs -> ws_p (via acc_s_half)
                            T.wait_flag("MTE3", "V", SIG_S_HALF)
                            T.copy(work_ub, acc_s_half)
                            T.set_flag("V", "MTE3", SIG_S_HALF)

                            T.wait_flag("V", "MTE3", SIG_S_HALF)
                            T.copy(acc_s_half, ws_p[cid, i, vid * half : vid * half + half, :])
                            T.set_flag("MTE3", "V", SIG_S_HALF)
                            if (i + 1) % cross_interval == 0 or i == batch_iters - 1:
                                T.set_cross_flag("MTE3", SEM_WS2_V2C)

                        T.set_cross_flag("MTE2", SEM_WS1_V2C)

                        # --- phase 2: O accumulation batch (read ws_o) ---
                        for i in T.serial(batch_iters):
                            T.tile.exp(r_factors[i, :, :], r_factors[i, :, :])
                            T.tile.mul(sumexp, sumexp, r_factors[i, :, :])
                            T.tile.add(sumexp, sumexp, sumexp_is[i, :, :])
                            T.tile.broadcast(buf_2d, r_factors[i, :, :])
                            T.tile.mul(acc_o, acc_o, buf_2d)

                            io_side = i % 2
                            T.wait_flag("V", "MTE2", SIG_IO_UB + io_side)
                            if i % cross_interval == 0:
                                T.wait_cross_flag(SEM_WS3_C2V)
                            T.copy(ws_o[cid, i, vid * half : vid * half + half, :], io_buf[io_side, :, :])
                            T.set_flag("MTE2", "V", SIG_IO_UB + io_side)

                            T.wait_flag("MTE2", "V", SIG_IO_UB + io_side)
                            T.copy(io_buf[io_side, :, :], work_ub)
                            T.set_flag("V", "MTE2", SIG_IO_UB + io_side)

                            T.tile.add(acc_o, acc_o, work_ub)

                        T.set_cross_flag("MTE2", SEM_WS3_V2C)

                    # --- final normalize + per-vid contiguous output box ---
                    # (Output rows bx*BM+vid*half .. +half form one contiguous
                    # [half, D] box in BHSD — a single runtime-base/static-
                    # extent copy; the runtime `vid` slice pattern is the one
                    # proven by the ws reads above. No scatter plan needed.)
                    # fully-masked-row guard (see
                    # _gqa_kernel_expert).
                    T.tile.add(sumexp, sumexp, 1e-30)
                    T.tile.broadcast(buf_2d, sumexp)
                    T.tile.div(acc_o, acc_o, buf_2d)
                    T.wait_flag("MTE3", "V", SIG_S_HALF)
                    T.copy(acc_o, acc_s_half)
                    T.set_flag("V", "MTE3", SIG_S_HALF)
                    T.wait_flag("V", "MTE3", SIG_S_HALF)
                    T.copy(
                        acc_s_half,
                        Output[bz, hq, bx * block_M + vid * half : bx * block_M + vid * half + half, :],
                    )
                    T.set_flag("MTE3", "V", SIG_S_HALF)

                # destroy: consume outstanding init-direction flags
                T.wait_flag("V", "MTE2", SIG_IO_UB)
                T.wait_flag("V", "MTE2", SIG_IO_UB + 1)
                T.wait_flag("MTE3", "V", SIG_S_HALF)

    return main


@tilelang.jit(out_idx=[4], workspace_idx=[5, 6, 7], pass_configs=P28_EXPERT_PASS_CONFIGS)
def _gqa_kernel_prefill_g2stack(
    B,
    S_padded,
    S_kv_padded,
    sm_scale,
    is_causal,
    N_q=8,
    N_kv=2,
    D=128,
    dtype="float16",
    block_M=64,
    skv_minus_s=0,
    use_mask=True,
    block_N=128,
    num_stages=4,
    cross_interval=2,
):
    """G_inner=2 M-stacked expert manual-flag deep-pipeline prefill kernel.

    One task serves the 2 Q heads (hq0, hq0+1) of one (bz, kg, gb) at the same
    bx: K/V/mask GM traffic and task count /2 vs the per-head expert; the
    per-iteration C<->V handshake packet cost amortizes over both heads
    (the scalar-pipe handshake was measured as the top stall source).

    Dispatch guard (wrapper): S>4, D==128, fp16/bf16, kv_iters==1 or even
    >=2, causal requires S_kv>=S, G%2==0 (no tail head-pair), stacked
    instance count B*N_kv*(G/2)*q_blocks >= NUM_CORES.

    Stacked GEMM form: M_stack = 2*block_M.  l0a [2, M_stack, D] f16 is
    EXACTLY the 65536B L0A limit; l0c [2, M_stack, block_N] f32 is EXACTLY
    the 131072B wmma.accumulator planning envelope (boundary-inclusive, the
    same total the g2 kernel runs as 4x[64,128]) — zero margin by design.

    V-side row mapping: stacked block rows [h*BM, (h+1)*BM) belong to head
    hq0+h; vid v processes the [v*half, v*half+half) strip of EACH head
    (leading-dim runtime indexing — the io_buf[io_side]/neg_sm[cur] proven
    pattern).  Tile temporaries (io_buf/work_ub/buf_2d/acc_s_half/mask_v)
    are shared by both heads with per-(i,h) complete flag ping-pongs; only
    the row STATE (acc_o/neg_sm/sumexp/sumexp_is/r_factors) is per head
    (leading-dim head index: acc_o [2,half,D], neg_sm [4,half,1] idx
    h*2+cur, sumexp [2,half,1], sumexp_is/r_factors [ns*2,half,1] idx
    i*2+h).  io_side = h keeps the MTE2/V io_buf ping-pong overlapped
    (h1's MTE2 prefetch runs under h0's V compute).

    Cross-core semaphore sequence per task/k: STRUCTURALLY IDENTICAL to the
    per-head expert — the inner h-loop never touches cross flags; the
    ci-grain waits/sets are hoisted outside it, so every set/wait pair count
    matches on the C and V sides by construction (verified bit-stable
    across repeated runs).

    Causal skip: runtime loop bound kv_useful (branch-free, identical
    formula on C and V scopes -> identical batch_iters sequence -> flag
    pair counts match).  Both heads cover the SAME Q-row range
    [bx*BM, (bx+1)*BM) -> same mask rows (mask depends only on (s, j)) ->
    same visible prefix; no new mask form.

    Softmax is max-after-mask (mask applied to raw scores before the
    running max) with the batched 2-phase form of the per-head expert;
    per-head recurrences are independent and keep the exact
    per-head update order (phase-2 h-loop inside the i-loop).  The neg_sm
    slot parity argument holds per head: ns even keeps the prv
    slot live across batch boundaries; ns==1 single-batch is harmless
    (r_factors[0] = neg_sm[0] - neg_sm_cap init -> exp() underflows to 0
    and multiplies only the zero-init accumulators).  Output: per (h, vid)
    contiguous [half, D] BHSD box
    (runtime-base/static-extent copy).
    """
    assert block_N == D, "g2stack prefill requires D == block_N (square l0b tiles)"
    assert num_stages == 1 or num_stages % 2 == 0, "num_stages must be 1 or even (neg_sm slot parity across batches)"
    assert num_stages <= ((S_kv_padded + block_N - 1) // block_N), "num_stages <= kv_iters"
    accum_dtype = "float"

    # max-after-mask neg_sm init cap.  This
    # kernel applies the additive mask to the RAW scores BEFORE the running
    # max, so a masked column can only win the max if its raw score exceeds
    # the visible max by more than |mask| — impossible with _BIG_NEG=-1e30
    # vs finite fp32 scores.  Fully-masked rows (mixed blocks) would still
    # cancel the additive mask in the shift (softmax shift-invariance), so
    # the running max is FLOORED at mask/2: neg_sm <= -sm_scale*_BIG_NEG/2.
    # Normal rows keep the exact max (their neg_sm <= ~5e10 << cap).
    neg_sm_cap = -sm_scale * _BIG_NEG * 0.5

    G = N_q // N_kv
    assert G % 2 == 0, "g2stack prefill requires even G (wrapper guards odd/tail)"
    G_batches = G // 2
    q_blocks = S_padded // block_M
    kv_iters = S_kv_padded // block_N
    assert kv_iters == 1 or (kv_iters >= 2 and kv_iters % 2 == 0), "g2stack prefill requires kv_iters == 1 or even kv_iters >= 2"
    assert (not is_causal) or skv_minus_s >= 0, "causal requires S_kv >= S"
    block_num = B * N_kv * G_batches * q_blocks
    assert block_num >= P28_NUM_CORES, "g2stack requires block_num >= NUM_CORES"
    num_outer = T.ceildiv(kv_iters, num_stages)
    half = block_M // 2  # per-vid row strip within one head's segment
    M_stack = 2 * block_M  # stacked GEMM M dim (2 Q heads)
    # causal skip boundary offset (original, pre-padding — same as per-head);
    # 1<<30 for non-causal subsumes the no-skip case into the clamp below
    skip_limit = skv_minus_s if is_causal else (1 << 30)

    q_tasks = block_num // P28_NUM_CORES
    r_tasks = block_num % P28_NUM_CORES

    # Cross-core semaphores (C <-> V) — verbatim per-head expert allocation
    SEM_WS1_C2V = 0  # ws_s ready:      C(FIX) -> V
    SEM_WS1_V2C = 1  # ws_s consumed:   V(MTE2) -> C
    SEM_WS2_V2C = 2  # ws_p ready:      V(MTE3) -> C
    SEM_WS2_C2V = 3  # ws_p consumed:   C(MTE2) -> V
    SEM_WS3_C2V = 4  # ws_o ready:      C(FIX) -> V
    SEM_WS3_V2C = 5  # ws_o consumed:   V(MTE2) -> C

    # Local directed event ids (per pipe pair) — verbatim per-head expert
    SIG_K_L1 = 0  # MTE2 <-> MTE1 (k_l1[side] ownership, slots 0/1)
    SIG_V_L1 = 2  # MTE2 <-> MTE1 (v_l1[side] ownership, slots 0/1)
    SIG_P_L1 = 4  # MTE2 <-> MTE1 (p_l1 ownership)
    SIG_Q_L1 = 5  # MTE2 <-> MTE1 (q_l1 ownership)
    SIG_L0AB = 0  # MTE1 <-> M (l0a/l0b slots 0/1)
    SIG_L0C = 0  # M <-> FIX (l0c slots 0/1)
    SIG_IO_UB = 0  # MTE2 <-> V (io_buf slots 0/1)
    SIG_MASK = 2  # MTE2 <-> V (mask_v ready/free)
    SIG_S_HALF = 0  # V <-> MTE3 (acc_s_half ownership)

    def task_range(cid_val):
        start = cid_val * q_tasks + T.if_then_else(cid_val < r_tasks, cid_val, r_tasks)
        count = q_tasks + T.if_then_else(cid_val < r_tasks, 1, 0)
        return start, count

    @T.prim_func
    def main(
        Q: T.Tensor([B, N_q, S_padded, D], dtype),  # type: ignore  BHSD
        K: T.Tensor([B, N_kv, S_kv_padded, D], dtype),  # type: ignore  BHSD
        V: T.Tensor([B, N_kv, S_kv_padded, D], dtype),  # type: ignore  BHSD
        Mask: T.Tensor([B, S_padded, S_kv_padded], accum_dtype),  # type: ignore
        Output: T.Tensor([B, N_q, S_padded, D], dtype),  # type: ignore  BHSD
        ws_s: T.Tensor([P28_NUM_CORES, num_stages, M_stack, block_N], accum_dtype),  # type: ignore
        ws_p: T.Tensor([P28_NUM_CORES, num_stages, M_stack, block_N], dtype),  # type: ignore
        ws_o: T.Tensor([P28_NUM_CORES, num_stages, M_stack, D], accum_dtype),  # type: ignore
    ):
        with T.Kernel(P28_NUM_CORES, is_npu=True) as (cid, vid):
            # ===== C-side buffers (per-head expert, M stacked) =====
            q_l1 = T.alloc_L1([M_stack, D], dtype)
            # k/v double buffering: two separate 2D buffers per operand,
            # selected by a runtime `if` (whole-buffer copies only — sliced
            # copies of layout-annotated L1 buffers lower incorrectly)
            k_l1_0 = T.alloc_L1([block_N, D], dtype)
            k_l1_1 = T.alloc_L1([block_N, D], dtype)
            v_l1_0 = T.alloc_L1([block_N, D], dtype)
            v_l1_1 = T.alloc_L1([block_N, D], dtype)
            p_l1 = T.alloc_L1([M_stack, block_N], dtype)

            T.annotate_layout(
                {
                    q_l1: make_zn_layout(q_l1),
                    k_l1_0: make_nz_layout(k_l1_0),
                    k_l1_1: make_nz_layout(k_l1_1),
                    p_l1: make_zn_layout(p_l1),
                    v_l1_0: make_zn_layout(v_l1_0),
                    v_l1_1: make_zn_layout(v_l1_1),
                }
            )

            # shared GEMM1/GEMM2 slots (D == block_N makes the shapes coincide);
            # l0a 64KB / l0c 131072B EXACTLY at their envelopes (zero margin
            # by design)
            l0a = T.alloc_L0A([2, M_stack, D], dtype)
            l0b = T.alloc_L0B([2, D, block_N], dtype)
            l0c = T.alloc_L0C([2, M_stack, block_N], accum_dtype)

            # ===== V-side buffers (per vid) =====
            # tile temporaries shared by both heads (complete per-(i,h) flag
            # ping-pongs); row state per head via leading-dim runtime index
            io_buf = T.alloc_ub([2, half, block_N], accum_dtype)
            work_ub = T.alloc_ub([half, block_N], accum_dtype)
            buf_2d = T.alloc_ub([half, block_N], accum_dtype)
            acc_s_half = T.alloc_ub([half, block_N], dtype)
            acc_o = T.alloc_ub([2, half, D], accum_dtype)
            mask_v = T.alloc_ub([half, block_N], accum_dtype)
            neg_sm = T.alloc_ub([4, half, 1], accum_dtype)  # idx h*2 + cur
            sumexp = T.alloc_ub([2, half, 1], accum_dtype)  # idx h
            sumexp_is = T.alloc_ub([num_stages * 2, half, 1], accum_dtype)  # idx i*2 + h
            r_factors = T.alloc_ub([num_stages * 2, half, 1], accum_dtype)  # idx i*2 + h

            my_start, my_count = task_range(cid)

            with T.Scope("C"):
                # init: pretend consumers already released everything
                T.set_cross_flag("MTE2", SEM_WS2_C2V)
                T.set_flag("MTE1", "MTE2", SIG_K_L1)
                T.set_flag("MTE1", "MTE2", SIG_K_L1 + 1)
                T.set_flag("MTE1", "MTE2", SIG_P_L1)
                T.set_flag("MTE1", "MTE2", SIG_V_L1)
                T.set_flag("MTE1", "MTE2", SIG_V_L1 + 1)
                T.set_flag("MTE1", "MTE2", SIG_Q_L1)
                T.set_flag("M", "MTE1", SIG_L0AB)
                T.set_flag("M", "MTE1", SIG_L0AB + 1)
                T.set_flag("FIX", "M", SIG_L0C)
                T.set_flag("FIX", "M", SIG_L0C + 1)

                for t in T.serial(my_count):
                    task_id = my_start + t
                    bx = task_id % q_blocks
                    gb = (task_id // q_blocks) % G_batches
                    kg = (task_id // (q_blocks * G_batches)) % N_kv
                    bz = task_id // (q_blocks * G_batches * N_kv)
                    hq0 = kg * G + gb * 2

                    # --- Q load: both heads' [block_M, D] boxes stacked into
                    # q_l1 (static-slice dest — the dense-expert proven
                    # sliced-L1 pattern; whole q_l1 then copies to l0a) ---
                    T.wait_flag("MTE1", "MTE2", SIG_Q_L1)
                    T.copy(
                        Q[bz, hq0, bx * block_M : (bx + 1) * block_M, :],
                        q_l1[0:block_M, :],
                    )
                    T.copy(
                        Q[bz, hq0 + 1, bx * block_M : (bx + 1) * block_M, :],
                        q_l1[block_M:M_stack, :],
                    )
                    T.set_flag("MTE2", "MTE1", SIG_Q_L1)
                    T.wait_flag("MTE2", "MTE1", SIG_Q_L1)

                    for k in T.serial(num_outer):
                        # causal skip as a RUNTIME loop bound (branch-free,
                        # identical on C and V scopes — see per-head expert)
                        bound = (bx + 1) * block_M + skip_limit
                        kv_u_raw = (bound + block_N - 1) // block_N
                        kv_useful = T.if_then_else(kv_u_raw > kv_iters, kv_iters, kv_u_raw)
                        _remaining = kv_useful - k * num_stages
                        _rem_c = T.if_then_else(_remaining < 0, 0, _remaining)
                        batch_iters = T.if_then_else(_rem_c < num_stages, _rem_c, num_stages)

                        # --- GEMM1 batch: K loads + mma(Q, K^T) -> ws_s ---
                        T.wait_cross_flag(SEM_WS1_V2C)
                        for i in T.serial(batch_iters):
                            side = i % 2
                            idx = k * num_stages + i

                            T.wait_flag("MTE1", "MTE2", SIG_K_L1 + side)
                            if i % 2 == 0:
                                T.copy(
                                    K[bz, kg, idx * block_N : (idx + 1) * block_N, :],
                                    k_l1_0,
                                )
                            else:
                                T.copy(
                                    K[bz, kg, idx * block_N : (idx + 1) * block_N, :],
                                    k_l1_1,
                                )
                            T.set_flag("MTE2", "MTE1", SIG_K_L1 + side)

                            T.wait_flag("M", "MTE1", SIG_L0AB + side)
                            if i < 2:
                                T.copy(q_l1, l0a[side, :, :])

                            T.wait_flag("MTE2", "MTE1", SIG_K_L1 + side)
                            if i % 2 == 0:
                                T.copy(k_l1_0, l0b[0, :, :], transpose=True)
                            else:
                                T.copy(k_l1_1, l0b[1, :, :], transpose=True)
                            T.set_flag("MTE1", "MTE2", SIG_K_L1 + side)
                            T.set_flag("MTE1", "M", SIG_L0AB + side)

                            T.wait_flag("MTE1", "M", SIG_L0AB + side)
                            T.wait_flag("FIX", "M", SIG_L0C + side)
                            T.mma(l0a[side, :, :], l0b[side, :, :], l0c[side, :, :], init=True)
                            T.set_flag("M", "MTE1", SIG_L0AB + side)
                            T.set_flag("M", "FIX", SIG_L0C + side)

                            T.wait_flag("M", "FIX", SIG_L0C + side)
                            T.copy(l0c[side, :, :], ws_s[cid, i, :, :])
                            T.set_flag("FIX", "M", SIG_L0C + side)
                            if (i + 1) % cross_interval == 0 or i == batch_iters - 1:
                                T.set_cross_flag("FIX", SEM_WS1_C2V)

                        # --- GEMM2 batch: V loads + ws_p -> p_l1 + mma(P, V) -> ws_o ---
                        T.wait_cross_flag(SEM_WS3_V2C)
                        for i in T.serial(batch_iters):
                            side = i % 2
                            idx = k * num_stages + i

                            T.wait_flag("MTE1", "MTE2", SIG_V_L1 + side)
                            if i % 2 == 0:
                                T.copy(
                                    V[bz, kg, idx * block_N : (idx + 1) * block_N, :],
                                    v_l1_0,
                                )
                            else:
                                T.copy(
                                    V[bz, kg, idx * block_N : (idx + 1) * block_N, :],
                                    v_l1_1,
                                )
                            T.set_flag("MTE2", "MTE1", SIG_V_L1 + side)

                            T.wait_flag("MTE1", "MTE2", SIG_P_L1)
                            if i % cross_interval == 0:
                                T.wait_cross_flag(SEM_WS2_V2C)
                            T.copy(ws_p[cid, i, :, :], p_l1)
                            T.set_flag("MTE2", "MTE1", SIG_P_L1)

                            T.wait_flag("MTE2", "MTE1", SIG_V_L1 + side)
                            T.wait_flag("M", "MTE1", SIG_L0AB + side)
                            if i % 2 == 0:
                                T.copy(v_l1_0, l0b[0, :, :])
                            else:
                                T.copy(v_l1_1, l0b[1, :, :])
                            T.set_flag("MTE1", "MTE2", SIG_V_L1 + side)

                            T.wait_flag("MTE2", "MTE1", SIG_P_L1)
                            T.copy(p_l1, l0a[side, :, :])
                            T.set_flag("MTE1", "MTE2", SIG_P_L1)
                            T.set_flag("MTE1", "M", SIG_L0AB + side)

                            T.wait_flag("MTE1", "M", SIG_L0AB + side)
                            T.wait_flag("FIX", "M", SIG_L0C + side)
                            T.mma(l0a[side, :, :], l0b[side, :, :], l0c[side, :, :], init=True)
                            T.set_flag("M", "MTE1", SIG_L0AB + side)
                            T.set_flag("M", "FIX", SIG_L0C + side)

                            T.wait_flag("M", "FIX", SIG_L0C + side)
                            T.copy(l0c[side, :, :], ws_o[cid, i, :, :])
                            T.set_flag("FIX", "M", SIG_L0C + side)
                            if (i + 1) % cross_interval == 0 or i == batch_iters - 1:
                                T.set_cross_flag("FIX", SEM_WS3_C2V)

                        T.set_cross_flag("MTE2", SEM_WS2_C2V)

                    # MTE1 no longer reads q_l1; return it before the next task reloads Q
                    T.set_flag("MTE1", "MTE2", SIG_Q_L1)

                # destroy: consume outstanding init-direction flags
                T.wait_flag("MTE1", "MTE2", SIG_K_L1)
                T.wait_flag("MTE1", "MTE2", SIG_K_L1 + 1)
                T.wait_flag("MTE1", "MTE2", SIG_P_L1)
                T.wait_flag("MTE1", "MTE2", SIG_V_L1)
                T.wait_flag("MTE1", "MTE2", SIG_V_L1 + 1)
                T.wait_flag("MTE1", "MTE2", SIG_Q_L1)
                T.wait_flag("M", "MTE1", SIG_L0AB)
                T.wait_flag("M", "MTE1", SIG_L0AB + 1)
                T.wait_flag("FIX", "M", SIG_L0C)
                T.wait_flag("FIX", "M", SIG_L0C + 1)

            with T.Scope("V"):
                # init: pretend producers already released everything
                T.set_cross_flag("MTE2", SEM_WS1_V2C)
                T.set_cross_flag("MTE2", SEM_WS3_V2C)
                T.set_flag("V", "MTE2", SIG_IO_UB)
                T.set_flag("V", "MTE2", SIG_IO_UB + 1)
                T.set_flag("MTE3", "V", SIG_S_HALF)

                for t in T.serial(my_count):
                    task_id = my_start + t
                    bx = task_id % q_blocks
                    gb = (task_id // q_blocks) % G_batches
                    kg = (task_id // (q_blocks * G_batches)) % N_kv
                    bz = task_id // (q_blocks * G_batches * N_kv)
                    hq0 = kg * G + gb * 2

                    T.tile.fill(acc_o, 0.0)
                    T.tile.fill(sumexp, 0.0)
                    T.tile.fill(neg_sm, neg_sm_cap)

                    for k in T.serial(num_outer):
                        # same unified runtime kv_useful bound as the C scope
                        # (identical formula + inputs -> identical batch_iters
                        # sequence -> flag pairs stay balanced on both sides)
                        bound = (bx + 1) * block_M + skip_limit
                        kv_u_raw = (bound + block_N - 1) // block_N
                        kv_useful = T.if_then_else(kv_u_raw > kv_iters, kv_iters, kv_u_raw)
                        _remaining = kv_useful - k * num_stages
                        _rem_c = T.if_then_else(_remaining < 0, 0, _remaining)
                        batch_iters = T.if_then_else(_rem_c < num_stages, _rem_c, num_stages)

                        # --- phase 1: softmax batch (read ws_s, write ws_p) ---
                        # ci-grain cross waits/sets hoisted OUTSIDE the h-loop
                        # (one flag pair per i, covering BOTH heads' strips of
                        # the stacked tile — the C side produces/consumes the
                        # whole [M_stack, BN] tile per flag)
                        T.wait_cross_flag(SEM_WS2_C2V)
                        for i in T.serial(batch_iters):
                            if i % cross_interval == 0:
                                T.wait_cross_flag(SEM_WS1_C2V)
                            for h in T.serial(2):
                                cur = i % 2
                                prv = 1 - cur
                                idx = k * num_stages + i
                                io_side = h
                                row0 = h * block_M + vid * half

                                T.wait_flag("V", "MTE2", SIG_IO_UB + io_side)
                                T.copy(ws_s[cid, i, row0 : row0 + half, :], io_buf[io_side, :, :])
                                T.set_flag("MTE2", "V", SIG_IO_UB + io_side)

                                if use_mask:
                                    # 2-flag mask sync (varlen pattern): MTE2
                                    # loads mask_v only after the previous
                                    # (i,h)'s add consumed it — mask rows are
                                    # h-independent (mask depends on (s, j))
                                    T.set_flag("V", "MTE2", SIG_MASK)
                                    T.wait_flag("V", "MTE2", SIG_MASK)
                                    T.copy(
                                        Mask[
                                            bz,
                                            bx * block_M + vid * half : bx * block_M + vid * half + half,
                                            idx * block_N : (idx + 1) * block_N,
                                        ],
                                        mask_v,
                                    )
                                    T.set_flag("MTE2", "V", SIG_MASK)

                                T.wait_flag("MTE2", "V", SIG_IO_UB + io_side)
                                T.copy(io_buf[io_side, :, :], work_ub)
                                T.set_flag("V", "MTE2", SIG_IO_UB + io_side)

                                # online softmax (batched 2-phase form) —
                                # max-after-mask; per-head state slots
                                # (neg_sm idx h*2+cur, prv = h*2+prv)
                                if use_mask:
                                    T.wait_flag("MTE2", "V", SIG_MASK)
                                    T.tile.add(work_ub, work_ub, mask_v)
                                # clear=True is kept (compiles correctly on
                                # all builds; see the _gqa_kernel_expert
                                # note).
                                T.reduce_max(work_ub, neg_sm[h * 2 + cur, :, :], dim=-1)
                                T.tile.mul(neg_sm[h * 2 + cur, :, :], neg_sm[h * 2 + cur, :, :], -sm_scale)
                                T.tile.min(neg_sm[h * 2 + cur, :, :], neg_sm[h * 2 + cur, :, :], neg_sm[h * 2 + prv, :, :])

                                T.tile.broadcast(buf_2d, neg_sm[h * 2 + cur, :, :])
                                T.tile.axpy(buf_2d, work_ub, sm_scale)
                                T.tile.exp(work_ub, buf_2d)

                                T.reduce_sum(work_ub, sumexp_is[i * 2 + h, :, :], dim=-1)
                                T.tile.sub(r_factors[i * 2 + h, :, :], neg_sm[h * 2 + cur, :, :], neg_sm[h * 2 + prv, :, :])

                                # write probs -> ws_p (via acc_s_half)
                                T.wait_flag("MTE3", "V", SIG_S_HALF)
                                T.copy(work_ub, acc_s_half)
                                T.set_flag("V", "MTE3", SIG_S_HALF)

                                T.wait_flag("V", "MTE3", SIG_S_HALF)
                                T.copy(acc_s_half, ws_p[cid, i, row0 : row0 + half, :])
                                T.set_flag("MTE3", "V", SIG_S_HALF)
                            if (i + 1) % cross_interval == 0 or i == batch_iters - 1:
                                T.set_cross_flag("MTE3", SEM_WS2_V2C)

                        T.set_cross_flag("MTE2", SEM_WS1_V2C)

                        # --- phase 2: O accumulation batch (read ws_o) ---
                        for i in T.serial(batch_iters):
                            if i % cross_interval == 0:
                                T.wait_cross_flag(SEM_WS3_C2V)
                            for h in T.serial(2):
                                # per-head recurrence in the exact per-head
                                # order (rescale by r_factors[i], then add
                                # ws_o[i]'s contribution)
                                T.tile.exp(r_factors[i * 2 + h, :, :], r_factors[i * 2 + h, :, :])
                                T.tile.mul(sumexp[h, :, :], sumexp[h, :, :], r_factors[i * 2 + h, :, :])
                                T.tile.add(sumexp[h, :, :], sumexp[h, :, :], sumexp_is[i * 2 + h, :, :])
                                T.tile.broadcast(buf_2d, r_factors[i * 2 + h, :, :])
                                T.tile.mul(acc_o[h, :, :], acc_o[h, :, :], buf_2d)

                                io_side = h
                                row0 = h * block_M + vid * half
                                T.wait_flag("V", "MTE2", SIG_IO_UB + io_side)
                                T.copy(ws_o[cid, i, row0 : row0 + half, :], io_buf[io_side, :, :])
                                T.set_flag("MTE2", "V", SIG_IO_UB + io_side)

                                T.wait_flag("MTE2", "V", SIG_IO_UB + io_side)
                                T.copy(io_buf[io_side, :, :], work_ub)
                                T.set_flag("V", "MTE2", SIG_IO_UB + io_side)

                                T.tile.add(acc_o[h, :, :], acc_o[h, :, :], work_ub)

                        T.set_cross_flag("MTE2", SEM_WS3_V2C)

                    # --- final normalize + per-(h, vid) contiguous output box ---
                    # (head h's rows [bx*BM + vid*half, +half) are one
                    # contiguous [half, D] box in BHSD — runtime-base/
                    # static-extent copy, the pattern proven by the ws reads)
                    for h in T.serial(2):
                        # fully-masked-row guard (see
                        # _gqa_kernel_expert).
                        T.tile.add(sumexp[h, :, :], sumexp[h, :, :], 1e-30)
                        T.tile.broadcast(buf_2d, sumexp[h, :, :])
                        T.tile.div(acc_o[h, :, :], acc_o[h, :, :], buf_2d)
                        T.wait_flag("MTE3", "V", SIG_S_HALF)
                        T.copy(acc_o[h, :, :], acc_s_half)
                        T.set_flag("V", "MTE3", SIG_S_HALF)
                        T.wait_flag("V", "MTE3", SIG_S_HALF)
                        T.copy(
                            acc_s_half,
                            Output[bz, hq0 + h, bx * block_M + vid * half : bx * block_M + vid * half + half, :],
                        )
                        T.set_flag("MTE3", "V", SIG_S_HALF)

                # destroy: consume outstanding init-direction flags
                T.wait_flag("V", "MTE2", SIG_IO_UB)
                T.wait_flag("V", "MTE2", SIG_IO_UB + 1)
                T.wait_flag("MTE3", "V", SIG_S_HALF)

    return main


_ZERO_POOL = {}
_ZERO_POOL_INITIAL_NUMEL = 1 << 23  # 8M elements (16MB fp16) — covers any pad slab


def _zero_block(dtype, device, numel):
    """Resident zero slab.

    torch.full(0) / torch.zeros with fp16/bf16 on NPU lowers to
    aclnnInplaceFillScalar / ZerosLike, whose binaries are missing on some
    CANN 9.x SoC packages (error 561103).  fp32 Fill IS available (the
    causal Mask_full construction relies on it — the binary dispatches by
    dtype).  Keep ONE 1D zero slab per (dtype, device), built ONCE on CPU +
    a single .to(device) transfer during warmup — one host->device copy per
    process for the whole pool, instead of a device-side fill kernel per
    pad region — and view it into every pad region.  The view assignment
    is an NPU-side slice copy (ViewCopy), whose fp16/bf16 binary is
    missing on SOME CANN 9.x SoC packages (561103) and present on others
    (e.g. CANN 9.2.0); on packages without it this fallback path is
    unavailable — the gqa() main path pads in-kernel instead (see
    _pad_along).
    """
    key = (dtype, str(device))
    blk = _ZERO_POOL.get(key)
    if blk is None or blk.numel() < numel:
        n = max(_ZERO_POOL_INITIAL_NUMEL, numel)
        n = 1 << (n - 1).bit_length()  # round up to a power of two
        blk = torch.zeros(n, dtype=dtype).to(device)
        _ZERO_POOL[key] = blk
    return blk[:numel]


def _npu_sync_tensor(t):
    """d2d-copy -> kernel-launch race guard.

    The tilelang kernel launch is NOT reliably stream-ordered against
    preceding torch device-to-device copy ops (slice assigns / ViewCopy):
    driving the padded (non-aligned) path repeatedly produces
    nondeterministic, partially-NaN outputs with BIT-IDENTICAL kernel
    inputs (with fixed inputs, sync-before-launch runs bit-stable while
    omitting the sync corrupts the output).  Aligned shapes never insert
    torch ops between the transposes and the kernel and are unaffected.  A
    synchronize() on the padded path costs at most the tiny pad-copy time
    (the launch would wait for the copies anyway if the ordering held).
    """
    if t is not None and str(t.device).startswith("npu"):
        torch.npu.synchronize()


def _pad_along(t, dim, target_len):
    """Zero-pad tensor `t` along `dim` up to `target_len` (returns t if no pad).

    torch.cat([t, torch.zeros(..., device=npu)]) lowers to ZerosLike +
    ConcatD / Cat_1_SliceAiCore, all missing on some CANN 9.x SoC packages
    (561103).  torch.full(0, fp16/bf16, device=npu) itself lowers to
    aclnnInplaceFillScalar (FillAiCore) whose fp16/bf16 binary is ALSO
    missing on those packages (561103; fp32 Fill remains available — the
    binary dispatches by dtype).  New construction: torch.empty (pure
    allocation, no device kernel) + valid-region slice assignment (d2d
    ViewCopy) + pad-region slice assigned from the resident _ZERO_POOL
    slab (NPU-side slice copy) — zero Fill kernels on the fp16/bf16
    padding path.

    Only invoked for non-block-aligned shapes; performance impact is limited
    to the padding fallback cases.

    Risk note: this fallback relies on the device-side slice copy, so it is
    unavailable on the SoC packages missing its fp16/bf16 binary (561103).
    The gqa() main path is unaffected — it moves padding into the custom
    kernels (in-kernel zero-fill); this function only serves direct factory
    callers and non-contiguous inputs.
    """
    orig = t.shape[dim]
    if target_len <= orig:
        return t
    shape = list(t.shape)
    shape[dim] = target_len
    pad_len = target_len - orig
    out = torch.empty(shape, dtype=t.dtype, device=t.device)
    slab = [slice(None)] * t.dim()
    slab[dim] = slice(0, orig)
    out[tuple(slab)] = t
    pad = [slice(None)] * t.dim()
    pad[dim] = slice(orig, target_len)
    pad_shape = [s if i != dim else pad_len for i, s in enumerate(shape)]
    pad_numel = 1
    for s in pad_shape:
        pad_numel *= s
    out[tuple(pad)] = _zero_block(t.dtype, t.device, pad_numel).view(*pad_shape)
    return out


def _prefill_run_padded(kernel_fn, Q, K, V, Mask, B, S, S_kv, N_q, N_kv, D, S_padded, S_kv_padded, use_mask, crop=True):
    """Shared BHSD prefill runner: implicit row/col padding + output crop.

    Extracted verbatim from the per-head _run (identical padding
    semantics for both prefill kernels):
      - Q padded along dim 2 (S) with zeros -> padded rows have Q=0
      - K/V padded along dim 2 (S_kv) with zeros
      - Mask_full (_BIG_NEG filled, real [0:S, 0:S_kv] copied in) built ONLY
        when the kernel actually reads the mask (use_mask=True: causal or
        column padding — fast path); otherwise the unpadded
        mask is passed through (kernel never dereferences it)
    """
    pad_S = S_padded - S
    pad_Skv = S_kv_padded - S_kv

    if pad_S > 0 or pad_Skv > 0:
        # Pad Q along dim 2 (S) with zeros → padded rows have Q=0.
        # NO-OP on the gqa() adapter path: transpose_qkv already emits the
        # padded BHSD tensors (S_out == S_padded), so _pad_along returns the
        # tensor unchanged (`target_len <= orig`) — zero torch device-side
        # ops.  The calls are RETAINED for factory callers supplying
        # real-S BHSD tensors, where they still pad via the torch
        # slice-assign pattern.
        if pad_S > 0:
            Q = _pad_along(Q, 2, S_padded)
        # Pad K/V along dim 2 (S_kv) with zeros
        if pad_Skv > 0:
            K = _pad_along(K, 2, S_kv_padded)
            V = _pad_along(V, 2, S_kv_padded)
        # Only build the padded Mask_full when the kernel
        # actually reads the Mask (use_mask=True: causal or column
        # padding). For row-padding-only non-causal cases use_mask=False
        # and the kernel never dereferences Mask (both T.copy(Mask) and
        # the additive T.tile.add are compile-time eliminated), so the
        # ~15us FillScalar+ViewCopy construction is pure waste. The
        # unpadded [B, S, S_kv] Mask is passed through unchanged — the JIT
        # call chain does not validate input shapes against the declared
        # [B, S_padded, S_kv_padded], and the kernel never dereferences
        # the Mask on this path, so the output is unaffected.
        # Semantics match the existing fast path: non-causal callers must
        # supply an all-zero (ignored) mask.
        if use_mask:
            # _BIG_NEG is -1e30 (full-range-score-dominant); fp32
            # FillScalar is available even on the CANN 9.x SoC packages
            # missing the fp16/bf16 fill binaries (dispatch by dtype).
            Mask_full = torch.full(
                (B, S_padded, S_kv_padded),
                _BIG_NEG,
                dtype=Mask.dtype,
                device=Mask.device,
            )
            Mask_full[:, :S, :S_kv] = Mask
            Mask = Mask_full
        # pad/mask slice-assigns (torch d2d) must complete before the
        # tilelang launch — see _npu_sync_tensor.  On the gqa() adapter
        # path the transposes are pre-padded and the _pad_along calls above are
        # no-ops, but this sync STILL (a) orders the fp32 Mask_full
        # Fill+ViewCopy writes against the main kernel launch and (b)
        # belt-and-braces orders the padding transpose (tilelang) itself
        # against the main kernel launch on the padded path.
        _npu_sync_tensor(Q)

    Out_padded = kernel_fn(Q, K, V, Mask)
    # the output crop is consumed by transpose_out's .contiguous()
    # (torch d2d) — same ordering hazard on the way OUT; transpose_out
    # syncs inside its non-contiguous branch.
    # crop=False (gqa() adapter, non-aligned S): return the FULL padded
    # output — transpose_out(s_real=...) reads it directly in-kernel, so no
    # crop materialization (fp16/bf16 ViewCopy d2d, binary missing on some
    # CANN 9.x SoC packages).  Factory callers keep the crop view.
    if not crop:
        return Out_padded
    return Out_padded[:, :, :S, :]


# ─────────────────────────────────────────────────────────────────────
#  Compile fallback chain.  Every @tilelang.jit factory in this module
#  compiles EAGERLY at construction (JITKernel.__init__ ->
#  _compile_and_create_ adapter), so compile failures —
#  AscendMemoryPlanning envelope overflow (e.g. D=512 per-head
#  "acc_o_l0c required: 262144"), bisheng C++ failures (seen on some
#  tilelang builds only) — raise at FACTORY time, before any tensor
#  is launched.  Trying the next candidate there can never mask a runtime
#  or precision defect: those surface at invocation time and propagate
#  unchanged.
# ─────────────────────────────────────────────────────────────────────

_KERNEL_BUILD_CACHE = {}

# g2 as a compile-fallback candidate only (never primary — see the note
# at the prefill dispatch).
_ENABLE_G2_FALLBACK = True

# Per-head L0C static budget: acc_s_l0c [2,BM,BN] + acc_o_l0c [2,BM,D]
# fp32 must fit 192KB — the tightest envelope accepted across tilelang
# builds (D256@BM64 = exactly 192KB passes; D512@BM64 = 320KB fails
# everywhere).
_PERHEAD_L0C_BUDGET = 196608


def _perhead_bm_ladder(D, block_N=128):
    """block_M candidates for the per-head kernel, largest first."""
    return [bm for bm in (64, 32, 16) if (2 * bm * block_N + 2 * bm * D) * 4 <= _PERHEAD_L0C_BUDGET]


def _resolve_kernel_chain(chain_key, specs):
    """Build (compile) the first factory that succeeds.

    specs: ordered [(name, builder)]; builder() -> (kernel_fn, extra).
    Returns (kernel_fn, extra, name).  The resolution is cached per
    chain_key so a shape never re-pays failed-compile attempts.
    """
    cached = _KERNEL_BUILD_CACHE.get(chain_key)
    if cached is not None:
        return cached
    last_err = None
    for name, build in specs:
        try:
            kernel_fn, extra = build()
            _KERNEL_BUILD_CACHE[chain_key] = (kernel_fn, extra, name)
            return kernel_fn, extra, name
        except Exception as e:  # noqa: BLE001 — compile-phase only (see above)
            last_err = e
            print(
                f"[gqa][fallback] kernel '{name}' failed to compile ({type(e).__name__}: {str(e)[:200]}); trying next candidate", flush=True
            )
    raise last_err


def gqa_expert(B, S, S_kv, sm_scale, is_causal, N_q=8, N_kv=2, D=128, dtype="float16"):
    """GQA forward wrapper: pads inputs to block-aligned shapes, crops output.

    Public API. Returns a callable whose LAYOUT CONTRACT depends on S:

        S <= 4 (decode, BSND direct supply — no transposes):
            out_bsnd = gqa_expert(...)( Q_bsnd, K_bsnd, V_bsnd, Mask_b )
        S >  4 (prefill, BHSD — caller transposes):
            out_bhsd = gqa_expert(...)( Q_bhsd, K_bhsd, V_bhsd, Mask_b )

    Dispatch on S:
        - 1 <= S <= 4 → dense M-packing kernel: one
                    instance per (bz, kv-group), G_inner Q heads densely
                    packed into the GEMM M dim (KV traffic & instance count
                    ÷ G_inner). Q/Output use the real S (no row padding / no
                    crop). K/V ingested via strided reads straight from the
                    caller's BSND tensors; output written as s-major
                    contiguous boxes back to BSND.
        - S  > 4 → G_inner=2 M-stacking expert kernel when
                    use_expert_prefill (D==128, fp16/bf16, kv_iters==1 or
                    even >=2, causal S_kv>=S, B*N_q*q_blocks >= NUM_CORES)
                    AND G % 2 == 0 AND B*N_kv*(G//2)*q_blocks >=
                    NUM_CORES: one instance per (bz, kv-group, bx,
                    head-pair) — KV traffic and instance count ÷2.
        - S  > 4, stacking-ineligible (G odd, D!=128, instance-short,
                    odd kv_iters>1, causal S_kv<S) → per-head expert
                    kernel when use_expert_prefill holds, else the
                    per-head AUTO block_M ladder (causal skip; L0C
                    double buffer); g2 stays the last compile fallback.
    """
    G = N_q // N_kv
    block_N = 128
    S_kv_padded = ((S_kv + block_N - 1) // block_N) * block_N

    # Mask fast path: the additive mask only
    # needs to be applied when causal (real _BIG_NEG structure) or when KV columns
    # are padded (padded columns must be excluded from softmax: their K=0 rows
    # would otherwise contribute score 0 weight). Row padding alone does NOT
    # need a mask: padded Q rows are zeroed by the wrapper (scores = 0 ->
    # uniform softmax -> output = V mean), which is finite (no inf/nan) and the
    # padded rows are cropped from the output anyway.
    use_mask = bool(is_causal or (S_kv_padded != S_kv))

    if 1 <= S <= 4:
        # ========== decode dense M-packing, BSND direct ==========
        G_inner, G_batches, M_tile = _dense_tile_params(G, S, D)
        kv_iters = S_kv_padded // block_N

        # Expert manual-flag deep-pipeline dispatch guard: the expert
        # kernel is eligible for fp16/bf16 D==128 shapes with even kv_iters
        # >= 4, no tail G_batch (the factory-level scatter plan assumes full
        # G_inner segments) and enough instances to keep all NUM_CORES busy.
        # Faster than the AUTO dense kernel on decode shapes;
        # tiny/odd/tail shapes fall back to the AUTO dense kernel
        # (the expert form is net-negative below ~4 KV iterations or
        # ~20 instances).
        use_expert = (
            D == 128
            and dtype in ("float16", "bfloat16")
            and kv_iters >= 4
            and kv_iters % 2 == 0
            and (G % G_inner) == 0
            and (B * N_kv * G_batches) >= P28_NUM_CORES
        )
        # D256 decode expert (BN=64 refit): same eligibility shape as the
        # D128 expert but the batch unit is 64 columns (kv64 = S_kv_padded//64,
        # always even because S_kv_padded is a 128 multiple).  Faster
        # than the AUTO dense kernel on D256 decode shapes; the
        # BN=64 form beats the BN=128 shared-half variant by keeping the
        # 512B-row K/V segment geometry.
        use_expert_d256 = (
            D == 256
            and dtype in ("float16", "bfloat16")
            and (S_kv_padded // 64) >= 4
            and (G % G_inner) == 0
            and (B * N_kv * G_batches) >= P28_NUM_CORES
        )

        # Decode compile-fallback chain: the tuned expert kernel
        # first, then the AUTO dense kernel (structurally different codegen —
        # a factory compile failure on other tilelang builds falls through
        # instead of failing the case).
        specs = []
        if use_expert:
            ns = min(16, kv_iters)
            # tuned: coarse cross chunks amortize the C<->V round trip
            # for small tiles with deep KV; fine chunks keep M_tile=64 (or
            # shallow-KV) pipelines flowing
            ci = 4 if (kv_iters >= 16 and M_tile <= 32) else 2
            specs.append(
                (
                    "dense_expert",
                    lambda ns=ns, ci=ci: (
                        _gqa_kernel_dense_expert(
                            B,
                            S,
                            S_kv_padded,
                            sm_scale,
                            is_causal,
                            N_q=N_q,
                            N_kv=N_kv,
                            D=D,
                            dtype=dtype,
                            use_mask=use_mask,
                            block_N=block_N,
                            M_tile=M_tile,
                            G_inner=G_inner,
                            num_stages=ns,
                            cross_interval=ci,
                        ),
                        None,
                    ),
                )
            )
        if use_expert_d256:
            kv64 = S_kv_padded // 64
            ns = min(16, kv64)
            ci = 4 if (kv64 >= 16 and M_tile <= 32) else 2
            specs.append(
                (
                    "dense_expert_d256",
                    lambda ns=ns, ci=ci: (
                        _gqa_kernel_dense_expert_d256(
                            B,
                            S,
                            S_kv_padded,
                            sm_scale,
                            is_causal,
                            N_q=N_q,
                            N_kv=N_kv,
                            D=D,
                            dtype=dtype,
                            use_mask=use_mask,
                            block_N=64,
                            M_tile=M_tile,
                            G_inner=G_inner,
                            num_stages=ns,
                            cross_interval=ci,
                        ),
                        None,
                    ),
                )
            )
        specs.append(
            (
                "dense_auto",
                lambda: (
                    _gqa_kernel_dense(
                        B,
                        S,
                        S_kv_padded,
                        sm_scale,
                        is_causal,
                        N_q=N_q,
                        N_kv=N_kv,
                        D=D,
                        dtype=dtype,
                        skv_minus_s=S_kv - S,  # original (pre-padding) causal boundary
                        use_mask=use_mask,
                        block_N=block_N,
                        M_tile=M_tile,
                        G_inner=G_inner,
                    ),
                    None,
                ),
            )
        )
        chain_key = ("decode", B, S, S_kv, N_q, N_kv, D, dtype, is_causal, sm_scale)
        kernel_fn, _, _chosen = _resolve_kernel_chain(chain_key, specs)

        def _run(Q, K, V, Mask):
            """Execute the dense M-packing kernel (S<=4 decode path).

            BSND layout contract — the caller's original
            [B, S, N_q, D] / [B, S_kv, N_kv, D] tensors are passed straight
            through (no BSND<->BHSD transposes anywhere on the decode path):

            Q:      [B, S, N_q, D]    BSND, real S
            K/V:    [B, S_kv, N_kv, D] BSND (column-padded here when needed)
            Mask:   [B, S, S_kv]      float32 additive, 0=visible/_BIG_NEG=masked
            returns [B, S, N_q, D]    BSND, real S (no crop)

            K/V are column-padded to S_kv_padded along dim 1.  When the
            kernel reads the mask (causal / column padding), mask rows are
            replicated s-major — packed row s*G_inner + g_i uses mask row s
            (the mask depends only on (s, j), shared across all G_inner head
            segments).
            """
            pad_Skv = S_kv_padded - S_kv
            if pad_Skv > 0:
                # Pad K/V along dim 1 (S_kv) via the dedicated 2-in-1
                # tilelang pad kernel — a plain _pad_along
                # slice-assign lowers to torch d2d ViewCopy, whose
                # fp16/bf16 binary is missing on some CANN 9.x SoC packages
                # (561103).  decode + non-aligned S_kv variants hit this
                # path.
                K, V = _pad_kv_npu(K, V, S_kv_padded)
                # tilelang pad kernel -> main kernel launch
                # ordering (see _npu_sync_tensor); the blocking mask H2D
                # below re-covers this when use_mask=True, but the sync is
                # kept unconditionally as belt-and-braces.
                _npu_sync_tensor(K)
            if use_mask:
                # torch.zeros(B, M_tile, S_kv_padded, device=device)
                # lowers to the ZerosLike / aclnnInplaceZero binary, which
                # is missing on some CANN 9.x SoC packages (error 561103),
                # so the NPU-side zeros build crashes the kernel launch.
                #
                # Strategy: build the mask on CPU (torch.zeros
                # on CPU uses a host allocator — no NPU binary needed),
                # then transfer via ONE clean .to(device) copy.  The
                # adapter already skips _build_causal_mask
                # for the decode path (uses _dummy_mask), so a
                # decode+causal call performs EXACTLY ONE host->device
                # mask transfer — the same single-transfer pattern as
                # the prefill _build_causal_mask.
                #
                # Missing-binary list on such SoC packages:
                # BroadcastToAiCore, GatherV3, Cat_1_SliceAiCore, ConcatD,
                # ZerosLike.  All avoided below (only torch.zeros on CPU +
                # slice assignment + .to(device)).
                if is_causal:
                    # [B, M_tile, S_kv_padded] replicated causal mask
                    # on CPU, single H2D transfer.
                    skv_minus_s = S_kv - S
                    mask_cpu = torch.zeros(
                        B,
                        M_tile,
                        S_kv_padded,
                        dtype=torch.float32,
                    )
                    for s in range(S):
                        mask_start_j = s + skv_minus_s + 1
                        # clamp: a negative start (decode S > S_kv)
                        # must mask the WHOLE row, not wrap around Python
                        # slice semantics
                        mask_start_j = max(0, mask_start_j)
                        for g in range(G_inner):
                            row_idx = s * G_inner + g
                            if mask_start_j < S_kv:
                                mask_cpu[:, row_idx, mask_start_j:S_kv] = _BIG_NEG
                            if S_kv < S_kv_padded:
                                mask_cpu[:, row_idx, S_kv:] = _BIG_NEG
                    # pad rows [S * G_inner, M_tile) stay 0: kernel discards them
                    # Single contiguous host->device transfer (the same
                    # one-transfer pattern as the prefill _build_causal_mask).
                    Mask = mask_cpu.to(Q.device)
                else:
                    # Non-causal column-padding path: replicated mask built
                    # on CPU + single H2D (avoids ZerosLike).  Real columns
                    # visible (0); the zero-filled K/V pad columns
                    # [S_kv, S_kv_padded) MUST carry _BIG_NEG — unmasked pad
                    # columns score 0 and their exp(0-shift) weight inflates
                    # the softmax denominator (systematic output shrink
                    # proportional to the pad ratio).  Mirrors the causal
                    # branch's pad-column handling above.  Pad rows
                    # [S * G_inner, M_tile) need no special value: the
                    # scatter plan never writes them back to Output.
                    mask_cpu = torch.zeros(
                        B,
                        M_tile,
                        S_kv_padded,
                        dtype=torch.float32,
                    )
                    if S_kv < S_kv_padded:
                        mask_cpu[:, :, S_kv:] = _BIG_NEG
                    Mask = mask_cpu.to(Q.device)  # blocking H2D: also covers the pad copies
            # (no separate `elif pad_Skv > 0` sync branch is needed:
            #  column padding implies use_mask=True, and the pad kernel sync
            #  above already covers the padded no-mask corner.)
            return kernel_fn(Q, K, V, Mask)

        return _run

    # ================ prefill path (S > 4) ================
    # ELIGIBLE shapes (fp16/bf16, D==128, kv_iters==1 or even
    # kv_iters>=2, causal requires S_kv>=S, >= 20 per-head instances
    # B*N_q*q_blocks) dispatch to _gqa_kernel_prefill_expert: a fixed-20-core
    # (MIX_AIC_1_2) manual-flag deep-pipeline per-head kernel (C/V scope
    # separation, MTE2<->MTE1<->M<->FIX directed flag chains, k/v L1 double
    # buffering, GM-workspace ns-deep C/V decoupling, 2-phase batched online
    # softmax with max-after-mask semantics; runtime kv_useful loop-bound
    # causal skip — prefix-block semantics identical to the g2 skip
    # predicate).  Faster than the AUTO kernels on prefill shapes at
    # ns=kv_iters, ci=2.
    # kv_iters==1 small prefill also routes here at ns=1 (faster there
    # than the g2/per-head AUTO kernels; odd kv>1, e.g. Skv=300, still
    # falls back).  Ineligible shapes (D=256, causal S_kv<S, <20 instances)
    # keep the existing paths below.
    block_M = 64
    S_padded = ((S + block_M - 1) // block_M) * block_M
    kv_iters = S_kv_padded // block_N
    q_blocks = S_padded // block_M
    use_expert_prefill = (
        D == 128
        and dtype in ("float16", "bfloat16")
        and (kv_iters == 1 or (kv_iters >= 2 and kv_iters % 2 == 0))
        and ((not is_causal) or (S_kv >= S))
        and (B * N_q * q_blocks) >= P28_NUM_CORES
    )

    # G_even expert-eligible shapes with enough STACKED
    # instances (B*N_kv*(G/2)*q_blocks >= NUM_CORES) dispatch to the
    # G_inner=2 M-stacked expert kernel instead: one task serves the 2 Q
    # heads of one (bz, kg, head-pair) at the same bx — K/V/mask GM traffic,
    # task count and per-task C<->V sync packets all /2 (l0a 64KB / l0c
    # 131072B exactly at their envelopes).  Consistently faster than the
    # per-head expert on stacked-eligible shapes
    # (flat within the noise band on kv=1 causal small shapes).
    # G odd / G==1 / stacked-instance-short shapes keep the per-head expert
    # below.
    use_g2_stack = use_expert_prefill and (G % 2) == 0 and (B * N_kv * (G // 2) * q_blocks) >= P28_NUM_CORES

    # ============ prefill compile-fallback chain ============
    # Ordered candidates: the tuned primary FIRST (zero behavior
    # change on any shape whose primary compiles), then
    # structurally different kernels as compile fallbacks for planner /
    # compiler differences across tilelang builds.  Every builder returns
    # (kernel_fn, S_padded) — the per-head ladder's smaller block_M needs
    # its own row padding.
    specs = []

    if use_g2_stack:
        ns = min(kv_iters, 8)
        specs.append(
            (
                "g2stack",
                lambda ns=ns: (
                    _gqa_kernel_prefill_g2stack(
                        B,
                        S_padded,
                        S_kv_padded,
                        sm_scale,
                        is_causal,
                        N_q=N_q,
                        N_kv=N_kv,
                        D=D,
                        dtype=dtype,
                        block_M=block_M,
                        skv_minus_s=S_kv - S,  # original (pre-padding) causal boundary
                        use_mask=use_mask,
                        block_N=block_N,
                        num_stages=ns,
                        cross_interval=2,
                    ),
                    S_padded,
                ),
            )
        )

    if use_expert_prefill:
        # tuned: ns = kv_iters (single full batch — one
        # deep-pipeline wave maximizes C/V overlap; capped at 8), ci = 2
        # (finer chunks keep the bigger V-side prefill work flowing; ci=1
        # over-synchronizes, ci=4 starves V).  kv_iters==1 -> ns=1 (odd
        # allowed in the single-batch form only).
        ns = min(kv_iters, 8)
        specs.append(
            (
                "prefill_expert",
                lambda ns=ns: (
                    _gqa_kernel_prefill_expert(
                        B,
                        S_padded,
                        S_kv_padded,
                        sm_scale,
                        is_causal,
                        N_q=N_q,
                        N_kv=N_kv,
                        D=D,
                        dtype=dtype,
                        block_M=block_M,
                        skv_minus_s=S_kv - S,  # original (pre-padding) causal boundary
                        use_mask=use_mask,
                        block_N=block_N,
                        num_stages=ns,
                        cross_interval=2,
                    ),
                    S_padded,
                ),
            )
        )

    # ---- per-head ladder (primary for NOT-expert-eligible shapes: D!=128,
    # odd kv_iters>1, causal S_kv<S, tiny instance counts, any G).  The
    # block_M is L0C-budget-gated: D=512@BM64 needs 320KB > the 192KB
    # envelope (compile error "acc_o_l0c required: 262144"), so the
    # ladder starts at the largest BM that fits; smaller BMs are retained
    # as further compile fallbacks.
    # Every ladder entry compiles with the UNIFIED S_padded
    # (ceil(S/64)*64, a multiple of 64/32/16) instead of its own
    # ceil(S/bm)*bm: the transpose emits Q padded to exactly this
    # length, and a smaller per-bm sp would declare tensor strides the
    # padded input does not match.  Extra rows are zero-filled by the
    # transpose and cropped from the output (correct, slightly more work
    # on fallback-only shapes). ----
    ladder = _perhead_bm_ladder(D, block_N)
    for bm in ladder:
        specs.append(
            (
                f"per_head_bm{bm}",
                (
                    lambda bm=bm: (
                        _gqa_kernel_expert(
                            B,
                            S_padded,
                            S_kv_padded,
                            sm_scale,
                            is_causal,
                            N_q=N_q,
                            N_kv=N_kv,
                            D=D,
                            dtype=dtype,
                            block_M=bm,
                            skv_minus_s=S_kv - S,  # original (pre-padding) causal boundary
                            use_mask=use_mask,
                            block_N=block_N,
                        ),
                        S_padded,
                    )
                ),
            )
        )

    # ---- g2 as the LAST compile fallback (never primary): its
    # bisheng C++ compile fails on some other tilelang builds and its
    # max-after-mask + init-floor softmax was reworked; eligible for
    # D<=128, G>=2.
    # Only reached when every candidate above failed to compile. ----
    if _ENABLE_G2_FALLBACK and G >= 2 and D <= 128:
        specs.append(
            (
                "g2",
                lambda: (
                    _gqa_kernel_prefill_g2(
                        B,
                        S_padded,
                        S_kv_padded,
                        sm_scale,
                        is_causal,
                        N_q=N_q,
                        N_kv=N_kv,
                        D=D,
                        dtype=dtype,
                        block_M=block_M,
                        skv_minus_s=S_kv - S,  # original (pre-padding) causal boundary
                        use_mask=use_mask,
                        block_N=block_N,
                    ),
                    S_padded,
                ),
            )
        )

    chain_key = ("prefill", B, S, S_kv, N_q, N_kv, D, dtype, is_causal, sm_scale)
    kernel_fn, S_padded_eff, chosen = _resolve_kernel_chain(chain_key, specs)

    def _run(Q, K, V, Mask, crop=True):
        """Execute the chosen prefill kernel (BHSD contract).

        Q: [B, N_q, S, D]  BHSD
        K: [B, N_kv, S_kv, D]  BHSD
        V: [B, N_kv, S_kv, D]  BHSD
        Mask: [B, S, S_kv]  float32, additive: 0=visible, _BIG_NEG=masked
        Returns [B, N_q, S, D] BHSD (cropped to real S).  crop=False
        returns the FULL padded [B, N_q, S_padded, D] (the gqa() adapter
        uses it with transpose_out(s_real=...) to skip the crop
        materialization).
        """
        return _prefill_run_padded(
            kernel_fn,
            Q,
            K,
            V,
            Mask,
            B,
            S,
            S_kv,
            N_q,
            N_kv,
            D,
            S_padded_eff,
            S_kv_padded,
            use_mask,
            crop=crop,
        )

    return _run


# ─────────────────────────────────────────────────────────────────────
#  Custom BSND<->BHSD transposes
# ─────────────────────────────────────────────────────────────────────

_TRANSPOSE_PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}

_VEC_NUM = 2
# 192KB per-core UB pool — the HOST-SIDE tiling budget constant used by
# _ub_rows() for the transpose/pad kernels.  Distinct quantity from the
# 196,352B shared.ub pool ceiling quoted in the g2 kernel's fused-form
# note (that one is what the g2 kernel's UB allocations plan against).
_UB_BYTES = 196608

_transpose_kernel_cache = {}


def _ceildiv(a, b):
    return (a + b - 1) // b


def _ub_rows(R):
    """Max tile rows of [bm, R] fitting the UB budget (2 bytes/elem)."""
    return max(1, _UB_BYTES // (R * 2))


# ---------------------------------------------------------------------------
# 3-in-1: Q/K/V BSND -> BHSD in one launch
# ---------------------------------------------------------------------------
@tilelang.jit(out_idx=[3, 4, 5], pass_configs=_TRANSPOSE_PASS_CONFIGS)
def _transpose_qkv(B, Sq, Nq, Skv, Nkv, R, q_bm, q_v1, q_bn, kv_bm, dtype, tail=False, Sq_out=None, Skv_out=None):
    # NOTE: Python `if` statements inside the prim_func body become TIR
    # IfThenElse blocks whose locals are branch-scoped — allocations MUST be
    # emitted unconditionally at kernel-body scope, and
    # every closure value referenced from a branch must be defined for BOTH
    # trace outcomes (dead constant-false branches are not traced at all).
    #
    # tail mode: shapes where S / S_kv are not divisible by any whole-block
    # bm would otherwise force the host back to aclnn Transpose (binary
    # missing on some CANN 9.x SoC packages, 561103).
    # tail=True switches the whole-block V2 copies to per-row GUARDED copies
    # (the exact _transpose_v1 pattern: scalar runtime row read + runtime-
    # base [1, R] write), so any S / S_kv is covered.  Input tensors are
    # viewed 3D (B*rows, heads, R) in tail mode so each row copy is a [1, R]
    # 2D region.  tail=False keeps the aligned whole-block path byte-for-byte
    # (zero impact on aligned shapes; the tail flag is part of the host cache
    # key).
    #
    # padded-output mode:
    # Sq_out / Skv_out (>= Sq / Skv) make the kernel emit the BLOCK-ALIGNED
    # BHSD outputs directly — output rows in [Sq, Sq_out) / [Skv, Skv_out)
    # are zero-filled IN-KERNEL (T.tile.fill on the staging tile + the same
    # [1, R] / [q_bn, R] row write), so the fp16/bf16 padding needs ZERO
    # torch device-side ops (a plain _pad_along slice-assign lowers
    # to ViewCopyAiCore, whose fp16/bf16 binary is missing on some CANN 9.x
    # SoC packages, 561103).  Only the per-row segments (tail, q_v1 Q) can
    # write pad rows; the host dispatch never routes a pad-needing segment
    # to a whole-block mode.  Pad code is guarded by the Python-level
    # constants pad_q/pad_kv (dead constant-false branches are not traced),
    # so no-pad traces are IDENTICAL to the kernel without pad support.
    Sq_out = Sq if Sq_out is None else Sq_out
    Skv_out = Skv if Skv_out is None else Skv_out
    pad_q = Sq_out - Sq
    pad_kv = Skv_out - Skv
    q_m = _ceildiv(Sq_out, q_bm)
    q_n = _ceildiv(Nq, q_bn) if q_v1 else 1
    if q_v1:
        q_total = B * q_m * q_n
    else:
        q_total = B * Nq * q_m
    kv_m = _ceildiv(Skv_out, kv_bm)
    kv_total = B * Nkv * kv_m
    total = q_total + 2 * kv_total
    block_num = _ceildiv(total, _VEC_NUM)

    q_in_shape = (B * Sq, Nq, R) if (q_v1 or tail) else (B * Sq, Nq * R)
    q_out_shape = (B * Nq, Sq_out * R) if q_v1 else (B * Nq, Sq_out, R)
    kv_in_shape = (B * Skv, Nkv, R) if tail else (B * Skv, Nkv * R)
    # in tail mode every copy stages through the [1, R] row buffer — the
    # whole-block `ub` tile is dead, shrink it to (1, R) to spare UB budget
    ub_rows_alloc = (kv_bm if q_v1 else q_bm) if not tail else 1

    @T.prim_func
    def kernel(
        qx: T.Tensor(q_in_shape, dtype),  # type: ignore
        kx: T.Tensor(kv_in_shape, dtype),  # type: ignore
        vx: T.Tensor(kv_in_shape, dtype),  # type: ignore
        qy: T.Tensor(q_out_shape, dtype),  # type: ignore
        ky: T.Tensor((B * Nkv, Skv_out, R), dtype),  # type: ignore
        vy: T.Tensor((B * Nkv, Skv_out, R), dtype),  # type: ignore
    ):
        with T.Kernel(block_num, is_npu=True) as (cid, vid):
            task = cid * _VEC_NUM + vid
            # unconditional top-scope allocations (branch-exclusive tasks
            # share them; q_v1=False uses one shared [q_bm, R] tile — the
            # host passes q_bm == kv_bm on that path)
            ub_q = T.alloc_ub((q_bn if q_v1 else 1, R), dtype)
            ub = T.alloc_ub((ub_rows_alloc, R), dtype)
            # per-row staging tile used by ALL tail-mode copies
            ub_row = T.alloc_ub((1, R), dtype)
            if task < total:
                if task < q_total:
                    if q_v1:
                        t = task
                        b = t // (q_m * q_n)
                        rem = t % (q_m * q_n)
                        bmb = rem // q_n
                        bnb = rem % q_n
                        with T.Scope("V"):
                            for mi in T.serial(q_bm):
                                m_idx = bmb * q_bm + mi
                                if m_idx < Sq:
                                    T.copy(
                                        qx[b * Sq + m_idx, bnb * q_bn : bnb * q_bn + q_bn, :],
                                        ub_q,
                                    )
                                    T.copy(
                                        ub_q,
                                        qy[
                                            b * Nq + bnb * q_bn : b * Nq + bnb * q_bn + q_bn,
                                            m_idx * R : (m_idx + 1) * R,
                                        ],
                                    )
                                else:
                                    # zero-fill the Q pad rows in-kernel.
                                    # Nested on purpose (noqa SIM102): the outer guard is a
                                    # trace-time Python constant, the inner one a runtime TIR
                                    # predicate — merging them would feed a mixed Python/TIR `and`
                                    # to the TVMScript parser and change this kernel's emitted code.
                                    if pad_q > 0:  # noqa: SIM102
                                        if m_idx < Sq_out:
                                            T.tile.fill(ub_q, 0.0)
                                            T.copy(
                                                ub_q,
                                                qy[
                                                    b * Nq + bnb * q_bn : b * Nq + bnb * q_bn + q_bn,
                                                    m_idx * R : (m_idx + 1) * R,
                                                ],
                                            )
                    elif tail:
                        # V2 Q segment, per-row guarded copies
                        t = task
                        b = t // (Nq * q_m)
                        rem = t % (Nq * q_m)
                        n = rem // q_m
                        bmb = rem % q_m
                        with T.Scope("V"):
                            for mi in T.serial(q_bm):
                                m_idx = bmb * q_bm + mi
                                if m_idx < Sq:
                                    T.copy(
                                        qx[b * Sq + m_idx, n : n + 1, :],
                                        ub_row,
                                    )
                                    T.copy(
                                        ub_row,
                                        qy[b * Nq + n, m_idx : m_idx + 1, :],
                                    )
                                else:
                                    # zero-fill the Q pad rows in-kernel.
                                    # Nested on purpose (noqa SIM102): the outer guard is a
                                    # trace-time Python constant, the inner one a runtime TIR
                                    # predicate — merging them would feed a mixed Python/TIR `and`
                                    # to the TVMScript parser and change this kernel's emitted code.
                                    if pad_q > 0:  # noqa: SIM102
                                        if m_idx < Sq_out:
                                            T.tile.fill(ub_row, 0.0)
                                            T.copy(
                                                ub_row,
                                                qy[b * Nq + n, m_idx : m_idx + 1, :],
                                            )
                    else:
                        t = task
                        b = t // (Nq * q_m)
                        rem = t % (Nq * q_m)
                        n = rem // q_m
                        bmb = rem % q_m
                        m0 = bmb * q_bm
                        T.copy(
                            qx[b * Sq + m0 : b * Sq + m0 + q_bm, n * R : (n + 1) * R],
                            ub,
                        )
                        T.copy(ub, qy[b * Nq + n, m0 : m0 + q_bm, :])
                else:
                    t = task - q_total
                    if t < kv_total:
                        b = t // (Nkv * kv_m)
                        rem = t % (Nkv * kv_m)
                        n = rem // kv_m
                        bmb = rem % kv_m
                        if tail:
                            # K segment, per-row guarded copies
                            with T.Scope("V"):
                                for mi in T.serial(kv_bm):
                                    m_idx = bmb * kv_bm + mi
                                    if m_idx < Skv:
                                        T.copy(
                                            kx[b * Skv + m_idx, n : n + 1, :],
                                            ub_row,
                                        )
                                        T.copy(
                                            ub_row,
                                            ky[b * Nkv + n, m_idx : m_idx + 1, :],
                                        )
                                    else:
                                        # zero-fill the K pad rows.
                                        # Nested on purpose (noqa SIM102): see the Q pad branch.
                                        if pad_kv > 0:  # noqa: SIM102
                                            if m_idx < Skv_out:
                                                T.tile.fill(ub_row, 0.0)
                                                T.copy(
                                                    ub_row,
                                                    ky[b * Nkv + n, m_idx : m_idx + 1, :],
                                                )
                        else:
                            m0 = bmb * kv_bm
                            T.copy(
                                kx[b * Skv + m0 : b * Skv + m0 + kv_bm, n * R : (n + 1) * R],
                                ub,
                            )
                            T.copy(ub, ky[b * Nkv + n, m0 : m0 + kv_bm, :])
                    else:
                        t = t - kv_total
                        b = t // (Nkv * kv_m)
                        rem = t % (Nkv * kv_m)
                        n = rem // kv_m
                        bmb = rem % kv_m
                        if tail:
                            # V segment, per-row guarded copies
                            with T.Scope("V"):
                                for mi in T.serial(kv_bm):
                                    m_idx = bmb * kv_bm + mi
                                    if m_idx < Skv:
                                        T.copy(
                                            vx[b * Skv + m_idx, n : n + 1, :],
                                            ub_row,
                                        )
                                        T.copy(
                                            ub_row,
                                            vy[b * Nkv + n, m_idx : m_idx + 1, :],
                                        )
                                    else:
                                        # zero-fill the V pad rows.
                                        # Nested on purpose (noqa SIM102): see the Q pad branch.
                                        if pad_kv > 0:  # noqa: SIM102
                                            if m_idx < Skv_out:
                                                T.tile.fill(ub_row, 0.0)
                                                T.copy(
                                                    ub_row,
                                                    vy[b * Nkv + n, m_idx : m_idx + 1, :],
                                                )
                        else:
                            m0 = bmb * kv_bm
                            T.copy(
                                vx[b * Skv + m0 : b * Skv + m0 + kv_bm, n * R : (n + 1) * R],
                                ub,
                            )
                            T.copy(ub, vy[b * Nkv + n, m0 : m0 + kv_bm, :])

    return kernel


# ---------------------------------------------------------------------------
# V1 single-tensor transpose (output direction BHSD -> BSND)
# ---------------------------------------------------------------------------
@tilelang.jit(out_idx=[1], pass_configs=_TRANSPOSE_PASS_CONFIGS)
def _transpose_v1(B, M, N, R, bm, bn, dtype, n_pad=None):
    """[B, M, N, R] -> [B, N, M, R] via contiguous reads + strided writes.

    x viewed (B*M, N, R); y viewed (B*N, M*R).  Task (b, bm-block, bn-block);
    serial inner loop over bm rows: contiguous [bn, R] read, strided [bn, R]
    write (dstGap M*R - R).  Tail-safe on the M dim (m_idx < M guard).

    n_pad (>= N): the INPUT tensor is padded
    along the N dim — x is declared (B*M, n_pad, R) so the reads use the
    padded strides, while the task grid and the y writes cover only the REAL
    N rows (the pad rows are never read).  This lets the output-direction
    transpose consume the padded kernel output DIRECTLY — no crop
    materialization (Out_padded[:, :, :S, :].contiguous() lowers to a
    fp16/bf16 ViewCopy d2d whose binary is missing on some CANN 9.x SoC
    packages, 561103).  n_pad == N traces identically to the no-pad form.
    """
    n_pad = N if n_pad is None else n_pad
    m_num = _ceildiv(M, bm)
    n_num = _ceildiv(N, bn)
    total = B * m_num * n_num
    block_num = _ceildiv(total, _VEC_NUM)

    @T.prim_func
    def kernel(
        x: T.Tensor((B * M, n_pad, R), dtype),  # type: ignore
        y: T.Tensor((B * N, M * R), dtype),  # type: ignore
    ):
        with T.Kernel(block_num, is_npu=True) as (cid, vid):
            task = cid * _VEC_NUM + vid
            ub = T.alloc_ub((bn, R), dtype)
            if task < total:
                b = task // (m_num * n_num)
                rem = task % (m_num * n_num)
                bmb = rem // n_num
                bnb = rem % n_num
                with T.Scope("V"):
                    for mi in T.serial(bm):
                        m_idx = bmb * bm + mi
                        if m_idx < M:
                            T.copy(
                                x[b * M + m_idx, bnb * bn : bnb * bn + bn, :],
                                ub,
                            )
                            T.copy(
                                ub,
                                y[
                                    b * N + bnb * bn : b * N + bnb * bn + bn,
                                    m_idx * R : (m_idx + 1) * R,
                                ],
                            )

    return kernel


# ---------------------------------------------------------------------------
# decode BSND dim-1 pad
# ---------------------------------------------------------------------------
@tilelang.jit(out_idx=[2, 3], pass_configs=_TRANSPOSE_PASS_CONFIGS)
def _pad_kv(B, S_in, S_out, N, R, bm, dtype):
    """[B, S_in, N, R] -> K'/V' [B, S_out, N, R], zero rows [S_in, S_out).

    2-in-1 (K and V in ONE launch — same task-grid pattern as
    _transpose_qkv's K/V segments).  Valid rows (m < S_in) stage through a
    [N, R] ub tile; pad rows are zero-filled in-kernel (T.tile.fill + the
    same row write), so the fp16/bf16 decode padding path needs ZERO torch
    device-side ops: a plain _pad_along slice-assign lowers to
    ViewCopyAiCore whose fp16/bf16 binary is missing on some CANN 9.x SoC
    packages (561103).
    """
    m_num = _ceildiv(S_out, bm)
    half = B * m_num
    total = 2 * half
    block_num = _ceildiv(total, _VEC_NUM)

    @T.prim_func
    def kernel(
        kx: T.Tensor((B * S_in, N, R), dtype),  # type: ignore
        vx: T.Tensor((B * S_in, N, R), dtype),  # type: ignore
        ky: T.Tensor((B * S_out, N, R), dtype),  # type: ignore
        vy: T.Tensor((B * S_out, N, R), dtype),  # type: ignore
    ):
        with T.Kernel(block_num, is_npu=True) as (cid, vid):
            task = cid * _VEC_NUM + vid
            ub = T.alloc_ub((N, R), dtype)
            if task < total:
                if task < half:
                    b = task // m_num
                    bmb = task % m_num
                    with T.Scope("V"):
                        for mi in T.serial(bm):
                            m_idx = bmb * bm + mi
                            if m_idx < S_out:
                                if m_idx < S_in:
                                    T.copy(kx[b * S_in + m_idx, :, :], ub)
                                else:
                                    T.tile.fill(ub, 0.0)
                                T.copy(ub, ky[b * S_out + m_idx, :, :])
                else:
                    t = task - half
                    b = t // m_num
                    bmb = t % m_num
                    with T.Scope("V"):
                        for mi in T.serial(bm):
                            m_idx = bmb * bm + mi
                            if m_idx < S_out:
                                if m_idx < S_in:
                                    T.copy(vx[b * S_in + m_idx, :, :], ub)
                                else:
                                    T.tile.fill(ub, 0.0)
                                T.copy(ub, vy[b * S_out + m_idx, :, :])

    return kernel


_pad_kv_kernel_cache = {}


def _pad_kv_npu(K, V, S_kv_padded):
    """Pad decode K/V [B, S_kv, N_kv, D] -> [B, S_kv_padded, N_kv, D] (both
    tensors, ONE tilelang launch).  Replaces the two _pad_along
    slice-assigns (torch d2d ViewCopy — fp16/bf16 binary missing on some
    CANN 9.x SoC packages, 561103).  Non-contiguous inputs fall back to
    _pad_along.
    """
    if not (K.is_contiguous() and V.is_contiguous()):
        return (_pad_along(K, 1, S_kv_padded), _pad_along(V, 1, S_kv_padded))
    B, S_kv, N, R = K.shape
    dtype_str = _dtype_str(K.dtype)
    if dtype_str is None:
        return (_pad_along(K, 1, S_kv_padded), _pad_along(V, 1, S_kv_padded))
    bm = max(1, min(64, _ub_rows(N * R)))
    key_ = ("padkv", B, S_kv, S_kv_padded, N, R, dtype_str)
    if key_ not in _pad_kv_kernel_cache:
        _pad_kv_kernel_cache[key_] = _pad_kv(B, S_kv, S_kv_padded, N, R, bm, dtype_str)
    fn = _pad_kv_kernel_cache[key_]
    ky, vy = fn(K.view(B * S_kv, N, R), V.view(B * S_kv, N, R))
    return (ky.view(B, S_kv_padded, N, R), vy.view(B, S_kv_padded, N, R))


# ---------------------------------------------------------------------------
# host-side dispatch
# ---------------------------------------------------------------------------
def _dtype_str(t):
    if t == torch.float16:
        return "float16"
    if t == torch.bfloat16:
        return "bfloat16"
    return None


def _shared_bm(S, Skv, R):
    """Shared V2 block size for the Q+KV segments (single [bm, R] ub)."""
    bm = min(512, S, Skv, _ub_rows(R))
    # require exact divisibility (no tail support in the V2 segment)
    while bm > 16 and (S % bm != 0 or Skv % bm != 0):
        bm //= 2
    if S % bm != 0 or Skv % bm != 0:
        return None
    return bm


def _tail_bm(R):
    """Fixed block size for the tail-mode (per-row) transpose path."""
    return max(16, min(128, _ub_rows(R)))


def transpose_qkv(query, key, value, S_out=None, S_kv_out=None):
    """BSND [B,S,Nq,D] / [B,Skv,Nkv,D] -> BHSD triple in ONE kernel launch.

    Returns (q_bhsd, k_bhsd, v_bhsd) or None when the shape/dtype is not
    covered by the dispatch rules (caller falls back to aclnn).

    Shapes where S / S_kv are
    not divisible by a shared whole-block bm would otherwise return None and
    the caller would fall back to aclnn Transpose — whose binary is MISSING
    on some CANN 9.x SoC packages (error 561103).  Such shapes route to the
    tail mode of the SAME kernel (fixed bm + per-row guarded copies), so the
    aclnn fallback is never taken for fp16/bf16 shapes with D*2 % 32 == 0
    and S > 4.

    S_out / S_kv_out (when given, >= S / S_kv) make the outputs come out
    BLOCK-ALIGNED PADDED — [B,Nq,S_out,D] / [B,Nkv,S_kv_out,D] with zero pad
    rows written in-kernel — eliminating every torch device-side copy op
    from the fp16/bf16 padding path (ViewCopyAiCore binary missing on some
    CANN 9.x SoC packages, 561103).  Whole-block segments cannot write pad
    rows, so any pad-needing segment is routed to a per-row (tail) mode;
    no-pad shapes keep the whole-block dispatch exactly (byte-for-byte).
    """
    B, S, Nq, D = query.shape
    Skv = key.shape[1]
    Nkv = key.shape[2]
    if key.shape[0] != B or value.shape != key.shape:
        return None
    if not (query.is_contiguous() and key.is_contiguous() and value.is_contiguous()):
        return None
    if D * 2 % 32 != 0 or S <= 4:
        return None
    dtype_str = _dtype_str(query.dtype)
    if dtype_str is None or key.dtype != query.dtype:
        return None

    Sq_out = S if S_out is None else S_out
    Skv_out = Skv if S_kv_out is None else S_kv_out
    pad_q = Sq_out > S
    pad_kv = Skv_out > Skv

    if S >= 256 or Nq <= 64:
        # Q segment V2, shared bm with the KV segments (single ub buffer).
        # whole-block copies cannot write pad rows — padded shapes fall
        # through to the universal tail mode below.
        bm = _shared_bm(S, Skv, D)
        if bm is not None and not pad_q and not pad_kv:
            key_ = ("qkv", B, S, Nq, Skv, Nkv, D, bm, False, 0, bm, dtype_str, False, Sq_out, Skv_out)
            if key_ not in _transpose_kernel_cache:
                _transpose_kernel_cache[key_] = _transpose_qkv(B, S, Nq, Skv, Nkv, D, bm, False, 0, bm, dtype_str, False, Sq_out, Skv_out)
            fn = _transpose_kernel_cache[key_]
            qy, ky, vy = fn(
                query.view(B * S, Nq * D),
                key.view(B * Skv, Nkv * D),
                value.view(B * Skv, Nkv * D),
            )
            return (qy.view(B, Nq, S, D), ky.view(B, Nkv, Skv, D), vy.view(B, Nkv, Skv, D))
        # no divisible shared bm (or padding needed)
        # -> universal tail fallback below
    else:
        # S < 256 and Nq > 64: Q segment V1, KV segments V2.
        # The V1 Q segment is per-row (tail-safe, pad-aware), so it may
        # serve padded outputs; only the KV segments need the pad check.
        q_bn = min(Nq, 128)
        kv_bm = min(512, Skv, _ub_rows(D))
        while kv_bm > 16 and Skv % kv_bm != 0:
            kv_bm //= 2
        if Nq % q_bn == 0:
            if Skv % kv_bm == 0 and not pad_kv:
                key_ = ("qkv", B, S, Nq, Skv, Nkv, D, 8, True, q_bn, kv_bm, dtype_str, False, Sq_out, Skv_out)
                if key_ not in _transpose_kernel_cache:
                    _transpose_kernel_cache[key_] = _transpose_qkv(
                        B, S, Nq, Skv, Nkv, D, 8, True, q_bn, kv_bm, dtype_str, False, Sq_out, Skv_out
                    )
                fn = _transpose_kernel_cache[key_]
                qy, ky, vy = fn(
                    query.view(B * S, Nq, D),
                    key.view(B * Skv, Nkv * D),
                    value.view(B * Skv, Nkv * D),
                )
                return (qy.view(B, Nq, Sq_out, D), ky.view(B, Nkv, Skv, D), vy.view(B, Nkv, Skv, D))
            # Q V1 (tail-safe by construction) + KV tail: per-row KV
            # copies at a fixed bm (pad-aware on both segments)
            kv_tail = _tail_bm(D)
            key_ = ("qkv", B, S, Nq, Skv, Nkv, D, 8, True, q_bn, kv_tail, dtype_str, True, Sq_out, Skv_out)
            if key_ not in _transpose_kernel_cache:
                _transpose_kernel_cache[key_] = _transpose_qkv(
                    B, S, Nq, Skv, Nkv, D, 8, True, q_bn, kv_tail, dtype_str, True, Sq_out, Skv_out
                )
            fn = _transpose_kernel_cache[key_]
            qy, ky, vy = fn(
                query.view(B * S, Nq, D),
                key.view(B * Skv, Nkv, D),
                value.view(B * Skv, Nkv, D),
            )
            return (qy.view(B, Nq, Sq_out, D), ky.view(B, Nkv, Skv_out, D), vy.view(B, Nkv, Skv_out, D))
        # Nq not divisible by q_bn -> universal tail fallback below

    # universal tail fallback: q_v1=False, fixed bm, per-row guarded
    # copies for BOTH the Q and KV segments (covers any S / S_kv;
    # pad-aware on all segments).
    bm = _tail_bm(D)
    key_ = ("qkv", B, S, Nq, Skv, Nkv, D, bm, False, 0, bm, dtype_str, True, Sq_out, Skv_out)
    if key_ not in _transpose_kernel_cache:
        _transpose_kernel_cache[key_] = _transpose_qkv(B, S, Nq, Skv, Nkv, D, bm, False, 0, bm, dtype_str, True, Sq_out, Skv_out)
    fn = _transpose_kernel_cache[key_]
    qy, ky, vy = fn(
        query.view(B * S, Nq, D),
        key.view(B * Skv, Nkv, D),
        value.view(B * Skv, Nkv, D),
    )
    return (qy.view(B, Nq, Sq_out, D), ky.view(B, Nkv, Skv_out, D), vy.view(B, Nkv, Skv_out, D))


def out_plan(B, Nq, S, D):
    """Output-direction (BHSD->BSND) plan: always ('custom', bm, bn).

    The custom V1 tilelang kernel is used for ALL shapes —
    never fall back to aclnn ``transpose(...).contiguous()``. On CANN 9.x the
    aclnn ``Transpose`` binary may be missing on some SoC packages (error
    code 561103 / "Op Transpose does not has any binary"), so any aclnn
    fallback is treated as a latent failure risk and avoided.

    bm: tile rows along Nq (M dim); m_idx < M guard handles tail rows.
    bn: tile cols along S (N dim); tile loop is *exact*, so bn must divide
    S. We pick the largest power-of-2 ≤ 128 that divides S.
    D * 2 must be a multiple of 32 for fp16 vector-load alignment (returns
    None otherwise — the caller then falls back to
    out_bhsd.transpose(1, 2).contiguous()).
    """
    bm = 16 if Nq >= 48 else 8
    # bn must divide S. Search largest power-of-2 in [1, 128].
    bn = 128
    while bn > 1 and S % bn != 0:
        bn //= 2
    if D * 2 % 32 != 0:
        return None  # V1 can't handle this alignment; caller uses the torch transpose
    return ("custom", bm, bn)


def transpose_out(out_bhsd, s_real=None):
    """BHSD [B,Nq,S,D] -> BSND [B,S,Nq,D] (custom V1 kernel, no aclnn fallback).

    Always routes through the tilelang V1 kernel; the
    aclnn fallback path (``out_bhsd.transpose(1, 2).contiguous()``)
    is retained *only* for truly unknown dtypes / alignment edge cases that
    V1 cannot satisfy. The V1 kernel is tilelang-compiled (no aclnn
    binary needed), so it works on all SoC packages, and produces the
    same layout as aclnn Transpose.

    s_real (when given): out_bhsd is the PADDED
    kernel output [B, Nq, S_pad, D] (contiguous) and only its first s_real
    rows are transposed — the V1 kernel reads the padded tensor directly
    (n_pad strides) and writes only the real-S output rows, so no crop
    materialization is needed (Out_padded[:, :, :S, :].contiguous() is a
    fp16/bf16 ViewCopy d2d whose binary is missing on some CANN 9.x SoC
    packages, 561103).
    """
    B, Nq, S, D = out_bhsd.shape
    dtype_str = _dtype_str(out_bhsd.dtype)
    if s_real is not None and s_real < S:
        # padded-input mode (gqa() adapter, non-aligned S)
        plan = out_plan(B, Nq, s_real, D) if dtype_str is not None else None
        if plan is not None and out_bhsd.is_contiguous():
            bm, bn = plan[1], plan[2]
            key_ = ("v1", B, Nq, s_real, D, bm, bn, dtype_str, S)
            if key_ not in _transpose_kernel_cache:
                _transpose_kernel_cache[key_] = _transpose_v1(B, Nq, s_real, D, bm, bn, dtype_str, n_pad=S)
            fn = _transpose_kernel_cache[key_]
            y = fn(out_bhsd.view(B * Nq, S, D))
            return y.view(B, s_real, Nq, D)
        # unreachable for contiguous fp16/bf16 outputs with D*2%32==0
        # (the V1 plan above covers them): crop and recurse through the
        # fallback path
        return transpose_out(out_bhsd[:, :, :s_real, :].contiguous())
    plan = out_plan(B, Nq, S, D) if dtype_str is not None else None
    if plan is None:
        # Unknown dtype or unaligned D — last-resort fallback (may fail on
        # CANN 9.x environments missing the aclnn Transpose binary; this
        # path should be essentially unreachable in practice: the V1 plan
        # covers every fp16/bf16 shape whose D passes the 32-byte
        # vector-load alignment gate D*2 % 32 == 0, i.e. any D that is a
        # multiple of 16).
        if not out_bhsd.is_contiguous():
            out_bhsd = out_bhsd.contiguous()
        return out_bhsd.transpose(1, 2).contiguous()
    if not out_bhsd.is_contiguous():
        # the crop view materializes via a torch d2d copy that reads
        # the tilelang kernel's output — the same launch-ordering hazard as
        # the padding path (see _npu_sync_tensor); aligned outputs are
        # contiguous and skip this entirely.
        _npu_sync_tensor(out_bhsd)
        # Make contiguous first (byte-level copy, no Transpose kernel invoked).
        out_bhsd = out_bhsd.contiguous()
    bm, bn = plan[1], plan[2]
    key_ = ("v1", B, Nq, S, D, bm, bn, dtype_str)
    if key_ not in _transpose_kernel_cache:
        _transpose_kernel_cache[key_] = _transpose_v1(B, Nq, S, D, bm, bn, dtype_str)
    fn = _transpose_kernel_cache[key_]
    y = fn(out_bhsd.view(B * Nq, S, D))
    return y.view(B, S, Nq, D)


# ─────────────────────────────────────────────────────────────────────
#  Tensor (BSND) interface — adapter constants + mask construction
# ─────────────────────────────────────────────────────────────────────

# The additive mask constant must dominate ANY finite fp32 score magnitude,
# not just unit-range scores.  With -(2^30) a full-range fp16 dot
# product (D=128 -> |s| up to ~5.5e11) lets a MASKED column win the
# max-after-mask running max (s_j - 2^30 > max(visible)), so the mask no
# longer underflows exp() and the row attends to future tokens.
# -1e30 restores mask dominance for every finite fp32 score; for normal
# ranges the softmax is bit-identical (both constants underflow exp()).
_BIG_NEG = -1e30

_KERNEL_BLOCK_N = 128  # use_mask elimination threshold for the adapter

_COMPILED_CACHE = {}

_DUMMY_MASK = {}


def _dummy_mask(device):
    """Shared uninitialized stand-in for the never-read Mask argument.

    Replaces the per-call torch.zeros(B, S, S_kv) whose ZerosLike fill
    kernel would be counted in the device-kernel-sum elapsed metric.
    """
    key = str(device)
    m = _DUMMY_MASK.get(key)
    if m is None:
        m = torch.empty(1, dtype=torch.float32, device=device)
        _DUMMY_MASK[key] = m
    return m


def _build_causal_mask(B, S, S_kv, device):
    """Build additive causal mask.

    Direct additive constructions (no binary * scalar step).
    [B, S, S_kv] float32: 0=visible, _BIG_NEG (=-1e30)=masked.
    Right-aligned convention: j > i + (S_kv - S) → masked.
    """
    # Special case: S=1 causal → no masked positions.
    # torch.zeros(..., device=npu) lowers to ZerosLike, missing on some
    # CANN 9.x SoC packages even for fp32 — build on CPU + single
    # .to(device) H2D.  (Dead path for the gqa() adapter — decode_direct
    # handles S<=4 — kept correct for factory callers.)
    if S == 1:
        return torch.zeros(B, 1, S_kv, dtype=torch.float32).to(device)

    # General case: direct additive constructions via index fill
    mask = torch.zeros(S, S_kv, dtype=torch.float32, device="cpu")
    i = torch.arange(S).unsqueeze(-1)
    j = torch.arange(S_kv).unsqueeze(0)
    masked = j > (i + (S_kv - S))
    mask[masked] = _BIG_NEG
    return mask.unsqueeze(0).expand(B, S, S_kv).contiguous().to(device)


# ─────────────────────────────────────────────────────────────────────
#  Public API (single merged-module entry points)
# ─────────────────────────────────────────────────────────────────────

__all__ = ["gqa", "gqa_expert"]


def gqa(
    query=None,
    key=None,
    value=None,
    scaleValue=-1.0,
    is_causal=False,
    *,
    B=None,
    S=None,
    S_kv=None,
    sm_scale=None,
    N_q=8,
    N_kv=2,
    D=128,
    dtype="float16",
):
    """GQA forward — dual-interface entry point.

    Supports two calling conventions:

    1. **Tensor (BSND) interface** (keyword, tensors):
           gqa(query=Q, key=K, value=V, scaleValue=..., is_causal=...)
       Returns output [B, S, N_q, D] in BSND layout directly.

    2. **Factory interface** (keyword, shapes — legacy):
           gqa(B=..., S=..., S_kv=..., sm_scale=..., is_causal=..., ...)
       Returns a callable ``out = kernel(Q, K, V, Mask)``.

    The dispatch is based on which set of required arguments is provided:
    if ``B`` is given (factory mode) the factory path is used; otherwise
    the tensor (BSND) adapter path is used (``query`` is required).  The
    adapter is implemented inline in this file (see the module docstring
    Note) — no cross-module imports.
    """

    if B is not None:
        # Factory / legacy interface — delegate to gqa_expert
        return gqa_expert(
            B,
            S,
            S_kv,
            sm_scale,
            is_causal,
            N_q=N_q,
            N_kv=N_kv,
            D=D,
            dtype=dtype,
        )

    # ================================================================
    # Tensor (BSND) interface — inline adapter (single-file example;
    # see the module docstring Note).
    # ================================================================
    q, k_t, v_t = query, key, value
    B_bs, S_bs, N_q_bs, D_bs = q.shape
    S_kv_bs = k_t.shape[1]
    N_kv_bs = k_t.shape[2]

    if q.dtype == torch.float16:
        dtype_str = "float16"
    elif q.dtype == torch.bfloat16:
        dtype_str = "bfloat16"
    else:
        raise ValueError(f"Unsupported dtype: {q.dtype}")

    sm_scale_val = scaleValue if scaleValue > 0 else 1.0 / (D_bs**0.5)

    decode_direct = 1 <= S_bs <= 4

    if not decode_direct:
        # Block-aligned pad targets for the transpose outputs — MUST
        # match gqa_expert's prefill S_padded (block_M=64) / S_kv_padded
        # (block_N=128) so the padded BHSD tensors the transpose emits are
        # exactly what the chosen kernel declares (every chain candidate is
        # built with the unified S_padded).  Pad rows are zero-filled
        # in-kernel: zero torch device-side copy ops on the fp16/bf16
        # non-aligned path (ViewCopyAiCore binary missing on some CANN 9.x
        # SoC packages, 561103).
        S_out = ((S_bs + 63) // 64) * 64
        S_kv_out = ((S_kv_bs + 127) // 128) * 128
        res = transpose_qkv(q, k_t, v_t, S_out=S_out, S_kv_out=S_kv_out)
        if res is not None:
            q_bhsd, k_bhsd, v_bhsd = res
        else:
            q_bhsd = q.transpose(1, 2).contiguous()
            k_bhsd = k_t.transpose(1, 2).contiguous()
            v_bhsd = v_t.transpose(1, 2).contiguous()

    if decode_direct:
        # Decode path (S<=4): the decode _run
        # always rebuilds the proper [B, M_tile, S_kv_padded] mask format
        # itself (its own single host->device transfer).  The Mask
        # argument passed in is DISCARDED by _run when use_mask=True
        # (causal / col padding), so there is NO need to build any mask
        # here.  Use the shared dummy stub: building a real mask here
        # would add a second per-call host->device transfer for a buffer
        # the kernel never reads.
        mask_npu = _dummy_mask(q.device)
    elif is_causal:
        mask_npu = _build_causal_mask(B_bs, S_bs, S_kv_bs, q.device)
    elif S_kv_bs % _KERNEL_BLOCK_N != 0:
        # Non-causal + non-128-aligned
        # S_kv needs an all-zero additive mask.  torch.zeros(..., device=npu)
        # lowers to the ZerosLike / aclnnInplaceZero binary, which is MISSING
        # from some CANN 9.x SoC packages (error 561103).
        # Build on CPU (host allocator, no NPU binary) and transfer with ONE
        # clean .to(device) H2D — the same pattern as _build_causal_mask.
        mask_npu = torch.zeros(
            B_bs,
            S_bs,
            S_kv_bs,
            dtype=torch.float32,
        ).to(q.device)
    else:
        mask_npu = _dummy_mask(q.device)

    cache_key = (B_bs, S_bs, S_kv_bs, N_q_bs, N_kv_bs, D_bs, dtype_str, is_causal, sm_scale_val)
    if cache_key not in _COMPILED_CACHE:
        _COMPILED_CACHE[cache_key] = gqa_expert(
            B=B_bs,
            S=S_bs,
            S_kv=S_kv_bs,
            sm_scale=sm_scale_val,
            is_causal=is_causal,
            N_q=N_q_bs,
            N_kv=N_kv_bs,
            D=D_bs,
            dtype=dtype_str,
        )
    kernel_fn = _COMPILED_CACHE[cache_key]

    if decode_direct:
        return kernel_fn(q, k_t, v_t, mask_npu)

    # crop=False: the prefill runner returns the FULL padded output;
    # padded shapes transpose it directly (transpose_out s_real mode, no
    # crop materialization), aligned shapes use it directly
    # (the full tensor IS the [B, Nq, S, D] output).
    out_padded = kernel_fn(q_bhsd, k_bhsd, v_bhsd, mask_npu, crop=False)
    if S_out > S_bs:
        return transpose_out(out_padded, s_real=S_bs)
    return transpose_out(out_padded)


# ─────────────────────────────────────────────────────────────────────
#  Self-verification entry point (`python gqa.py`)
# ─────────────────────────────────────────────────────────────────────


def _ref_gqa(query, key, value, scaleValue=-1.0, is_causal=False):
    """PyTorch reference implementing the GQA forward contract.

    Computes in the dtype/device of the tensors it is given.  The
    self-check below feeds it host float32 tensors, so the comparison
    never touches a device helper binary (the same 561103 concern
    documented at the transpose / pad call sites above).

    Contract (identical to the kernel semantics in the module docstring):
      - GQA group folding: query head ``n_kv * G + g`` shares KV head
        ``n_kv``, with ``G = N_q // N_kv``.  K/V are NOT materialized to
        N_q heads — the group dim is folded into the matmul M dim.
      - ``scaleValue <= 0`` -> ``1 / sqrt(D)``.
      - ``is_causal``: right-bottom-aligned mask — score (i, j) with
        ``j > i + (S_kv - S)`` is set to -inf before the softmax.
      - Fully-masked rows (causal with S > S_kv): the softmax of an
        all -inf row is 0/0 = NaN and is replaced by 0, so those rows
        output exactly 0.
      - NaN/Inf inputs propagate unsanitized.

    Args:
        query: [B, S, N_q, D]
        key: [B, S_kv, N_kv, D]
        value: [B, S_kv, N_kv, D]

    Returns:
        output: [B, S, N_q, D], contiguous
    """
    B, S, N_q, D = query.shape
    S_kv = key.shape[1]
    N_kv = key.shape[2]

    if scaleValue <= 0:
        scaleValue = 1.0 / (D**0.5)

    G = N_q // N_kv
    q = query.reshape(B, S, N_kv, G, D).permute(0, 2, 3, 1, 4).reshape(B, N_kv, G * S, D)
    k = key.permute(0, 2, 1, 3)  # [B, N_kv, S_kv, D]
    v = value.permute(0, 2, 1, 3)  # [B, N_kv, S_kv, D]

    scores = torch.matmul(q, k.transpose(-2, -1)) * scaleValue
    scores = scores.reshape(B, N_kv, G, S, S_kv)
    if is_causal:
        i = torch.arange(S, device=scores.device).unsqueeze(-1)
        j = torch.arange(S_kv, device=scores.device).unsqueeze(0)
        scores = scores.masked_fill(j > (i + (S_kv - S)), float("-inf"))
    scores_max = scores.max(dim=-1, keepdim=True).values
    all_masked = torch.isinf(scores_max) & (scores_max < 0)
    weights = torch.softmax(scores, dim=-1)
    weights = torch.where(all_masked, torch.zeros_like(weights), weights)
    weights = weights.reshape(B, N_kv, G * S, S_kv)
    out = torch.matmul(weights, v)  # [B, N_kv, G*S, D]

    return out.reshape(B, N_kv, G, S, D).permute(0, 3, 1, 2, 4).reshape(B, S, N_q, D).contiguous()


if __name__ == "__main__":
    # Self-check for `python gqa.py` — the examples/bench_test.sh CI entry,
    # which requires the process to exit 0 AND print a completion marker.
    # One case per dispatch path, both shapes taken from the bench matrix:
    #   prefill causal  B=2  S=128 S_kv=128  N_q=32 N_kv=8 D=128 fp16
    #                   -> transpose_qkv + g2stack + transpose_out
    #   decode          B=16 S=1   S_kv=2048 N_q=32 N_kv=8 D=128 fp16
    #                   -> dense_expert, BSND supplied directly (no helper)
    # Every import below stays inside the guard so that importing this
    # module (e.g. as part of a packaged wheel) performs no device work.
    import sys
    import time

    import torch_npu  # noqa: F401  (registers the npu backend)

    tilelang.disable_cache()

    # (name, B, S, S_kv, N_q, N_kv, D, is_causal, dtype)
    cases = [
        ("prefill_causal", 2, 128, 128, 32, 8, 128, True, torch.float16),
        ("decode_noncausal", 16, 1, 2048, 32, 8, 128, False, torch.float16),
    ]
    atol, rtol = 1e-3, 1e-2  # fp16 mixed tolerance, zero violations allowed
    failures = []

    for name, B, S, S_kv, N_q, N_kv, D, is_causal, dtype in cases:
        tag = f"{name} B={B} S={S} S_kv={S_kv} N_q={N_q} N_kv={N_kv} D={D} causal={is_causal} fp16"
        torch.manual_seed(42 + B + S + S_kv)  # uniform [-1, 1], reproducible
        q = (torch.rand(B, S, N_q, D) * 2 - 1).to(dtype).npu()
        k = (torch.rand(B, S_kv, N_kv, D) * 2 - 1).to(dtype).npu()
        v = (torch.rand(B, S_kv, N_kv, D) * 2 - 1).to(dtype).npu()
        torch.npu.synchronize()

        start = time.time()
        out = gqa(query=q, key=k, value=v, scaleValue=-1.0, is_causal=is_causal)
        torch.npu.synchronize()
        elapsed = time.time() - start

        ref = _ref_gqa(q.cpu().float(), k.cpu().float(), v.cpu().float(), -1.0, is_causal)
        got = out.float().cpu()

        problems = []
        if tuple(out.shape) != (B, S, N_q, D):
            problems.append(f"shape {tuple(out.shape)} != {(B, S, N_q, D)}")
        if out.dtype != dtype:
            problems.append(f"dtype {out.dtype} != {dtype}")
        if not out.is_contiguous():
            problems.append("output not contiguous")
        n_nan, n_inf = int(torch.isnan(got).sum()), int(torch.isinf(got).sum())
        if n_nan or n_inf:
            problems.append(f"nan={n_nan} inf={n_inf}")
        diff = (got - ref).abs()
        max_diff = diff.max().item()
        viol = int((diff > atol + rtol * ref.abs()).sum())
        if viol:
            problems.append(f"tolviol={viol}")

        status = "PASS" if not problems else "FAIL: " + "; ".join(problems)
        print(f"[gqa] {tag} max_diff={max_diff:.3e} viol={viol} launch={elapsed:.1f}s -> {status}", flush=True)
        if problems:
            failures.append(name)

    if failures:
        print(f"[gqa] self-check FAILED for: {', '.join(failures)}", flush=True)
        sys.exit(1)
    print("Test Passed!")
