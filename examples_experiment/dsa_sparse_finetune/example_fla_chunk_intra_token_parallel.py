"""fla_chunk_intra_token_parallel: KDA intra-chunk per-token decay inner
products Aqk/Akk on Ascend 910B3 (Cube GEMM offload + fixed-core C/V
software pipeline).

Semantics (upstream: the FLA/FLA_KDA chunk_kda_fwd_kernel_intra_token_parallel
Triton kernel and its TileLang CUDA port in tile-ai/tilelang). For token i
in subchunk [i_ts, i_ts+SCS) and j in [i_ts, min(i+1, i_ts+SCS)):

    kgj[h, d] = k[b, j, h, d] * 2^(g[b, i, h, d] - g[b, j, h, d])
    Aqk[b, i, h, j % CS]   = scale * sum_d q[b, i, h, d] * kgj[h, d]
    Akk[b, i, h, j - i_ts] = (j < i) ? sum_d (beta[b,i,h]*k[b,i,h,d]) * kgj[h, d] : 0

Aqk columns not covered by any j stay 0 (cleared-row semantics, matching
the upstream TileLang version); the Akk diagonal (j == i) stays 0. scale is
fixed at 1.0 (upstream test configuration; not passed through).

Ascend restructuring (all deviations from the upstream CUDA code are
deliberate design decisions, each verified on 910B3):

1. Subchunk-parallel logical tasks: one (batch, subchunk, head-group) per
   task; one-dimensional T.Kernel with (cid, vid) (3-D kernels are
   unsupported on this backend, threads are restricted to 1/2; vid splits
   the BH-head group into BH_V = BH // 2 per AIV).
2. exp2(x) as exp(x * ln2) (exp2 has no Ascend backend mapping); the
   subtraction feeding each exp runs in the ORIGINAL log2 domain BEFORE
   the ln2 scaling -- at large |g| the scaled values round to the same
   fp32 (ulp of |g*ln2| can exceed the gate step, e.g. g ~ -1e8 with a
   step of 8) and the decay difference would be lost, while the
   raw-domain difference of adjacent fp32 gates is exact down to the
   1-ulp boundary. All bf16<->fp32 conversions UB-internally; fp32
   staging + block CAST_RINT + per-head 2D strided writeback bursts.
3. Factored decay (mid-anchor a = SCS//2): e_ij = E+_i * E-_j with
   E+ = exp((g_i - g_a) * ln2), E- = exp((g_a - g_j) * ln2);
   per-head one-shot slabs Q' = q*E+, K' = k*E-, K'' = k*E+.
   Documented domain restriction: the within-subchunk half-span gate
   decay |g_i - g_a| < ~127 (log2 domain) keeps E+/E- finite in fp32.
   (The |g| magnitude itself is unconstrained: the anchor subtraction
   happens in the raw gate domain, so arbitrarily deep absolute gates
   survive as long as their differences are fp32-representable.)
4. GEMM OFFLOAD to the Cube cores: the per-(i,j) dot products leave the
   Vector pipe entirely --
      Aqk_h = tril(Q'_h @ K'_h^T),   Akk_h = strict_tril((beta*K'')_h @ K'_h^T)
   via one gemm_v0 per GGRP-head group with block-diagonal batching:
      A = [Q'; K''] rows of GGRP heads stacked  [GGRP*2*SCS, DK]
      B = K' rows of the same GGRP heads        [GGRP*SCS,  DK]
      C = A @ B^T (transpose_B) -> [GGRP*2SCS, GGRP*SCS]; the useful output
      is the per-head diagonal [2*SCS, SCS] blocks (4x FLOP waste at
      GGRP=4, measured cheaper than 4x more gemm calls: ggrp=1 benches
      +168us over ggrp=4 at the bench shape).
   beta is folded into K'' on the V side (one-hot select + row reduce +
   column broadcast, replacing 64 scalar BufferLoad/muls per task) so the
   Akk GEMM result needs no post-folding.
   fp32 operands (not bf16): bf16 rounding of Q'/K'/K'' alone fails the
   Akk matched-ratio gate (CPU-simulated 0.9848 < 0.99 vs fp32 1.0000)
   and costs the same transit bytes as split-bf16.
   GGRP sizing: the largest DIVISOR of BH satisfying the L0A
   single-K-step budget (GGRP*2*SCS*DK*4B <= 64KB) and the L0C/L0B
   N >= 16 fractal rule (N = GGRP*SCS >= 16). Divisibility is a
   hard requirement: the block-diagonal tiling must cover the BH heads
   exactly -- a non-divisor leaves tail heads reading out-of-range gemm
   groups in o_relay. No divisor satisfying both constraints is a
   hard error (the wrapper rejects BH*SCS < 16 up front).
5. Post-GEMM masking on the V side via arith_progression + broadcast +
   compare + select (VSEL_TENSOR_SCALAR_MODE): the block-diagonal garbage
   cells (j > i) can be +-inf under deep decay (E+_i*E-_j up to 2^180 in
   the accumulator) -- a multiply mask would give inf*0 = NaN; select
   REPLACES with 0, matching the golden's masked_fill semantics.
6. Memory-planner interaction: region-dst writes do not extend a buffer's
   planner liveness -- the per-vid operand accumulators a_agg/b_agg must
   be whole-buffer-filled first (the fill anchors the interval),
   otherwise the planner overlays them with the in-loop fp32 scratches
   and the staged operands get corrupted. Per-head load staging
   (q/k/gk loaded inside the factorize loop) keeps the peak UB ~124KB.
   The drain staging uses single-head [SCS, *] buffers refilled per h
   (instead of a [BH_V, SCS, *] whole-task layout): the planner's
   ExtendKillIndex forces drain+produce buffers to coexist over the whole
   task loop, and the 22.5KB this frees is what unblocks the beta-fold
   scratch.
7. FIXED-CORE C/V SOFTWARE PIPELINE (the naive per-task
   V(construct)->ws->C(gemm)->ws->V(postproc) chain is fully serial):
   - FIXED-CORE grid: T.Kernel(CORE_EFF = min(20, NBLK)) with a strided
     task map tid = cid + t*CORE_EFF; out-of-range tasks are redirected
     BRANCH-FREE (Select) to the core's own first task -- the recompute
     is bit-identical and the garbage drains are overwritten by the real
     ones, so ZERO IfThenElse statements remain in the loop (each
     IfThenElse costs two PipeBarrier<PIPE_ALL> drains under AUTO_SYNC).
   - SOFTWARE-PIPELINED task loop with a 2-TASK-DEFERRED drain: the V
     side post-processes task t-2 at the TOP of iteration t, then
     constructs task t's operands; the C side gemms task t in the same
     iteration. Steady state: V drain(t-2)/produce(t) overlap C
     gemm(t-1)/gemm(t); the manual drain flag (set at C's iteration t-2)
     is long fired when V arrives, so neither core stalls on the other's
     CURRENT phase.
   - DOUBLE-BUFFERED relay slots (NSLOT=2, t % NSLOT): workspace_a/b
     (V->C operand relay) keep their AUTO_CV_SYNC cross-core flags (the
     pass keys on the literal buffer-name substring "workspace"; produce
     and consume are in the SAME loop iteration, so the pass's FIFO
     index pairing is exactly the required same-iteration edge). The
     result relay is named o_relay (NON-"workspace" name => NO auto
     flag): its C->V handoff uses MANUAL cross-core flags
     T.set_cross_flag("FIX", 3 + t % 2) / T.wait_cross_flag(...) with
     alternating event IDs so the 2-deferred drain pairs with the FIX of
     iteration t-2 (auto FIFO flags would pair same-iteration and
     deadlock: V's drain would wait FIX(t) which waits V's own
     produce(t)).
   - WAR safety at NSLOT=2 (both reuses transitively ordered, no extra
     flags): (a) V produce(t+2) overwrites the ws slot that C gemm(t)
     reads -- V produce(t+2) follows V drain(t) in V's stream, which
     waits the manual FIX(t) flag, which C sets only after gemm(t)'s L1
     loads are done; (b) C FIX(t+2) overwrites the o_relay slot V
     drain(t) reads -- FIX(t+2) follows C's wait on V's produce(t+2)
     relay flag, and V's produce(t+2) MTE3 follows (in-order MTE3 pipe)
     V drain(t)'s MTE2->Vector->MTE3 chain.
   - Loop-invariant hoists: the lt/slt masks and the beta one-hot
     compare source are computed ONCE per core.
   - Workspace capacity is per-core ping-pong [CORE_EFF, NSLOT, ...]
     (~5 MiB at the bench shape, permanently L2-resident) instead of
     per-task [NBLK, ...] (~640 MiB); per-task relay traffic is
     unchanged.
   - Peeled fill iterations 0/1 (produce + gemm + FIX + a trailing
     garbage drain with NO wait): avoid the t=0 deadlock (produce must
     precede the first drain) and keep every relay's access-site counts
     balanced.
8. C-SIDE DOUBLE-SLOT DECOMPOSED PIPELINE: the gemm_v0 template calls
   are replaced by a hand-written decomposed sequence (GM->L1, L1->L0A/B,
   mma, L0C->GM) on INDEPENDENT double-slot L0C accumulators (even head
   groups -> C_l0c_0, odd -> C_l0c_1; packed [2,*] slots corrupt in the
   zN fractal layout). The AUTO_SYNC pass keys WAR/RAW flag pairs per
   buffer, so the two slots' chains interleave in the steady state:
   mma(g1) overlaps fixpipe(g0), and the next task's GM->L1 slab
   prefetch overlaps the current mma chain. L1 staging is a single
   full-slab buffer loaded by ONE workspace copy per relay per section
   (keeps the AUTO_CV_SYNC cube/vec site counts balanced) and row-sliced
   per group by the L1->L0A/B copies. L0A/L0B stay single-slot: the
   fp32 A tile [AM, DK] already fills the 64KB L0A budget at GGRP=4.
   No hand-written flags are added on either side (they would collide
   with the AUTO_SYNC pass's round-robin event IDs).

Supported domain: fixed-length sequences with S % CS == 0, CS % SCS == 0,
CS and SCS powers of two (tile alignment), DK % 16 == 0, DK >= 32, and
BH * SCS >= 16 (gemm N-fractal lower bound). BH adapts 8 -> 4 -> 2
subject to BOTH head divisibility (H % BH == 0) and the 192KB unified-
buffer coexistence budget (the a_agg/b_agg operand accumulators plus
the DK-proportional staging; e.g. DK=256 drops to BH=2, DK >= 512 is
rejected with an explicit budget message); q/k/beta bfloat16 + gk
float32 inputs, bfloat16 outputs. varlen (cu_seqlens) is out of scope.
"""

import torch
import tilelang
from tilelang import language as T

# 2^g is computed as exp(g * ln2): exp2 has no Ascend backend mapping.
LN2 = 0.6931471805599453

# Ascend 910B3: 20 AI Cube cores (+ 40 AI Vector cores). The fixed-core
# grid launches one block per cube core; the (cid, vid) binding splits
# each task's vector work across the 2 AIV sub-blocks of the core pair.
CORE_NUM = 20

# Relay slot ring per core (t % NSLOT). NSLOT=2 is transitively WAR-safe
# at the 2-task-deferred drain (see the pipeline section of the module
# docstring); bump to 4 only as a diagnostic if a reuse race is suspected.
NSLOT = 2

# Manual cross-core flag IDs for the o_relay C->V handoff start at 3:
# IDs 0-2 stay reserved for the AUTO_CV_SYNC-generated flags.
FLAG_BASE = 3

PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
    tilelang.PassConfigKey.TL_ASCEND_VECTOR_MASK_REUSE: False,
}


# 910B3 unified buffer per AIV: 192KB. The memory planner pins the
# produce-side staging, the per-vid operand accumulators and the
# drain-side staging over the whole task loop (ExtendKillIndex), so the
# FULL coexistence budget must fit.
UB_TOTAL_BYTES = 196_352


def _ub_bytes(BH_V, SCS, DK, CS):
    """UB bytes coexisting in the steady state: bf16/fp32 load staging
    rows, the fp32 operand accumulators (a_agg + b_agg, the BH_V-
    proportional part) and the fixed scratches/masks/drain staging
    (AKK_W = max(SCS, 16))."""
    akk_w = max(SCS, 16)
    staging = 2 * SCS * DK * 2 + 6 * SCS * DK * 4  # q/k rows + 6 fp32 rows
    accum = 3 * SCS * BH_V * DK * 4  # a_agg (2*SCS*BH_V) + b_agg (SCS*BH_V)
    fixed = (
        SCS * 16 * (2 + 4 + 1 + 4 + 4)
        + SCS * 4  # beta_h/f, eq_m, beta_sel, j16b, beta_col
        + 4 * SCS
        + 4 * 16  # idx + idx16
        + 4 * SCS * SCS * 2
        + SCS * SCS * 2  # i_grid/j_grid + lt_m/slt_m
        + 2 * SCS * SCS * 4
        + 2 * SCS * SCS * 4  # r_ub + r1_m/a_m
        + SCS * CS * 4
        + SCS * akk_w * 4  # aqk_s + akk_s
        + SCS * CS * 2
        + SCS * akk_w * 2  # aqk_h + akk_h
    )
    return staging + accum + fixed


def _pick_bh(H, SCS, DK, CS):
    """BH is a performance tiling parameter, not a dispatch predicate:
    pick the largest supported group that divides H (8 -> 4 -> 2) AND
    keeps the coexisting UB budget within 192KB (smaller groups shrink
    the operand accumulators while the staging stays DK-proportional)."""
    for bh in (8, 4, 2):
        if H % bh != 0:
            continue
        if _ub_bytes(bh // 2, SCS, DK, CS) <= UB_TOTAL_BYTES:
            return bh
    return 0


def _pick_ggrp(BH, SCS, DK):
    """Heads per gemm group: amortize the per-call cube overhead with
    block-diagonal batching. GGRP MUST DIVIDE BH -- the block-diagonal
    tiling covers the BH heads exactly, and a non-divisor leaves tail
    heads reading out-of-range gemm groups in o_relay (with inconsistent
    workspace/L1 row counts). Among the divisors of BH, prefer the
    largest that satisfies the L0A single-K-step budget (fp32 A tile
    GGRP*2*SCS x DK x 4B <= 64KB) and the L0B/L0C N >= 16 fractal rule
    (N = GGRP*SCS)."""
    for g in (8, 4, 2, 1):
        if BH % g != 0:
            continue
        if g * 2 * SCS * DK * 4 > 65536:
            continue
        if g * SCS < 16:
            continue
        return g
    raise ValueError(
        f"no divisor of BH={BH} satisfies the gemm constraints "
        f"(L0A budget GGRP*2*SCS*DK*4B <= 64KB and N=GGRP*SCS >= 16) "
        f"with SCS={SCS}, DK={DK}"
    )


@tilelang.jit(out_idx=[-2, -1], workspace_idx=[4, 5, 6], pass_configs=PASS_CONFIGS)
def fla_chunk_intra_token_parallel_kernel(
    B,
    S,
    H,
    DK,
    CS=64,
    SCS=16,
    BH=8,
    input_dtype="bfloat16",
    gate_dtype="float32",
    output_dtype="bfloat16",
):
    """JIT kernel generator. Returns a kernel taking (Q, K, GK, Beta) and
    producing (Aqk [B,S,H,CS], Akk [B,S,H,SCS]), both output_dtype. The
    runtime allocates the three fp32 GM relays (per-core ping-pong
    operand A/B transit + L0C result transit) alongside the outputs."""
    VEC_NUM = 2  # default dual vector threads: (cid, vid)
    BH_V = BH // VEC_NUM  # heads handled by each vid
    NSC = S // SCS  # subchunks per sequence
    HG = H // BH  # head groups
    NBLK = B * NSC * HG  # logical (batch, subchunk, head-group) tasks
    GGRP = _pick_ggrp(BH, SCS, DK)  # heads per gemm group
    NG = BH // GGRP  # gemm groups per task
    AM = GGRP * 2 * SCS  # gemm A rows (M)
    BN = GGRP * SCS  # gemm B rows (N)
    AKK_W = max(SCS, 16)
    AMV = BH_V * 2 * SCS  # per-vid a_agg rows
    BNV = BH_V * SCS  # per-vid b_agg rows
    op_dtype = gate_dtype  # fp32 GEMM operands (docstring item 4)
    # fixed-core pipeline mapping (docstring item 7)
    CORE_EFF = min(CORE_NUM, NBLK)
    TPT = (NBLK + CORE_EFF - 1) // CORE_EFF  # produce iterations per core
    anc = SCS // 2  # factored-decay mid anchor (loop-invariant)

    Q_shape = (B, S, H, DK)
    K_shape = (B, S, H, DK)
    GK_shape = (B, S, H, DK)
    Beta_shape = (B, S, H)
    Aqk_shape = (B, S, H, CS)
    Akk_shape = (B, S, H, SCS)

    @T.prim_func
    def main(
        Q: T.Tensor(Q_shape, dtype=input_dtype),  # type: ignore
        K: T.Tensor(K_shape, dtype=input_dtype),  # type: ignore
        GK: T.Tensor(GK_shape, dtype=gate_dtype),  # type: ignore
        Beta: T.Tensor(Beta_shape, dtype=input_dtype),  # type: ignore
        workspace_a: T.Tensor([CORE_EFF, NSLOT, BH * 2 * SCS, DK], dtype=op_dtype),  # type: ignore
        workspace_b: T.Tensor([CORE_EFF, NSLOT, BH * SCS, DK], dtype=op_dtype),  # type: ignore
        o_relay: T.Tensor([CORE_EFF, NSLOT, NG, AM, BN], dtype=gate_dtype),  # type: ignore
        Aqk: T.Tensor(Aqk_shape, dtype=output_dtype),  # type: ignore
        Akk: T.Tensor(Akk_shape, dtype=output_dtype),  # type: ignore
    ):
        with T.Kernel(CORE_EFF, is_npu=True) as (cid, vid):
            # ---- V-side UB buffers ----
            # per-head load staging + fp32 rows (consumed within one
            # factorize iteration; whole-block slabs would blow the UB
            # budget once the accumulators' liveness is anchored)
            q_h_row = T.alloc_ub([SCS, DK], input_dtype)
            k_h_row = T.alloc_ub([SCS, DK], input_dtype)
            q_f_row = T.alloc_ub([SCS, DK], gate_dtype)
            k_f_row = T.alloc_ub([SCS, DK], gate_dtype)
            g2_row = T.alloc_ub([SCS, DK], gate_dtype)
            beta_h = T.alloc_ub([SCS, 16], input_dtype)
            beta_f = T.alloc_ub([SCS, 16], gate_dtype)
            e = T.alloc_ub([SCS, DK], gate_dtype)  # E-/E+/K'' chain
            kgj = T.alloc_ub([SCS, DK], gate_dtype)  # K' = k * E-
            bc_row = T.alloc_ub([SCS, DK], gate_dtype)  # anchor broadcast
            # beta-fold vectorization scratch (one-hot select + row reduce +
            # column broadcast; replaces 16 scalar muls/head)
            eq_m = T.alloc_ub([SCS, 16], "uint8")  # (j == h) one-hot mask
            beta_sel = T.alloc_ub([SCS, 16], gate_dtype)
            beta_col = T.alloc_ub([SCS, 1], gate_dtype)
            # per-task j-index grid for the one-hot compare (built from the
            # hoisted idx values; a dedicated produce-live buffer keeps the
            # hoist-only j_grid liveness from extending over the loop)
            j16b = T.alloc_ub([SCS, 16], gate_dtype)
            # per-vid fp32 operand accumulators (fill-anchored liveness)
            a_agg = T.alloc_ub([AMV, DK], op_dtype)
            b_agg = T.alloc_ub([BNV, DK], op_dtype)
            # masks / index grids (attention_sink pattern); loop-invariant
            # -> computed ONCE per core
            idx = T.alloc_ub([SCS], gate_dtype)
            idx16 = T.alloc_ub([16], gate_dtype)  # beta staging width is fixed 16
            i_grid = T.alloc_ub([SCS, SCS], gate_dtype)
            j_grid = T.alloc_ub([SCS, SCS], gate_dtype)
            lt_m = T.alloc_ub([SCS, SCS], "uint8")  # lt[i,j] = (j <= i)
            slt_m = T.alloc_ub([SCS, SCS], "uint8")  # slt[i,j] = (j < i)
            # result processing
            r_ub = T.alloc_ub([2 * SCS, SCS], gate_dtype)
            r1_m = T.alloc_ub([SCS, SCS], gate_dtype)
            a_m = T.alloc_ub([SCS, SCS], gate_dtype)
            # drain staging, PER-HEAD REUSE: single-head [SCS, *]
            # buffers refilled/drained inside the h loop (replaces the
            # [BH_V, SCS, *] whole-task layout, 30KB -> 7.5KB; the
            # planner's ExtendKillIndex makes drain+produce buffers
            # coexist over the whole task loop, and the freed 22.5KB is
            # what unblocks the beta-fold scratch below)
            aqk_s = T.alloc_ub([SCS, CS], gate_dtype)
            akk_s = T.alloc_ub([SCS, AKK_W], gate_dtype)
            aqk_h = T.alloc_ub([SCS, CS], output_dtype)
            akk_h = T.alloc_ub([SCS, AKK_W], output_dtype)

            # ---- C-side buffers (full-slab L1 staging + dual
            # independent L0C slots; L0A/L0B single-slot -- the fp32 A
            # tile [AM, DK] fills the 64KB L0A budget at GGRP=4) ----
            A_l1 = T.alloc_L1([NG * AM, DK], op_dtype)
            B_l1 = T.alloc_L1([NG * BN, DK], op_dtype)
            l0a = T.alloc_L0A([AM, DK], op_dtype)
            l0b = T.alloc_L0B([DK, BN], op_dtype)
            C_l0c_0 = T.alloc_L0C([AM, BN], gate_dtype)  # even head groups
            C_l0c_1 = T.alloc_L0C([AM, BN], gate_dtype)  # odd head groups

            # ==== loop-invariant mask hoist (once per core) ====
            T.tile.arith_progression(idx, 0.0, 1.0, SCS)
            T.tile.broadcast(j_grid, idx, axis=0)  # j_grid[r, c] = c
            T.tile.broadcast(i_grid, idx, axis=1)  # i_grid[r, c] = r
            T.tile.compare(lt_m, i_grid, j_grid, "GE")  # r >= c (j <= i)
            T.tile.compare(slt_m, i_grid, j_grid, "GT")  # r > c (j < i)
            # The one-hot compare source [SCS,16] (j16b[r, c] = c)
            # is loop-invariant too -- hoisting its broadcast kills one
            # vector op per task section; the buffer's planner interval
            # is unchanged (ExtendKillIndex already extends it from the
            # peeled-0 touch to the loop tail, same as lt_m/slt_m).
            # idx16 (not idx!): the beta staging width is a FIXED 16
            # columns, so the broadcast source must be length 16 -- idx
            # only works at SCS == 16 by coincidence
            T.tile.arith_progression(idx16, 0.0, 1.0, 16)
            T.tile.broadcast(j16b, idx16, axis=0)

            # ==== peeled fill iteration 0: task cid, relay slot 0 ====
            # task coords (subchunk-major decomposition)
            sc = cid % NSC
            hg = (cid // NSC) % HG
            bb = cid // (NSC * HG)
            c = sc // (CS // SCS)
            i_s = sc % (CS // SCS)
            i_ts = c * CS + i_s * SCS
            h0v = hg * BH + vid * BH_V
            # -- V: produce task cid (beta staging + fill-anchored
            # accumulators + per-head factored construction + relay) --
            T.tile.fill(beta_h, 0.0)  # deterministic pad columns
            T.copy(Beta[bb, i_ts, h0v], beta_h[:, 0:BH_V])  # [SCS, BH_V] row-gap copy
            T.copy(beta_h, beta_f)  # UB<->UB cross-dtype cast copy (flat)
            T.tile.fill(a_agg, 0.0)
            T.tile.fill(b_agg, 0.0)
            for h in T.serial(BH_V):
                T.copy(Q[bb, i_ts : i_ts + SCS, h0v + h, :], q_h_row)
                T.copy(K[bb, i_ts : i_ts + SCS, h0v + h, :], k_h_row)
                T.copy(GK[bb, i_ts : i_ts + SCS, h0v + h, :], g2_row)
                T.tile.cast(q_f_row, q_h_row, "CAST_NONE", SCS * DK)
                T.tile.cast(k_f_row, k_h_row, "CAST_NONE", SCS * DK)
                # E- = exp((g_a - g_j) * ln2): the subtraction runs in the
                # ORIGINAL log2 domain BEFORE the ln2 scaling -- at large
                # |g| the scaled values round to the same fp32 (ulp of
                # |g*ln2| can exceed the gate step, e.g. g ~ -1e8 with a
                # step of 8) and the decay difference is lost; the
                # raw-domain difference of adjacent fp32 gates is exact
                # down to the 1-ulp boundary.
                T.tile.broadcast(bc_row, g2_row[anc : anc + 1, :], axis=0)
                T.tile.sub(e, bc_row, g2_row)
                T.tile.mul(e, e, LN2)
                T.tile.exp(e, e)
                T.tile.mul(kgj, k_f_row, e)  # K' = k * E-
                # E+ = exp((g_i - g_a) * ln2)
                T.tile.sub(e, g2_row, bc_row)
                T.tile.mul(e, e, LN2)
                T.tile.exp(e, e)
                T.tile.mul(q_f_row, q_f_row, e)  # Q' = q * E+
                T.tile.mul(e, k_f_row, e)  # K'' = k * E+
                # fold beta into K'' rows (Akk = (beta*K'') @ K'^T)
                # fold beta into K'' rows: one-hot select + row reduce +
                # column broadcast -- bit-identical to the former per-row
                # scalar muls (select keeps beta_f bits, reduce only adds
                # zeros) and kills 16 scalar GetValue + tiny Muls per head
                # (the largest V-side item before this change)
                T.tile.compare(eq_m, j16b, h, "EQ")
                T.tile.select(beta_sel, eq_m, beta_f, 0.0, "VSEL_TENSOR_SCALAR_MODE")
                T.reduce_sum(beta_sel, beta_col, dim=-1)
                T.tile.broadcast(bc_row, beta_col, axis=1)
                T.tile.mul(e, e, bc_row)
                # stage fp32 operands into the fill-anchored accumulators
                T.copy(q_f_row, a_agg[2 * SCS * h : 2 * SCS * h + SCS, :])
                T.copy(e, a_agg[2 * SCS * h + SCS : 2 * SCS * h + 2 * SCS, :])
                T.copy(kgj, b_agg[SCS * h : SCS * h + SCS, :])
            # per-vid aggregate -> GM relay slot 0 (auto cross-core flags)
            T.copy(a_agg, workspace_a[cid, 0, vid * AMV : vid * AMV + AMV, :])
            T.copy(b_agg, workspace_b[cid, 0, vid * BNV : vid * BNV + BNV, :])
            # -- C: block-diagonal gemm per head group -> o_relay slot 0
            # (ONE full-slab relay load per buffer + a decomposed
            # per-group sequence on double L0C slots, pass-driven flags) --
            T.copy(workspace_a[cid, 0, :, :], A_l1)
            T.copy(workspace_b[cid, 0, :, :], B_l1)
            for g in T.serial((NG + 1) // 2):
                gg = 2 * g
                T.copy(A_l1[gg * AM : (gg + 1) * AM, :], l0a)
                T.copy(B_l1[gg * BN : (gg + 1) * BN, :], l0b, transpose=True)
                T.mma(l0a, l0b, C_l0c_0, init=True)
                T.copy(C_l0c_0, o_relay[cid, 0, gg, :, :])
            if NG >= 2:
                for g in T.serial(NG // 2):
                    gg = 2 * g + 1
                    T.copy(A_l1[gg * AM : (gg + 1) * AM, :], l0a)
                    T.copy(B_l1[gg * BN : (gg + 1) * BN, :], l0b, transpose=True)
                    T.mma(l0a, l0b, C_l0c_1, init=True)
                    T.copy(C_l0c_1, o_relay[cid, 0, gg, :, :])
            T.set_cross_flag("FIX", FLAG_BASE)
            # -- V: trailing garbage drain (slot 2 % NSLOT, task cid's
            # region, NO wait): with NSLOT=2 this races C's FIX(0) and
            # writes stale/garbage values that the real drain at t == 2
            # (dt == cid) overwrites bit-identically; keeps the o_relay
            # V-side read-site counts balanced with the C-side writes.
            # per-head drain: fill/select/cast/writeback per h on
            # the single-head [SCS, *] buffers (bit-identical: same
            # elementwise ops on the same values; cleared-row zeros
            # re-established by the per-head fill) --
            w = i_s * SCS
            for h in T.serial(BH_V):
                gh = vid * BH_V + h
                gg = gh // GGRP
                wi = gh % GGRP
                T.copy(o_relay[cid, 2 % NSLOT, gg, 2 * SCS * wi : 2 * SCS * wi + 2 * SCS, SCS * wi : SCS * wi + SCS], r_ub)
                T.tile.fill(aqk_s, 0.0)
                T.tile.fill(akk_s, 0.0)
                T.tile.select(r1_m, lt_m, r_ub[0:SCS, :], 0.0, "VSEL_TENSOR_SCALAR_MODE")
                T.copy(r1_m, aqk_s[:, w : w + SCS])
                T.tile.select(a_m, slt_m, r_ub[SCS : 2 * SCS, :], 0.0, "VSEL_TENSOR_SCALAR_MODE")
                T.copy(a_m, akk_s[:, 0:SCS])
                T.tile.cast(aqk_h, aqk_s, "CAST_RINT", SCS * CS)
                T.tile.cast(akk_h, akk_s, "CAST_RINT", SCS * AKK_W)
                T.copy(aqk_h, Aqk[bb, i_ts : i_ts + SCS, h0v + h, :])
                T.copy(akk_h[:, 0:SCS], Akk[bb, i_ts : i_ts + SCS, h0v + h, :])

            # ==== peeled fill iteration 1: task cid + CORE_EFF, slot 1 ====
            pt1 = T.if_then_else(cid + CORE_EFF < NBLK, cid + CORE_EFF, cid)
            sc1 = pt1 % NSC
            hg1 = (pt1 // NSC) % HG
            bb1 = pt1 // (NSC * HG)
            c1 = sc1 // (CS // SCS)
            i_s1 = sc1 % (CS // SCS)
            i_ts1 = c1 * CS + i_s1 * SCS
            h0v1 = hg1 * BH + vid * BH_V
            # -- V: produce task pt1 -> relay slot 1 --
            T.tile.fill(beta_h, 0.0)
            T.copy(Beta[bb1, i_ts1, h0v1], beta_h[:, 0:BH_V])
            T.copy(beta_h, beta_f)
            T.tile.fill(a_agg, 0.0)
            T.tile.fill(b_agg, 0.0)
            for h in T.serial(BH_V):
                T.copy(Q[bb1, i_ts1 : i_ts1 + SCS, h0v1 + h, :], q_h_row)
                T.copy(K[bb1, i_ts1 : i_ts1 + SCS, h0v1 + h, :], k_h_row)
                T.copy(GK[bb1, i_ts1 : i_ts1 + SCS, h0v1 + h, :], g2_row)
                T.tile.cast(q_f_row, q_h_row, "CAST_NONE", SCS * DK)
                T.tile.cast(k_f_row, k_h_row, "CAST_NONE", SCS * DK)
                T.tile.broadcast(bc_row, g2_row[anc : anc + 1, :], axis=0)
                T.tile.sub(e, bc_row, g2_row)
                T.tile.mul(e, e, LN2)
                T.tile.exp(e, e)
                T.tile.mul(kgj, k_f_row, e)
                T.tile.sub(e, g2_row, bc_row)
                T.tile.mul(e, e, LN2)
                T.tile.exp(e, e)
                T.tile.mul(q_f_row, q_f_row, e)
                T.tile.mul(e, k_f_row, e)
                # fold beta into K'' rows: one-hot select + row reduce +
                # column broadcast -- bit-identical to the former per-row
                # scalar muls (select keeps beta_f bits, reduce only adds
                # zeros) and kills 16 scalar GetValue + tiny Muls per head
                # (the largest V-side item before this change)
                T.tile.compare(eq_m, j16b, h, "EQ")
                T.tile.select(beta_sel, eq_m, beta_f, 0.0, "VSEL_TENSOR_SCALAR_MODE")
                T.reduce_sum(beta_sel, beta_col, dim=-1)
                T.tile.broadcast(bc_row, beta_col, axis=1)
                T.tile.mul(e, e, bc_row)
                T.copy(q_f_row, a_agg[2 * SCS * h : 2 * SCS * h + SCS, :])
                T.copy(e, a_agg[2 * SCS * h + SCS : 2 * SCS * h + 2 * SCS, :])
                T.copy(kgj, b_agg[SCS * h : SCS * h + SCS, :])
            T.copy(a_agg, workspace_a[cid, 1 % NSLOT, vid * AMV : vid * AMV + AMV, :])
            T.copy(b_agg, workspace_b[cid, 1 % NSLOT, vid * BNV : vid * BNV + BNV, :])
            # -- C: gemm task pt1 -> o_relay slot 1 --
            T.copy(workspace_a[cid, 1 % NSLOT, :, :], A_l1)
            T.copy(workspace_b[cid, 1 % NSLOT, :, :], B_l1)
            for g in T.serial((NG + 1) // 2):
                gg = 2 * g
                T.copy(A_l1[gg * AM : (gg + 1) * AM, :], l0a)
                T.copy(B_l1[gg * BN : (gg + 1) * BN, :], l0b, transpose=True)
                T.mma(l0a, l0b, C_l0c_0, init=True)
                T.copy(C_l0c_0, o_relay[cid, 1 % NSLOT, gg, :, :])
            if NG >= 2:
                for g in T.serial(NG // 2):
                    gg = 2 * g + 1
                    T.copy(A_l1[gg * AM : (gg + 1) * AM, :], l0a)
                    T.copy(B_l1[gg * BN : (gg + 1) * BN, :], l0b, transpose=True)
                    T.mma(l0a, l0b, C_l0c_1, init=True)
                    T.copy(C_l0c_1, o_relay[cid, 1 % NSLOT, gg, :, :])
            T.set_cross_flag("FIX", FLAG_BASE + 1)
            # -- V: trailing garbage drain (slot 3 % NSLOT, task pt1's
            # region, NO wait; overwritten by the real drain at t == 3);
            # per-head drain (see the peeled-0 site) --
            w1 = i_s1 * SCS
            for h in T.serial(BH_V):
                gh = vid * BH_V + h
                gg = gh // GGRP
                wi = gh % GGRP
                T.copy(o_relay[cid, 3 % NSLOT, gg, 2 * SCS * wi : 2 * SCS * wi + 2 * SCS, SCS * wi : SCS * wi + SCS], r_ub)
                T.tile.fill(aqk_s, 0.0)
                T.tile.fill(akk_s, 0.0)
                T.tile.select(r1_m, lt_m, r_ub[0:SCS, :], 0.0, "VSEL_TENSOR_SCALAR_MODE")
                T.copy(r1_m, aqk_s[:, w1 : w1 + SCS])
                T.tile.select(a_m, slt_m, r_ub[SCS : 2 * SCS, :], 0.0, "VSEL_TENSOR_SCALAR_MODE")
                T.copy(a_m, akk_s[:, 0:SCS])
                T.tile.cast(aqk_h, aqk_s, "CAST_RINT", SCS * CS)
                T.tile.cast(akk_h, akk_s, "CAST_RINT", SCS * AKK_W)
                T.copy(aqk_h, Aqk[bb1, i_ts1 : i_ts1 + SCS, h0v1 + h, :])
                T.copy(akk_h[:, 0:SCS], Akk[bb1, i_ts1 : i_ts1 + SCS, h0v1 + h, :])

            # ==== steady-state pipeline loop: t = 2 .. TPT + 1 ====
            # V: drain(t-2) [manual flag from iteration t-2] -> produce(t)
            #    [auto-flag relay]; C: gemm(t) -> FIX(t) [manual flag].
            # Out-of-range produce/drain targets redirect branch-free to
            # the core's own first task (bit-identical recompute; garbage
            # drains are overwritten by the real ones).
            for tt in T.serial(TPT):
                t = tt + 2
                tid = cid + t * CORE_EFF
                pt = T.if_then_else(tid < NBLK, tid, cid)
                dt_raw = tid - 2 * CORE_EFF
                dt = T.if_then_else(dt_raw < NBLK, dt_raw, cid)
                slot = t % NSLOT  # produce/gemm slot == (t-2) % NSLOT drain slot
                dslot = (t + 2) % NSLOT  # == (t - 2) % NSLOT
                # drain task coords
                scd = dt % NSC
                hgd = (dt // NSC) % HG
                bbd = dt // (NSC * HG)
                cd = scd // (CS // SCS)
                i_sd = scd % (CS // SCS)
                i_tsd = cd * CS + i_sd * SCS
                h0vd = hgd * BH + vid * BH_V
                # produce task coords
                scp = pt % NSC
                hgp = (pt // NSC) % HG
                bbp = pt // (NSC * HG)
                cp = scp // (CS // SCS)
                i_sp = scp % (CS // SCS)
                i_tsp = cp * CS + i_sp * SCS
                h0vp = hgp * BH + vid * BH_V

                # -- V: drain task dt (post-process t-2's gemm results;
                # manual flag pairs with iteration t-2's FIX; the "V" pipe
                # argument attributes the wait to the Vector core for the
                # CombineCV resource verifier) --
                T.wait_cross_flag(FLAG_BASE + t % 2, "V")
                # per-head drain (see the peeled-0 site): fill/
                # select/cast/writeback per h on single-head buffers
                wd = i_sd * SCS
                for h in T.serial(BH_V):
                    gh = vid * BH_V + h
                    gg = gh // GGRP
                    wi = gh % GGRP
                    T.copy(o_relay[cid, dslot, gg, 2 * SCS * wi : 2 * SCS * wi + 2 * SCS, SCS * wi : SCS * wi + SCS], r_ub)
                    T.tile.fill(aqk_s, 0.0)
                    T.tile.fill(akk_s, 0.0)
                    T.tile.select(r1_m, lt_m, r_ub[0:SCS, :], 0.0, "VSEL_TENSOR_SCALAR_MODE")
                    T.copy(r1_m, aqk_s[:, wd : wd + SCS])
                    T.tile.select(a_m, slt_m, r_ub[SCS : 2 * SCS, :], 0.0, "VSEL_TENSOR_SCALAR_MODE")
                    T.copy(a_m, akk_s[:, 0:SCS])
                    T.tile.cast(aqk_h, aqk_s, "CAST_RINT", SCS * CS)
                    T.tile.cast(akk_h, akk_s, "CAST_RINT", SCS * AKK_W)
                    T.copy(aqk_h, Aqk[bbd, i_tsd : i_tsd + SCS, h0vd + h, :])
                    T.copy(akk_h[:, 0:SCS], Akk[bbd, i_tsd : i_tsd + SCS, h0vd + h, :])

                # -- V: produce task pt (construct + relay, slot t%NSLOT) --
                T.tile.fill(beta_h, 0.0)
                T.copy(Beta[bbp, i_tsp, h0vp], beta_h[:, 0:BH_V])
                T.copy(beta_h, beta_f)
                T.tile.fill(a_agg, 0.0)
                T.tile.fill(b_agg, 0.0)
                for h in T.serial(BH_V):
                    T.copy(Q[bbp, i_tsp : i_tsp + SCS, h0vp + h, :], q_h_row)
                    T.copy(K[bbp, i_tsp : i_tsp + SCS, h0vp + h, :], k_h_row)
                    T.copy(GK[bbp, i_tsp : i_tsp + SCS, h0vp + h, :], g2_row)
                    T.tile.cast(q_f_row, q_h_row, "CAST_NONE", SCS * DK)
                    T.tile.cast(k_f_row, k_h_row, "CAST_NONE", SCS * DK)
                    T.tile.broadcast(bc_row, g2_row[anc : anc + 1, :], axis=0)
                    T.tile.sub(e, bc_row, g2_row)
                    T.tile.mul(e, e, LN2)
                    T.tile.exp(e, e)
                    T.tile.mul(kgj, k_f_row, e)
                    T.tile.sub(e, g2_row, bc_row)
                    T.tile.mul(e, e, LN2)
                    T.tile.exp(e, e)
                    T.tile.mul(q_f_row, q_f_row, e)
                    T.tile.mul(e, k_f_row, e)
                    # fold beta into K'' rows: one-hot select + row reduce +
                    # column broadcast -- bit-identical to the former per-row
                    # scalar muls (select keeps beta_f bits, reduce only adds
                    # zeros) and kills 16 scalar GetValue + tiny Muls per head
                    # (the largest V-side item before this change)
                    T.tile.compare(eq_m, j16b, h, "EQ")
                    T.tile.select(beta_sel, eq_m, beta_f, 0.0, "VSEL_TENSOR_SCALAR_MODE")
                    T.reduce_sum(beta_sel, beta_col, dim=-1)
                    T.tile.broadcast(bc_row, beta_col, axis=1)
                    T.tile.mul(e, e, bc_row)
                    T.copy(q_f_row, a_agg[2 * SCS * h : 2 * SCS * h + SCS, :])
                    T.copy(e, a_agg[2 * SCS * h + SCS : 2 * SCS * h + 2 * SCS, :])
                    T.copy(kgj, b_agg[SCS * h : SCS * h + SCS, :])
                T.copy(a_agg, workspace_a[cid, slot, vid * AMV : vid * AMV + AMV, :])
                T.copy(b_agg, workspace_b[cid, slot, vid * BNV : vid * BNV + BNV, :])

                # -- C: gemm task pt -> o_relay slot t%NSLOT (
                # full-slab load + double-slot decomposed sequence; the
                # steady state overlaps mma(g1)/fixpipe(g0) and the next
                # task's slab prefetch) --
                T.copy(workspace_a[cid, slot, :, :], A_l1)
                T.copy(workspace_b[cid, slot, :, :], B_l1)
                for g in T.serial((NG + 1) // 2):
                    gg = 2 * g
                    T.copy(A_l1[gg * AM : (gg + 1) * AM, :], l0a)
                    T.copy(B_l1[gg * BN : (gg + 1) * BN, :], l0b, transpose=True)
                    T.mma(l0a, l0b, C_l0c_0, init=True)
                    T.copy(C_l0c_0, o_relay[cid, slot, gg, :, :])
                if NG >= 2:
                    for g in T.serial(NG // 2):
                        gg = 2 * g + 1
                        T.copy(A_l1[gg * AM : (gg + 1) * AM, :], l0a)
                        T.copy(B_l1[gg * BN : (gg + 1) * BN, :], l0b, transpose=True)
                        T.mma(l0a, l0b, C_l0c_1, init=True)
                        T.copy(C_l0c_1, o_relay[cid, slot, gg, :, :])
                T.set_cross_flag("FIX", FLAG_BASE + t % 2)

    return main


def fla_chunk_intra_token_parallel(q, k, gk, beta, chunk_size=64, sub_chunk_size=16):
    """Host wrapper: domain asserts (pure Python integer/dtype checks) +
    JIT launch. Returns (Aqk [B,S,H,chunk_size], Akk [B,S,H,sub_chunk_size]),
    both bfloat16, allocated by tilelang (out_idx=[-2, -1])."""
    B, S, H, DK = q.shape
    CS, SCS = chunk_size, sub_chunk_size

    if S % CS != 0:
        raise ValueError(f"S={S} must be divisible by chunk_size={CS}")
    if CS % SCS != 0:
        raise ValueError(f"chunk_size={CS} must be divisible by sub_chunk_size={SCS}")
    if SCS < 1 or CS < SCS:
        raise ValueError(f"invalid chunk/sub_chunk sizes: {CS}/{SCS}")
    if CS & (CS - 1) or SCS & (SCS - 1):
        # tile alignment: the [SCS, *] staging rows and the [SCS, CS]
        # drain row must be pow2 (non-pow2 chunk sizes crash the TVM
        # compiler instead of failing cleanly)
        raise ValueError(f"chunk_size={CS} and sub_chunk_size={SCS} must be powers of two")
    if DK % 16 != 0 or DK < 32:
        raise ValueError(f"DK={DK} must be a multiple of 16 and >= 32")
    if H % 2 != 0:
        raise ValueError(f"H={H} must be divisible by 2 (minimum head group)")
    BH = _pick_bh(H, SCS, DK, CS)
    if BH == 0:
        raise ValueError(
            f"DK={DK} exceeds the 192KB unified-buffer budget at every "
            f"head-group size (SCS={SCS}: even BH=2 needs "
            f"{_ub_bytes(1, SCS, DK, CS)} bytes of coexisting buffers)"
        )
    if BH * SCS < 16:
        # gemm_v0 L0B/L0C N-fractal lower bound with GGRP <= BH
        # (excludes only SCS < 8 at BH=2 corners)
        raise ValueError(f"BH*SCS={BH * SCS} must be >= 16 (gemm N-fractal lower bound; got BH={BH}, SCS={SCS})")
    if q.dtype is not torch.bfloat16 or k.dtype is not torch.bfloat16:
        raise ValueError("q/k must be bfloat16")
    if gk.dtype is not torch.float32:
        raise ValueError("gk must be float32")
    if beta.dtype is not torch.bfloat16:
        raise ValueError("beta must be bfloat16")
    for name, t in (("q", q), ("k", k), ("gk", gk), ("beta", beta)):
        if not t.is_contiguous():
            raise ValueError(f"input '{name}' must be contiguous")

    kernel = fla_chunk_intra_token_parallel_kernel(B, S, H, DK, CS, SCS, BH)
    return kernel(q, k, gk, beta)


if __name__ == "__main__":
    # Smoke entry (the CI legacy runner executes this file with `python`):
    # minimal fixed-length shape, output-shape/finite checks, "Test Passed!"
    # on success. Numerical accuracy is covered by
    # test_example_fla_chunk_intra_token_parallel.py (PyTorch golden).
    import torch_npu  # noqa: F401  (registers .npu() on torch)

    tilelang.cache.clear_cache()
    B, S, H, DK, CS, SCS = 1, 64, 8, 64, 64, 16

    gen = torch.Generator().manual_seed(42)
    q = torch.randn(B, S, H, DK, generator=gen).to(torch.bfloat16)
    k = torch.randn(B, S, H, DK, generator=gen).to(torch.bfloat16)
    beta = torch.randn(B, S, H, generator=gen).to(torch.bfloat16)
    gk = torch.nn.functional.logsigmoid(torch.randn(B, S, H, DK, generator=gen, dtype=torch.float32))
    gk = gk.view(B, S // CS, CS, H, DK).cumsum(dim=2).view(B, S, H, DK)

    aqk, akk = fla_chunk_intra_token_parallel(q.npu(), k.npu(), gk.npu(), beta.npu())
    torch.npu.synchronize()
    assert aqk.shape == (B, S, H, CS) and aqk.dtype is torch.bfloat16
    assert akk.shape == (B, S, H, SCS) and akk.dtype is torch.bfloat16
    for name, out in (("Aqk", aqk), ("Akk", akk)):
        assert torch.isfinite(out.float()).all().item(), f"{name} contains NaN/Inf"
    print(f"[SMOKE_PASS] fla_chunk_intra_token_parallel B={B} S={S} H={H} DK={DK} chunk={CS} sub_chunk={SCS} bf16")
    print("Test Passed!")


if __name__ == "__main__":
    # Smoke entry (the CI legacy runner executes this file with `python`):
    # minimal fixed-length shape, output-shape/finite checks, "Test Passed!"
    # on success. Numerical accuracy is covered by
    # test_example_fla_chunk_intra_token_parallel.py (PyTorch golden).
    import torch_npu  # noqa: F401  (registers .npu() on torch)

    tilelang.cache.clear_cache()
    B, S, H, DK, CS, SCS = 1, 64, 8, 64, 64, 16

    gen = torch.Generator().manual_seed(42)
    q = torch.randn(B, S, H, DK, generator=gen).to(torch.bfloat16)
    k = torch.randn(B, S, H, DK, generator=gen).to(torch.bfloat16)
    beta = torch.randn(B, S, H, generator=gen).to(torch.bfloat16)
    gk = torch.nn.functional.logsigmoid(torch.randn(B, S, H, DK, generator=gen, dtype=torch.float32))
    gk = gk.view(B, S // CS, CS, H, DK).cumsum(dim=2).view(B, S, H, DK)

    aqk, akk = fla_chunk_intra_token_parallel(q.npu(), k.npu(), gk.npu(), beta.npu())
    torch.npu.synchronize()
    assert aqk.shape == (B, S, H, CS) and aqk.dtype is torch.bfloat16
    assert akk.shape == (B, S, H, SCS) and akk.dtype is torch.bfloat16
    for name, out in (("Aqk", aqk), ("Akk", akk)):
        assert torch.isfinite(out.float()).all().item(), f"{name} contains NaN/Inf"
    print(f"[SMOKE_PASS] fla_chunk_intra_token_parallel B={B} S={S} H={H} DK={DK} chunk={CS} sub_chunk={SCS} bf16")
    print("Test Passed!")
