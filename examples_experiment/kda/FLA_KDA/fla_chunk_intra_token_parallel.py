"""fla_chunk_intra_token_parallel: KDA intra-chunk per-token decay inner
products Aqk/Akk on Ascend 910B3 (Cube GEMM offload + fixed-core C/V
software pipeline).

Semantics (upstream FLA/FLA_KDA Triton kernel; for token i in subchunk
[i_ts, i_ts+SCS), j in [i_ts, min(i+1, i_ts+SCS))):

    kgj = k[b, j] * 2^(g[b, i] - g[b, j])
    Aqk[b, i, h, j % CS]   = scale * sum_d q[b, i, h, d] * kgj[h, d]
    Akk[b, i, h, j - i_ts] = (j < i) ? sum_d (beta*k[b, i]) * kgj : 0

Aqk columns not covered by any j stay 0; the Akk diagonal stays 0.

Implementation, in brief:
- Tasks are (batch, subchunk, head-group) on a 1-D fixed-core grid
  (cid, vid), vid splitting the BH head group across the 2 AIVs.
- Numerics: exp2(x) = exp(x * ln2) with the anchor subtraction done in
  the RAW log2 domain BEFORE the ln2 scaling (scaled differences
  collapse to identical fp32 at large |g|); mid-anchor factored decay
  e_ij = E+_i * E-_j (|g_i - g_anchor| < ~127 keeps E finite); GEMM
  operands stay fp32 (bf16 rounding alone fails the Akk gate); outputs
  stage fp32 and CAST_RINT to bf16 in UB.
- The dot products are offloaded to the Cube cores as block-diagonal
  batched GEMMs per GGRP-head group (A = [Q'; K''], B = K'; the useful
  output is the per-head diagonal blocks). GGRP must DIVIDE BH and fit
  the L0A budget (GGRP*2*SCS*DK*4B <= 64KB) with GGRP*SCS >= 16.
- Post-GEMM masking is a V-side compare+select (garbage cells can be
  +-inf under deep decay; a multiply mask would give inf*0 = NaN).
- Fixed-core task loop with a 2-task-deferred drain: V post-processes
  task t-2 while constructing task t; C gemms task t. Relays are
  double-buffered per-core ping-pong slots (workspace_a/b with auto
  CV-sync; o_relay with manual alternating cross-core flags). Peeled
  fill iterations avoid the t=0 drain deadlock.
- The C side consumes the relays with the high-level T.gemm_v0
  intrinsic (one block-diagonal call per GGRP-head group, operands
  loaded from the relay slots into L1, results staged through one L0C
  accumulator into o_relay).

Supported domain: fixed-length sequences, S % CS == 0, CS % SCS == 0,
CS and SCS powers of two, DK % 16 == 0 with 32 <= DK <= 128 (the 192KB
UB coexistence at the largest head group), BH * SCS >= 16; BH adapts
8 -> 4 -> 2 under head divisibility. q/k/beta bfloat16 + gk float32
in, bfloat16 out; varlen out of scope.
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


def _pick_head_group(H):
    """BH is a performance tiling parameter, not a dispatch predicate:
    pick the largest supported group that divides H (8 -> 4 -> 2)."""
    for bh in (8, 4, 2):
        if H % bh == 0:
            return bh
    return 0


def _pick_ggrp(BH, SCS, DK):
    """Heads per gemm group: amortize the per-call cube overhead with
    block-diagonal batching. GGRP MUST DIVIDE BH -- the block-diagonal
    tiling covers the BH heads exactly, and a non-divisor leaves tail
    heads reading out-of-range gemm groups in o_relay (with inconsistent
    workspace/L1 row counts). Among the divisors of BH, prefer the
    largest that satisfies gemm_v0's L0A/L0B ping-pong slot budget and
    the L0B/L0C N >= 16 fractal rule.

    Slot budget (mirrors the static_asserts in gemm_v0): the slots are
    sized (M x kL0Size) / (nTile x kL0Size) with kL0Size FIXED at 128 --
    even when K < kL0Size the slot still reserves the full 128 columns --
    and the per-slot budget halves (64KB -> 32KB) once K needs more than
    one L0 step. With fp32 operands this bounds M = GGRP*2*SCS to 128
    elements at DK <= 128 (and 64 above)."""
    k_l0 = 128
    steps = (DK + k_l0 - 1) // k_l0
    slot_elems = (65536 // 4) // (2 if steps > 1 else 1)
    for g in (8, 4, 2, 1):
        if BH % g != 0:
            continue
        if g * 2 * SCS * k_l0 > slot_elems:  # A tile: M x kL0Size
            continue
        if g * SCS * k_l0 > slot_elems:  # B tile: N x kL0Size
            continue
        if g * SCS < 16:  # L0B/L0C N-fractal lower bound
            continue
        return g
    raise ValueError(
        f"no divisor of BH={BH} satisfies the gemm constraints (L0A/L0B slot budget at DK={DK} and N=GGRP*SCS >= 16) with SCS={SCS}"
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

            # ---- C-side buffers: per-group L1 operand staging + the
            # single L0C accumulator backing T.gemm_v0 ----
            A_l1 = T.alloc_L1([AM, DK], op_dtype)
            B_l1 = T.alloc_L1([BN, DK], op_dtype)
            C_l0c = T.alloc_L0C([AM, BN], gate_dtype)

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
            # -- C: block-diagonal gemm per head group -> o_relay slot 0 --
            for gg in T.serial(NG):
                T.copy(workspace_a[cid, 0, gg * AM : gg * AM + AM, :], A_l1)
                T.copy(workspace_b[cid, 0, gg * BN : gg * BN + BN, :], B_l1)
                T.gemm_v0(A_l1, B_l1, C_l0c, transpose_B=True, init=True)
                T.copy(C_l0c, o_relay[cid, 0, gg, :, :])
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
            for gg in T.serial(NG):
                T.copy(workspace_a[cid, 1 % NSLOT, gg * AM : gg * AM + AM, :], A_l1)
                T.copy(workspace_b[cid, 1 % NSLOT, gg * BN : gg * BN + BN, :], B_l1)
                T.gemm_v0(A_l1, B_l1, C_l0c, transpose_B=True, init=True)
                T.copy(C_l0c, o_relay[cid, 1 % NSLOT, gg, :, :])
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

                # -- C: gemm task pt -> o_relay slot t%NSLOT --
                for gg in T.serial(NG):
                    T.copy(workspace_a[cid, slot, gg * AM : gg * AM + AM, :], A_l1)
                    T.copy(workspace_b[cid, slot, gg * BN : gg * BN + BN, :], B_l1)
                    T.gemm_v0(A_l1, B_l1, C_l0c, transpose_B=True, init=True)
                    T.copy(C_l0c, o_relay[cid, slot, gg, :, :])
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
    # 192KB UB coexistence at the largest head group (BH=8) bounds DK to
    # 128; a larger DK would need a smaller head group, which is out of
    # scope for this example (the memory planner still hard-fails on any
    # shape that slips through, so this gate is a readable early error).
    if DK > 128:
        raise ValueError(f"DK={DK} exceeds the 192KB unified-buffer budget of the BH=8 head group (supported: DK <= 128)")
    if H % 2 != 0:
        raise ValueError(f"H={H} must be divisible by 2 (minimum head group)")
    BH = _pick_head_group(H)
    if BH == 0:
        raise ValueError(f"H={H} must be divisible by 2 (minimum head group)")
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
    # test_fla_chunk_intra_token_parallel.py (PyTorch golden).
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
