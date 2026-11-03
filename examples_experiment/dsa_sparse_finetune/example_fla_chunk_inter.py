"""fla_chunk_inter: KDA chunked backward inter-chunk gradients (dq/dk/dw/dg).

Single fused Cube+Vector kernel, Developer mode (alloc_shared/fragment +
auto C/V split), for the flash-linear-attention KDA backward on Ascend
910B3. Semantics follow the upstream Triton reference
(chunk_kda_bwd_kernel_inter, fixed length, all fp32):

    dq = (do @ h^T) * scale * 2^g             [64, DK] per chunk/head
    dk = (v  @ dh^T) * 2^(gn - g)              gn = g[chunk last row]
    dw = -(dv @ h^T)                           sign folded into the dv load
    dg = q*dq - k*dk + m_last * pp,            pp = 2^gn * sum_v(h*dh)
                                               + sum_t(dk * k)

q/k/g [B,S,H,DK], v/dv/do [B,S,H,DV], h/dh [B,S//64,H,DK,DV] -> dq/dk/dw/dg
[B,S,H,DK], all fp32, chunk_size = 64 (S % 64 == 0 asserted: a trailing
partial chunk has no h/dh state and is undefined in the reference).

Performance structure (bench B=1, S=8192, H=64, DK=DV=128: 19.556 ms ->
6.742 ms, NPU-event mean of 10 reps after 20 warmups):

* FIXED-CORE grid: T.Kernel(CORE_EFF <= 20, threads=1) with one in-core
  T.serial task loop. The 8192 logical (chunk, b, h) tasks are striped
  over the cores; one block instance per core amortizes dispatch/TPipe
  overhead and keeps the relay slot ring permanently L2-resident.
* NK-MERGED TASKS: one task per (b, chunk, head) covers ALL dk-blocks, so
  the DK-less operands (v/do/dv) load/split/relay once instead of NK
  times, and the Cube side holds the left (A) gemm operands in L1 across
  the per-bk groups.
* EXPLICIT RELAYS: 10 fp16 operand relays (V->C, auto cross-core flags
  paired by the workspace naming) + ONE fused fp32 drain relay (C->V,
  dq|dk|diag in a single readback) under a manual 1-late cross-core flag.
* 1-LATE FLAG PAIRING: drain(j) consumes the relays that iteration j-1
  produced (prologue head-set primes j=0), so the Vector side never
  stalls through the Cube's same-iteration consume chain.
* BATCHED LEFT LOADS: the three GM loads fire back-to-back and the hi/lo
  split chains run afterwards, keeping the MTE2 stream continuous.

Toolchain-mandated implementation notes (all verified on this repo):
- scratch-free red line: ZERO T.reduce_sum / T.tile.broadcast. Those ops
  trigger the auto memory planner's injected shared scratch, whose UB
  packing is compile-context dependent and can land illegal placements
  (device Illegal instruction). Substitutes: the column reduce is a
  literal halving tree; the row-vector gate is transpose +
  row_expand_sub (bDK=64) or an unrolled row-Sub chain (bDK=32); the
  last-row injection is a single-row add; diag(h@dh^T) is a Cube-side
  gemm group + gather extraction.
- fp32 gemm_v0 operands are broken on this toolchain -> fp16 hi/lo
  3-gemm per logical GEMM (x = hi + lo) with fp32 L0C accumulation
  (relative error ~2^-22 per operand pair).
- exp2 has no Ascend backend -> 2^x is exp(x * ln2) with ln2 pre-folded
  on g/gn.
- block_DK is an adaptive performance parameter (64 when DK%64==0 and
  DV>=128, else 32); both paths cover the whole declared domain.
- MANUAL FLAG SLOT 10: the 10 operand relays make the auto allocator
  occupy flag ids 0..9, so the manual drain flag uses the single
  uncollided slot 10 (ids 11..14 are system-reserved).

MAINTENANCE NOTE: the exact text of this file influences the auto memory
planner's UB packing. The current text lands a verified-legal packing;
do not reformat or reword casually -- after any edit, re-run the minimal
gate case (B=1, S=128, H=2, DK=64, DV=64) and the bench case to confirm
the packing still compiles and runs, and never reintroduce
T.reduce_sum / T.tile.broadcast.
"""

import tilelang
import torch
from tilelang import language as T

# 2^x is computed as exp(x * ln2): exp2 has no Ascend backend mapping.
LN2 = 0.6931471805599453
# fixed-core grid ceiling (910B3 AI Cube cores) and relay slot-ring depth
# (cube lag upper bound 1 + margin).
CORE_NUM = 20
NSLOT = 4

PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}


def _check_shape(B, S, H, DK, DV, chunk_size, block_DK=None):
    """Domain gate: deterministic rejection replaces the reference
    implementation's undefined OOB behavior for S % 64 != 0.

    Returns the adaptive block_DK (a performance parameter, not a dispatch
    gate: both the 64 and 32 paths cover the whole declared domain).
    """
    if chunk_size != 64:
        raise ValueError(f"chunk_size={chunk_size} unsupported: only 64 (fixed chunked data model)")
    if S % chunk_size != 0:
        raise ValueError(
            f"S={S} not divisible by chunk_size={chunk_size}: mathematically undefined domain (a trailing partial chunk has no h/dh state)"
        )
    if DK % 32 != 0:
        raise ValueError(f"DK={DK} must be a multiple of 32 (block_DK exact split)")
    if DV % 16 != 0 or DV < 64:
        raise ValueError(f"DV={DV} must be a multiple of 16 and >= 64 (gemm K fractals + 32B alignment)")
    if B < 1 or H < 1:
        raise ValueError(f"B={B}, H={H} must be >= 1")
    if block_DK is None:
        block_DK = 64 if (DK % 64 == 0 and DV >= 128) else 32
    if block_DK not in (32, 64):
        raise ValueError(f"block_DK={block_DK} must be 32 or 64")
    if DK % block_DK != 0:
        raise ValueError(f"DK={DK} not divisible by block_DK={block_DK}")
    return block_DK


# ============================================================================
# single fused kernel: out_idx=[19..22] (dq/dk/dw/dg), workspace_idx=[8..18]
# (10 operand relays + the fused ddr relay; absolute indices, never negative).
# ============================================================================
@tilelang.jit(
    out_idx=[19, 20, 21, 22],
    workspace_idx=[8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
    pass_configs=PASS_CONFIGS,
)
def fla_chunk_inter(B, S, H, DK, DV, scale=1.0, chunk_size=64, block_DK=None):
    block_DK = _check_shape(B, S, H, DK, DV, chunk_size, block_DK)
    BS = chunk_size
    NT = S // BS
    NK = DK // block_DK
    GRID = NT * B * H
    # fixed-core striped decomposition: core cid handles tasks cid + t*CORE_EFF
    CORE_EFF = min(CORE_NUM, GRID)
    L = GRID // CORE_EFF + 2  # +2 slack: 1-late drain tail + redirect redo
    WS_L = (CORE_EFF, NSLOT, BS, DV)  # fp16 left-operand relay (one slot/task)
    WS_R = (CORE_EFF, NSLOT * NK, block_DK, DV)  # fp16 right relay (flat task,bk)
    WS_F = (CORE_EFF, NSLOT * NK, 2 * BS + block_DK, block_DK)  # fused drain relay

    @T.prim_func
    def main(
        Q: T.Tensor((B, S, H, DK), "float32"),  # type: ignore
        K: T.Tensor((B, S, H, DK), "float32"),  # type: ignore
        V: T.Tensor((B, S, H, DV), "float32"),  # type: ignore
        G: T.Tensor((B, S, H, DK), "float32"),  # type: ignore
        h: T.Tensor((B, NT, H, DK, DV), "float32"),  # type: ignore
        dv: T.Tensor((B, S, H, DV), "float32"),  # type: ignore
        DO: T.Tensor((B, S, H, DV), "float32"),  # type: ignore
        Dh: T.Tensor((B, NT, H, DK, DV), "float32"),  # type: ignore
        workspace_1: T.Tensor(WS_L, "float16"),  # type: ignore  # do hi
        workspace_2: T.Tensor(WS_L, "float16"),  # type: ignore  # do lo
        workspace_3: T.Tensor(WS_R, "float16"),  # type: ignore  # h hi (dq+dw share)
        workspace_4: T.Tensor(WS_R, "float16"),  # type: ignore  # h lo (dq+dw share)
        workspace_5: T.Tensor(WS_L, "float16"),  # type: ignore  # v hi
        workspace_6: T.Tensor(WS_L, "float16"),  # type: ignore  # v lo
        workspace_7: T.Tensor(WS_R, "float16"),  # type: ignore  # dh hi
        workspace_8: T.Tensor(WS_R, "float16"),  # type: ignore  # dh lo
        workspace_9: T.Tensor(WS_L, "float16"),  # type: ignore  # (-dv) hi
        workspace_10: T.Tensor(WS_L, "float16"),  # type: ignore  # (-dv) lo
        ddr_relay: T.Tensor(WS_F, "float32"),  # type: ignore
        dq: T.Tensor((B, S, H, DK), "float32"),  # type: ignore
        dk: T.Tensor((B, S, H, DK), "float32"),  # type: ignore
        dw: T.Tensor((B, S, H, DK), "float32"),  # type: ignore
        dg: T.Tensor((B, S, H, DK), "float32"),  # type: ignore
    ):
        with T.Kernel(CORE_EFF, threads=1, is_npu=True) as (cid):
            # ---- L0C frags (fp32 accumulators, reused every iteration) ----
            dq_frag = T.alloc_fragment((BS, block_DK), "float32")
            dk_frag = T.alloc_fragment((BS, block_DK), "float32")
            dw_frag = T.alloc_fragment((BS, block_DK), "float32")
            dgm_frag = T.alloc_fragment((block_DK, block_DK), "float32")
            # ---- produce side: fp32 load domain ----
            h_ub = T.alloc_shared((block_DK, DV), "float32")
            dh_ub = T.alloc_shared((block_DK, DV), "float32")
            v_ub = T.alloc_shared((BS, DV), "float32")
            do_ub = T.alloc_shared((BS, DV), "float32")
            dvn_ub = T.alloc_shared((BS, DV), "float32")
            # shared split temps (left / right operands)
            t_hi = T.alloc_shared((BS, DV), "float32")
            t_lo = T.alloc_shared((BS, DV), "float32")
            th_hi = T.alloc_shared((block_DK, DV), "float32")
            th_lo = T.alloc_shared((block_DK, DV), "float32")
            # fp16 hi/lo staging = dedicated relay sources (one per workspace;
            # dead once relayed -- the workspace pass keys relays by name)
            do_hi_ub = T.alloc_shared((BS, DV), "float16")
            do_lo_ub = T.alloc_shared((BS, DV), "float16")
            v_hi_ub = T.alloc_shared((BS, DV), "float16")
            v_lo_ub = T.alloc_shared((BS, DV), "float16")
            dvn_hi_ub = T.alloc_shared((BS, DV), "float16")
            dvn_lo_ub = T.alloc_shared((BS, DV), "float16")
            h_hi_ub = T.alloc_shared((block_DK, DV), "float16")
            h_lo_ub = T.alloc_shared((block_DK, DV), "float16")
            dh_hi_ub = T.alloc_shared((block_DK, DV), "float16")
            dh_lo_ub = T.alloc_shared((block_DK, DV), "float16")
            # ---- L1 gemm operands (fp16; left reused across bk) ----
            do_hi_l1 = T.alloc_shared((BS, DV), "float16")
            do_lo_l1 = T.alloc_shared((BS, DV), "float16")
            v_hi_l1 = T.alloc_shared((BS, DV), "float16")
            v_lo_l1 = T.alloc_shared((BS, DV), "float16")
            dvn_hi_l1 = T.alloc_shared((BS, DV), "float16")
            dvn_lo_l1 = T.alloc_shared((BS, DV), "float16")
            h_hi_l1 = T.alloc_shared((block_DK, DV), "float16")
            h_lo_l1 = T.alloc_shared((block_DK, DV), "float16")
            dh_hi_l1 = T.alloc_shared((block_DK, DV), "float16")
            dh_lo_l1 = T.alloc_shared((block_DK, DV), "float16")
            # ---- drain side ----
            ddr_ub = T.alloc_shared((2 * BS + block_DK, block_DK), "float32")
            dgkn_ub = T.alloc_shared((block_DK,), "float32")
            gn_ub = T.alloc_shared((block_DK,), "float32")
            gnexp_ub = T.alloc_shared((block_DK,), "float32")
            dgk_off_i32 = T.alloc_shared((block_DK,), "int32")
            dgk_off_u32 = T.alloc_shared((block_DK,), "uint32")
            g_ub = T.alloc_shared((BS, block_DK), "float32")
            gexp_ub = T.alloc_shared((BS, block_DK), "float32")
            gnd_ub = T.alloc_shared((BS, block_DK), "float32")
            q_ub = T.alloc_shared((BS, block_DK), "float32")
            k_ub = T.alloc_shared((BS, block_DK), "float32")
            dkk_ub = T.alloc_shared((BS, block_DK), "float32")
            dg_ub = T.alloc_shared((BS, block_DK), "float32")
            gexp_tmp = T.alloc_shared((block_DK, 8), "float32")

            # ---- prologue: gather byte offsets (live code) + head-set that
            # primes drain(0)'s 1-late wait (flag counts stay balanced) ----
            T.tile.arith_progression(dgk_off_i32, 0, (block_DK + 1) * 4, block_DK)
            T.reinterpretcast(dgk_off_u32, dgk_off_i32, "uint32_t")
            # Manual drain flag on the dedicated slot 10: the 10 operand
            # relays make the auto flag allocator occupy ids 0..9, and ids
            # 11..14 are system-reserved -- slot 10 is the only uncollided
            # user slot (its GetffstMsg 0xf mask is itself). A single
            # dedicated slot keeps the 1-late protocol sound: every wait is
            # guaranteed the previous iteration's set fired (pending <= 2).
            T.set_cross_flag("FIX", 10)

            # ============ single task loop: produce(j) + drain(j-1) ============
            for j in T.serial(L):
                tid_p = cid + j * CORE_EFF
                pt = T.if_then_else(tid_p < GRID, tid_p, cid)  # redirect redo
                slot = j % NSLOT
                # ---- produce task pt ----
                bh = pt % H
                bt = pt // H
                bs = bt % NT
                bb = bt // NT
                s0 = bs * BS
                # right operands per bk: load -> hi/lo split -> flat (task,bk) relay
                for bk in range(NK):
                    dk0 = bk * block_DK
                    T.copy(h[bb, bs, bh, dk0 : dk0 + block_DK, 0:DV], h_ub)
                    T.copy(Dh[bb, bs, bh, dk0 : dk0 + block_DK, 0:DV], dh_ub)
                    # dh split (cast -> sub -> cast, all UB<->UB)
                    T.copy(dh_ub, dh_hi_ub)
                    T.copy(dh_hi_ub, th_hi)
                    T.tile.sub(th_lo, dh_ub, th_hi)
                    T.copy(th_lo, dh_lo_ub)
                    T.copy(dh_hi_ub, workspace_7[cid, slot * NK + bk, 0:block_DK, 0:DV])
                    T.copy(dh_lo_ub, workspace_8[cid, slot * NK + bk, 0:block_DK, 0:DV])
                    # h split (the dq AND dw groups share this relay: one UB
                    # source -> single L1 destination)
                    T.copy(h_ub, h_hi_ub)
                    T.copy(h_hi_ub, th_hi)
                    T.tile.sub(th_lo, h_ub, th_hi)
                    T.copy(th_lo, h_lo_ub)
                    T.copy(h_hi_ub, workspace_3[cid, slot * NK + bk, 0:block_DK, 0:DV])
                    T.copy(h_lo_ub, workspace_4[cid, slot * NK + bk, 0:block_DK, 0:DV])
                # left operands once per task (NK-merge: no DK axis).
                # BATCHED LOADS: the three GM loads fire back-to-back (MTE2
                # burst) and the split chains run afterwards, so the load
                # stream is not interleaved with split vec ops.
                T.copy(DO[bb, s0 : s0 + BS, bh, 0:DV], do_ub)
                T.copy(V[bb, s0 : s0 + BS, bh, 0:DV], v_ub)
                T.copy(dv[bb, s0 : s0 + BS, bh, 0:DV], dvn_ub)
                T.tile.mul(dvn_ub, dvn_ub, -1.0)  # dw sign fold
                # do split
                T.copy(do_ub, do_hi_ub)
                T.copy(do_hi_ub, t_hi)
                T.tile.sub(t_lo, do_ub, t_hi)
                T.copy(t_lo, do_lo_ub)
                T.copy(do_hi_ub, workspace_1[cid, slot, 0:BS, 0:DV])
                T.copy(do_lo_ub, workspace_2[cid, slot, 0:BS, 0:DV])
                # v split
                T.copy(v_ub, v_hi_ub)
                T.copy(v_hi_ub, t_hi)
                T.tile.sub(t_lo, v_ub, t_hi)
                T.copy(t_lo, v_lo_ub)
                T.copy(v_hi_ub, workspace_5[cid, slot, 0:BS, 0:DV])
                T.copy(v_lo_ub, workspace_6[cid, slot, 0:BS, 0:DV])
                # dvn split
                T.copy(dvn_ub, dvn_hi_ub)
                T.copy(dvn_hi_ub, t_hi)
                T.tile.sub(t_lo, dvn_ub, t_hi)
                T.copy(t_lo, dvn_lo_ub)
                T.copy(dvn_hi_ub, workspace_9[cid, slot, 0:BS, 0:DV])
                T.copy(dvn_lo_ub, workspace_10[cid, slot, 0:BS, 0:DV])
                # ---- C: left relays -> L1 (once per task, A reused across bk) ----
                T.copy(workspace_1[cid, slot, 0:BS, 0:DV], do_hi_l1)
                T.copy(workspace_2[cid, slot, 0:BS, 0:DV], do_lo_l1)
                T.copy(workspace_5[cid, slot, 0:BS, 0:DV], v_hi_l1)
                T.copy(workspace_6[cid, slot, 0:BS, 0:DV], v_lo_l1)
                T.copy(workspace_9[cid, slot, 0:BS, 0:DV], dvn_hi_l1)
                T.copy(workspace_10[cid, slot, 0:BS, 0:DV], dvn_lo_l1)
                # ---- C: per bk right relays + 12 gemms + stores ----
                for bk in range(NK):
                    dk0 = bk * block_DK
                    # dq group: do @ h^T
                    T.copy(workspace_3[cid, slot * NK + bk, 0:block_DK, 0:DV], h_hi_l1)
                    T.gemm_v0(do_hi_l1, h_hi_l1, dq_frag, transpose_B=True, init=True)
                    T.copy(workspace_4[cid, slot * NK + bk, 0:block_DK, 0:DV], h_lo_l1)
                    T.gemm_v0(do_hi_l1, h_lo_l1, dq_frag, transpose_B=True, init=False)
                    T.gemm_v0(do_lo_l1, h_hi_l1, dq_frag, transpose_B=True, init=False)
                    # dk group: v @ dh^T
                    T.copy(workspace_7[cid, slot * NK + bk, 0:block_DK, 0:DV], dh_hi_l1)
                    T.gemm_v0(v_hi_l1, dh_hi_l1, dk_frag, transpose_B=True, init=True)
                    T.copy(workspace_8[cid, slot * NK + bk, 0:block_DK, 0:DV], dh_lo_l1)
                    T.gemm_v0(v_hi_l1, dh_lo_l1, dk_frag, transpose_B=True, init=False)
                    T.gemm_v0(v_lo_l1, dh_hi_l1, dk_frag, transpose_B=True, init=False)
                    # dw group: -(dv @ h^T) with the sign folded into dvn
                    T.gemm_v0(dvn_hi_l1, h_hi_l1, dw_frag, transpose_B=True, init=True)
                    T.gemm_v0(dvn_hi_l1, h_lo_l1, dw_frag, transpose_B=True, init=False)
                    T.gemm_v0(dvn_lo_l1, h_hi_l1, dw_frag, transpose_B=True, init=False)
                    # dgkn diagonal group: dgm = h @ dh^T (reuses the L1 operands)
                    T.gemm_v0(h_hi_l1, dh_hi_l1, dgm_frag, transpose_B=True, init=True)
                    T.gemm_v0(h_hi_l1, dh_lo_l1, dgm_frag, transpose_B=True, init=False)
                    T.gemm_v0(h_lo_l1, dh_hi_l1, dgm_frag, transpose_B=True, init=False)
                    # dw direct store (same dtype L0C->GM, sign already folded)
                    T.copy(dw_frag, dw[bb, s0 : s0 + BS, bh, dk0 : dk0 + block_DK])
                    # fused drain relay rows: [0,BS)=dq | [BS,2BS)=dk | dgm tail
                    T.copy(dq_frag, ddr_relay[cid, slot * NK + bk, 0:BS, 0:block_DK])
                    T.copy(dk_frag, ddr_relay[cid, slot * NK + bk, BS : 2 * BS, 0:block_DK])
                    T.copy(
                        dgm_frag,
                        ddr_relay[cid, slot * NK + bk, 2 * BS : 2 * BS + block_DK, 0:block_DK],
                    )
                # 1-late drain handoff on the dedicated uncollided slot 10
                T.set_cross_flag("FIX", 10)

                # ---- drain task j-1 (reads dslot; garbage at j=0, overwritten
                # idempotently by the redirect-redo structure) ----
                tid_d = cid + (j - 1) * CORE_EFF
                dt = T.if_then_else(tid_d < GRID, T.if_then_else(tid_d >= 0, tid_d, cid), cid)
                dslot = (j + NSLOT - 1) % NSLOT
                bh = dt % H
                bt = dt // H
                bs = bt % NT
                bb = bt // NT
                s0 = bs * BS
                # 1-late pairing: drain(j) waits the set raised at iteration
                # j-1 (and drain(0) waits the prologue head-set) on slot 10
                T.wait_cross_flag(10, "V")
                for bk in range(NK):
                    dk0 = bk * block_DK
                    # single fused readback (dq | dk | dgm in one copy)
                    T.copy(
                        ddr_relay[cid, dslot * NK + bk, 0 : 2 * BS + block_DK, 0:block_DK],
                        ddr_ub,
                    )
                    T.copy(G[bb, s0 : s0 + BS, bh, dk0 : dk0 + block_DK], g_ub)
                    T.copy(G[bb, s0 + BS - 1, bh, dk0 : dk0 + block_DK], gn_ub)
                    T.copy(Q[bb, s0 : s0 + BS, bh, dk0 : dk0 + block_DK], q_ub)
                    T.copy(K[bb, s0 : s0 + BS, bh, dk0 : dk0 + block_DK], k_ub)
                    # ln2 pre-fold (one [BS,bDK] mul saved; rounding ~2e-7 << rtol)
                    T.tile.mul(g_ub, g_ub, LN2)
                    T.tile.mul(gn_ub, gn_ub, LN2)
                    # dq = dq_raw * scale * 2^g (in place in the ddr row band)
                    T.tile.exp(gexp_ub, g_ub)
                    T.tile.mul(ddr_ub[0:BS, 0:block_DK], ddr_ub[0:BS, 0:block_DK], gexp_ub)
                    if scale != 1.0:
                        T.tile.mul(ddr_ub[0:BS, 0:block_DK], ddr_ub[0:BS, 0:block_DK], scale)
                    # dgk = diag(dgm) * 2^gn: scratch-free gather extraction
                    T.tile.gather(
                        dgkn_ub,
                        ddr_ub[2 * BS : 2 * BS + block_DK, 0:block_DK],
                        dgk_off_u32,
                        0,
                    )
                    T.tile.exp(gnexp_ub, gn_ub)
                    T.tile.mul(dgkn_ub, dgkn_ub, gnexp_ub)
                    # dk = dk_raw * 2^(gn - g): broadcast gate without scratch
                    if block_DK == 64:
                        # 256B row window: transpose + row_expand_sub + sign fix
                        T.tile.transpose(gexp_ub, g_ub)  # gT (gexp is dead here)
                        T.tile.row_expand_sub_experiment(gexp_ub, gexp_ub, gn_ub, gexp_tmp)
                        T.tile.mul(gexp_ub, gexp_ub, -1.0)  # keep exp arg exact
                        T.tile.exp(gexp_ub, gexp_ub)
                        T.tile.transpose(gnd_ub, gexp_ub)
                    else:
                        # bDK=32 (128B rows): literal row-Sub chain
                        for t in T.unroll(BS):
                            T.tile.sub(gnd_ub[t, 0:block_DK], gn_ub, g_ub[t, 0:block_DK])
                        T.tile.exp(gnd_ub, gnd_ub)
                    T.tile.mul(
                        ddr_ub[BS : 2 * BS, 0:block_DK],
                        ddr_ub[BS : 2 * BS, 0:block_DK],
                        gnd_ub,
                    )
                    # dkk = dk * k (double duty: dg negative term + tree input)
                    T.tile.mul(dkk_ub, ddr_ub[BS : 2 * BS, 0:block_DK], k_ub)
                    T.tile.mul(dg_ub, q_ub, ddr_ub[0:BS, 0:block_DK])
                    T.tile.sub(dg_ub, dg_ub, dkk_ub)
                    # pp = sum_t dkk: literal 6-level halving tree (no scratch)
                    for t0, kk0 in T.Parallel(32, block_DK):
                        dkk_ub[t0, kk0] = dkk_ub[t0, kk0] + dkk_ub[t0 + 32, kk0]
                    for t1, kk1 in T.Parallel(16, block_DK):
                        dkk_ub[t1, kk1] = dkk_ub[t1, kk1] + dkk_ub[t1 + 16, kk1]
                    for t2, kk2 in T.Parallel(8, block_DK):
                        dkk_ub[t2, kk2] = dkk_ub[t2, kk2] + dkk_ub[t2 + 8, kk2]
                    for t3, kk3 in T.Parallel(4, block_DK):
                        dkk_ub[t3, kk3] = dkk_ub[t3, kk3] + dkk_ub[t3 + 4, kk3]
                    for t4, kk4 in T.Parallel(2, block_DK):
                        dkk_ub[t4, kk4] = dkk_ub[t4, kk4] + dkk_ub[t4 + 2, kk4]
                    for kk5 in T.Parallel(block_DK):
                        dkk_ub[0, kk5] = dkk_ub[0, kk5] + dkk_ub[1, kk5]
                    # pp += dgk; last-row single-row injection (no broadcast)
                    T.tile.add(dkk_ub[0, 0:block_DK], dkk_ub[0, 0:block_DK], dgkn_ub[0:block_DK])
                    T.tile.add(
                        dg_ub[BS - 1, 0:block_DK],
                        dg_ub[BS - 1, 0:block_DK],
                        dkk_ub[0, 0:block_DK],
                    )
                    # 3 GM writes (dw was stored by the C side)
                    T.copy(ddr_ub[0:BS, 0:block_DK], dq[bb, s0 : s0 + BS, bh, dk0 : dk0 + block_DK])
                    T.copy(ddr_ub[BS : 2 * BS, 0:block_DK], dk[bb, s0 : s0 + BS, bh, dk0 : dk0 + block_DK])
                    T.copy(dg_ub, dg[bb, s0 : s0 + BS, bh, dk0 : dk0 + block_DK])

    return main


def fla_chunk_inter_bwd(q, k, v, g, h, dv, do, dh, scale=1.0, chunk_size=64, block_DK=None):
    """Host API (upstream chunk_kda_bwd_dqkwg semantics, minus the unused w).

    All fp32, bshd/5D contiguous layouts, directly consumed by the kernel
    (zero view / zero reshape / zero aclnn on the host path). Returns
    (dq, dk, dw, dg), each [B, S, H, DK] fp32.
    """
    B, S, H, DK = q.shape
    DV = v.shape[-1]
    block_DK = _check_shape(B, S, H, DK, DV, chunk_size, block_DK)
    for name, t in (
        ("q", q),
        ("k", k),
        ("v", v),
        ("g", g),
        ("h", h),
        ("dv", dv),
        ("do", do),
        ("dh", dh),
    ):
        if t.dtype is not torch.float32:
            raise ValueError(f"input '{name}' must be float32 (fp32-only operator)")
        if not t.is_contiguous():
            raise ValueError(f"input '{name}' must be contiguous")
    if tuple(h.shape[1:3]) != (S // chunk_size, H) or h.shape[3] != DK:
        raise ValueError(f"h shape {tuple(h.shape)} mismatched with (B, S//64, H, DK, DV)")
    if tuple(dh.shape) != tuple(h.shape):
        raise ValueError(f"dh shape {tuple(dh.shape)} must equal h shape {tuple(h.shape)}")
    kernel = fla_chunk_inter(B, S, H, DK, DV, scale, chunk_size, block_DK)
    return kernel(q, k, v, g, h, dv, do, dh)


__all__ = [
    "fla_chunk_inter",
    "fla_chunk_inter_bwd",
    "_check_shape",
    "PASS_CONFIGS",
]
