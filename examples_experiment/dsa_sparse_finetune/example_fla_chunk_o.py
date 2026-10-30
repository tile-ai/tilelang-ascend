"""fla_chunk_o: KDA chunked forward output O (single kernel) on Ascend 910B3.

Semantics (upstream flash-linear-attention chunk_gla_fwd_kernel_o,
use_exp2=True, fixed length): for each 64-token chunk c and head h,

    qg[t, :] = (q[t, :] * scale) * 2^g[t, :]          # [64, DK], scale = DK**-0.5
    o[t, :]  = qg @ h[c, h]                            # inter-chunk
             + tril_incl_diag(A[t, :]) @ v[t, :]       # intra-chunk (t >= j)

q/v/A/h bf16 (consumed-only, produced upstream), g fp32 (per-token
cumulative log2 decay), o bf16 [B, S, H, DV].

Developer mode, threads=2, fixed-core grid with an in-core task loop;
bshd layout consumed via zero-copy 3D views (no host-side materialization,
no aclnn calls in the wrapper).

Performance structure (bench B=1, S=8192, H=64, DK=DV=128, chunk=64:
3.285 ms -> 1.417 ms, NPU-event mean of 10 reps after 10 warmups):

* FIXED-CORE grid: T.Kernel(CORE_EFF <= 20 cube cores, threads=2) with an
  in-core task loop. The 8192 logical (chunk, b, h) tasks are striped over
  the cores (tid = cid + t * CORE_EFF); one block instance per core
  amortizes the per-block dispatch/TPipe-init overhead and lets the
  MTE2/MTE3/V queues flow across tasks.
* EXPLICIT relay workspaces (workspace_idx): qg_hi/qg_lo/a_mask/v are
  staged through small named GM tensors ([CORE, NSLOT, ...], a few MiB --
  permanently L2-resident) instead of the AUTO_CV_COMBINE auto-injected
  per-block workspaces (~768 MiB per launch). Their V->C handoffs keep
  the auto-inserted cross-core flags (AUTO_CV_SYNC matches buffers named
  "workspace_*"). The v relay moves the strided 64x256B v read onto the
  Vector's MTE2 (the Cube's MTE2 runs at ~16 GB/s on that strided pattern
  vs ~48 GB/s contiguous); the Cube reads the contiguous relay instead.
* SOFTWARE-PIPELINED 3-CHUNK BATCHED task loop, Vector throttled TWO
  BATCHES behind the Cube. One pipeline step consumes three consecutive
  tasks: the per-iteration sync scaffolding amortizes over three tasks,
  successive h loads issue right after the previous gemms (distinct L1
  buffer names -> no forced M-pipe wait), and the L0C accumulator is
  ping-ponged per chunk parity. The output is written DIRECTLY by the
  Cube-side fixpipe: copy_l0c_to_gm converts the fp32 L0C accumulator to
  bf16 and stores into the strided O rows -- no GM relay round-trip for
  the output. The single manual cross-core flag (one set per batch, IDs
  outside the auto pass's range) is a pure THROTTLE: the Vector waits it
  at the top of batch k, gating its batch-k relay writes on the Cube's
  first FIX of batch k-2, keeping every auto relay flag pre-set when the
  Cube arrives. Peeled prologue/fill iterations keep every flag pair's
  site counts balanced and seed the pipeline.
* Branch-free tail handling: out-of-range iterations (the last core's
  remainder plus the final fill iterations) are redirected with Select
  expressions to the core's own first task (always in-range): the
  recomputed relay value is bit-identical and the redundant O writes are
  exact recomputes. ZERO IfThenElse statements remain in the loop -- the
  AUTO_SYNC pass drains all pipes around EVERY IfThenElse, which dominates
  the per-task overhead.
* UNROLLED full-DK inter term: q/g are loaded once per task as [64, DK]
  rows and the hi/lo pair is applied with two K=DK gemms (the upstream
  i_k Pipelined loop never software-pipelined on this toolchain).
* Overlap micro-structure: batched A/Q/G/V loads; the a_mask relay is
  issued right after the mask chain (before the gate chain) so the Cube's
  intra gemm overlaps the Vector's gate math; the inter relay reads
  (hi+lo) are issued back-to-back; the next batch's loads are prefetched
  at the end of each body (latency hides under the next throttle wait).

Ascend toolchain adaptations (all empirically verified on 910B3):

1. One-dimensional T.Kernel with cid decomposed on-core (3-D kernels
   unsupported; threads restricted to 1/2). threads=2 splits each task's
   Vector rows across the 2 AIV sub-blocks of the core pair (910B3 has
   20 cube + 40 vector cores, so threads=2 keeps all vector cores busy).
2. All gemm_v0 inputs are bf16 (fp32 A/B produces garbage on this
   toolchain). qg is split into a bf16 hi/lo pair and accumulated with
   two gemms (~2^-18 relative on o_inter): a single bf16 cast (~2^-9
   relative) measurably trips the bf16 mixed tolerance on cancellation
   elements (|o| << |o_inter|).
3. Every bf16<->fp32 conversion happens UB-internally via a two-step copy
   chain; GM<->UB cast-on-copy codegen is broken (mismatched template).
4. The output is written directly by the Cube-side fixpipe as BF16 (the
   fp32 L0C -> bf16 GM conversion rides the FIX instruction, verified
   exact on this path).
5. The single manual cross-core flag pair classifies onto the right core
   via its pipe argument: T.set_cross_flag("FIX", ...) -> Cube,
   T.wait_cross_flag(..., "V") -> Vector (ResourceForPipe accepts exactly
   "V"/"M"/"MTE1"/"FIX"; anything else needs T.Scope). Manual flag IDs
   must start well above the auto pass's allocation range: the auto pass
   does NOT track user-issued IDs, and an ID collision silently corrupts
   the handshake (observed and fixed; auto IDs observed in the 0..40
   range on this kernel, user IDs start at 48).
6. The tril mask is applied with an element-wise Select (T.tile.select,
   VSEL_TENSOR_SCALAR_MODE) against a bit-packed uint8 constant
   (Msk[BS*BS//8]: bit (t*BS+j), LSB-first, is 1 iff t >= j, inclusive
   diagonal). A multiplicative 0/1 mask CANNOT clear non-finite
   upper-triangle values (Inf*0 = NaN pollutes the intra-chunk gemm),
   while Select zeroes them exactly -- matching the upstream tl.where
   semantics. The VSEL selMask must be a packed bitmap, not per-element
   0/1 values (compare/select share the bit-packed convention). The
   select operates on the HALF-HEIGHT region [0:BS/2, 0:BS]: the
   vid-reduction pass splits recognized binary ops (mul et al.) to the
   per-sub-block half automatically, but NOT tile.select -- a full-tile
   select would run 4096 elements against each vid's 256B mask half and
   overflow the destination buffer by 8KB. The half-region makes the
   element count match each vid's own 32 rows and its 256B mask bits
   exactly. The mask load is hoisted out of the task loop
   (loop-invariant).
7. Element math uses buffer-level T.tile.* instructions only (element
   T.Parallel loops are a measured -26%..-71%/kernel antipattern).
8. Q/G/A/O are consumed through zero-copy 3D views [B, S, H*D] of the
   bshd tensors: the vid-reduction pass adds the sub-block row offset to
   GM dimension (gm_dims - ub_dims); with 4D bshd slices that lands on
   the H index (reads a different head -- silent data corruption), while
   3D row slices put the S rows exactly there. V/Hstate are consumed
   Cube-side (GM->L1, not vid-split) and stay 4D/5D.
9. The host wrapper caches the compiled-kernel launch path (_FastLaunch):
   the generic JIT adapter re-validates every argument shape and rebuilds
   a TVM Arith Analyzer on EVERY call (~0.8 ms host time -- exposed in
   device-side event timing whenever the queue runs dry). The cached
   path reuses the relay workspaces, re-checks every argument (dtype /
   shape / contiguity / device), bitwise-verifies the first launch
   against the generic path, and falls back to it on any mismatch or
   exception. Only stable public adapter attributes are touched (same
   pattern as examples/torch_tl_ascend/compile_tl_op/flash_attention.py).
"""

import ctypes

import tilelang
import torch
from tilelang import language as T

# 2^g is computed as exp(g * ln2): exp2 has no Ascend backend mapping.
LN2 = 0.6931471805599453

# Ascend 910B3: 20 AI Cube cores (+ 40 AI Vector cores). The fixed-core
# grid launches one block per cube core; threads=2 pairs each with its 2
# vector sub-blocks.
CORE_NUM = 20

# Relay slot ring per core (t % NSLOT). The Vector is throttled TWO
# BATCHES (4 tasks) behind the Cube (its batch-k relay writes are gated by
# the Cube's FIX-A of batch k-2), so every relay flag is long set when the
# Cube arrives -- the measured ~470ns x 4 relay-flag waits per task (777us
# total, the hi/lo pair alone ~490us) collapse to protocol overhead. The
# ring must then hold THREE in-flight batches (V at k, C at k-1, k-2's
# slots being retired): NSLOT=8 gives 4 batches of margin.
# (An 8-bare-set probe at kernel start -- giving the Vector credit to run
# ~10 batches ahead, within the documented 15-set flag counter capacity --
# measured neutral: the Vector is throttle-bound, not production-bound.)
# Ring safety for throttle lag L requires NSLOT > 2L+1 (V's earliest write
# at C's FIX-A(c) touches slot 2(c+L)%NSLOT, which must stay clear of C's
# pending reads [2c+1, 2(c+L)-1]); L=2, NSLOT=9 satisfies it with margin.
# 3-chunk batching. Ring: batch k uses slots {3k, 3k+1, 3k+2} mod
# NSLOT=9 (3 batches of ring; conflict condition 3d == m (mod 9) for
# lead d and slot offset m in [0,2] first binds at d=3, while the
# throttle (lag 2, production-bound V) holds the lead at <= 2).
NSLOT = 9

PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,  # automatic C/V split
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,  # cross-core flags on the relays
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,  # intra-core pipe sync
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,  # UB/L1 lifetime planning
}


# Manual cross-core THROTTLE flag ID. The AUTO_CV_SYNC pass assigns
# cross-core flag IDs 0..N-1 to every (workspace, site) relay pair in
# statement order (ascend_combinecv.cc, global_sync_flag_id) with NO
# awareness of user-specified IDs -- this kernel's 4 workspaces x 8
# chunk-sites occupy IDs 0-31. The historical ID 12 COLLIDED with the auto
# flag of the loop's chunk-A ws2 (lo relay) site: per the CANN docs each
# cross-core flagId owns ONE hardware counter shared across directions, so
# the manual set/wait pair and the auto relay pair of the same ID scrambled
# each other's accounting -- the measured effect was the Vector delivering
# relays just-in-time (Cube relay-flag waits ~780us/core in msprof r2c)
# instead of the designed 2-batch lead. ID 40 sits above the current auto
# range with margin; ANY restructuring that adds relay sites (e.g. 3-chunk
# batching, +16 sites) MUST re-check this bound.
THROTTLE_FLAG_ID = 48


def _pick_block_dk(dim: int) -> int:
    """Domain gate for DK (historically the exact-2 Pipelined block split).

    The kernel computes the inter term with full-K gemms (no i_k
    loop), so any 16-multiple K >= 32 would be fractal-legal; the
    {64, 96, 128} domain is kept unchanged from the delivered contract
    (out-of-domain shapes are rejected by the wrapper's asserts).
    """
    blk = dim // 2
    if dim % 2 == 0 and blk in (32, 48, 64):
        return blk
    raise ValueError(f"DK={dim} unsupported: the delivered contract restricts DK to one of {{64, 96, 128}} (documented limitation)")


def _pick_block_dv(dim: int) -> int:
    """block_DV = DV single block (IO-unique: every tensor read/written once).

    Constraints: the intra/inter gemm N = block_DV must satisfy the bf16
    gemm_v0 N >= 16 fractal; a single block keeps the o accumulator
    [64, DV] fp32 within the 128KB L0C budget -> DV <= 128. Larger DV
    would need a nested bv loop (no precedent, out of scope).
    """
    if dim % 16 == 0 and 16 <= dim <= 128:
        return dim
    raise ValueError(
        f"DV={dim} unsupported: DV must be a 16-multiple in [16, 128]"
        " (bf16 gemm N-fractal + single-block L0C budget; documented"
        " limitation)"
    )


# out_idx/workspace_idx use ABSOLUTE indices over the declared signature
# (Q,V,G,A,Hstate,Msk,workspace_1..3,O): no UB->L1 / L0C->UB copies remain
# (all V<->C exchanges go through the explicit named workspaces), so
# AUTO_CV_COMBINE injects no extra GM params and the indices stay stable.
# The output is written DIRECTLY by the Cube-side fixpipe
# (copy_l0c_to_gm with realDstN = H*DV row stride, derived by
# compute_strideN in src/op/ascend.cc) -- no o_relay GM round-trip.
@tilelang.jit(out_idx=[10], workspace_idx=[6, 7, 8, 9], pass_configs=PASS_CONFIGS)
def fla_chunk_o_kernel(
    B,
    S,
    H,
    DK,
    DV,
    chunk_size=64,
    scale=None,
    block_DK=None,
    block_DV=None,
):
    """Single kernel: o = qg @ h + tril_incl_diag(A) @ v per (chunk, b, h)."""
    if scale is None:
        scale = DK**-0.5
    BS = chunk_size
    NT = S // BS
    bDV = block_DV if block_DV is not None else _pick_block_dv(DV)
    GRID = NT * B * H  # logical (chunk, b, h) tasks
    # Fixed-core grid: CORE_EFF blocks (<= cube-core count, <= task count),
    # each running a TPT+1 iteration T.serial task loop with a STRIDED
    # task map tid = cid + t * CORE_EFF. The stride keeps the branch-free
    # Select fallbacks below always in-bounds ("own first task" == cid <
    # CORE_EFF <= GRID), which removes EVERY IfThenElse from the loop --
    # the AUTO_SYNC pass drains all pipes (PipeBarrier<PIPE_ALL>) around
    # each IfThenElse, measured as the dominant per-task overhead.
    CORE_EFF = min(CORE_NUM, GRID)
    TPT = (GRID + CORE_EFF - 1) // CORE_EFF  # produce slots t = 0 .. TPT+1
    # pipeline step k consumes tasks in batches (see 3-chunk below)
    # of the strided map (tid = cid + t*CORE_EFF). NB batch steps cover the
    # TPT produce slots from t=2 up (the first two are the peeled fills).
    # Select-free fast path: for k <= NB_MAIN every core's produce AND
    # prefetch targets are in-range ((2k+4)*CORE_EFF <= GRID), so the body
    # needs no Select guards; the remaining tail batches keep the
    # branch-free Select redirects to the core's own first task.
    # 3-chunk batching: pipeline step k consumes tasks
    # (3k, 3k+1, 3k+2) of the strided map (tid = cid + t*CORE_EFF). The
    # produce slots t = 0 .. TPT+1 are covered by batches k = 0 .. NB
    # (NB = ceil((TPT+2)/3) - 1). Batches 0 and 1 are PEELED full batches
    # (they seed the pipeline and keep the throttle-flag accounting at the
    # designed 2-batch lag: the loop's k-th wait pairs the Cube's set from
    # batch k-2 exactly); the loops cover k = 2 .. NB.
    NB = (TPT + 2 + 2) // 3 - 1  # last batch index (tasks 3k..3k+2)
    # Select-free bound: batch k is guard-free when its own tasks
    # 3k..3k+2 AND its prefetch targets 3k+3..3k+5 are all in-range,
    # i.e. (3k+6)*CORE_EFF <= GRID.
    K_MAX = (GRID // CORE_EFF - 6) // 3
    N_MAIN = max(0, min(K_MAX, NB) - 1)  # Select-free batches k = 2 .. K_MAX
    N_TAIL = NB - max(1, min(K_MAX, NB))  # Select-guarded remainder

    @T.prim_func
    def main(
        # Q/G/A/O as zero-copy 3D views [B, S, H*D] of the bshd tensors
        # (vid-reduction splits the S rows only when they sit at GM dim
        # gm_dims - ub_dims, see module docstring item 7).
        Q: T.Tensor((B, S, H * DK), "bfloat16"),  # type: ignore
        V: T.Tensor((B, S, H * DV), "bfloat16"),  # type: ignore
        G: T.Tensor((B, S, H * DK), "float32"),  # type: ignore
        A: T.Tensor((B, S, H * BS), "bfloat16"),  # type: ignore
        Hstate: T.Tensor((B, NT, H, DK, DV), "bfloat16"),  # type: ignore
        Msk: T.Tensor((BS * BS // 8,), "uint8"),  # type: ignore
        # Explicit V->C relay workspaces (L2-resident, slot ring).
        workspace_1: T.Tensor((CORE_EFF, NSLOT, BS, DK), "bfloat16"),  # type: ignore
        workspace_2: T.Tensor((CORE_EFF, NSLOT, BS, DK), "bfloat16"),  # type: ignore
        workspace_3: T.Tensor((CORE_EFF, NSLOT, BS, BS), "bfloat16"),  # type: ignore
        workspace_4: T.Tensor((CORE_EFF, NSLOT, BS, bDV), "bfloat16"),  # type: ignore
        O: T.Tensor((B, S, H * DV), "bfloat16"),  # type: ignore
    ):
        with T.Kernel(CORE_EFF, threads=2, is_npu=True) as (cid):
            # ---- buffers (placement planned by the compiler; the *_l1
            # names mark gemm operands that must land in L1, the fp32
            # math domains land in UB) ----
            # L1 gemm operands are PING-PONGED BY NAME (chunk A/_a,
            # chunk B/_b). The AUTO_SYNC WAR analysis (HasDataDependency,
            # ascend_sync_insert.cc) is per buffer NAME: with shared names
            # every inter-chunk load->gemm pair gets a forced M_MTE2
            # serialization (measured: the MTE2 queue and the M pipe take
            # strict turns, zero overlap); with distinct names chunk B's
            # h/v loads drain under chunk A's gemm/FIX tail and the next
            # iteration's A-loads overlap B's gemms. L1 total 176KB/512KB.
            A_l1_a = T.alloc_shared([BS, BS], "bfloat16")  # intra LHS, chunk 0
            A_l1_b = T.alloc_shared([BS, BS], "bfloat16")  # intra LHS, chunk 1
            A_l1_c = T.alloc_shared([BS, BS], "bfloat16")  # intra LHS, chunk 2
            V_l1_a = T.alloc_shared([BS, bDV], "bfloat16")  # intra RHS, chunk 0
            V_l1_b = T.alloc_shared([BS, bDV], "bfloat16")  # intra RHS, chunk 1
            V_l1_c = T.alloc_shared([BS, bDV], "bfloat16")  # intra RHS, chunk 2
            H_l1_a = T.alloc_shared([DK, bDV], "bfloat16")  # inter RHS, chunk 0
            H_l1_b = T.alloc_shared([DK, bDV], "bfloat16")  # inter RHS, chunk 1
            H_l1_c = T.alloc_shared([DK, bDV], "bfloat16")  # inter RHS, chunk 2
            qg_hi_l1_a = T.alloc_shared([BS, DK], "bfloat16")  # inter LHS hi, 0
            qg_hi_l1_b = T.alloc_shared([BS, DK], "bfloat16")  # inter LHS hi, 1
            qg_hi_l1_c = T.alloc_shared([BS, DK], "bfloat16")  # inter LHS hi, 2
            qg_lo_l1_a = T.alloc_shared([BS, DK], "bfloat16")  # inter LHS lo, 0
            qg_lo_l1_b = T.alloc_shared([BS, DK], "bfloat16")  # inter LHS lo, 1
            qg_lo_l1_c = T.alloc_shared([BS, DK], "bfloat16")  # inter LHS lo, 2
            a_half = T.alloc_shared([BS, BS], "bfloat16")  # A load staging (chunk A)
            a_f32 = T.alloc_shared([BS, BS], "float32")  # A fp32 domain
            msk_bits = T.alloc_shared([BS * BS // 8], "uint8")  # tril bits, packed
            a_mask = T.alloc_shared([BS, BS], "bfloat16")  # masked A (bf16, exact)
            q_half = T.alloc_shared([BS, DK], "bfloat16")  # q load staging (chunk A)
            q_ub = T.alloc_shared([BS, DK], "float32")  # q fp32 (reused: qg, lo)
            g_ub = T.alloc_shared([BS, DK], "float32")  # g (A; reused: g*ln2, 2^g)
            v_half = T.alloc_shared([BS, bDV], "bfloat16")  # v staging (chunk A, prefetched)
            v_half2 = T.alloc_shared([BS, bDV], "bfloat16")  # v staging (chunk 1)
            a_half2 = T.alloc_shared([BS, BS], "bfloat16")  # A staging (chunk 1)
            q_half2 = T.alloc_shared([BS, DK], "bfloat16")  # q staging (chunk 1)
            g_ub2 = T.alloc_shared([BS, DK], "float32")  # g (chunk 1; reused)
            v_half3 = T.alloc_shared([BS, bDV], "bfloat16")  # v staging (chunk 2)
            a_half3 = T.alloc_shared([BS, BS], "bfloat16")  # A staging (chunk 2)
            q_half3 = T.alloc_shared([BS, DK], "bfloat16")  # q staging (chunk 2)
            g_ub3 = T.alloc_shared([BS, DK], "float32")  # g (chunk 2; reused)
            qg_hi_ub = T.alloc_shared([BS, DK], "bfloat16")  # qg hi (bf16)
            qg_hi_f32 = T.alloc_shared([BS, DK], "float32")  # hi, exact fp32
            qg_lo_ub = T.alloc_shared([BS, DK], "bfloat16")  # qg lo (bf16)
            # The L0C accumulator is PING-PONGED per chunk parity
            # (2 x 32KB <= 128KB L0C): chunk B's gemms no longer WAR chunk
            # A's FIX read of the accumulator, so the FIX pipe overlaps the
            # M pipe's next gemm group instead of serializing behind it
            # (the per-boundary FIX_M/M_FIX handshake pairs disappear).
            o_frag_a = T.alloc_fragment([BS, bDV], "float32")  # L0C, chunk A
            o_frag_b = T.alloc_fragment([BS, bDV], "float32")  # L0C, chunk B

            # ---- loop-invariant hoist: the tril constant loads once per core ----
            T.copy(Msk[0 : BS * BS // 8], msk_bits)

            # ---- prologue: stage batch 0's three chunks. Batch 0 = strided tasks
            # t = 0,1,2 (tid = cid + t*CORE_EFF); chunk 0 (tid = cid) is always
            # in-range, chunks 1/2 are Select-redirected to the core's own first
            # task when out of range (bit-identical relay recomputes).
            p0t1 = T.if_then_else(cid + CORE_EFF < GRID, cid + CORE_EFF, cid)
            p0t2 = T.if_then_else(cid + 2 * CORE_EFF < GRID, cid + 2 * CORE_EFF, cid)
            p0c0_bs_ = cid % NT
            p0c0_bbh = cid // NT
            p0c0_bb = p0c0_bbh // H
            p0c0_bh = p0c0_bbh % H
            p0c0_s0 = p0c0_bs_ * BS
            T.copy(A[p0c0_bb, p0c0_s0 : p0c0_s0 + BS, p0c0_bh * BS : (p0c0_bh + 1) * BS], a_half)
            T.copy(Q[p0c0_bb, p0c0_s0 : p0c0_s0 + BS, p0c0_bh * DK : (p0c0_bh + 1) * DK], q_half)
            T.copy(G[p0c0_bb, p0c0_s0 : p0c0_s0 + BS, p0c0_bh * DK : (p0c0_bh + 1) * DK], g_ub)
            T.copy(V[p0c0_bb, p0c0_s0 : p0c0_s0 + BS, p0c0_bh * bDV : (p0c0_bh + 1) * bDV], v_half)
            p0c1_bs_ = p0t1 % NT
            p0c1_bbh = p0t1 // NT
            p0c1_bb = p0c1_bbh // H
            p0c1_bh = p0c1_bbh % H
            p0c1_s0 = p0c1_bs_ * BS
            T.copy(A[p0c1_bb, p0c1_s0 : p0c1_s0 + BS, p0c1_bh * BS : (p0c1_bh + 1) * BS], a_half2)
            T.copy(Q[p0c1_bb, p0c1_s0 : p0c1_s0 + BS, p0c1_bh * DK : (p0c1_bh + 1) * DK], q_half2)
            T.copy(G[p0c1_bb, p0c1_s0 : p0c1_s0 + BS, p0c1_bh * DK : (p0c1_bh + 1) * DK], g_ub2)
            T.copy(V[p0c1_bb, p0c1_s0 : p0c1_s0 + BS, p0c1_bh * bDV : (p0c1_bh + 1) * bDV], v_half2)
            p0c2_bs_ = p0t2 % NT
            p0c2_bbh = p0t2 // NT
            p0c2_bb = p0c2_bbh // H
            p0c2_bh = p0c2_bbh % H
            p0c2_s0 = p0c2_bs_ * BS
            T.copy(A[p0c2_bb, p0c2_s0 : p0c2_s0 + BS, p0c2_bh * BS : (p0c2_bh + 1) * BS], a_half3)
            T.copy(Q[p0c2_bb, p0c2_s0 : p0c2_s0 + BS, p0c2_bh * DK : (p0c2_bh + 1) * DK], q_half3)
            T.copy(G[p0c2_bb, p0c2_s0 : p0c2_s0 + BS, p0c2_bh * DK : (p0c2_bh + 1) * DK], g_ub3)
            T.copy(V[p0c2_bb, p0c2_s0 : p0c2_s0 + BS, p0c2_bh * bDV : (p0c2_bh + 1) * bDV], v_half3)
            # ---- peeled batch 0 (tasks 0,1,2) (Select-guarded) ----
            ta0_ = 3 * (0) + 0
            tida0_ = cid + ta0_ * CORE_EFF
            pt0_ = T.if_then_else(tida0_ < GRID, tida0_, cid)
            c0_bs_ = pt0_ % NT
            c0_bbh = pt0_ // NT
            c0_bb = c0_bbh // H
            c0_bh = c0_bbh % H
            c0_s0 = c0_bs_ * BS
            ta1_ = 3 * (0) + 1
            tida1_ = cid + ta1_ * CORE_EFF
            pt1_ = T.if_then_else(tida1_ < GRID, tida1_, cid)
            c1_bs_ = pt1_ % NT
            c1_bbh = pt1_ // NT
            c1_bb = c1_bbh // H
            c1_bh = c1_bbh % H
            c1_s0 = c1_bs_ * BS
            ta2_ = 3 * (0) + 2
            tida2_ = cid + ta2_ * CORE_EFF
            pt2_ = T.if_then_else(tida2_ < GRID, tida2_, cid)
            c2_bs_ = pt2_ % NT
            c2_bbh = pt2_ // NT
            c2_bb = c2_bbh // H
            c2_bh = c2_bbh % H
            c2_s0 = c2_bs_ * BS
            nta0_ = ta0_ + 3
            ntida0_ = cid + nta0_ * CORE_EFF
            np0_ = T.if_then_else(ntida0_ < GRID, ntida0_, cid)
            np0_bs_ = np0_ % NT
            np0_bbh = np0_ // NT
            np0_bb = np0_bbh // H
            np0_bh = np0_bbh % H
            np0_s0 = np0_bs_ * BS
            nta1_ = ta1_ + 3
            ntida1_ = cid + nta1_ * CORE_EFF
            np1_ = T.if_then_else(ntida1_ < GRID, ntida1_, cid)
            np1_bs_ = np1_ % NT
            np1_bbh = np1_ // NT
            np1_bb = np1_bbh // H
            np1_bh = np1_bbh % H
            np1_s0 = np1_bs_ * BS
            nta2_ = ta2_ + 3
            ntida2_ = cid + nta2_ * CORE_EFF
            np2_ = T.if_then_else(ntida2_ < GRID, ntida2_, cid)
            np2_bs_ = np2_ % NT
            np2_bbh = np2_ // NT
            np2_bb = np2_bbh // H
            np2_bh = np2_bbh % H
            np2_s0 = np2_bs_ * BS
            slot0_ = ta0_ % NSLOT
            slot1_ = ta1_ % NSLOT
            slot2_ = ta2_ % NSLOT
            # -- Vector: v relays x3 first (staging-only) --
            T.copy(v_half, workspace_4[cid, slot0_, 0:BS, 0:bDV])
            T.copy(v_half2, workspace_4[cid, slot1_, 0:BS, 0:bDV])
            T.copy(v_half3, workspace_4[cid, slot2_, 0:BS, 0:bDV])
            # -- Vector: early v loads for the next batch (drain hides under
            # the mask/gate chains; removes the exposed end-of-body MTE2
            # drain from the serial chain) --
            T.copy(V[np0_bb, np0_s0 : np0_s0 + BS, np0_bh * bDV : (np0_bh + 1) * bDV], v_half)
            T.copy(V[np1_bb, np1_s0 : np1_s0 + BS, np1_bh * bDV : (np1_bh + 1) * bDV], v_half2)
            T.copy(V[np2_bb, np2_s0 : np2_s0 + BS, np2_bh * bDV : (np2_bh + 1) * bDV], v_half3)
            # -- Cube: hoisted loads: 3x Hstate (flag-free) --
            T.copy(Hstate[c0_bb, c0_bs_, c0_bh, 0:DK, 0:bDV], H_l1_a)
            T.copy(Hstate[c1_bb, c1_bs_, c1_bh, 0:DK, 0:bDV], H_l1_b)
            T.copy(Hstate[c2_bb, c2_bs_, c2_bh, 0:DK, 0:bDV], H_l1_c)
            # -- Cube: hoisted relay loads: workspace_4 c0/c1/c2 --
            T.copy(workspace_4[cid, slot0_, 0:BS, 0:bDV], V_l1_a)
            T.copy(workspace_4[cid, slot1_, 0:BS, 0:bDV], V_l1_b)
            T.copy(workspace_4[cid, slot2_, 0:BS, 0:bDV], V_l1_c)
            # -- Cube: hoisted relay loads: workspace_3 c0/c1/c2 --
            T.copy(workspace_3[cid, slot0_, 0:BS, 0:BS], A_l1_a)
            T.copy(workspace_3[cid, slot1_, 0:BS, 0:BS], A_l1_b)
            T.copy(workspace_3[cid, slot2_, 0:BS, 0:BS], A_l1_c)
            # -- Cube: hoisted relay loads: workspace_1 c0/c1/c2 --
            T.copy(workspace_1[cid, slot0_, 0:BS, 0:DK], qg_hi_l1_a)
            T.copy(workspace_1[cid, slot1_, 0:BS, 0:DK], qg_hi_l1_b)
            T.copy(workspace_1[cid, slot2_, 0:BS, 0:DK], qg_hi_l1_c)
            # -- Cube: hoisted relay loads: workspace_2 c0/c1/c2 --
            T.copy(workspace_2[cid, slot0_, 0:BS, 0:DK], qg_lo_l1_a)
            T.copy(workspace_2[cid, slot1_, 0:BS, 0:DK], qg_lo_l1_b)
            T.copy(workspace_2[cid, slot2_, 0:BS, 0:DK], qg_lo_l1_c)
            # -- Vector: chunk 0: mask + a relay --
            T.copy(a_half, a_f32)
            T.tile.select(
                a_f32[0 : BS // 2, 0:BS],
                msk_bits,
                a_f32[0 : BS // 2, 0:BS],
                0.0,
                "VSEL_TENSOR_SCALAR_MODE",
            )
            T.copy(a_f32, a_mask)
            T.copy(a_mask, workspace_3[cid, slot0_, 0:BS, 0:BS])
            # -- Vector: chunk 0: gate chain + hi/lo relays --
            T.copy(q_half, q_ub)
            T.tile.mul(g_ub, g_ub, LN2)
            T.tile.exp(g_ub, g_ub)
            T.tile.mul(q_ub, q_ub, scale)
            T.tile.mul(q_ub, q_ub, g_ub)
            T.copy(q_ub, qg_hi_ub)
            T.copy(qg_hi_ub, qg_hi_f32)
            T.tile.sub(q_ub, q_ub, qg_hi_f32)
            T.copy(q_ub, qg_lo_ub)
            T.copy(qg_hi_ub, workspace_1[cid, slot0_, 0:BS, 0:DK])
            T.copy(qg_lo_ub, workspace_2[cid, slot0_, 0:BS, 0:DK])
            # -- Vector: prefetch next batch's chunk 0 a/q/g --
            T.copy(A[np0_bb, np0_s0 : np0_s0 + BS, np0_bh * BS : (np0_bh + 1) * BS], a_half)
            T.copy(Q[np0_bb, np0_s0 : np0_s0 + BS, np0_bh * DK : (np0_bh + 1) * DK], q_half)
            T.copy(G[np0_bb, np0_s0 : np0_s0 + BS, np0_bh * DK : (np0_bh + 1) * DK], g_ub)
            # -- Cube: gemm chain chunk 0 (L1 set _a) --
            T.gemm_v0(A_l1_a, V_l1_a, o_frag_a, init=True)
            T.gemm_v0(qg_hi_l1_a, H_l1_a, o_frag_a, init=False)
            T.gemm_v0(qg_lo_l1_a, H_l1_a, o_frag_a, init=False)
            # -- Cube: FIX chunk 0 (direct strided O store) --
            T.copy(o_frag_a, O[c0_bb, c0_s0 : c0_s0 + BS, c0_bh * bDV : (c0_bh + 1) * bDV])
            T.set_cross_flag("FIX", THROTTLE_FLAG_ID)
            # -- Vector: chunk 1: mask + a relay --
            T.copy(a_half2, a_f32)
            T.tile.select(
                a_f32[0 : BS // 2, 0:BS],
                msk_bits,
                a_f32[0 : BS // 2, 0:BS],
                0.0,
                "VSEL_TENSOR_SCALAR_MODE",
            )
            T.copy(a_f32, a_mask)
            T.copy(a_mask, workspace_3[cid, slot1_, 0:BS, 0:BS])
            # -- Vector: chunk 1: gate chain + hi/lo relays --
            T.copy(q_half2, q_ub)
            T.tile.mul(g_ub2, g_ub2, LN2)
            T.tile.exp(g_ub2, g_ub2)
            T.tile.mul(q_ub, q_ub, scale)
            T.tile.mul(q_ub, q_ub, g_ub2)
            T.copy(q_ub, qg_hi_ub)
            T.copy(qg_hi_ub, qg_hi_f32)
            T.tile.sub(q_ub, q_ub, qg_hi_f32)
            T.copy(q_ub, qg_lo_ub)
            T.copy(qg_hi_ub, workspace_1[cid, slot1_, 0:BS, 0:DK])
            T.copy(qg_lo_ub, workspace_2[cid, slot1_, 0:BS, 0:DK])
            # -- Vector: prefetch next batch's chunk 1 a/q/g --
            T.copy(A[np1_bb, np1_s0 : np1_s0 + BS, np1_bh * BS : (np1_bh + 1) * BS], a_half2)
            T.copy(Q[np1_bb, np1_s0 : np1_s0 + BS, np1_bh * DK : (np1_bh + 1) * DK], q_half2)
            T.copy(G[np1_bb, np1_s0 : np1_s0 + BS, np1_bh * DK : (np1_bh + 1) * DK], g_ub2)
            # -- Cube: gemm chain chunk 1 (L1 set _b) --
            T.gemm_v0(A_l1_b, V_l1_b, o_frag_b, init=True)
            T.gemm_v0(qg_hi_l1_b, H_l1_b, o_frag_b, init=False)
            T.gemm_v0(qg_lo_l1_b, H_l1_b, o_frag_b, init=False)
            # -- Cube: FIX chunk 1 (direct strided O store) --
            T.copy(o_frag_b, O[c1_bb, c1_s0 : c1_s0 + BS, c1_bh * bDV : (c1_bh + 1) * bDV])
            # -- Vector: chunk 2: mask + a relay --
            T.copy(a_half3, a_f32)
            T.tile.select(
                a_f32[0 : BS // 2, 0:BS],
                msk_bits,
                a_f32[0 : BS // 2, 0:BS],
                0.0,
                "VSEL_TENSOR_SCALAR_MODE",
            )
            T.copy(a_f32, a_mask)
            T.copy(a_mask, workspace_3[cid, slot2_, 0:BS, 0:BS])
            # -- Vector: chunk 2: gate chain + hi/lo relays --
            T.copy(q_half3, q_ub)
            T.tile.mul(g_ub3, g_ub3, LN2)
            T.tile.exp(g_ub3, g_ub3)
            T.tile.mul(q_ub, q_ub, scale)
            T.tile.mul(q_ub, q_ub, g_ub3)
            T.copy(q_ub, qg_hi_ub)
            T.copy(qg_hi_ub, qg_hi_f32)
            T.tile.sub(q_ub, q_ub, qg_hi_f32)
            T.copy(q_ub, qg_lo_ub)
            T.copy(qg_hi_ub, workspace_1[cid, slot2_, 0:BS, 0:DK])
            T.copy(qg_lo_ub, workspace_2[cid, slot2_, 0:BS, 0:DK])
            # -- Vector: prefetch next batch's chunk 2 a/q/g --
            T.copy(A[np2_bb, np2_s0 : np2_s0 + BS, np2_bh * BS : (np2_bh + 1) * BS], a_half3)
            T.copy(Q[np2_bb, np2_s0 : np2_s0 + BS, np2_bh * DK : (np2_bh + 1) * DK], q_half3)
            T.copy(G[np2_bb, np2_s0 : np2_s0 + BS, np2_bh * DK : (np2_bh + 1) * DK], g_ub3)
            # -- Cube: gemm chain chunk 2 (L1 set _c) --
            T.gemm_v0(A_l1_c, V_l1_c, o_frag_a, init=True)
            T.gemm_v0(qg_hi_l1_c, H_l1_c, o_frag_a, init=False)
            T.gemm_v0(qg_lo_l1_c, H_l1_c, o_frag_a, init=False)
            # -- Cube: FIX chunk 2 (direct strided O store) --
            T.copy(o_frag_a, O[c2_bb, c2_s0 : c2_s0 + BS, c2_bh * bDV : (c2_bh + 1) * bDV])

            # ---- peeled batch 1 (tasks 3,4,5) (Select-guarded) ----
            ta0_ = 3 * (1) + 0
            tida0_ = cid + ta0_ * CORE_EFF
            pt0_ = T.if_then_else(tida0_ < GRID, tida0_, cid)
            c0_bs_ = pt0_ % NT
            c0_bbh = pt0_ // NT
            c0_bb = c0_bbh // H
            c0_bh = c0_bbh % H
            c0_s0 = c0_bs_ * BS
            ta1_ = 3 * (1) + 1
            tida1_ = cid + ta1_ * CORE_EFF
            pt1_ = T.if_then_else(tida1_ < GRID, tida1_, cid)
            c1_bs_ = pt1_ % NT
            c1_bbh = pt1_ // NT
            c1_bb = c1_bbh // H
            c1_bh = c1_bbh % H
            c1_s0 = c1_bs_ * BS
            ta2_ = 3 * (1) + 2
            tida2_ = cid + ta2_ * CORE_EFF
            pt2_ = T.if_then_else(tida2_ < GRID, tida2_, cid)
            c2_bs_ = pt2_ % NT
            c2_bbh = pt2_ // NT
            c2_bb = c2_bbh // H
            c2_bh = c2_bbh % H
            c2_s0 = c2_bs_ * BS
            nta0_ = ta0_ + 3
            ntida0_ = cid + nta0_ * CORE_EFF
            np0_ = T.if_then_else(ntida0_ < GRID, ntida0_, cid)
            np0_bs_ = np0_ % NT
            np0_bbh = np0_ // NT
            np0_bb = np0_bbh // H
            np0_bh = np0_bbh % H
            np0_s0 = np0_bs_ * BS
            nta1_ = ta1_ + 3
            ntida1_ = cid + nta1_ * CORE_EFF
            np1_ = T.if_then_else(ntida1_ < GRID, ntida1_, cid)
            np1_bs_ = np1_ % NT
            np1_bbh = np1_ // NT
            np1_bb = np1_bbh // H
            np1_bh = np1_bbh % H
            np1_s0 = np1_bs_ * BS
            nta2_ = ta2_ + 3
            ntida2_ = cid + nta2_ * CORE_EFF
            np2_ = T.if_then_else(ntida2_ < GRID, ntida2_, cid)
            np2_bs_ = np2_ % NT
            np2_bbh = np2_ // NT
            np2_bb = np2_bbh // H
            np2_bh = np2_bbh % H
            np2_s0 = np2_bs_ * BS
            slot0_ = ta0_ % NSLOT
            slot1_ = ta1_ % NSLOT
            slot2_ = ta2_ % NSLOT
            # -- Vector: v relays x3 first (staging-only) --
            T.copy(v_half, workspace_4[cid, slot0_, 0:BS, 0:bDV])
            T.copy(v_half2, workspace_4[cid, slot1_, 0:BS, 0:bDV])
            T.copy(v_half3, workspace_4[cid, slot2_, 0:BS, 0:bDV])
            # -- Vector: early v loads for the next batch (drain hides under
            # the mask/gate chains; removes the exposed end-of-body MTE2
            # drain from the serial chain) --
            T.copy(V[np0_bb, np0_s0 : np0_s0 + BS, np0_bh * bDV : (np0_bh + 1) * bDV], v_half)
            T.copy(V[np1_bb, np1_s0 : np1_s0 + BS, np1_bh * bDV : (np1_bh + 1) * bDV], v_half2)
            T.copy(V[np2_bb, np2_s0 : np2_s0 + BS, np2_bh * bDV : (np2_bh + 1) * bDV], v_half3)
            # -- Cube: hoisted loads: 3x Hstate (flag-free) --
            T.copy(Hstate[c0_bb, c0_bs_, c0_bh, 0:DK, 0:bDV], H_l1_a)
            T.copy(Hstate[c1_bb, c1_bs_, c1_bh, 0:DK, 0:bDV], H_l1_b)
            T.copy(Hstate[c2_bb, c2_bs_, c2_bh, 0:DK, 0:bDV], H_l1_c)
            # -- Cube: hoisted relay loads: workspace_4 c0/c1/c2 --
            T.copy(workspace_4[cid, slot0_, 0:BS, 0:bDV], V_l1_a)
            T.copy(workspace_4[cid, slot1_, 0:BS, 0:bDV], V_l1_b)
            T.copy(workspace_4[cid, slot2_, 0:BS, 0:bDV], V_l1_c)
            # -- Cube: hoisted relay loads: workspace_3 c0/c1/c2 --
            T.copy(workspace_3[cid, slot0_, 0:BS, 0:BS], A_l1_a)
            T.copy(workspace_3[cid, slot1_, 0:BS, 0:BS], A_l1_b)
            T.copy(workspace_3[cid, slot2_, 0:BS, 0:BS], A_l1_c)
            # -- Cube: hoisted relay loads: workspace_1 c0/c1/c2 --
            T.copy(workspace_1[cid, slot0_, 0:BS, 0:DK], qg_hi_l1_a)
            T.copy(workspace_1[cid, slot1_, 0:BS, 0:DK], qg_hi_l1_b)
            T.copy(workspace_1[cid, slot2_, 0:BS, 0:DK], qg_hi_l1_c)
            # -- Cube: hoisted relay loads: workspace_2 c0/c1/c2 --
            T.copy(workspace_2[cid, slot0_, 0:BS, 0:DK], qg_lo_l1_a)
            T.copy(workspace_2[cid, slot1_, 0:BS, 0:DK], qg_lo_l1_b)
            T.copy(workspace_2[cid, slot2_, 0:BS, 0:DK], qg_lo_l1_c)
            # -- Vector: chunk 0: mask + a relay --
            T.copy(a_half, a_f32)
            T.tile.select(
                a_f32[0 : BS // 2, 0:BS],
                msk_bits,
                a_f32[0 : BS // 2, 0:BS],
                0.0,
                "VSEL_TENSOR_SCALAR_MODE",
            )
            T.copy(a_f32, a_mask)
            T.copy(a_mask, workspace_3[cid, slot0_, 0:BS, 0:BS])
            # -- Vector: chunk 0: gate chain + hi/lo relays --
            T.copy(q_half, q_ub)
            T.tile.mul(g_ub, g_ub, LN2)
            T.tile.exp(g_ub, g_ub)
            T.tile.mul(q_ub, q_ub, scale)
            T.tile.mul(q_ub, q_ub, g_ub)
            T.copy(q_ub, qg_hi_ub)
            T.copy(qg_hi_ub, qg_hi_f32)
            T.tile.sub(q_ub, q_ub, qg_hi_f32)
            T.copy(q_ub, qg_lo_ub)
            T.copy(qg_hi_ub, workspace_1[cid, slot0_, 0:BS, 0:DK])
            T.copy(qg_lo_ub, workspace_2[cid, slot0_, 0:BS, 0:DK])
            # -- Vector: prefetch next batch's chunk 0 a/q/g --
            T.copy(A[np0_bb, np0_s0 : np0_s0 + BS, np0_bh * BS : (np0_bh + 1) * BS], a_half)
            T.copy(Q[np0_bb, np0_s0 : np0_s0 + BS, np0_bh * DK : (np0_bh + 1) * DK], q_half)
            T.copy(G[np0_bb, np0_s0 : np0_s0 + BS, np0_bh * DK : (np0_bh + 1) * DK], g_ub)
            # -- Cube: gemm chain chunk 0 (L1 set _a) --
            T.gemm_v0(A_l1_a, V_l1_a, o_frag_a, init=True)
            T.gemm_v0(qg_hi_l1_a, H_l1_a, o_frag_a, init=False)
            T.gemm_v0(qg_lo_l1_a, H_l1_a, o_frag_a, init=False)
            # -- Cube: FIX chunk 0 (direct strided O store) --
            T.copy(o_frag_a, O[c0_bb, c0_s0 : c0_s0 + BS, c0_bh * bDV : (c0_bh + 1) * bDV])
            T.set_cross_flag("FIX", THROTTLE_FLAG_ID)
            # -- Vector: chunk 1: mask + a relay --
            T.copy(a_half2, a_f32)
            T.tile.select(
                a_f32[0 : BS // 2, 0:BS],
                msk_bits,
                a_f32[0 : BS // 2, 0:BS],
                0.0,
                "VSEL_TENSOR_SCALAR_MODE",
            )
            T.copy(a_f32, a_mask)
            T.copy(a_mask, workspace_3[cid, slot1_, 0:BS, 0:BS])
            # -- Vector: chunk 1: gate chain + hi/lo relays --
            T.copy(q_half2, q_ub)
            T.tile.mul(g_ub2, g_ub2, LN2)
            T.tile.exp(g_ub2, g_ub2)
            T.tile.mul(q_ub, q_ub, scale)
            T.tile.mul(q_ub, q_ub, g_ub2)
            T.copy(q_ub, qg_hi_ub)
            T.copy(qg_hi_ub, qg_hi_f32)
            T.tile.sub(q_ub, q_ub, qg_hi_f32)
            T.copy(q_ub, qg_lo_ub)
            T.copy(qg_hi_ub, workspace_1[cid, slot1_, 0:BS, 0:DK])
            T.copy(qg_lo_ub, workspace_2[cid, slot1_, 0:BS, 0:DK])
            # -- Vector: prefetch next batch's chunk 1 a/q/g --
            T.copy(A[np1_bb, np1_s0 : np1_s0 + BS, np1_bh * BS : (np1_bh + 1) * BS], a_half2)
            T.copy(Q[np1_bb, np1_s0 : np1_s0 + BS, np1_bh * DK : (np1_bh + 1) * DK], q_half2)
            T.copy(G[np1_bb, np1_s0 : np1_s0 + BS, np1_bh * DK : (np1_bh + 1) * DK], g_ub2)
            # -- Cube: gemm chain chunk 1 (L1 set _b) --
            T.gemm_v0(A_l1_b, V_l1_b, o_frag_b, init=True)
            T.gemm_v0(qg_hi_l1_b, H_l1_b, o_frag_b, init=False)
            T.gemm_v0(qg_lo_l1_b, H_l1_b, o_frag_b, init=False)
            # -- Cube: FIX chunk 1 (direct strided O store) --
            T.copy(o_frag_b, O[c1_bb, c1_s0 : c1_s0 + BS, c1_bh * bDV : (c1_bh + 1) * bDV])
            # -- Vector: chunk 2: mask + a relay --
            T.copy(a_half3, a_f32)
            T.tile.select(
                a_f32[0 : BS // 2, 0:BS],
                msk_bits,
                a_f32[0 : BS // 2, 0:BS],
                0.0,
                "VSEL_TENSOR_SCALAR_MODE",
            )
            T.copy(a_f32, a_mask)
            T.copy(a_mask, workspace_3[cid, slot2_, 0:BS, 0:BS])
            # -- Vector: chunk 2: gate chain + hi/lo relays --
            T.copy(q_half3, q_ub)
            T.tile.mul(g_ub3, g_ub3, LN2)
            T.tile.exp(g_ub3, g_ub3)
            T.tile.mul(q_ub, q_ub, scale)
            T.tile.mul(q_ub, q_ub, g_ub3)
            T.copy(q_ub, qg_hi_ub)
            T.copy(qg_hi_ub, qg_hi_f32)
            T.tile.sub(q_ub, q_ub, qg_hi_f32)
            T.copy(q_ub, qg_lo_ub)
            T.copy(qg_hi_ub, workspace_1[cid, slot2_, 0:BS, 0:DK])
            T.copy(qg_lo_ub, workspace_2[cid, slot2_, 0:BS, 0:DK])
            # -- Vector: prefetch next batch's chunk 2 a/q/g --
            T.copy(A[np2_bb, np2_s0 : np2_s0 + BS, np2_bh * BS : (np2_bh + 1) * BS], a_half3)
            T.copy(Q[np2_bb, np2_s0 : np2_s0 + BS, np2_bh * DK : (np2_bh + 1) * DK], q_half3)
            T.copy(G[np2_bb, np2_s0 : np2_s0 + BS, np2_bh * DK : (np2_bh + 1) * DK], g_ub3)
            # -- Cube: gemm chain chunk 2 (L1 set _c) --
            T.gemm_v0(A_l1_c, V_l1_c, o_frag_a, init=True)
            T.gemm_v0(qg_hi_l1_c, H_l1_c, o_frag_a, init=False)
            T.gemm_v0(qg_lo_l1_c, H_l1_c, o_frag_a, init=False)
            # -- Cube: FIX chunk 2 (direct strided O store) --
            T.copy(o_frag_a, O[c2_bb, c2_s0 : c2_s0 + BS, c2_bh * bDV : (c2_bh + 1) * bDV])

            # ---- steady-state main loop: Select-free fast path (k = 2 .. K_MAX).
            # One pipeline step consumes THREE strided tasks (3k, 3k+1, 3k+2); the
            # per-batch sync scaffolding (throttle wait/set, loop control + address
            # chain, top-of-iteration WAR pairs) amortizes over 3 tasks (per-task
            # compute is nearly free -- the per-batch
            # fixed sync protocol IS the wall).
            for kk in T.serial(N_MAIN):
                # ---- main batch (Select-free) ----
                ta0_ = 3 * (kk + 2) + 0
                tida0_ = cid + ta0_ * CORE_EFF
                c0_bs_ = tida0_ % NT
                c0_bbh = tida0_ // NT
                c0_bb = c0_bbh // H
                c0_bh = c0_bbh % H
                c0_s0 = c0_bs_ * BS
                ta1_ = 3 * (kk + 2) + 1
                tida1_ = cid + ta1_ * CORE_EFF
                c1_bs_ = tida1_ % NT
                c1_bbh = tida1_ // NT
                c1_bb = c1_bbh // H
                c1_bh = c1_bbh % H
                c1_s0 = c1_bs_ * BS
                ta2_ = 3 * (kk + 2) + 2
                tida2_ = cid + ta2_ * CORE_EFF
                c2_bs_ = tida2_ % NT
                c2_bbh = tida2_ // NT
                c2_bb = c2_bbh // H
                c2_bh = c2_bbh % H
                c2_s0 = c2_bs_ * BS
                nta0_ = ta0_ + 3
                ntida0_ = cid + nta0_ * CORE_EFF
                np0_bs_ = ntida0_ % NT
                np0_bbh = ntida0_ // NT
                np0_bb = np0_bbh // H
                np0_bh = np0_bbh % H
                np0_s0 = np0_bs_ * BS
                nta1_ = ta1_ + 3
                ntida1_ = cid + nta1_ * CORE_EFF
                np1_bs_ = ntida1_ % NT
                np1_bbh = ntida1_ // NT
                np1_bb = np1_bbh // H
                np1_bh = np1_bbh % H
                np1_s0 = np1_bs_ * BS
                nta2_ = ta2_ + 3
                ntida2_ = cid + nta2_ * CORE_EFF
                np2_bs_ = ntida2_ % NT
                np2_bbh = ntida2_ // NT
                np2_bb = np2_bbh // H
                np2_bh = np2_bbh % H
                np2_s0 = np2_bs_ * BS
                slot0_ = ta0_ % NSLOT
                slot1_ = ta1_ % NSLOT
                slot2_ = ta2_ % NSLOT
                # -- Vector: throttle (Cube's chunk-0 FIX of batch k-2) --
                T.wait_cross_flag(THROTTLE_FLAG_ID, "V")
                # -- Vector: v relays x3 first (staging-only) --
                T.copy(v_half, workspace_4[cid, slot0_, 0:BS, 0:bDV])
                T.copy(v_half2, workspace_4[cid, slot1_, 0:BS, 0:bDV])
                T.copy(v_half3, workspace_4[cid, slot2_, 0:BS, 0:bDV])
                # -- Vector: early v loads for the next batch (drain hides under
                # the mask/gate chains; removes the exposed end-of-body MTE2
                # drain from the serial chain) --
                T.copy(V[np0_bb, np0_s0 : np0_s0 + BS, np0_bh * bDV : (np0_bh + 1) * bDV], v_half)
                T.copy(V[np1_bb, np1_s0 : np1_s0 + BS, np1_bh * bDV : (np1_bh + 1) * bDV], v_half2)
                T.copy(V[np2_bb, np2_s0 : np2_s0 + BS, np2_bh * bDV : (np2_bh + 1) * bDV], v_half3)
                # -- Cube: hoisted loads: 3x Hstate (flag-free) --
                T.copy(Hstate[c0_bb, c0_bs_, c0_bh, 0:DK, 0:bDV], H_l1_a)
                T.copy(Hstate[c1_bb, c1_bs_, c1_bh, 0:DK, 0:bDV], H_l1_b)
                T.copy(Hstate[c2_bb, c2_bs_, c2_bh, 0:DK, 0:bDV], H_l1_c)
                # -- Cube: hoisted relay loads: workspace_4 c0/c1/c2 --
                T.copy(workspace_4[cid, slot0_, 0:BS, 0:bDV], V_l1_a)
                T.copy(workspace_4[cid, slot1_, 0:BS, 0:bDV], V_l1_b)
                T.copy(workspace_4[cid, slot2_, 0:BS, 0:bDV], V_l1_c)
                # -- Cube: hoisted relay loads: workspace_3 c0/c1/c2 --
                T.copy(workspace_3[cid, slot0_, 0:BS, 0:BS], A_l1_a)
                T.copy(workspace_3[cid, slot1_, 0:BS, 0:BS], A_l1_b)
                T.copy(workspace_3[cid, slot2_, 0:BS, 0:BS], A_l1_c)
                # -- Cube: hoisted relay loads: workspace_1 c0/c1/c2 --
                T.copy(workspace_1[cid, slot0_, 0:BS, 0:DK], qg_hi_l1_a)
                T.copy(workspace_1[cid, slot1_, 0:BS, 0:DK], qg_hi_l1_b)
                T.copy(workspace_1[cid, slot2_, 0:BS, 0:DK], qg_hi_l1_c)
                # -- Cube: hoisted relay loads: workspace_2 c0/c1/c2 --
                T.copy(workspace_2[cid, slot0_, 0:BS, 0:DK], qg_lo_l1_a)
                T.copy(workspace_2[cid, slot1_, 0:BS, 0:DK], qg_lo_l1_b)
                T.copy(workspace_2[cid, slot2_, 0:BS, 0:DK], qg_lo_l1_c)
                # -- Vector: chunk 0: mask + a relay --
                T.copy(a_half, a_f32)
                T.tile.select(
                    a_f32[0 : BS // 2, 0:BS],
                    msk_bits,
                    a_f32[0 : BS // 2, 0:BS],
                    0.0,
                    "VSEL_TENSOR_SCALAR_MODE",
                )
                T.copy(a_f32, a_mask)
                T.copy(a_mask, workspace_3[cid, slot0_, 0:BS, 0:BS])
                # -- Vector: chunk 0: gate chain + hi/lo relays --
                T.copy(q_half, q_ub)
                T.tile.mul(g_ub, g_ub, LN2)
                T.tile.exp(g_ub, g_ub)
                T.tile.mul(q_ub, q_ub, scale)
                T.tile.mul(q_ub, q_ub, g_ub)
                T.copy(q_ub, qg_hi_ub)
                T.copy(qg_hi_ub, qg_hi_f32)
                T.tile.sub(q_ub, q_ub, qg_hi_f32)
                T.copy(q_ub, qg_lo_ub)
                T.copy(qg_hi_ub, workspace_1[cid, slot0_, 0:BS, 0:DK])
                T.copy(qg_lo_ub, workspace_2[cid, slot0_, 0:BS, 0:DK])
                # -- Vector: prefetch next batch's chunk 0 a/q/g --
                T.copy(A[np0_bb, np0_s0 : np0_s0 + BS, np0_bh * BS : (np0_bh + 1) * BS], a_half)
                T.copy(Q[np0_bb, np0_s0 : np0_s0 + BS, np0_bh * DK : (np0_bh + 1) * DK], q_half)
                T.copy(G[np0_bb, np0_s0 : np0_s0 + BS, np0_bh * DK : (np0_bh + 1) * DK], g_ub)
                # -- Cube: gemm chain chunk 0 (L1 set _a) --
                T.gemm_v0(A_l1_a, V_l1_a, o_frag_a, init=True)
                T.gemm_v0(qg_hi_l1_a, H_l1_a, o_frag_a, init=False)
                T.gemm_v0(qg_lo_l1_a, H_l1_a, o_frag_a, init=False)
                # -- Cube: FIX chunk 0 (direct strided O store) --
                T.copy(o_frag_a, O[c0_bb, c0_s0 : c0_s0 + BS, c0_bh * bDV : (c0_bh + 1) * bDV])
                T.set_cross_flag("FIX", THROTTLE_FLAG_ID)
                # -- Vector: chunk 1: mask + a relay --
                T.copy(a_half2, a_f32)
                T.tile.select(
                    a_f32[0 : BS // 2, 0:BS],
                    msk_bits,
                    a_f32[0 : BS // 2, 0:BS],
                    0.0,
                    "VSEL_TENSOR_SCALAR_MODE",
                )
                T.copy(a_f32, a_mask)
                T.copy(a_mask, workspace_3[cid, slot1_, 0:BS, 0:BS])
                # -- Vector: chunk 1: gate chain + hi/lo relays --
                T.copy(q_half2, q_ub)
                T.tile.mul(g_ub2, g_ub2, LN2)
                T.tile.exp(g_ub2, g_ub2)
                T.tile.mul(q_ub, q_ub, scale)
                T.tile.mul(q_ub, q_ub, g_ub2)
                T.copy(q_ub, qg_hi_ub)
                T.copy(qg_hi_ub, qg_hi_f32)
                T.tile.sub(q_ub, q_ub, qg_hi_f32)
                T.copy(q_ub, qg_lo_ub)
                T.copy(qg_hi_ub, workspace_1[cid, slot1_, 0:BS, 0:DK])
                T.copy(qg_lo_ub, workspace_2[cid, slot1_, 0:BS, 0:DK])
                # -- Vector: prefetch next batch's chunk 1 a/q/g --
                T.copy(A[np1_bb, np1_s0 : np1_s0 + BS, np1_bh * BS : (np1_bh + 1) * BS], a_half2)
                T.copy(Q[np1_bb, np1_s0 : np1_s0 + BS, np1_bh * DK : (np1_bh + 1) * DK], q_half2)
                T.copy(G[np1_bb, np1_s0 : np1_s0 + BS, np1_bh * DK : (np1_bh + 1) * DK], g_ub2)
                # -- Cube: gemm chain chunk 1 (L1 set _b) --
                T.gemm_v0(A_l1_b, V_l1_b, o_frag_b, init=True)
                T.gemm_v0(qg_hi_l1_b, H_l1_b, o_frag_b, init=False)
                T.gemm_v0(qg_lo_l1_b, H_l1_b, o_frag_b, init=False)
                # -- Cube: FIX chunk 1 (direct strided O store) --
                T.copy(o_frag_b, O[c1_bb, c1_s0 : c1_s0 + BS, c1_bh * bDV : (c1_bh + 1) * bDV])
                # -- Vector: chunk 2: mask + a relay --
                T.copy(a_half3, a_f32)
                T.tile.select(
                    a_f32[0 : BS // 2, 0:BS],
                    msk_bits,
                    a_f32[0 : BS // 2, 0:BS],
                    0.0,
                    "VSEL_TENSOR_SCALAR_MODE",
                )
                T.copy(a_f32, a_mask)
                T.copy(a_mask, workspace_3[cid, slot2_, 0:BS, 0:BS])
                # -- Vector: chunk 2: gate chain + hi/lo relays --
                T.copy(q_half3, q_ub)
                T.tile.mul(g_ub3, g_ub3, LN2)
                T.tile.exp(g_ub3, g_ub3)
                T.tile.mul(q_ub, q_ub, scale)
                T.tile.mul(q_ub, q_ub, g_ub3)
                T.copy(q_ub, qg_hi_ub)
                T.copy(qg_hi_ub, qg_hi_f32)
                T.tile.sub(q_ub, q_ub, qg_hi_f32)
                T.copy(q_ub, qg_lo_ub)
                T.copy(qg_hi_ub, workspace_1[cid, slot2_, 0:BS, 0:DK])
                T.copy(qg_lo_ub, workspace_2[cid, slot2_, 0:BS, 0:DK])
                # -- Vector: prefetch next batch's chunk 2 a/q/g --
                T.copy(A[np2_bb, np2_s0 : np2_s0 + BS, np2_bh * BS : (np2_bh + 1) * BS], a_half3)
                T.copy(Q[np2_bb, np2_s0 : np2_s0 + BS, np2_bh * DK : (np2_bh + 1) * DK], q_half3)
                T.copy(G[np2_bb, np2_s0 : np2_s0 + BS, np2_bh * DK : (np2_bh + 1) * DK], g_ub3)
                # -- Cube: gemm chain chunk 2 (L1 set _c) --
                T.gemm_v0(A_l1_c, V_l1_c, o_frag_a, init=True)
                T.gemm_v0(qg_hi_l1_c, H_l1_c, o_frag_a, init=False)
                T.gemm_v0(qg_lo_l1_c, H_l1_c, o_frag_a, init=False)
                # -- Cube: FIX chunk 2 (direct strided O store) --
                T.copy(o_frag_a, O[c2_bb, c2_s0 : c2_s0 + BS, c2_bh * bDV : (c2_bh + 1) * bDV])
            # ---- tail loop: Select-guarded redirects ----
            for kk2 in T.serial(N_TAIL):
                # ---- tail batch (Select-guarded) ----
                ta0_ = 3 * (kk2 + max(1, min(K_MAX, NB)) + 1) + 0
                tida0_ = cid + ta0_ * CORE_EFF
                pt0_ = T.if_then_else(tida0_ < GRID, tida0_, cid)
                c0_bs_ = pt0_ % NT
                c0_bbh = pt0_ // NT
                c0_bb = c0_bbh // H
                c0_bh = c0_bbh % H
                c0_s0 = c0_bs_ * BS
                ta1_ = 3 * (kk2 + max(1, min(K_MAX, NB)) + 1) + 1
                tida1_ = cid + ta1_ * CORE_EFF
                pt1_ = T.if_then_else(tida1_ < GRID, tida1_, cid)
                c1_bs_ = pt1_ % NT
                c1_bbh = pt1_ // NT
                c1_bb = c1_bbh // H
                c1_bh = c1_bbh % H
                c1_s0 = c1_bs_ * BS
                ta2_ = 3 * (kk2 + max(1, min(K_MAX, NB)) + 1) + 2
                tida2_ = cid + ta2_ * CORE_EFF
                pt2_ = T.if_then_else(tida2_ < GRID, tida2_, cid)
                c2_bs_ = pt2_ % NT
                c2_bbh = pt2_ // NT
                c2_bb = c2_bbh // H
                c2_bh = c2_bbh % H
                c2_s0 = c2_bs_ * BS
                nta0_ = ta0_ + 3
                ntida0_ = cid + nta0_ * CORE_EFF
                np0_ = T.if_then_else(ntida0_ < GRID, ntida0_, cid)
                np0_bs_ = np0_ % NT
                np0_bbh = np0_ // NT
                np0_bb = np0_bbh // H
                np0_bh = np0_bbh % H
                np0_s0 = np0_bs_ * BS
                nta1_ = ta1_ + 3
                ntida1_ = cid + nta1_ * CORE_EFF
                np1_ = T.if_then_else(ntida1_ < GRID, ntida1_, cid)
                np1_bs_ = np1_ % NT
                np1_bbh = np1_ // NT
                np1_bb = np1_bbh // H
                np1_bh = np1_bbh % H
                np1_s0 = np1_bs_ * BS
                nta2_ = ta2_ + 3
                ntida2_ = cid + nta2_ * CORE_EFF
                np2_ = T.if_then_else(ntida2_ < GRID, ntida2_, cid)
                np2_bs_ = np2_ % NT
                np2_bbh = np2_ // NT
                np2_bb = np2_bbh // H
                np2_bh = np2_bbh % H
                np2_s0 = np2_bs_ * BS
                slot0_ = ta0_ % NSLOT
                slot1_ = ta1_ % NSLOT
                slot2_ = ta2_ % NSLOT
                # -- Vector: throttle (Cube's chunk-0 FIX of batch k-2) --
                T.wait_cross_flag(THROTTLE_FLAG_ID, "V")
                # -- Vector: v relays x3 first (staging-only) --
                T.copy(v_half, workspace_4[cid, slot0_, 0:BS, 0:bDV])
                T.copy(v_half2, workspace_4[cid, slot1_, 0:BS, 0:bDV])
                T.copy(v_half3, workspace_4[cid, slot2_, 0:BS, 0:bDV])
                # -- Vector: early v loads for the next batch (drain hides under
                # the mask/gate chains; removes the exposed end-of-body MTE2
                # drain from the serial chain) --
                T.copy(V[np0_bb, np0_s0 : np0_s0 + BS, np0_bh * bDV : (np0_bh + 1) * bDV], v_half)
                T.copy(V[np1_bb, np1_s0 : np1_s0 + BS, np1_bh * bDV : (np1_bh + 1) * bDV], v_half2)
                T.copy(V[np2_bb, np2_s0 : np2_s0 + BS, np2_bh * bDV : (np2_bh + 1) * bDV], v_half3)
                # -- Cube: hoisted loads: 3x Hstate (flag-free) --
                T.copy(Hstate[c0_bb, c0_bs_, c0_bh, 0:DK, 0:bDV], H_l1_a)
                T.copy(Hstate[c1_bb, c1_bs_, c1_bh, 0:DK, 0:bDV], H_l1_b)
                T.copy(Hstate[c2_bb, c2_bs_, c2_bh, 0:DK, 0:bDV], H_l1_c)
                # -- Cube: hoisted relay loads: workspace_4 c0/c1/c2 --
                T.copy(workspace_4[cid, slot0_, 0:BS, 0:bDV], V_l1_a)
                T.copy(workspace_4[cid, slot1_, 0:BS, 0:bDV], V_l1_b)
                T.copy(workspace_4[cid, slot2_, 0:BS, 0:bDV], V_l1_c)
                # -- Cube: hoisted relay loads: workspace_3 c0/c1/c2 --
                T.copy(workspace_3[cid, slot0_, 0:BS, 0:BS], A_l1_a)
                T.copy(workspace_3[cid, slot1_, 0:BS, 0:BS], A_l1_b)
                T.copy(workspace_3[cid, slot2_, 0:BS, 0:BS], A_l1_c)
                # -- Cube: hoisted relay loads: workspace_1 c0/c1/c2 --
                T.copy(workspace_1[cid, slot0_, 0:BS, 0:DK], qg_hi_l1_a)
                T.copy(workspace_1[cid, slot1_, 0:BS, 0:DK], qg_hi_l1_b)
                T.copy(workspace_1[cid, slot2_, 0:BS, 0:DK], qg_hi_l1_c)
                # -- Cube: hoisted relay loads: workspace_2 c0/c1/c2 --
                T.copy(workspace_2[cid, slot0_, 0:BS, 0:DK], qg_lo_l1_a)
                T.copy(workspace_2[cid, slot1_, 0:BS, 0:DK], qg_lo_l1_b)
                T.copy(workspace_2[cid, slot2_, 0:BS, 0:DK], qg_lo_l1_c)
                # -- Vector: chunk 0: mask + a relay --
                T.copy(a_half, a_f32)
                T.tile.select(
                    a_f32[0 : BS // 2, 0:BS],
                    msk_bits,
                    a_f32[0 : BS // 2, 0:BS],
                    0.0,
                    "VSEL_TENSOR_SCALAR_MODE",
                )
                T.copy(a_f32, a_mask)
                T.copy(a_mask, workspace_3[cid, slot0_, 0:BS, 0:BS])
                # -- Vector: chunk 0: gate chain + hi/lo relays --
                T.copy(q_half, q_ub)
                T.tile.mul(g_ub, g_ub, LN2)
                T.tile.exp(g_ub, g_ub)
                T.tile.mul(q_ub, q_ub, scale)
                T.tile.mul(q_ub, q_ub, g_ub)
                T.copy(q_ub, qg_hi_ub)
                T.copy(qg_hi_ub, qg_hi_f32)
                T.tile.sub(q_ub, q_ub, qg_hi_f32)
                T.copy(q_ub, qg_lo_ub)
                T.copy(qg_hi_ub, workspace_1[cid, slot0_, 0:BS, 0:DK])
                T.copy(qg_lo_ub, workspace_2[cid, slot0_, 0:BS, 0:DK])
                # -- Vector: prefetch next batch's chunk 0 a/q/g --
                T.copy(A[np0_bb, np0_s0 : np0_s0 + BS, np0_bh * BS : (np0_bh + 1) * BS], a_half)
                T.copy(Q[np0_bb, np0_s0 : np0_s0 + BS, np0_bh * DK : (np0_bh + 1) * DK], q_half)
                T.copy(G[np0_bb, np0_s0 : np0_s0 + BS, np0_bh * DK : (np0_bh + 1) * DK], g_ub)
                # -- Cube: gemm chain chunk 0 (L1 set _a) --
                T.gemm_v0(A_l1_a, V_l1_a, o_frag_a, init=True)
                T.gemm_v0(qg_hi_l1_a, H_l1_a, o_frag_a, init=False)
                T.gemm_v0(qg_lo_l1_a, H_l1_a, o_frag_a, init=False)
                # -- Cube: FIX chunk 0 (direct strided O store) --
                T.copy(o_frag_a, O[c0_bb, c0_s0 : c0_s0 + BS, c0_bh * bDV : (c0_bh + 1) * bDV])
                T.set_cross_flag("FIX", THROTTLE_FLAG_ID)
                # -- Vector: chunk 1: mask + a relay --
                T.copy(a_half2, a_f32)
                T.tile.select(
                    a_f32[0 : BS // 2, 0:BS],
                    msk_bits,
                    a_f32[0 : BS // 2, 0:BS],
                    0.0,
                    "VSEL_TENSOR_SCALAR_MODE",
                )
                T.copy(a_f32, a_mask)
                T.copy(a_mask, workspace_3[cid, slot1_, 0:BS, 0:BS])
                # -- Vector: chunk 1: gate chain + hi/lo relays --
                T.copy(q_half2, q_ub)
                T.tile.mul(g_ub2, g_ub2, LN2)
                T.tile.exp(g_ub2, g_ub2)
                T.tile.mul(q_ub, q_ub, scale)
                T.tile.mul(q_ub, q_ub, g_ub2)
                T.copy(q_ub, qg_hi_ub)
                T.copy(qg_hi_ub, qg_hi_f32)
                T.tile.sub(q_ub, q_ub, qg_hi_f32)
                T.copy(q_ub, qg_lo_ub)
                T.copy(qg_hi_ub, workspace_1[cid, slot1_, 0:BS, 0:DK])
                T.copy(qg_lo_ub, workspace_2[cid, slot1_, 0:BS, 0:DK])
                # -- Vector: prefetch next batch's chunk 1 a/q/g --
                T.copy(A[np1_bb, np1_s0 : np1_s0 + BS, np1_bh * BS : (np1_bh + 1) * BS], a_half2)
                T.copy(Q[np1_bb, np1_s0 : np1_s0 + BS, np1_bh * DK : (np1_bh + 1) * DK], q_half2)
                T.copy(G[np1_bb, np1_s0 : np1_s0 + BS, np1_bh * DK : (np1_bh + 1) * DK], g_ub2)
                # -- Cube: gemm chain chunk 1 (L1 set _b) --
                T.gemm_v0(A_l1_b, V_l1_b, o_frag_b, init=True)
                T.gemm_v0(qg_hi_l1_b, H_l1_b, o_frag_b, init=False)
                T.gemm_v0(qg_lo_l1_b, H_l1_b, o_frag_b, init=False)
                # -- Cube: FIX chunk 1 (direct strided O store) --
                T.copy(o_frag_b, O[c1_bb, c1_s0 : c1_s0 + BS, c1_bh * bDV : (c1_bh + 1) * bDV])
                # -- Vector: chunk 2: mask + a relay --
                T.copy(a_half3, a_f32)
                T.tile.select(
                    a_f32[0 : BS // 2, 0:BS],
                    msk_bits,
                    a_f32[0 : BS // 2, 0:BS],
                    0.0,
                    "VSEL_TENSOR_SCALAR_MODE",
                )
                T.copy(a_f32, a_mask)
                T.copy(a_mask, workspace_3[cid, slot2_, 0:BS, 0:BS])
                # -- Vector: chunk 2: gate chain + hi/lo relays --
                T.copy(q_half3, q_ub)
                T.tile.mul(g_ub3, g_ub3, LN2)
                T.tile.exp(g_ub3, g_ub3)
                T.tile.mul(q_ub, q_ub, scale)
                T.tile.mul(q_ub, q_ub, g_ub3)
                T.copy(q_ub, qg_hi_ub)
                T.copy(qg_hi_ub, qg_hi_f32)
                T.tile.sub(q_ub, q_ub, qg_hi_f32)
                T.copy(q_ub, qg_lo_ub)
                T.copy(qg_hi_ub, workspace_1[cid, slot2_, 0:BS, 0:DK])
                T.copy(qg_lo_ub, workspace_2[cid, slot2_, 0:BS, 0:DK])
                # -- Vector: prefetch next batch's chunk 2 a/q/g --
                T.copy(A[np2_bb, np2_s0 : np2_s0 + BS, np2_bh * BS : (np2_bh + 1) * BS], a_half3)
                T.copy(Q[np2_bb, np2_s0 : np2_s0 + BS, np2_bh * DK : (np2_bh + 1) * DK], q_half3)
                T.copy(G[np2_bb, np2_s0 : np2_s0 + BS, np2_bh * DK : (np2_bh + 1) * DK], g_ub3)
                # -- Cube: gemm chain chunk 2 (L1 set _c) --
                T.gemm_v0(A_l1_c, V_l1_c, o_frag_a, init=True)
                T.gemm_v0(qg_hi_l1_c, H_l1_c, o_frag_a, init=False)
                T.gemm_v0(qg_lo_l1_c, H_l1_c, o_frag_a, init=False)
                # -- Cube: FIX chunk 2 (direct strided O store) --
                T.copy(o_frag_a, O[c2_bb, c2_s0 : c2_s0 + BS, c2_bh * bDV : (c2_bh + 1) * bDV])

            # (end of task loop; o_frag ping-pong by chunk parity: chunks 0/2 -> a,
            # chunk 1 -> b -- chunk 2's gemms only WAR chunk 0's FIX, which has
            # long drained by then.)

    return main


# ============================================================================
# Host wrapper (upstream naming: chunk_gla_fwd_o_gk). Only shape checks +
# Msk constant caching + kernel invocation; no data mutation on the host.
#
# Host-dispatch bypass: the public JIT dispatch path
# (JITKernel.__call__ -> CythonKernelWrapper.forward, tilelang/jit/) repeats
# per-call work that is pure overhead for this FULLY-STATIC kernel: it
# rebuilds a tvm.arith.Analyzer on every call (~290us, only needed to
# simplify symbolic shapes -- this kernel has none), re-resolves all 11
# param shapes through Python isinstance chains, re-validates dtypes and
# devices, and re-allocates the four relay workspaces (~65us). Measured
# total: ~800us host time per call, vs 2.4us for the JIT cache lookup and
# 9us for the actual lib.call enqueue. With a ~1.4ms device-side kernel
# that cost hides behind the device except on the FIRST timed rep after
# an idle sync,
# where it is fully exposed (~+0.83ms on the official 10-rep mean's rep
# 0, i.e. ~+83us on the mean). _FastLaunch below caches, per
# (shape-spec, device, stream): the compiled lib handle, the four relay
# workspaces (internal write-then-read staging scratch -- never returned,
# same-stream launches are device-side serialized, so reuse is safe), and
# prebuilt launch state; the output stays a FRESH torch.empty per call
# (successive calls must not alias, upstream wrapper semantics).
# Correctness guardrails: (1) per-call re-validation of every input's
# shape/dtype/contiguity/device; (2) the cache is armed only after a
# bitwise self-check (public result == two consecutive fast launches
# reusing the same workspaces); (3) ANY exception or guard failure --
# including upstream tilelang refactors touching the adapter internals --
# falls back to the public dispatch path, which owns error messages and
# re-specialization. The launch ABI used is the generated library's
# extern "C" void call(<11 buffer ptrs>, aclrtStream), the same entry the
# cython adapter itself invokes (kernel.adapter access has in-repo
# precedent: examples/torch_tl_ascend/compile_tl_op).
_WS_IDX = [6, 7, 8, 9]  # workspace_idx (absolute param indices)
_OUT_IDX = [10]  # out_idx (absolute param index)


def _check_shape(B, S, H, chunk_size, A):
    assert S % chunk_size == 0, "S must be divisible by chunk_size"
    assert chunk_size == 64, "chunk_size fixed at 64 (block_S semantic)"
    assert A.shape[-1] == chunk_size, "A last dim must equal chunk_size"


_MSK_CACHE = {}


def _get_msk(device, chunk_size):
    """Inclusive-tril bit-packed constant (diagonal=0), built ONCE per
    (device, chunk). Packed LSB-first into chunk*chunk//8 bytes (the VSEL
    selMask convention): bit (t*chunk + j) of the little-endian byte stream
    is 1 iff t >= j. A packed bitmap (not a 0/1 float tensor) is required
    by T.tile.select's selMask operand.
    """
    key = (str(device), chunk_size)
    if key not in _MSK_CACHE:
        n = chunk_size * chunk_size
        m = torch.tril(torch.ones(chunk_size, chunk_size, dtype=torch.uint8), 0).reshape(-1)
        bits = torch.zeros(n // 8, dtype=torch.uint8)
        for b in range(8):
            bits |= (m[b::8] << b).to(torch.uint8)
        _MSK_CACHE[key] = bits.to(device)
    return _MSK_CACHE[key]


class _GuardFail(Exception):
    """Fast-path input guard failed -> caller falls back to the public path."""


class _FastLaunch:
    """Cached low-level launch state for one (spec, device, stream) key.

    Slots-only, no per-call allocation beyond the output tensor. The
    stream is part of the cache key, so the cached raw stream value is
    always the caller's current stream on a hit.
    """

    __slots__ = (
        "lib",
        "ws",
        "ws_ptrs",
        "out_shape",
        "out_dtype",
        "dev",
        "dev_idx",
        "stream_raw",
        "eq_q",
        "eq_v",
        "eq_g",
        "eq_A",
        "eq_h",
        "msk",
    )

    def __init__(self, kernel, dev_idx, stream_raw, msk):
        adapter = kernel.adapter
        params = adapter.params
        # The bypass assumes the compiled signature is exactly the 11
        # static buffers + stream: no dynamic symbolics (they append int64
        # args), no auto-GM injections, no symbolic shape dims.
        if len(params) != 11 or adapter.dynamic_symbolic_map or getattr(adapter, "auto_gm_idx", []):
            raise _GuardFail("unsupported kernel signature")
        if not all(isinstance(d, int) for p in params for d in p.shape):
            raise _GuardFail("symbolic shape dims")
        self.dev_idx = dev_idx
        self.dev = torch.device("npu", dev_idx)
        self.stream_raw = stream_raw
        self.msk = msk
        self.lib = adapter.lib
        # ctypes: with argtypes set, plain ints convert safely to 64-bit
        # pointers (the cython path passes c_void_p objects, which stay
        # compatible).
        self.lib.call.argtypes = [ctypes.c_void_p] * 12
        self.ws = [torch.empty(tuple(params[i].shape), dtype=params[i].dtype, device=self.dev) for i in _WS_IDX]
        self.ws_ptrs = [t.data_ptr() for t in self.ws]
        p_out = params[_OUT_IDX[0]]
        self.out_shape = tuple(p_out.shape)
        self.out_dtype = p_out.dtype

    def set_expect(self, q, v, g, A, h):
        """Snapshot the validated input shapes (arm time)."""
        self.eq_q, self.eq_v, self.eq_g, self.eq_A, self.eq_h = (
            tuple(q.shape),
            tuple(v.shape),
            tuple(g.shape),
            tuple(A.shape),
            tuple(h.shape),
        )

    def launch(self, q, v, g, A, h, msk):
        """Guarded fast launch. Raises _GuardFail -> public path fallback."""
        if (
            q.shape != self.eq_q
            or v.shape != self.eq_v
            or g.shape != self.eq_g
            or A.shape != self.eq_A
            or h.shape != self.eq_h
            or q.dtype is not torch.bfloat16
            or v.dtype is not torch.bfloat16
            or g.dtype is not torch.float32
            or A.dtype is not torch.bfloat16
            or h.dtype is not torch.bfloat16
            or not (q.is_contiguous() and v.is_contiguous() and g.is_contiguous() and A.is_contiguous() and h.is_contiguous())
            or q.device.index != self.dev_idx
            or msk is not self.msk
        ):
            raise _GuardFail("input guard")
        o = torch.empty(self.out_shape, dtype=self.out_dtype, device=self.dev)
        wp = self.ws_ptrs
        self.lib.call(
            q.data_ptr(),
            v.data_ptr(),
            g.data_ptr(),
            A.data_ptr(),
            h.data_ptr(),
            msk.data_ptr(),
            wp[0],
            wp[1],
            wp[2],
            wp[3],
            o.data_ptr(),
            self.stream_raw,
        )
        return o


# None = bypass structurally unavailable for this process (first failure);
# entry None = poisoned for that key (self-check failed) -> public path.
_FAST: dict = {}


def fla_chunk_o(q, v, g, A, h, scale=None, chunk_size=64):
    """Host API (upstream: chunk_gla_fwd_o_gk, use_exp2=True, fixed length).

    q[B,S,H,DK] bf16, v[B,S,H,DV] bf16, g[B,S,H,DK] fp32 (cumulative log2
    decay), A[B,S,H,64] bf16 (intra-chunk attention), h[B,S//64,H,DK,DV]
    bf16 (chunk-level hidden state) -> o[B,S,H,DV] bf16.
    """
    B, S, H, DK = q.shape
    DV = v.shape[-1]
    _check_shape(B, S, H, chunk_size, A)
    if scale is None:
        scale = DK**-0.5
    _pick_block_dk(DK)  # domain gate (contract unchanged)
    _pick_block_dv(DV)
    msk = _get_msk(q.device, chunk_size)
    # ---- fast dispatch (guarded; falls back to the public path) ----
    key = None
    if _FAST is not None:
        try:
            stream_raw = torch.npu.current_stream().npu_stream
            key = (B, S, H, DK, DV, chunk_size, scale, q.device.index, stream_raw)
            entry = _FAST.get(key)
            if entry is not None:
                # no internal synchronize: upstream wrapper parity (the
                # do_bench harness owns synchronization)
                return entry.launch(q, v, g, A, h, msk).view(B, S, H, DV)
        except Exception:
            pass  # guard/launch failure -> public path below
    kernel = fla_chunk_o_kernel(B, S, H, DK, DV, chunk_size, scale, None, None)
    # Zero-copy 3D views [B, S, H*D] of the contiguous bshd tensors (pure
    # metadata, no materialization): puts the S row dim at GM position
    # (gm_dims - ub_dims) so the threads=2 vid-reduction splits rows, not
    # heads (see the kernel docstring item 7). v/h stay 4D/5D (Cube-side).
    o = kernel(
        q.view(B, S, H * DK),
        v.view(B, S, H * DV),  # 3D view: vid-split lands on S rows (item 7)
        g.view(B, S, H * DK),
        A.view(B, S, H * chunk_size),
        h,
        msk,
    )
    # ---- arm the fast path: bitwise self-check vs the public result ----
    if _FAST is not None and key is not None and key not in _FAST:
        try:
            entry = _FastLaunch(kernel, q.device.index, stream_raw, msk)
            entry.set_expect(q, v, g, A, h)
            o_f1 = entry.launch(q, v, g, A, h, msk)
            o_f2 = entry.launch(q, v, g, A, h, msk)  # workspace reuse probe
            if torch.equal(o, o_f1) and torch.equal(o_f1, o_f2):
                _FAST[key] = entry
            else:
                _FAST[key] = None  # poisoned: keep the public path forever
        except Exception:
            _FAST[key] = None
    # no internal synchronize: upstream wrapper parity (the do_bench harness
    # owns synchronization; an internal sync would expose host-side JIT
    # dispatch latency inside the device-side measurement)
    return o.view(B, S, H, DV)


__all__ = [
    "fla_chunk_o",
    "fla_chunk_o_kernel",
    "PASS_CONFIGS",
]


if __name__ == "__main__":
    # Smoke entry (the CI legacy runner executes this file with `python`):
    # L0-gate shape, output-shape/finite checks, "Test Passed!" on success.
    # Numerical accuracy is covered by test_example_fla_chunk_o.py
    # (PyTorch golden).
    import torch_npu  # noqa: F401  (registers .npu() on torch)

    tilelang.cache.clear_cache()
    B, S, H, DK, DV, CS = 1, 128, 2, 64, 64, 64

    gen = torch.Generator().manual_seed(42)
    q = torch.randn(B, S, H, DK, generator=gen)
    v = torch.randn(B, S, H, DV, generator=gen)
    g = torch.randn(B, S, H, DK, generator=gen)
    A = torch.randn(B, S, H, CS, generator=gen)
    h = torch.randn(B, S // CS, H, DK, DV, generator=gen)
    q, v, A, h = (t.to(torch.bfloat16) for t in (q, v, A, h))

    o = fla_chunk_o(q.npu(), v.npu(), g.npu(), A.npu(), h.npu())
    torch.npu.synchronize()
    assert o.shape == (B, S, H, DV) and o.dtype is torch.bfloat16
    assert torch.isfinite(o.float()).all().item(), "o contains NaN/Inf"
    print(f"[SMOKE_PASS] fla_chunk_o B={B} S={S} H={H} DK={DK} DV={DV} chunk={CS} bf16 (gate fp32)")
    print("Test Passed!")
