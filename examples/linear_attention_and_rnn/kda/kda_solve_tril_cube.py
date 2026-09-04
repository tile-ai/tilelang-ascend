"""A = (I + L)^{-1} on the cube, by a doubling Neumann series.

``kda_solve_tril.py`` dispatches here for fp16 / bf16 inputs with C >= 16; it
keeps the vector forward substitution for fp32 (a cube operand cannot be fp32)
and for C below the cube's fractal granularity of 16.

Why this stage moved
--------------------
The vector form runs C-2 = 62 *serial* iterations per [C, C] chunk matrix, each
a [C, C] multiply plus a reduction -- 167 us, entirely on the vector cores.  The
hand-written AscendC operator puts this stage on the cube (``prepare.h:101``
static_assert, ``:1503`` BlockMmadTla), and it is one of the larger terms in our
1065 us of vector time against its 160 us.

The maths: an identity, not an approximation
--------------------------------------------
L is strictly lower triangular, hence nilpotent: L^C = 0.  With M = -L,

    (I + L)^{-1} = sum_k M^k = (I + M)(I + M^2)(I + M^4)(I + M^8) ...

Each factor doubles the powers covered, so three steps already reach M^15.  As a
recurrence:

    S0 = I - L                      P1 = L @ L
    S1 = S0 @ I + S0 @ P1           P2 = P1 @ P1
    S2 = S1 @ I + S1 @ P2           P3 = P2 @ P2
    S3 = S2 @ I + S2 @ P3

Nine [C, C] matmuls in place of 62 serial forward-substitution rows.

How many steps is enough was measured, not assumed --
``PERF/probes/probe_neumann_precision.py`` sweeps the step count under the
normal / forget / extreme / keep gates.  With fp16 operands and fp32
accumulation the gap to forward substitution is 2.7e-5 / 3.5e-5 / 8.1e-6 /
1.66e-4 against a 5e-3 tolerance.  The reason is that L's entries are far below
1, so the powers decay fast (this file's own acceptance run measures P1 5.4e-2,
P2 2.1e-3, P3 6.6e-7); from the fourth step on they are all zero in fp16.

Not one line of arithmetic on the vector side
----------------------------------------------
Even S0 = I - L is built on the cube:

    acc  = I @ I        init=True    -> I
    acc += L @ (-I)     init=False   -> I - L

I and -I are compile-time constant inputs.  This shape was forced by
``PERF/probes/probe_v_sub.py``: computing I - L in the V phase produced a wrong
chain, while the cube form is exact (relerr 0).

The V phase is left with two moves only: reshaping the *strided* [B, SEQ, HV, C]
block into a contiguous workspace (the cube's GM->L1 path only takes contiguous
blocks), and carrying the result back.

Synchronisation
---------------
Every ``T.copy(acc, ws)`` is immediately followed by a GM->L1 read of the same
block.  AUTO_SYNC does not cover the FIX-to-MTE2 hop, so
``set_flag("fix", "mte2")`` / ``wait_flag`` have to be written by hand.  Missing
one shows up as nan, not as a tolerance failure.

Ragged tail
-----------
With SEQ % C != 0 the last chunk has only R valid rows; the load is clamped by
compute_valid_extent and rows R..C-1 keep whatever was in UB.  Those rows are
*not* inert: the matmul sums over every j, so a valid row i < R picks up
L[i, j] * X[j, :] for j >= R.  The second factor is 0 by strict lower
triangularity, but 0 * inf is nan -- hence the fill before the load, for the
same reason as in the vector version.
"""

import os
import sys

import torch
import tilelang
from tilelang import language as T

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import kda_chunk_ref  # noqa: E402
import kda_varlen as _VL  # noqa: E402
import kda_solve_tril as _VEC  # noqa: E402

pass_configs = {tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True}

VEC_NUM = 2
# Doubling steps; the series covers M^(2^(STEPS+1) - 1).  The default of 2
# (covering M^7) is measured, not picked: probe_neumann_precision.py sweeps the
# step count under the hardest gate (keep) and reports
#
#     covers L^3     fp32 residual 6.189e-05    fp16 input 1.664e-04
#     covers L^7     fp32 residual 9.450e-08    fp16 input 1.664e-04   <- here
#     covers L^15    fp32 residual 9.450e-08    fp16 input 1.664e-04
#
# L^7 is where the truncation residual first sits three orders below fp16
# rounding, and no further step changes a digit.  L^3 is still far off in fp32
# terms, which is what pins the lower bound.  The tolerance is 5e-3.
#
# STEPS = 1 was tried and *fails on the keep gate in the pipeline*: 8.260e-03
# against the 5e-3 tolerance.  The table above comes from the probe's synthetic
# L, which understates the truncation residual of the real keep gate (alpha -> 1
# leaves L's entries larger, so the Neumann series converges more slowly).  The
# "fp16 column is identical at L^3 and L^7" is a property of the probe's data,
# not of the pipeline -- do not use it to justify dropping a step.
#
# Each extra step costs about 30 us in the profile (145.26 vs 115.56).  The
# environment override exists so the chain length can be bisected when something
# goes wrong.
STEPS = int(os.environ.get("KDA_SOLVE_STEPS", "2"))
assert 1 <= STEPS <= 3, "only three steps are unrolled explicitly in the kernel"


@tilelang.jit(
    out_idx=[-1],
    workspace_idx=[-9, -8, -7, -6, -5, -4, -3, -2],
    pass_configs=pass_configs,
)
def kda_solve_tril_cube_ker(B, SEQ, HV, C, dtype="float16", accum_dtype="float"):
    N = -(-SEQ // C)  # chunks per sequence; the last one may be ragged
    R = SEQ % C
    RAGGED = R != 0
    total = B * HV * N  # one [C, C] chunk matrix per task
    blocks = (total + VEC_NUM - 1) // VEC_NUM

    @T.prim_func
    def main(
        Ltri: T.Tensor([B, SEQ, HV, C], dtype),  # type: ignore
        Idt: T.Tensor([C, C], dtype),  # type: ignore    identity; a cube operand must be dtype
        NegI: T.Tensor([C, C], dtype),  # type: ignore   -I
        ws_l: T.Tensor([total, C, C], dtype),  # type: ignore   L, flattened by V
        ws_s0: T.Tensor([total, C, C], dtype),  # type: ignore
        ws_s1: T.Tensor([total, C, C], dtype),  # type: ignore
        ws_s2: T.Tensor([total, C, C], dtype),  # type: ignore
        ws_s3: T.Tensor([total, C, C], dtype),  # type: ignore
        ws_p1: T.Tensor([total, C, C], dtype),  # type: ignore
        ws_p2: T.Tensor([total, C, C], dtype),  # type: ignore
        ws_p3: T.Tensor([total, C, C], dtype),  # type: ignore
        A: T.Tensor([B, SEQ, HV, C], dtype),  # type: ignore
    ):
        # One block is 1 AIC plus 2 AIVs.  Each vector core flattens one chunk
        # matrix; the cube then works through both of them in turn.
        with T.Kernel(blocks, is_npu=True) as (cid, vid):
            l_l1 = T.alloc_L1([C, C], dtype)
            a_l1 = T.alloc_L1([C, C], dtype)
            b_l1 = T.alloc_L1([C, C], dtype)
            s_l1 = T.alloc_L1([C, C], dtype)
            i_l1 = T.alloc_L1([C, C], dtype)
            n_l1 = T.alloc_L1([C, C], dtype)
            acc = T.alloc_L0C([C, C], accum_dtype)  # the S chain
            acc2 = T.alloc_L0C([C, C], accum_dtype)  # the P chain, concurrent with S
            x_io = T.alloc_ub([C, C], dtype)

            with T.Scope("V"):
                raw = cid * VEC_NUM + vid
                # An odd task count leaves one core over.  Letting it redo the
                # last task, writing the same bytes to the same address, is safer
                # than run-time control flow -- same handling as the vector version.
                pid = T.if_then_else(raw < total, raw, total - 1)
                n = pid % N
                hv = (pid // N) % HV
                bz = pid // (N * HV)
                t0 = n * C

                # Python `and` short-circuits: with RAGGED False the whole block
                # disappears at trace time.
                if RAGGED and n == N - 1:
                    T.tile.fill(x_io, 0)
                # Strided load: the token axis and the C axis sit either side of
                # HV, so the row pitch is HV * C.
                T.copy(Ltri[bz, t0 : t0 + C, hv, :], x_io)
                # Flatten to a contiguous block; the cube's GM->L1 only takes those.
                T.copy(x_io, ws_l[pid, 0, 0])
                T.set_cross_flag("MTE3", 0)

                T.wait_cross_flag(1)
                T.copy(ws_s3[pid, 0, 0], x_io)
                T.copy(x_io, A[bz, t0 : t0 + C, hv, :])

            with T.Scope("C"):
                T.wait_cross_flag(0)
                # I and -I are loaded once per block and shared by both tasks.
                T.copy(Idt, i_l1)
                T.copy(NegI, n_l1)

                # Grouped by dependency, not by the order of the formulas.
                #
                # Written out term by term the chain is
                # S0 -> P1 -> S1 -> P2 -> S2 -> P3 -> S3, and every term needs a
                # FIX->GM->MTE2 round trip after it: 14 per block.  Measurement
                # says the time goes *entirely* into those round trips rather than
                # into the matmuls -- the three STEPS = 1/2/3 points are almost
                # perfectly linear at 6.36 us per matmul, while every pipe sits
                # below 20% utilisation.
                #
                # But the dependencies are looser than that order suggests:
                # P1 = L @ L does not depend on S0, and P_{st+1} = P_st @ P_st
                # depends only on P_st.  So the terms regroup into rounds whose
                # inputs are all ready before the previous barrier:
                #
                #     group 0   S0 = I - L        P1 = L @ L     (needs only ws_l)
                #     barrier
                #     group 1   S1 = S0(I + P1)   P2 = P1 @ P1
                #     barrier
                #     group 2   S2 = S1(I + P2)   P3 = P2 @ P2
                #     barrier
                #     tail      S3 = S2(I + P3)
                #
                # That takes the barriers from 14 down to STEPS, and the two tasks
                # of a group (VEC_NUM = 2) issue back to back, giving the DMA two
                # independent streams to overlap.
                #
                # Every buffer choice below is a separate `if st == n:` rather than
                # a conditional expression or a list subscript: TVMScript's parser
                # rejects a ternary on buffers *at parse time*
                # (`ws_a if cond else ws_b` -> parse error).  st comes from range()
                # and is a plain Python int, so these ifs fold away during tracing
                # and the emitted IR matches a hand-written full unroll.
                #
                # Two accumulators, acc and acc2: S and P are computed in the same
                # group, and only separate L0C tiles let the two matmuls overlap
                # instead of being serialised by a write-after-read on one buffer.
                # AUTO_SYNC sees L0C hazards and inserts its own barriers; the only
                # hop it cannot see is the one through GM, which is why the
                # hand-written flag appears once per group and nowhere else.

                # ---- group 0: S0 and P1, both need only the ws_l that V staged ----
                for v in range(VEC_NUM):
                    raw_c = cid * VEC_NUM + v
                    pc = T.if_then_else(raw_c < total, raw_c, total - 1)
                    T.copy(ws_l[pc, 0, 0], l_l1)
                    T.copy(ws_l[pc, 0, 0], b_l1)
                    T.gemm_v0(i_l1, i_l1, acc, init=True)  # I
                    T.gemm_v0(l_l1, n_l1, acc, init=False)  # - L
                    T.gemm_v0(l_l1, b_l1, acc2, init=True)  # L @ L
                    T.copy(acc, ws_s0[pc, 0, 0])
                    T.copy(acc2, ws_p1[pc, 0, 0])
                # One flag is enough: FIX is an in-order pipe, so the last store
                # completing implies every earlier one has.
                T.set_flag("fix", "mte2", 0)
                T.wait_flag("fix", "mte2", 0)

                # ---- group st (st = 1 .. STEPS-1): S_st and P_{st+1} ----
                for st in range(1, STEPS):
                    for v in range(VEC_NUM):
                        raw_c = cid * VEC_NUM + v
                        pc = T.if_then_else(raw_c < total, raw_c, total - 1)
                        if st == 1:
                            T.copy(ws_s0[pc, 0, 0], s_l1)
                            T.copy(ws_p1[pc, 0, 0], a_l1)
                            T.copy(ws_p1[pc, 0, 0], b_l1)
                        if st == 2:
                            T.copy(ws_s1[pc, 0, 0], s_l1)
                            T.copy(ws_p2[pc, 0, 0], a_l1)
                            T.copy(ws_p2[pc, 0, 0], b_l1)
                        T.gemm_v0(s_l1, i_l1, acc, init=True)  # S @ I
                        T.gemm_v0(s_l1, a_l1, acc, init=False)  # + S @ P
                        T.gemm_v0(a_l1, b_l1, acc2, init=True)  # P @ P
                        if st == 1:
                            T.copy(acc, ws_s1[pc, 0, 0])
                            T.copy(acc2, ws_p2[pc, 0, 0])
                        if st == 2:
                            T.copy(acc, ws_s2[pc, 0, 0])
                            T.copy(acc2, ws_p3[pc, 0, 0])
                    T.set_flag("fix", "mte2", 1)
                    T.wait_flag("fix", "mte2", 1)

                # ---- tail: S_STEPS.  Only S here, no further P, and it always
                # lands in ws_s3 -- the V phase reads that one buffer whatever
                # STEPS is. ----
                for v in range(VEC_NUM):
                    raw_c = cid * VEC_NUM + v
                    pc = T.if_then_else(raw_c < total, raw_c, total - 1)
                    if STEPS == 1:
                        T.copy(ws_s0[pc, 0, 0], s_l1)
                        T.copy(ws_p1[pc, 0, 0], a_l1)
                    if STEPS == 2:
                        T.copy(ws_s1[pc, 0, 0], s_l1)
                        T.copy(ws_p2[pc, 0, 0], a_l1)
                    if STEPS == 3:
                        T.copy(ws_s2[pc, 0, 0], s_l1)
                        T.copy(ws_p3[pc, 0, 0], a_l1)
                    T.gemm_v0(s_l1, i_l1, acc, init=True)
                    T.gemm_v0(s_l1, a_l1, acc, init=False)
                    T.copy(acc, ws_s3[pc, 0, 0])

                T.set_cross_flag("FIX", 1)

    return main


@tilelang.jit(
    out_idx=[-1],
    workspace_idx=[-9, -8, -7, -6, -5, -4, -3, -2],
    pass_configs=pass_configs,
)
def kda_solve_tril_cube_ker_varlen(B, SEQ, HV, C, NT_TOTAL, dtype="float16", accum_dtype="float"):
    """A = (I + L)^{-1} per chunk on the cube, varlen.  Twin of the builder above.

    The cube phase is *word for word identical* -- inverting a triangle does not
    care where the block came from.  Only the V phase changes: which rows belong
    to this block has to be read from Meta rather than computed as n * C.

    Doing varlen too is not optional.  The acceptance criterion for varlen is
    *exactly 0*: a batched varlen run must be bit-identical to running each
    sequence on its own.  Single runs take the fixed-length path, which is now on
    the cube; leaving varlen on the vector version makes the two paths diverge --
    measured |dO| = 1.5e-05, and the criterion goes red immediately.

    It is a separate @tilelang.jit builder rather than one builder with a flag
    because a negative out_idx is normalised in place into the decorator's own
    list, so two prim_funcs of different arity under one decorator corrupt each
    other's output indices.  Same reason as
    kda_solve_tril.kda_solve_tril_ker_varlen.
    """
    total = HV * NT_TOTAL  # B is always 1 under varlen
    blocks = (total + VEC_NUM - 1) // VEC_NUM

    @T.prim_func
    def main(
        Ltri: T.Tensor([B, SEQ, HV, C], dtype),  # type: ignore
        Idt: T.Tensor([C, C], dtype),  # type: ignore
        NegI: T.Tensor([C, C], dtype),  # type: ignore
        Meta: T.Tensor([NT_TOTAL, _VL.META_COLS], "int32"),  # type: ignore
        ws_l: T.Tensor([total, C, C], dtype),  # type: ignore
        ws_s0: T.Tensor([total, C, C], dtype),  # type: ignore
        ws_s1: T.Tensor([total, C, C], dtype),  # type: ignore
        ws_s2: T.Tensor([total, C, C], dtype),  # type: ignore
        ws_s3: T.Tensor([total, C, C], dtype),  # type: ignore
        ws_p1: T.Tensor([total, C, C], dtype),  # type: ignore
        ws_p2: T.Tensor([total, C, C], dtype),  # type: ignore
        ws_p3: T.Tensor([total, C, C], dtype),  # type: ignore
        A: T.Tensor([B, SEQ, HV, C], dtype),  # type: ignore
    ):
        with T.Kernel(blocks, is_npu=True) as (cid, vid):
            l_l1 = T.alloc_L1([C, C], dtype)
            a_l1 = T.alloc_L1([C, C], dtype)
            b_l1 = T.alloc_L1([C, C], dtype)
            s_l1 = T.alloc_L1([C, C], dtype)
            i_l1 = T.alloc_L1([C, C], dtype)
            n_l1 = T.alloc_L1([C, C], dtype)
            acc = T.alloc_L0C([C, C], accum_dtype)  # the S chain
            acc2 = T.alloc_L0C([C, C], accum_dtype)  # the P chain, concurrent with S
            x_io = T.alloc_ub([C, C], dtype)

            with T.Scope("V"):
                raw = cid * VEC_NUM + vid
                pid = T.if_then_else(raw < total, raw, total - 1)
                ic = pid % NT_TOTAL
                hv = pid // NT_TOTAL
                t0 = Meta[ic, _VL.META_T0]
                rows = Meta[ic, _VL.META_ROWS]

                # Unconditional fill.  Under varlen "is this the ragged chunk" is
                # a run-time property: at most N chunks are ragged and they land on
                # arbitrary flat indices, so a compile-time predicate would fire on
                # the wrong block.  After the fill, rows rows..C-1 are true zeros,
                # which is what keeps 0 * a neighbour's stale value out of nan.
                T.tile.fill(x_io, 0)
                T.copy(Ltri[0, t0 : t0 + rows, hv, :], x_io)
                T.copy(x_io, ws_l[pid, 0, 0])
                T.set_cross_flag("MTE3", 0)

                T.wait_cross_flag(1)
                T.copy(ws_s3[pid, 0, 0], x_io)
                # Bounded store: rows past this chunk's count belong to the next
                # sequence and are written by that sequence's own block, so they
                # stay in x_io and never reach GM.
                T.copy(x_io, A[0, t0 : t0 + rows, hv, :])

            with T.Scope("C"):
                T.wait_cross_flag(0)
                # I and -I are loaded once per block and shared by both tasks.
                T.copy(Idt, i_l1)
                T.copy(NegI, n_l1)

                # Grouped by dependency, not by the order of the formulas.
                #
                # Written out term by term the chain is
                # S0 -> P1 -> S1 -> P2 -> S2 -> P3 -> S3, and every term needs a
                # FIX->GM->MTE2 round trip after it: 14 per block.  Measurement
                # says the time goes *entirely* into those round trips rather than
                # into the matmuls -- the three STEPS = 1/2/3 points are almost
                # perfectly linear at 6.36 us per matmul, while every pipe sits
                # below 20% utilisation.
                #
                # But the dependencies are looser than that order suggests:
                # P1 = L @ L does not depend on S0, and P_{st+1} = P_st @ P_st
                # depends only on P_st.  So the terms regroup into rounds whose
                # inputs are all ready before the previous barrier:
                #
                #     group 0   S0 = I - L        P1 = L @ L     (needs only ws_l)
                #     barrier
                #     group 1   S1 = S0(I + P1)   P2 = P1 @ P1
                #     barrier
                #     group 2   S2 = S1(I + P2)   P3 = P2 @ P2
                #     barrier
                #     tail      S3 = S2(I + P3)
                #
                # That takes the barriers from 14 down to STEPS, and the two tasks
                # of a group (VEC_NUM = 2) issue back to back, giving the DMA two
                # independent streams to overlap.
                #
                # Every buffer choice below is a separate `if st == n:` rather than
                # a conditional expression or a list subscript: TVMScript's parser
                # rejects a ternary on buffers *at parse time*
                # (`ws_a if cond else ws_b` -> parse error).  st comes from range()
                # and is a plain Python int, so these ifs fold away during tracing
                # and the emitted IR matches a hand-written full unroll.
                #
                # Two accumulators, acc and acc2: S and P are computed in the same
                # group, and only separate L0C tiles let the two matmuls overlap
                # instead of being serialised by a write-after-read on one buffer.
                # AUTO_SYNC sees L0C hazards and inserts its own barriers; the only
                # hop it cannot see is the one through GM, which is why the
                # hand-written flag appears once per group and nowhere else.

                # ---- group 0: S0 and P1, both need only the ws_l that V staged ----
                for v in range(VEC_NUM):
                    raw_c = cid * VEC_NUM + v
                    pc = T.if_then_else(raw_c < total, raw_c, total - 1)
                    T.copy(ws_l[pc, 0, 0], l_l1)
                    T.copy(ws_l[pc, 0, 0], b_l1)
                    T.gemm_v0(i_l1, i_l1, acc, init=True)  # I
                    T.gemm_v0(l_l1, n_l1, acc, init=False)  # - L
                    T.gemm_v0(l_l1, b_l1, acc2, init=True)  # L @ L
                    T.copy(acc, ws_s0[pc, 0, 0])
                    T.copy(acc2, ws_p1[pc, 0, 0])
                # One flag is enough: FIX is an in-order pipe, so the last store
                # completing implies every earlier one has.
                T.set_flag("fix", "mte2", 0)
                T.wait_flag("fix", "mte2", 0)

                # ---- group st (st = 1 .. STEPS-1): S_st and P_{st+1} ----
                for st in range(1, STEPS):
                    for v in range(VEC_NUM):
                        raw_c = cid * VEC_NUM + v
                        pc = T.if_then_else(raw_c < total, raw_c, total - 1)
                        if st == 1:
                            T.copy(ws_s0[pc, 0, 0], s_l1)
                            T.copy(ws_p1[pc, 0, 0], a_l1)
                            T.copy(ws_p1[pc, 0, 0], b_l1)
                        if st == 2:
                            T.copy(ws_s1[pc, 0, 0], s_l1)
                            T.copy(ws_p2[pc, 0, 0], a_l1)
                            T.copy(ws_p2[pc, 0, 0], b_l1)
                        T.gemm_v0(s_l1, i_l1, acc, init=True)  # S @ I
                        T.gemm_v0(s_l1, a_l1, acc, init=False)  # + S @ P
                        T.gemm_v0(a_l1, b_l1, acc2, init=True)  # P @ P
                        if st == 1:
                            T.copy(acc, ws_s1[pc, 0, 0])
                            T.copy(acc2, ws_p2[pc, 0, 0])
                        if st == 2:
                            T.copy(acc, ws_s2[pc, 0, 0])
                            T.copy(acc2, ws_p3[pc, 0, 0])
                    T.set_flag("fix", "mte2", 1)
                    T.wait_flag("fix", "mte2", 1)

                # ---- tail: S_STEPS.  Only S here, no further P, and it always
                # lands in ws_s3 -- the V phase reads that one buffer whatever
                # STEPS is. ----
                for v in range(VEC_NUM):
                    raw_c = cid * VEC_NUM + v
                    pc = T.if_then_else(raw_c < total, raw_c, total - 1)
                    if STEPS == 1:
                        T.copy(ws_s0[pc, 0, 0], s_l1)
                        T.copy(ws_p1[pc, 0, 0], a_l1)
                    if STEPS == 2:
                        T.copy(ws_s1[pc, 0, 0], s_l1)
                        T.copy(ws_p2[pc, 0, 0], a_l1)
                    if STEPS == 3:
                        T.copy(ws_s2[pc, 0, 0], s_l1)
                        T.copy(ws_p3[pc, 0, 0], a_l1)
                    T.gemm_v0(s_l1, i_l1, acc, init=True)
                    T.gemm_v0(s_l1, a_l1, acc, init=False)
                    T.copy(acc, ws_s3[pc, 0, 0])

                T.set_cross_flag("FIX", 1)

    return main


# ---------------------------------------------------------------------- host

_DTYPE = _VEC._DTYPE
_CONST_CACHE = {}


def _const_pair(C, device, dtype):
    """[C, C] identity and its negation, in the data's dtype (a cube operand
    cannot be fp32)."""
    key = (C, str(device), str(dtype))
    got = _CONST_CACHE.get(key)
    if got is None:
        eye = torch.eye(C, device=device, dtype=dtype)
        got = (eye, -eye)
        _CONST_CACHE[key] = got
    return got


def kda_solve_tril_cube(Ltri, cu_seqlens=None):
    """A = (I + L)^{-1} per chunk, with both the fixed-length and the varlen path
    on the cube.

    Doing varlen at the same time is not optional: the acceptance criterion is
    that a batched varlen run is *bit-identical* to running each sequence on its
    own (exactly 0).  Single runs take the fixed-length path, which is on the
    cube; leaving varlen on the vector version splits the two immediately --
    measured |dO| = 1.5e-05.
    """
    assert Ltri.dim() == 4, f"expected [B, SEQ, HV, C], got {tuple(Ltri.shape)}"
    B, SEQ, HV, C = Ltri.shape
    assert Ltri.dtype in _DTYPE, f"unsupported dtype {Ltri.dtype}"
    assert Ltri.is_contiguous(), "Ltri must be contiguous in [B, SEQ, HV, C]"
    assert C % 16 == 0, f"C must be a multiple of 16 to keep rows 32B aligned, got {C}"
    # The cube's fractal granularity is 16: a C below that cannot fill even one.
    assert C >= 16, f"C={C} is below the Cube fractal granularity"
    # A cube operand can only be fp16 / bf16.  Passing fp32 does not raise -- it
    # silently produces nan, which is exactly how the first version failed: the
    # test harness took L from stage_tensors as fp32, the kernel was instantiated
    # as fp32, and all nine matmuls came back nan.  In the pipeline L is produced
    # by kkt in k.dtype and is already fp16 / bf16, so this asserts rather than
    # quietly converting.
    assert Ltri.dtype in (torch.float16, torch.bfloat16), f"Cube operands must be fp16/bf16, got {Ltri.dtype}; use kda_solve_tril for fp32"

    if SEQ == 0:
        return torch.empty((B, 0, HV, C), device=Ltri.device, dtype=Ltri.dtype)

    idt, negi = _const_pair(C, Ltri.device, Ltri.dtype)
    if cu_seqlens is None:
        ker = kda_solve_tril_cube_ker(B, SEQ, HV, C, dtype=_DTYPE[Ltri.dtype])
        return ker(Ltri, idt, negi)

    bounds = _VL.varlen_bounds(cu_seqlens, q=Ltri)
    meta = _VL.chunk_meta(bounds, C, Ltri.device)
    nt_total = meta.shape[0]
    # Every token row must be written by exactly one block, or the caller gets
    # dirty memory back: the output buffer is torch.empty, not zeros.
    assert nt_total > 0, "a non-empty batch must produce at least one chunk"
    assert int(meta[:, _VL.META_ROWS].sum()) == SEQ, "chunk metadata does not cover every token exactly once"
    ker = kda_solve_tril_cube_ker_varlen(B, SEQ, HV, C, nt_total, dtype=_DTYPE[Ltri.dtype])
    return ker(Ltri, idt, negi, meta)


# ------------------------------------------------------------------- testing

_TAG = _VEC._TAG
_TOL = _VEC._TOL


def _relerr(got, ref):
    got, ref = got.float(), ref.float()
    return (got - ref).abs().max().item() / max(ref.abs().max().item(), 1e-9)


def _case(B, SEQ, H, HV, K, V, C, gate, dtype):
    """One shape, against the stage-3 golden from the L1 reference pipeline."""
    import kda_ref as L0

    q, k, v, g, beta, _ = L0.make_inputs(B, SEQ, H, HV, K, V, device="npu", dtype=dtype, gate=gate, seed=0)
    st = kda_chunk_ref.stage_tensors(q, k, v, g, beta, C=C)
    # stage_tensors returns L in fp32 (the reference layer accumulates in fp32
    # throughout), but in the pipeline stage 2 emits k.dtype, so stage 3 receives
    # fp16 / bf16 -- which is also all a cube operand can be.  Cast once here so
    # both implementations are fed the same input the pipeline would give them.
    Ltri, ref = st["L"].to(dtype).contiguous(), st["A"]

    got = kda_solve_tril_cube(Ltri)
    torch.npu.synchronize()
    e_cube = _relerr(got, ref)

    # Cross-check against the vector version on the same fp16 input: both are
    # measured against the golden and should land in the same ballpark.
    # kda_solve_tril now forwards here by default, so the switch has to be held
    # down explicitly -- otherwise the "control" would be the same code as the
    # subject and the comparison would be vacuous.
    _prev = os.environ.get("KDA_SOLVE_CUBE")
    os.environ["KDA_SOLVE_CUBE"] = "0"
    try:
        got_v = _VEC.kda_solve_tril(Ltri)
    finally:
        if _prev is None:
            del os.environ["KDA_SOLVE_CUBE"]
        else:
            os.environ["KDA_SOLVE_CUBE"] = _prev
    torch.npu.synchronize()
    e_vec = _relerr(got_v, ref)

    tol = _TOL[dtype]
    ok = e_cube < tol
    print(
        "  B=%d SEQ=%-5d HV=%d C=%-3d %-8s %-5s  cube=%.3e  vec=%.3e  %s"
        % (B, SEQ, HV, C, gate, _TAG[dtype], e_cube, e_vec, "OK" if ok else "FAIL")
    )
    return ok


def main():
    import torch_npu  # noqa: F401

    torch.npu.set_device(0)
    print(
        "solve_tril on the cube: doubling Neumann, %d steps, %d matmuls per "
        "chunk, %d unrolled in the cube scope" % (STEPS, 2 + 3 * STEPS, VEC_NUM * (2 + 3 * STEPS))
    )
    print("  %-38s %-22s" % ("", "relerr vs golden"))
    ok = True
    # For bisecting a hang: one shape only, so a failure does not cost ten compiles
    if "--quick" in sys.argv:
        ok = _case(1, 512, 4, 4, 128, 128, 64, "normal", torch.float16)
        print()
        if not ok:
            raise SystemExit(1)
        print("Kernel Output Match!")
        return
    # Four gates: normal is the baseline, forget/keep are the two edges, extreme
    # is the worst case
    for gate in ("normal", "forget", "keep", "extreme"):
        ok &= _case(1, 512, 4, 4, 128, 128, 64, gate, torch.float16)
    # the shape the profile uses
    ok &= _case(1, 4096, 4, 4, 128, 128, 64, "normal", torch.float16)
    # small C, more heads, more batches
    ok &= _case(2, 256, 4, 4, 128, 128, 32, "normal", torch.float16)
    ok &= _case(1, 128, 8, 8, 128, 128, 64, "normal", torch.float16)
    # ragged tail: SEQ % C != 0
    ok &= _case(1, 200, 4, 4, 128, 128, 64, "normal", torch.float16)
    ok &= _case(1, 300, 4, 4, 128, 128, 32, "normal", torch.float16)
    # bf16
    ok &= _case(1, 512, 4, 4, 128, 128, 64, "normal", torch.bfloat16)

    print()
    if ok:
        print("Kernel Output Match!")
    else:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
