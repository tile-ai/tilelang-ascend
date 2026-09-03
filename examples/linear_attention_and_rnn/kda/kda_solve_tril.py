"""KDA L1 stage 3: A = (I + L)^{-1}, L strictly lower triangular, per chunk.

This is the one stage whose *math* needs no porting at all.  The gate never
enters the formula -- L already carries every decay factor, baked in by stage 2
-- so the recurrence below is the GDN kernel body unchanged.  What changed is
the interface: the tensor layout, the head axis, and the grid decomposition.

Algorithm (row-by-row forward substitution, on the vector core)
--------------------------------------------------------------
    let A = I + X;  (I + L)(I + X) = I  =>  X = -L - L X
    row i:          X_i = -L_i - L_i X_{0..i-1}

The kernel keeps the GDN sign convention: it holds X' = -X in ``x_ub``,
recurses ``x_ub[i] <- L[i] - sum_j x_ub[j] * x_ub[i, j]``, and finishes with
A = I - X'.  Substituting X' = -X reproduces ``kda_chunk_ref.ref_solve_tril``
line for line, so the two are the same computation, not merely equivalent ones.

Forward substitution, not ``linalg.inv``: there is no NPU implementation of
inv, so it silently falls back to CPU and then NaN-poisons the whole matrix.
Rows 0 and 1 need no update -- row 0 is empty and row 1 has only the already
final row 0 to its left -- which is why the serial loop starts at 2.

Interface (frozen FLA contract, external layout)
------------------------------------------------
    Ltri  [B, SEQ, HV, C]   dtype   stage 2 output, == stage_tensors()["L"]
    Idt   [C, C]            fp32    identity, a compile-time constant
    A     [B, SEQ, HV, C]   dtype   stage 3 output, == stage_tensors()["A"]

    dtype is taken from the input tensor (fp16 / bf16 / fp32), never hardcoded.
    Only HV and C matter here: H, K, V, GRP and scale do not enter this stage,
    because both tensors are indexed by the value head hv only.  There is no
    Q/Kt read, hence no GVA remapping.

The strided tile
----------------
The token axis and the C axis sit on either side of HV, so one [C, C] chunk
matrix is a *strided* transfer: C contiguous elements per row, HV * C elements
between rows.  Slicing the token axis while pinning hv expresses exactly that:

    T.copy(Ltri[bz, t0 : t0 + C, hv, :], x_io)

The backend turns the resulting region extents [1, C, 1, C] into a row stride
of HV * C (every trailing unit extent folds the matching shape dim into the
stride), which is the number we want.  Rows are C * itemsize bytes, always a
multiple of 32 for C % 16 == 0, so both directions stay 32B aligned.

Assumption inherited from stage 2: L is *exactly* zero on and above the
diagonal.  The recurrence sums over all C rows rather than just j < i, and the
terms with j >= i vanish only because x_ub[i, j] is exactly 0 there.  The
reference masks explicitly, and the mask is folded into the exponent in the kkt
kernel, so the zeros are true zeros rather than underflowed ones.

Ragged tail
-----------
``SEQ % C != 0`` is supported.  The grid uses ``ceildiv(SEQ, C)`` chunks and the
load and store of the last chunk matrix are clamped to its R valid rows by
``compute_valid_extent`` (src/op/ascend.cc:410).  The tile stays [C, C], so
``C % 16 == 0`` and the 32B row alignment are untouched.

The math needs no change: (I + L) for a ragged chunk is block lower triangular
with the pad block absent, so forward substitution over the leading R rows is
exactly the length-R answer.  One thing *is* needed though, and it is not
obvious -- see the fill in the kernel.  The recurrence sums over every j, so a
valid row i < R picks up terms ``x[j,k] * x[i,j]`` with j >= R.  The second
factor is 0 by strict lower triangularity, but 0 * inf is NaN, so the tail rows
have to be genuine zeros rather than leftover UB.

UB budget (limit 196352 B)
--------------------------
    x_ub, i_ub, mul_ub  3 * C * C * 4        C=64 -> 49152 B, C=32 -> 12288 B
    red_ub              C * 4                C=64 ->   256 B, C=32 ->   128 B
    x_io                C * C * itemsize     C=64 -> 8192 B (fp16) / 16384 (fp32)
    total, worst case C=64 fp32                        65792 B
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

# Only AUTO_SYNC, same as the six GDN kernels.  MEMORY_PLANNING is deliberately
# left off: on the backward dot it aliased a reduction target with a temporary
# tile, and the stores wrote zeros while the registers held the right values.
pass_configs = {tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True}

VEC_NUM = 2


@tilelang.jit(out_idx=[-1], pass_configs=pass_configs)
def kda_solve_tril_ker(B, SEQ, HV, C, dtype="float16", accum_dtype="float"):
    # ceil, not floor: the last chunk may be ragged.  SEQ is a Python int at
    # trace time, so this stays a compile-time constant.
    N = -(-SEQ // C)  # chunks per sequence
    R = SEQ % C  # 0 when aligned; else the valid row count of the last chunk
    RAGGED = R != 0
    total = B * HV * N  # one [C, C] chunk matrix per task
    blocks = (total + VEC_NUM - 1) // VEC_NUM

    @T.prim_func
    def main(
        Ltri: T.Tensor([B, SEQ, HV, C], dtype),  # type: ignore
        Idt: T.Tensor([C, C], accum_dtype),  # type: ignore
        A: T.Tensor([B, SEQ, HV, C], dtype),  # type: ignore
    ):
        # The row recurrence is serial, so the two vector cores take two whole
        # chunk matrices rather than splitting one.  The chunk index is the
        # fastest-moving part of the task id, so for N > 1 the two cores of a
        # block sit in the same (batch, head) and read neighbouring GM.
        with T.Kernel(blocks, is_npu=True) as (cid, vid):
            raw = cid * VEC_NUM + vid
            # An odd task count leaves one core over.  Clamping is safe where
            # skipping would need runtime control flow: the surplus core redoes
            # the last task and stores byte-identical values to the same address,
            # so no interleaving of the two stores can produce a wrong result.
            pid = T.if_then_else(raw < total, raw, total - 1)
            n = pid % N
            hv = (pid // N) % HV
            bz = pid // (N * HV)
            t0 = n * C

            x_ub = T.alloc_ub([C, C], accum_dtype)  # L, then I - A, then A
            i_ub = T.alloc_ub([C, C], accum_dtype)  # identity
            mul_ub = T.alloc_ub([C, C], accum_dtype)  # reduction operand
            red_ub = T.alloc_ub([C], accum_dtype)  # C * 4 >= 32 B
            row_ub = T.alloc_ub([C], accum_dtype)  # row i of x_ub, lifted out
            row_b = T.alloc_ub([C, C], accum_dtype)  # that row, broadcast across columns
            x_io = T.alloc_ub([C, C], dtype)  # GM staging, dtype side

            with T.Scope("V"):
                # Python `and` short-circuits: with RAGGED False the whole block is
                # dropped at trace time, so the aligned path emits identical IR.
                # The load below is clamped to R rows on the last chunk, so
                # rows R..C-1 keep whatever UB held.  Those rows are NOT
                # inert here: the reduction
                #     red[k] = sum_j x[j,k] * x[i,j]
                # runs over every j, so for a *valid* row i < R the j >= R
                # terms contribute x[j,k] * x[i,j].  The second factor is 0
                # (L is strictly lower triangular and j >= R > i), but
                # inf * 0 is NaN -- one non-finite tail row would poison
                # every valid row.  Zeroing makes those terms exactly 0.
                if RAGGED and n == N - 1:
                    T.tile.fill(x_io, 0)

                # Strided load of one chunk matrix; see the module docstring.
                # On the tail chunk this transfers R rows, not C.
                T.copy(Ltri[bz, t0 : t0 + C, hv, :], x_io)
                T.copy(x_io, x_ub)
                T.copy(Idt[0, 0], i_ub)

                for i in range(2, C):
                    # mul_ub is fully overwritten below, and T.reduce_sum
                    # defaults to clear=True and initialises red_ub itself, so
                    # neither buffer gets a manual fill.
                    #
                    # ★ The factor x_ub[i, j] is indexed by the OUTER T.Parallel
                    # variable, which is the worst broadcast form in this dialect:
                    # one narrow vector op per row, each preceded by a
                    # PipeBarrier<PIPE_ALL> and a GetValue.  At C = 64 that is
                    # (C-2) * C = 3968 scalar reads per block, and it is why this
                    # stage's scalar pipe (55.0% of core active time) does MORE
                    # work than its vector pipe (45.7%) -- worse in ratio terms
                    # than kkt was.
                    #
                    # Lift row i out and broadcast it across the columns, so the
                    # multiply becomes tile-vs-tile.  The row index is a run-time
                    # value (range() lowers to T.serial), and an older note in
                    # this codebase warns that a UB->UB copy at a run-time row
                    # offset is unreliable -- so this was measured before it was
                    # written: PERF/probes/probe_a.py shows GetValue 1->0,
                    # PipeBarrier<PIPE_ALL> 1->0, op widths [64] -> [64, 4096],
                    # and the result bit-identical to the old form (max|diff| =
                    # 0.0, both 1.49e-08 against a CPU forward substitution).
                    T.copy(x_ub[i, 0:C], row_ub)
                    T.tile.broadcast(row_b, row_ub, axis=1)
                    for j, k in T.Parallel(C, C):
                        mul_ub[j, k] = x_ub[j, k] * row_b[j, k]
                    T.reduce_sum(mul_ub, red_ub, dim=0)
                    for j in T.Parallel(C):
                        x_ub[i, j] = x_ub[i, j] - red_ub[j]

                for i, j in T.Parallel(C, C):
                    x_ub[i, j] = i_ub[i, j] - x_ub[i, j]

                T.copy(x_ub, x_io)
                # Clamped on the tail chunk: rows R..C-1 are never written, so
                # the identity rows the loop leaves there stay out of GM.
                T.copy(x_io, A[bz, t0 : t0 + C, hv, :])

    return main


@tilelang.jit(out_idx=[-1], pass_configs=pass_configs)
def kda_solve_tril_ker_varlen(B, SEQ, HV, C, NT_TOTAL, dtype="float16", accum_dtype="float"):
    """A = (I + L)^{-1} per chunk, varlen.  Twin of kda_solve_tril_ker above.

    Stage 3 is the stage varlen changes least: it inverts one [C, C] matrix per
    chunk and never looks at a neighbouring chunk, so nothing about the algebra
    moves.  What changes is only which rows of GM a block owns, and that has to
    become explicit, because the row block that a chunk does NOT own is no
    longer past the end of the tensor -- it belongs to the next sequence.

    That turns what was a harmless over-read into two real defects:

      * The load would pull the next sequence's L rows into x_io.  Those rows
        are written by a DIFFERENT block with no ordering between them, so the
        value read is whatever the schedule happens to produce.  They are not
        inert either: the reduction  red[k] = sum_j x[j,k] * x[i,j]  runs over
        every j, so a foreign row j >= rows contributes to a valid row i.  It
        survives today only because L is strictly lower triangular and x[i,j]
        is 0 for j > i -- but 0 * inf is NaN, and a foreign row can be anything.
      * The store would write C rows, of which up to C-1 land on the next
        sequence's A.  Those rows hold `I - X'` built from a mixture of two
        sequences, so the corruption is finite, plausible and silent.

    Both are fixed by the same thing: take the row extent from the metadata
    instead of leaving it at C.  The bounded load then also gap-fills the unused
    rows with zeros for free (PROBES/probe_varlen3.log), which is exactly the
    invariant the reduction needs.

    See kda_chunk_cumsum.cumsum_ker_varlen's docstring for why this is a
    separate @tilelang.jit builder rather than a flag on the existing one -- the
    short version is that a negative out_idx is normalised in place into the
    decorator's own list, so two prim_funcs of different arity under one
    decorator corrupt each other's output index.
    """
    total = HV * NT_TOTAL  # one [C, C] chunk matrix per task; B is 1 under varlen
    blocks = (total + VEC_NUM - 1) // VEC_NUM

    @T.prim_func
    def main(
        Ltri: T.Tensor([B, SEQ, HV, C], dtype),  # type: ignore
        Idt: T.Tensor([C, C], accum_dtype),  # type: ignore
        Meta: T.Tensor([NT_TOTAL, _VL.META_COLS], "int32"),  # type: ignore
        A: T.Tensor([B, SEQ, HV, C], dtype),  # type: ignore
    ):
        # The row recurrence is serial, so the two vector cores take two whole
        # chunk matrices rather than splitting one.  The chunk index is the
        # fastest-moving part of the task id, so the two cores of a block sit in
        # the same head and read neighbouring GM.
        with T.Kernel(blocks, is_npu=True) as (cid, vid):
            raw = cid * VEC_NUM + vid
            # An odd task count leaves one core over.  Clamping is safe where
            # skipping would need runtime control flow: the surplus core redoes
            # the last task and stores byte-identical values to the same address,
            # so no interleaving of the two stores can produce a wrong result.
            # Still true under varlen -- same task id means the same Meta row,
            # hence the same t0, the same rows and the same destination.
            pid = T.if_then_else(raw < total, raw, total - 1)
            ic = pid % NT_TOTAL
            hv = pid // NT_TOTAL

            t0 = Meta[ic, _VL.META_T0]
            rows = Meta[ic, _VL.META_ROWS]

            x_ub = T.alloc_ub([C, C], accum_dtype)  # L, then I - A, then A
            i_ub = T.alloc_ub([C, C], accum_dtype)  # identity
            mul_ub = T.alloc_ub([C, C], accum_dtype)  # reduction operand
            red_ub = T.alloc_ub([C], accum_dtype)  # C * 4 >= 32 B
            row_ub = T.alloc_ub([C], accum_dtype)  # row i of x_ub, lifted out
            row_b = T.alloc_ub([C, C], accum_dtype)  # that row, broadcast across columns
            x_io = T.alloc_ub([C, C], dtype)  # GM staging, dtype side

            with T.Scope("V"):
                # Unconditional.  Raggedness is a run-time property under varlen
                # -- up to N chunks are ragged and they sit at arbitrary flat
                # indices -- so a compile-time `n == N - 1` predicate would fire
                # on the wrong block and skip the fill on every block that
                # actually needs it.
                T.tile.fill(x_io, 0)

                # Strided load of one chunk matrix, bounded to the rows this
                # chunk owns.  Rows rows..C-1 come back as zeros, so the
                # reduction's j >= rows terms are exactly 0 rather than
                # 0 * (whatever the neighbour left there).
                T.copy(Ltri[0, t0 : t0 + rows, hv, :], x_io)
                T.copy(x_io, x_ub)
                T.copy(Idt[0, 0], i_ub)

                for i in range(2, C):
                    # mul_ub is fully overwritten below, and T.reduce_sum
                    # defaults to clear=True and initialises red_ub itself, so
                    # neither buffer gets a manual fill.
                    #
                    # ★ The factor x_ub[i, j] is indexed by the OUTER T.Parallel
                    # variable, which is the worst broadcast form in this dialect:
                    # one narrow vector op per row, each preceded by a
                    # PipeBarrier<PIPE_ALL> and a GetValue.  At C = 64 that is
                    # (C-2) * C = 3968 scalar reads per block, and it is why this
                    # stage's scalar pipe (55.0% of core active time) does MORE
                    # work than its vector pipe (45.7%) -- worse in ratio terms
                    # than kkt was.
                    #
                    # Lift row i out and broadcast it across the columns, so the
                    # multiply becomes tile-vs-tile.  The row index is a run-time
                    # value (range() lowers to T.serial), and an older note in
                    # this codebase warns that a UB->UB copy at a run-time row
                    # offset is unreliable -- so this was measured before it was
                    # written: PERF/probes/probe_a.py shows GetValue 1->0,
                    # PipeBarrier<PIPE_ALL> 1->0, op widths [64] -> [64, 4096],
                    # and the result bit-identical to the old form (max|diff| =
                    # 0.0, both 1.49e-08 against a CPU forward substitution).
                    T.copy(x_ub[i, 0:C], row_ub)
                    T.tile.broadcast(row_b, row_ub, axis=1)
                    for j, k in T.Parallel(C, C):
                        mul_ub[j, k] = x_ub[j, k] * row_b[j, k]
                    T.reduce_sum(mul_ub, red_ub, dim=0)
                    for j in T.Parallel(C):
                        x_ub[i, j] = x_ub[i, j] - red_ub[j]

                for i, j in T.Parallel(C, C):
                    x_ub[i, j] = i_ub[i, j] - x_ub[i, j]

                T.copy(x_ub, x_io)
                # Bounded store: rows past this chunk's count belong to the next
                # sequence and are written by that sequence's own block, so the
                # identity rows the loop leaves in x_ub stay out of GM.
                T.copy(x_io, A[0, t0 : t0 + rows, hv, :])

    return main


# --------------------------------------------------------------- host wrapper
_DTYPE = {
    torch.float16: "float16",
    torch.bfloat16: "bfloat16",
    torch.float32: "float",
}

_IDT_CACHE = {}


def _identity(C, device):
    """Cached [C, C] fp32 identity.

    A compile-time constant, not a transform of the data: at most 64x64 fp32 =
    16 KB, the same trick the GDN kernels use for their causal masks.  Cached so
    a pipeline that calls this per chunk does not reallocate it every time.
    """
    key = (C, str(device))
    idt = _IDT_CACHE.get(key)
    if idt is None:
        idt = torch.eye(C, device=device, dtype=torch.float)
        _IDT_CACHE[key] = idt
    return idt


def kda_solve_tril(Ltri, cu_seqlens=None):
    """A = (I + L)^{-1} per chunk, both in external [B, SEQ, HV, C] layout.

    C is the trailing dimension of the input -- stage 2 emits one C-wide row per
    token -- so there is no separate chunk-length argument to keep in sync.

    With ``cu_seqlens`` the input is a flattened varlen batch (B == 1).  The
    layout is unchanged: chunk matrices are still one C-wide row per token, laid
    out in flattened token order.  Only the block-to-chunk mapping moves.
    """
    assert Ltri.dim() == 4, f"expected [B, SEQ, HV, C], got {tuple(Ltri.shape)}"
    B, SEQ, HV, C = Ltri.shape
    assert Ltri.dtype in _DTYPE, f"unsupported dtype {Ltri.dtype}"
    # The kernel derives its GM strides from the declared shape, so a transposed
    # or sliced view would be read with the wrong row stride.  Fail loudly.
    assert Ltri.is_contiguous(), "Ltri must be contiguous in [B, SEQ, HV, C]"
    assert C % 16 == 0, f"C must be a multiple of 16 to keep rows 32B aligned, got {C}"
    # 3 fp32 [C, C] tiles plus the dtype staging tile must fit in 196352 B.
    # 4 fp32 [C, C] tiles (x_ub, i_ub, mul_ub, row_b) + the dtype staging tile.
    assert C <= 64, f"C={C} needs {4 * C * C * 4 + C * C * 4} B of UB, over budget"

    # SEQ == 0 slips past the assert above (0 % C == 0) and would launch a
    # zero-block grid over an unwritten output.  A zero-length sequence is legal
    # input; there is no block to invert, so A is empty along the token axis.
    #
    # Under varlen this fires only for a wholly empty batch.  A single empty
    # sequence inside a non-empty one needs nothing: it contributes zero chunks,
    # so no block is created for it.
    if SEQ == 0:
        return torch.empty((B, 0, HV, C), device=Ltri.device, dtype=Ltri.dtype)

    # The cube version replaces these 62 serial forward-substitution rows with a
    # doubling Neumann series of 8 matmuls.  Measured in the pipeline at
    # 165.76 -> 69.98 us, with ten shapes (including ragged tails, C=32 and bf16)
    # landing in the same error band as this implementation and eight of them
    # bit-identical -- see kda_solve_tril_cube.py.
    #
    # Both the fixed-length and the varlen path go to the cube.  Moving only one
    # would immediately break the varlen acceptance criterion, which requires a
    # batched run to be bit-identical (exactly 0) to running each sequence alone.
    #
    # Two conditions keep a call on the vector version, both from the cube itself:
    # its operands can only be fp16 / bf16, and its fractal granularity is 16.
    #   * fp32 input
    #   * C < 16
    # KDA_SOLVE_CUBE=0 forces the vector version, for A/B comparisons.
    if os.environ.get("KDA_SOLVE_CUBE", "1") != "0" and Ltri.dtype in (torch.float16, torch.bfloat16) and C >= 16:
        # Lazy import: that module imports this one back (it reuses _DTYPE and
        # _TOL), so a top-level import would be circular.
        from kda_solve_tril_cube import kda_solve_tril_cube

        return kda_solve_tril_cube(Ltri, cu_seqlens=cu_seqlens)

    if cu_seqlens is None:
        ker = kda_solve_tril_ker(B, SEQ, HV, C, dtype=_DTYPE[Ltri.dtype])
        return ker(Ltri, _identity(C, Ltri.device))

    bounds = _VL.varlen_bounds(cu_seqlens, q=Ltri)
    meta = _VL.chunk_meta(bounds, C, Ltri.device)
    nt_total = meta.shape[0]
    # Every token row is written by exactly one block, or the caller gets dirty
    # memory back: the output buffer is torch.empty, not zeros.
    assert nt_total > 0, "a non-empty batch must produce at least one chunk"
    assert int(meta[:, _VL.META_ROWS].sum()) == SEQ, "chunk metadata does not cover every token exactly once"
    ker = kda_solve_tril_ker_varlen(B, SEQ, HV, C, nt_total, dtype=_DTYPE[Ltri.dtype])
    return ker(Ltri, _identity(C, Ltri.device), meta)


# ------------------------------------------------------------------- testing
_TAG = {torch.float16: "fp16", torch.bfloat16: "bf16", torch.float32: "fp32"}

# Kernel-only budget: the recurrence runs in fp32 internally, so this is
# dominated by rounding A back to the transport dtype on the way out.
_TOL = {torch.float16: 5e-3, torch.bfloat16: 3e-2, torch.float32: 1e-5}


def _relerr(got, ref):
    got, ref = got.float(), ref.float()
    return (got - ref).abs().max().item() / max(ref.abs().max().item(), 1e-9)


def _case(B, SEQ, H, HV, K, V, C, gate, dtype):
    """One shape against the stage-3 golden out of the L1 reference pipeline.

    Inputs are built on CPU in fp32 so the reference is computed exactly; only
    the tensor the kernel actually consumes goes to the device.
    """
    q, k, v, g, beta, _ = kda_chunk_ref.make_inputs(B, SEQ, H, HV, K, V, device="cpu", dtype=torch.float32, gate=gate)
    st = kda_chunk_ref.stage_tensors(q, k, v, g, beta, C=C)

    # stage_tensors returns external layout as a transposed view.  .contiguous()
    # is a test-side materialisation, not part of the kernel path: in the real
    # pipeline L arrives already contiguous in [B, SEQ, HV, C] from the kkt
    # kernel, which is why the wrapper asserts contiguity rather than fixing it.
    l_in = st["L"].contiguous().to(device="npu", dtype=dtype)
    got = kda_solve_tril(l_in).float().cpu()

    # Two references, because two different errors are worth separating.
    # a_same re-runs the reference on the *rounded* matrix the kernel was handed,
    # so e_kern is the kernel alone.  e_gold is against the pipeline golden and
    # additionally carries the amplification of that rounding through A dL A,
    # which e_sens measures on the reference itself -- hence the adaptive bound
    # instead of a hand-picked loose constant.  For fp32 the input is exact,
    # e_sens is 0, and the golden check tightens to _TOL.
    li = l_in.float().cpu().transpose(1, 2)  # [B, HV, SEQ, C], the layout ref_solve_tril wants
    pad = (-SEQ) % C
    if pad:
        # ref_solve_tril chunks the token axis, so it needs a whole number of
        # chunks.  Padding L with zero rows is neutral for exactly the reason
        # the kernel's zeroed tail rows are: a zero row of L yields an identity
        # row of A and contributes nothing to the substitution for any row above
        # it.  The padded rows are sliced off again below.
        li = torch.cat([li, li.new_zeros(li.shape[0], li.shape[1], pad, li.shape[3])], dim=2)
    a_same = kda_chunk_ref.ref_solve_tril(li).transpose(1, 2)
    if pad:
        a_same = a_same[:, :SEQ].contiguous()
    e_kern = _relerr(got, a_same)
    e_gold = _relerr(got, st["A"])
    e_sens = _relerr(a_same, st["A"])

    finite = bool(torch.isfinite(got).all().item())
    ok = finite and e_kern < _TOL[dtype] and e_gold <= 4.0 * e_sens + _TOL[dtype]
    print(
        f"  B{B} T{SEQ:<4d} H{H} HV{HV} K{K:<3d} V{V:<3d} C{C:<2d} "
        f"{_TAG[dtype]} {gate:8s} kern={e_kern:.2e} gold={e_gold:.2e} "
        f"sens={e_sens:.2e}  {'ok' if ok else 'FAIL'}"
    )
    return ok


def _vcase(seqlens, H, HV, K, V, C, gate, dtype, note=""):
    """One varlen batch against the stage-3 golden, over the WHOLE flat token axis.

    Comparing every token rather than per sequence is what makes this able to
    see a spill.  A chunk that stores C rows instead of its own count writes
    plausible, finite `I - X'` values onto the NEXT sequence's A rows; that
    sequence's own block writes the same rows too, so which value survives is a
    scheduling accident.  Against the full-axis golden the bad rows are simply
    wrong, and a row nobody wrote is dirty memory, which is also wrong -- one
    comparison covers both.
    """
    q, k, v, g, beta, _, cu = kda_chunk_ref.make_varlen_inputs(seqlens, H, HV, K, V, device="cpu", dtype=torch.float32, gate=gate)
    st = kda_chunk_ref.stage_tensors(q, k, v, g, beta, C=C, cu_seqlens=cu)

    l_in = st["L"].contiguous().to(device="npu", dtype=dtype)
    got = kda_solve_tril(l_in, cu_seqlens=cu.npu()).float().cpu()

    e_gold = _relerr(got, st["A"])
    finite = bool(torch.isfinite(got).all().item())
    shape_ok = tuple(got.shape) == tuple(st["A"].shape)
    ok = finite and shape_ok and e_gold < _TOL[dtype]
    print(f"  {str(seqlens):24s} HV{HV} C{C:<2d} {_TAG[dtype]} {gate:8s} gold={e_gold:.2e}  {'ok' if ok else 'FAIL'}  {note}")
    return ok


def main():
    tilelang.disable_cache()
    torch.manual_seed(0)

    ok = True

    print("== chunk length x GVA (fp16) ==")
    for C in (32, 64):
        for HV in (2, 4):  # HV == H, then HV == 2 * H
            for gate in ("normal", "forget"):
                ok &= _case(2, 256, 2, HV, 64, 64, C, gate, torch.float16)

    print("== K3 head dimension, K = V = 128 (fp16) ==")
    for gate in ("normal", "forget"):
        ok &= _case(1, 256, 1, 1, 128, 128, 64, gate, torch.float16)

    print("== grid edges (fp16) ==")
    # total == 1: odd task count, exercises the surplus-core clamp
    ok &= _case(1, 32, 1, 1, 64, 64, 32, "normal", torch.float16)
    # HV == 3 * H and HV not a power of two
    ok &= _case(1, 128, 2, 6, 64, 64, 64, "forget", torch.float16)

    print("== ragged tail (SEQ % C != 0) ==")
    ok &= _case(2, 70, 1, 2, 64, 64, 64, "normal", torch.float16)  # 70 = 64 + 6
    ok &= _case(1, 33, 1, 1, 64, 64, 32, "forget", torch.float16)  # 33 = 32 + 1, one valid tail row
    ok &= _case(1, 65, 1, 1, 128, 128, 64, "forget", torch.float16)  # K3 dim, one valid tail row
    ok &= _case(2, 100, 2, 4, 64, 64, 32, "extreme", torch.float32)  # fp32, tight 1e-5 bound

    print("== dtype passthrough ==")
    ok &= _case(2, 256, 2, 2, 64, 64, 64, "normal", torch.bfloat16)
    ok &= _case(2, 256, 2, 4, 64, 64, 32, "forget", torch.bfloat16)
    # fp32 in, fp32 out: the input is exact, so these two gate the kernel
    # straight against the pipeline golden at 1e-5.
    ok &= _case(2, 256, 2, 2, 64, 64, 64, "forget", torch.float32)
    ok &= _case(1, 256, 1, 1, 128, 128, 64, "normal", torch.float32)

    print("== varlen (cu_seqlens) ==")
    ok &= _vcase([64, 64, 64], 1, 2, 64, 64, 64, "normal", torch.float32, "equal, chunk-aligned")
    ok &= _vcase([70, 33, 129], 1, 2, 64, 64, 64, "normal", torch.float32, "every sequence ragged -- interior tails")
    ok &= _vcase([70, 0, 129], 1, 2, 64, 64, 64, "forget", torch.float32, "empty sequence in the middle")
    ok &= _vcase([0, 70], 1, 2, 64, 64, 64, "normal", torch.float32, "empty sequence first")
    ok &= _vcase([70, 0], 1, 2, 64, 64, 64, "normal", torch.float32, "empty sequence last")
    ok &= _vcase([1, 200], 1, 2, 64, 64, 64, "forget", torch.float32, "one token, then a long sequence")
    ok &= _vcase([32], 1, 1, 64, 64, 32, "normal", torch.float32, "total == 1, surplus-core clamp")
    ok &= _vcase([100, 28], 2, 4, 64, 64, 32, "extreme", torch.float32, "GVA + extreme gate, C = 32")
    ok &= _vcase([65, 65], 1, 1, 128, 128, 64, "forget", torch.float16, "K3 dim, fp16")
    ok &= _vcase([70, 33], 1, 2, 64, 64, 64, "forget", torch.bfloat16, "bf16 passthrough")

    if ok:
        print("Kernel Output Match!")
    else:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
