"""KDA L1 stage 4 (wy_fast): the UT transform, at the frozen L0 interface.

    M = A . Diag(beta)
    U = M @ V
    W = M @ (K . e^{G})

GDN scales row i of K by the scalar e^{gamma_i}; KDA scales element (i,d) by
e^{gamma_{i,d}}, so GDN's one row-broadcast multiply becomes a full [C, K]
elementwise multiply.  Everything else -- the two gemms, the workspace hand-off,
the cross-core flag -- is structurally the same as the GDN kernel.

Only W and U are produced here.  kg = K . e^{G_C - G}, qg and Aqk are computed
in place by the downstream chunk_h / chunk_o kernels, which saves three GM
round trips; ``kda_chunk_ref.ref_wy_fast`` returns them as well, but this stage is
only checked against its ``["W"]`` / ``["U"]`` entries.

What changed against the earlier chunkwise port (interface only, math intact)
----------------------------------------------------------------------------
    * layout [B, H, L, D] -> [B, SEQ, HV, D].  T and D are no longer adjacent,
      so every tile load is a strided 2-D DMA: one row is D contiguous
      elements, consecutive rows are HV*D apart.  The token extent has to be
      put on axis 1 with an explicit slice (see the note in the V scope).
    * GVA: Q/Kt carry H heads, everything else HV.  A value head hv reads key
      head hq = hv // GRP.
    * beta arrives as [B, SEQ, HV, 8] fp32 (32B-aligned, data in column 0)
      instead of [B, H, L] in the input dtype.
    * A arrives as [B, SEQ, HV, C] in the input dtype, straight from the
      solve_tril kernel -- the [C, C] tile per (chunk, head) is now strided too.
    * dtype is threaded through from the caller (fp16 / bf16), never hardcoded.
    * the two workspaces are compact, one [C, K] / [C, V] slab per block,
      indexed by the block id.  That keeps both workspace copies byte-identical
      in shape to the proven GDN ones (contiguous rows, row pitch == width) and
      confines the new strided patterns to the real inputs and outputs.

Numerics
--------
e^{G} <= 1 here (G is a chunk-local cumsum of non-positive gates), so this
stage cannot overflow.  It can underflow to zero deep inside a chunk, which is
the right answer: that token's contribution really has decayed away.  Nothing
here computes a ratio of two exponentials, and no mask is applied after an
exp -- both traps live in stage 2 (kkt), not here.

Ragged tail
-----------
``SEQ % C != 0`` is supported.  The grid uses ``ceildiv(SEQ, C)``; every GM copy
on the token axis -- the four vector-side loads, the A load into L1 and the two
L0C stores of W and U -- is clamped to the R valid rows by ``compute_valid_extent``
(src/op/ascend.cc:410), and ``copy_gm_to_l1`` additionally zeroes the tail rows
of the A tile.  So A's pad rows are exact zeros, the pad rows of W and U come out
zero, and the clamped fixpipe never writes them.

Two things the framework does not do for us, both handled in the V scope:

  * The UB tail rows are stale, and ``g_ub`` is exponentiated and published to
    ``ws_k``, which the cube reads as a FULL [C, BK] operand -- a garbage row
    becomes inf and NaN-poisons every valid row of W.  The four tiles are
    pre-filled with zero before the clamped loads overwrite the valid part.
  * When a core's whole CV window starts past SEQ the clamp gives validRow == 0,
    i.e. ``DataCopyExtParams`` with blockCount 0.  That is outside the documented
    [1, 4095] range and is not exercised anywhere in this repo, so the DMA is
    skipped instead of being issued with a zero count.

★ The skip covers the DMAs ONLY.  ``set_cross_flag`` / ``wait_cross_flag`` are
never put inside a branch: if one vector core skipped its flag, the cube would
wait on it forever.
"""

import os
import sys

import torch
import tilelang
from tilelang import language as T

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kda_chunk_ref  # noqa: E402

# One pass config only, the same one all six GDN kernels use.  MEMORY_PLANNING
# is deliberately off: on the backward bwd_dot kernel it aliased a reduction
# destination with a scratch tile and stored all-zero results.
pass_configs = {tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True}

VEC_NUM = 2
BETA_PAD = 8  # beta's padded last dim: 8 fp32 = 32B, the minimum UB alignment


@tilelang.jit(out_idx=[-2, -1], workspace_idx=[-4, -3], pass_configs=pass_configs)
def wy_fast_ker(B, SEQ, H, HV, K, V, C, BK, BV, dtype="float16", accum_dtype="float"):
    # ceil, not floor: the last chunk may be ragged.  SEQ is a Python int at
    # trace time, so this stays a compile-time constant, and grid / ws_k / ws_v
    # first dims all follow from it.
    chunk_num = -(-SEQ // C)
    R = SEQ % C  # 0 when aligned; else the valid row count of the last chunk
    RAGGED = R != 0
    bk_num = K // BK
    bv_num = V // BV
    GRP = HV // H  # value heads per qk head (GVA)
    CV = C // VEC_NUM  # rows of the chunk owned by one vector core
    grid = B * HV * chunk_num

    @T.prim_func
    def main(
        Kt: T.Tensor([B, SEQ, H, K], dtype),  # type: ignore
        Vt: T.Tensor([B, SEQ, HV, V], dtype),  # type: ignore
        # beta is one fp32 per (token, head), padded to 8 so each row is a whole
        # 32B block: a [.., 1] tile would be a 4B UB row and the DMA would land
        # the rows 32B apart anyway, walking over whatever follows.
        Beta: T.Tensor([B, SEQ, HV, BETA_PAD], accum_dtype),  # type: ignore
        G: T.Tensor([B, SEQ, HV, K], accum_dtype),  # type: ignore  chunk-local cumsum
        A: T.Tensor([B, SEQ, HV, C], dtype),  # type: ignore  (I + L)^{-1}
        # Workspaces: the vector cores publish the scaled operands here for the
        # cube to pick up.  torch.empty (dirty) memory is fine because every
        # element is written before the flag is set -- vid 0 writes rows
        # [0, CV), vid 1 writes [CV, C), both full width.  Nothing here needs
        # zero-initialised GM, which is what would force it to be a real input.
        ws_k: T.Tensor([grid, C, K], dtype),  # type: ignore
        ws_v: T.Tensor([grid, C, V], dtype),  # type: ignore
        W: T.Tensor([B, SEQ, HV, K], dtype),  # type: ignore
        U: T.Tensor([B, SEQ, HV, V], dtype),  # type: ignore
    ):
        with T.Kernel(grid, is_npu=True) as (cid, vid):
            bx = cid % chunk_num  # chunk index along the sequence
            hv = (cid // chunk_num) % HV  # value head
            bz = (cid // chunk_num) // HV  # batch
            hq = hv // GRP  # matching qk head
            t0 = bx * C  # first token of this chunk

            kg_ub = T.alloc_ub([CV, K], accum_dtype)  # beta_i * e^{G} * K
            v_ub = T.alloc_ub([CV, V], accum_dtype)  # beta_i * V
            g_ub = T.alloc_ub([CV, K], accum_dtype)
            beta8_ub = T.alloc_ub([CV, BETA_PAD], accum_dtype)
            beta_ub = T.alloc_ub([CV], accum_dtype)  # CV*4 >= 32B for C >= 32
            k_half = T.alloc_ub([CV, K], dtype)
            v_half = T.alloc_ub([CV, V], dtype)
            # Not named tmp_*: compound T.Parallel expressions get lowered into
            # auto-allocated scratch tiles called tmp_ub, and the memory planner
            # then rejects the duplicate name.

            a_l1 = T.alloc_L1([C, C], dtype)
            k_l1 = T.alloc_L1([C, BK], dtype)
            v_l1 = T.alloc_L1([C, BV], dtype)
            w_l0 = T.alloc_L0C([C, BK], accum_dtype)
            u_l0 = T.alloc_L0C([C, BV], accum_dtype)

            with T.Scope("V"):
                # Strided tile loads.  The explicit token-axis slice is what
                # puts the CV extent on axis 1; a bare ``Kt[bz, row, hq, 0]``
                # index would expand the region over the *last two* axes and
                # read CV consecutive heads of one token instead.  With the
                # slice the region extents are [1, CV, 1, K], from which the
                # backend derives row pitch = HV*K (it folds every trailing
                # unit-extent axis into the stride) and one 2-D DMA of CV rows.
                # Ragged tail, and there are two distinct hazards here.
                #
                # (1) g_ub goes through T.exp below and the result is published
                #     to ws_k, which the cube then reads as a FULL [C, BK]
                #     operand.  A stale UB tail row becomes exp(garbage) = inf
                #     and NaN-poisons every valid row of W.  So pre-fill: the
                #     clamped DMA overwrites only the rows that exist, and the
                #     rest stay the zeros we put there.  Zero is also correct
                #     semantically -- g = 0 is alpha = 1, and beta = 0 writes
                #     nothing through the delta rule.
                #
                # (2) when this core's whole CV window starts past SEQ the
                #     clamp yields validRow == 0, i.e. DataCopyExtParams with
                #     blockCount 0.  That is outside the documented [1, 4095]
                #     and is not exercised anywhere in this repo, so the DMA is
                #     skipped rather than issued with a zero count.
                #
                # ONLY the DMAs are skipped.  The cross-core flag below is NOT
                # inside any branch: one core failing to set it deadlocks the
                # cube on wait_cross_flag(1) forever.
                if RAGGED and bx == chunk_num - 1:
                    T.tile.fill(k_half, 0)
                    T.tile.fill(v_half, 0)
                    T.tile.fill(g_ub, 0.0)
                    T.tile.fill(beta8_ub, 0.0)

                if (not RAGGED) or t0 + vid * CV < SEQ:
                    T.copy(Kt[bz, t0 + vid * CV : t0 + (vid + 1) * CV, hq, :], k_half)
                    T.copy(Vt[bz, t0 + vid * CV : t0 + (vid + 1) * CV, hv, :], v_half)
                    T.copy(G[bz, t0 + vid * CV : t0 + (vid + 1) * CV, hv, :], g_ub)
                    T.copy(Beta[bz, t0 + vid * CV : t0 + (vid + 1) * CV, hv, :], beta8_ub)
                T.copy(k_half, kg_ub)  # dtype -> fp32
                T.copy(v_half, v_ub)

                # beta sits in column 0 of a zero-padded 8-wide row, so the row
                # sum *is* the column-0 extract -- and it lands in a 1-D buffer,
                # which is the only shape the per-row broadcast below accepts
                # (a [CV, 1] column cannot be read strided by the vector unit).
                # reduce_sum defaults to clear=True and initialises beta_ub, so
                # no fill here; passing clear=False is what asks for accumulate.
                # If a caller ever hands over a Beta whose pad slots are not
                # zero, load ``Beta[bz, lo:hi, hv, 0:1]`` into this same [CV, 8]
                # tile instead: a 1-wide GM region makes the DMA pre-fill the
                # tile with its pad value (0) and then write only column 0.
                T.reduce_sum(beta8_ub, beta_ub, dim=-1)

                # One operation per T.Parallel, as the GDN kernels do: compound
                # expressions allocate a scratch tile each.
                for i, j in T.Parallel(CV, K):
                    g_ub[i, j] = T.exp(g_ub[i, j])
                # GDN: k_ub[i, j] *= g_ub[i]   (one scalar per row)
                for i, j in T.Parallel(CV, K):
                    kg_ub[i, j] = kg_ub[i, j] * g_ub[i, j]
                # Row broadcast of a 1-D buffer has to be in place: writing
                # ``out[i, j] = tile[i, j] * vec[i]`` to a second buffer does
                # not survive the vector lowering.
                for i, j in T.Parallel(CV, K):
                    kg_ub[i, j] = kg_ub[i, j] * beta_ub[i]
                for i, j in T.Parallel(CV, V):
                    v_ub[i, j] = v_ub[i, j] * beta_ub[i]

                T.copy(kg_ub, k_half)  # fp32 -> dtype for the cube
                T.copy(v_ub, v_half)
                T.copy(k_half, ws_k[cid, vid * CV : (vid + 1) * CV, :])
                T.copy(v_half, ws_v[cid, vid * CV : (vid + 1) * CV, :])
                T.set_cross_flag("MTE3", 1)

            with T.Scope("C"):
                # A is already in GM from solve_tril; issue its load before the
                # wait so this MTE2 overlaps the vector cores' work.  [C, C]
                # tile, row pitch HV*C.
                T.copy(A[bz, t0 : t0 + C, hv, :], a_l1)
                T.wait_cross_flag(1)

                for i in T.serial(bk_num):
                    T.copy(ws_k[cid, :, i * BK : (i + 1) * BK], k_l1)
                    T.gemm_v0(a_l1, k_l1, w_l0, init=True)
                    T.copy(w_l0, W[bz, t0 : t0 + C, hv, i * BK : (i + 1) * BK])

                for i in T.serial(bv_num):
                    T.copy(ws_v[cid, :, i * BV : (i + 1) * BV], v_l1)
                    T.gemm_v0(a_l1, v_l1, u_l0, init=True)
                    T.copy(u_l0, U[bz, t0 : t0 + C, hv, i * BV : (i + 1) * BV])

    return main


_DTYPES = {torch.float16: "float16", torch.bfloat16: "bfloat16"}


def wy_fast(k, v, beta, G, A, C, BK=None, BV=None):
    """Host wrapper.  All tensors are in the frozen external layout.

        k    [B, SEQ, H,  K]  dtype    key, not GVA-expanded
        v    [B, SEQ, HV, V]  dtype
        beta [B, SEQ, HV]     any float, read as fp32
        G    [B, SEQ, HV, K]  fp32     stage-1 output (chunk-local cumsum)
        A    [B, SEQ, HV, C]  dtype    stage-3 output
        ->   W [B, SEQ, HV, K], U [B, SEQ, HV, V], both in dtype

    The host does no layout surgery: it only pads beta's last dim to 8 (the
    32B UB alignment the kernel needs) and looks the dtype up.  Everything
    else is indexed in place by the kernel.
    """
    B, SEQ, H, K = k.shape
    HV, V = v.shape[2], v.shape[-1]
    assert HV % H == 0, "HV must be divisible by H (GVA)"
    assert C % (VEC_NUM * 16) == 0, f"C must be a multiple of {VEC_NUM * 16}, got {C}"
    assert G.shape == (B, SEQ, HV, K) and G.dtype == torch.float, "G must be fp32 [B, SEQ, HV, K]"
    assert A.shape == (B, SEQ, HV, C) and A.dtype == k.dtype, "A must be dtype [B, SEQ, HV, C]"

    BK = K if BK is None else BK
    BV = V if BV is None else BV
    assert K % BK == 0 and BK % 16 == 0, f"need K % BK == 0 and BK % 16 == 0, got K={K} BK={BK}"
    assert V % BV == 0 and BV % 16 == 0, f"need V % BV == 0 and BV % 16 == 0, got V={V} BV={BV}"

    # SEQ == 0 slips past the assert above (0 % C == 0) and would launch a
    # zero-block grid over unwritten outputs.  A zero-length sequence is legal
    # input; both UT-transform outputs are empty along the token axis.
    if SEQ == 0:
        return (
            torch.empty((B, 0, HV, K), device=k.device, dtype=k.dtype),
            torch.empty((B, 0, HV, V), device=k.device, dtype=k.dtype),
        )

    # beta: one fp32 per (token, head) in column 0, the other 7 slots zero.
    # The zeros are load-bearing -- the kernel recovers column 0 as the row sum
    # of the padded row.  Padding to [B, SEQ, HV, 8] and not [B, SEQ, HV + 8]:
    # the latter starts head hv at byte offset 4*hv, which is not 32B aligned.
    beta_p = torch.zeros((B, SEQ, HV, BETA_PAD), device=k.device, dtype=torch.float)
    beta_p[..., 0] = beta.float()

    dt = _DTYPES[k.dtype]
    ker = wy_fast_ker(B, SEQ, H, HV, K, V, C, BK, BV, dtype=dt)
    return ker(k, v, beta_p, G, A)


# ----------------------------------------------------------------- test
def _relerr(x, r):
    r = r.float()
    return (x.float() - r).abs().max().item() / max(r.abs().max().item(), 1e-9)


def _case(B, SEQ, H, HV, K, V, C, gate, dtype):
    q, k, v, g, beta, _ = kda_chunk_ref.make_inputs(B, SEQ, H, HV, K, V, device="npu", dtype=dtype, gate=gate)
    ref = kda_chunk_ref.stage_tensors(q, k, v, g, beta, C=C)

    # stage_tensors hands everything back in external layout, but through a
    # transpose, so the views are not contiguous.  contiguous() here is a
    # harness artefact only: in the real pipeline G and A come straight out of
    # the cumsum and solve_tril kernels already laid out this way.
    G = ref["G"].contiguous()
    A = ref["A"].to(dtype).contiguous()

    W, U = wy_fast(k, v, beta, G, A, C)

    # The golden multiplies by an fp32 A; the kernel gets the same A rounded to
    # the input dtype (the cube needs it there), so that quantisation shows up
    # in the comparison alongside the kernel's own fp16/bf16 gemm operands.
    tol = 5e-3 if dtype == torch.float16 else 3e-2
    eW, eU = _relerr(W, ref["W"]), _relerr(U, ref["U"])
    finite = bool(torch.isfinite(W.float()).all() and torch.isfinite(U.float()).all())
    ok = finite and eW < tol and eU < tol
    tag = "bf16" if dtype == torch.bfloat16 else "fp16"
    print(
        f"  B{B} T{SEQ:<4d} H{H} HV{HV} K{K:<3d} V{V:<3d} C{C:<2d} {tag} {gate:8s} "
        f"W={eW:.2e} U={eU:.2e} finite={finite}  {'ok' if ok else 'FAIL'}"
    )
    return ok


def main():
    tilelang.disable_cache()
    torch.manual_seed(0)

    ok = True
    print("== shapes x gates (fp16) ==")
    for B, SEQ, H, HV, K, V, C, gate in [
        (2, 128, 2, 2, 64, 64, 32, "normal"),  # HV == H
        (2, 128, 2, 2, 64, 64, 64, "normal"),
        (2, 128, 2, 4, 64, 64, 32, "forget"),  # HV == 2H, deep decay
        (2, 128, 2, 4, 64, 64, 64, "forget"),
        (1, 256, 1, 1, 128, 128, 64, "forget"),  # K3 head dim
        (1, 128, 2, 4, 128, 128, 32, "normal"),  # K3 head dim + GVA
    ]:
        ok &= _case(B, SEQ, H, HV, K, V, C, gate, torch.float16)

    print("== ragged tail (SEQ % C != 0) ==")
    ok &= _case(2, 70, 1, 2, 64, 64, 64, "normal", torch.float16)  # R=6: core 0 partly valid, core 1 entirely empty
    ok &= _case(1, 33, 1, 1, 64, 64, 32, "forget", torch.float16)  # R=1, core 1 gets validRow==0
    ok &= _case(1, 65, 1, 1, 128, 128, 64, "forget", torch.float16)  # K3 dim, R=1
    ok &= _case(2, 100, 2, 4, 64, 64, 32, "extreme", torch.float16)  # R=4, GVA, extreme gate
    ok &= _case(1, 96, 1, 1, 64, 64, 64, "normal", torch.float16)  # R=32 == CV, exact core boundary

    print("== bf16 (dtype must be threaded through, not hardcoded) ==")
    for gate in ("normal", "forget"):
        ok &= _case(2, 128, 2, 4, 64, 64, 64, gate, torch.bfloat16)

    if ok:
        print("Kernel Output Match!")
    else:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
