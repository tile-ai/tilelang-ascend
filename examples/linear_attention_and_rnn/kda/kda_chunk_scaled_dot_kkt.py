"""KDA L1 stage 2: the decayed Gram matrix, at the frozen L0 interface.

    L[i, j] = beta_i * sum_d k[i, d] k[j, d] exp(G[i, d] - G[j, d]),   j < i

This is the stage where KDA stops being GDN.  GDN's decay factor exp(g_i - g_j)
is one scalar per (i, j), so its kernel lets the cube compute a plain K K^T and
rescales the result on the vector unit afterwards.  Per channel the factor is
exp(G[i, d] - G[j, d]): it sits *inside* the sum over d and cannot be hoisted
out of the matmul.

Why not the one-shot fold
-------------------------
Algebraically the factor splits as e^{G_i} . e^{-G_j} and can be folded into the
two operands, which keeps the whole thing a single matmul.  Numerically that is
a trap: G is a cumulative sum of non-positive gates, so e^{-G_j} grows without
bound down the chunk.  Measured on CPU with the reference layer, the naive fold
produces 2624 non-finite entries at the "forget" gate (min Gamma_C = -209) and
3840 at "extreme" (-841).  ``kda_chunk_ref.test_naive_fold_blows_up`` demonstrates
it rather than merely asserting it.

What this kernel does instead
-----------------------------
It evaluates the contraction directly, one output row at a time, with the causal
mask folded into the exponent *before* exp():

    e      = (G[i, :] - G[j, :]) * mask[i, j]   -> <= 0 for j < i, exactly 0 else
    L[i, j] = mask[i, j] * beta_i * sum_d k[i, d] k[j, d] e^{e}

Two properties make this safe by construction, and both are load-bearing:

  * Only differences of G are ever exponentiated, never a ratio of two exps.
    Deep inside a chunk e^{G_i} and e^{G_j} both underflow to 0 and 0/0 is NaN.
  * The mask multiplies the *exponent*, not the result.  G is non-increasing in
    i, so the j < i half of the exponent is non-positive for free, and the j > i
    half is crushed to exactly 0 before exp() can overflow it to inf.  Masking
    after exp() lets a single steep channel turn the discarded half into inf and
    then 0 * inf = NaN poisons the half that is kept.

This is strictly safer than the BC=16 anchored blocking used by
``kda_chunk_ref._decayed_dot`` (and by fla): the anchored form still bounds both
folded factors by 1, but this form never forms a positive exponent at all.  The
two agree to fp32 rounding, so the anchored golden validates this kernel
directly.  The reason to prefer anchoring is *speed*, not numerics -- it is what
lets the off-diagonal strips go to the cube.  See "known limitations".

Frozen interface (FLA contract)
-------------------------------
    Kt   [B, SEQ, H,  K]   dtype (fp16 / bf16)      read with hq = hv // GRP
    G    [B, SEQ, HV, K]   fp32, chunk-local cumsum of the log gate, <= 0
    Beta [B, SEQ, HV, 8]   fp32, only [..., 0] is used (32B alignment padding)
    Msk  [C, C]            fp32, strictly lower triangular
    L    [B, SEQ, HV, C]   dtype -- consumed as-is by stage 3 solve_tril

Note the token axis and the head-dim axis have the head axis between them, so a
[C, K] tile of one head is a *strided* read: C rows of K contiguous elements,
H * K apart in Kt and HV * K apart in G.  ``T.copy`` infers its region from the
*trailing* dims of the source, so the bare ``T.copy(Kt[bz, base, hq, 0], tile)``
that worked for the old [B, H, L, DK] layout would now read C *heads* of one
token, silently.  The token range is therefore written out as a slice --
``Kt[bz, base:base + C, hq, 0:K]``, region [1, C, 1, K].  A unit extent to the
left of the innermost one gets folded into the source row stride
(``compute_strideN`` in src/op/ascend.cc), so this lowers to a single
``DataCopyPad`` of C blocks with srcStride = (H - 1) * K elements.  Same index
shape as the GM->L1 reads in examples/deepseek_v4/lightning_indexer.py.
Single rows (query row, gate row, beta, output row) stay plain BufferLoads: a
one-row region needs no stride and is the pattern the L0 kernel already uses.

Parallel decomposition
----------------------
grid = B * HV * chunk_num, one block per (batch, value head, chunk).  The two
vector cores split the *rows* of the C x C output between them; both need every
key of the chunk, so both load the whole resident tile.  Nothing is contracted
across cores, so there is no cross-core communication and no workspace.

Known limitations
-----------------
  * ``SEQ % C == 0`` only.  Tail blocks are a later pass (fixed length first,
    varlen later); ``kda_chunk_ref.kda_chunk_ref`` already pads for them on host,
    but this kernel does not accept a ragged chunk.
  * Runs entirely on the vector unit, cube idle -- the slowest stage by a wide
    margin.  The follow-up is the anchored BC=16 decomposition: anchor each row
    sub-block at its first row, hand the off-diagonal strips to ``T.gemm_v0``
    (both folded factors are then bounded by 1), keep this element-wise path for
    the diagonal blocks.
  * Both vector cores load the full [C, K] key/gate tiles although core 0 only
    ever reads columns j < C/2.  Harmless duplication, kept for index clarity.
  * C must be even (two vector cores split C rows) and K % 16 == 0 / C % 16 == 0
    so every row copy starts on a 32B boundary.
"""

import os
import sys

import torch
import tilelang
from tilelang import language as T

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import kda_chunk_ref  # noqa: E402

# Only AUTO_SYNC, matching all six GDN kernels.  MEMORY_PLANNING is deliberately
# left off: on the backward bwd_dot it aliased a reduction target with a scratch
# tile -- the registers held the right values and only the store wrote zeros.
pass_configs = {tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True}

UB_LIMIT = 196352


def _ub_bytes(C, K, elem):
    """Byte budget of the allocations below, plus one worst-case scratch tile."""
    big_f32 = 3 * C * K * 4  # kfull_ub, gfull_ub, prod_ub
    big_low = C * K * elem  # kfull_half
    rows = K * 4 + K * elem + K * 4  # krow_ub, krow_half, grow_ub
    cols = C * 4 + C * 4 + C * elem  # mrow_ub, red_ub, arow_half
    scratch = C * K * 4  # a compound T.Parallel would add one
    return big_f32 + big_low + rows + cols + 32 + scratch


@tilelang.jit(out_idx=[-1], pass_configs=pass_configs)
def kkt_ker(B, SEQ, H, HV, K, C, dtype="float16", accum_dtype="float"):
    assert SEQ % C == 0, "first pass supports whole chunks only"
    assert HV % H == 0, "HV must be divisible by H (GVA)"
    assert C % 2 == 0, "two vector cores split the C rows"

    chunk_num = SEQ // C
    VEC_NUM = 2
    CV = C // VEC_NUM
    GRP = HV // H  # GRP value heads share one qk head

    @T.prim_func
    def main(
        Kt: T.Tensor([B, SEQ, H, K], dtype),  # type: ignore
        G: T.Tensor([B, SEQ, HV, K], accum_dtype),  # type: ignore
        # beta gets a whole 32B slot per token: a [1] fp32 buffer is 4B and
        # skews every later allocation ("The UB address accessed by the VEC
        # instruction is not aligned").  Padded on host, only [0] is read.
        Beta: T.Tensor([B, SEQ, HV, 8], accum_dtype),  # type: ignore
        Msk: T.Tensor([C, C], accum_dtype),  # type: ignore  strictly lower
        L: T.Tensor([B, SEQ, HV, C], dtype),  # type: ignore
    ):
        with T.Kernel(B * HV * chunk_num, is_npu=True) as (cid, vid):
            bx = cid % chunk_num  # chunk index
            hv = (cid // chunk_num) % HV  # value head
            bz = (cid // chunk_num) // HV  # batch
            hq = hv // GRP  # qk head behind this hv
            base = bx * C  # first token of the chunk

            kfull_ub = T.alloc_ub([C, K], accum_dtype)
            gfull_ub = T.alloc_ub([C, K], accum_dtype)
            # Not named tmp_*: compound T.Parallel expressions get lowered into
            # auto-allocated scratch tiles called tmp_ub, and the memory planner
            # rejects the duplicate name.
            prod_ub = T.alloc_ub([C, K], accum_dtype)
            kfull_half = T.alloc_ub([C, K], dtype)

            krow_ub = T.alloc_ub([K], accum_dtype)
            krow_half = T.alloc_ub([K], dtype)
            grow_ub = T.alloc_ub([K], accum_dtype)
            mrow_ub = T.alloc_ub([C], accum_dtype)
            red_ub = T.alloc_ub([C], accum_dtype)
            arow_half = T.alloc_ub([C], dtype)
            beta_ub = T.alloc_ub([8], accum_dtype)  # 32B, see Beta above

            with T.Scope("V"):
                # Resident tiles: every key and gate of the chunk.  The explicit
                # slice is load-bearing.  `T.copy(Kt[bz, base, hq, 0], tile)`
                # would infer the region from the *trailing* dims and put the C
                # extent on the head axis -- C heads of one token instead of C
                # tokens of one head, silently.  Writing the token range out as
                # a slice makes the region [1, C, 1, K]; the unit head extent is
                # then folded into the source row stride (H * K), which is
                # exactly one strided DataCopyPad of C blocks.
                #
                # If a future toolchain ever mis-strides this, the drop-in
                # fallback is to replace each of the two strided copies below
                # with a T.serial(C) loop of single-row copies indexed by
                # base + tj, which carry no stride at all -- C launches per tile
                # instead of one, but nothing left for the region inference to
                # get wrong.
                T.copy(Kt[bz, base : base + C, hq, 0:K], kfull_half)
                T.copy(G[bz, base : base + C, hv, 0:K], gfull_ub)
                T.copy(kfull_half, kfull_ub)  # cast dtype -> fp32

                for r in T.serial(CV):
                    ri = vid * CV + r  # this core's output row

                    # Row ri is re-read from global memory rather than sliced
                    # out of the resident tile: a UB->UB copy at a runtime row
                    # offset is not a pattern any kernel in this repo uses, and
                    # it does not do what it looks like it does.
                    T.copy(Kt[bz, base + ri, hq, 0], krow_half)
                    T.copy(krow_half, krow_ub)
                    T.copy(G[bz, base + ri, hv, 0], grow_ub)
                    T.copy(Msk[ri, 0], mrow_ub)
                    T.copy(Beta[bz, base + ri, hv, 0], beta_ub)

                    # One operation per T.Parallel, as the GDN kernels do:
                    # a compound expression allocates a scratch tile each.
                    #
                    # grow_ub[d] broadcasts along the *inner* loop variable, so
                    # writing to a different tile is fine here.  mrow_ub[j]
                    # broadcasts along the *outer* one and must be applied in
                    # place -- `out[j,d] = prod[j,d] * mrow[j]` does not lower.
                    for j, d in T.Parallel(C, K):
                        prod_ub[j, d] = grow_ub[d] - gfull_ub[j, d]
                    for j, d in T.Parallel(C, K):
                        prod_ub[j, d] = prod_ub[j, d] * mrow_ub[j]
                    for j, d in T.Parallel(C, K):
                        prod_ub[j, d] = T.exp(prod_ub[j, d])
                    for j, d in T.Parallel(C, K):
                        prod_ub[j, d] = prod_ub[j, d] * krow_ub[d]
                    for j, d in T.Parallel(C, K):
                        prod_ub[j, d] = prod_ub[j, d] * kfull_ub[j, d]
                    # Second mask: the j >= i exponents were flattened to 0, so
                    # exp() gave them 1 and they still carry k_i . k_j.  This is
                    # what actually discards them.
                    for j, d in T.Parallel(C, K):
                        prod_ub[j, d] = prod_ub[j, d] * mrow_ub[j]

                    # reduce_sum defaults to clear=True, i.e. it initialises the
                    # output itself (tilelang/language/reduce.py:79).  Do not
                    # pre-fill red_ub; pass clear=False if accumulation is what
                    # is wanted.
                    T.reduce_sum(prod_ub, red_ub, dim=-1)

                    # beta_i is one scalar for the whole row; a loop-invariant
                    # UB read broadcasts, the way the L0 kernel's
                    # `u_ub[j] *= beta_ub[0]` does.
                    for j in T.Parallel(C):
                        red_ub[j] = red_ub[j] * beta_ub[0]

                    # Store the finished row straight to global memory.  Staging
                    # it in a [CV, C] tile does not work: a pure write of the
                    # form `tile[r, j] = vec[j]` drops the row index and every
                    # iteration lands on row 0.
                    T.copy(red_ub, arow_half)
                    T.copy(arow_half, L[bz, base + ri, hv, 0])

    return main


def chunk_scaled_dot_kkt(k, G, beta, C=64):
    """Host wrapper.  Returns L [B, SEQ, HV, C] in the dtype of ``k``.

    Arguments are the frozen external tensors:
        k     [B, SEQ, H,  K]  fp16 / bf16
        G     [B, SEQ, HV, K]  fp32, chunk-local cumsum of the log gate (stage 1)
        beta  [B, SEQ, HV] or an already padded [B, SEQ, HV, 8] fp32

    Host does no transposing, reshaping or state movement -- the kernel indexes
    [B, SEQ, HV, *] directly.  The only host work is the beta 32B padding, the
    constant C x C mask and a dtype lookup.
    """
    B, SEQ, H, K = k.shape
    HV = G.shape[2]
    assert G.shape == (B, SEQ, HV, K), f"G must be [B, SEQ, HV, K], got {tuple(G.shape)}"
    assert HV % H == 0, "HV must be divisible by H (GVA)"
    assert SEQ % C == 0, f"this kernel needs SEQ % C == 0, got SEQ={SEQ} C={C}"
    assert C % 2 == 0 and C % 16 == 0, "C must be even and 32B-aligned in dtype"
    assert K % 16 == 0, "K must be 32B-aligned in dtype"

    elem = torch.finfo(k.dtype).bits // 8
    need = _ub_bytes(C, K, elem)
    assert need <= UB_LIMIT, f"UB budget {need} > {UB_LIMIT} for C={C} K={K}; lower C"

    # SEQ == 0 slips past the assert above (0 % C == 0) and would launch a
    # zero-block grid over an unwritten output.  A zero-length sequence is legal
    # input; there are no token pairs, so L is empty along the token axis.
    if SEQ == 0:
        return torch.empty((B, 0, HV, C), device=k.device, dtype=k.dtype)

    # beta: one 32B slot per token, only lane 0 carries a value.  Padding the
    # last axis to 8 rather than the head axis to HV + 8 is deliberate: the
    # latter starts head hv at byte 4 * hv, which is not a multiple of 32.
    if beta.dim() == 3:
        beta_p = torch.zeros((B, SEQ, HV, 8), device=beta.device, dtype=torch.float)
        beta_p[..., 0] = beta.float()
    else:
        assert beta.shape == (B, SEQ, HV, 8), f"padded beta must be [B,SEQ,HV,8], got {tuple(beta.shape)}"
        beta_p = beta.float()

    msk = torch.tril(torch.ones((C, C), device=k.device, dtype=torch.float), diagonal=-1)

    dt = {torch.float16: "float16", torch.bfloat16: "bfloat16"}[k.dtype]
    return kkt_ker(B, SEQ, H, HV, K, C, dtype=dt)(k, G.float(), beta_p, msk)


# Alias matching the stage name used by the GDN pipeline.
kkt = chunk_scaled_dot_kkt


# ----------------------------------------------------------------- test
def _relerr(got, ref):
    got = got.float().cpu()
    ref = ref.float().cpu()
    return (got - ref).abs().max().item() / max(ref.abs().max().item(), 1e-9)


def _case(B, SEQ, H, HV, K, V, C, gate, dtype):
    # Inputs and golden are built on CPU in fp32: stage_tensors on the NPU
    # dispatches einsum to matmul with reduced-precision accumulation and drifts
    # two identical fp32 references by ~3e-4, which would blur the comparison.
    # The tensors handed to the kernel are bit-identical copies of these.
    q, k, v, g, beta, _ = kda_chunk_ref.make_inputs(B, SEQ, H, HV, K, V, device="cpu", dtype=dtype, gate=gate)
    st = kda_chunk_ref.stage_tensors(q, k, v, g, beta, C=C)
    G, ref = st["G"], st["L"]

    got = chunk_scaled_dot_kkt(k.npu(), G.npu(), beta.npu(), C=C)

    err = _relerr(got, ref)
    finite = bool(torch.isfinite(got.float()).all())
    tol = 5e-3 if dtype == torch.float16 else 3e-2  # bf16 has 8 mantissa bits
    ok = finite and err < tol
    tag = "bf16" if dtype == torch.bfloat16 else "fp16"
    print(
        f"  B{B} T{SEQ:<4d} H{H} HV{HV} K{K:<3d} C{C:<2d} {tag} {gate:8s} "
        f"relerr={err:.2e} finite={'Y' if finite else 'N'}  "
        f"{'ok' if ok else 'FAIL'}"
    )
    return ok


def main():
    tilelang.disable_cache()
    torch.manual_seed(0)

    ok = True
    print("== C=32 / C=64, HV == H, two gate settings ==")
    for gate in ("normal", "forget"):
        ok &= _case(2, 256, 4, 4, 64, 64, 32, gate, torch.float16)
        ok &= _case(2, 256, 4, 4, 64, 64, 64, gate, torch.float16)

    print("== GVA: HV == 2H ==")
    for gate in ("normal", "forget"):
        ok &= _case(2, 128, 2, 4, 64, 64, 32, gate, torch.float16)
        ok &= _case(2, 128, 2, 4, 64, 64, 64, gate, torch.float16)

    print("== single head, and the gate extremes ==")
    ok &= _case(1, 64, 1, 1, 64, 64, 64, "normal", torch.float16)
    for gate in ("keep", "extreme"):
        ok &= _case(2, 128, 2, 2, 64, 64, 64, gate, torch.float16)

    print("== K3 spec: K = V = 128 ==")
    ok &= _case(1, 256, 2, 2, 128, 128, 64, "forget", torch.float16)
    ok &= _case(1, 256, 1, 2, 128, 128, 64, "normal", torch.float16)  # + GVA
    ok &= _case(1, 128, 1, 1, 128, 128, 32, "forget", torch.float16)

    print("== bf16 dtype passthrough ==")
    for gate in ("normal", "forget"):
        ok &= _case(2, 128, 2, 4, 64, 64, 64, gate, torch.bfloat16)

    if ok:
        print("Kernel Output Match!")
    else:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
