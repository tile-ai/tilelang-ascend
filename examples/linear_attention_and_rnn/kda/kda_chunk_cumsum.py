"""KDA L1 stage 1/6: chunk-local inclusive prefix sum of the log-domain gate.

    in   G  [B, SEQ, HV, K]  fp32, log-domain, g <= 0
    out  G  [B, SEQ, HV, K]  fp32, inclusive prefix sum restarted every C tokens

golden = kda_l1_ref.stage_tensors(q, k, v, g, beta, C=C)["G"]

GDN carries one scalar gate per token, so its cumsum is a C-step scalar chain.
KDA carries K of them, so every step here is a K-wide vector add and the chain
moves K times as much data -- the gate tensor grows from [B,H,L] to [B,T,HV,K].

Restarting the sum at every chunk boundary is deliberate, not an optimisation:
it caps how far the exponents can travel inside a chunk, which is what keeps
every exp() in the five later stages in range.  See the numerical rules in
kda_l1_ref.py.

What changed vs the earlier chunkwise port (the math is untouched)
-----------------------------------------------------------------
1. Layout.  [B, H, L, DK] -> [B, SEQ, HV, K].  T and K are no longer adjacent:
   HV sits between them, so a [C, K] tile is a *strided* transfer -- C bursts of
   K contiguous fp32, with a gap of (HV - 1) * K fp32 between bursts.  It is
   expressed as an explicit region slice on both sides,
       T.copy(G[bz, t0 : t0 + C, hv, ko : ko + BK], g_ub)
   which the Ascend copy lowering turns into exactly that one-level-strided
   DataCopyPad (src/op/ascend.cc:compute_strideN folds every interior extent-1
   dim of the region into the row pitch, giving strideN = HV * K).  Nothing is
   transposed or gathered on the host.
2. Head axis.  The head range is HV (value heads), not H.  This stage never
   touches Q/Kt, so GVA / GRP / scale do not appear here at all -- the gate is
   already per value head in the frozen contract.
3. Core split.  The old kernel gave each vector core a different head
   (by = ... * 2 + vid), which silently requires an even head count.  The
   token-axis chain is independent per channel, so this version splits the K
   axis instead: both cores own the same (batch, head, chunk) and half of the
   channels each.  Consequences:
     * HV parity is now irrelevant -- HV = 1 or 3 work, no host assert, no
       single-core degradation.  (This is the third option for the "HV odd"
       question; it removes the constraint instead of handling it.)
     * cid decomposes to (bz, hv, i_c) alone, so the grid math no longer
       depends on VEC_NUM.  The five later stages can reuse the decomposition
       verbatim and put vid on whatever inner axis suits them (V for chunk_h /
       chunk_o, rows of C for kkt), which is what they need anyway since none of
       them can split an odd head axis either.
     * At the K3 spec (K = 128) BK = 64 fp32 = one full 256B vector
       instruction, so the split costs nothing there.  At K = 64 each step is a
       half-width vector op.

Known limitations (first pass, on purpose)
------------------------------------------
* ``SEQ % C == 0``.  Tail blocks are the next round (task spec: fixed length
  first, varlen later).  kda_l1_ref.kda_chunk_ref pads on the host, but
  stage_tensors -- the golden for this file -- refuses a partial tail too.
* ``K % 16 == 0`` so that BK * 4 bytes is a multiple of 32: the UB row pitch of
  the tile has to stay 32B aligned or the per-row VEC instructions are
  misaligned.  K in {64, 128} passes.
* G must already be fp32.  It is fp32 by the frozen contract (it is a
  log-domain quantity); the stages that consume q/k/v pass their dtype through
  a lookup table the way the L0 decode kernel does.  Nothing here is
  hardwired to fp16.
"""

import os
import sys

import torch
import tilelang
from tilelang import language as T

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import kda_l1_ref  # noqa: E402

# Only AUTO_SYNC, same as the six GDN kernels in the repo.  MEMORY_PLANNING is
# deliberately left off: it aliased a reduction target with a scratch tile in
# the backward dot kernel (right values in registers, all-zero stores).
pass_configs = {tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True}

VEC_NUM = 2


@tilelang.jit(out_idx=[-1], pass_configs=pass_configs)
def cumsum_ker(B, SEQ, HV, K, C, accum_dtype="float"):
    chunk_num = SEQ // C  # SEQ % C == 0 is asserted on the host
    BK = K // VEC_NUM  # channels per vector core

    @T.prim_func
    def main(
        G: T.Tensor([B, SEQ, HV, K], accum_dtype),  # type: ignore
        Gsum: T.Tensor([B, SEQ, HV, K], accum_dtype),  # type: ignore
    ):
        # grid = B * HV * chunk_num; one block owns one (batch, value head,
        # chunk).  The prefix chain is serial along tokens, so the two vector
        # cores split the channel axis -- the chains of different channels never
        # talk to each other.
        with T.Kernel(B * HV * chunk_num, is_npu=True) as (cid, vid):
            i_c = cid % chunk_num
            hv = (cid // chunk_num) % HV
            bz = cid // (chunk_num * HV)

            t0 = i_c * C  # first token of this chunk
            ko = vid * BK  # first channel of this vector core

            g_ub = T.alloc_ub([C, BK], accum_dtype)
            s_ub = T.alloc_ub([C, BK], accum_dtype)

            with T.Scope("V"):
                # Strided load: C rows of BK fp32, row pitch HV * K fp32.
                T.copy(G[bz, t0 : t0 + C, hv, ko : ko + BK], g_ub)

                # Inclusive prefix sum down the token axis, one vector add per
                # token.  Ascending only: a descending range() maps onto a
                # step-less T.serial and does not compile, and a descending
                # recurrence chain compiles but computes the wrong thing.
                for j in T.Parallel(BK):
                    s_ub[0, j] = g_ub[0, j]
                for i in range(1, C):
                    # Reads row i-1 of s_ub (already final) and row i of g_ub,
                    # writes row i of s_ub -- disjoint rows, so the serial token
                    # loop is the only ordering needed.  Two buffers rather than
                    # one in-place tile: this is the shape the old kernel was
                    # validated in.
                    for j in T.Parallel(BK):
                        s_ub[i, j] = s_ub[i - 1, j] + g_ub[i, j]

                # Strided store back into the same [B, SEQ, HV, K] window.
                T.copy(s_ub, Gsum[bz, t0 : t0 + C, hv, ko : ko + BK])

    return main


def chunk_cumsum(g, C=64):
    """Chunk-local inclusive prefix sum of g along the token axis.

    g is the external [B, SEQ, HV, K] fp32 log gate; the result has the same
    shape and layout and is what stages 2/4/5 consume as ``G``.
    """
    B, SEQ, HV, K = g.shape
    assert g.dtype == torch.float32, f"G is fp32 in the frozen contract (log-domain gate), got {g.dtype}"
    assert SEQ % C == 0, f"first pass needs SEQ % C == 0 (no tail block yet), got SEQ={SEQ} C={C}"
    assert K % (VEC_NUM * 8) == 0, f"K/{VEC_NUM} fp32 must be a multiple of 32B for the UB row pitch, got K={K}"

    # SEQ == 0 has to be caught here, not left to the assert above: 0 % C == 0
    # passes, chunk_num would come out 0, and the kernel would launch a
    # zero-block grid and hand back an allocated-but-never-written buffer.  A
    # zero-length sequence is legal input -- it is what a varlen batch with an
    # empty entry produces, and FLA's fused_recurrent returns early on it too.
    # Nothing to accumulate, so the result is empty along the token axis.
    if SEQ == 0:
        return torch.empty((B, 0, HV, K), device=g.device, dtype=torch.float32)

    return cumsum_ker(B, SEQ, HV, K, C)(g)


# ----------------------------------------------------------------------- test
def _relerr(got, ref):
    ref = ref.float()
    d = (got.float().cpu() - ref).abs().max().item()
    return d / max(ref.abs().max().item(), 1e-9)


def _case(B, SEQ, H, HV, K, V, C, gate, note=""):
    # Inputs are built on the host and the golden is taken on the host in fp32:
    # the reference pipeline is exact there, while an NPU einsum would drift the
    # two fp32 references apart by ~3e-4 (see kda_l1_ref.py).  Only g crosses to
    # the device, and only as a plain contiguous copy.
    q, k, v, g, beta, _ = kda_l1_ref.make_inputs(B, SEQ, H, HV, K, V, device="cpu", dtype=torch.float16, gate=gate)

    ref = kda_l1_ref.stage_tensors(q, k, v, g, beta, C=C)["G"]
    got = chunk_cumsum(g.npu(), C)

    e = _relerr(got, ref)
    ok = e < 1e-5 and bool(torch.isfinite(got.float()).all().item())
    print(f"  B{B} T{SEQ:<4d} H{H} HV{HV} K{K:<4d} C{C:<3d} {gate:8s} rel={e:.2e}  {'ok' if ok else 'FAIL'}  {note}")
    return ok


def main():
    tilelang.disable_cache()
    torch.manual_seed(0)

    ok = True
    print("== HV == H, both chunk lengths ==")
    ok &= _case(2, 256, 4, 4, 64, 64, 32, "normal")
    ok &= _case(2, 256, 4, 4, 64, 64, 64, "normal")
    ok &= _case(2, 256, 4, 4, 64, 64, 32, "forget")
    ok &= _case(2, 256, 4, 4, 64, 64, 64, "forget")

    print("== HV == 2H (GVA), both chunk lengths ==")
    ok &= _case(2, 128, 2, 4, 64, 64, 32, "normal")
    ok &= _case(2, 128, 2, 4, 64, 64, 64, "normal")
    ok &= _case(2, 128, 2, 4, 64, 64, 32, "forget")
    ok &= _case(2, 128, 2, 4, 64, 64, 64, "forget")

    print("== K3 spec K = V = 128 ==")
    ok &= _case(1, 128, 1, 1, 128, 128, 64, "forget", "HV=1, odd head count")
    ok &= _case(1, 256, 2, 4, 128, 128, 64, "normal", "GVA")

    print("== odd HV / magnitude stress ==")
    ok &= _case(1, 64, 1, 3, 64, 64, 32, "normal", "HV=3, needs the K split")
    ok &= _case(1, 128, 1, 1, 64, 64, 64, "extreme", "exponents die at once")

    if ok:
        print("Kernel Output Match!")
    else:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
