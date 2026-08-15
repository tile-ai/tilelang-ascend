"""KDA: the six chunkwise stages chained into one prefill forward pass.

Mirrors gdn_full.py: the stage kernels live in kda/, this driver sits one level
up and chains them.  It calls the six stages in order and checks the result
against two independent goldens:

    1. kda_chunk_ref.kda_chunk_ref  the chunkwise reference (same decomposition)
    2. kda_ref.kda_ref              the token-by-token recurrence

Agreement with (2) is the real acceptance criterion.  The two implementations
share no code path: one walks tokens one at a time carrying a [K, V] state, the
other cuts the sequence into chunks, builds a C x C decayed Gram matrix, inverts
a unit-lower-triangular block and propagates only per-chunk entry states.  If
they agree, the chunkwise algebra is right.

Stage order and what each one produces
--------------------------------------
    1  chunk_cumsum         G       chunk-local cumulative log gate
    2  chunk_scaled_dot_kkt L       strictLower(Diag(beta) . decayed K K^T)
    3  solve_tril           A       (I + L)^{-1}
    4  wy_fast              W, U    UT transform
    5  chunk_h              states, V', SF   per-chunk entry states + final state
    6  chunk_o              O       (scale.Q . e^G) states + Aqk V'

Note the kernels fold differently from the reference: kg is computed inside
chunk_h and qg / Aqk inside chunk_o, rather than being materialised by wy_fast.
That saves three GM round trips and is why wy_fast only emits W and U.

Known limitations of this first pass
------------------------------------
    * SEQ % C == 0 required.  Tail blocks and varlen / cu_seqlens are the next
      round -- the task spec says fixed length first.
    * No performance claim is made here; no msprof data has been collected.

Zero-length sequences
---------------------
SEQ == 0 is supported and is *not* covered by the SEQ % C == 0 rule above:
0 % C == 0, so a zero-length input would pass every guard, launch six
zero-block grids and return allocated-but-never-written memory.  Every host
wrapper therefore tests it explicitly and returns without touching the device.
The contract is that no token was consumed, so all token-axis outputs are empty
and the final state equals the initial state (zeros when none was supplied).
"""

import os
import sys

import torch
import tilelang  # noqa: F401  (imported for its torch_npu side effect)

# kda/ goes on sys.path before the stage modules are imported.  Each stage file
# has to be runnable on its own -- CI executes every .py in the example tree as
# a standalone script -- so they import the reference layer flat, as
# ``import kda_chunk_ref``.  Putting kda/ on the path here means this file picks
# up the *same* module object they do; importing it as ``kda.kda_chunk_ref``
# instead would create a second, independent copy of the reference layer.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "kda"))

from kda.kda_chunk_cumsum import chunk_cumsum  # noqa: E402
from kda.kda_chunk_scaled_dot_kkt import chunk_scaled_dot_kkt  # noqa: E402
from kda.kda_solve_tril import kda_solve_tril  # noqa: E402
from kda.kda_wy_fast import wy_fast  # noqa: E402
from kda.kda_chunk_h import chunk_h  # noqa: E402
from kda.kda_chunk_o import chunk_o  # noqa: E402

import kda_chunk_ref as R  # noqa: E402
import kda_ref as _L0  # noqa: E402


# ----------------------------------------------------------------- pipeline
def kda_chunk_fwd(q, k, v, g, beta, C=64, BC=16, scale=None, initial_state=None, output_final_state=False):
    """Chunkwise KDA forward on the NPU.  Interface matches kda_ref.kda_ref.

    All tensors are the frozen external layout:
        q, k [B,SEQ,H,K]   v [B,SEQ,HV,V]   g [B,SEQ,HV,K] fp32
        beta [B,SEQ,HV]    initial_state [B,HV,K,V] fp32
    """
    B, SEQ, H, K = q.shape
    HV = v.shape[2]
    assert HV % H == 0, "HV must be divisible by H (GVA)"
    assert SEQ % C == 0, f"first pass needs SEQ % C == 0, got SEQ={SEQ} C={C}"
    if scale is None:
        scale = K**-0.5

    # Contiguity is asserted, not repaired.  Slicing the token axis (q[:, :cut],
    # the natural way to feed a prefix) leaves stride[0] at the *original*
    # T*H*K, so the result is a non-contiguous view and the jit wrapper rejects
    # it with "Input tensor at index 0 must be contiguous".  Calling
    # .contiguous() here would work, but it is a full copy of every input on the
    # host -- exactly the kind of hidden cost the acceptance gate forbids
    # ("no large host-side state transforms masking kernel cost").  Make the
    # caller pay for it visibly instead.
    for name, t in (("q", q), ("k", k), ("v", v), ("g", g), ("beta", beta)):
        assert t.is_contiguous(), (
            f"{name} must be contiguous; a token-axis slice is not. "
            f"Call .contiguous() at the call site -- this wrapper will not copy "
            f"for you, so the cost stays visible."
        )

    # A zero-length sequence is legal input: it is what a varlen batch with an
    # empty entry produces, and FLA's fused_recurrent returns early on it.  It
    # must be handled explicitly rather than left to the assert above, because
    # 0 % C == 0 passes -- every one of the six kernels would then launch a
    # zero-block grid and the output would be allocated but never written, so
    # the caller would silently receive uninitialised device memory.  No token
    # is consumed, so the state passes through untouched: the final state IS
    # the initial state.  Each of the six stage wrappers carries the same guard
    # independently, since each is also a public entry point.
    if SEQ == 0:
        V = v.shape[-1]
        o = torch.empty((B, 0, HV, V), device=v.device, dtype=v.dtype)
        if not output_final_state:
            return o, None
        if initial_state is not None:
            sf = initial_state.float().clone()
        else:
            sf = torch.zeros((B, HV, K, V), device=v.device, dtype=torch.float32)
        return o, sf

    G = chunk_cumsum(g.float(), C=C)  # stage 1
    L = chunk_scaled_dot_kkt(k, G, beta, C=C)  # stage 2
    A = kda_solve_tril(L)  # stage 3
    W, U = wy_fast(k, v, beta, G, A, C)  # stage 4
    states, Vnew, SF = chunk_h(k, W, U, G, C=C, initial_state=initial_state)  # stage 5
    O = chunk_o(q, k, Vnew, states, G, C=C, BC=BC, scale=scale)  # stage 6

    return O, (SF if output_final_state else None)


# -------------------------------------------------------------------- tests
def _mk(B, SEQ, H, HV, K, V, gate, dtype, with_state=False):
    """Inputs on CPU (goldens must not run on the NPU -- see note in main)."""
    return _L0.make_inputs(B, SEQ, H, HV, K, V, device="cpu", dtype=dtype, gate=gate, seed=0, with_state=with_state)


def _rel(x, ref):
    x, ref = x.float().cpu(), ref.float().cpu()
    return (x - ref).abs().max().item() / max(ref.abs().max().item(), 1e-9)


def _tol(dtype):
    return 3e-2 if dtype is torch.bfloat16 else 5e-3


def _case(B, SEQ, H, HV, K, V, C, gate, dtype, with_state=False, note=""):
    q, k, v, g, beta, s0 = _mk(B, SEQ, H, HV, K, V, gate, dtype, with_state)

    # two CPU goldens
    ref_l0, _ = _L0.kda_ref(q, k, v, g, beta, initial_state=s0)
    ref_ch, _ = R.kda_chunk_ref(q, k, v, g, beta, C=C, initial_state=s0)

    got, _ = kda_chunk_fwd(
        q.npu(), k.npu(), v.npu(), g.npu().float(), beta.npu(), C=C, initial_state=None if s0 is None else s0.npu().float()
    )

    e0, ec = _rel(got, ref_l0), _rel(got, ref_ch)
    # how far the two goldens are from each other: the floor this kernel can hit
    floor = _rel(ref_ch, ref_l0)
    finite = bool(torch.isfinite(got.float()).all())
    tol = _tol(dtype)
    ok = finite and e0 < tol and ec < tol

    tag = "bf16" if dtype is torch.bfloat16 else ("fp32" if dtype is torch.float32 else "fp16")
    print(
        f"  B{B} T{SEQ:<4d} H{H} HV{HV} K{K:<3d} V{V:<3d} C{C:<2d} {tag} {gate:8s} "
        f"S0={'Y' if with_state else 'N'}  vsL0={e0:.2e} vsChunk={ec:.2e} "
        f"floor={floor:.2e} finite={'Y' if finite else 'N'}  "
        f"{'ok' if ok else 'FAIL'}  {note}"
    )
    return ok


def test_vs_both_goldens():
    print("== L1 kernel pipeline  vs  L0 recurrence  and  chunkwise reference ==")
    ok = True
    for gate in ("normal", "forget"):
        ok &= _case(1, 128, 1, 1, 64, 64, 64, gate, torch.float16)
        ok &= _case(2, 128, 2, 2, 64, 64, 64, gate, torch.float16)
        ok &= _case(2, 256, 2, 4, 64, 64, 32, gate, torch.float16, note="GVA HV=2H")
    print("  -- batch and chunk-count edges --")
    # B=4 is named in the test matrix; every other case here runs B in {1, 2}.
    ok &= _case(4, 128, 1, 1, 64, 64, 64, "normal", torch.float16, note="B=4")
    # SEQ == C is the single-chunk path: the chunk loop runs exactly once, which
    # is where an off-by-one in the cross-chunk carry would hide.
    ok &= _case(1, 64, 1, 1, 64, 64, 64, "normal", torch.float16, note="SEQ == C, one chunk")
    ok &= _case(2, 64, 2, 4, 64, 64, 64, "forget", torch.float16, note="SEQ == C + GVA")
    print("  -- gate extremes (the NaN traps) --")
    ok &= _case(1, 128, 1, 1, 64, 64, 64, "keep", torch.float16, note="alpha->1")
    ok &= _case(1, 128, 1, 2, 64, 64, 64, "extreme", torch.float16, note="state dies at once")
    print("  -- K3 spec --")
    ok &= _case(1, 256, 1, 1, 128, 128, 64, "forget", torch.float16, note="K=V=128")
    ok &= _case(1, 128, 2, 4, 128, 128, 64, "normal", torch.float16, note="K3 + GVA")
    print("  -- K != V --")
    ok &= _case(1, 128, 1, 1, 64, 128, 64, "normal", torch.float16)
    print("  -- dtype passthrough --")
    ok &= _case(2, 128, 2, 4, 64, 64, 64, "normal", torch.bfloat16)
    ok &= _case(1, 128, 1, 1, 128, 128, 64, "forget", torch.bfloat16)
    print("  -- non-zero initial state --")
    ok &= _case(2, 128, 2, 2, 64, 64, 64, "normal", torch.float16, with_state=True)
    ok &= _case(1, 256, 1, 1, 128, 128, 64, "forget", torch.float16, with_state=True)
    return ok


def test_state_relay():
    """Whole sequence in one call vs two calls relaying final_state.

    One shot verifies the initial_state entry, the final_state exit and the
    cross-chunk carry all at once -- the highest value-per-line test there is.
    """
    print("== whole sequence  vs  two-segment relay through final_state ==")
    ok = True
    for SEQ, cut, C, HV, gate in (
        (128, 64, 64, 2, "normal"),
        (256, 128, 64, 2, "forget"),
        (256, 128, 32, 4, "normal"),
        (128, 64, 64, 1, "extreme"),
    ):
        q, k, v, g, beta, _ = _mk(2, SEQ, 1, HV, 64, 64, gate, torch.float16)
        qa, ka, va, ga, ba = (x.npu() for x in (q, k, v, g.float(), beta))

        # .contiguous() on every slice: a token-axis slice keeps the original
        # stride[0] and is therefore a non-contiguous view.  The wrapper asserts
        # rather than repairing this, so the copy is explicit here.
        # cut is bound as a default so the closure captures this iteration's
        # value, not the loop variable (ruff B023).  Both are called in this
        # same iteration, so behaviour is unchanged.
        def pre(x, cut=cut):
            return x[:, :cut].contiguous()

        def post(x, cut=cut):
            return x[:, cut:].contiguous()

        whole, _ = kda_chunk_fwd(qa, ka, va, ga, ba, C=C)
        a, sa = kda_chunk_fwd(pre(qa), pre(ka), pre(va), pre(ga), pre(ba), C=C, output_final_state=True)
        b, _ = kda_chunk_fwd(post(qa), post(ka), post(va), post(ga), post(ba), C=C, initial_state=sa)
        seg = torch.cat([a, b], dim=1)
        e = _rel(seg, whole)
        # Exact equality, not a tolerance.  Every cut here is a chunk boundary,
        # and chunks are independent given their entry state, so the segmented
        # run performs literally the same arithmetic in the same order as the
        # one-shot run -- any difference at all would mean the entry state did
        # not survive the round trip through final_state / initial_state.
        # Asserting the tolerance instead would let a real regression through:
        # a drift to 4e-3 would still print "ok" while the README claims the
        # result is bit-identical.
        good = e == 0.0
        ok &= good
        print(f"  T={SEQ} cut={cut} C={C} HV={HV} {gate:8s} rel={e:.2e}  {'ok (bit-identical)' if good else 'FAIL'}")
    return ok


def test_zero_state_equals_none():
    """Passing an all-zero initial_state must equal passing none at all."""
    print("== zero initial_state  vs  no initial_state ==")
    ok = True
    for SEQ, C, HV in ((128, 64, 2), (256, 32, 4)):
        q, k, v, g, beta, _ = _mk(1, SEQ, 1, HV, 64, 64, "normal", torch.float16)
        qa, ka, va, ga, ba = (x.npu() for x in (q, k, v, g.float(), beta))
        z = torch.zeros((1, HV, 64, 64), device="npu", dtype=torch.float32)
        a, _ = kda_chunk_fwd(qa, ka, va, ga, ba, C=C)
        b, _ = kda_chunk_fwd(qa, ka, va, ga, ba, C=C, initial_state=z)
        e = (a.float() - b.float()).abs().max().item()
        good = e == 0.0
        ok &= good
        print(f"  T={SEQ} C={C} HV={HV}  max|diff|={e:.1e}  {'ok (bit-identical)' if good else 'FAIL'}")
    return ok


def test_empty_sequence():
    """T == 0 must be accepted, must launch nothing, and must pass the state through.

    A zero-length sequence is what a varlen batch with an empty entry produces.
    It is the one degenerate shape that slips through ``SEQ % C == 0`` (0 % C is
    0), so without an explicit guard every stage would launch a zero-block grid
    and the caller would get uninitialised device memory back instead of a loud
    failure.  Both levels are checked here: the pipeline entry point, and each
    of the six stage wrappers called directly, since each is also public.
    """
    print("== zero-length sequence (T = 0) ==")
    ok = True
    B, H, HV, K, V, C = 2, 1, 2, 64, 64, 64
    dt = torch.float16

    def _e(*shape, dtype=dt):
        return torch.empty(shape, device="npu", dtype=dtype)

    q, k, v = _e(B, 0, H, K), _e(B, 0, H, K), _e(B, 0, HV, V)
    g, beta = _e(B, 0, HV, K, dtype=torch.float32), _e(B, 0, HV)

    o, sf = kda_chunk_fwd(q, k, v, g, beta, C=C, output_final_state=True)
    m = float(sf.abs().max())
    good = tuple(o.shape) == (B, 0, HV, V) and tuple(sf.shape) == (B, HV, K, V) and m == 0.0
    ok &= good
    print(f"  pipeline S0=none    o{tuple(o.shape)} sf{tuple(sf.shape)} max|sf|={m:.1e}  {'ok' if good else 'FAIL'}")

    # A non-zero initial state must come back bit-identical: no token was
    # consumed, so nothing may decay it and nothing may be written into it.
    s0 = torch.randn((B, HV, K, V), device="npu", dtype=torch.float32)
    o, sf = kda_chunk_fwd(q, k, v, g, beta, C=C, initial_state=s0, output_final_state=True)
    d = float((sf - s0).abs().max())
    good = tuple(o.shape) == (B, 0, HV, V) and d == 0.0
    ok &= good
    print(f"  pipeline S0=random  max|sf-S0|={d:.1e}  {'ok (bit-identical passthrough)' if good else 'FAIL'}")

    # and it must be a copy: a caller relaying it into the next segment must not
    # be able to mutate its own input through the returned handle.
    sf.zero_()
    good = float(s0.abs().max()) > 0.0
    ok &= good
    print(f"  final_state is a copy, not an alias of initial_state  {'ok' if good else 'FAIL'}")

    # the CPU reference must carry the identical contract
    qc, kc, vc, gc, bc, s0c = (x.cpu() for x in (q, k, v, g, beta, s0))
    ro, rs = R.kda_chunk_ref(qc, kc, vc, gc, bc, C=C, initial_state=s0c, output_final_state=True)
    d = float((rs - s0c).abs().max())
    good = tuple(ro.shape) == (B, 0, HV, V) and d == 0.0
    ok &= good
    print(f"  reference agrees    o{tuple(ro.shape)} max|sf-S0|={d:.1e}  {'ok' if good else 'FAIL'}")

    # every stage wrapper, called directly
    G = chunk_cumsum(g, C=C)
    L = chunk_scaled_dot_kkt(k, G, beta, C=C)
    A = kda_solve_tril(L)
    W, U = wy_fast(k, v, beta, G, A, C)
    states, Vnew, SF = chunk_h(k, W, U, G, C=C, initial_state=s0)
    O = chunk_o(q, k, Vnew, states, G, C=C)
    for name, got_s, want_s in (
        ("1 cumsum G", tuple(G.shape), (B, 0, HV, K)),
        ("2 kkt L", tuple(L.shape), (B, 0, HV, C)),
        ("3 solve A", tuple(A.shape), (B, 0, HV, C)),
        ("4 wy W", tuple(W.shape), (B, 0, HV, K)),
        ("4 wy U", tuple(U.shape), (B, 0, HV, V)),
        ("5 h states", tuple(states.shape), (B, HV, 0, K, V)),
        ("5 h Vnew", tuple(Vnew.shape), (B, 0, HV, V)),
        ("5 h SF", tuple(SF.shape), (B, HV, K, V)),
        ("6 o O", tuple(O.shape), (B, 0, HV, V)),
    ):
        good = got_s == want_s
        ok &= good
        print(f"  stage {name:11s} {str(got_s):20s} want {str(want_s):20s}  {'ok' if good else 'FAIL'}")

    d = float((SF - s0).abs().max())
    good = d == 0.0
    ok &= good
    print(f"  stage 5 passes the state through  max|SF-S0|={d:.1e}  {'ok' if good else 'FAIL'}")
    return ok


def main():
    tilelang.disable_cache()
    torch.manual_seed(0)

    # Goldens run on CPU on purpose: an NPU einsum dispatches to matmul with
    # reduced-precision accumulation and drifts two exact fp32 references by
    # ~3e-4, which is the same order as the quantity being measured.
    ok = True
    ok &= test_vs_both_goldens()
    print()
    ok &= test_state_relay()
    print()
    ok &= test_zero_state_equals_none()
    print()
    ok &= test_empty_sequence()
    print()

    if ok:
        print("Kernel Output Match!")
    else:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
