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

Ragged tail
-----------
SEQ need not be a multiple of C.  Every stage runs a ceildiv(SEQ, C) grid and
the last chunk simply moves fewer rows: compute_valid_extent (src/op/ascend.cc)
clamps validRow on each GM transfer to SEQ - t0.  No tile shape changes, so
every alignment constraint is untouched, and nothing is padded on the host --
which is what the acceptance gate rules out.

Three things the framework does not cover, handled per stage:
  * single-row reads put the token axis on a unit-extent dim, which
    find_active_dim_indices never bounds-checks (stages 2, 5, 6);
  * UB tail rows reach exp() and then the cube as full-width operands, so they
    are zero-filled first (stages 1-6);
  * chunk_h must read the chunk decay at the last *valid* token, not at row
    C - 1 -- the one genuine off-by-one of the change.

Varlen (cu_seqlens)
-------------------
A flattened varlen batch is supported.  The convention is FlashAttention's, the
one FLA follows: B == 1, every sequence concatenated onto the token axis, and
cu_seqlens[i] .. cu_seqlens[i+1] delimiting sequence i.  Chunking restarts at
every sequence start, so a sequence's last chunk is ragged in the MIDDLE of the
flattened tensor -- which is what makes varlen more than "the ragged tail again":
compute_valid_extent clamps against the end of the whole tensor, not against eos,
so nothing bounds an interior chunk for us.  Every stage takes its valid row
count from per-chunk metadata instead (kda/kda_varlen.py) and passes it as a
run-time copy extent.

Two shapes change under varlen, and only two -- the ones not indexed by token:

    initial_state / final_state   [N, HV, K, V]            one per SEQUENCE
    states (stage 5 -> stage 6)   [1, HV, NT_TOTAL, K, V]  chunk axis spans the
                                                           whole batch

Everything on the token axis keeps its [1, T_total, HV, *] layout, because a
flattened varlen batch already IS that layout.

An empty sequence (T_i == 0) is legal and needs no special case in five of the
six stages: it contributes zero chunks, so no block is ever created for it.
Stage 5 is the exception -- its grid is per sequence -- and there it falls out
of a zero-trip chunk loop, with the final state passing through untouched.

Known limitations of this pass
------------------------------
    * No performance claim is made here; no msprof data has been collected.

Zero-length sequences
---------------------
SEQ == 0 is supported, and it is worth calling out separately from the ragged
tail above: ceildiv(0, C) is 0, so a zero-length input passes every guard and
would launch six zero-block grids, returning allocated-but-never-written memory.  Every host
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
_K = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kda")
sys.path.insert(0, _K)

from kda.kda_chunk_cumsum import chunk_cumsum  # noqa: E402
from kda.kda_chunk_scaled_dot_kkt import chunk_scaled_dot_kkt  # noqa: E402
from kda.kda_solve_tril import kda_solve_tril  # noqa: E402
from kda.kda_wy_fast import wy_fast  # noqa: E402
from kda.kda_chunk_h import chunk_h  # noqa: E402
from kda.kda_chunk_o import chunk_o  # noqa: E402

import kda_chunk_ref as R  # noqa: E402
import kda_ref as _L0  # noqa: E402


# ----------------------------------------------------------------- pipeline
def kda_chunk_fwd(
    q, k, v, g, beta, C=64, BC=16, BV=None, scale=None, initial_state=None, output_final_state=False, cu_seqlens=None, route_b=False
):
    """Chunkwise KDA forward on the NPU.  Interface matches kda_ref.kda_ref.

    All tensors are the frozen external layout:
        q, k [B,SEQ,H,K]   v [B,SEQ,HV,V]   g [B,SEQ,HV,K] fp32
        beta [B,SEQ,HV]    initial_state [B,HV,K,V] fp32

    With cu_seqlens the batch is flattened (B == 1) and initial_state /
    final_state are [N, HV, K, V], one per sequence.
    """
    B, SEQ, H, K = q.shape
    HV = v.shape[2]
    assert HV % H == 0, "HV must be divisible by H (GVA)"
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
    # empty entry produces, and FLA's fused_recurrent returns early on it.  The
    # ceil grid does not save us here -- ceildiv(0, C) is 0, so every one of the
    # six kernels would launch a
    # zero-block grid and the output would be allocated but never written, so
    # the caller would silently receive uninitialised device memory.  No token
    # is consumed, so the state passes through untouched: the final state IS
    # the initial state.  Each of the six stage wrappers carries the same guard
    # independently, since each is also a public entry point.
    #
    # Under varlen this fires only when the WHOLE batch is empty.  A single empty
    # sequence inside a non-empty batch is handled inside the stages.
    if SEQ == 0:
        V = v.shape[-1]
        n_lead = B if cu_seqlens is None else (cu_seqlens.numel() - 1)
        o = torch.empty((B, 0, HV, V), device=v.device, dtype=v.dtype)
        if not output_final_state:
            return o, None
        if initial_state is not None:
            sf = initial_state.float().clone()
        else:
            sf = torch.zeros((n_lead, HV, K, V), device=v.device, dtype=torch.float32)
        return o, sf

    # cu_seqlens threads through all six stages unchanged.  Each wrapper
    # rebuilds the per-chunk metadata from it rather than being handed a
    # prebuilt table: one source of truth, and no way for two stages to disagree
    # about where a chunk starts.  The cost is six O(N) device-to-host reads of
    # cu_seqlens per forward -- the grid extent is a trace-time Python int, so
    # the boundaries have to reach the host somehow.  FLA solves the same
    # problem by carrying a separate cu_seqlens_cpu the whole way down; passing
    # cu_seqlens already on the CPU here has the same effect.
    G = chunk_cumsum(g.float(), C=C, cu_seqlens=cu_seqlens)  # stage 1
    # BC is stage 2's anchor width and it sets how much stays on the vector
    # unit: the diagonal blocks are BC wide, everything to their left is a
    # cube matmul.  Halving BC halves the vector arithmetic and adds more,
    # smaller strips -- which the cube has room for at 2.4% MAC occupancy.
    # route_b puts stage 2's diagonal blocks on the cube.  Off by default; see
    # chunk_scaled_dot_kkt for what it costs and when it is safe to ask for.
    L = chunk_scaled_dot_kkt(k, G, beta, C=C, BC=BC, cu_seqlens=cu_seqlens, route_b=route_b)  # stage 2
    A = kda_solve_tril(L, cu_seqlens=cu_seqlens)  # stage 3
    W, U = wy_fast(k, v, beta, G, A, C, cu_seqlens=cu_seqlens)  # stage 4
    # BV shards the state along V and IS the whole parallel decomposition of
    # stage 5: its grid is B * HV * (V // BV).  At the default BV = min(V, 64)
    # and B=1, HV=4, V=128 that is EIGHT blocks on a 20-core part, so twelve
    # cores sit idle for the whole stage.  Exposed here so the caller can pick.
    states, Vnew, SF = chunk_h(k, W, U, G, C=C, BV=BV, initial_state=initial_state, cu_seqlens=cu_seqlens)  # stage 5
    O = chunk_o(q, k, Vnew, states, G, C=C, BC=BC, scale=scale, cu_seqlens=cu_seqlens)  # stage 6

    return O, (SF if output_final_state else None)


# -------------------------------------------------------------------- tests
def _mk(B, SEQ, H, HV, K, V, gate, dtype, with_state=False):
    """Inputs on CPU (goldens must not run on the NPU -- see note in main)."""
    return _L0.make_inputs(B, SEQ, H, HV, K, V, device="cpu", dtype=dtype, gate=gate, seed=0, with_state=with_state)


def _rel(x, ref):
    x, ref = x.float().cpu(), ref.float().cpu()
    # A varlen batch in which every sequence is empty legitimately produces
    # zero-element outputs, and .max() raises on those rather than returning 0.
    if x.numel() == 0 and ref.numel() == 0:
        return 0.0
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
    print("  -- ragged tail (SEQ % C != 0) --")
    ok &= _case(2, 70, 1, 2, 64, 64, 64, "normal", torch.float16, note="70 = 64 + 6")
    ok &= _case(1, 33, 1, 1, 64, 64, 32, "forget", torch.float16, note="33 = 32 + 1, one valid tail row")
    ok &= _case(1, 65, 1, 1, 128, 128, 64, "forget", torch.float16, note="K3 dim, one valid tail row")
    ok &= _case(2, 100, 2, 4, 64, 64, 32, "extreme", torch.float16, note="GVA + extreme gate on the tail")
    ok &= _case(1, 96, 1, 1, 64, 64, 64, "normal", torch.float16, note="R = 32, exact core boundary")
    ok &= _case(2, 130, 2, 2, 64, 64, 64, "forget", torch.float16, with_state=True, note="tail + non-zero initial state")

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


def _mkv(seqlens, H, HV, K, V, gate, dtype, with_state=False):
    return _L0.make_varlen_inputs(seqlens, H, HV, K, V, device="cpu", dtype=dtype, gate=gate, seed=0, with_state=with_state)


def _vcase(seqlens, H, HV, K, V, C, gate, dtype, with_state=False, note=""):
    """One varlen batch of the full pipeline against BOTH goldens.

    Compared over the whole flat token axis, never per sequence: a chunk that
    writes past its own eos lands finite, plausible values on the NEXT
    sequence's tokens, so a per-sequence comparison would pass on a corrupt
    batch.  The empty sequences additionally have to pass their state through
    bit for bit -- no token was consumed, so final_state IS initial_state.
    """
    q, k, v, g, beta, s0, cu = _mkv(seqlens, H, HV, K, V, gate, dtype, with_state)

    ref_l0, sf_l0 = _L0.kda_ref(q, k, v, g, beta, initial_state=s0, output_final_state=True, cu_seqlens=cu)
    ref_ch, _ = R.kda_chunk_ref(q, k, v, g, beta, C=C, initial_state=s0, output_final_state=True, cu_seqlens=cu)

    got, sf = kda_chunk_fwd(
        q.npu(),
        k.npu(),
        v.npu(),
        g.npu().float(),
        beta.npu(),
        C=C,
        initial_state=None if s0 is None else s0.npu().float(),
        output_final_state=True,
        cu_seqlens=cu.npu(),
    )

    e0, ec = _rel(got, ref_l0), _rel(got, ref_ch)
    ef = _rel(sf, sf_l0)
    finite = bool(torch.isfinite(got.float()).all())
    N, T_total = len(seqlens), int(sum(seqlens))
    shape_ok = tuple(got.shape) == (1, T_total, HV, V) and tuple(sf.shape) == (N, HV, K, V)

    passthru = True
    if with_state:
        for i, n in enumerate(seqlens):
            if n == 0:
                passthru &= bool(torch.equal(sf[i].cpu(), s0[i].float().cpu()))

    tol = _tol(dtype)
    ok = finite and shape_ok and passthru and e0 < tol and ec < tol and ef < tol
    tag = "bf16" if dtype is torch.bfloat16 else "fp16"
    print(
        f"  {str(seqlens):22s} HV{HV} K{K:<4d} C{C:<2d} {tag} {gate:8s} S0={'Y' if with_state else 'N'} "
        f"vsL0={e0:.2e} vsChunk={ec:.2e} SF={ef:.2e} pass={'Y' if passthru else 'N'}  {'ok' if ok else 'FAIL'}  {note}"
    )
    return ok


def test_varlen_vs_both_goldens():
    print("== varlen pipeline  vs  L0 recurrence  and  chunkwise reference ==")
    ok = True
    ok &= _vcase([64, 64, 64], 1, 2, 64, 64, 64, "normal", torch.float16, note="equal, chunk-aligned")
    ok &= _vcase([70, 33, 129], 1, 2, 64, 64, 64, "normal", torch.float16, note="every sequence ragged")
    ok &= _vcase([70, 33, 129], 1, 2, 64, 64, 64, "forget", torch.float16, True, "ragged + per-sequence S0")
    ok &= _vcase([1, 200], 1, 2, 64, 64, 64, "forget", torch.float16, note="one token, then a long sequence")
    ok &= _vcase([20, 20], 1, 2, 64, 64, 64, "normal", torch.float16, note="both shorter than C/2")
    ok &= _vcase([5], 1, 1, 64, 64, 64, "normal", torch.float16, note="N = 1, shorter than a chunk")
    ok &= _vcase([256], 1, 1, 64, 64, 64, "normal", torch.float16, note="N = 1, several chunks")
    print("  -- empty sequences --")
    ok &= _vcase([70, 0, 129], 1, 2, 64, 64, 64, "forget", torch.float16, True, "empty in the middle")
    ok &= _vcase([0, 70], 1, 2, 64, 64, 64, "normal", torch.float16, True, "empty first")
    ok &= _vcase([70, 0], 1, 2, 64, 64, 64, "normal", torch.float16, True, "empty last")
    ok &= _vcase([0, 0], 1, 2, 64, 64, 64, "normal", torch.float16, True, "every sequence empty")
    ok &= _vcase([0, 70, 0, 33, 0], 1, 2, 64, 64, 64, "forget", torch.float16, True, "empties interleaved")
    print("  -- gate extremes, GVA, K3, dtypes --")
    ok &= _vcase([70, 33], 1, 2, 64, 64, 64, "extreme", torch.float16, note="extreme gate on partial blocks")
    ok &= _vcase([100, 28], 2, 4, 64, 64, 32, "extreme", torch.float16, note="GVA HV=2H + C = 32")
    ok &= _vcase([65, 65], 1, 1, 128, 128, 64, "forget", torch.float16, True, "K3 spec K=V=128")
    ok &= _vcase([128, 64], 2, 4, 128, 128, 64, "normal", torch.float16, note="K3 + GVA")
    ok &= _vcase([70, 33], 2, 4, 64, 64, 64, "forget", torch.bfloat16, True, "bf16 + GVA")
    ok &= _vcase([70, 33, 129], 1, 2, 64, 64, 64, "keep", torch.float16, note="alpha -> 1")
    return ok


def test_varlen_equals_per_sequence_calls():
    """★ The acceptance test for varlen, and the only one that is not a tolerance.

    A varlen batch must produce EXACTLY what the same sequences produce when run
    one at a time through the fixed-length path -- bit for bit, not within a
    tolerance.  Nothing approximate is involved: chunking restarts at every
    sequence start and no state crosses a boundary, so the varlen run performs
    literally the same arithmetic in the same order on the same numbers.  Any
    difference at all means a chunk read or wrote outside its own sequence.

    A tolerance here would be worse than no test: the corruption this is built
    to catch -- an unbounded tile copy spilling into the neighbour -- produces
    finite, plausible, small-looking differences.
    """
    print("== varlen batch  vs  the same sequences run one at a time ==")
    ok = True
    cases = [
        ([64, 64, 64], 64, False, "equal, chunk-aligned"),
        ([70, 33, 129], 64, False, "every sequence ragged"),
        ([70, 33, 129], 64, True, "ragged + per-sequence S0"),
        ([70, 0, 129], 64, True, "empty in the middle"),
        ([1, 200], 64, False, "one token, then a long sequence"),
        ([100, 28], 32, False, "C = 32"),
        ([20, 20, 20], 64, True, "all shorter than C/2"),
    ]
    for seqlens, C, ws, note in cases:
        q, k, v, g, beta, s0, cu = _mkv(seqlens, 1, 2, 64, 64, "forget", torch.float16, ws)
        qa, ka, va, ga, ba = (x.npu() for x in (q, k, v, g.float(), beta))
        sa = None if s0 is None else s0.npu().float()

        o_v, sf_v = kda_chunk_fwd(qa, ka, va, ga, ba, C=C, initial_state=sa, output_final_state=True, cu_seqlens=cu.npu())

        outs, finals = [], []
        pos = 0
        for i, n in enumerate(seqlens):
            sl = slice(pos, pos + n)
            pos += n
            s_i = None if sa is None else sa[i : i + 1]
            # .contiguous() on the slices: with B == 1 they are already
            # contiguous views, so this is a no-op the wrapper's assert accepts.
            o_i, sf_i = kda_chunk_fwd(
                qa[:, sl].contiguous(),
                ka[:, sl].contiguous(),
                va[:, sl].contiguous(),
                ga[:, sl].contiguous(),
                ba[:, sl].contiguous(),
                C=C,
                initial_state=s_i,
                output_final_state=True,
            )
            outs.append(o_i)
            finals.append(sf_i)

        o_s = torch.cat(outs, dim=1)
        sf_s = torch.cat(finals, dim=0)
        d_o = (o_v.float() - o_s.float()).abs().max().item()
        d_s = (sf_v.float() - sf_s.float()).abs().max().item()
        good = d_o == 0.0 and d_s == 0.0
        ok &= good
        print(
            f"  {str(seqlens):22s} C={C:<2d} S0={'Y' if ws else 'N'}  |dO|={d_o:.1e} |dSF|={d_s:.1e}  {'ok (bit-identical)' if good else 'FAIL'}  {note}"
        )
    return ok


def test_varlen_equals_fixed_batch():
    """N equal-length sequences under varlen must equal a B = N fixed batch.

    Exercises the flatten/split mapping and the [N, HV, K, V] state indexing --
    the parts the per-sequence test above cannot see, because it splits the same
    way varlen does.  Exact equality again, and for the same reason.
    """
    print("== varlen (N equal sequences)  vs  fixed-length B = N batch ==")
    ok = True
    for seqlens, C in (([64, 64, 64], 64), ([70, 70], 64), ([128, 128], 32)):
        N, Lq = len(seqlens), seqlens[0]
        q, k, v, g, beta, s0, cu = _mkv(seqlens, 1, 2, 64, 64, "normal", torch.float16, True)
        qa, ka, va, ga, ba = (x.npu() for x in (q, k, v, g.float(), beta))
        sa = s0.npu().float()

        o_v, sf_v = kda_chunk_fwd(qa, ka, va, ga, ba, C=C, initial_state=sa, output_final_state=True, cu_seqlens=cu.npu())

        def rs(x, N=N, Lq=Lq):
            return x.reshape(N, Lq, *x.shape[2:]).contiguous()

        o_b, sf_b = kda_chunk_fwd(rs(qa), rs(ka), rs(va), rs(ga), rs(ba), C=C, initial_state=sa, output_final_state=True)
        d_o = (o_v.reshape(N, Lq, 2, 64).float() - o_b.float()).abs().max().item()
        d_s = (sf_v.float() - sf_b.float()).abs().max().item()
        good = d_o == 0.0 and d_s == 0.0
        ok &= good
        print(f"  {str(seqlens):22s} C={C:<2d}  |dO|={d_o:.1e} |dSF|={d_s:.1e}  {'ok (bit-identical)' if good else 'FAIL'}")
    return ok


def test_varlen_state_relay():
    """Relaying final_state across a call boundary must still work under varlen.

    Cut every sequence at a chunk boundary, run the two halves as two varlen
    batches, and relay the [N, HV, K, V] state between them.  Exact, for the
    same reason as the fixed-length relay test: the cut is a chunk boundary and
    chunks are independent given their entry state.
    """
    print("== varlen whole  vs  two-segment relay through final_state ==")
    ok = True
    for seqlens, cut, C in (([128, 128], 64, 64), ([192, 64], 64, 64), ([128, 256], 128, 64)):
        q, k, v, g, beta, _, cu = _mkv(seqlens, 1, 2, 64, 64, "forget", torch.float16)
        qa, ka, va, ga, ba = (x.npu() for x in (q, k, v, g.float(), beta))

        whole, _ = kda_chunk_fwd(qa, ka, va, ga, ba, C=C, cu_seqlens=cu.npu())

        # Split every sequence at `cut`; both halves are themselves varlen
        # batches with their own cu_seqlens.
        # seqlens and cut bound as defaults rather than captured: a closure
        # over a loop variable reads its LAST value, which happens to be
        # harmless here only because every call is in the same iteration.
        def halves(x, first, seqlens=seqlens, cut=cut):
            parts, pos = [], 0
            for n in seqlens:
                sl = slice(pos, pos + cut) if first else slice(pos + cut, pos + n)
                pos += n
                parts.append(x[:, sl])
            return torch.cat(parts, dim=1).contiguous()

        cu_a = torch.tensor([0] + [cut * (i + 1) for i in range(len(seqlens))], dtype=torch.int32).npu()
        tail = [n - cut for n in seqlens]
        cu_b = torch.tensor([0] + [sum(tail[: i + 1]) for i in range(len(tail))], dtype=torch.int32).npu()

        a, sa = kda_chunk_fwd(*(halves(x, True) for x in (qa, ka, va, ga, ba)), C=C, output_final_state=True, cu_seqlens=cu_a)
        b, _ = kda_chunk_fwd(*(halves(x, False) for x in (qa, ka, va, ga, ba)), C=C, initial_state=sa, cu_seqlens=cu_b)

        # Reassemble into flattened order to compare against the whole run.
        seg, pos_a, pos_b = [], 0, 0
        for n in seqlens:
            seg.append(a[:, pos_a : pos_a + cut])
            seg.append(b[:, pos_b : pos_b + (n - cut)])
            pos_a += cut
            pos_b += n - cut
        seg = torch.cat(seg, dim=1)
        e = (seg.float() - whole.float()).abs().max().item()
        good = e == 0.0
        ok &= good
        print(f"  {str(seqlens):22s} cut={cut} C={C}  max|diff|={e:.1e}  {'ok (bit-identical)' if good else 'FAIL'}")
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
    ok &= test_varlen_vs_both_goldens()
    print()
    ok &= test_varlen_equals_per_sequence_calls()
    print()
    ok &= test_varlen_equals_fixed_batch()
    print()
    ok &= test_varlen_state_relay()
    print()

    if ok:
        print("Kernel Output Match!")
    else:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
