"""KDA L1 reference: the six-stage chunkwise pipeline at the frozen L0 interface.

This is the *reference layer* of L1.  It contains no kernel code -- it is pure
PyTorch and runs on CPU.  Its job is to be the per-stage golden that every L1
kernel is checked against, and to prove that the chunkwise math is still correct
after the interface change.

Relationship to the other reference files
-----------------------------------------
    kda_ref.py               L0: token-by-token recurrence at the frozen
                             interface, shipped alongside this file.  The
                             ground truth for everything here.
    (not shipped)            an earlier chunkwise reference from the port.  Its
                             math was verified line by line against FLA, but it
                             used the old layout [B,H,L,DK], had no scale / GVA
                             / initial state, and required L % C == 0.

The inner math here is a direct port of that verified body.  It is deliberately
*not* rewritten: the layout adapter, GVA expansion, scale, initial state and
tail-block padding all live in a thin shell around it, so the part that was
already proven correct stays byte-for-byte the same.

Frozen interface (FLA contract; frozen on day 1 of the port)
----------------------------------------------------
    q, k          [B, T, H,  K]     fp16 / bf16
    v             [B, T, HV, V]     HV % H == 0   (GVA)
    g             [B, T, HV, K]     fp32, log-domain, g <= 0
    beta          [B, T, HV]        fp16 / bf16
    initial_state [B, HV, K, V]     fp32, optional
    o             [B, T, HV, V]     same dtype as input
    final_state   [B, HV, K, V]     fp32
    scale         defaults to K ** -0.5, applied to q

Numerical rules that are load-bearing (do not "simplify" these)
---------------------------------------------------------------
    * exp of a *difference*, never a ratio of two exps.  Deep inside a chunk
      both e^{G_i} and e^{G_j} underflow to 0, and 0/0 is NaN.
    * fold the causal mask into the exponent, never multiply a mask after exp.
      The i<j half overflows to inf first and 0*inf is NaN.  One channel with a
      steep gate is enough to trigger it.
    * restart the cumulative sum at every chunk boundary.  That is what bounds
      how far the exponents can travel, and it is the first line of defence,
      not an optimisation.
"""

import os
import sys

import torch


def _load_l0_ref():
    """Import the L0 recurrent reference, whichever layout we are unpacked into.

    Tried in order: same directory, sibling ``kda/``, ``examples/kda`` two levels
    up (the in-repo layout), then whatever is already on sys.path.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    for cand in (
        here,
        os.path.join(here, "..", "kda"),
        os.path.join(here, "..", "..", "examples", "kda"),
        os.path.join(here, "..", "..", "kda"),
    ):
        if os.path.isfile(os.path.join(cand, "kda_ref.py")):
            sys.path.insert(0, os.path.abspath(cand))
            break
    import kda_ref

    return kda_ref


_L0 = _load_l0_ref()

# Safe here and not at the top of the file: _load_l0_ref is what puts this
# directory on sys.path, so the flat import only resolves once it has run.
import kda_varlen as _VL  # noqa: E402


# --------------------------------------------------------------- layout shell
def _in(x):
    """external [B, T, H, ...] -> internal [B, H, T, ...]"""
    return x.transpose(1, 2)


def _out(x):
    """internal [B, H, T, ...] -> external [B, T, H, ...], contiguous.

    ``.contiguous()`` is not cosmetic.  A bare ``transpose`` returns a
    non-contiguous view, and the tilelang jit wrapper rejects those outright
    ("Input tensor at index N must be contiguous").  More importantly it would
    be *wrong* as a fixture even if it were accepted: in the real pipeline every
    inter-stage tensor is a fresh kernel output and therefore contiguous, so a
    transposed view is an artifact of this reference alone.  Handing one to a
    kernel tests a memory layout that production never produces.
    """
    return x.transpose(1, 2).contiguous()


def _expand_qk(x, grp):
    """GVA: [B, H, T, K] -> [B, HV, T, K] by repeating each qk head grp times."""
    return x if grp == 1 else x.repeat_interleave(grp, dim=1)


def _pad_to_chunk(q, k, v, g, beta, C):
    """Zero-pad the token axis up to a multiple of C.

    The padding is neutral by construction: beta = 0 writes nothing through the
    delta rule, and g = 0 gives alpha = 1 so the state is not decayed either.
    Therefore the final state after the padded tokens equals the final state
    after the real ones, and only the padded rows of the output are garbage --
    they are sliced off by the caller.
    """
    T_ = q.shape[2]
    pad = (-T_) % C
    if pad == 0:
        return q, k, v, g, beta, 0

    def z(x, n):
        return torch.cat([x, x.new_zeros(*x.shape[:2], n, *x.shape[3:])], dim=2)

    return z(q, pad), z(k, pad), z(v, pad), z(g, pad), z(beta, pad), pad


# ------------------------------------------------------------- chunk helpers
def _chunks(x, C):
    """[B,H,L,*] -> [B,H,N,C,*]"""
    B, H, L = x.shape[:3]
    return x.reshape(B, H, L // C, C, *x.shape[3:])


def _unchunk(x):
    """[B,H,N,C,D] -> [B,H,L,D]"""
    B, H, N, C, D = x.shape
    return x.reshape(B, H, N * C, D)


# ----------------------------------------------------- stage 1: chunk_cumsum
def ref_chunk_cumsum(g, C):
    """Chunk-local inclusive prefix sum along the token axis, per channel.

    g is internal layout [B, HV, T, K] with T a multiple of C.
    """
    return _unchunk(_chunks(g.float(), C).cumsum(-2))


# ------------------------------------------------- decayed dot (the hard part)
def _decayed_dot_naive(X, Y, G, mask):
    """The tempting one-liner: fold e^{+G} into X, e^{-G} into Y, one matmul.

    Algebraically exact and numerically useless -- e^{-G_j} grows without bound
    down the chunk.  Kept so the test suite can demonstrate the failure instead
    of merely asserting it.
    """
    A = torch.einsum("...id,...jd->...ij", X * G.exp(), Y * (-G).exp())
    return A * mask


def _decayed_dot(X, Y, G, mask, BC):
    """(i,j) |-> sum_d X[i,d] Y[j,d] exp(G[i,d] - G[j,d]), masked.

    For a scalar gate this factorises as exp(g_i - g_j) * (X Y^T), i.e. a plain
    matmul with a cheap post-scale -- exactly what the GDN kernel does.  With a
    per-channel gate the decay sits *inside* the sum over d, so it has to be
    folded into the two operands, which is safe only while both folded factors
    stay bounded.

    Anchoring, with G non-increasing in i:
      off-diagonal strip (row block a, all columns j < start of a):
          anchor = first row of block a  ->  both exponents <= 0
      diagonal block:
          no anchor bounds both sides, so it is evaluated column by column,
          directly as exp(G[i] - G[j]) restricted to i >= j (exponent <= 0).
    """
    *lead, C, _ = X.shape
    A = X.new_zeros(*lead, C, C)

    for a in range(0, C, BC):
        b = min(a + BC, C)
        anchor = G[..., a, :].unsqueeze(-2)  # [...,1,K]

        if a > 0:  # off-diagonal strip: one matmul against every earlier column
            Xa = X[..., a:b, :] * (G[..., a:b, :] - anchor).exp()
            Yb = Y[..., :a, :] * (anchor - G[..., :a, :]).exp()
            A[..., a:b, :a] = torch.einsum("...id,...jd->...ij", Xa, Yb)

        for j in range(a, b):  # diagonal block, one column at a time
            d = (G[..., j:b, :] - G[..., j, :].unsqueeze(-2)).exp()
            A[..., j:b, j] = (X[..., j:b, :] * Y[..., j, :].unsqueeze(-2) * d).sum(-1)

    return A * mask


def _dot(X, Y, G, mask, BC):
    """BC=None selects the naive one-shot fold, any int selects anchored blocks."""
    return _decayed_dot_naive(X, Y, G, mask) if BC is None else _decayed_dot(X, Y, G, mask, BC)


# ------------------------------------------------------------- stage 2: kkt
def ref_kkt(k, beta, G, C, BC=16):
    """L = strictLower(Diag(beta) . decayed_dot(K, K)).

    All arguments are internal layout with HV heads.
    """
    k, beta, G = k.float(), beta.float(), G.float()
    kc, gc, bc = _chunks(k, C), _chunks(G, C), _chunks(beta, C)
    idx = torch.arange(C, device=k.device)
    mask = (idx[:, None] > idx[None, :]).float()  # strictly lower
    A = _dot(kc, kc, gc, mask, BC) * bc.unsqueeze(-1)
    return _unchunk(A)


# ------------------------------------------------------ stage 3: solve_tril
def ref_solve_tril(L_):
    """A = (I + L)^{-1}, L strictly lower triangular.

    Forward substitution rather than linalg.inv: it mirrors what the kernel does
    row by row, and linalg.inv has no NPU implementation (it silently falls back
    to CPU and then NaN-poisons the matrix).  Identical to GDN -- the gate never
    enters here, which is why this is the one stage whose *math* needs no
    porting at all.
    """
    B, H, L, C = L_.shape
    A = -_chunks(L_.float(), C).clone()
    for i in range(1, C):
        A[..., i, :i] += (A[..., i, :, None] * A[..., :, :i]).sum(-2)
    return _unchunk(A + torch.eye(C, dtype=A.dtype, device=A.device))


# -------------------------------------------------------- stage 4: wy_fast
def ref_wy_fast(k, v, beta, G, A, C, BC=16, q=None):
    """UT transform.  Also emits the folded tensors the later stages need:
    kg = K . e^{G_C - G}, qg = Q . e^G, and the causal Aqk."""
    k, v, beta, G, A = (x.float() for x in (k, v, beta, G, A))
    kc, vc, gc = _chunks(k, C), _chunks(v, C), _chunks(G, C)
    bc, Ac = _chunks(beta, C), _chunks(A, C)

    M = Ac * bc.unsqueeze(-2)  # A . Diag(beta)
    U = M @ vc
    W = M @ (kc * gc.exp())
    kg = kc * (gc[..., -1:, :] - gc).exp()  # exponent <= 0

    out = [_unchunk(W), _unchunk(U), _unchunk(kg)]
    if q is not None:
        qc = _chunks(q.float(), C)
        idx = torch.arange(C, device=k.device)
        mask = (idx[:, None] >= idx[None, :]).float()  # causal, inclusive
        out.append(_unchunk(qc * gc.exp()))  # qg
        out.append(_unchunk(_dot(qc, kc, gc, mask, BC)))  # Aqk
    return tuple(out)


# -------------------------------------------------------- stage 5: chunk_h
def ref_chunk_h(W, U, kg, G, C, initial_state=None):
    """Per-chunk entry states, the pseudo-values V' = U - W S, and the final state.

    initial_state is internal-order [B, HV, K, V] fp32 (same as the external
    contract -- the head axis is already dim 1 there).
    """
    W, U, kg, G = (x.float() for x in (W, U, kg, G))
    B, H, L, K = W.shape
    V = U.shape[-1]
    N = L // C
    Wc, Uc, kgc, gc = _chunks(W, C), _chunks(U, C), _chunks(kg, C), _chunks(G, C)

    if initial_state is None:
        S = torch.zeros((B, H, K, V), dtype=torch.float, device=W.device)
    else:
        S = initial_state.float().clone()

    states = torch.zeros((B, H, N, K, V), dtype=torch.float, device=W.device)
    Vt = torch.empty((B, H, N, C, V), dtype=torch.float, device=W.device)

    for n in range(N):
        states[:, :, n] = S
        Vt[:, :, n] = Uc[:, :, n] - Wc[:, :, n] @ S
        # per-channel state decay: scales row d of S (GDN scales the whole S)
        S = S * gc[:, :, n, -1, :].exp().unsqueeze(-1) + kgc[:, :, n].transpose(-1, -2) @ Vt[:, :, n]

    return states, _unchunk(Vt), S


# -------------------------------------------------------- stage 6: chunk_o
def ref_chunk_o(qg, Aqk, Vt, states, C):
    """O = (Q . e^G) S + Aqk V'."""
    qg, Aqk, Vt = qg.float(), Aqk.float(), Vt.float()
    qgc, Ac, Vc = _chunks(qg, C), _chunks(Aqk, C), _chunks(Vt, C)
    return _unchunk(qgc @ states + Ac @ Vc)


# ---------------------------------------------------------- full L1 pipeline
def kda_chunk_ref(q, k, v, g, beta, C=64, BC=16, scale=None, initial_state=None, output_final_state=False, cu_seqlens=None):
    """Six-stage chunkwise KDA at the frozen L0 interface.

    Signature mirrors kda_ref.kda_ref, plus the chunk length C and the
    sub-chunk anchor width BC.  Returns (o, final_state | None).

    With ``cu_seqlens`` the inputs are a flattened varlen batch (B == 1) and the
    final state is [N, HV, K, V], one per sequence.  See _kda_chunk_ref_varlen.
    """
    if cu_seqlens is not None:
        return _kda_chunk_ref_varlen(q, k, v, g, beta, C, BC, scale, initial_state, output_final_state, cu_seqlens)

    B, SEQ, H, K = q.shape
    HV, _V = v.shape[2], v.shape[-1]
    assert HV % H == 0, "HV must be divisible by H (GVA)"
    grp = HV // H
    if scale is None:
        scale = K**-0.5

    # SEQ == 0 is guarded here so the reference carries exactly the same
    # contract as the kernel path: no token is consumed, every token-axis
    # output is empty, and the final state is the initial state untouched.
    # Without this the chunk helpers would reshape a zero-element tensor into
    # zero chunks and the stage algebra would run on empty operands -- which
    # happens to work in torch, but only by accident and not in every stage.
    if SEQ == 0:
        o = torch.empty((B, 0, HV, _V), dtype=torch.float32)
        if not output_final_state:
            return o, None
        S = initial_state.float().clone() if initial_state is not None else torch.zeros((B, HV, K, _V), dtype=torch.float32)
        return o, S

    # external -> internal, GVA expansion, scale on q
    qi = _expand_qk(_in(q.float()), grp) * scale
    ki = _expand_qk(_in(k.float()), grp)
    vi, gi, bi = _in(v.float()), _in(g.float()), _in(beta.float())

    qi, ki, vi, gi, bi, pad = _pad_to_chunk(qi, ki, vi, gi, bi, C)

    G = ref_chunk_cumsum(gi, C)
    A = ref_kkt(ki, bi, G, C, BC)
    A = ref_solve_tril(A)
    W, U, kg, qg, Aqk = ref_wy_fast(ki, vi, bi, G, A, C, BC, q=qi)
    states, Vt, S = ref_chunk_h(W, U, kg, G, C, initial_state=initial_state)
    o = ref_chunk_o(qg, Aqk, Vt, states, C)

    if pad:
        o = o[:, :, :SEQ]
    return _out(o), (S if output_final_state else None)


# ------------------------------------------------- stage tensors for kernel tests
def stage_tensors(q, k, v, g, beta, C=64, BC=16, scale=None, initial_state=None, cu_seqlens=None):
    """Every intermediate of the six stages, in **external** [B, T, HV, *] layout.

    This is the single source of truth for per-stage kernel tests: a kernel test
    calls this once, feeds the kernel the same external tensors the pipeline
    would, and compares against the matching entry here.  Keeping the layout
    conversion in one place is deliberate -- six separate tests each doing their
    own transpose is how the head axis ends up swapped in exactly one of them.

    A ragged tail is supported.  When ``SEQ % C != 0`` the token axis is padded
    up to a whole number of chunks with ``_pad_to_chunk`` -- neutral by
    construction, since the padded tokens carry ``beta = 0`` (the delta rule
    writes nothing) and ``g = 0`` (``alpha = 1``, the state is not decayed) --
    and every token-axis output is sliced back to ``SEQ`` on the way out.

    Padding here is on the host and that is fine: this is the golden, not the
    kernel path.  The acceptance gate rules out host-side padding because it
    would hide kernel cost, which is not a concern for a CPU reference.

    ``states`` is *not* sliced.  It legitimately gains a chunk, because the tail
    is a real chunk with a real entry state, and the kernels index it the same
    way.  ``SF`` is unaffected -- the padded tokens leave the state untouched,
    so the final state after them is the final state after the real ones.

    Returned dict, with N = ceil(SEQ / C):

        G       [B, T, HV, K]     stage 1 out: chunk-local cumulative log gate
        L       [B, T, HV, C]     stage 2 out: strictly-lower Diag(beta).KK^T
        A       [B, T, HV, C]     stage 3 out: (I + L)^{-1}
        W       [B, T, HV, K]     stage 4 out
        U       [B, T, HV, V]     stage 4 out
        kg      [B, T, HV, K]     stage 4 out: K . e^{G_C - G}
        qg      [B, T, HV, K]     stage 4 out: (scale.Q) . e^{G}
        Aqk     [B, T, HV, C]     stage 4 out: causal decayed Q K^T
        states  [B, HV, N, K, V]  stage 5 out: per-chunk entry states
        Vt      [B, T, HV, V]     stage 5 out: pseudo-values U - W S
        SF      [B, HV, K, V]     stage 5 out: final state
        o       [B, T, HV, V]     stage 6 out
    """
    if cu_seqlens is not None:
        return _stage_tensors_varlen(q, k, v, g, beta, C, BC, scale, initial_state, cu_seqlens)

    B, SEQ, H, K = q.shape
    HV, _V = v.shape[2], v.shape[-1]
    assert HV % H == 0, "HV must be divisible by H (GVA)"
    grp = HV // H
    if scale is None:
        scale = K**-0.5

    # SEQ == 0 gets the same explicit guard kda_chunk_ref carries, and for the
    # same reason: _pad_to_chunk leaves a zero-token tensor alone (pad = 0), the
    # chunk helpers then reshape it into zero chunks, and the stage algebra runs
    # on empty operands -- which torch tolerates in some stages by accident and
    # not in others.  Under varlen this stopped being a corner case: a
    # cu_seqlens entry with T_i == 0 routes straight through here.
    if SEQ == 0:

        def _empty(width):
            return torch.zeros((B, 0, HV, width), dtype=torch.float32)

        sf = initial_state.float().clone() if initial_state is not None else torch.zeros((B, HV, K, _V), dtype=torch.float32)
        return {
            "G": _empty(K),
            "L": _empty(C),
            "A": _empty(C),
            "W": _empty(K),
            "U": _empty(_V),
            "kg": _empty(K),
            "qg": _empty(K),
            "Aqk": _empty(C),
            "states": torch.zeros((B, HV, 0, K, _V), dtype=torch.float32),
            "Vt": _empty(_V),
            "SF": sf,
            "o": _empty(_V),
        }

    qi = _expand_qk(_in(q.float()), grp) * scale
    ki = _expand_qk(_in(k.float()), grp)
    vi, gi, bi = _in(v.float()), _in(g.float()), _in(beta.float())

    # pad == 0 short-circuits inside _pad_to_chunk, so the aligned path is
    # bit-for-bit what it was before the tail was supported.
    qi, ki, vi, gi, bi, pad = _pad_to_chunk(qi, ki, vi, gi, bi, C)

    G = ref_chunk_cumsum(gi, C)
    L = ref_kkt(ki, bi, G, C, BC)
    A = ref_solve_tril(L)
    W, U, kg, qg, Aqk = ref_wy_fast(ki, vi, bi, G, A, C, BC, q=qi)
    states, Vt, SF = ref_chunk_h(W, U, kg, G, C, initial_state=initial_state)
    o = ref_chunk_o(qg, Aqk, Vt, states, C)

    # Internal layout is [B, HV, T, *]: the token axis is dim 2.  states and SF
    # have no token axis and are returned whole.
    def _cut(x):
        return x if pad == 0 else x[:, :, :SEQ]

    return {
        "G": _out(_cut(G)),
        "L": _out(_cut(L)),
        "A": _out(_cut(A)),
        "W": _out(_cut(W)),
        "U": _out(_cut(U)),
        "kg": _out(_cut(kg)),
        "qg": _out(_cut(qg)),
        "Aqk": _out(_cut(Aqk)),
        "states": states,
        "Vt": _out(_cut(Vt)),
        "SF": SF,
        "o": _out(_cut(o)),
    }


def make_inputs(*a, **kw):
    """Re-export of the L0 input factory so kernel tests need one import only."""
    return _L0.make_inputs(*a, **kw)


def make_varlen_inputs(*a, **kw):
    """Re-export of the L0 varlen input factory, same reason."""
    return _L0.make_varlen_inputs(*a, **kw)


# ---------------------------------------------------------------------- varlen
# Both varlen entry points below run the fixed-length body once per sequence and
# reassemble.  That is exact, not an approximation: chunking restarts at every
# bos and no state crosses a sequence boundary, so a per-sequence run performs
# the same arithmetic in the same order.  It is also the only construction worth
# having in a golden -- a second, varlen-aware copy of the stage algebra would
# just be a second thing to be wrong, and it is a CPU reference, so the loop
# costs nothing that matters.
#
# The slices are views, not copies: with B == 1 torch ignores the stride of the
# extent-1 batch axis when deciding contiguity, so ``q[:, bos:eos]`` stays
# contiguous.
_TOKEN_AXIS_KEYS = ("G", "L", "A", "W", "U", "kg", "qg", "Aqk", "Vt", "o")


def _seq_slices(q, k, v, g, beta, bos, eos):
    return q[:, bos:eos], k[:, bos:eos], v[:, bos:eos], g[:, bos:eos], beta[:, bos:eos]


def _kda_chunk_ref_varlen(q, k, v, g, beta, C, BC, scale, initial_state, output_final_state, cu_seqlens):
    bounds = _L0.varlen_bounds(cu_seqlens, q, k, v, g, beta, initial_state)
    outs, finals = [], []
    for i, (bos, eos) in enumerate(bounds):
        qs, ks, vs, gs, bs = _seq_slices(q, k, v, g, beta, bos, eos)
        s0 = None if initial_state is None else initial_state[i : i + 1]
        o_i, s_i = kda_chunk_ref(qs, ks, vs, gs, bs, C=C, BC=BC, scale=scale, initial_state=s0, output_final_state=True)
        outs.append(o_i)
        finals.append(s_i)
    o = torch.cat(outs, dim=1) if len(outs) > 1 else outs[0]
    return o, (torch.cat(finals, dim=0) if output_final_state else None)


def _stage_tensors_varlen(q, k, v, g, beta, C, BC, scale, initial_state, cu_seqlens):
    """Per-stage fixtures for a varlen batch, in the layout the kernels produce.

    Token-axis intermediates concatenate along the token axis and therefore keep
    the flattened [1, T_total, HV, *] layout unchanged -- which is the whole
    reason the varlen kernels do not need a layout change either.

    ``states`` is the exception.  It is indexed by chunk, not by token, so the
    per-sequence blocks concatenate along the chunk axis into a single
    NT_total-long run.  Sequence n's chunk i therefore lives at slot
    ``chunk_offsets[n] + i`` where ``chunk_offsets[n] = sum_{m<n} ceil(T_m / C)``
    -- the same addressing FLA's prepare_chunk_offsets produces, and the same one
    the stage-5 kernel computes on the device.  An empty sequence contributes
    zero slots, so it simply does not appear.
    """
    bounds = _L0.varlen_bounds(cu_seqlens, q, k, v, g, beta, initial_state)
    parts = []
    for i, (bos, eos) in enumerate(bounds):
        qs, ks, vs, gs, bs = _seq_slices(q, k, v, g, beta, bos, eos)
        s0 = None if initial_state is None else initial_state[i : i + 1]
        parts.append(stage_tensors(qs, ks, vs, gs, bs, C=C, BC=BC, scale=scale, initial_state=s0))

    out = {}
    for key in _TOKEN_AXIS_KEYS:
        out[key] = torch.cat([p[key] for p in parts], dim=1)
    # states is [B, HV, N, K, V]; the chunk axis is dim 2.
    out["states"] = torch.cat([p["states"] for p in parts], dim=2)
    # SF is [1, HV, K, V] per sequence -> [N, HV, K, V].
    out["SF"] = torch.cat([p["SF"] for p in parts], dim=0)
    return out


def chunk_offsets(cu_seqlens, C):
    """(first chunk slot of each sequence, total chunk count) -- see kda_varlen.

    Kept here as a convenience for the tests in this file; the arithmetic itself
    lives in kda_varlen so the kernels and the goldens cannot drift apart on it.
    """
    return _VL.chunk_layout(_VL.varlen_bounds(cu_seqlens), C)


# ------------------------------------------------------------------- selftest
def _err(a, b):
    return (a - b).abs().max().item()


def test_chunk_vs_recurrent():
    """The load-bearing test: chunkwise must reproduce the token loop.

    Both references are independent implementations of the same recurrence, so
    agreement here means the chunkwise math survived the interface change.
    Runs on CPU in fp32 -- NPU einsum dispatches to matmul with reduced-precision
    accumulation and drifts two identical fp32 references by ~3e-4.
    """
    l0 = _L0

    print("== chunkwise ref  vs  L0 recurrent ref (fp32, CPU) ==")
    ok = True
    cases = [
        # B  SEQ   H  HV   K    V    C   gate
        (1, 64, 1, 1, 64, 64, 64, "normal"),
        (2, 128, 2, 4, 64, 64, 64, "normal"),
        (1, 128, 2, 2, 64, 128, 32, "forget"),
        (2, 70, 1, 2, 64, 64, 64, "normal"),  # tail block 70 = 64 + 6
        (1, 33, 1, 1, 64, 64, 32, "keep"),  # tail block 33 = 32 + 1
        (1, 256, 1, 1, 128, 128, 64, "forget"),  # K3 head dim
        (1, 128, 1, 1, 64, 64, 64, "extreme"),  # exponents die immediately
    ]
    for B, SEQ, H, HV, K, V, C, gate in cases:
        q, k, v, g, beta, _ = _mk(B, SEQ, H, HV, K, V, gate)
        ref, _ = l0.kda_ref(q, k, v, g, beta)
        got, _ = kda_chunk_ref(q, k, v, g, beta, C=C)
        e = _err(got, ref)
        rel = e / ref.abs().max().item()
        good = rel < 1e-5
        ok &= good
        print(
            f"  B={B} T={SEQ:3d} H={H} HV={HV} K={K:3d} V={V:3d} C={C:2d} "
            f"{gate:8s} max|err|={e:.3e} rel={rel:.2e}  {'ok' if good else 'FAIL'}"
        )
    return ok


def test_state_relay():
    """Whole sequence in one call must equal two calls relaying final_state."""
    print("== whole  vs  two-segment relay (chunkwise ref) ==")
    ok = True
    for SEQ, cut, C in ((128, 64, 64), (256, 128, 64), (128, 96, 32)):
        q, k, v, g, beta, _ = _mk(2, SEQ, 2, 2, 64, 64, "normal")
        whole, _ = kda_chunk_ref(q, k, v, g, beta, C=C)
        a, sa = kda_chunk_ref(q[:, :cut], k[:, :cut], v[:, :cut], g[:, :cut], beta[:, :cut], C=C, output_final_state=True)
        b, _ = kda_chunk_ref(q[:, cut:], k[:, cut:], v[:, cut:], g[:, cut:], beta[:, cut:], C=C, initial_state=sa)
        seg = torch.cat([a, b], dim=1)
        e = _err(whole, seg)
        good = e < 1e-4
        ok &= good
        print(f"  T={SEQ} cut={cut} C={C}  max|err|={e:.3e}  {'ok' if good else 'FAIL'}")
    return ok


def test_empty_sequence():
    """T == 0 is legal and must leave the state untouched.

    0 % C == 0, so a zero-length sequence passes every chunk-divisibility guard
    on the way in.  It has to be caught explicitly or the chunk helpers reshape
    a zero-element tensor into zero chunks and the stage algebra silently runs
    on empty operands.  The contract is that no token was consumed: token-axis
    outputs are empty and final_state == initial_state.
    """
    print("== zero-length sequence (T = 0, chunkwise ref) ==")
    ok = True
    B, H, HV, K, V, C = 2, 1, 2, 64, 64, 64
    q, k, v = torch.empty(B, 0, H, K), torch.empty(B, 0, H, K), torch.empty(B, 0, HV, V)
    g, beta = torch.empty(B, 0, HV, K), torch.empty(B, 0, HV)

    o, sf = kda_chunk_ref(q, k, v, g, beta, C=C, output_final_state=True)
    m = sf.abs().max().item()
    good = tuple(o.shape) == (B, 0, HV, V) and tuple(sf.shape) == (B, HV, K, V) and m == 0.0
    ok &= good
    print(f"  S0=none    o{tuple(o.shape)} sf{tuple(sf.shape)} max|sf|={m:.1e}  {'ok' if good else 'FAIL'}")

    s0 = torch.randn(B, HV, K, V)
    o, sf = kda_chunk_ref(q, k, v, g, beta, C=C, initial_state=s0, output_final_state=True)
    e = _err(sf, s0)
    good = tuple(o.shape) == (B, 0, HV, V) and e == 0.0
    ok &= good
    print(f"  S0=random  max|sf-S0|={e:.1e}  {'ok (bit-identical passthrough)' if good else 'FAIL'}")

    sf.zero_()
    good = s0.abs().max().item() > 0.0
    ok &= good
    print(f"  final_state is a copy, not an alias  {'ok' if good else 'FAIL'}")
    return ok


def test_naive_fold_blows_up():
    """Demonstrate, not assert, why the anchored form is required.

    BC=None selects the naive fold.  Under a strong gate it must produce a
    non-finite value while the anchored form stays exact.
    """
    print("== naive fold (BC=None) vs anchored (BC=16) under strong gates ==")
    ok = True
    for gate in ("normal", "forget", "extreme"):
        q, k, v, g, beta, _ = _mk(1, 64, 1, 1, 64, 64, gate)
        gi = _in(g.float())
        G = ref_chunk_cumsum(gi, 64)
        gmin = G[..., -1, :].min().item()
        bad = ref_kkt(_expand_qk(_in(k.float()), 1), _in(beta.float()), G, 64, None)
        good = ref_kkt(_expand_qk(_in(k.float()), 1), _in(beta.float()), G, 64, 16)
        nf = (~torch.isfinite(bad)).sum().item()
        print(f"  {gate:8s} min(Gamma_C)={gmin:8.1f}   naive non-finite={nf:6d}   anchored finite={bool(torch.isfinite(good).all())}")
        ok &= bool(torch.isfinite(good).all())
    return ok


def test_varlen():
    """A flattened varlen batch must reproduce the L0 recurrence sequence by sequence.

    Checked three ways, because each catches something the others cannot:

      1. vs the L0 recurrence run per sequence -- the actual correctness claim.
      2. chunk_offsets / NT_total against a hand-computed expectation, including
         an empty sequence, since that arithmetic is what the stage-5 kernel will
         reproduce on the device and an off-by-one there is silent.
      3. shapes of every stage_tensors entry, because the varlen fixtures are
         what the per-stage kernel tests will compare against; a wrong shape here
         would show up six stages later as a mysterious kernel failure.
    """
    print("== varlen (chunkwise ref) vs L0 recurrent ref, per sequence ==")
    ok = True
    cases = [
        ([64, 64], 64, "equal, chunk-aligned"),
        ([70, 33, 129], 64, "all ragged, interior tails"),
        ([70, 0, 129], 64, "empty sequence in the middle"),
        ([0, 64], 64, "empty sequence first"),
        ([64, 0], 64, "empty sequence last"),
        ([1, 200], 64, "one token, then a long sequence"),
        ([5], 64, "N = 1, shorter than a chunk"),
        ([100, 28], 32, "C = 32"),
    ]
    for seqlens, C, note in cases:
        q, k, v, g, beta, s0, cu = _L0.make_varlen_inputs(seqlens, 2, 4, 64, 64, device="cpu", dtype=torch.float32, with_state=True)
        ref, ref_sf = _L0.kda_ref(q, k, v, g, beta, initial_state=s0, output_final_state=True, cu_seqlens=cu)
        got, got_sf = kda_chunk_ref(q, k, v, g, beta, C=C, initial_state=s0, output_final_state=True, cu_seqlens=cu)

        denom = max(ref.abs().max().item(), 1e-9)
        rel = _err(got, ref) / denom
        rel_sf = _err(got_sf, ref_sf) / max(ref_sf.abs().max().item(), 1e-9)

        offs, nt_total = chunk_offsets(cu, C)
        want_offs, want_total = [], 0
        for n in seqlens:
            want_offs.append(want_total)
            want_total += -(-n // C)
        meta_ok = offs == want_offs and nt_total == want_total

        st = stage_tensors(q, k, v, g, beta, C=C, initial_state=s0, cu_seqlens=cu)
        T_total = int(sum(seqlens))
        shape_ok = all(st[key].shape[:3] == (1, T_total, 4) for key in _TOKEN_AXIS_KEYS)
        shape_ok &= tuple(st["states"].shape) == (1, 4, nt_total, 64, 64)
        shape_ok &= tuple(st["SF"].shape) == (len(seqlens), 4, 64, 64)

        # Every empty sequence passes its state through untouched, bit for bit.
        empty_ok = all(bool(torch.equal(got_sf[i], s0[i])) for i, n in enumerate(seqlens) if n == 0)

        good = rel < 1e-5 and rel_sf < 1e-5 and meta_ok and shape_ok and empty_ok
        ok &= good
        flags = f"meta={'ok' if meta_ok else 'BAD'} shapes={'ok' if shape_ok else 'BAD'} empty={'ok' if empty_ok else 'BAD'}"
        print(
            f"  {str(seqlens):18s} C={C:2d} NT={nt_total:2d} rel(o)={rel:.2e} rel(SF)={rel_sf:.2e} "
            f"{flags}  {'ok' if good else 'FAIL'}  {note}"
        )
    return ok


def test_varlen_equals_fixed_batch():
    """N equal-length sequences under varlen must equal a B = N fixed batch, bit for bit.

    This is the one varlen test that is not a tautology.  The varlen reference is
    itself a per-sequence loop, so comparing it against a per-sequence loop proves
    nothing; comparing it against the *batched* path exercises the flatten/split
    mapping and the [N, HV, K, V] state indexing, which is where a real mistake
    would live.  Exact equality, not a tolerance: the two runs perform the same
    arithmetic in the same order on the same numbers.
    """
    print("== varlen (N equal sequences)  vs  fixed-length B = N batch ==")
    ok = True
    for seqlens, C in (([64, 64, 64], 64), ([70, 70], 64), ([128, 128], 32)):
        N, L = len(seqlens), seqlens[0]
        q, k, v, g, beta, s0, cu = _L0.make_varlen_inputs(seqlens, 2, 4, 64, 64, device="cpu", dtype=torch.float32, with_state=True)
        o_v, sf_v = kda_chunk_ref(q, k, v, g, beta, C=C, initial_state=s0, output_final_state=True, cu_seqlens=cu)

        def rs(x, N=N, L=L):
            return x.reshape(N, L, *x.shape[2:])

        o_b, sf_b = kda_chunk_ref(rs(q), rs(k), rs(v), rs(g), rs(beta), C=C, initial_state=s0, output_final_state=True)
        good = bool(torch.equal(o_v.reshape(N, L, 4, 64), o_b)) and bool(torch.equal(sf_v, sf_b))
        ok &= good
        print(f"  {str(seqlens):18s} C={C:2d}  {'ok (bit-identical)' if good else 'FAIL'}")
    return ok


def _mk(B, SEQ, H, HV, K, V, gate):
    return _L0.make_inputs(B, SEQ, H, HV, K, V, device="cpu", dtype=torch.float32, gate=gate, seed=0)


def main():
    torch.manual_seed(0)
    ok = True
    ok &= test_chunk_vs_recurrent()
    print()
    ok &= test_state_relay()
    print()
    ok &= test_empty_sequence()
    print()
    ok &= test_varlen()
    print()
    ok &= test_varlen_equals_fixed_batch()
    print()
    ok &= test_naive_fold_blows_up()
    print()
    if ok:
        print("Kernel Output Match!")
    else:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
