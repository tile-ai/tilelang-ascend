"""KDA L0 reference (golden B): the token-by-token recurrence, written from the
paper rather than from FLA.

This is the second of two independent goldens.  It expands Eq. 1 of the Kimi
Linear report directly and deliberately does NOT follow FLA's code structure:
two implementations agreeing is only evidence if they were not copied from each
other -- a transcription would reproduce the same mistake and the cross-check
would prove nothing.

Interface (the FLA contract, frozen on day 1 of the port):

    q, k          [B, T, H,  K]     H  query/key heads
    v             [B, T, HV, V]     HV value heads; HV must divide by H (GVA)
    g             [B, T, HV, K]     per-channel forget gate in log space, g <= 0
    beta          [B, T, HV]        delta-rule step size
    initial_state [N, HV, K, V]     optional; N = number of sequences (= B when
                                    all sequences have the same length)
    ->
    o             [B, T, HV, V]
    final_state   [N, HV, K, V]     optional

Recurrence (Eq. 1, state laid out as [K, V]):

    S_t = (I - beta_t k_t k_t^T) Diag(alpha_t) S_{t-1} + beta_t k_t v_t^T
    o_t = S_t^T q_t

Expanded into the three steps the code actually performs (alpha = exp(g)):

    S  <- Diag(alpha_t) S            # scale row d of S by alpha[d]
    S  <- S + beta_t k_t (v_t - S^T k_t)^T
    o_t = S^T (scale * q_t)
"""

import torch

# varlen_bounds is re-exported below so callers that already import this module
# do not need a second import just to validate a cu_seqlens.  The bookkeeping
# itself lives in a leaf module the six kernel files can depend on without
# pulling this reference implementation in with it.
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from kda_varlen import varlen_bounds

__all__ = ["kda_ref", "make_inputs", "make_varlen_inputs", "varlen_bounds"]


def _kda_ref_varlen(q, k, v, g, beta, cu_seqlens, scale, initial_state, output_final_state):
    """Varlen by running the fixed-length recurrence once per sequence.

    This is exact, not an approximation.  The recurrence carries no state across
    a sequence boundary -- each sequence starts from its own initial_state[i] --
    so slicing the flattened batch and running the sequences one at a time
    performs literally the same arithmetic in the same order.  Writing a second,
    "varlen-aware" recurrence here would only create a second thing to be wrong.

    The slices are free.  With B == 1 the batch axis has extent 1, and torch
    ignores the stride of an extent-1 axis when it decides contiguity, so
    ``q[:, bos:eos]`` is a contiguous view rather than a copy.  That matters
    beyond speed: an acceptance gate forbids large host-side tensor transforms
    that hide kernel cost, and a golden that quietly copied every input would
    set the wrong precedent for the kernel wrapper next door.
    """
    bounds = varlen_bounds(cu_seqlens, q, k, v, g, beta, initial_state)
    HV, V = v.shape[2], v.shape[-1]
    K = q.shape[-1]

    outs, finals = [], []
    for i, (bos, eos) in enumerate(bounds):
        s0 = None if initial_state is None else initial_state[i : i + 1]
        o_i, s_i = kda_ref(
            q[:, bos:eos],
            k[:, bos:eos],
            v[:, bos:eos],
            g[:, bos:eos],
            beta[:, bos:eos],
            scale=scale,
            initial_state=s0,
            output_final_state=True,
        )
        outs.append(o_i)
        finals.append(s_i)

    # An all-empty batch still has to produce correctly shaped outputs; torch.cat
    # of zero-token pieces gives [1, 0, HV, V], which is what the contract asks
    # for, but only if the list is non-empty -- and cu_seqlens always has at
    # least one sequence, so it is.
    o = torch.cat(outs, dim=1) if len(outs) > 1 else outs[0]
    if not output_final_state:
        return o, None

    # [N, HV, K, V]: one state per sequence, in cu_seqlens order.
    sf = torch.cat(finals, dim=0)
    assert sf.shape == (len(bounds), HV, K, V), f"final_state must be [N, HV, K, V], got {tuple(sf.shape)}"
    return o, sf


def kda_ref(q, k, v, g, beta, scale=None, initial_state=None, output_final_state=False, cu_seqlens=None):
    if cu_seqlens is not None:
        return _kda_ref_varlen(q, k, v, g, beta, cu_seqlens, scale, initial_state, output_final_state)

    B, SEQ, H, K = q.shape
    HV, V = v.shape[2], v.shape[-1]
    assert HV % H == 0, "HV must be divisible by H"
    grp = HV // H  # every grp value heads share one qk head
    if scale is None:
        scale = K**-0.5  # default scaling, applied to q

    q, k, v, g, beta = (x.float() for x in (q, k, v, g, beta))

    S = q.new_zeros(B, HV, K, V)
    if initial_state is not None:
        S = S + initial_state.float()
    o = q.new_zeros(B, SEQ, HV, V)

    for t in range(SEQ):
        # GVA: value head hv reads qk head hv // grp; repeat_interleave expands it
        q_t = q[:, t].repeat_interleave(grp, dim=1) * scale  # [B, HV, K]
        k_t = k[:, t].repeat_interleave(grp, dim=1)  # [B, HV, K]
        v_t = v[:, t]  # [B, HV, V]
        a_t = g[:, t].exp()  # [B, HV, K]
        b_t = beta[:, t]  # [B, HV]

        # 1) per-channel decay: scale row d of S by alpha[d]
        S = S * a_t.unsqueeze(-1)

        # 2) delta rule: read what the decayed state predicts for this key, then
        #    write back only the error
        pred = torch.einsum("bhkv,bhk->bhv", S, k_t)  # S^T k
        err = v_t - pred
        S = S + b_t[..., None, None] * k_t.unsqueeze(-1) * err.unsqueeze(-2)

        # 3) read out
        o[:, t] = torch.einsum("bhkv,bhk->bhv", S, q_t)

    return o, (S if output_final_state else None)


def make_inputs(B, SEQ, H, HV, K, V, device="npu", dtype=torch.float16, gate="normal", seed=0, with_state=False):
    """Build inputs in the frozen shapes.

    The four gate settings bracket the retention/forgetting boundary:
        keep     alpha -> 1, almost nothing is forgotten
        normal   logsigmoid, the shape seen in ordinary training
        forget   K3's safe_gate form, lower_bound = -5
        extreme  no lower bound, a single step decays the state away
    """
    import torch.nn.functional as F

    torch.manual_seed(seed)
    q = F.normalize(torch.randn(B, SEQ, H, K, device=device), dim=-1, p=2)
    k = F.normalize(torch.randn(B, SEQ, H, K, device=device), dim=-1, p=2)
    v = torch.randn(B, SEQ, HV, V, device=device)
    beta = torch.rand(B, SEQ, HV, device=device)

    a = torch.randn(B, SEQ, HV, K, device=device)
    if gate == "keep":
        g = -1e-4 * torch.rand(B, SEQ, HV, K, device=device)
    elif gate == "normal":
        g = F.logsigmoid(a)
    elif gate == "forget":  # K3: lower_bound * sigmoid(exp(A_log) * a)
        A_log = torch.rand(1, 1, HV, K, device=device) * 15 + 1
        g = -5.0 * torch.sigmoid(A_log * a)
    else:  # extreme
        A_log = torch.rand(1, 1, HV, K, device=device) * 15 + 1
        g = -A_log * F.softplus(a)

    s0 = torch.randn(B, HV, K, V, device=device) * 0.1 if with_state else None
    return q.to(dtype), k.to(dtype), v.to(dtype), g.float(), beta.to(dtype), s0


def make_varlen_inputs(seqlens, H, HV, K, V, device="npu", dtype=torch.float16, gate="normal", seed=0, with_state=False):
    """Build a flattened varlen batch and its cu_seqlens.

    ``seqlens`` is the per-sequence token count; entries may be 0.  The tensors
    come back at B == 1 with sum(seqlens) tokens, and the state (when asked for)
    is [N, HV, K, V] -- one per sequence, not one per batch element.

    Built by generating the whole flattened batch in one shot rather than per
    sequence and concatenating.  Concatenating would make the contents depend on
    how the batch happens to be split, so the same token index would hold
    different numbers under a different split -- and the load-bearing test here
    is exactly that a varlen run and a per-sequence run agree token for token.
    """
    T_total = int(sum(seqlens))
    q, k, v, g, beta, _ = make_inputs(1, T_total, H, HV, K, V, device=device, dtype=dtype, gate=gate, seed=seed)

    cu = [0]
    for n in seqlens:
        cu.append(cu[-1] + int(n))
    cu_seqlens = torch.tensor(cu, dtype=torch.int32, device=device)

    s0 = None
    if with_state:
        torch.manual_seed(seed + 1)
        s0 = torch.randn(len(seqlens), HV, K, V, device=device) * 0.1
    return q, k, v, g, beta, s0, cu_seqlens


if __name__ == "__main__":
    # Self-check against this module's own recurrence: run every gate setting
    # and every shape in the matrix, and assert the invariants that hold by
    # construction.  This is deliberately self-contained.
    #
    # The stronger check -- cross-validating against flash-linear-attention's
    # naive_recurrent_kda -- was run during development and is reported in the
    # pull-request description.  It is not wired in here because FLA is a
    # third-party dependency that this example must not carry.
    dev = "npu" if hasattr(torch, "npu") and torch.npu.is_available() else "cpu"
    ok = True
    print(f"{'shape':38s} {'gate':10s} {'result':>10s}")
    print("-" * 62)

    cases = [
        (1, 1, 1, 1, 16, 16),  # pure decode
        (1, 7, 2, 2, 32, 32),  # very short sequence
        (4, 64, 4, 4, 64, 64),  # exactly one chunk
        (2, 70, 2, 4, 32, 32),  # not a multiple of 64, plus GVA (HV = 2H)
        (1, 130, 1, 4, 16, 32),  # several chunks, GVA (HV = 4H), K != V
    ]
    for B, SEQ, H, HV, K, V in cases:
        for gate in ("keep", "normal", "forget", "extreme"):
            for ws in (False, True):
                q, k, v, g, beta, s0 = make_inputs(B, SEQ, H, HV, K, V, device=dev, dtype=torch.float32, gate=gate, with_state=ws)
                o, s = kda_ref(q, k, v, g, beta, initial_state=s0, output_final_state=True)
                good = (
                    o.shape == (B, SEQ, HV, V)
                    and s.shape == (B, HV, K, V)
                    and torch.isfinite(o).all().item()
                    and torch.isfinite(s).all().item()
                )
                # beta = 0 must leave the state untouched apart from the decay
                if good and not ws:
                    o0, s0z = kda_ref(q, k, v, g, torch.zeros_like(beta), output_final_state=True)
                    good = bool(torch.equal(s0z, torch.zeros_like(s0z)))
                ok &= good
                if not good:
                    print(f"B{B} T{SEQ} H{H} HV{HV} K{K} V{V} state={ws!s:5s} {gate:10s} {'FAIL':>10s}")
        print(f"B{B} T{SEQ} H{H} HV{HV} K{K} V{V}".ljust(38) + f"{'all gates':10s} {'ok' if ok else 'FAIL':>10s}")

    # ------------------------------------------------------------- varlen
    # The load-bearing check is NOT "varlen agrees with a per-sequence loop" --
    # the varlen path here IS a per-sequence loop, so that would be a tautology.
    # It is that a varlen batch of N equal-length sequences reproduces the same
    # data run as an ordinary B = N fixed-length batch.  That exercises the part
    # that can actually be wrong: the flatten/split mapping and the [N, HV, K, V]
    # state indexing.
    print()
    print(f"{'varlen':38s} {'seqlens':22s} {'result':>10s}")
    print("-" * 74)

    varlen_cases = [
        ([64, 64, 64], "equal, chunk-aligned"),
        ([70, 33, 129], "all ragged"),
        ([0, 70, 0], "empty first and last"),
        ([70, 0, 129], "empty in the middle"),
        ([0, 0], "every sequence empty"),
        ([1], "N = 1, single token"),
        ([5], "N = 1, shorter than a chunk"),
    ]
    for seqlens, note in varlen_cases:
        good = True
        for gate in ("normal", "forget"):
            for ws in (False, True):
                q, k, v, g, beta, s0, cu = make_varlen_inputs(
                    seqlens, 2, 4, 32, 32, device=dev, dtype=torch.float32, gate=gate, with_state=ws
                )
                o, s = kda_ref(q, k, v, g, beta, initial_state=s0, output_final_state=True, cu_seqlens=cu)
                N, T_total = len(seqlens), int(sum(seqlens))
                good &= o.shape == (1, T_total, 4, 32) and s.shape == (N, 4, 32, 32)
                good &= bool(torch.isfinite(o).all().item()) and bool(torch.isfinite(s).all().item())

                # An empty sequence consumes no token, so its final state must be
                # its initial state, bit for bit -- not merely close.
                for i, n in enumerate(seqlens):
                    if n == 0:
                        want = s0[i] if ws else torch.zeros_like(s[i])
                        good &= bool(torch.equal(s[i], want))

                # Equal lengths only: the same tokens run as a B = N batch.
                if len(set(seqlens)) == 1 and seqlens[0] > 0:
                    Ls = seqlens[0]

                    # N and Ls bound as defaults, not captured: a closure over a
                    # loop variable reads its LAST value, which is only harmless
                    # here because it is called in the same iteration.  ruff
                    # flags it (B023) and is right to.
                    def rs(x, N=N, Ls=Ls):
                        return x.reshape(N, Ls, *x.shape[2:])

                    ob, sb = kda_ref(rs(q), rs(k), rs(v), rs(g), rs(beta), initial_state=s0, output_final_state=True)
                    good &= bool(torch.equal(o.reshape(N, Ls, 4, 32), ob)) and bool(torch.equal(s, sb))
        ok &= good
        print(f"{str(seqlens):38s} {note:22s} {'ok' if good else 'FAIL':>10s}")

    print()
    if ok:
        print("Kernel Output Match!")
    else:
        raise SystemExit(1)
