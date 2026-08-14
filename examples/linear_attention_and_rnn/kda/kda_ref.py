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

__all__ = ["kda_ref", "make_inputs"]


def kda_ref(q, k, v, g, beta, scale=None, initial_state=None, output_final_state=False):
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

    print()
    print("Test Passed!" if ok else "FAILED")
