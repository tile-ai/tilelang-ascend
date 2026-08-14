"""KDA L0 acceptance test.

Four checks, in the order they buy the most confidence:

  [0] the two goldens against each other -- our own recurrence vs FLA's
      naive_recurrent_kda (fp32 inputs, strict 1e-5)
  [1] the kernel against both goldens (fp16 / bf16 inputs)
  [2] one shot vs segmented with the state relayed through final_state
  [3] an all-zero initial_state must equal passing none at all

Shape coverage: B=1/4, single and multi head, GVA (HV=2H / 4H), pure decode
(T=1), very short sequences, one full chunk length, several chunk lengths,
a non-divisible tail, K != V, fp16 and bf16 inputs, fp32 reference and state,
zero and non-zero initial state, four gate regimes (keep boundary / normal /
the lower_bound=-5 form K3 uses / extreme).

Two notes on method, both of which cost real debugging time to learn:

  * Both goldens run on CPU, always.  torch.einsum on the NPU dispatches to
    matmul and may accumulate at reduced precision -- measured, that drifts two
    references that should agree by 3e-4 at K=64/V=128, T=128.  A reference
    exists to define "correct"; it must not inherit the hardware's rounding.
  * The golden-vs-golden check must use fp32 inputs.  FLA's naive ends with
    `return o.to(dtype)`, so comparing at fp16 compares an fp32 result against
    a rounded one and reports fp16's relative precision (~5e-4) -- a number
    that says nothing about the algorithm.
"""

import contextlib
import os
import sys

import torch
import tilelang

from kda_ref import kda_ref as golden_b, make_inputs
from kda_recurrent import kda_recurrent

# FLA is the semantics and golden reference the task points at.  Lookup order:
#   1) an installed flash-linear-attention package
#   2) a local fla_ref/ copy next to this file (development only)
#   3) neither -- skip the FLA comparison; our own golden still runs
# Do not commit a copy of the FLA sources: they are third-party MIT code.
golden_a = None
try:
    from fla.ops.kda.naive import naive_recurrent_kda as golden_a  # noqa: E402
except ImportError:
    _local = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fla_ref")
    if os.path.isdir(_local):
        sys.path.insert(0, _local)
        with contextlib.suppress(ImportError):
            from naive import naive_recurrent_kda as golden_a  # noqa: E402

FP16_TOL, BF16_TOL = 5e-3, 3e-2

# B, SEQ, H, HV, K, V
CASES = [
    (1, 1, 1, 1, 64, 64),  # pure decode
    (1, 7, 2, 2, 64, 64),  # very short sequence + HV == H
    (4, 64, 4, 4, 64, 64),  # B=4, one full chunk length
    (2, 70, 2, 4, 64, 64),  # non-divisible tail + GVA (HV=2H)
    (1, 33, 1, 4, 128, 128),  # K3 head dim 128 + GVA (HV=4H)
    (2, 128, 2, 2, 64, 128),  # K != V, several chunk lengths
]
GATES = ("keep", "normal", "forget", "extreme")


def rel(x, r):
    x, r = x.detach().float().cpu(), r.detach().float().cpu()
    return (x - r).abs().max().item() / max(r.abs().max().item(), 1e-9)


def on_cpu(fn, q, k, v, g, beta, s0):
    args = [x.cpu() for x in (q, k, v, g, beta)]
    st = s0.cpu() if s0 is not None else None
    return fn(*args, initial_state=st, output_final_state=True)


def test_goldens_agree():
    print("[0] the two goldens against each other (fp32 inputs, strict 1e-5)")
    if golden_a is None:
        print("    FLA not found, skipping.  Install with: pip install flash-linear-attention")
        return True
    ok = True
    for B, SEQ, H, HV, K, V in CASES:
        worst = 0.0
        for gate in GATES:
            for ws in (False, True):
                q, k, v, g, beta, s0 = make_inputs(B, SEQ, H, HV, K, V, dtype=torch.float32, gate=gate, with_state=ws)
                o_b, s_b = on_cpu(golden_b, q, k, v, g, beta, s0)
                o_a, s_a = on_cpu(golden_a, q, k, v, g, beta, s0)
                worst = max(worst, rel(o_b, o_a), rel(s_b, s_a))
        good = worst < 1e-5
        ok &= good
        print(f"    B{B} T{SEQ:<4d} H{H} HV{HV} K{K:<3d} V{V:<3d}  max|A-B| = {worst:.2e}  {'ok' if good else 'FAIL'}")
    return ok


def test_kernel_vs_goldens():
    print("[1] kernel against both goldens (fp16 / bf16 inputs)")
    ok = True
    for B, SEQ, H, HV, K, V in CASES:
        for dt, tol in ((torch.float16, FP16_TOL), (torch.bfloat16, BF16_TOL)):
            worst_a = worst_b = 0.0
            for gate in GATES:
                for ws in (False, True):
                    q, k, v, g, beta, s0 = make_inputs(B, SEQ, H, HV, K, V, dtype=dt, gate=gate, with_state=ws)
                    o_k, s_k = kda_recurrent(q, k, v, g, beta, initial_state=s0, output_final_state=True)
                    o_b, s_b = on_cpu(golden_b, q, k, v, g, beta, s0)
                    worst_b = max(worst_b, rel(o_k, o_b), rel(s_k, s_b))
                    if golden_a is not None:
                        o_a, s_a = on_cpu(golden_a, q, k, v, g, beta, s0)
                        worst_a = max(worst_a, rel(o_k, o_a), rel(s_k, s_a))
            good = worst_b < tol and (golden_a is None or worst_a < tol)
            ok &= good
            tag = "bf16" if dt == torch.bfloat16 else "fp16"
            fla = "vs FLA=(skipped)  " if golden_a is None else f"vs FLA={worst_a:.2e}  "
            print(f"    B{B} T{SEQ:<4d} H{H} HV{HV} K{K:<3d} V{V:<3d} {tag}  {fla}vs ours={worst_b:.2e}  {'ok' if good else 'FAIL'}")
    return ok


def test_segmented():
    """Split into segments relayed through final_state; must equal one shot.

    One test covers three things at once: that initial_state is read correctly,
    that final_state really is the state at the end of that segment, and that
    nothing is lost handing the state between segments.
    """
    print("[2] one shot vs segmented with the state relayed")
    ok = True
    for B, SEQ, H, HV, K, V, cuts in [
        (2, 128, 2, 4, 64, 64, [64]),  # in half
        (2, 128, 2, 4, 64, 64, [1, 65]),  # decode-shaped: one token, then continue
        (1, 96, 1, 2, 64, 64, [30, 60]),  # three segments, cuts off the chunk grid
        (1, 70, 1, 4, 128, 128, [33]),  # K3 head dim + non-divisible
    ]:
        for gate in ("normal", "forget"):
            q, k, v, g, beta, s0 = make_inputs(B, SEQ, H, HV, K, V, dtype=torch.float16, gate=gate, with_state=True)
            o_all, s_all = kda_recurrent(q, k, v, g, beta, initial_state=s0, output_final_state=True)

            bounds = [0] + cuts + [SEQ]
            state, outs = s0, []
            for a, b in zip(bounds[:-1], bounds[1:]):
                # a token-axis slice is a non-contiguous view; the kernel wants contiguous
                sl = [x[:, a:b].contiguous() for x in (q, k, v, g, beta)]
                o_seg, state = kda_recurrent(*sl, initial_state=state, output_final_state=True)
                outs.append(o_seg)

            eo = rel(torch.cat(outs, dim=1), o_all)
            es = rel(state, s_all)
            good = eo < 1e-6 and es < 1e-6  # same kernel, same precision: should be bit-identical
            ok &= good
            print(f"    B{B} T{SEQ:<4d} cuts={str(cuts):12s} {gate:8s} o={eo:.1e} S={es:.1e}  {'ok' if good else 'FAIL'}")
    return ok


def test_zero_state():
    print("[3] an all-zero initial_state must equal passing none")
    q, k, v, g, beta, _ = make_inputs(2, 32, 2, 4, 64, 64, dtype=torch.float16)
    z = torch.zeros(2, 4, 64, 64, device=q.device, dtype=torch.float)
    o1, s1 = kda_recurrent(q, k, v, g, beta, output_final_state=True)
    o2, s2 = kda_recurrent(q, k, v, g, beta, initial_state=z, output_final_state=True)
    ok = torch.equal(o1, o2) and torch.equal(s1, s2)
    print(f"    bit-identical: {ok}")
    return ok


def main():
    tilelang.disable_cache()
    torch.manual_seed(0)

    ok = test_goldens_agree()
    ok &= test_kernel_vs_goldens()
    ok &= test_segmented()
    ok &= test_zero_state()

    print()
    if ok:
        print("Kernel Output Match!")
    else:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
