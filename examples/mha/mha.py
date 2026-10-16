"""MHA (Multi-Head Attention) forward, BSHD layout, Expert mode.

    y = softmax(Q @ K^T * s) @ V,   s = scaleValue if scaleValue > 0 else 1/sqrt(D)

is_causal masks scores[..., i, j] to -inf for j > i + (S_kv - S): the mask is
bottom-right aligned. All-masked rows are reachable only out-of-domain and yield
0 through the sumexp + 1e-30 guard. Q/K/V are fp16 or bf16; the score mainline
and the O accumulator are fp32.

File layout. This file is the wrapper layer: the host-side metadata contract, the
three-way form dispatch, and the runnable self-check. The kernels live in four
sibling modules, and this file holds no T.* code at all:

    mha_common.py            pass_configs, NUM_STAGES / CROSS_INTERVAL,
                             _cube_core_num(), the entire host-side config
                             derivation (derive_config -> MhaConfig), the
                             dispatch predicate (select_form), and make_emit_v,
                             the factory for the shared V-scope @T.macro
    mha_form_a.py            mha_form_a_kernel          Form A
    mha_form_b_mma.py        mha_form_b_mma_kernel      Form B-mma
    mha_form_b_gemm_v0.py    mha_form_b_gemm_v0_kernel  Form B gemm_v0

mha_kernel() below keeps the signature and the return-value semantics it had when
all of this was one file: it picks the form and returns that form's compiled
kernel, so `mha_kernel(B=..., S=...)` still yields an object exposing
.get_kernel_source() and still callable as kernel(query, key, value). Calling one
of the three form factories directly builds the identical kernel; test_mha.py
guards that the dispatch is mutually exclusive and exhaustive.

Architecture. The host selects one of three mutually exclusive and exhaustive
C-scope forms from (S, D, S_kv) alone, so a generated kernel holds exactly one.
Form A (S > 16, D <= 128) is a T.mma full pipeline: L0A/L0B/L0C double slots with
manual side = i % 2 ping-pong, ZN/NZ annotated L1 operands and fine-grained
per-pipe-pair event flags. When block_N == D the GEMM2 L0 pool aliases the GEMM1
pool, and in the single-KV-block sub-case the FIX pipe writes Output straight
from L0C with a runtime row extent, because O = P_hat @ V is already final there.
Form B-mma (S > 16, D == 256) is a manual K/N-split T.mma pipeline: GEMM1 splits
K into two KC = 128 chunks chained by mma(init=True -> False), GEMM2 splits N
into two NC = 128 chunks that accumulate into a narrow L0C pool and are each
written to their own workspace_3 column segment. The split exists because
gemm_v0 drains the M pipe at every step at this width, and because an mma
targeting an L0C column sub-region is rejected by the cube unit. Form B gemm_v0
(S <= 16, all D) is the decode fallback: T.gemm_v0 manages L0A/L0B internally
through the kL0Size tiers 32/64/128, and its self-balanced prologue/epilogue pipe
flags cover every segment boundary, so no coarse barrier is needed.

The V scope is shared by all three forms through emit_v, which is a @T.macro and
must stay one. TVMScript resolves T.* frames at parse time, so a plain Python
emitter called from inside a prim_func has its T.* emissions silently dropped:
the kernel still compiles, but the generated body is missing and the aicore hangs
on an orphaned wait.
It is defined once, in mha_common.py, and bound to one configuration by
make_emit_v(); no form module holds a copy, and all three obtain theirs from that
single call.

Pipeline shape. C and V run NUM_STAGES = 8 deep batches and exchange data through
three explicit GM workspaces indexed [core, stage]: workspace_1 = fp32 scores,
workspace_2 = P, workspace_3 = fp32 PV. Inline comments abbreviate these as
w1/w2/w3, and a "colseg" write is a column-segment store. T.copy(L0C -> UB) and
T.copy(UB -> L1) have no device implementation, so that GM round trip is the only
physical C/V route. T.Pipelined is unavailable in this mode; the manual
side = i % 2 slot rotation expresses the pipeline instead. Each AIV owns
half_M = block_M // 2 rows (the vid row split) and writes back
min(half_M, max(0, valid_m - vid * half_M)) rows under a runtime guard.

Multi-KV-block prefill walks a flat (task, kv) slot stream so a slot batch can
cross task boundaries without draining. Under causal masking eff_kv_iters varies
per task at runtime, so the flat index -> (task, kv) mapping is a scalar walk
state machine that C and V compute identically, never via a workspace.

Precision invariants. These must stay bit-identical to the golden reference; do
not reorder or refold any of them.
1. Score double-rounding grid: fp32 QK result -> round to dtype (r16(qk)) ->
   multiply by s in fp32 -> round to dtype again (r16(scores)).
2. Single-KV-block specialization: P is divided by (sumexp + 1e-30) BEFORE the
   dtype rounding, matching the golden one-shot softmax; phase 2 then skips the
   rescale and the final division.
3. sumexp + 1e-30 on every final normalization, so an all-masked row gives 0
   instead of 0/0.
4. Mask chain semantics: a pre-exp -inf select built from a 2D compare against
   hoisted, task-invariant row thresholds, with fully visible KV blocks skipped
   at runtime by a block-uniform condition.

Sync model. Every event id is statically budgeted: init pre-sets the tokens a
first consumer would otherwise wait for, destroy consumes exactly the outstanding
ones, and each iteration is net zero. Cross-core semaphores are strictly 1:1
Set/Wait. A conditional DMA must carry ALL its flags inside the same condition --
a copy guarded while its flag is not (or the reverse) corrupts the run.

Constraints. D in {64, 128, 256}; fp16/bf16; contiguous BSHD; 1 <= B <= 128,
1 <= S <= 2048, 1 <= S_kv <= 4096, 1 <= N <= 64; is_causal requires S <= S_kv.
Every trip count is a compile-time constant or derived from one, never from tensor
data. The 32 KB L0B ping-pong budget and the 511.75 KB L1 budget are what fix the
block_N and gemm1_kL0 tiers. MTE2_MTE1 event ids 6 and 7 are unusable in
static-tensor code. The kv_db decode path (S <= 16, D == 256, more than one KV
block) runs on stock tilelang-ascend: its T.gemm_v0 calls keep the default
prologue, whose whole-MTE2-pipe drain makes every operand enqueued before the
call resident, so that path needs no framework-side patch.

DESIGN.md in this directory derives the tiling tiers, the event-id budget and
the precision gates that test_mha.py checks.

Run this file directly for a self-checking example against a torch reference:
    python examples/mha/mha.py                    # fp16, B=2 S=S_kv=128 N=8 D=64
    python examples/mha/mha.py --causal --D 128   # --help lists every option
"""

import argparse

import torch

from mha_common import FORM_A, FORM_B_MMA, FORM_B_GEMM_V0, select_form
from mha_form_a import mha_form_a_kernel
from mha_form_b_gemm_v0 import mha_form_b_gemm_v0_kernel
from mha_form_b_mma import mha_form_b_mma_kernel

# Form -> jit factory. Built once at import; the decorator is lazy, so importing
# this module compiles nothing.
_FORM_FACTORIES = {
    FORM_A: mha_form_a_kernel,
    FORM_B_MMA: mha_form_b_mma_kernel,
    FORM_B_GEMM_V0: mha_form_b_gemm_v0_kernel,
}


def mha_kernel(
    B,
    S,
    S_kv,
    N,
    D,
    dtype="float16",
    scaleValue=-1.0,
    is_causal=False,
):
    """Build the kernel for one (shape, dtype, attr) point.

    Wrapper-layer dispatcher. The form is chosen from (S, D) by
    mha_common.select_form() -- the same single source the dispatch-guard test
    reads -- and the corresponding form factory builds the kernel. Host-side
    config derivation happens inside that factory (mha_common.derive_config()),
    so nothing is derived twice with a chance to diverge.

    Signature and return value are unchanged from the single-file layout of this
    operator: the result exposes .get_kernel_source() and is callable as
    kernel(query, key, value).
    """
    form = select_form(B, S, S_kv, N, D, dtype, scaleValue, is_causal)
    return _FORM_FACTORIES[form](
        B=B,
        S=S,
        S_kv=S_kv,
        N=N,
        D=D,
        dtype=dtype,
        scaleValue=scaleValue,
        is_causal=is_causal,
    )


def mha(query, key, value, scaleValue=-1.0, is_causal=False):
    """Host wrapper: metadata validation + JIT dispatch only (zero tensor ops).

    Args:
        query: [B, S, N, D] float16/bfloat16, contiguous, finite values.
        key: [B, S_kv, N, D] same dtype; B/N/D must match query.
        value: [B, S_kv, N, D] same as key.
        scaleValue: <= 0 -> auto 1/sqrt(D); > 0 -> used as-is (JIT constant).
        is_causal: bottom-right aligned causal mask; requires S <= S_kv.

    Returns:
        [B, S, N, D] output, same dtype as query, contiguous (framework-allocated).
    """
    assert query.dtype == key.dtype == value.dtype, "Q/K/V dtypes must match"
    assert query.dtype in (torch.float16, torch.bfloat16), f"unsupported dtype {query.dtype}; only float16/bfloat16"
    assert query.is_contiguous() and key.is_contiguous() and value.is_contiguous(), "Q/K/V must be contiguous BSHD tensors"
    B, S, N, D = query.shape
    S_kv = key.shape[1]
    assert tuple(key.shape) == (B, S_kv, N, D), "key shape must be (B, S_kv, N, D)"
    assert tuple(value.shape) == (B, S_kv, N, D), "value shape must equal key shape"
    assert D in (64, 128, 256), f"D must be in {{64, 128, 256}}, got {D}"
    assert 1 <= B <= 128, f"B out of range [1, 128]: {B}"
    assert 1 <= S <= 2048, f"S out of range [1, 2048]: {S}"
    assert 1 <= S_kv <= 4096, f"S_kv out of range [1, 4096]: {S_kv}"
    assert 1 <= N <= 64, f"N out of range [1, 64]: {N}"
    if is_causal:
        assert S_kv >= S, "is_causal=True requires S <= S_kv (else all-masked rows)"
    kernel = mha_kernel(
        B=B,
        S=S,
        S_kv=S_kv,
        N=N,
        D=D,
        dtype=str(query.dtype).removeprefix("torch."),
        scaleValue=float(scaleValue),
        is_causal=bool(is_causal),
    )
    return kernel(query, key, value)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Self-checking MHA forward example (BSHD layout).")
    parser.add_argument("--B", type=int, default=2, help="batch size, 1..128")
    parser.add_argument("--S", type=int, default=128, help="query length, 1..2048; S <= 16 selects the decode path")
    parser.add_argument("--S-kv", dest="S_kv", type=int, default=128, help="key/value length, 1..4096")
    parser.add_argument("--N", type=int, default=8, help="head count, 1..64")
    parser.add_argument("--D", type=int, default=64, choices=(64, 128, 256), help="head dim")
    parser.add_argument("--dtype", default="float16", choices=("float16", "bfloat16"), help="Q/K/V dtype")
    parser.add_argument("--scale", type=float, default=-1.0, help="softmax scale; <= 0 selects 1/sqrt(D)")
    parser.add_argument("--causal", action="store_true", help="apply the bottom-right aligned causal mask")
    args = parser.parse_args()

    def ref_mha(q, k, v, scale, causal):
        """Reference on the kernel's own rounding grid: QK^T -> *scale -> softmax -> @V,
        computed in the input dtype over [B, N, S, S_kv] scores."""
        S, S_kv = q.shape[1], k.shape[1]
        scores = torch.matmul(q.transpose(1, 2), k.transpose(1, 2).transpose(-2, -1)) * scale
        if causal:
            i = torch.arange(S).unsqueeze(-1)
            j = torch.arange(S_kv).unsqueeze(0)
            scores = scores.masked_fill(j > i + (S_kv - S), float("-inf"))
        # A fully masked row would leave softmax with 0/0; force its weights to 0.
        row_max = scores.max(dim=-1, keepdim=True).values
        weights = torch.nn.functional.softmax(scores, dim=-1)
        weights = torch.where(torch.isinf(row_max) & (row_max < 0), torch.zeros_like(weights), weights)
        return torch.matmul(weights, v.transpose(1, 2)).transpose(1, 2).contiguous()

    scale = args.scale if args.scale > 0 else 1.0 / args.D**0.5
    dtype = getattr(torch, args.dtype)
    torch.manual_seed(0)
    query = torch.randn(args.B, args.S, args.N, args.D, dtype=dtype)
    key = torch.randn(args.B, args.S_kv, args.N, args.D, dtype=dtype)
    value = torch.randn(args.B, args.S_kv, args.N, args.D, dtype=dtype)

    out = mha(query.to("npu"), key.to("npu"), value.to("npu"), scaleValue=args.scale, is_causal=args.causal)
    torch.npu.synchronize()
    ref = ref_mha(query, key, value, scale, args.causal)

    # The mixed-tolerance gate test_mha.py uses: at least 99% of the elements
    # inside atol + rtol*|ref|, and no element past the dtype's absolute ceiling.
    atol, rtol, max_abs_limit = {"float16": (2**-14, 2**-9, 1e-1), "bfloat16": (2**-10, 2**-6, 1e0)}[args.dtype]
    out_f, ref_f = out.detach().cpu().float(), ref.float()
    abs_err = (out_f - ref_f).abs()
    matched = (abs_err <= atol + rtol * ref_f.abs()).float().mean().item()
    max_abs = abs_err.max().item()

    print(f"MHA B={args.B} S={args.S} S_kv={args.S_kv} N={args.N} D={args.D} dtype={args.dtype} causal={args.causal} scale={scale:.6f}")
    print(f"matched={matched:.5f} (gate >= 0.99)   max_abs_err={max_abs:.3e} (gate <= {max_abs_limit:g})")
    assert matched >= 0.99, f"only {matched:.5f} of elements are within tolerance (gate 0.99)"
    assert max_abs <= max_abs_limit, f"max_abs_err {max_abs:.3e} exceeds the {max_abs_limit:g} ceiling"
    print("Test Passed!")
