"""KDA L1 stage 3: A = (I + L)^{-1}, L strictly lower triangular, per chunk.

This is the one stage whose *math* needs no porting at all.  The gate never
enters the formula -- L already carries every decay factor, baked in by stage 2
-- so the recurrence below is the GDN kernel body unchanged.  What changed is
the interface: the tensor layout, the head axis, and the grid decomposition.

Algorithm (row-by-row forward substitution, on the vector core)
--------------------------------------------------------------
    let A = I + X;  (I + L)(I + X) = I  =>  X = -L - L X
    row i:          X_i = -L_i - L_i X_{0..i-1}

The kernel keeps the GDN sign convention: it holds X' = -X in ``x_ub``,
recurses ``x_ub[i] <- L[i] - sum_j x_ub[j] * x_ub[i, j]``, and finishes with
A = I - X'.  Substituting X' = -X reproduces ``kda_l1_ref.ref_solve_tril``
line for line, so the two are the same computation, not merely equivalent ones.

Forward substitution, not ``linalg.inv``: there is no NPU implementation of
inv, so it silently falls back to CPU and then NaN-poisons the whole matrix.
Rows 0 and 1 need no update -- row 0 is empty and row 1 has only the already
final row 0 to its left -- which is why the serial loop starts at 2.

Interface (frozen FLA contract, external layout)
------------------------------------------------
    Ltri  [B, SEQ, HV, C]   dtype   stage 2 output, == stage_tensors()["L"]
    Idt   [C, C]            fp32    identity, a compile-time constant
    A     [B, SEQ, HV, C]   dtype   stage 3 output, == stage_tensors()["A"]

    dtype is taken from the input tensor (fp16 / bf16 / fp32), never hardcoded.
    Only HV and C matter here: H, K, V, GRP and scale do not enter this stage,
    because both tensors are indexed by the value head hv only.  There is no
    Q/Kt read, hence no GVA remapping.

The strided tile
----------------
The token axis and the C axis sit on either side of HV, so one [C, C] chunk
matrix is a *strided* transfer: C contiguous elements per row, HV * C elements
between rows.  Slicing the token axis while pinning hv expresses exactly that:

    T.copy(Ltri[bz, t0 : t0 + C, hv, :], x_io)

The backend turns the resulting region extents [1, C, 1, C] into a row stride
of HV * C (every trailing unit extent folds the matching shape dim into the
stride), which is the number we want.  Rows are C * itemsize bytes, always a
multiple of 32 for C % 16 == 0, so both directions stay 32B aligned.

Assumption inherited from stage 2: L is *exactly* zero on and above the
diagonal.  The recurrence sums over all C rows rather than just j < i, and the
terms with j >= i vanish only because x_ub[i, j] is exactly 0 there.  The
reference masks explicitly, and the mask is folded into the exponent in the kkt
kernel, so the zeros are true zeros rather than underflowed ones.

Known limitation
----------------
First pass requires ``SEQ % C == 0``.  Tail blocks are deliberately left to the
next round (task spec: fixed length first, varlen later); ``kda_l1_ref`` already
pads for them, and the padding is neutral, so nothing here has to change when
that support lands -- only the grid bound does.

UB budget (limit 196352 B)
--------------------------
    x_ub, i_ub, mul_ub  3 * C * C * 4        C=64 -> 49152 B, C=32 -> 12288 B
    red_ub              C * 4                C=64 ->   256 B, C=32 ->   128 B
    x_io                C * C * itemsize     C=64 -> 8192 B (fp16) / 16384 (fp32)
    total, worst case C=64 fp32                        65792 B
"""

import os
import sys

import torch
import tilelang
from tilelang import language as T

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import kda_l1_ref  # noqa: E402

# Only AUTO_SYNC, same as the six GDN kernels.  MEMORY_PLANNING is deliberately
# left off: on the backward dot it aliased a reduction target with a temporary
# tile, and the stores wrote zeros while the registers held the right values.
pass_configs = {tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True}

VEC_NUM = 2


@tilelang.jit(out_idx=[-1], pass_configs=pass_configs)
def kda_solve_tril_ker(B, SEQ, HV, C, dtype="float16", accum_dtype="float"):
    N = SEQ // C  # chunks per sequence
    total = B * HV * N  # one [C, C] chunk matrix per task
    blocks = (total + VEC_NUM - 1) // VEC_NUM

    @T.prim_func
    def main(
        Ltri: T.Tensor([B, SEQ, HV, C], dtype),  # type: ignore
        Idt: T.Tensor([C, C], accum_dtype),  # type: ignore
        A: T.Tensor([B, SEQ, HV, C], dtype),  # type: ignore
    ):
        # The row recurrence is serial, so the two vector cores take two whole
        # chunk matrices rather than splitting one.  The chunk index is the
        # fastest-moving part of the task id, so for N > 1 the two cores of a
        # block sit in the same (batch, head) and read neighbouring GM.
        with T.Kernel(blocks, is_npu=True) as (cid, vid):
            raw = cid * VEC_NUM + vid
            # An odd task count leaves one core over.  Clamping is safe where
            # skipping would need runtime control flow: the surplus core redoes
            # the last task and stores byte-identical values to the same address,
            # so no interleaving of the two stores can produce a wrong result.
            pid = T.if_then_else(raw < total, raw, total - 1)
            n = pid % N
            hv = (pid // N) % HV
            bz = pid // (N * HV)
            t0 = n * C

            x_ub = T.alloc_ub([C, C], accum_dtype)  # L, then I - A, then A
            i_ub = T.alloc_ub([C, C], accum_dtype)  # identity
            mul_ub = T.alloc_ub([C, C], accum_dtype)  # reduction operand
            red_ub = T.alloc_ub([C], accum_dtype)  # C * 4 >= 32 B
            x_io = T.alloc_ub([C, C], dtype)  # GM staging, dtype side

            with T.Scope("V"):
                # Strided load of one chunk matrix; see the module docstring.
                T.copy(Ltri[bz, t0 : t0 + C, hv, :], x_io)
                T.copy(x_io, x_ub)
                T.copy(Idt[0, 0], i_ub)

                for i in range(2, C):
                    # mul_ub is fully overwritten below, and T.reduce_sum
                    # defaults to clear=True and initialises red_ub itself, so
                    # neither buffer gets a manual fill.
                    for j, k in T.Parallel(C, C):
                        mul_ub[j, k] = x_ub[j, k] * x_ub[i, j]
                    T.reduce_sum(mul_ub, red_ub, dim=0)
                    for j in T.Parallel(C):
                        x_ub[i, j] = x_ub[i, j] - red_ub[j]

                for i, j in T.Parallel(C, C):
                    x_ub[i, j] = i_ub[i, j] - x_ub[i, j]

                T.copy(x_ub, x_io)
                T.copy(x_io, A[bz, t0 : t0 + C, hv, :])

    return main


# --------------------------------------------------------------- host wrapper
_DTYPE = {
    torch.float16: "float16",
    torch.bfloat16: "bfloat16",
    torch.float32: "float",
}

_IDT_CACHE = {}


def _identity(C, device):
    """Cached [C, C] fp32 identity.

    A compile-time constant, not a transform of the data: at most 64x64 fp32 =
    16 KB, the same trick the GDN kernels use for their causal masks.  Cached so
    a pipeline that calls this per chunk does not reallocate it every time.
    """
    key = (C, str(device))
    idt = _IDT_CACHE.get(key)
    if idt is None:
        idt = torch.eye(C, device=device, dtype=torch.float)
        _IDT_CACHE[key] = idt
    return idt


def kda_solve_tril(Ltri):
    """A = (I + L)^{-1} per chunk, both in external [B, SEQ, HV, C] layout.

    C is the trailing dimension of the input -- stage 2 emits one C-wide row per
    token -- so there is no separate chunk-length argument to keep in sync.
    """
    assert Ltri.dim() == 4, f"expected [B, SEQ, HV, C], got {tuple(Ltri.shape)}"
    B, SEQ, HV, C = Ltri.shape
    assert Ltri.dtype in _DTYPE, f"unsupported dtype {Ltri.dtype}"
    # The kernel derives its GM strides from the declared shape, so a transposed
    # or sliced view would be read with the wrong row stride.  Fail loudly.
    assert Ltri.is_contiguous(), "Ltri must be contiguous in [B, SEQ, HV, C]"
    assert SEQ % C == 0, f"first pass needs SEQ % C == 0, got SEQ={SEQ} C={C}"
    assert C % 16 == 0, f"C must be a multiple of 16 to keep rows 32B aligned, got {C}"
    # 3 fp32 [C, C] tiles plus the dtype staging tile must fit in 196352 B.
    assert C <= 64, f"C={C} needs {3 * C * C * 4 + C * C * 4} B of UB, over budget"

    # SEQ == 0 slips past the assert above (0 % C == 0) and would launch a
    # zero-block grid over an unwritten output.  A zero-length sequence is legal
    # input; there is no block to invert, so A is empty along the token axis.
    if SEQ == 0:
        return torch.empty((B, 0, HV, C), device=Ltri.device, dtype=Ltri.dtype)

    ker = kda_solve_tril_ker(B, SEQ, HV, C, dtype=_DTYPE[Ltri.dtype])
    return ker(Ltri, _identity(C, Ltri.device))


# ------------------------------------------------------------------- testing
_TAG = {torch.float16: "fp16", torch.bfloat16: "bf16", torch.float32: "fp32"}

# Kernel-only budget: the recurrence runs in fp32 internally, so this is
# dominated by rounding A back to the transport dtype on the way out.
_TOL = {torch.float16: 5e-3, torch.bfloat16: 3e-2, torch.float32: 1e-5}


def _relerr(got, ref):
    got, ref = got.float(), ref.float()
    return (got - ref).abs().max().item() / max(ref.abs().max().item(), 1e-9)


def _case(B, SEQ, H, HV, K, V, C, gate, dtype):
    """One shape against the stage-3 golden out of the L1 reference pipeline.

    Inputs are built on CPU in fp32 so the reference is computed exactly; only
    the tensor the kernel actually consumes goes to the device.
    """
    q, k, v, g, beta, _ = kda_l1_ref.make_inputs(B, SEQ, H, HV, K, V, device="cpu", dtype=torch.float32, gate=gate)
    st = kda_l1_ref.stage_tensors(q, k, v, g, beta, C=C)

    # stage_tensors returns external layout as a transposed view.  .contiguous()
    # is a test-side materialisation, not part of the kernel path: in the real
    # pipeline L arrives already contiguous in [B, SEQ, HV, C] from the kkt
    # kernel, which is why the wrapper asserts contiguity rather than fixing it.
    l_in = st["L"].contiguous().to(device="npu", dtype=dtype)
    got = kda_solve_tril(l_in).float().cpu()

    # Two references, because two different errors are worth separating.
    # a_same re-runs the reference on the *rounded* matrix the kernel was handed,
    # so e_kern is the kernel alone.  e_gold is against the pipeline golden and
    # additionally carries the amplification of that rounding through A dL A,
    # which e_sens measures on the reference itself -- hence the adaptive bound
    # instead of a hand-picked loose constant.  For fp32 the input is exact,
    # e_sens is 0, and the golden check tightens to _TOL.
    a_same = kda_l1_ref.ref_solve_tril(l_in.float().cpu().transpose(1, 2)).transpose(1, 2)
    e_kern = _relerr(got, a_same)
    e_gold = _relerr(got, st["A"])
    e_sens = _relerr(a_same, st["A"])

    finite = bool(torch.isfinite(got).all().item())
    ok = finite and e_kern < _TOL[dtype] and e_gold <= 4.0 * e_sens + _TOL[dtype]
    print(
        f"  B{B} T{SEQ:<4d} H{H} HV{HV} K{K:<3d} V{V:<3d} C{C:<2d} "
        f"{_TAG[dtype]} {gate:8s} kern={e_kern:.2e} gold={e_gold:.2e} "
        f"sens={e_sens:.2e}  {'ok' if ok else 'FAIL'}"
    )
    return ok


def main():
    tilelang.disable_cache()
    torch.manual_seed(0)

    ok = True

    print("== chunk length x GVA (fp16) ==")
    for C in (32, 64):
        for HV in (2, 4):  # HV == H, then HV == 2 * H
            for gate in ("normal", "forget"):
                ok &= _case(2, 256, 2, HV, 64, 64, C, gate, torch.float16)

    print("== K3 head dimension, K = V = 128 (fp16) ==")
    for gate in ("normal", "forget"):
        ok &= _case(1, 256, 1, 1, 128, 128, 64, gate, torch.float16)

    print("== grid edges (fp16) ==")
    # total == 1: odd task count, exercises the surplus-core clamp
    ok &= _case(1, 32, 1, 1, 64, 64, 32, "normal", torch.float16)
    # HV == 3 * H and HV not a power of two
    ok &= _case(1, 128, 2, 6, 64, 64, 64, "forget", torch.float16)

    print("== dtype passthrough ==")
    ok &= _case(2, 256, 2, 2, 64, 64, 64, "normal", torch.bfloat16)
    ok &= _case(2, 256, 2, 4, 64, 64, 32, "forget", torch.bfloat16)
    # fp32 in, fp32 out: the input is exact, so these two gate the kernel
    # straight against the pipeline golden at 1e-5.
    ok &= _case(2, 256, 2, 2, 64, 64, 64, "forget", torch.float32)
    ok &= _case(1, 256, 1, 1, 128, 128, 64, "normal", torch.float32)

    if ok:
        print("Kernel Output Match!")
    else:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
