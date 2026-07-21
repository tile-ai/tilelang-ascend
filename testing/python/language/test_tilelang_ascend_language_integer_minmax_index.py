import pytest

import torch

import tilelang
import tilelang.language as T

"""
Regression test for integer scalar ``max`` / ``min`` in the AscendC codegen.

A kernel that uses a *runtime* (non-constant-foldable) scalar ``T.max`` / ``T.min``
on a grid-block index lowers to a ``Max`` / ``Min`` IR node whose operands have
mismatched integer widths -- an int64 block variable against an int literal.

The non-PTO ``CodeGenTileLangAscend`` backend inherits ``CodeGenC``'s default and
prints these as a bare ``max(a, b)`` / ``min(a, b)``.  bisheng then rejects the
generated kernel with ``error: call to 'max' is ambiguous`` (the int64/int
overloads are equally viable), so the kernel fails to compile.  Emitting the
integer case as a ternary ``(a > b ? a : b)`` / ``(a < b ? a : b)`` removes the
overload resolution entirely.

The clamped index is used as a DMA row offset so it survives dead-code
elimination and actually reaches codegen.  ``ascendc`` exercises the affected
path; ``pto`` already emitted ``std::max`` / ``std::min`` and is included to show
it stays correct.
"""

# CV-combine is harmless for a pure-vector (no Cube) kernel and matches the
# passing element-wise / gm-to-ub suites.
VEC_PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}

TARGETS = ["ascendc", "pto"]


@pytest.fixture(scope="session", autouse=True)
def clear_cache():
    """Clear the tilelang cache before the session (stale kernels would mask a
    codegen regression -- a rebuilt kernel could otherwise return a cached one)."""
    tilelang.cache.clear_cache()
    yield


def gm_ub_gm_int_minmax_index(M, N, block_M, op, dtype, use_vid=True):
    """GM -> UB -> GM tile gather whose *source* tile is chosen by a clamped
    integer index.

    Each block ``cid`` stores to its own tile ``cid`` but loads from a neighbour
    tile picked by a scalar ``T.max`` / ``T.min`` clamp on ``cid``.  The clamp is
    the whole point of the test: it forces a runtime integer max/min into the
    generated code."""
    VEC_NUM = 2
    m_num = M // block_M
    rows = block_M // VEC_NUM if use_vid else block_M

    @T.prim_func
    def main(A: T.Tensor((M, N), dtype), C: T.Tensor((M, N), dtype)):  # type: ignore
        with T.Kernel(m_num, is_npu=True) as (cid, vid):
            # --- regression trigger --------------------------------------
            # ``cid`` is an int64 grid-block variable.  Mixing it with an int
            # literal inside a scalar max/min yields a mismatched-width integer
            # Max/Min node.  Used as a row index below, so it cannot be folded
            # away and must be printed by codegen.  ``op`` is a Python string,
            # so this ternary is resolved at trace time (like the ``use_vid``
            # ternary below) and yields a single Max/Min node; a statement-level
            # ``if`` would instead parse as device-side control flow and scope
            # ``src`` to the branch.
            src = T.max(cid - 1, 0) if op == "max" else T.min(cid + 1, m_num - 1)

            a_ub = T.alloc_ub((rows, N), dtype)
            voff = vid * rows if use_vid else 0
            # Gather the clamped source tile, then store to this block's tile.
            T.copy(A[src * block_M + voff, 0], a_ub)
            T.copy(a_ub, C[cid * block_M + voff, 0])

    return main


def _ref_clamped_gather(a, block_M, m_num, op):
    """Golden: tile ``cid`` of the output equals the clamped-neighbour tile of
    the input, matching the kernel's ``T.max`` / ``T.min`` index."""
    ref = torch.empty_like(a)
    for cid in range(m_num):
        src = max(cid - 1, 0) if op == "max" else min(cid + 1, m_num - 1)
        ref[cid * block_M : (cid + 1) * block_M] = a[src * block_M : (src + 1) * block_M]
    return ref


def run_test_int_minmax_index(M, N, block_M, op, dtype, target, use_vid=True):
    torch.manual_seed(0)
    func = gm_ub_gm_int_minmax_index(M, N, block_M, op, dtype, use_vid)
    func = tilelang.compile(func, out_idx=[-1], pass_configs=VEC_PASS_CONFIGS, target=target)
    a = torch.randn(M, N, dtype=torch.float32).npu()
    torch.npu.synchronize()
    c = func(a)
    ref = _ref_clamped_gather(a, block_M, M // block_M, op)
    torch.testing.assert_close(c, ref, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize("op", ["max", "min"])
def test_int_minmax_index(op, target):
    # 1024x256 fp32, 128-row tiles -> 8 blocks.  ``cid`` is an int64 grid var, so
    # the clamp lowers to an integer max/min that codegen must emit without an
    # ambiguous bare call.  ``ascendc`` covers the fixed path; ``pto`` proves it
    # is unaffected.
    run_test_int_minmax_index(1024, 256, 128, op, "float", target)
