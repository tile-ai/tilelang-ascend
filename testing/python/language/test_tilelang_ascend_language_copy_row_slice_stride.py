import re

import pytest
import torch

import tilelang
import tilelang.language as T

"""
Regression test for the row-slice strideN bug (issue #1263) in the Ascend
GM<->UB copy / atomic-add lowering (src/op/ascend.cc :: compute_strideN).

Feature under test
------------------
A GM copy is emitted as a 2D DMA: ``blockCount`` rows (the second-to-last
active dim, i.e. the highest dim below the last whose extent != 1) x
``blockLen`` columns (the last dim), with one uniform stride between
consecutive rows. That stride is the product of every buffer dim below the row
dim, so extent-1 dims sitting under the row dim must be folded into it (e.g.
``Query[cid, m:m+BM, n2, g*D:(g+1)*D]`` folds the scalar ``n2`` dim).

The bug: when EVERY leading extent is 1 -- a row slice like ``C2[row, 0:N]``
of a 2D ``(rows, N)`` buffer -- the old ``compute_strideN`` folded the leading
dims in as well and returned the WHOLE BUFFER SIZE (``N * rows``) instead of
the row pitch ``N``. At issue scale this produced ``realDstN = 2**30`` where
``128`` was expected, and the DMA corrupted data on the reporter's device
(CANN 8.3.RC2).

The fix: fold the extent-1 run only when a real row dim (extent != 1) exists
below it; for a row slice the stride is just the last buffer dim.

How these tests trigger it
--------------------------
- ``_scatter_kernel`` is the issue's kernel (scaled down): 3D and 2D row-slice
  GM<->UB copies through a packed 2D buffer. Codegen tests assert the emitted
  strideN equals the row width (128), not the buffer size; the runtime test
  checks the scatter-add numerics in-place (``out_idx=[]``).
- ``_atomic_kernel`` drives the same row-slice shape through
  ``AscendAtomicAdd::Lower`` (``T.tile.atomic_add`` with a row-slice GM dst).
- ``_fold_kernel`` is the lightning-indexer-style pattern the folding logic
  was built for (a scalar dim between the row dim and the column dim). Its
  stride MUST stay folded (``N2 * G * D``); this guards against over-fixing
  the bug by always returning the last dim.

Codegen tests use ``tilelang.lower`` (no device needed) and run on both
backends; runtime correctness tests target ascendc (see the note above them for
the PTO toolchain situation) and require an Ascend NPU.
"""

BLOCK_M, BLOCK_N = 8, 128
N_TILES = 2  # N // BLOCK_N
M_TILES = 2  # M // BLOCK_M
HALF_N_TILES = N_TILES // 2
HALF_TILES = M_TILES * HALF_N_TILES
PACKED_ROWS = M_TILES * BLOCK_M * N_TILES  # 32
DTYPE = "float32"

# Lightning-indexer-style fold pattern geometry.
F_BM, F_N2, F_G, F_D = 4, 3, 2, 64

PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: False,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}


@pytest.fixture(scope="session", autouse=True)
def clear_cache():
    tilelang.disable_cache()
    yield


def _scatter_kernel():
    @T.prim_func
    def scatter_add_left(
        CP: T.Tensor((HALF_TILES, BLOCK_M, BLOCK_N), DTYPE),  # type: ignore
        C2: T.Tensor((PACKED_ROWS, BLOCK_N), DTYPE),  # type: ignore
    ):
        with T.Kernel(HALF_TILES, threads=1, is_npu=True) as cid:
            bm = cid // HALF_N_TILES
            bn = cid % HALF_N_TILES
            row = bm * BLOCK_M
            cp_ub = T.alloc_ub((BLOCK_N,), DTYPE)
            c_ub = T.alloc_ub((BLOCK_N,), DTYPE)
            for rr in T.serial(BLOCK_M):
                out_row = (row + rr) * N_TILES + bn
                T.copy(CP[cid, rr, 0:BLOCK_N], cp_ub)
                T.copy(C2[out_row, 0:BLOCK_N], c_ub)
                T.tile.add(c_ub, c_ub, cp_ub)
                T.copy(c_ub, C2[out_row, 0:BLOCK_N])

    return scatter_add_left


def _atomic_kernel(rows=16):
    @T.prim_func
    def atomic_row_slice(
        SRC: T.Tensor((rows, BLOCK_N), DTYPE),  # type: ignore
        C2: T.Tensor((rows, BLOCK_N), DTYPE),  # type: ignore
    ):
        with T.Kernel(rows, is_npu=True) as (cid, vid):
            row_ub = T.alloc_ub((BLOCK_N,), DTYPE)
            T.copy(SRC[cid, 0:BLOCK_N], row_ub)
            T.tile.atomic_add(C2[cid, 0:BLOCK_N], row_ub)

    return atomic_row_slice


def _fold_kernel():
    @T.prim_func
    def fold_middle_scalar(
        Q: T.Tensor((1, F_BM, F_N2, F_G * F_D), DTYPE),  # type: ignore
        O: T.Tensor((1, F_BM, F_G, F_D), DTYPE),  # type: ignore
    ):
        with T.Kernel(F_G, is_npu=True) as (cid, vid):
            q_ub = T.alloc_ub((F_BM, F_D), DTYPE)
            T.copy(Q[0, 0:F_BM, 1, cid * F_D : (cid + 1) * F_D], q_ub)
            T.copy(q_ub, O[0, 0:F_BM, cid, 0:F_D])

    return fold_middle_scalar


def _ascendc_copy_strides(source):
    """Extract strideN (the first runtime arg after the two tensor operands)
    from every GM copy / atomic-add call in generated AscendC source.

    The printed form is ``copy_gm_to_ub<T, ...>(dst[off], src[off], strideN,
    validRow, validCol, ...)``; the first ``"], <int>, "`` in the line is the
    boundary between the source tensor operand and strideN (offset
    expressions only contain parentheses, and the UB operand is followed by
    the GM operand's name, not a digit)."""
    strides = []
    for line in source.splitlines():
        if not any(op in line for op in ("copy_gm_to_ub<", "copy_ub_to_gm<", "atomic_add_ub_to_gm<")):
            continue
        match = re.search(r"\], (\d+), ", line)
        assert match is not None, f"no strideN found in copy call: {line.strip()}"
        strides.append(int(match.group(1)))
    return strides


def test_row_slice_copy_stride_codegen_ascendc():
    """Row-slice GM<->UB copies must use the row width (128) as strideN, not
    the whole-buffer size (32*128 for C2, 2*8*128 for CP)."""
    source = tilelang.lower(_scatter_kernel(), target="ascendc").kernel_source
    strides = _ascendc_copy_strides(source)
    # CP gm2ub, C2 gm2ub, C2 ub2gm.
    assert len(strides) == 3, f"expected 3 GM copy calls, got {len(strides)}: {strides}"
    assert strides == [BLOCK_N, BLOCK_N, BLOCK_N], (
        f"row-slice strideN must be the row width {BLOCK_N}, got {strides} (the pre-fix bug returned the whole-buffer size)"
    )


def test_row_slice_atomic_add_stride_codegen_ascendc():
    """The AscendAtomicAdd fallback path shares the same strideN logic."""
    source = tilelang.lower(_atomic_kernel(), target="ascendc").kernel_source
    strides = _ascendc_copy_strides(source)
    assert len(strides) == 2, f"expected 2 GM calls, got {len(strides)}: {strides}"
    assert strides == [BLOCK_N, BLOCK_N], f"row-slice atomic-add strideN must be {BLOCK_N}, got {strides}"


def test_fold_middle_scalar_stride_codegen_ascendc():
    """An extent-1 dim between the row dim and the column dim must still be
    folded into the stride: for Q[1, BM, N2, G*D] the stride is N2*G*D."""
    source = tilelang.lower(_fold_kernel(), target="ascendc").kernel_source
    strides = _ascendc_copy_strides(source)
    assert len(strides) == 2, f"expected 2 GM copy calls, got {len(strides)}: {strides}"
    # gm2ub folds the scalar N2 dim below the row dim; ub2gm into O folds the
    # G dim of O[1, BM, G, D] the same way.
    assert strides == [F_N2 * F_G * F_D, F_G * F_D], (
        f"folded strides expected, got {strides} (always-return-last-dim would break the multi-row DMA)"
    )


@pytest.mark.parametrize("target", ["ascendc", "pto"])
def test_row_slice_copy_stride_codegen(target):
    """Both backends consume the same lowered strideN arg. On the PTO side it
    drives MergeShapeBySrcN/ComputeStrides: the row-slice stride (128) keeps
    the buffer's true row-major strides, while the pre-fix whole-buffer stride
    merged the entire buffer into one flat dim (the row-pitch slot held the
    total element count instead of the row width)."""
    source = tilelang.lower(_scatter_kernel(), target=target).kernel_source
    if target == "ascendc":
        assert _ascendc_copy_strides(source) == [BLOCK_N] * 3
    else:
        # C2 is (32, 128): row-major strides (4096, 128, 1) survive; pre-fix
        # this call collapsed to Stride<1, 1, 1, 4096, 1>.
        assert f"pto::Stride<1, 1, {PACKED_ROWS * BLOCK_N}, {BLOCK_N}, 1>" in source
        # CP is (2, 8, 128): row-major strides (2048, 1024, 128, 1) survive;
        # pre-fix this call collapsed to Stride<1, 1, 1, 2048, 1>.
        assert "pto::Stride<1, 2048, 1024, 128, 1>" in source
        assert f"pto::Stride<1, 1, 1, {PACKED_ROWS * BLOCK_N}, 1>" not in source
        assert "pto::Stride<1, 1, 1, 2048, 1>" not in source


@pytest.mark.parametrize("target", ["ascendc", "pto"])
def test_fold_middle_scalar_stride_codegen(target):
    """The fold pattern's strides keep the folded pitch (Q: N2*G*D = 384
    between rows; O: G*D = 128) instead of a last-dim-only stride."""
    source = tilelang.lower(_fold_kernel(), target=target).kernel_source
    if target == "ascendc":
        assert _ascendc_copy_strides(source) == [F_N2 * F_G * F_D, F_G * F_D]
    else:
        # Q merges to (1, BM, N2*G*D): the row-pitch slot holds N2*G*D.
        assert (f"pto::Stride<1, {F_BM * F_N2 * F_G * F_D}, {F_BM * F_N2 * F_G * F_D}, {F_N2 * F_G * F_D}, 1>") in source
        # O merges to (1, BM, G*D): the row-pitch slot holds G*D.
        assert (f"pto::Stride<1, {F_BM * F_G * F_D}, {F_BM * F_G * F_D}, {F_G * F_D}, 1>") in source


# The runtime correctness tests below target the ascendc backend only.
#
# On the PTO side the corrected strideN changes the emitted
# copy_gm_to_ub_dynamic template from the pre-fix flat Stride<..., 1, total, 1>
# (the whole-buffer stride merged every dim into one flat dim) to the true
# row-major Stride<..., total, row_width, 1>. That strided TLOAD instantiation
# fails to COMPILE on the CI NPU runner's toolchain (CANN 9.1.0-beta.1 bisheng
# rejects pto-isa a2a3 intrinsics -- set_mov_pad_val / set_vector_mask /
# vector_dup -- with "does not support the given target feature", inside
# 3rdparty/pto-isa headers, before any generated code runs). The same error
# class reproduces locally on CANN 9.0.0 even for flat PTO kernels, so it is a
# pto-isa/bisheng version incompatibility, not a property of the stride fix.
# PTO stride coverage is retained by the codegen tests above, which assert the
# exact pto::Stride tuples via tilelang.lower and pass on CI.


@pytest.mark.skipif(
    not (hasattr(torch, "npu") and torch.npu.is_available()),
    reason="row-slice scatter correctness requires an Ascend NPU runtime",
)
def test_row_slice_scatter_add_correctness():
    """The issue's kernel: read-modify-write rows of a packed 2D GM buffer
    through 1D UB tiles (out_idx=[] -- in-place buffer mutation)."""
    func = tilelang.compile(_scatter_kernel(), out_idx=[], pass_configs=PASS_CONFIGS, target="ascendc")
    torch.manual_seed(0)
    cp = torch.randn(HALF_TILES, BLOCK_M, BLOCK_N, dtype=torch.float32).npu()
    c2 = torch.randn(PACKED_ROWS, BLOCK_N, dtype=torch.float32).npu()
    torch.npu.synchronize()
    c2_orig = c2.cpu().clone()

    func(cp, c2)
    torch.npu.synchronize()

    golden = c2_orig
    for cid in range(HALF_TILES):
        bm, bn = cid // HALF_N_TILES, cid % HALF_N_TILES
        for rr in range(BLOCK_M):
            out_row = (bm * BLOCK_M + rr) * N_TILES + bn
            golden[out_row] += cp[cid, rr].cpu()
    torch.testing.assert_close(c2.cpu(), golden, rtol=1e-4, atol=1e-4)


@pytest.mark.skipif(
    not (hasattr(torch, "npu") and torch.npu.is_available()),
    reason="row-slice atomic_add correctness requires an Ascend NPU runtime",
)
def test_row_slice_atomic_add_correctness():
    """Row-slice atomic accumulation into a packed 2D GM buffer. Both vector
    cores run each block (the VEC_NUM=2 convention of the tile atomic_add
    tests), so every row is added twice."""
    rows = 16
    func = tilelang.compile(_atomic_kernel(rows), pass_configs=PASS_CONFIGS, target="ascendc")
    torch.manual_seed(0)
    src = torch.randn(rows, BLOCK_N, dtype=torch.float32).npu()
    c2 = torch.zeros(rows, BLOCK_N, dtype=torch.float32).npu()
    torch.npu.synchronize()

    func(src, c2)
    torch.npu.synchronize()
    torch.testing.assert_close(c2.cpu(), src.cpu() * 2, rtol=1e-4, atol=1e-4)


@pytest.mark.skipif(
    not (hasattr(torch, "npu") and torch.npu.is_available()),
    reason="folded-stride copy correctness requires an Ascend NPU runtime",
)
def test_fold_middle_scalar_copy_correctness():
    """Multi-row copies whose rows straddle a scalar dim: the folded stride
    (N2*G*D) is what the DMA actually uses between rows."""
    func = tilelang.compile(_fold_kernel(), out_idx=[-1], pass_configs=PASS_CONFIGS, target="ascendc")
    torch.manual_seed(0)
    q = torch.randn(1, F_BM, F_N2, F_G * F_D, dtype=torch.float32).npu()
    torch.npu.synchronize()
    out = func(q)
    torch.npu.synchronize()
    golden = q.cpu()[0, :, 1, :].reshape(1, F_BM, F_G, F_D)
    torch.testing.assert_close(out.cpu(), golden, rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
