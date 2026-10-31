import pytest
import tilelang
import tilelang.language as T
import torch

"""
GM -> UB column-slice copy correctness suite (the dstStride "flattening" fix).

Feature under test
------------------
``T.copy(gm_src, ub[:, c0:c0+W])`` with a COMPILE-TIME-CONSTANT column-slice
destination narrower than the UB buffer's row pitch.

Before the fix, codegen instantiated ``tl::ascend::copy_gm_to_ub<T, dstN,
dstM>`` with ``dstN = slice width`` instead of the buffer's physical row
pitch.  Inside the template the DataCopyExtParams destination stride is
``(dstN - maskShapeN) * sizeof(T) / 32`` 32B blocks, so ``dstN == slice
width`` made dstStride 0: consecutive rows landed PACKED from the slice
offset, silently overwriting the neighbour columns (row 1 of the slice wrote
over row 0's right neighbour, and so on -- the "flattening" bug).

The fix (src/op/ascend.cc + src/tl_templates/ascend/common.h):
  * a proper const column slice instantiates the template with the buffer's
    row pitch (dstN = pitch), reproducing the production AscendC DataCopyPad
    pattern (blockLen/srcStride in BYTES, UB dstStride in 32B BLOCKS of the
    physical pitch);
  * the call is emitted with the new defaulted template arg ``padGap=false``
    so the tail-pad Duplicate pre-fill (which fills dstM*dstN elements and
    uses fixed event ids 0) does NOT wipe the whole buffer -- the gap columns
    of a slice destination belong to neighbouring slices' data.

For a [16, 512] fp32 UB buffer and four [16, 128] slices the emitted call is
``copy_gm_to_ub<float, 512, 16, false>(ub[l*128], A[l], 128, 16, 128, 0)``,
i.e. DataCopyExtParams(blockCount=16, blockLen=512B, srcStride=0,
dstStride=48 x 32B) -- rows keep the 2048B physical pitch.

Scenarios covered (target = ascendc / pto, real NPU)
----------------------------------------------------
  1. Assemble: four GM [16,128] planes copied into the four column slices of
     one UB [16,512], stored whole; result must be EXACT (bit-identical to
     the CPU-assembled concatenation).  This is the case that flattened
     before the fix.
  2. Const-offset slice into a pre-filled UB: proves padGap=false preserves
     the neighbour columns (the Duplicate pre-fill would wipe them).
  3. Full-width copy: unchanged behaviour baseline (extent == pitch, no
     slice, no padGap arg emitted).
  4. Row-slice destination: innermost extent == pitch -> NOT a column slice,
     unchanged behaviour (no Duplicate fires, rows below the slice keep the
     pre-fill).
  5. Codegen source assertion (ascendc): the slice copy instantiates the
     template with the pitch and padGap=false, and no slice-width dstN
     instantiation remains.
"""

# Same Vector-scope config as test_tilelang_ascend_language_gm_to_ub.py.
VEC_PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}

TARGETS = ["ascendc", "pto"]


@pytest.fixture(scope="session", autouse=True)
def clear_cache():
    """Clear tilelang cache before the session."""
    tilelang.cache.clear_cache()
    yield


# =============================================================================
# 1. Column-slice assemble (the flattening-bug reproducer)   [risk: medium]
# =============================================================================
def col_slice_assemble(L, M, W, dtype):
    """Copy L GM planes [M, W] into the L consecutive column slices of one UB
    [M, L*W] buffer, then store the whole UB.  The slice offset l*W is a
    runtime loop value; the slice WIDTH W is compile-time constant -- the
    exact shape the dstStride fix targets."""
    pitch = L * W

    @T.prim_func
    def main(A: T.Tensor((L, M, W), dtype), C: T.Tensor((M, pitch), dtype)):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            ub = T.alloc_ub((M, pitch), dtype)
            for l in T.serial(L):
                T.copy(A[l, 0, 0], ub[0:M, l * W : (l + 1) * W])
            T.copy(ub, C[0, 0])

    return main


@pytest.mark.parametrize("target", TARGETS)
def test_col_slice_assemble(target):
    torch.manual_seed(0)
    L, M, W = 4, 16, 128
    func = col_slice_assemble(L, M, W, "float")
    func = tilelang.compile(func, out_idx=[-1], pass_configs=VEC_PASS_CONFIGS, target=target)
    a = torch.randn(L, M, W, dtype=torch.float32).npu()
    torch.npu.synchronize()
    c = func(a)
    # CPU-assembled expectation: C[:, l*W:(l+1)*W] == A[l]  ->  EXACT match
    # (float32 DMA copies are bit-exact; any dstStride flattening shows up as
    # misplaced rows immediately).
    expected = a.cpu().permute(1, 0, 2).reshape(M, L * W)
    torch.testing.assert_close(c.cpu(), expected, rtol=0, atol=0)


# =============================================================================
# 2. Const-offset slice preserves neighbours (padGap=false)  [risk: medium]
# =============================================================================
def col_slice_const_offset(L, M, W, off, dtype):
    """Pre-fill the whole UB [M, pitch] with 7.0, copy ONE GM plane into the
    const column slice [off, off+W), store the whole UB.  The neighbours must
    still read 7.0: with padGap=true the Duplicate pre-fill would wipe all
    M*pitch elements (and use fixed event ids 0)."""
    pitch = L * W

    @T.prim_func
    def main(A: T.Tensor((L, M, W), dtype), C: T.Tensor((M, pitch), dtype)):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            ub = T.alloc_ub((M, pitch), dtype)
            T.tile.fill(ub, 7.0)
            T.copy(A[1, 0, 0], ub[0:M, off : off + W])
            T.copy(ub, C[0, 0])

    return main


@pytest.mark.parametrize("target", TARGETS)
def test_col_slice_const_offset_preserves_neighbours(target):
    torch.manual_seed(0)
    L, M, W, off = 4, 16, 128, 256  # offset 256 fp32 = 1024B, 32B-aligned
    func = col_slice_const_offset(L, M, W, off, "float")
    func = tilelang.compile(func, out_idx=[-1], pass_configs=VEC_PASS_CONFIGS, target=target)
    a = torch.randn(L, M, W, dtype=torch.float32).npu()
    torch.npu.synchronize()
    c = func(a).cpu()
    torch.testing.assert_close(c[:, off : off + W], a[1].cpu(), rtol=0, atol=0)
    assert torch.all(c[:, :off] == 7.0), "left neighbour columns were wiped"
    assert torch.all(c[:, off + W :] == 7.0), "right neighbour columns were wiped"


# =============================================================================
# 3. Full-width copy baseline (unchanged behaviour)          [risk: low]
# =============================================================================
def full_width_copy(M, N, dtype):
    @T.prim_func
    def main(A: T.Tensor((M, N), dtype), C: T.Tensor((M, N), dtype)):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            ub = T.alloc_ub((M, N), dtype)
            T.copy(A[0, 0], ub)
            T.copy(ub, C[0, 0])

    return main


@pytest.mark.parametrize("target", TARGETS)
def test_full_width_copy(target):
    torch.manual_seed(0)
    M, N = 16, 128
    func = full_width_copy(M, N, "float")
    func = tilelang.compile(func, out_idx=[-1], pass_configs=VEC_PASS_CONFIGS, target=target)
    a = torch.randn(M, N, dtype=torch.float32).npu()
    torch.npu.synchronize()
    c = func(a)
    torch.testing.assert_close(c.cpu(), a.cpu(), rtol=0, atol=0)


# =============================================================================
# 4. Row-slice destination (unchanged behaviour)             [risk: low]
# =============================================================================
def row_slice_dst(M, N, dtype):
    """Row slice: innermost extent == buffer pitch -> NOT a column slice, the
    template keeps dstN = pitch with padGap defaulted true and no Duplicate
    fires (maskShapeM == dstM after compute_blocklen clamping).  Rows below
    the slice must keep the pre-fill."""

    @T.prim_func
    def main(A: T.Tensor((M, N), dtype), C: T.Tensor((M, N), dtype)):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            ub = T.alloc_ub((M, N), dtype)
            T.tile.fill(ub, 9.0)
            T.copy(A[0 : M // 2, 0:N], ub[0 : M // 2, 0:N])
            T.copy(ub, C[0, 0])

    return main


@pytest.mark.parametrize("target", TARGETS)
def test_row_slice_dst(target):
    torch.manual_seed(0)
    M, N = 16, 128
    func = row_slice_dst(M, N, "float")
    func = tilelang.compile(func, out_idx=[-1], pass_configs=VEC_PASS_CONFIGS, target=target)
    a = torch.randn(M, N, dtype=torch.float32).npu()
    torch.npu.synchronize()
    c = func(a).cpu()
    torch.testing.assert_close(c[: M // 2], a.cpu()[: M // 2], rtol=0, atol=0)
    assert torch.all(c[M // 2 :] == 9.0), "rows below the row slice were wiped"


# =============================================================================
# 5. Codegen source assertion (ascendc only)                 [risk: low]
# =============================================================================
def test_col_slice_codegen_source():
    """The emitted call must instantiate the template with the PHYSICAL pitch
    (512) and padGap=false -- never the slice width (128) as dstN."""
    tilelang.disable_cache()
    try:
        func = col_slice_assemble(4, 16, 128, "float")
        kern = tilelang.compile(func, out_idx=[-1], pass_configs=VEC_PASS_CONFIGS, target="ascendc")
        src = kern.get_kernel_source()
    finally:
        tilelang.enable_cache()
    assert "copy_gm_to_ub<float, 512, 16, false>" in src, "column-slice copy not emitted with pitch dstN + padGap=false:\n" + src
    assert "copy_gm_to_ub<float, 128" not in src, "slice-width dstN instantiation still emitted:\n" + src


# =============================================================================
# 6. UB -> GM column-slice disassemble (read-side mirror)    [risk: medium]
# =============================================================================
def col_slice_disassemble(L, M, W, dtype):
    """Store direction: load the whole UB [M, L*W] from one GM tensor, then
    copy L column slices out to L GM planes [M, W].  With srcN == slice width
    the copy_ub_to_gm packed check collapsed each multi-row slice store into
    ONE flat burst reading the UB rows contiguously from the slice offset
    (neighbour columns' data); with srcN == pitch the multi-burst srcStride
    is (pitch - W) * sizeof(T) / 32 32B blocks."""
    pitch = L * W

    @T.prim_func
    def main(A: T.Tensor((M, pitch), dtype), C: T.Tensor((L, M, W), dtype)):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            ub = T.alloc_ub((M, pitch), dtype)
            T.copy(A[0, 0], ub)
            for l in T.serial(L):
                T.copy(ub[0:M, l * W : (l + 1) * W], C[l, 0, 0])

    return main


@pytest.mark.parametrize("target", TARGETS)
def test_col_slice_disassemble(target):
    torch.manual_seed(0)
    L, M, W = 4, 16, 128
    func = col_slice_disassemble(L, M, W, "float")
    func = tilelang.compile(func, out_idx=[-1], pass_configs=VEC_PASS_CONFIGS, target=target)
    a = torch.randn(M, L * W, dtype=torch.float32).npu()
    torch.npu.synchronize()
    c = func(a).cpu()
    expected = a.cpu().reshape(M, L, W).permute(1, 0, 2)
    torch.testing.assert_close(c, expected, rtol=0, atol=0)


def test_col_slice_disassemble_codegen_source():
    """The store-side slice copy must instantiate the template with the
    PHYSICAL pitch (512), never the slice width (128) as srcN."""
    tilelang.disable_cache()
    try:
        func = col_slice_disassemble(4, 16, 128, "float")
        kern = tilelang.compile(func, out_idx=[-1], pass_configs=VEC_PASS_CONFIGS, target="ascendc")
        src = kern.get_kernel_source()
    finally:
        tilelang.enable_cache()
    assert "copy_ub_to_gm<float, 512, 16>" in src, "column-slice store not emitted with pitch srcN:\n" + src
    assert "copy_ub_to_gm<float, 128" not in src, "slice-width srcN instantiation still emitted:\n" + src


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
