import pytest
import tilelang
import tilelang.language as T
import torch

"""
Regression test for a runtime inner-extent (dynamic column width) in the AscendC
GM<->UB copy lowering (src/op/ascend.cc :: AscendCopy::Lower).

Feature under test
------------------
The contiguous (last) dim of a ``copy_gm_to_ub`` / ``copy_ub_to_gm`` call is a
**compile-time template argument**. The lowering built that template dim from the
BufferRegion's extent. When the extent is a *runtime* expression -- a copy sliced
to a dynamic width ``[..., 0:n]`` where ``n`` is only known at run time (e.g. a
softmax's actual window width, smaller than the fixed workspace/UB width) -- a
non-constant expression ended up inside the template argument, producing invalid
generated C++ (``error: use of undeclared identifier 'T'``), so the kernel failed
to compile.

The fix: if the inner extent is a compile-time ``IntImm`` use it (every existing
caller -- full-width copies and constant slices -- stays byte-identical); if it is
a runtime expression, use the buffer's compile-time shape for the template dim.
The actual runtime width is already carried by the ``maskShapeN`` argument
(``validCol``), so the DMA still transfers exactly the runtime number of columns.

How this test triggers it
-------------------------
Mirrors the real caller (a fixed-width GM workspace and UB tile, both allocated at
a compile-time width, copied over only the runtime-valid columns ``[..., 0:n]``).
Both the GM tensors and the UB tile are a fixed compile-time width ``UB_WIDTH``;
the runtime slice width ``n`` (``< UB_WIDTH``) is a ``T.symbolic`` carried by a
dummy input tensor's shape. Both the gm2ub load (UB is the destination) and the
ub2gm store (UB is the source) slice to ``0:n``, so both runtime-inner-extent
paths are exercised. Before the fix this fails to compile; after it, the first
``n`` columns round-trip correctly for any runtime ``n``.

This targets the ascendc backend only (the fix is in the non-PTO copy lowering).
"""

TARGET = "ascendc"

VEC_PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}


@pytest.fixture(scope="session", autouse=True)
def clear_cache():
    """Clear the tilelang cache before the session (a stale kernel could mask a
    codegen regression -- a rebuilt kernel could otherwise return a cached one)."""
    tilelang.cache.clear_cache()
    yield


def _torch_dtype(dtype):
    return {
        "float": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[dtype]


def gm_ub_gm_runtime_inner_extent(M, block_M, ub_width, dtype):
    """Copy only the first ``n`` (runtime) columns through a fixed-width UB tile.

    ``A`` / ``C`` and the UB tile are all a fixed compile-time width ``ub_width``;
    ``n`` is a symbolic runtime value (``<= ub_width``) carried by the dummy input
    ``NW``'s shape. Slicing ``[..., 0:n]`` makes the inner extent a runtime
    expression on both the gm2ub load and the ub2gm store -- the case that
    previously put a non-const expr into the compile-time template column dim.
    Row and column are both sliced (matching the real caller), so the copy extent
    is unambiguous."""
    n = T.symbolic("n")

    @T.prim_func
    def main(
        A: T.Tensor((M, ub_width), dtype),  # type: ignore  fixed-width GM
        NW: T.Tensor((n,), "int32"),  # type: ignore  dummy: carries runtime n
        C: T.Tensor((M, ub_width), dtype),  # type: ignore
    ):
        with T.Kernel(M // block_M, is_npu=True) as (cid, vid):
            a_ub = T.alloc_ub((block_M, ub_width), dtype)
            # gm2ub: dst UB sliced to runtime n (dst inner extent is runtime).
            T.copy(A[cid * block_M : cid * block_M + block_M, 0:n], a_ub[:, 0:n])
            # ub2gm: src UB sliced to runtime n (src inner extent is runtime).
            T.copy(a_ub[:, 0:n], C[cid * block_M : cid * block_M + block_M, 0:n])

    return main


def run_test_runtime_inner_extent(M, block_M, ub_width, dtype, n_values):
    torch.manual_seed(0)
    # Disable cache for the symbolic-var kernel (same rationale as the gm_to_ub
    # dynamic-shape suite: the disk cache is not process-safe under pytest-xdist).
    tilelang.disable_cache()
    try:
        func = gm_ub_gm_runtime_inner_extent(M, block_M, ub_width, dtype)
        func = tilelang.compile(func, out_idx=[-1], pass_configs=VEC_PASS_CONFIGS, target=TARGET)
    finally:
        tilelang.enable_cache()
    td = _torch_dtype(dtype)
    for n in n_values:
        a = torch.randn(M, ub_width, dtype=td).npu()
        nw = torch.zeros(n, dtype=torch.int32).npu()  # dummy: shape carries n
        torch.npu.synchronize()
        c = func(a, nw)
        # Only the first n columns are copied; the rest of C is untouched.
        torch.testing.assert_close(c[:, :n].cpu(), a[:, :n].cpu(), rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("dtype", ["float", "float16"])
def test_runtime_inner_extent(dtype):
    # Fixed 512-wide GM/UB; runtime slice widths 64/128/256 are all < 512 (a genuine
    # narrow runtime inner-extent) and 32-Byte aligned. Compiled once (symbolic n).
    run_test_runtime_inner_extent(128, 64, 512, dtype, n_values=[64, 128, 256])


def gm_atomic_add_runtime_inner_extent(M, block_M, ub_width, dtype):
    """Atomic-add only the first ``n`` (runtime) columns of a UB tile into a
    fixed-width GM band -- the ``AscendAtomicAdd::Lower`` counterpart of the copy
    above. The atomic_add's source UB is sliced to a runtime width ``[:, 0:n]``,
    so the source inner extent is a runtime expression that previously landed in
    the compile-time template column dim (``atomic_add_ub_to_gm<T, N>``) and
    produced invalid C++. Each block writes its own disjoint row band, so every
    GM element is added exactly once and the result is deterministic."""
    n = T.symbolic("n")

    @T.prim_func
    def main(
        SRC: T.Tensor((M, ub_width), dtype),  # type: ignore  source values
        NW: T.Tensor((n,), "int32"),  # type: ignore  dummy: carries runtime n
        C: T.Tensor((M, ub_width), dtype),  # type: ignore  atomic-add target (pre-zeroed)
    ):
        with T.Kernel(M // block_M, is_npu=True) as (cid, vid):
            s_ub = T.alloc_ub((block_M, ub_width), dtype)
            T.copy(SRC[cid * block_M : cid * block_M + block_M, 0:n], s_ub[:, 0:n])
            # atomic-add only the first n (runtime) columns into this block's band
            T.tile.atomic_add(C[cid * block_M : cid * block_M + block_M, 0:n], s_ub[:, 0:n])

    return main


def run_test_atomic_add_runtime_inner_extent(M, block_M, ub_width, dtype, n_values):
    torch.manual_seed(0)
    tilelang.disable_cache()
    try:
        func = gm_atomic_add_runtime_inner_extent(M, block_M, ub_width, dtype)
        func = tilelang.compile(func, pass_configs=VEC_PASS_CONFIGS, target=TARGET)
    finally:
        tilelang.enable_cache()
    td = _torch_dtype(dtype)
    # Both vector cores of the cube run the kernel body, so each band is
    # atomic-added VEC_NUM (2) times -- the same doubling the existing tile
    # atomic_add tests bake into their golden.
    VEC_NUM = 2
    for n in n_values:
        src = torch.randn(M, ub_width, dtype=td).npu()
        nw = torch.zeros(n, dtype=torch.int32).npu()  # dummy: shape carries n
        c = torch.zeros(M, ub_width, dtype=td).npu()  # atomic-add target, pre-zeroed
        torch.npu.synchronize()
        func(src, nw, c)
        torch.npu.synchronize()
        # The first n columns are added (VEC_NUM times); the [n:ub_width] tail is
        # untouched (still zero) -- so the runtime width, not the compile-time
        # template width, drove how many columns were added.
        torch.testing.assert_close(c[:, :n].cpu(), (src[:, :n] * VEC_NUM).cpu(), rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(c[:, n:].cpu(), torch.zeros(M, ub_width - n, dtype=td), rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("dtype", ["float", "float16"])
def test_atomic_add_runtime_inner_extent(dtype):
    # Same fixed 512-wide GM/UB and runtime slice widths as the copy test, on the
    # atomic_add (ub_to_gm) runtime-inner-extent path.
    run_test_atomic_add_runtime_inner_extent(128, 64, 512, dtype, n_values=[64, 128, 256])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
