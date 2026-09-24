"""Issue #1304 regression: bulk T.copy + scalar tails over W=126 fp32.

A scalar GM store (BufferStore lowering to an S-pipe SetValue) goes through a
write-back cache that MTE3 DMA bypasses. When a kernel mixes scalar stores
and bulk T.copy writes to the same GM buffer, the scalar store's cache-line
fill (read-modify-write) can snapshot GM before the DMA lands, and the dirty
line's later eviction stamps stale bytes over the freshly copied data. In a
multi-block kernel whose row pitch is not a cache-line multiple, the race is
also cross-core (adjacent rows on different cores share cache lines), which
no per-core synchronization can fix.

Two complementary treatments, both opt-in via pass_configs:

* AscendSyncInsert (TL_ASCEND_AUTO_SYNC): detects the mixed-write pattern
  and inserts a dcci before the DMA plus an MTE3_S ordering after it. Fixes
  the same-core hazard; the cross-core residual remains.
* AscendScalarStoreToDma (TL_ASCEND_SCALAR_STORE_TO_DMA): rewrites the
  contiguous scalar tail stores into UB staging + one DMA burst. DMA-vs-DMA
  writes go through the same L2 path and are coherent, so the hazard is
  removed entirely.
"""

import torch

import tilelang
import tilelang.language as T


@tilelang.jit(
    out_idx=[0],
    target="ascendc",
    pass_configs={tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True},
)
def fill_rows_with_tails(oc_n: int = 8, h: int = 12, w: int = 126, main_w: int = 112, dtype: str = "float32"):
    """The single-core issue #1304 pattern: an aligned bulk T.copy per row plus
    scalar tail stores, with w % 8 != 0 so consecutive rows share GM lines."""

    @T.prim_func
    def main(Y: T.Tensor((oc_n, h, w), dtype)):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            row_ub = T.alloc_ub((main_w,), dtype)
            with T.Scope("V"):
                for oc in T.serial(oc_n):
                    for oh in T.serial(h):
                        for ow in T.serial(main_w):
                            row_ub[ow] = (
                                T.float32(0.125)
                                + T.cast(oc, "float32") * T.float32(0.0001)
                                + T.cast(oh, "float32") * T.float32(0.00001)
                                + T.cast(ow, "float32") * T.float32(0.000001)
                            )
                        T.copy(row_ub, Y[oc, oh, 0])
                        for tw in T.serial(w - main_w):
                            ow_tail = main_w + tw
                            Y[oc, oh, ow_tail] = (
                                T.float32(0.125)
                                + T.cast(oc, "float32") * T.float32(0.0001)
                                + T.cast(oh, "float32") * T.float32(0.00001)
                                + T.cast(ow_tail, "float32") * T.float32(0.000001)
                            )

    return main


@tilelang.jit(
    out_idx=[0],
    target="ascendc",
    pass_configs={
        tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
        tilelang.PassConfigKey.TL_ASCEND_SCALAR_STORE_TO_DMA: True,
    },
)
def fill_rows_with_tails_dma(oc_n: int = 8, h: int = 12, w: int = 126, main_w: int = 112, dtype: str = "float32"):
    """Same kernel as fill_rows_with_tails; the ScalarStoreToDma pass rewrites
    the tail loop into UB staging + one ascend_copy burst at compile time."""

    @T.prim_func
    def main(Y: T.Tensor((oc_n, h, w), dtype)):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            row_ub = T.alloc_ub((main_w,), dtype)
            with T.Scope("V"):
                for oc in T.serial(oc_n):
                    for oh in T.serial(h):
                        for ow in T.serial(main_w):
                            row_ub[ow] = (
                                T.float32(0.125)
                                + T.cast(oc, "float32") * T.float32(0.0001)
                                + T.cast(oh, "float32") * T.float32(0.00001)
                                + T.cast(ow, "float32") * T.float32(0.000001)
                            )
                        T.copy(row_ub, Y[oc, oh, 0])
                        for tw in T.serial(w - main_w):
                            ow_tail = main_w + tw
                            Y[oc, oh, ow_tail] = (
                                T.float32(0.125)
                                + T.cast(oc, "float32") * T.float32(0.0001)
                                + T.cast(oh, "float32") * T.float32(0.00001)
                                + T.cast(ow_tail, "float32") * T.float32(0.000001)
                            )

    return main


def _reference(oc_n, h, w):
    oc_g = torch.arange(oc_n, dtype=torch.float32).view(oc_n, 1, 1)
    oh_g = torch.arange(h, dtype=torch.float32).view(1, h, 1)
    ow_g = torch.arange(w, dtype=torch.float32).view(1, 1, w)
    return 0.125 + oc_g * 0.0001 + oh_g * 0.00001 + ow_g * 0.000001


def test_copy_ub_to_gm_with_scalar_tails_w126():
    oc_n, h, w = 8, 12, 126
    out = fill_rows_with_tails(oc_n, h, w)()
    torch.npu.synchronize()
    assert torch.equal(out.cpu(), _reference(oc_n, h, w))


def test_copy_ub_to_gm_scalar_tails_dma_rewrite():
    oc_n, h, w = 8, 12, 126
    out = fill_rows_with_tails_dma(oc_n, h, w)()
    torch.npu.synchronize()
    assert torch.equal(out.cpu(), _reference(oc_n, h, w))
