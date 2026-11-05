import pytest
import tilelang
import tilelang.language as T

"""
T.copy source-region active-dimension guard.

Feature under test
------------------
``tl.ascend_copy`` lowering (src/op/ascend.cc ``find_active_dim_indices``) only
honours the **last two** active (extent != 1) dimensions of the source region.
A source region with >= 3 active dimensions -- e.g. a 4-D global slice
``Q[b, s0:s0+bs, g0:g0+G, :]`` with extents ``[1, bs, G, D]`` -- silently drops
the leading active dimension(s): only ``[G, D]`` is DMA'd, the ``bs`` extent is
never transferred, no error is raised, and the result is silently wrong.

The guard
---------
``src/op/ascend.cc::AscendCopy`` now raises a lowering-time error when the
*source* region of a copy has more than two active dimensions (message contains
"active dims").  This turns the silent data loss into an explicit, searchable
compile-time error.

Legal patterns that must NOT trip the guard
-------------------------------------------
  * Leading dims with extent == 1 (the FA/GQA pattern):
    ``Q[bz, s, g0:g0+G, :]`` -> extents ``[1, 1, G, D]`` -> 2 active dims.
  * Symbolic extents count as active (they may be != 1 at runtime), so a
    symbolic leading dim on the source is rejected -- this is intentional: the
    lowering cannot prove it is 1, and silently dropping it would be wrong.
  * The destination side is not restricted by this guard (L1 row-splice writes
    into a 2-D buffer slice remain legal).
"""

TARGET = "ascendc"

DEV_CONFIGS = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}


def copy_src_3_active_dims(batch, bs, groups, dim, dtype):
    """Illegal: source region [1, bs, G, D] has 3 active dims (bs, G, D).

    The destination is a UB buffer so the copy itself is otherwise legal
    (GM -> UB): without the guard this compiles silently and drops the bs
    extent, producing wrong results with no error.
    """

    @T.prim_func
    def main(
        Q: T.Tensor([batch, bs, groups, dim], dtype),  # type: ignore
        Out: T.Tensor([bs, groups, dim], dtype),  # type: ignore
    ):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            buf = T.alloc_ub([groups, dim], dtype)
            # extents [1, bs, G, D] -> 3 active dims -> must raise
            T.copy(Q[0, 0:bs, 0:groups, :], buf)
            T.copy(buf, Out[0, :, :])

    return main


def copy_src_leading_dim_one(batch, seq, groups, dim, dtype):
    """Legal: source region [1, 1, G, D] has only 2 active dims (G, D).

    This is the FA/GQA load pattern; the guard must not trip on it.  The
    destination is UB so the whole kernel is a supported GM -> UB -> GM flow.
    """

    @T.prim_func
    def main(
        Q: T.Tensor([batch, seq, groups, dim], dtype),  # type: ignore
        Out: T.Tensor([groups, dim], dtype),  # type: ignore
    ):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            buf = T.alloc_ub([groups, dim], dtype)
            T.copy(Q[0, 0, 0:groups, :], buf)
            T.copy(buf, Out[:, :])

    return main


def test_copy_src_3_active_dims_raises():
    """A >=3-active-dim source region must raise a compile-time error."""
    func = copy_src_3_active_dims(1, 4, 8, 128, "float16")
    with pytest.raises(Exception, match="active dims"):
        tilelang.compile(func, pass_configs=DEV_CONFIGS, target=TARGET)


def test_copy_src_leading_dim_one_ok():
    """The FA/GQA pattern (leading extent-1 dims) must still compile."""
    func = copy_src_leading_dim_one(1, 4, 8, 128, "float16")
    # Should compile without raising; no need to run on device.
    tilelang.compile(func, out_idx=[-1], pass_configs=DEV_CONFIGS, target=TARGET)


if __name__ == "__main__":
    test_copy_src_3_active_dims_raises()
    test_copy_src_leading_dim_one_ok()
    print("PASS")
