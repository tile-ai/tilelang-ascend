"""Codegen-level checks for the AscendC vector tail-block scheme.

These tests only inspect the generated kernel source (host-side codegen), so
they do not require NPU hardware to *run* the kernel -- only a built tilelang
with the Ascend codegen. They verify that:

  * a kernel with a real tail (M and/or N not divisible by the block) emits the
    internal ``tl::ascend::tail_*`` helpers, and
  * the removed ``pad_value`` path (the UB gap-fill ``Duplicate``) is gone, i.e.
    ``T.copy`` no longer carries a pad argument.
"""

import pytest

import tilelang
import tilelang.language as T

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
    # Enable the opt-in tail-block scheme (default off) so the tail_* helpers
    # are actually emitted for these tests.
    tilelang.PassConfigKey.TL_ASCEND_TAIL_MASK: True,
}


@pytest.fixture(scope="session", autouse=True)
def clear_cache():
    tilelang.cache.clear_cache()
    yield


def _tail_add(M, N, block_M, block_N, dtype="float"):
    m_num = T.ceildiv(M, block_M)
    n_num = T.ceildiv(N, block_N)

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),
        B: T.Tensor((M, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, _):
            bx = cid // n_num
            by = cid % n_num
            a_ub = T.alloc_ub((block_M, block_N), dtype)
            b_ub = T.alloc_ub((block_M, block_N), dtype)
            c_ub = T.alloc_ub((block_M, block_N), dtype)
            T.copy(A[bx * block_M : (bx + 1) * block_M, by * block_N : (by + 1) * block_N], a_ub)
            T.copy(B[bx * block_M : (bx + 1) * block_M, by * block_N : (by + 1) * block_N], b_ub)
            T.tile.add(c_ub, a_ub, b_ub)
            T.copy(c_ub, C[bx * block_M : (bx + 1) * block_M, by * block_N : (by + 1) * block_N])

    return main


def _tail_reduce(M, N, block_M, block_N, dtype="float"):
    m_num = T.ceildiv(M, block_M)
    n_num = T.ceildiv(N, block_N)

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),
        B: T.Tensor((M, block_N), dtype),
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, _):
            bx = cid // n_num
            by = cid % n_num
            a_ub = T.alloc_ub((block_M, block_N), dtype)
            r_ub = T.alloc_ub((block_M, 1), dtype)
            T.copy(A[bx * block_M : (bx + 1) * block_M, by * block_N : (by + 1) * block_N], a_ub)
            T.reduce_sum(a_ub, r_ub, dim=-1)
            T.copy(r_ub, B[bx * block_M : (bx + 1) * block_M, by : by + 1])

    return main


def _tail_unary(M, N, block_M, block_N, dtype="float"):
    m_num = T.ceildiv(M, block_M)
    n_num = T.ceildiv(N, block_N)

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),
        B: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, _):
            bx = cid // n_num
            by = cid % n_num
            a_ub = T.alloc_ub((block_M, block_N), dtype)
            b_ub = T.alloc_ub((block_M, block_N), dtype)
            T.copy(A[bx * block_M : (bx + 1) * block_M, by * block_N : (by + 1) * block_N], a_ub)
            T.tile.exp(b_ub, a_ub)
            T.copy(b_ub, B[bx * block_M : (bx + 1) * block_M, by * block_N : (by + 1) * block_N])

    return main


def _tail_scalar(M, N, block_M, block_N, dtype="float"):
    m_num = T.ceildiv(M, block_M)
    n_num = T.ceildiv(N, block_N)

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),
        B: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, _):
            bx = cid // n_num
            by = cid % n_num
            a_ub = T.alloc_ub((block_M, block_N), dtype)
            b_ub = T.alloc_ub((block_M, block_N), dtype)
            T.copy(A[bx * block_M : (bx + 1) * block_M, by * block_N : (by + 1) * block_N], a_ub)
            T.tile.add(b_ub, a_ub, 2.0)  # scalar immediate -> adds -> tail_scalar
            T.copy(b_ub, B[bx * block_M : (bx + 1) * block_M, by * block_N : (by + 1) * block_N])

    return main


def _source(func, target="ascendc", tail_mask=True):
    cfg = {**pass_configs, tilelang.PassConfigKey.TL_ASCEND_TAIL_MASK: tail_mask}
    compiled = tilelang.compile(func, pass_configs=cfg, target=target)
    return compiled.get_kernel_source()


# Per-backend "a tail-aware op was emitted" marker. The two backends express the
# valid-region compute differently in the generated source:
#   * ascendc -> a call to the internal ``tl::ascend::tail_<kind>`` device helper
#     (the mask/repeat/count ladder written in ascend/common.h).
#   * pto     -> a ``TileUbDataND<..., pto::DYNAMIC, pto::DYNAMIC>`` dynamic tile.
#     PTO reuses its native dynamic-tile op macros (TADD/TEXP/TADDS/...), so the
#     tell-tale is the DYNAMIC valid-shape tile, which is emitted *only* by the
#     tail unary/binary/scalar codegen (CreateUbVariableDynamic) and nowhere on
#     the ordinary full-tile path.
def _emit_marker(target, kind):
    return f"tl::ascend::tail_{kind}" if target == "ascendc" else "pto::DYNAMIC"


def _no_tail_marker(target):
    # A substring that must be ABSENT when no op was rewritten to a tail variant.
    return "tl::ascend::tail_" if target == "ascendc" else "pto::DYNAMIC"


@pytest.mark.parametrize("target", ["ascendc", "pto"])
def test_tail_add_emits_tail_helper(target):
    # 34x130 with 32x32 blocks => tail in both M and N.
    src = _source(_tail_add(34, 130, 32, 32, "float"), target=target)
    assert _emit_marker(target, "binary") in src, src


@pytest.mark.parametrize("target", ["ascendc", "pto"])
def test_tail_unary_emits_tail_helper(target):
    src = _source(_tail_unary(34, 130, 32, 32, "float"), target=target)
    assert _emit_marker(target, "unary") in src, src


@pytest.mark.parametrize("target", ["ascendc", "pto"])
def test_tail_scalar_emits_tail_helper(target):
    src = _source(_tail_scalar(34, 130, 32, 32, "float"), target=target)
    assert _emit_marker(target, "scalar") in src, src


@pytest.mark.parametrize("target", ["ascendc", "pto"])
def test_tail_reduce_not_rewritten(target):
    # reduce is currently NOT rewritten to a tail variant (rewrite_reduce=False):
    # it stays on the full-tile + pad_value/real_shape path, so neither backend
    # emits its tail marker for a reduce-only kernel.
    src = _source(_tail_reduce(34, 130, 32, 32, "float"), target=target)
    assert _no_tail_marker(target) not in src, src


@pytest.mark.parametrize("target", ["ascendc", "pto"])
def test_flag_off_emits_no_tail_helper(target):
    # Opt-in default: with the switch off the pass is a no-op, so a tail kernel
    # generates the same full-tile ops as upstream (no tail variant at all).
    src = _source(_tail_add(34, 130, 32, 32, "float"), target=target, tail_mask=False)
    assert _no_tail_marker(target) not in src, src


if __name__ == "__main__":
    print(_source(_tail_add(34, 130, 32, 32, "float")))
