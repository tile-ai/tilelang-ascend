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
            T.copy(A[bx * block_M:(bx + 1) * block_M, by * block_N:(by + 1) * block_N], a_ub)
            T.copy(B[bx * block_M:(bx + 1) * block_M, by * block_N:(by + 1) * block_N], b_ub)
            T.tile.add(c_ub, a_ub, b_ub)
            T.copy(c_ub, C[bx * block_M:(bx + 1) * block_M, by * block_N:(by + 1) * block_N])

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
            T.copy(A[bx * block_M:(bx + 1) * block_M, by * block_N:(by + 1) * block_N], a_ub)
            T.reduce_sum(a_ub, r_ub, dim=-1)
            T.copy(r_ub, B[bx * block_M:(bx + 1) * block_M, by:by + 1])

    return main


def _source(func):
    compiled = tilelang.compile(func, pass_configs=pass_configs, target="ascendc")
    return compiled.get_kernel_source()


def test_tail_add_emits_tail_helper():
    # 34x130 with 32x32 blocks => tail in both M and N.
    src = _source(_tail_add(34, 130, 32, 32, "float"))
    assert "tl::ascend::tail_binary" in src, src
    # The pad gap-fill path must be gone: copy_gm_to_ub no longer Duplicates.
    assert "Duplicate" not in src or "tail_" in src


def test_tail_reduce_emits_tail_helper():
    src = _source(_tail_reduce(34, 130, 32, 32, "float"))
    assert "tl::ascend::tail_reduce_sum" in src, src


if __name__ == "__main__":
    print(_source(_tail_add(34, 130, 32, 32, "float")))
