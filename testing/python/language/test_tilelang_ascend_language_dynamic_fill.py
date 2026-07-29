"""Runtime-dynamic T.tile.fill regression test (issue #1207).

Exercises T.tile.fill on a slice whose length is only known at runtime, e.g.
``T.tile.fill(c_ub[0, 0:idx], 1.0)`` where ``idx`` depends on a value read from
A. The tile is zero-initialised first so the region past ``idx`` is a known 0,
which lets us assert the fill length is exactly ``idx`` (guarding against the
PTO over-fill where the whole tile row would be filled). Runs on both the
ascendc and pto codegen targets.
"""

import os

import pytest

import torch

import tilelang
import tilelang.language as T

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}


@pytest.fixture(scope="session", autouse=True)
def clear_cache():
    """Clear tilelang cache before tests."""
    tilelang.cache.clear_cache()
    yield


def dynamic_fill(M, N, block_M, block_N, dtype="float"):
    m_num = M // block_M
    n_num = N // block_N

    VEC_NUM = 2

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num

            c_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
            # Zero the tile so the region past idx is a known 0.
            T.tile.fill(c_ub, 0.0)
            # Runtime-dynamic fill length: 32 or 64 depending on a value in A.
            idx = T.if_then_else(A[0, 0] > 0, 32, 64)
            T.tile.fill(c_ub[0, 0:idx], 1.0)

            T.copy(c_ub, C[bx * block_M + vid * block_M // VEC_NUM, by * block_N])

    return main


def run_test_dynamic_fill(M, N, block_M, block_N, dtype, target):
    func = dynamic_fill(M, N, block_M, block_N, dtype)
    func = tilelang.compile(func, out_idx=[-1], pass_configs=pass_configs, target=target)

    torch.manual_seed(0)
    a = torch.randn(M, N).npu()

    torch.npu.synchronize()
    c = func(a).cpu()

    # A[0,0] > 0 -> fill 32 cols of row 0, else 64. Row 0 of the first tile:
    # [0:fill_len) == 1.0 and [fill_len:block_N) == 0.0 (exact-length check).
    fill_len = 32 if a[0, 0].item() > 0 else 64
    torch.testing.assert_close(c[0, :fill_len], torch.ones(fill_len), rtol=0, atol=0)
    torch.testing.assert_close(c[0, fill_len:block_N], torch.zeros(block_N - fill_len), rtol=0, atol=0)


@pytest.mark.parametrize("target", ["ascendc", "pto"])
@pytest.mark.parametrize("shape", [(1024, 1024)])
def test_dynamic_fill(target, shape):
    M, N = shape
    run_test_dynamic_fill(M, N, 128, 256, "float", target=target)


def dynamic_fill_multirow(ROWS, COLS, dtype="float"):
    """Dynamic fill whose length can span multiple rows.

    ``c_ub`` has a small column count (COLS) so a runtime ``idx`` may exceed it,
    exercising the multi-row path. ``idx = 48`` (COLS=32) is one full row plus a
    16-element tail (non-column-aligned), ``idx = 64`` is two full rows (aligned).
    """

    @T.prim_func
    def main(
        A: T.Tensor((1, 1), dtype),
        C: T.Tensor((ROWS, COLS), dtype),
    ):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            c_ub = T.alloc_ub((ROWS, COLS), dtype)
            # Zero the tile so the region past idx is a known 0.
            T.tile.fill(c_ub, 0.0)
            # idx can exceed COLS, wrapping across rows: 48 = 1*32 + 16 (tail),
            # 64 = 2*32 (aligned). Guards the full-rows + tail split.
            idx = T.if_then_else(A[0, 0] > 0, 48, 64)
            T.tile.fill(c_ub[0, 0:idx], 1.0)
            T.copy(c_ub, C[0, 0])

    return main


def run_test_dynamic_fill_multirow(ROWS, COLS, dtype, target, fill_len):
    func = dynamic_fill_multirow(ROWS, COLS, dtype)
    func = tilelang.compile(func, out_idx=[-1], pass_configs=pass_configs, target=target)

    # Force the branch yielding fill_len: 48 when A[0,0] > 0, else 64.
    a_val = 1.0 if fill_len == 48 else -1.0
    a = torch.tensor([[a_val]], dtype=torch.float32).npu()
    torch.npu.synchronize()
    c = func(a).cpu().flatten()

    total = ROWS * COLS
    # Exact-length check: [0:fill_len) == 1.0, the rest stays 0.0 (the tail
    # split must neither drop elements nor over-fill into the next row).
    torch.testing.assert_close(c[:fill_len], torch.ones(fill_len, dtype=c.dtype), rtol=0, atol=0)
    torch.testing.assert_close(c[fill_len:total], torch.zeros(total - fill_len, dtype=c.dtype), rtol=0, atol=0)


@pytest.mark.parametrize("target", ["ascendc", "pto"])
@pytest.mark.parametrize("fill_len", [48, 64])
def test_dynamic_fill_multirow(target, fill_len):
    # COLS=32 so idx in {48, 64} spans 2 rows; ROWS=8 leaves headroom.
    run_test_dynamic_fill_multirow(8, 32, "float", target=target, fill_len=fill_len)


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    dynamic_fill_test_path = os.path.join(current_dir, "test_tilelang_ascend_language_dynamic_fill.py")
    pytest.main(["--forked", dynamic_fill_test_path])
