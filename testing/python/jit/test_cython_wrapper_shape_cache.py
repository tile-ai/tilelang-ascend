# Copyright (c) Tile-AI Organization.
# Licensed under the MIT License.
"""Regression tests for the symbolic shape-resolution cache in the cython JIT
wrapper.

The wrapper caches resolved output/workspace shapes keyed by the tuple of
symbolic variable values, so that repeated calls of the same kernel with
identical shapes skip the TVM FFI resolution (Analyzer() + undefined_vars /
substitute / simplify per dim) entirely. These tests pin down the behavioral
contract of that cache:

  * alternating shapes populate distinct cache entries and every call still
    resolves the correct output shape and values;
  * repeated calls with the same shape return freshly allocated, independent
    output tensors (the cache must not alias or reuse output buffers);
  * static-shape kernels share the same code path (empty symbolic map);
  * the cache is actually populated (the fix itself, not just correct output).
"""

import pytest
import torch

import tilelang
import tilelang.language as T

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}

BLOCK_M, BLOCK_N = 64, 128

# (M, N) must be multiples of (BLOCK_M, BLOCK_N) so the tiled kernel covers
# the full tensor; different shapes exercise different cache entries.
SHAPES = [(128, 256), (256, 512), (64, 128)]


def symbolic_add(dtype="float16"):
    M = T.symbolic("M")
    N = T.symbolic("N")
    m_num = T.ceildiv(M, BLOCK_M)
    n_num = T.ceildiv(N, BLOCK_N)
    VEC_NUM = 2

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),  # type: ignore
        B: T.Tensor((M, N), dtype),  # type: ignore
        C: T.Tensor((M, N), dtype),  # type: ignore
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num

            a_ub = T.alloc_ub((BLOCK_M // VEC_NUM, BLOCK_N), dtype)
            b_ub = T.alloc_ub((BLOCK_M // VEC_NUM, BLOCK_N), dtype)
            c_ub = T.alloc_ub((BLOCK_M // VEC_NUM, BLOCK_N), dtype)

            T.copy(A[bx * BLOCK_M + vid * BLOCK_M // VEC_NUM, by * BLOCK_N], a_ub)
            T.copy(B[bx * BLOCK_M + vid * BLOCK_M // VEC_NUM, by * BLOCK_N], b_ub)

            T.tile.add(c_ub, a_ub, b_ub)

            T.copy(c_ub, C[bx * BLOCK_M + vid * BLOCK_M // VEC_NUM, by * BLOCK_N])

    return main


def static_add(M, N, dtype="float16"):
    m_num = M // BLOCK_M
    n_num = N // BLOCK_N
    VEC_NUM = 2

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),  # type: ignore
        B: T.Tensor((M, N), dtype),  # type: ignore
        C: T.Tensor((M, N), dtype),  # type: ignore
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num

            a_ub = T.alloc_ub((BLOCK_M // VEC_NUM, BLOCK_N), dtype)
            b_ub = T.alloc_ub((BLOCK_M // VEC_NUM, BLOCK_N), dtype)
            c_ub = T.alloc_ub((BLOCK_M // VEC_NUM, BLOCK_N), dtype)

            T.copy(A[bx * BLOCK_M + vid * BLOCK_M // VEC_NUM, by * BLOCK_N], a_ub)
            T.copy(B[bx * BLOCK_M + vid * BLOCK_M // VEC_NUM, by * BLOCK_N], b_ub)

            T.tile.add(c_ub, a_ub, b_ub)

            T.copy(c_ub, C[bx * BLOCK_M + vid * BLOCK_M // VEC_NUM, by * BLOCK_N])

    return main


def _make_inputs(M, N):
    a = torch.randn(M, N, dtype=torch.float16).npu()
    b = torch.randn(M, N, dtype=torch.float16).npu()
    torch.npu.synchronize()
    return a, b


@pytest.fixture(scope="module")
def symbolic_func():
    func = tilelang.compile(symbolic_add(), out_idx=[-1], pass_configs=pass_configs, target="ascendc")
    yield func
    tilelang.cache.clear_cache()


def test_alternating_shapes_are_all_correct(symbolic_func):
    """Calls with interleaved shapes must hit distinct cache entries and still
    resolve the right output shape and values every time."""
    order = [SHAPES[0], SHAPES[1], SHAPES[0], SHAPES[2], SHAPES[1], SHAPES[0]]
    for M, N in order:
        a, b = _make_inputs(M, N)
        c = symbolic_func(a, b)
        torch.npu.synchronize()
        assert tuple(c.shape) == (M, N), f"bad output shape {tuple(c.shape)} for input ({M}, {N})"
        torch.testing.assert_close(c, a + b, rtol=1e-2, atol=1e-2)


def test_repeated_shape_returns_independent_outputs(symbolic_func):
    """The cache stores shapes, not tensors: every call must allocate a fresh
    output buffer."""
    M, N = SHAPES[0]
    a, b = _make_inputs(M, N)
    outs = [symbolic_func(a, b) for _ in range(3)]
    torch.npu.synchronize()
    assert len({o.data_ptr() for o in outs}) == 3, "outputs must be distinct allocations"
    for o in outs:
        torch.testing.assert_close(o, a + b, rtol=1e-2, atol=1e-2)


def test_cache_is_populated_per_distinct_shape(symbolic_func):
    """The cache itself, not just correct output: the wrapper's shape cache
    must hold exactly one entry per distinct shape."""
    order = [SHAPES[0], SHAPES[1], SHAPES[0], SHAPES[2], SHAPES[1], SHAPES[0]]
    for M, N in order:
        a, b = _make_inputs(M, N)
        symbolic_func(a, b)
    torch.npu.synchronize()

    wrapper = symbolic_func.adapter.cython_wrapper
    shape_cache = getattr(wrapper, "shape_cache", None)
    assert shape_cache is not None, "cython wrapper should expose shape_cache"
    assert len(shape_cache) == len(SHAPES), f"expected {len(SHAPES)} cache entries (one per distinct shape), got {len(shape_cache)}"


def test_static_shape_kernel_shares_cache_path():
    """Static-shape kernels go through the same forward() path with an empty
    symbolic map (cache key ()); they must stay correct across calls."""
    M, N = SHAPES[0]
    func = tilelang.compile(static_add(M, N), out_idx=[-1], pass_configs=pass_configs, target="ascendc")
    for _ in range(2):
        a, b = _make_inputs(M, N)
        c = func(a, b)
        torch.npu.synchronize()
        assert tuple(c.shape) == (M, N)
        torch.testing.assert_close(c, a + b, rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
