"""Regression tests for T.Pipelined extent/tail handling (issue #1922).

CrossCorePipeline chunks a ``T.Pipelined`` loop into stage loops of
``num_stages`` iterations.  The outer loop used ``extent / num_stages``
(floor division), which dropped the tail:

* ``extent == 1`` produced an empty outer loop.  ``b_l1`` was then never
  touched, AscendMemoryPlanning assigned it no address, and codegen failed
  with ``Cannot find pre-allocated address for buffer: b_l1``.
* ``extent == 3`` ran only 2 iterations, leaving the third output slice zero.

The outer loop must use ``ceil(extent / num_stages)`` and the inner stage
loop must be clamped to the remaining iterations.
"""

import pytest
import torch
import tilelang
import tilelang.language as T

PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}

TARGET = "ascendc"


@pytest.fixture(scope="module", autouse=True)
def _disable_cache():
    tilelang.disable_cache()
    yield
    tilelang.enable_cache()


def _gemm_loop(D, M=64, bD=32):
    """The issue's kernel: a GEMM loop with D // bD pipeline iterations."""

    @T.prim_func
    def main(
        A: T.Tensor((M, M), "bfloat16"),
        B: T.Tensor((M, D), "bfloat16"),
        C: T.Tensor((M, D), "bfloat16"),
    ):
        with T.Kernel(1, threads=1, is_npu=True):
            a_l1 = T.alloc_shared([M, M], "bfloat16")
            b_l1 = T.alloc_shared([M, bD], "bfloat16")
            c_frag = T.alloc_fragment([M, bD], "float32")
            c_ub = T.alloc_shared([M, bD], "bfloat16")
            T.copy(A, a_l1)
            for i in T.Pipelined(D // bD, num_stages=2):
                T.copy(B[0:M, i * bD : (i + 1) * bD], b_l1)
                T.gemm_v0(a_l1, b_l1, c_frag, init=True)
                T.copy(c_frag, c_ub)
                T.copy(c_ub, C[0:M, i * bD : (i + 1) * bD])

    return main


def _run(D):
    M = 64
    kernel = tilelang.compile(_gemm_loop(D), pass_configs=PASS_CONFIGS, target=TARGET)
    a = torch.randn(M, M, dtype=torch.bfloat16, device="npu")
    b = torch.randn(M, D, dtype=torch.bfloat16, device="npu")
    c = torch.zeros(M, D, dtype=torch.bfloat16, device="npu")
    torch.npu.synchronize()
    kernel(a, b, c)
    torch.npu.synchronize()
    ref = a.float() @ b.float()
    torch.testing.assert_close(c.float(), ref, rtol=2e-2, atol=2e-2)


@pytest.mark.skipif(
    not (hasattr(torch, "npu") and torch.npu.is_available()),
    reason="T.Pipelined correctness requires an Ascend NPU runtime",
)
@pytest.mark.parametrize("D", [32, 64, 96])
def test_pipeline_extent_covers_tail(D):
    """D=32 -> 1 iteration, D=96 -> 3 iterations (both from issue #1922).

    D=128 (4 iterations, two complete waves) fails on the base revision as
    well; multi-wave correctness is a separate issue and is not covered here.
    """
    _run(D)


@pytest.mark.skipif(
    not (hasattr(torch, "npu") and torch.npu.is_available()),
    reason="T.Pipelined correctness requires an Ascend NPU runtime",
)
@pytest.mark.parametrize("extent,num_stages", [(1, 2), (2, 2), (3, 2), (1, 3)])
def test_vector_pipeline_short_extent(extent, num_stages):
    """A pure-vector pipeline with fewer iterations than its depth must lower
    to a serial loop instead of a prologue-only pipeline."""

    @T.prim_func
    def main(X: T.Tensor((1, 128), "float32"), Y: T.Tensor((1, 128), "float32")):
        with T.Kernel(1, threads=1, is_npu=True):
            x_ub = T.alloc_shared([1, 128], "float32")
            y_ub = T.alloc_shared([1, 128], "float32")
            T.tile.fill(y_ub, 0.0)
            for _ in T.Pipelined(extent, num_stages=num_stages):
                T.copy(X, x_ub)
                T.tile.add(y_ub, y_ub, x_ub)
            T.copy(y_ub, Y)

    kernel = tilelang.compile(main, pass_configs=PASS_CONFIGS, target=TARGET)
    x = torch.randn(1, 128, dtype=torch.float32, device="npu")
    y = torch.zeros(1, 128, dtype=torch.float32, device="npu")
    torch.npu.synchronize()
    kernel(x, y)
    torch.npu.synchronize()
    torch.testing.assert_close(y, x * extent, rtol=1e-3, atol=1e-3)
