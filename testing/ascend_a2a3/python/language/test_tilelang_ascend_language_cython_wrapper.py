"""Cython wrapper forward-path regression: lazy TVM Analyzer construction.

The default ``execution_backend="cython"`` routes every kernel call through
``CythonKernelWrapper.forward()`` (tilelang/jit/adapter/cython/cython_wrapper.pyx).
The TA wrapper (modern A2/A3 allocation resolver) used to construct a TVM ``Analyzer()`` unconditionally on every
call (~0.28 ms of host overhead per launch on Ascend 910B3) even though the
analyzer is only needed when a result/workspace shape is a symbolic
expression: the ``dynamic_symbolic_map`` must be non-empty AND a shape dim
must be a ``tir.PrimExpr`` (the only ``analyzer.simplify`` call site).
Static-shape kernels never reach it, so they must not pay the construction.

These tests pin both sides of the lazy construction:

  1. static-shape path: ``Analyzer()`` must NOT be constructed at all (the
     wrapper module's global is patched with a stub that raises on
     construction);
  2. symbolic-shape path: ``Analyzer()`` is constructed on demand and the
     symbolic output shape is still resolved (patched with a counting
     subclass that delegates to the real analyzer).

The behavior under test lives in the Python/Cython wrapper and is identical
for every codegen target, so both kernels below run on the default
``ascendc`` target (keeps the compile budget at two small kernels).

NOTE: these cases execute on real NPU hardware (``.npu()``); they cannot run
in a CPU-only environment.
"""

import pytest
import torch

import tilelang
import tilelang.ascend.a2a3.language as T
from tvm.arith import Analyzer

BLOCK_M = 64
BLOCK_N = 64

# Same vector-scope config as test_tilelang_ascend_language_gm_to_ub.py: the
# auto-sync pass inserts the UB load -> store dependency for the copy pairs.
VEC_PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}


def static_copy_kernel():
    """Static-shape GM -> UB -> GM identity copy (empty dynamic_symbolic_map)."""

    @T.prim_func
    def main(A: T.Tensor((BLOCK_M, BLOCK_N), "float32"), C: T.Tensor((BLOCK_M, BLOCK_N), "float32")):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            a_ub = T.alloc_ub((BLOCK_M, BLOCK_N), "float32")
            T.copy(A[0, 0], a_ub)
            T.copy(a_ub, C[0, 0])

    return main


def dyn_m_copy_kernel():
    """Symbolic-M GM -> UB -> GM identity copy: the result shape dim M is a
    tir.PrimExpr and goes through analyzer.simplify in forward()."""

    M = T.symbolic("M")
    m_num = T.ceildiv(M, BLOCK_M)

    @T.prim_func
    def main(A: T.Tensor((M, BLOCK_N), "float32"), C: T.Tensor((M, BLOCK_N), "float32")):
        with T.Kernel(m_num, is_npu=True) as (cid, vid):
            a_ub = T.alloc_ub((BLOCK_M, BLOCK_N), "float32")
            T.copy(A[cid * BLOCK_M, 0], a_ub)
            T.copy(a_ub, C[cid * BLOCK_M, 0])

    return main


def _wrapper_module(kernel):
    """Patch the Analyzer factory used by modern A2/A3 allocation resolution."""
    from tilelang.ascend.a2a3 import runtime

    return runtime.tvm.arith


@pytest.mark.a2a3_device
def test_static_shape_forward_does_not_construct_analyzer(monkeypatch):
    kernel = tilelang.compile(static_copy_kernel(), out_idx=[-1], pass_configs=VEC_PASS_CONFIGS, target="ascendc")
    wrapper_module = _wrapper_module(kernel)
    assert hasattr(wrapper_module, "Analyzer")

    class _BoomAnalyzer:
        def __init__(self, *args, **kwargs):
            raise AssertionError("Analyzer() must not be constructed for static-shape kernels")

    monkeypatch.setattr(wrapper_module, "Analyzer", _BoomAnalyzer)

    a = torch.randn(BLOCK_M, BLOCK_N).npu()
    c = kernel(a)
    torch.testing.assert_close(c, a, rtol=1e-2, atol=1e-2)


@pytest.mark.a2a3_device
def test_symbolic_output_shape_constructs_analyzer_lazily(monkeypatch):
    torch.manual_seed(0)
    # Symbolic-var kernels must compile without the disk cache: pytest-xdist
    # workers racing on one cache entry can lose the dynamic_symbolic_map
    # (same rationale as run_test_dyn_m in test_tilelang_ascend_language_gm_to_ub.py).
    tilelang.disable_cache()
    kernel = tilelang.compile(dyn_m_copy_kernel(), out_idx=[-1], pass_configs=VEC_PASS_CONFIGS, target="ascendc")
    tilelang.enable_cache()

    wrapper_module = _wrapper_module(kernel)
    assert hasattr(wrapper_module, "Analyzer")

    constructions = []

    class _CountingAnalyzer(Analyzer):
        def __init__(self, *args, **kwargs):
            constructions.append(1)
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(wrapper_module, "Analyzer", _CountingAnalyzer)

    for m in (64, 128, 50):
        a = torch.randn(m, BLOCK_N).npu()
        c = kernel(a)
        assert c.shape == (m, BLOCK_N)
        torch.testing.assert_close(c, a, rtol=1e-2, atol=1e-2)
    assert constructions, "Analyzer() must be constructed when a symbolic shape needs simplification"
