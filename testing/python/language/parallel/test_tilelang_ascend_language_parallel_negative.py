"""T.Parallel constraint / negative tests.

These assert that the documented T.Parallel constraints are actually enforced at
compile time:

* >= 3D parallel loops are rejected
  (src/transform/ascend_lower_parallel_to_vector.cc: LOG(FATAL)).
* ``coalesced_width`` that does not divide the inferred vector size is rejected
  (src/op/parallel.cc: LOG(FATAL)).
* A ``local.fragment`` buffer accessed with structurally inconsistent indices is
  rejected (src/op/parallel.cc: ICHECK).

The errors are raised by the C++ backend during ``tilelang.compile`` (TVM
``LOG(FATAL)`` / ``ICHECK`` surface as ``tvm.TVMError``), so they require the
Ascend toolchain to reproduce.  The exact exception type may need adjustment if
the backend wraps it differently -- if so, widen the ``pytest.raises`` tuple.
"""

import pytest
import tilelang
import tilelang.language as T
import tvm

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}

# C++ LOG(FATAL)/ICHECK surface as tvm.TVMError; keep ValueError for any
# Python-side validation that might front-run the backend check.
COMPILE_ERRORS = (tvm.TVMError, ValueError)

TARGET = "ascendc"


@pytest.fixture(scope="session", autouse=True)
def clear_cache():
    """Clear tilelang cache before tests."""
    tilelang.cache.clear_cache()
    yield


def _compile(func):
    return tilelang.compile(func, out_idx=[-1], pass_configs=pass_configs, target=TARGET)


# ---------------------------------------------------------------------------
# >= 3D parallel loop must be rejected.
# ---------------------------------------------------------------------------
def make_parallel_3d_func(dtype="float"):
    D0, D1, D2 = 4, 8, 64

    @T.prim_func
    def main(A: T.Tensor((D0, D1, D2), dtype), C: T.Tensor((D0, D1, D2), dtype)):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            a_ub = T.alloc_ub((D0, D1, D2), dtype)
            c_ub = T.alloc_ub((D0, D1, D2), dtype)
            with T.Scope("V"):
                for i, j, k in T.Parallel(D0, D1, D2):
                    a_ub[i, j, k] = A[i, j, k]
                for i, j, k in T.Parallel(D0, D1, D2):
                    c_ub[i, j, k] = a_ub[i, j, k] + 1.0
                for i, j, k in T.Parallel(D0, D1, D2):
                    C[i, j, k] = c_ub[i, j, k]

    return main


def test_parallel_3d_is_rejected():
    func = make_parallel_3d_func()
    with pytest.raises(COMPILE_ERRORS):
        _compile(func)


# ---------------------------------------------------------------------------
# coalesced_width that does not divide the inferred vector size must be rejected.
# 7 is prime and cannot divide the power-of-two vector size of this loop.
# ---------------------------------------------------------------------------
def make_bad_coalesced_width_func(M=1024, N=1024, block_M=128, block_N=128, dtype="float"):
    m_num = M // block_M
    n_num = N // block_N
    rows = block_M // 2

    @T.prim_func
    def main(A: T.Tensor((M, N), dtype), C: T.Tensor((M, N), dtype)):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num
            a_ub = T.alloc_ub((rows, block_N), dtype)
            c_ub = T.alloc_ub((rows, block_N), dtype)
            with T.Scope("V"):
                T.copy(A[bx * block_M + vid * rows, by * block_N], a_ub)
                for i, j in T.Parallel(rows, block_N, coalesced_width=7):
                    c_ub[i, j] = a_ub[i, j] + 1.0
                T.copy(c_ub, C[bx * block_M + vid * rows, by * block_N])

    return main


def test_parallel_bad_coalesced_width_is_rejected():
    func = make_bad_coalesced_width_func()
    with pytest.raises(COMPILE_ERRORS):
        _compile(func)


# ---------------------------------------------------------------------------
# A local.fragment buffer written with structurally inconsistent indices
# (frag[i, j] vs frag[j, i]) must be rejected by the index-consistency ICHECK.
# Note: the guard only fires for the "local.fragment" scope (alloc_fragment),
# not for UB (alloc_ub).
# ---------------------------------------------------------------------------
def make_inconsistent_fragment_index_func(BM=64, dtype="float"):
    @T.prim_func
    def main(A: T.Tensor((BM, BM), dtype), C: T.Tensor((BM, BM), dtype)):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            a_ub = T.alloc_ub((BM, BM), dtype)
            frag = T.alloc_fragment((BM, BM), dtype)
            with T.Scope("V"):
                T.copy(A[0, 0], a_ub)
                for i, j in T.Parallel(BM, BM):
                    frag[i, j] = a_ub[i, j]
                    frag[j, i] = a_ub[i, j] + 1.0  # inconsistent index -> ICHECK
                for i, j in T.Parallel(BM, BM):
                    C[i, j] = frag[i, j]

    return main


def test_parallel_inconsistent_fragment_index_is_rejected():
    func = make_inconsistent_fragment_index_func()
    with pytest.raises(COMPILE_ERRORS):
        _compile(func)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
