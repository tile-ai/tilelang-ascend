"""T.Parallel constraint / negative tests.

These assert that the documented T.Parallel constraints are actually enforced at
compile time:

* >= 3D parallel loops are rejected
  (src/transform/ascend_lower_parallel_to_vector.cc: LOG(FATAL)).

The errors are raised by the C++ backend during ``tilelang.compile`` (TVM
``LOG(FATAL)`` surfaces as ``tvm.TVMError``), so they require the Ascend
toolchain to reproduce.  The exact exception type may need adjustment if the
backend wraps it differently -- if so, widen the ``pytest.raises`` tuple.
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
