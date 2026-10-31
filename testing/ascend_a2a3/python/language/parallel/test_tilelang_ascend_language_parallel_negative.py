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
import tilelang.ascend.a2a3.language as T
from tilelang.ascend.a2a3 import kernel_cache
import tvm

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}

# C++ LOG(FATAL)/ICHECK surface as tvm.TVMError; keep ValueError for any
# Python-side validation that might front-run the backend check.
COMPILE_ERRORS = (tvm.TVMError, ValueError)

pytestmark = pytest.mark.a2a3_native

TARGET = {"kind": "ascend", "arch": "dav-2201", "model": "A3", "keys": ["ascendc", "ascend"]}


@pytest.fixture(autouse=True)
def clear_cache(isolated_a2a3_cache):
    """Clear tilelang cache before tests."""
    from tilelang.ascend.a2a3 import compiler

    # Public compile freezes its toolchain before lowering. The invalid loop
    # must fail before invoking it, but discovery still needs the source SDK.
    try:
        compiler.resolve_compile_config(tvm.target.Target(TARGET))
    except FileNotFoundError as error:
        pytest.skip(f"requires the actual AscendC toolchain configuration: {error}")
    kernel_cache.clear_cache()
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


def test_parallel_3d_is_rejected(monkeypatch):
    from tilelang.ascend.a2a3 import compiler

    monkeypatch.setattr(
        compiler.LibraryGenerator, "compile_lib", lambda *args, **kwargs: pytest.fail("3D rejection must precede SDK compilation")
    )
    func = make_parallel_3d_func()
    with pytest.raises(COMPILE_ERRORS, match="3D or higher dimensional parallel loops"):
        _compile(func)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
