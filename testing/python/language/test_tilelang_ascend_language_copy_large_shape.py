"""Regression tests for Ascend copies from logically large GM buffers."""

import tilelang
import tilelang.language as T


@T.prim_func
def copy_one_row_from_large_gm(
    A: T.Tensor((4, 4225, 5, 256, 128), "float16"),
    C: T.Tensor((128,), "float16"),
):
    with T.Kernel(1, is_npu=True) as (cid, vid):
        a_ub = T.alloc_ub((128,), "float16")
        with T.Scope("V"):
            T.copy(A[3, 4224, 4, 255, 0], a_ub)
            T.copy(a_ub, C)


def test_copy_one_row_from_large_gm_lowers_without_int32_overflow():
    artifact = tilelang.lower(copy_one_row_from_large_gm, target="ascendc")
    assert "2768895872" in artifact.kernel_source
    assert "2768896000" in artifact.kernel_source
    assert "-1526071424" not in artifact.kernel_source
    assert "-1526071296" not in artifact.kernel_source
