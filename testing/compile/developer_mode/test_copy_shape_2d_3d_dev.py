# Copyright (c) Huawei Technologies Co., Ltd. 2025.
import pytest
import tilelang

from testcommon import npuir_compile_to_bin

pytestmark = [pytest.mark.mode("Developer")]
import tilelang.language as T

tilelang.cache.clear_cache()
dtype = "float16"


def copy_shape_2d_3d(M, N, block_M, block_N):
    @T.prim_func
    def copyShape2D3D(
        A: T.Tensor((1, M, N), dtype),
        B: T.Tensor((1, M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(M, block_M) * T.ceildiv(N, block_N), is_npu=True) as (
            cid,
            _,
        ):
            blockx = cid // T.ceildiv(N, block_N)
            blocky = cid % T.ceildiv(N, block_N)
            by = blocky * block_N

            A_BUF = T.alloc_shared((1, block_N), dtype)

            for i in T.Parallel(block_M):
                bx = blockx * block_M + i
                T.copy(A[0, bx, by], A_BUF)
                T.copy(A_BUF, B[0, bx, by])

    return copyShape2D3D


def test_copy_shape_2d_3d():
    # In the future, Developer mode and Expert Mode will transition smoothly without
    # requiring explicit declarations.
    M = 256
    N = 1024
    # In the future, it will be optimized to automatically derive the workspace size.
    func = copy_shape_2d_3d(M, N, 32, 32)
    kernel = tilelang.engine.lower(func, target="npuir")
    result = npuir_compile_to_bin(kernel)
    assert result is not None and len(result) > 0, (
        "npuir compile failed or returned empty"
    )


if __name__ == "__main__":
    test_copy_shape_2d_3d()
