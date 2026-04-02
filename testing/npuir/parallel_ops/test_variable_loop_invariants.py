# Copyright (c) Huawei Technologies Co., Ltd. 2025.
import pytest
import torch

import tilelang
import tilelang.language as T

from testcommon import assert_close, gen_tensor

pytestmark = [
    pytest.mark.op("parallel"),
    pytest.mark.mode("Developer"),
]

DTYPES = ["float16"]


def parallel_add_scalar_invariant_dev(M, N, dtype):
    @T.prim_func
    def parallelAddScalarInvariantDev(
        A: T.Tensor((M, N), dtype),
        B: T.Tensor((M, N), dtype),
        C: T.float32,
    ):
        with T.Kernel(1, is_npu=True) as (cid, _):
            a = T.alloc_shared((M, N), dtype)
            b = T.alloc_shared((M, N), dtype)
            c_buf = T.alloc_shared((1,), dtype)

            T.copy(A, a)

            # materialize runtime scalar to buffer
            c_buf[0] = T.Cast(dtype, C)

            # 2D parallel elementwise add
            for i, j in T.Parallel(M, N):
                b[i, j] = a[i, j] + c_buf[0]

            T.copy(b, B)

    return parallelAddScalarInvariantDev


@pytest.mark.parametrize("dtype", DTYPES)
def test_parallel_add_scalar_invariant_dev(dtype):
    M, N = 16, 16

    A = gen_tensor((M, N), dtype, kind="randn")
    B = gen_tensor((M, N), dtype, kind="zeros")
    C = 2.0

    func = parallel_add_scalar_invariant_dev(M=M, N=N, dtype=dtype)
    compiled = tilelang.compile(func, target="npuir")
    compiled(A, B, C)

    c_ref = torch.tensor(C, dtype=A.dtype, device=A.device)
    ref = A + c_ref
    assert_close(B, ref, rtol=1e-2, atol=1e-2)
