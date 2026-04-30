# Copyright (c) Huawei Technologies Co., Ltd. 2025.
import pytest
import tilelang
import tilelang.language as T

from testcommon import assert_close, gen_tensor

pytestmark = [pytest.mark.mode("Expert")]
DTYPE_CASES = ["float16", "float32"]


def ref_program(a, b):
    k, _, _ = b.shape
    o = a.clone().detach()
    for i in range(k):
        o += b[i]
    return o


@tilelang.jit(target="npuir", out_idx=[2])
def reduce_add(M, N, K, block_M, block_N, dtype="float32"):
    @T.prim_func
    def reduceAdd(
        A: T.Tensor((M, N), dtype),
        B: T.Tensor((K, M, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N) * T.ceildiv(M, block_M), is_npu=True) as (
            cid,
            _,
        ):
            mid = cid // T.ceildiv(N, block_N)
            nid = cid % T.ceildiv(N, block_N)
            offset_m = mid * block_M
            offset_n = nid * block_N

            a = T.alloc_ub((block_M, block_N), dtype)
            b = T.alloc_ub((K, block_M, block_N), dtype)

            T.copy(A[offset_m : offset_m + block_M, offset_n : offset_n + block_N], a)
            T.copy(
                B[:, offset_m : offset_m + block_M, offset_n : offset_n + block_N], b
            )
            for i in T.serial(K):
                T.vadd(a, b[i, :, :], a)
            T.copy(a, C[offset_m : offset_m + block_M, offset_n : offset_n + block_N])

    return reduceAdd


@pytest.mark.op("vadd_rank_reduce_exp")
@pytest.mark.parametrize("dtype", DTYPE_CASES)
def test_vadd_rank_reduce_exp(dtype):
    M, N, K, BLOCK_M, BLOCK_N = 512, 512, 2, 64, 64
    a = gen_tensor((M, N), dtype, kind="randn")
    b = gen_tensor((K, M, N), dtype, kind="randn")

    kernel = reduce_add(M, N, K, BLOCK_M, BLOCK_N, dtype)
    o = kernel(a, b)

    assert_close(o, ref_program(a, b), rtol=1e-2, atol=1e-2)
