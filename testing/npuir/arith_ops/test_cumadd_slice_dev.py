# Copyright (c) Huawei Technologies Co., Ltd. 2025.
import torch
import tilelang
import tilelang.language as T

import pytest
import testcommon as tc

DATATYPE_CASES = ["float16", "float32"]
pytestmark = [pytest.mark.mode("Developer")]


@tilelang.jit(target="npuir")
def slice_add(block_M, block_N, dtype="float16"):
    M = T.symbolic("M")
    N = T.symbolic("N")
    BLOCK_SIZE = 1

    @T.prim_func
    def sliceAdd(Input: T.Tensor((M, N), dtype), Output: T.Tensor((1, N), dtype)):
        with T.Kernel(BLOCK_SIZE, is_npu=True) as (cid, _):
            src = T.alloc_shared([block_M, block_N], dtype=dtype)
            dst = T.alloc_shared([1, block_N], dtype=dtype)

            for i in T.serial(T.ceildiv(N, block_N)):
                offset_n = i * block_N
                remain_n = T.min(block_N, N - offset_n)
                value_zero = 0
                T.npuir_brc(value_zero, dst)
                for j in T.serial(T.ceildiv(M, block_M)):
                    offset_m = j * block_M
                    remain_m = T.min(block_M, M - offset_m)
                    T.copy(
                        Input[
                            offset_m : offset_m + remain_m,
                            offset_n : offset_n + remain_n,
                        ],
                        src[0:remain_m, 0:remain_n],
                    )
                    for k in T.serial(remain_m):
                        T.npuir_add(src[k : k + 1, :], dst, dst)
                T.copy(dst[0, 0:remain_n], Output[0:1, offset_n : offset_n + remain_n])

    return sliceAdd


@pytest.mark.op("slice_add_dev")
@pytest.mark.parametrize("dtype", DATATYPE_CASES)
def test_slice_add(dtype):
    kernel = slice_add(32, 32, dtype)

    datatype = tc.resolve_dtype(dtype)

    # case 1
    M, N = 2, 256
    input = torch.randn([M, N], dtype=datatype).npu()
    output = torch.randn([1, N], dtype=datatype).npu()
    kernel(input, output)
    ref_output = torch.sum(input, dim=0, keepdim=True)

    tc.assert_close(output, ref_output, rtol=1e-2, atol=1e-2)

    # case 2
    M, N = 3, 466
    input = torch.randn([M, N], dtype=datatype).npu()
    output = torch.randn([1, N], dtype=datatype).npu()
    kernel(input, output)
    ref_output = torch.sum(input, dim=0, keepdim=True)

    tc.assert_close(output, ref_output, rtol=1e-2, atol=1e-2)
