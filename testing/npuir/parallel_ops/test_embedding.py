# Copyright (c) Huawei Technologies Co., Ltd. 2025.
#
# This unit test verifies the support of T.Parallel for the following scenarios:
# 1. Vectorization of T.copy scenarios
# 2. Vectorization of T.vbrc scenarios
# 3. Correct vectorization of scenarios using expressions as indices
# 4. Correct exclusion (i.e., no vectorization) of indirect memory access scenarios
#

import pytest
import torch
import torch_npu  # noqa: F401

import tilelang
import tilelang.language as T
import testcommon as tc

pytestmark = [
    pytest.mark.op("parallel_embedding"),
    pytest.mark.mode("Developer"),
]

EMBEDDING_CASES = [
    (1001, 4441, 128, 64, 88888888),
]


@tilelang.jit(target="npuir")
def kernel_embedding_1d(dim, block):
    dtype = "float32"
    idx_dtype = "int32"

    seq_len = T.symbolic("seqLen")
    table_len = T.symbolic("tableLen")

    @T.prim_func
    def embedding1d(
            indices: T.Tensor((seq_len), idx_dtype),
            table: T.Tensor((table_len, dim), dtype),
            output: T.Tensor((seq_len, dim), dtype),
    ):
        with T.Kernel(T.ceildiv(seq_len, block), is_npu=True) as (cid, _):
            real_block = T.min(block, seq_len - cid * block)
            indices_shared = T.alloc_shared((block,), idx_dtype)
            output_shared = T.alloc_shared((block, dim), dtype)

            # Target: T.copy(indices[cid * block:cid * block + block], indices_shared[:])
            for i in T.Parallel(real_block):
                indices_shared[i] = indices[cid * block + i]
            # Target: T.vbrc(0, output_shared)
            for i, j in T.Parallel(block, dim):
                output_shared[i, j] = 0
            # Target: T.copy(table[indices_shared[i], :], output_shared[i, :])
            for i, j in T.Parallel(real_block, dim):
                output_shared[i, j] = table[indices_shared[i], j]

            T.copy(output_shared[:real_block, :], output[cid * block:cid * block + real_block, :])

    return embedding1d


@pytest.mark.parametrize("seq_len, table_len, dim, block, seed", EMBEDDING_CASES)
def test_parallel_embedding_1d(seq_len, table_len, dim, block, seed):
    kernel = kernel_embedding_1d(
        dim,
        block,
    )

    torch.manual_seed(seed)
    idx_dtype = torch.int32
    dtype = torch.float32

    indices = torch.randint(size=(seq_len,), low=0, high=table_len, dtype=idx_dtype, device="npu")
    table = torch.randn(size=(table_len, dim), dtype=dtype, device="npu")
    output = torch.zeros(size=(seq_len, dim), dtype=dtype, device="npu")

    ref_output = torch.nn.functional.embedding(indices, table)
    kernel(indices, table, output)

    tc.assert_close(output, ref_output, rtol=1e-2, atol=1e-2)
