import pytest
import torch
import torch_npu  # noqa: F401

import tilelang
import tilelang.language as T

from testcommon import assert_close, gen_tensor


pytestmark = [
    pytest.mark.op("embedding"),
    pytest.mark.mode("Developer"),
]

INDEXED_LOAD_CASES = [(1024, 256, 512, 64)]


@tilelang.jit(target="npuir")
def kernel_embedding_1d(seq_len, table_len, dim, block_s):
    dtype = "float32"
    idx_dtype = "int32"

    @T.prim_func
    def embedding1d(
        indices: T.Tensor((seq_len), idx_dtype),
        table: T.Tensor((table_len, dim), dtype),
        output: T.Tensor((seq_len, dim), dtype),
    ):
        with T.Kernel(T.ceildiv(seq_len, block_s), is_npu=True) as (cid, _):
            indices_shared = T.alloc_shared((block_s,), idx_dtype)
            output_shared = T.alloc_shared((block_s, dim), dtype)

            T.copy(indices[cid * block_s], indices_shared)

            for i in T.serial(block_s):
                idx = indices_shared[i]
                for j in T.serial(dim):
                    output_shared[i, j] = table[idx, j]

            T.copy(output_shared, output[cid * block_s, 0])

    return embedding1d


@pytest.mark.parametrize("seq_len, table_len, dim, block_s", INDEXED_LOAD_CASES)
def test_indexed_load(seq_len, table_len, dim, block_s):
    kernel = kernel_embedding_1d(seq_len, table_len, dim, block_s)

    indices = gen_tensor((seq_len,), "int32", kind="randint", low=0, high=table_len)
    table = gen_tensor((table_len, dim), "float32", kind="randn")
    output = gen_tensor((seq_len, dim), "float32", kind="zeros")

    ref_output = torch.nn.functional.embedding(indices, table)
    kernel(indices, table, output)

    assert_close(output.cpu(), ref_output.cpu(), dtype="float32", rtol=1e-2, atol=1e-2)
