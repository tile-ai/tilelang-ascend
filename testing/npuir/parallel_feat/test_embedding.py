# Copyright (c) Huawei Technologies Co., Ltd. 2025.
#
# This unit test verifies the support of T.Parallel for the following scenarios:
# 1. Vectorization of T.copy scenarios
# 2. Vectorization of T.vbrc scenarios
# 3. Correct vectorization of scenarios using expressions as indices
# 4. Correct exclusion (i.e., no vectorization) of indirect memory access scenarios
#

import os
import argparse
import torch

import tilelang
import tilelang.language as T

parser = argparse.ArgumentParser(description="NPU Kernel Compilation")
parser.add_argument("--seq_len", type=int, default=1001,
                    help="Length of the input sequence")
parser.add_argument("--table_len", type=int, default=4441,
                    help="Size of the embedding table (number of embeddings)")
parser.add_argument("--dim", type=int, default=128,
                    help="Dimension of each embedding vector")
parser.add_argument("--block", type=int, default=64,
                    help="Block size for parallel processing")
parser.add_argument("--seed", type=int, default=88888888,
                    help="Random seed for reproducibility")


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
            for i in T.Parallel(block):
                indices_shared[i] = indices[cid * block + i]
            # Target: T.vbrc(0, output_shared)
            for i, j in T.Parallel(block, dim):
                output_shared[i, j] = 0
            # Target: T.copy(table[indices_shared[i], :], output_shared[i, :])
            for i, j in T.Parallel(real_block, dim):
                output_shared[i, j] = table[indices_shared[i], j]

            T.copy(output_shared[:real_block, :], output[cid * block:cid * block + real_block, :])

    return embedding1d


def run_test(main_args):
    kernel = kernel_embedding_1d(
        main_args.dim,
        main_args.block,
    )

    torch.manual_seed(main_args.seed)  # set the random seed for torch
    idx_dtype = torch.int32
    dtype = torch.float32

    indices = torch.randint(size=(main_args.seq_len,), low=0, high=main_args.table_len, dtype=idx_dtype, device="npu")
    table = torch.randn(size=(main_args.table_len, main_args.dim), dtype=dtype, device="npu")
    output = torch.zeros(size=(main_args.seq_len, main_args.dim), dtype=dtype, device="npu")

    ref_output = torch.nn.functional.embedding(indices, table)
    kernel(indices, table, output)

    torch.testing.assert_close(output, ref_output, rtol=1e-2, atol=1e-2)
    print("\033[92mAll check passed!\033[0m")


if __name__ == "__main__":
    os.environ['TILELANG_ASCEND_MODE'] = 'Developer'
    args = parser.parse_args()
    run_test(args)
