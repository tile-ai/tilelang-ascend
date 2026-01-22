# Copyright (c) Huawei Technologies Co., Ltd. 2025.
import os
import argparse
import torch

import tilelang
import tilelang.language as T

parser = argparse.ArgumentParser(description="NPU Kernel Compilation - Embedding 1D Test")
parser.add_argument("--seq_len", type=int, default=1024,
                    help="Sequence length (number of indices to look up)")
parser.add_argument("--table_len", type=int, default=256,
                    help="Embedding table length (vocabulary size)")
parser.add_argument("--dim", type=int, default=512,
                    help="Embedding dimension (feature size)")
parser.add_argument("--block_s", type=int, default=64,
                    help="Block size for kernel parallelization")
parser.add_argument("--seed", type=int, default=88888888,
                    help="Random seed for reproducibility")


@tilelang.jit(target="npuir")
def kernel_embedding_1d(seq_len, table_len, dim, block_s):
    """
    TileLang-compiled 1D embedding kernel function
    Demonstrates optimization patterns using for-loops for scalar reads

    Args:
        seq_len: Sequence length
        table_len: Embedding table length
        dim: Embedding dimension
        block_s: Block size for parallel processing

    Returns:
        TileLang compiled kernel function
    """
    dtype = "float32"
    idx_dtype = "int32"

    @T.prim_func
    def embedding1d(
            indices: T.Tensor((seq_len), idx_dtype),  # Input indices
            table: T.Tensor((table_len, dim), dtype),  # Embedding table
            output: T.Tensor((seq_len, dim), dtype),  # Output embedding vectors
    ):
        """
        TileLang primitive function definition
        Implements scalar-level data reads and computation using for-loops
        """
        # Use block parallel strategy to process the sequence
        with T.Kernel(T.ceildiv(seq_len, block_s), is_npu=True) as (cid, _):
            # Allocate shared memory
            indices_shared = T.alloc_shared((block_s,), idx_dtype)
            output_shared = T.alloc_shared((block_s, dim), dtype)

            # Copy indices from global memory to shared memory
            T.copy(indices[cid * block_s], indices_shared)

            # Core computation: lookup embedding vectors for each index
            for i in T.serial(block_s):
                idx = indices_shared[i]  # Indexed load from ub
                for j in T.serial(dim):
                    # Indexed load from gm and indexed store to ub
                    output_shared[i, j] = table[idx, j]

            # Write results from shared memory back to global memory
            T.copy(output_shared, output[cid * block_s, 0])

    return embedding1d


def run_test(main_args):
    """
    Run embedding test with verification

    Args:
        main_args: Command line arguments object
    """
    # Compile TileLang kernel function
    kernel = kernel_embedding_1d(
        main_args.seq_len,
        main_args.table_len,
        main_args.dim,
        main_args.block_s,
    )

    # Set random seed for reproducibility
    torch.manual_seed(main_args.seed)

    # Define data types
    idx_dtype = torch.int32
    dtype = torch.float32

    # Create test data on NPU
    indices = torch.randint(
        size=(main_args.seq_len,),
        low=0,
        high=main_args.table_len,
        dtype=idx_dtype,
        device="npu"
    )
    table = torch.randn(
        size=(main_args.table_len, main_args.dim),
        dtype=dtype,
        device="npu"
    )
    output = torch.zeros(
        size=(main_args.seq_len, main_args.dim),
        dtype=dtype,
        device="npu"
    )

    # Compute reference output using PyTorch implementation
    ref_output = torch.nn.functional.embedding(indices, table)

    # Run TileLang kernel function
    kernel(indices, table, output)

    # Print results
    print("Actual Result (TileLang Kernel):")
    print(output)
    print("\nExpected Result (PyTorch Reference):")
    print(ref_output)

    # Verify correctness of results
    torch.testing.assert_close(output, ref_output, rtol=1e-2, atol=1e-2)
    print("\033[92mAll checks passed!\033[0m")


if __name__ == "__main__":
    # Set TileLang developer mode
    os.environ['TILELANG_ASCEND_MODE'] = 'Developer'

    # Parse command line arguments and run test
    args = parser.parse_args()
    run_test(args)