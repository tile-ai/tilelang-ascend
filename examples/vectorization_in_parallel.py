# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.

import os

import tilelang
import tilelang.language as T

import torch
import torch_npu

# Clear any previously cached compiled kernels to ensure a clean run
tilelang.cache.clear_cache()

# Define data type and sequence length for the vector addition
dtype = "float32"
seq_len = 4096  # Length of the vectors to be added


def binary_simple(N, block_N, dtype="float32"):
    n_num = N // block_N  # Number of blocks (each block processes `block_N` elements)

    @T.prim_func
    def binarySimple(A: T.Tensor((N), dtype), B: T.Tensor((N), dtype), C: T.Tensor((N), dtype), shape: T.int32):
        # Launch kernel with `n_num` parallel threads on the NPU
        with T.Kernel(n_num, is_npu=True) as (cid, _):
            # Allocate on-chip Unified Buffer (UB) for local computation
            A_VEC = T.alloc_ub((block_N), dtype)
            B_VEC = T.alloc_ub((block_N), dtype)
            C_VEC = T.alloc_ub((block_N), dtype)

            # Calculate the starting index for this thread
            start_idx = cid * block_N
            # Compute remaining elements from this start index to the end of the tensor
            remaining = shape - start_idx
            # Determine how many elements this thread should actually process (handles tail)
            tail_size = T.min(block_N, remaining)

            # Copy data from global memory (A, B) into on-chip buffers (A_VEC, B_VEC)
            T.copy(A[start_idx], A_VEC, [tail_size])
            T.copy(B[start_idx], B_VEC, [tail_size])

            for i in T.Parallel(block_N):
                C_VEC[i] = A_VEC[i] + B_VEC[i]

            # Write the result back from on-chip buffer (C_VEC) to global memory (C)
            T.copy(C_VEC, C[start_idx], [tail_size])

    return binarySimple


def binary_compound(N, block_N, dtype="float32"):
    n_num = N // block_N  # Number of blocks (each block processes `block_N` elements)

    @T.prim_func
    def binaryCompound(A: T.Tensor((N), dtype), B: T.Tensor((N), dtype), C: T.Tensor((N), dtype), shape: T.int32):
        # Launch kernel with `n_num` parallel threads on the NPU
        with T.Kernel(n_num, is_npu=True) as (cid, _):
            # Allocate on-chip Unified Buffer (UB) for local computation
            A_VEC = T.alloc_ub((block_N), dtype)
            B_VEC = T.alloc_ub((block_N), dtype)
            C_VEC = T.alloc_ub((block_N), dtype)

            # Calculate the starting index for this thread
            start_idx = cid * block_N
            # Compute remaining elements from this start index to the end of the tensor
            remaining = shape - start_idx
            # Determine how many elements this thread should actually process (handles tail)
            tail_size = T.min(block_N, remaining)

            # Copy data from global memory (A, B) into on-chip buffers (A_VEC, B_VEC)
            T.copy(A[start_idx], A_VEC, [tail_size])
            T.copy(B[start_idx], B_VEC, [tail_size])

            for i in T.Parallel(block_N):
                C_VEC[i] = A_VEC[i] + B_VEC[i] * 3.14

            # Write the result back from on-chip buffer (C_VEC) to global memory (C)
            T.copy(C_VEC, C[start_idx], [tail_size])

    return binaryCompound


def binary_compound_elementwise(N, block_N, dtype="float32"):
    n_num = N // block_N  # Number of blocks (each block processes `block_N` elements)

    @T.prim_func
    def binaryCompoundElementwise(A: T.Tensor((N), dtype), B: T.Tensor((N), dtype), C: T.Tensor((N), dtype), shape: T.int32):
        # Launch kernel with `n_num` parallel threads on the NPU
        with T.Kernel(n_num, is_npu=True) as (cid, _):
            # Allocate on-chip Unified Buffer (UB) for local computation
            A_VEC = T.alloc_ub((block_N), dtype)
            B_VEC = T.alloc_ub((block_N), dtype)
            C_VEC = T.alloc_ub((block_N), dtype)

            # Calculate the starting index for this thread
            start_idx = cid * block_N
            # Compute remaining elements from this start index to the end of the tensor
            remaining = shape - start_idx
            # Determine how many elements this thread should actually process (handles tail)
            tail_size = T.min(block_N, remaining)

            # Copy data from global memory (A, B) into on-chip buffers (A_VEC, B_VEC)
            T.copy(A[start_idx], A_VEC, [tail_size])
            T.copy(B[start_idx], B_VEC, [tail_size])

            for i in T.Parallel(block_N):
                C_VEC[i] = A_VEC[i] * B_VEC[i] + A_VEC[i]

            # Write the result back from on-chip buffer (C_VEC) to global memory (C)
            T.copy(C_VEC, C[start_idx], [tail_size])

    return binaryCompoundElementwise


def test_binary_simple(v1, v2, v3):
    # Instantiate the vector addition kernel for the full sequence length (single block)
    func = binary_simple(seq_len, seq_len)

    # Compile the TileLang function to NPU IR for execution on the NPU
    compiled_kernel = tilelang.compile(func, target="npuir")

    # Compute reference result using PyTorch's native addition (on NPU)
    y_ref = v1 + v2

    # Launch the compiled TileLang kernel
    compiled_kernel(v1, v2, v3, seq_len)

    # Print both results for visual comparison (should be nearly identical)
    print("Reference result (PyTorch):")
    print(y_ref)
    print("TileLang kernel result:")
    print(v3)


def test_binary_compound(v1, v2, v3):
    # Instantiate the vector addition kernel for the full sequence length (single block)
    func = binary_compound(seq_len, seq_len)

    # Compile the TileLang function to NPU IR for execution on the NPU
    compiled_kernel = tilelang.compile(func, target="npuir")

    # Compute reference result using PyTorch's native addition (on NPU)
    y_ref = v1 + v2 * 3.14

    # Launch the compiled TileLang kernel
    compiled_kernel(v1, v2, v3, seq_len)

    # Print both results for visual comparison (should be nearly identical)
    print("Reference result (PyTorch):")
    print(y_ref)
    print("TileLang kernel result:")
    print(v3)


def test_binary_compound_elementwise(v1, v2, v3):
    # Instantiate the vector addition kernel for the full sequence length (single block)
    func = binary_compound_elementwise(seq_len, seq_len)

    # Compile the TileLang function to NPU IR for execution on the NPU
    compiled_kernel = tilelang.compile(func, target="npuir")

    # Compute reference result using PyTorch's native addition (on NPU)
    y_ref = v1 * v2 + v1

    # Launch the compiled TileLang kernel
    compiled_kernel(v1, v2, v3, seq_len)

    # Print both results for visual comparison (should be nearly identical)
    print("Reference result (PyTorch):")
    print(y_ref)
    print("TileLang kernel result:")
    print(v3)


if __name__ == "__main__":
    torch.npu.set_device(6)
    # Create random input tensors on the NPU
    v1 = torch.randn(size=[seq_len], dtype=eval("torch." + dtype)).npu()
    v2 = torch.randn(size=[seq_len], dtype=eval("torch." + dtype)).npu()
    v3 = torch.zeros(size=[seq_len], dtype=eval("torch." + dtype)).npu()  # Output buffer

    test_binary_simple(v1, v2, v3)
    test_binary_compound(v1, v2, v3)
    test_binary_compound_elementwise(v1, v2, v3)

