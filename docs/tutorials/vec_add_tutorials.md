TileLang is a domain-specific language designed to simplify the process of writing high-performance kernels for various hardware accelerators. This tutorial demonstrates how to write and optimize a vector addition (VecAdd) kernel using TileLang for NPU (Neural Processing Unit) hardware.

## Table of Contents

1. [Introduction](#introduction)
2. [Prerequisites](#prerequisites)
3. [Vector Addition Example](#vector-addition-example)
   - [Code Walkthrough](#code-walkthrough)
   - [Compiling and Running](#compiling-and-running)
4. [Understanding the VecAdd Kernel](#understanding-the-vecadd-kernel)
5. [Performance Considerations](#performance-considerations)
6. [Verifying Correctness](#verifying-correctness)
8. [Summary](#summary)

---

## Introduction

Vector addition is one of the simplest yet fundamental operations in parallel computing. While basic in concept, implementing it efficiently on specialized hardware like NPUs requires careful consideration of memory access patterns and hardware-specific optimizations.

This tutorial shows how to implement a vector addition kernel using TileLang, targeting Huawei's NPU hardware through the NPUIR (NPU Intermediate Representation) backend.

## Prerequisites

- **Python 3.8+**
- **Huawei NPU** with appropriate drivers installed
- **PyTorch** with NPU support

### Installation

```bash
pip install torch_npu
```

Note: You may need to install additional dependencies specific to your NPU environment.

---

## Vector Addition Example

Below is a complete implementation of vector addition using TileLang:

```python
# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
import os

import tilelang
import tilelang.language as T

import torch
import torch_npu

tilelang.cache.clear_cache()

dtype = "float32"
seq_len = 4096

def vec_add(N, block_N, dtype="float32"):
    n_num = N // block_N

    @T.prim_func
    def main(
        A: T.Tensor((N), dtype),
        B: T.Tensor((N), dtype),
        C: T.Tensor((N), dtype),
        shape: T.int32,
    ):
        with T.Kernel(n_num, is_npu=True) as (cid, _):
            A_VEC = T.alloc_ub((block_N), dtype)
            B_VEC = T.alloc_ub((block_N), dtype)
            C_VEC = T.alloc_ub((block_N), dtype)
            t0 = cid * block_N
            t0 = shape - t0
            tail_size = T.min(block_N, t0)
            T.copy(A[cid * block_N : cid * block_N + tail_size], A_VEC[0:tail_size])
            T.copy(B[cid * block_N : cid * block_N + tail_size], B_VEC[0:tail_size])

            T.npuir_add(A_VEC, B_VEC, C_VEC)
            T.copy(C_VEC[0:tail_size], C[cid * block_N : cid * block_N + tail_size])
    return main

def test_vec_add():
    torch.npu.set_device(6)
    func = vec_add(seq_len, seq_len)
    compiled_kernel = tilelang.compile(func, target="npuir")

    v1 = torch.randn(size=[seq_len], dtype=eval("torch." + dtype)).npu()
    v2 = torch.randn(size=[seq_len], dtype=eval("torch." + dtype)).npu()
    v3 = torch.zeros(size=[seq_len], dtype=eval("torch." + dtype)).npu()

    y_ref = v1 + v2
    compiled_kernel(v1, v2, v3, seq_len)

    print(y_ref)
    print(v3)

if __name__ == "__main__":
    test_vec_add()
```

### Code Walkthrough

1. **Kernel Definition:**
   ```python
   def vec_add(N, block_N, dtype="float32"):
       n_num = N // block_N
       
       @T.prim_func
       def main(
           A: T.Tensor((N), dtype),
           B: T.Tensor((N), dtype),
           C: T.Tensor((N), dtype),
           shape: T.int32,
       ):
   ```
   The `vec_add` function returns a TileLang primitive function that adds two vectors. It takes the vector length `N`, block size `block_N`, and data type as parameters.

2. **Kernel Launch Configuration:**
   ```python
   with T.Kernel(n_num, is_npu=True) as (cid, _):
   ```
   This creates a 1D grid of `n_num` blocks (where `n_num = N // block_N`), with each block processing `block_N` elements. The `is_npu=True` parameter indicates this kernel targets NPU hardware.

3. **Memory Allocation:**
   ```python
   A_VEC = T.alloc_ub((block_N), dtype)
   B_VEC = T.alloc_ub((block_N), dtype)
   C_VEC = T.alloc_ub((block_N), dtype)
   ```
   These lines allocate memory in the NPU's unified buffer (UB) for temporary storage of vector chunks during computation.

4. **Boundary Handling:**
   ```python
   t0 = cid * block_N
   t0 = shape - t0
   tail_size = T.min(block_N, t0)
   ```
   This code handles the case where the vector length isn't evenly divisible by the block size, ensuring we don't access memory beyond the vector boundaries.

5. **Data Movement:**
   ```python
   T.copy(A[cid * block_N : cid * block_N + tail_size], A_VEC[0:tail_size])
   T.copy(B[cid * block_N : cid * block_N + tail_size], B_VEC[0:tail_size])
   ```
   These copy operations move chunks of the input vectors from global memory to the faster UB memory.

6. **Computation:**
   ```python
   T.npuir_add(A_VEC, B_VEC, C_VEC)
   ```
   This is the actual vector addition operation, performed using the NPUIR intrinsic for addition.

7. **Result Storage:**
   ```python
   T.copy(C_VEC[0:tail_size], C[cid * block_N : cid * block_N + tail_size])
   ```
   The result is copied back from UB to global memory.

### Compiling and Running

The `test_vec_add()` function demonstrates how to compile and run the kernel:

1. **Set Device:**
   ```python
   torch.npu.set_device(6)
   ```
   Selects which NPU device to use.

2. **Compilation:**
   ```python
   func = vec_add(seq_len, seq_len)
   compiled_kernel = tilelang.compile(func, target="npuir")
   ```
   Compiles the TileLang function to NPUIR.

3. **Data Preparation:**
   ```python
   v1 = torch.randn(size=[seq_len], dtype=eval("torch." + dtype)).npu()
   v2 = torch.randn(size=[seq_len], dtype=eval("torch." + dtype)).npu()
   v3 = torch.zeros(size=[seq_len], dtype=eval("torch." + dtype)).npu()
   ```
   Creates input and output tensors on the NPU.

5. **Kernel Execution:**
   ```python
   compiled_kernel(v1, v2, v3, seq_len)
   ```
   Executes the compiled TileLang kernel.

---

## Understanding the VecAdd Kernel

The vector addition kernel demonstrates several important concepts in TileLang programming for NPUs:

1. **Block-based Processing:** The vector is divided into blocks, with each block processed by a separate hardware unit. This enables parallel execution.

2. **Memory Hierarchy:** The kernel efficiently uses the NPU's memory hierarchy by:
   - Loading data from global memory to unified buffer (UB)
   - Performing computations on data in UB
   - Storing results back to global memory

3. **Boundary Handling:** The kernel properly handles cases where the vector size isn't evenly divisible by the block size.

4. **Hardware Intrinsics:** The `T.npuir_add` intrinsic maps directly to hardware instructions for efficient computation.

---

## Performance Considerations

When optimizing vector operations on NPUs:

1. **Block Size Selection:** Choose a block size that:
   - Maximizes parallelization
   - Fits within the UB memory constraints
   - Aligns with hardware capabilities

2. **Memory Access Patterns:** Ensure contiguous memory access where possible to maximize bandwidth utilization.

3. **Data Type Selection:** Use appropriate data types for your workload (e.g., float16 for machine learning workloads where precision requirements allow).

---

## Verifying Correctness

The example includes a basic correctness check by comparing the TileLang result with a PyTorch reference implementation:

```python
y_ref = v1 + v2
compiled_kernel(v1, v2, v3, seq_len)

print(y_ref)
print(v3)
```

---

## Summary

This tutorial provides a foundation for writing efficient vector operations using TileLang on NPU hardware. As you become more familiar with TileLang and NPU programming, you can explore more complex operations and optimizations to fully leverage the capabilities of your hardware.
