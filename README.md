<img src=./images/logo-row.svg />

<div align="center">

# TileLang-Ascend


[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/tile-ai/tilelang-ascend)

</div>

Tile Language Ascend (**tilelang-ascend**) is a specialized variant of the tile-lang domain-specific language, specifically optimized for Huawei Ascend NPU (Neural Processing Unit) architecture. Built upon the foundation of tile-lang's Pythonic syntax and [TVM](https://tvm.apache.org/) compiler infrastructure, tilelang-ascend enables developers to efficiently create high-performance AI compute kernels tailored for Ascend processors, including operations like GEMM, vector operations, and attention mechanisms. Tilelang-ascend allows developers to focus on productivity without sacrificing the low-level optimizations necessary for state-of-the-art performance on the NPU. The compiler backend supports two technical routes: [Ascend C & PTO](https://github.com/tile-ai/tilelang-ascend/tree/ascendc_pto) and [AscendNPU IR](https://github.com/tile-ai/tilelang-ascend/tree/npuir).

<p align="center">
  <img src="./images/tl-ascend-gemm.png" width="100%" alt="image">

</p>

## Latest News
- 12/08/2025 ✨: Added [T.Parallel](https://github.com/tile-ai/tilelang-ascend?tab=readme-ov-file#tparallel) support, check out [Pull Request#113](
https://github.com/tile-ai/tilelang-ascend/pull/113) for details.
- 11/25/2025 ✨: [Automatic buffer reuse](https://github.com/tile-ai/tilelang-ascend?tab=readme-ov-file#automatic-buffer-reuse) support, see [Pull Request#101](
https://github.com/tile-ai/tilelang-ascend/pull/101)!
- 11/17/2025 ✨: Added debug tools for tilelang-ascend—`T.printf` and `T.dump_tensor`, enabling users to [print and dump](https://github.com/tile-ai/tilelang-ascend/tree/ascendc_pto/examples/print) device-side buffers for easier inspection and troubleshooting.
- 11/07/2025 ✨: [Automatic insertion of intra-kernel synchronization](https://github.com/tile-ai/tilelang-ascend?tab=readme-ov-file#automatic-insertion-of-synchronization-instruction) support, see [Pull Request#74](
https://github.com/tile-ai/tilelang-ascend/pull/74)!
- 10/28/2025 🚀: Enhanced the performance of tl_templates and completed a high-performance [GEMM kernel](https://github.com/tile-ai/tilelang-ascend/blob/ascendc_pto/examples/gemm/example_gemm_intrinsic.py).
- 09/29/2025 🚀: We are excited to announce that tilelang-ascend, a dsl for high performance AI workloads on Ascend NPUs, is now open source and available to the public!

## Tested Devices
Although tilelang-ascend aims to be portable across a range of Ascend devices, it has been specifically tested and validated on the following NPUs: A2 and A3.

## OP Implementation Examples
**tilelang-ascend** provides the building blocks to implement a wide variety of operators on the NPU.
Some examples include:

- [Matrix Multiplication](./examples/gemm/)
- [Vector Add](./examples/elementwise/)
- [Flash Attention](./examples/flash_attention/)


Within the `examples` directory, you will also find additional complex kernels—such as [LightningIndexer](./examples/lightning_indexer/) and [SparseFlashAttention](./examples/sparse_flash_attention/), more operators will continuously be added.


## Installation

### Environment Preparation
We assume you already have an ascend environment with CANN (at least [8.3.RC1](https://www.hiascend.com/developer/download/community/result?cann=8.3.RC1&product=1&model=30)) and torch-npu (at least 2.6.0.RC1) installed. Firstly, set cann environment variables.

  ```bash
  source {your-cann-installed-path}/ascend-toolkit/set_env.sh
  ```

### TileLang-Ascend Installation

Here we use the method of compiling from source code for installation.

#### a) Download

    git clone --recursive https://github.com/tile-ai/tilelang-ascend.git
    cd tilelang-ascend

#### b) Compile and Install
    bash install_ascend.sh

#### c) Environment Variable Setup

    source set_env.sh

## Run


In this section, you will learn how to call NPU TileLang operators.

Here we use the **Matrix Multiplication** operator as an example for introduction.


```
cd examples/gemm
python example_gemm.py
```

Upon success, it will print:

```
Kernel Output Match!
```

## Comparison with NVIDIA Backend Implementation

GPUs primarily feature a three-level memory hierarchy that can be analogously mapped to NPU hardware architecture as follows:

**Memory Hierarchy Mapping:**
- `global memory` ↔ `global memory`
- `shared memory` ↔ `L1 buffer on cube core and unified buffer on vector core`  
- `register memory` ↔ `L0A/B/C buffer`

**Memory Management:**
TileLang-Ascend provides memory allocation primitives similar to the GPU version. For example, `alloc_{L1/ub/...}` functions allow on-chip memory allocation in a manner comparable to GPU programming.

**Execution Model Differences:**
At the execution level, NPUs lack thread-level abstractions. Therefore, we currently provide computation primitives operating at the `tile` granularity on vector cores. While the GPU version enables automatic parallelization of internal computations (e.g., addition) across different threads using `T.Parallel`, the NPU version requires manual vectorization through primitives like `T.add`.

**Cross-Core Communication:**
Additionally, since cube and vector cores on NPUs can only exchange data through global memory/L2 cache, the current implementation requires explicit specification of execution code for different cores using the `T.Scope` primitive. Synchronization between cores is managed through `T.set_cross_flag` and `T.wait_cross_flag`, and intermediate data transfer global tensors must be explicitly specified during kernel definition.


## Quick Start

In this section, you'll learn how to write and execute a straightforward GEMM (matrix multiplication) kernel using tilelang-ascend, The next chapter will introduce how to write a high-performance gemm kernel.

### GEMM Example with Annotations

Below is an example that demonstrates how to quickly implement a gemm on the ascend.

```python
@tilelang.jit(out_idx=[-1])
def matmul(M, N, K, block_M, block_N, K_L1, dtype="float16", accum_dtype="float"):

    # Calculate number of blocks in M and N dimensions
    m_num = M // block_M
    n_num = N // block_N

    @T.prim_func
    def main(
            A: T.Tensor((M, K), dtype),  # Input matrix A (M×K)
            B: T.Tensor((K, N), dtype),  # Input matrix B (K×N)
            C: T.Tensor((M, N), dtype),  # Output matrix C (M×N)
    ):

        with T.Kernel(m_num * n_num, is_npu=True) as (cid, _):

            bx = cid // n_num  # Block row index
            by = cid % n_num   # Block column index

            # Allocate L1 cache buffers for input matrices
            A_L1 = T.alloc_L1((block_M, K_L1), dtype)      # A block in L1
            B_L1 = T.alloc_L1((K_L1, block_N), dtype)      # B block in L1

            # Allocate L0C buffer for accumulation
            C_L0 = T.alloc_L0C((block_M, block_N), accum_dtype)

            # Core computation scope
            with T.Scope("C"):
                # Calculate number of K-dimension iterations
                loop_k = T.ceildiv(K, K_L1)

                # Iterate over K dimension blocks
                for k in T.serial(loop_k):
                    # Copy A and B blocks from global memory to L1 cache
                    T.copy(A[bx * block_M, k * K_L1], A_L1)
                    T.copy(B[k * K_L1, by * block_N], B_L1)

                    # Synchronize all cores before computation
                    T.barrier_all()

                    # Perform matrix multiplication
                    T.gemm_v0(A_L1, B_L1, C_L0, init=(k == 0))

                    # Synchronize all cores after computation
                    T.barrier_all()

                # Copy final result from L0C to global memory
                T.copy(C_L0, C[bx * block_M, by * block_N])

    return main
```

### High Performance GEMM Example. (Layout, L2 Cache Swizzling, and Pipelining, etc.)

Below is an example that demonstrates more advanced features: layout annotation, parallelized copy, and swizzle for improved L2 cache locality. This snippet shows how to adapt your kernel to maximize performance on complex hardware.

```python
@tilelang.jit(out_idx=[-1])
def matmul(M, N, K, block_M, block_N, block_K, K_L1, S1, S2, dtype="float16", accum_dtype="float"):
    m_num = M // block_M
    n_num = N // block_N

    core_num = 20

    @T.macro
    def init_flag():
        T.set_flag("mte1", "mte2", 0)
        T.set_flag("mte1", "mte2", 1)
        T.set_flag("m", "mte1", 0)
        T.set_flag("m", "mte1", 1)
        T.set_flag("fix", "m", 0)

    @T.macro
    def clear_flag():
        T.wait_flag("mte1", "mte2", 0)
        T.wait_flag("mte1", "mte2", 1)
        T.wait_flag("m", "mte1", 0)
        T.wait_flag("m", "mte1", 1)
        T.wait_flag("fix", "m", 0)

    @T.prim_func
    def main(
            A: T.Tensor((M, K), dtype),
            B: T.Tensor((K, N), dtype),
            C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(core_num, is_npu=True) as (cid, _):
            A_L1 = T.alloc_L1((S1, block_M, K_L1), dtype)
            B_L1 = T.alloc_L1((S1, K_L1, block_N), dtype)

            T.annotate_layout({
                A_L1: make_zn_layout(A_L1),
                B_L1: make_zn_layout(B_L1),
            })

            A_L0 = T.alloc_L0A((S2, block_M, block_K), dtype)
            B_L0 = T.alloc_L0B((S2, block_K, block_N), dtype)
            C_L0 = T.alloc_L0C((block_M, block_N), accum_dtype)

            with T.Scope("C"):
                init_flag()

                for i in T.serial(T.ceildiv(m_num * n_num, core_num)):
                    T.use_swizzle(
                        i * core_num + cid, M, N, K, block_M, block_N, off=3, in_loop=True)
                    bx = cid // n_num
                    by = cid % n_num

                    loop_k = T.ceildiv(K, K_L1)

                    T.wait_flag("mte1", "mte2", 0)
                    T.copy(A[bx * block_M, 0], A_L1[0, :, :])
                    T.copy(B[0, by * block_N], B_L1[0, :, :])
                    T.set_flag("mte2", "mte1", 0)
                    T.wait_flag("fix", "m", 0)
                    for k in T.serial(loop_k):
                        if k < loop_k - 1:
                            T.wait_flag("mte1", "mte2", (k + 1) % S1)
                            T.copy(A[bx * block_M, (k + 1) * K_L1], A_L1[(k + 1) % S1, :, :])
                            T.copy(B[(k + 1) * K_L1, by * block_N], B_L1[(k + 1) % S1, :, :])
                            T.set_flag("mte2", "mte1", (k + 1) % S1)

                        loop_kk = T.ceildiv(K_L1, block_K)

                        for kk in T.serial(loop_kk):
                            if kk == 0:
                                T.wait_flag("mte2", "mte1", k % S1)
                            T.wait_flag("m", "mte1", kk % S2)
                            T.copy(A_L1[k % S1, 0, kk * block_K], A_L0[kk % S2, :, :])
                            T.copy(B_L1[k % S1, kk * block_K, 0], B_L0[kk % S2, :, :])
                            if kk == 3:
                                T.set_flag("mte1", "mte2", k % S1)
                            T.set_flag("mte1", "m", kk % S2)
                            T.wait_flag("mte1", "m", kk % S2)

                            if k == 0 and kk == 0:
                                T.mma(A_L0[kk % S2, :, :], B_L0[kk % S2, :, :], C_L0, init=True)
                            else:
                                T.mma(A_L0[kk % S2, :, :], B_L0[kk % S2, :, :], C_L0)

                            T.set_flag("m", "mte1", kk % S2)

                    T.set_flag("m", "fix", 0)
                    T.wait_flag("m", "fix", 0)
                    T.copy(C_L0, C[bx * block_M, by * block_N])
                    T.set_flag("fix", "m", 0)

                clear_flag()

                T.barrier_all()

    return main
```

### Automatic insertion of synchronization instruction

We have supported automatic insertion of synchronization instructions within the core, which can be enabled by setting the TL_ASCEND_AUTO_SYNC attribute in the JIT's pass_configs parameter. Here is a simple example:

```python
pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True
}
@tilelang.jit(out_idx=[-1], pass_configs=pass_configs)
def vec_add(M, N, block_M, block_N, dtype="float"):
    m_num = M // block_M
    n_num = N // block_N

    VEC_NUM = 2

    @T.prim_func
    def main(
            A: T.Tensor((M, N), dtype),
            B: T.Tensor((M, N), dtype),
            C: T.Tensor((M, N), dtype),
    ):
        # Enable by setting the enable_auto_sync attribute
        T.func_attr({"enable_auto_sync": True})
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num

            a_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
            b_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
            c_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
            with T.Scope("V"):
                T.copy(A[bx * block_M + vid * block_M // VEC_NUM, by * block_N], a_ub)
                T.copy(B[bx * block_M + vid * block_M // VEC_NUM, by * block_N], b_ub)

                T.add(c_ub, a_ub, b_ub)

                T.copy(c_ub, C[bx * block_M + vid * block_M // VEC_NUM, by * block_N])

    return main
```

### Automatic Buffer Reuse

We have supported automatic buffer reuse, which can be enabled by setting the TL_ASCEND_MEMORY_PLANNING attribute in the JIT's pass_configs parameter. Here is a example based on example_sparse_flash_attn.py:

```python
pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True
}
@tilelang.jit(out_idx=[3], pass_configs=pass_configs)
def sparse_attention_fwd(
    # other code...
    
    #Manual configuration of T.annotate_address is no longer needed.

    # T.annotate_address({
    #             # L1 address
    #             q_l1: 0,
    #             q_tail_l1: 65536,
    #             kv_l1: 73728,
    #             kv_tail_l1: 139264,
    #             acc_s_l1: 139264,

    #             # L0C address
    #             acc_s_l0c: 0,
    #             acc_o_l0c: 0,

    #             ## ub address
    #             acc_o: 0,
    #             sumexp: 65536,
    #             m_i: 65664,
    #             indices_ub_: 65792,
    #             kv_ub: 66048,
    #             kv_tail_ub: 67072,
    #             acc_s_ub: 66048,
    #             m_i_prev: 74240,
    #             acc_s_ub_: 74368,
    #             tmp_ub: 74368,
    #             sumexp_i_ub: 98944,
    #             acc_s_half: 98944,
    #             acc_o_ub: 98944,
    #             acc_o_half: 98944
    #         })
    
    # other code...
)
```
### T.Parallel
We have supported [T.parallel](https://github.com/tile-ai/tilelang-ascend/blob/ascendc_pto/docs/tutorials/t_parallel.md), which transforms the parallel iteration space into vectorized operations that are lowered into AscendC vector instructions. Here is an example based on [example_sparse_flash_attn.py](https://github.com/tile-ai/tilelang-ascend/blob/ascendc_pto/examples/sparse_flash_attention/example_sparse_flash_attn.py):

```python
pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True
}
@tilelang.jit(out_idx=[3], pass_configs=pass_configs)
def sparse_attention_fwd(
    ...
    # T.tile.add(acc_s_ub, acc_s_ub, acc_s_ub_)
    for (i, j) in T.Parallel(v_block, BI):
        acc_s_ub[i, j] = acc_s_ub[i, j] + acc_s_ub_[i, j]
    ...

    # T.tile.mul(acc_s_ub, acc_s_ub, sm_scale)
    for (i, j) in T.Parallel(v_block, BI):
        acc_s_ub[i, j] = acc_s_ub[i, j] * sm_scale
    ...

    # T.tile.max(m_i, m_i, m_i_prev)
    for i in T.Parallel(v_block):
        m_i[i] = T.max(m_i[i], m_i_prev[i])
    ...

    # for h_i in range(v_block):
        # T.tile.sub(acc_s_ub[h_i, :], acc_s_ub[h_i, :], m_i[h_i])
    for (h_i, j) in T.Parallel(v_block, D):
        acc_s_ub[h_i, j] = acc_s_ub[h_i, j] - m_i[h_i]
)
```

### Auto-allocated Workspace
We now support [automatic workspace allocation](./docs/tutorials/automatic_workspace_allocation.md), enabling users to call operators without managing workspace or output tensor allocation—they only need to handle input tensors. Refer to [example_sparse_flash_attn.py](https://github.com/tile-ai/tilelang-ascend/blob/ascendc_pto/examples/sparse_flash_attention/example_sparse_flash_attn.py) for a concrete example.
```python
# Specify workspace positions in parameter list via workspace_idx
@tilelang.jit(out_idx=[3], workspace_idx=[4,5,6,7,8])
def sparse_attention_fwd(...):
    @T.prim_func
    def main(
            # --- Input tensors ---
            Q: T.Tensor(q_shape, dtype),  
            KV: T.Tensor(kv_shape, dtype),  
            Indices: T.Tensor(indices_shape, indices_dtype), 

            # --- Auto-allocated output (index 3 in out_idx) --- 
            Output: T.Tensor(o_shape, dtype),  

            # --- Auto-allocated workspaces (indices 4-8 in workspace_idx) ---
            # These are temporary buffers managed by the runtime
            workspace_1: T.Tensor([block_num, BI, D], dtype),
            workspace_2: T.Tensor([block_num, BI, D_tail], dtype),
            workspace_3: T.Tensor([block_num, H_per_block, BI], accum_dtype),
            workspace_4: T.Tensor([block_num, H_per_block, BI], dtype),
            workspace_5: T.Tensor([block_num, H_per_block, D], accum_dtype),
    ):

    ...

# Instantiate sparse attention function
func = sparse_attention_fwd(
    heads=128,
    dim=512,
    tail_dim=64,
    topk=2048,
    kv_stride=1,
)

# Prepare input tensors
q = torch.randn((B, S, H, DQK), dtype=dtype)
kv = torch.randn((B, SKV, HKV, DQK), dtype=dtype)
indices = torch.full((B, S, HKV, topk), SKV, dtype=torch.int32)
for b in range(B):
    for t in range(S):
        for h in range(HKV):
            i_i = torch.randperm(max(1, ((t + q_start_s_index) // KV_stride)))[:topk]
            indices[b, t, h, :len(i_i)] = i_i

# Call operator - output and workspaces are automatically allocated!
output = func(q, kv, indices)
```
### Dive Deep into TileLang Beyond GEMM

In addition to GEMM, we provide a variety of examples to showcase the versatility and power of TileLang-Ascend, including:

- [FlashAttention](./examples/flash_attention/): Implementations of FlashAttention with TileLang-Ascend.
- [LightningIndexer](./examples/lightning_indexer/): Implementations of LightningIndexer with TileLang-Ascend.
- [SparseFlashAttention](./examples/sparse_flash_attention/): Implementations of SparseFlashAttention with TileLang-Ascend.

### Automatic insert synchronization flags between AIC and AIV, such as CrossCoreSetFlag / CrossCoreWaitFlag.

Two switches need to be turned on:
```python
pass_configs = {
    "tl.ascend_auto_cv_combine": True,
    "tl.ascend_auto_cross_core_sync": True,
}
```

Here is an example:
- [FlashAttention](./examples/flash_attention/flash_attn_bhsd_cc_sync.py): Implementations of FlashAttention without inserting synchronization flags manually.

**Attention**

Currently, accessing to the same workspace address in a for loop is not supported.

## Upcoming Features

Check our [tilelang-ascend development plan](https://github.com/tile-ai/tilelang-ascend/issues/3) for upcoming features.


## Acknowledgements
We gratefully acknowledge the valuable support provided by Huawei's HiSilicon, ICT, Compiler and Programming Language Lab and the Peking University Kunpeng & Ascend Center for Excellence in Science, Education, and Innovation.
