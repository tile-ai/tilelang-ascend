# Tilelang.language.alloc_shared

## 1. OP概述

简介：`tilelang.language.alloc_shared` 申请一块shared memory的内存，Ascend中对应ub/L1上的内存

```
T.alloc_shared(shape, dtype) [Developer mode]
T.alloc_ub(shape, dtype) [Expert mode]
T.alloc_L1(shape, dtype) [Expert mode]
```

## 2. OP规格

### 2.1 参数说明

| 参数名  | 类型  | 说明  |
| ------------ | ------------ | ------------ |
| `shape` | `shape`| 用于指定申请shared memroy (UB/L1)的维度形状，整数元组|
| `dtype` | `str`| 数据类型，例如`float32`,`float16`|

### 2.2 支持规格

#### 2.2.1 DataType支持

|   | uint8 | int8 | uint16 | int16 | uint32 | int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool/int1 |
| ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ |
| Ascend | ×  | × |  × |  × | ×  | ×  | ×  | ×  | √  | √ |  ×  | ×  |

#### 2.2.2 Shape支持

|   | 支持维度范围|
| ------------ | ------------ |
| Ascend A2/A3 |  1D~2D|

注意：3D~5D有待充分验证

### 2.3 特殊限制说明

无

### 2.4 使用方法

示例1：以下示例实现了一个形状为(M,K)的tensor和一个形状为(K,N)的tensor矩阵乘，其中`T.alloc_shared`申请了Ascend L1内存

```
def matmul(M, N, K, block_M, block_N, block_K, dtype="float16", accum_dtype="float32"):
    @T.prim_func
    def gemm(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N) * T.ceildiv(M, block_M), is_npu=True) as (cid, _):
            by = cid // T.ceildiv(N, block_N)
            bx = cid % T.ceildiv(N, block_N)

            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=2):
                T.copy(A[by * block_M, k * block_K], A_shared)
                T.copy(B[k * block_K, bx * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_local, initC=(k == 0))

            T.copy(C_local, C[by * block_M, bx * block_N])

    return gemm

```

示例2：以下示例实现了一个形状为(M,N)的tensor和一个形状为(M,N)的tensor向量减，其中`T.alloc_shared`申请了Ascend UB内存

```
def vecsub(M, N, block_M, block_N, dtype="float16"):
    @T.prim_func
    def vecsub_(
        A: T.Tensor((M, N), dtype),
        B: T.Tensor((M, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N) * T.ceildiv(M, block_M), is_npu=True) as (cid, _):
            by = cid // T.ceildiv(N, block_N)
            bx = cid % T.ceildiv(N, block_N)

            A_shared = T.alloc_shared((block_M, block_N), dtype)
            B_shared = T.alloc_shared((block_M, block_N), dtype)
            C_shared = T.alloc_shared((block_M, block_N), dtype)

            T.copy(A[by * block_M, bx * block_N], A_shared)
            T.copy(B[by * block_M, bx * block_N], B_shared)
            T.vsub(A_shared, B_shared, C_shared)
            T.copy(C_shared, C[by * block_M, bx * block_N])

    return vecsub_

```

## 3. Tilelang Op到Ascend NPU IR Op的转换

**T.alloc_shared**将被下降为**tensor.empty**