# Tilelang.language.alloc_ub

## 1. OP概述

简介：`tilelang.language.alloc_ub` 申请一块Ascend ub上的内存

```
T.alloc_shared(shape, dtype) [Developer mode]
T.alloc_ub(shape, dtype) [Expert mode]
```

## 2. OP规格

### 2.1 参数说明

| 参数名  | 类型  | 说明  |
| ------------ | ------------ | ------------ |
| `shape` | `shape`| 用于指定申请UB的维度形状，整数元组|
| `dtype` | `str`| 数据类型，例如`float32`,`float16`|

### 2.2 支持规格

#### 2.2.1 DataType支持

|   | uint8 | int8 | uint16 | int16 | uint32 | int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool/int1 |
| ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ |
| Ascend | ×  | × |  × |  × | ×  | ×  | ×  | ×  | √  | √ |  ×  | ×  |

#### 2.2.2 Shape支持

结论：目前支持1D ~ 2D, 3D ~ 5D有待充分验证

### 2.3 特殊限制说明

无

### 2.4 使用方法

已下示例实现了一个形状为(M,N)的tensor和一个形状为(M,N)的tensor向量减，其中`T.alloc_ub`申请了Ascend UB内存

```
import torch
import torch_npu
import tilelang
import tilelang.language as T

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

            A_BUF = T.alloc_ub((block_M, block_N), dtype)
            B_BUF = T.alloc_ub((block_M, block_N), dtype)
            C_BUF = T.alloc_ub((block_M, block_N), dtype)

            T.copy(A[by * block_M, bx * block_N], A_BUF)
            T.copy(B[by * block_M, bx * block_N], B_BUF)
            T.vsub(A_BUF, B_BUF, C_BUF)
            T.copy(C_BUF, C[by * block_M, bx * block_N])

    return vecsub_

```

## 3. Tilelang Op到Ascend NPU IR Op的转换

**T.alloc_ub**将被下降为**memref::AllocOp**