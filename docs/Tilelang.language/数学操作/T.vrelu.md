# Tilelang.language.vrelu

## 1. OP概述

简介：`tilelang.language.vsrelu`用于计算张量的逐元素ReLU值

```
T.vrelu(src, dst) [Developer mode]
T.vrelu(src, dst) [Expert mode]
```

## 2. OP规格

### 2.1 参数说明

| 参数名  | 类型  | 说明  |
| ------------ | ------------ | ------------ |
| `src` | `tensor`| 源张量 |
| `dst` | `tensor`| 目的张量|

### 2.2 支持规格

#### 2.2.1 DataType支持

|   | uint8 | int8 | uint16 | int16 | uint32 | int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool/int1 |
| ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ |
| Ascend | ×  | × |  × |  × | ×  | ×  | ×  | ×  | √  | √ |  ×  | ×  |

#### 2.2.2 Shape支持

支持1D~5D的`tensor`

### 2.3 特殊限制说明

### 2.4 使用方法

以下示例实现了计算张量`src`中每个元素的ReLU值并输出到张量`dst`中

```
import torch
import torch_npu
import tilelang
import tilelang.language as T

def vecrelu(M, N, block_M, block_N, dtype="float16"):
    @T.prim_func
    def vecrelu_(
        A: T.Tensor((M, N), dtype),
        B: T.Tensor((M, N), dtype)
    ):
        with T.Kernel(T.ceildiv(N, block_N) * T.ceildiv(M, block_M), is_npu=True) as (cid, _):
            by = cid // T.ceildiv(N, block_N)
            bx = cid % T.ceildiv(N, block_N)

            A_BUF = T.alloc_shared((block_M, block_N), dtype)
            B_BUF = T.alloc_shared((block_M, block_N), dtype)

            T.copy(A[by * block_M, bx * block_N], A_BUF)
            T.vrelu(A_BUF, B_BUF)
            T.copy(B_BUF, B[by * block_M, bx * block_N])

    return vecrelu_

```

## 3. Tilelang Op到Ascend NPU IR Op的转换

**T.vrelu**将被下降为**hivm.hir.vrelu**