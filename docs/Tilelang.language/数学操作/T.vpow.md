# Tilelang.language.vpow

## 1. OP概述

简介：`tilelang.language.vpow`用于对输入张量`A`按元素幂运算，指数由另一输入`B`指定，结果写入输出`C`

```
T.vpow(A, B, C) [Developer mode]
T.vpow(A, B, C) [Expert mode]
```

## 2. OP规格

### 2.1 参数说明

| 参数名  | 类型  | 说明  |
| ------------ | ------------ | ------------ |
| `A` | `tensor`| 底数，输入`tensor` |
| `B` | `tensor`| 指数，输入`tensor`|
| `C` | `tensor`| 计算结果，输出`tensor`|

### 2.2 支持规格

#### 2.2.1 DataType支持

|   | uint8 | int8 | uint16 | int16 | uint32 | int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool/int1 |
| ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ |
| Ascend | ×  | × |  × |  × | ×  | √   | ×  | ×  | ×  | × |  ×  | ×  |

#### 2.2.2 Shape支持

支持1D~5D的`tensor`

### 2.3 特殊限制说明

### 2.4 使用方法

以下示例实现了逐元素以`A`为底数，`B`为指数的幂运算

```
def vecpow(M, N, block_M, block_N, dtype="int32"):
    @T.prim_func
    def vecpow_(
        A: T.Tensor((M, N), dtype),
        B: T.Tensor((M, N), dtype),
        C: T.Tensor((M, N), dtype)
    ):
        with T.Kernel(T.ceildiv(N, block_N) * T.ceildiv(M, block_M), is_npu=True) as (cid, _):
            by = cid // T.ceildiv(N, block_N)
            bx = cid % T.ceildiv(N, block_N)

            A_BUF = T.alloc_shared((block_M, block_N), dtype)
            B_BUF = T.alloc_shared((block_M, block_N), dtype)
            C_BUF = T.alloc_shared((block_M, block_N), dtype)

            T.copy(A[by * block_M, bx * block_N], A_BUF)
            T.copy(B[by * block_M, bx * block_N], B_BUF)
            T.vpow(A_BUF, B_BUF, C_BUF)
            T.copy(C_BUF, C[by * block_M, bx * block_N])

    return vecpow_

```

## 3. Tilelang Op到Ascend NPU IR Op的转换

**T.vpow**将被下降为**hivm.hir.vpow**