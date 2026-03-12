# Tilelang.language.brc

## 1. OP概述

简介：`tilelang.language.brc` 返回输入向量/标量基于输出形状的广播broadcast计算结果

```
T.vbrc(src, dst)
```

## 2. OP规格

### 2.1 参数说明

| 参数名  | 类型  | 说明  |
| ------------ | ------------ | ------------ |
| `src` | `tensor` ,`scalar`| 输入tensor  |
| `dst` | `tensor` | 输出tensor  |

### 2.2 支持规格

#### 2.2.1 DataType支持

|   | uint8 | int8 | uint16 | int16 | uint32 | int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool/int1 |
| ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ |
| Ascend | ×  | × |  × |  × | ×  | ×  | ×  | ×  | √  | √ |  ×  | ×  |

#### 2.2.2 Shape支持

结论：

1. 当 src 为标量（scalar）时：
   对输入（input）与输出（output）的 shape 无特殊限制。
2. 当 src 为向量（或张量）时：
   需同时满足以下两个条件：
   Rank 一致性：input 与 output 的张量阶数（rank）必须相同；
   Broadcast 兼容性：二者有且仅有一个维度的 shape 不同，且在该维度上，src 的尺寸必须为 1。
   示例：
   ✅ 合法：
   src: (M, 1, K) → dst: (M, N, K)
   src: (1, N, K) → dst: (M, N, K)
   ❌ 非法：
   src: (M, N, K) → dst: (M, N, K)（无维度为 1，无法广播）
   src: (L, 1, K) → dst: (M, N, K)（两个维度不同，违反“仅一个维度不同”）

### 2.3 特殊限制说明

无

### 2.4 使用方法

示例1：实现了将value = 3 广播到一个形状为(M, K)的tensor

```
def vec_brc(M, N, dtype):
    dtype = "float16"
    BLOCK_SIZE = 1

    @T.prim_func
    def main(
            A: T.Tensor((M, N), dtype)
    ):
        with T.Kernel(BLOCK_SIZE, is_npu=True) as (cid, _):
            A_ub = T.alloc_ub((M, N), dtype)
            T.copy(A, A_ub)
            brc_value = 3
            T.vbrc(brc_value, A_ub)
            T.copy(A_ub, A)

    return main
```

示例2：实现了将vector(1, N) 广播到形状(M, N)

```
def vec_brc(M, N, dtype):
    dtype = "float16"
    BLOCK_SIZE = 1

    @T.prim_func
    def main(
            A: T.Tensor((1, N), dtype),
            B: T.Tensor((M, N), dtype)
    ):
        with T.Kernel(BLOCK_SIZE, is_npu=True) as (cid, _):
            A_ub = T.alloc_ub((1, N), dtype)
            B_ub = T.alloc_ub((M, N), dtype)
           
             T.copy(A, A_ub)
             T.vbrc(A_ub, B_ub)
             T.copy(B_ub, B)

    return main

```

## 3. Tilelang Op到Ascend NPU IR Op的转换

**tilelang::vbrcOp**将被下降为hivm::VBrcOp