# Tilelang.language.min

## 1. OP概述

简介：`tilelang.language.min` 返回给定标量A和B的最小值。

```
T.min(A, B)
```

## 2. OP规格

### 2.1 参数说明

| 参数名  | 类型  | 说明  |
| ------------ | ------------ | ------------ |
| `A` | `scalar` | 输入scalar|
| `B` | `scalar` | 输入scalar|

### 2.2 支持规格

#### 2.2.1 DataType支持

|   | uint8 | int8 | uint16 | int16 | uint32 | int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool/int1 |
| ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ |
| Ascend | ×  | √ |  × |  √ | ×  | √  | ×  | √  | √  | √ |  ×  | ×  |

#### 2.2.2 Shape支持

标量运算，不支持shape

### 2.3 特殊限制说明

无

### 2.4 使用方法

以下示例实现了两个形状为(M,)的tensor的逐元素min计算

```
def Tmin(M, dtype="float16"):
    BLOCK_SIZE = 1

    @T.prim_func
    def main(A: T.Tensor((M, ), dtype),
             B: T.Tensor((M, ), dtype),
             C: T.Tensor((M, ), dtype),):

        with T.Kernel(BLOCK_SIZE, is_npu=True) as (cid, _):
            for i in range(M):
                C[i] = T.min(A[i], B[i])

    return main
```

## Tilelang Op到Ascend NPU IR Op的转换

**tilelang.language.min**将被下降为arith::MinimumFOp