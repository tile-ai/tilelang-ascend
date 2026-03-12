# Tilelang.language.reduce

## 1. OP概述

简介：`tilelang.language.reduce` 对张量在指定维度上进行聚合运算（如 `sum`、`max`、`min` 等），将多个元素压缩为更少元素或标量的操作

```markup
T.reduce(src, dst, dims=1, reduce_mode="sum", clear=True, size=[m,n])
T.reduce_max(src, dst, dim=0, clear = False)
T.reduce_min(src, dst, dim=0, clear = False)
T.reduce_sum(src, dst, dim=0, clear = False)
```

## 2. OP规格

### 2.1 参数说明


| 参数名        | 类型         | 说明                                                    |
| ------------- | ------------ | ------------------------------------------------------- |
| `src`         | `tensor`     | 输入tensor                                              |
| `dst`         | `tensor`     | 输出tensor                                              |
| `dims`        | `list/tuple` | 需要reduce的维度                                        |
| `reduce_mode` | `str`        | reduce操作类型(`sum`、`max`、`min`、`abssum`、`absmax`) |
| `clear`       | `bool`       | 是否在reduce前对目标张量进行初始化                      |
| `size`        | `list`       | 控制 reduce 实际参与计算的数据范围                      |

### 2.2 支持规格

#### 2.2.1 DataType支持


|        | uint8 | int8 | uint16 | int16 | uint32 | int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool/int1 |
| ------ | ----- | ---- | ------ | ----- | ------ | ----- | ------ | ----- | ---- | ---- | ---- | --------- |
| Ascend | ×    | ×   | ×     | ×    | ×     | ×    | ×     | ×    | √   | √   | ×   | ×        |

#### 2.2.2 Shape支持

1. Rank 一致性：src与dst 的张量阶数（rank）必须相同；
2. Reduce 兼容性：src与dst有且仅有一个维度的 shape 不同，且在该维度上，dst的尺寸必须为 1
   示例：
   ✅ 合法：
   src: (M, N, K) → dst: (M, 1, K)
   src: (M, N, K) → dst: (1, N, K)
   ❌ 非法：
   src: (M, N, K) → dst: (M, N, K)（无维度为 1，无法reduce）
   src: (M, N, K) → dst: (M, 1, L)（两个维度不同，违反“仅一个维度不同”）

### 2.3 使用方法

以下示例实现了对输入矩阵沿第 1 维执行  sum 归约，最终得到形状为 (M,1) 的结果。

```markup
import torch
import torch_npu
import tilelang
import tilelang.language as T

def reduce(M, N, dtype = "float16"):
    BLOCK_SIZE = 1

    @T.prim_func
    def main(A: T.Tensor((M, N), dtype),
                   B: T.Tensor((M, 1), dtype)):
  
        with T.Kernel(BLOCK_SIZE, is_npu=True) as (cid, _):
            a = T.alloc_shared((M,N), dtype)
            b = T.alloc_shared((M,1), dtype)
            T.copy(A, a)
  
            T.reduce(a, b, dims=[1], reduce_mode="sum", clear = False)

            T.copy(s, O)

    return main
```

## 3. Tilelang Op到Ascend NPU IR Op的转换

**tilelang::reduceOp**将被下降为hivm::VReduceOp