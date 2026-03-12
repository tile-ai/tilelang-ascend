# Tilelang.language.reduce_max

## 1. OP概述

简介：`tilelang.language.reduce_max`对输入Tensor的指定维度进行最大值归约。

```
T.reduce_max(buffer, out, dim=-1, size=[], clear=True)
```

## 2. OP规格

### 2.1 参数说明

| 参数名 | 类型 | 说明 |
| - | - | - |
| `buffer` | `tensor` | 输入Tensor |
| `out`  | `tensor` | 输出Tensor |
| `dim`  | `int` | 可选，需要进行归约的维度索引 |
| `size`  | `list` | 可选，手动指定Shape |
| `clear`  | `bool` | 可选，是否在归约前清空输出 |

### 2.2 支持规格

#### 2.2.1 DataType支持

|   | uint8 | int8 | uint16 | int16 | uint32 | int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool/int1 |
| - | - | - | - | - | - | - | - | - | - | - | - | - |
| Ascend | × | × | × | × | × | × | × | × | √ | √ | × | × |

#### 2.2.2 Shape支持

在shape方面无特殊要求

### 2.3 特殊限制说明

无

### 2.4 使用方法

以下示例实现了一个reduce_max计算

```python
dtype = "float16"
accum_dtype =  "float16"

def reduce_max_kernel(M, N, block_M):
    BLOCK_SIZE = 1

    @T.prim_func
    def main(
        B:T.Tensor((M, N), dtype),
        O:T.Tensor((M, 1), accum_dtype),
    ):
        with T.Kernel(BLOCK_SIZE, is_npu=True) as (cid, _):

            b = T.alloc_shared((M, N), dtype)
            s = T.alloc_shared((M, 1), accum_dtype)

            T.copy(B, b)
            T.reduce_max(b, s, dim=1, clear=False)
            T.copy(s, O)

    return main
```

## 3. Tilelang Op到Ascend NPU IR Op的转换

**tilelang::reduce_maxOp**将被转换为hivm::ReduceOperation::max