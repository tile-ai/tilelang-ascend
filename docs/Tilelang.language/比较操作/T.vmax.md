# Tilelang.language.vmax

## 1. OP概述

简介：`tilelang.language.vmax`按向量元素取最大值操作，该算子对比两个输入源的对应元素，并将较大值写入输出目标。

```
T.vmax(src0, src1, dst)
```

## 2. OP规格

### 2.1 参数说明

| 参数名 | 类型 | 说明 |
| - | - | - |
| `src0` | `tensor` | 输入Tensor0 |
| `src1`  | `tensor` | 输入Tensor1 |
| `dst`  | `tensor` | 输出Tensor |

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

以下示例实现了一个vmax计算

```python
def vmax_kernel(M, N, dtype):
    BLOCK_SIZE = 1

    @T.prim_func
    def main(
        src0:T.Tensor((M, N), dtype),
        src1:T.Tensor((M, N), dtype),
        dst:T.Tensor((M, N), dtype),
    ):
        with T.Kernel(BLOCK_SIZE, is_npu=True) as (cid, _):

            src0_ub = T.alloc_shared((M, N), dtype)
            src1_ub = T.alloc_shared((M, N), dtype)
            dst_ub = T.alloc_shared((M, N), dtype)

            T.copy(src0, src0_ub)
            T.copy(src1, src1_ub)

            T.vmax(src0_ub, src1_ub, dst_ub)
            T.copy(dst_ub, dst)

    return main
```

## 3. Tilelang Op到Ascend NPU IR Op的转换

**tilelang::vmaxOp**将被编译为hivm::VMaxOp