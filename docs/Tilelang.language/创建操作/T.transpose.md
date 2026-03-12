# Tilelang.language.transpose

## 1. OP概述

简介：`tilelang.language.transpose`根据给定的维度排列对输入Tensor的维度进行转置。
```
T.transpose(src, dst, permutation, size=[])
```

## 2. OP规格

### 2.1 参数说明

| 参数名 | 类型 | 说明 |
| - | - | - |
| `src` | `tensor` | 输入Tensor |
| `dst`  | `tensor` | 输出Tensor |
| `permutation`  | `list` | 维度排列序列 |
| `size`  | `list` | 可选参数，手动指定shape |

### 2.2 支持规格

#### 2.2.1 DataType支持

|   | uint8 | int8 | uint16 | int16 | uint32 | int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool/int1 |
| - | - | - | - | - | - | - | - | - | - | - | - | - |
| Ascend | × | × | × | × | × | × | × | × | √ | √ | × | × |

#### 2.2.2 Shape支持

输入`src`和输出`dst`的秩相同

### 2.3 特殊限制说明

无

### 2.4 使用方法

以下示例实现了一个transpose计算

```python
def transpose_kernel(M, N, dtype):
    BLOCK_SIZE = 1

    @T.prim_func
    def main(
        src:T.Tensor((M, N), dtype),
        dst:T.Tensor((N, M), dtype),
    ):
        with T.Kernel(BLOCK_SIZE, is_npu=True) as (cid, _):

            src_ub = T.alloc_shared((M, N), dtype)
            dst_ub = T.alloc_shared((N, M), dtype)

            T.copy(src, src_ub)
            T.transpose(src_ub, dst_ub, permutation=[1, 0])
            T.copy(dst_ub, dst)

    return main
```

## 3. Tilelang Op到Ascend NPU IR Op的转换

**tilelang::transposeOp**将被编译为hivm::VTransposeOp