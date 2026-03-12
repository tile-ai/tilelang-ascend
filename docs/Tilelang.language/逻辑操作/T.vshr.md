# Tilelang.language.vshr

## 1. OP概述

简介：`tilelang.language.vshr` 对输入张量`A`执行按位右移（bitwise right shift），移位位数由另一个输入`B`指定，结果写入输出`C`

```
T.vshr(A, B, C)
```

## 2. OP规格

### 2.1 参数说明

| 参数名  | 类型  | 说明  |
| ------------ | ------------ | ------------ |
| `A` | `tensor` | 输入tensor  |
| `B` | `tensor`, `scalar` | 输入tensor  |
| `C` | `tensor` | 输出tensor  |

### 2.2 支持规格

#### 2.2.1 DataType支持

|   | uint8 | int8 | uint16 | int16 | uint32 | int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool/int1 |
| ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ |
| Ascend | ×  | × |  × |  √ | ×  | √  | ×  | √  | ×  | × |  ×  | ×  |

#### 2.2.2 Shape支持

结论：

1. `C`: shape必须和`A`一致
2. `B`: shape需要和`A`一致或为标量

### 2.3 特殊限制说明

无

### 2.4 使用方法

示例1：B和A有相同的shape

```
import torch
import torch_npu
import tilelang
import tilelang.language as T

def vshr_kernel(M, N, dtype):
    BLOCK_SIZE=1

    @T.prim_func
    def main(
        A: T.Tensor((N,), dtype),
        B: T.Tensor((N,), dtype),
        Out: T.Tensor((N,), dtype),
    ):
        with T.Kernel(BLOCK_SIZE, is_npu=True) as (cid, _):

            acc_A  = T.alloc_shared((N,), dtype)
            acc_B  = T.alloc_shared((N,), dtype)
            out_ub = T.alloc_shared((N,), dtype)

            T.copy(A, acc_A)
            T.copy(B, acc_B)

            T.shr(acc_A, acc_B, out_ub)
            T.copy(out_ub, Out)

    return main
```

示例2：B是标量

```
def vshr_kernel(M, N):
    BLOCK_SIZE=1

    @T.prim_func
    def main(
        A: T.Tensor((N,), dtype),
        B: T.Tensor((1,), dtype),
        Out: T.Tensor((N,), dtype),
    ):
        with T.Kernel(BLOCK_SIZE, is_npu=True) as (cid, _):
            acc_A  = T.alloc_shared((N,), dtype)
            acc_B  = T.alloc_shared((1,), dtype)
            out_ub = T.alloc_shared((N,), dtype)

            T.copy(A, acc_A)
            T.copy(B, acc_B)
            T.shr(acc_A, acc_B, out_ub)
            T.copy(out_ub, Out)

    return main

```

## 3. Tilelang Op到Ascend NPU IR Op的转换

**tilelang::vshrOp**将被下降为hivm::VShROp