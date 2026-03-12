# Tilelang.language.reshape

## 1. OP概述

简介：`tilelang.language.reshape` 在不改变数据内容与存储顺序的前提下，仅调整张量的形状视图

```markup
T.reshape(src,dst)
```

## 2. OP规格

### 2.1 参数说明


| 参数名 | 类型     | 说明                                    |
| ------ | -------- | --------------------------------------- |
| `src`  | `tensor` | 输入tensor                              |
| `dst`  | `tensor` | 输出tensor(dst的shape即为reshape的形状) |

### 2.2 支持规格

#### 2.2.1 DataType支持


|        | uint8 | int8 | uint16 | int16 | uint32 | int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool |
| ------ | ----- | ---- | ------ | ----- | ------ | ----- | ------ | ----- | ---- | ---- | ---- | ---- |
| Ascend | ×    | √   | ×     | √    | ×     | √    | ×     | √    | √   | √   | √   | ×   |

### 2.3 使用方法

以下示例实现了对[M,N]矩阵做reshape得到[N,M]矩阵

```markup
import torch
import torch_npu
import tilelang
import tilelang.language as T

def reshape_dev(M, N, dtype="float16"):
    BLOCK_SIZE = 1

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),
        B: T.Tensor((N, M), dtype)
    ):
        with T.Kernel(BLOCK_SIZE, is_npu=True) as (cid, _):
            a = T.alloc_shared((M, N), dtype)
            b = T.alloc_shared((N, M), dtype)
            T.copy(A,a)
            T.reshape(a,b)
            T.copy(b,B)
    return main
```

## 3. Tilelang Op到Ascend NPU IR Op的转换

**[DEV]: tilelang::gatherOp**将被下降为mlir::tensor::CollapseShapeOp和mlir::tensor::ExpandShapeOp

**[EXPERT]: tilelang::gatherOp** 将被下降为**mlir::memref::ReinterpretCastOp**