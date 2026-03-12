# Tilelang.language.vnot

## 1. OP概述

简介：`tilelang.language.vnot` 对输入向量中的元素**逐元素按位取反**

```markup
T.vnot(src, dst)
```

## 2. OP规格

### 2.1 参数说明


| 参数名 | 类型     | 说明       |
| ------ | -------- | ---------- |
| `src`  | `tensor` | 输入tensor |
| `dst`  | `tensor` | 输出tensor |

### 2.2 支持规格

#### 2.2.1 DataType支持


|        | uint8 | int8 | uint16 | int16 | uint32 | int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool/int1 |
| ------ | ----- | ---- | ------ | ----- | ------ | ----- | ------ | ----- | ---- | ---- | ---- | --------- |
| Ascend | √    | √   | √     | √    | √     | √    | √     | √    | √   | √   | √   | √        |

### 2.3 使用方法

以下示例实现了对输入矩阵做reshape得到[N,M]矩阵

```markup
import torch
import torch_npu
import tilelang
import tilelang.language as T

def vnot_dev(M, N, dtype = "float16"):
    BLOCK_SIZE = 1

    @T.prim_func
    def main(
            A: T.Tensor((M, N), dtype),
            B: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(BLOCK_SIZE, is_npu=True) as (cid, _):

            A_VEC = T.alloc_shared((M, N), dtype)
            B_VEC = T.alloc_shared((M, N), dtype)
            T.copy(A, A_VEC)
            T.vnot(A_VEC, B_VEC)
            T.copy(B_VEC, B)

    return main
```

## 3. Tilelang Op到Ascend NPU IR Op的转换

**tilelang::vnotOP**将被下降为hivm::VNotOp