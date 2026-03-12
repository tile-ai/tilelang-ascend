# Tilelang.language.gather

## 1. OP概述

简介：`tilelang.language.gather` 根据给定的索引，从沿最后一个维度张量或 memref 中取出相应元素，并将这些元素存储到另一个张量或 memref 中。

```markup
T.gather(src, dst, indices,[m,n])
```

## 2. OP规格

### 2.1 参数说明


| 参数名    | 类型         | 说明                     |
| --------- | ------------ | ------------------------ |
| `src`     | `tensor`     | 输入tensor               |
| `dst`     | `tensor`     | 输出tensor               |
| `indices` | `list/tuple` | gather 操作的索引数组    |
| `size`    | `list`       | 实际参与gather的数据范围 |

### 2.2 支持规格

#### 2.2.1 DataType支持


|        | uint8 | int8 | uint16 | int16 | uint32 | int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool/int1 |
| ------ | ----- | ---- | ------ | ----- | ------ | ----- | ------ | ----- | ---- | ---- | ---- | --------- |
| Ascend | ×    | ×   | ×     | √    | ×     | √    | ×     | ×    | √   | √   | √   | ×        |

### 2.3 使用方法

以下示例实现了对输入矩阵沿最后一维做gather操作，indices固定为1的结果。

```markup
import tilelang
import tilelang.language as T

dtype = "float16"

def gather_dev(M, N):
    BLOCK_SIZE = 1

    @T.prim_func
    def main(
            A: T.Tensor((M, N), dtype),
            B: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(BLOCK_SIZE, is_npu=True) as (cid, _):

            A_VEC = T.alloc_shared((M, N), dtype)
            B_VEC = T.alloc_shared((M, N), dtype)
            indices = T.alloc_shared((M, N), "int32")
            value_one = 1
            T.npuir_brc(value_one, indices)
            T.copy(A, A_VEC)
            T.gather(A_VEC, B_VEC, indices, [M,N])
            T.copy(B_VEC, B)

    return main
```

## 3. Tilelang Op到Ascend NPU IR Op的转换

**tilelang::gatherOp**将被下降为hivm::VGatherOp