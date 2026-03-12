# Tilelang.language.vabs

## 1. OP概述

简介：`tilelang.language.vabs` 该算子返回输入向量中的元素**逐元素的绝对值**

```
T.vabs(src, dst) [Developer Op]
T.vabs(src, dst) [Expert Op]
```

## 2. OP规格

### 2.1 参数说明

| 参数名    | 类型         | 说明       |
| ----------- | -------------- | ------------ |
| `src` | `tensor` | 输入tensor |
| `dst` | `tensor` | 输出tensor |

### 2.2 支持规格

#### 2.2.1 DataType支持

|        | uint8 | int8 | uint16 | int16 | uint32 | int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool |
| -------- | ------- | ------ | -------- | ------- | -------- | ------- | -------- | ------- | ------ | ------ | ------ | ----------- |
| Ascend |√    | ×   |×     | ×    | ×     | √    | ×    | √   | √   |√   | ×   | ×        |

#### 2.2.2 Shape支持

结论：输入（input）与输出（output）shape要一致。

### 2.3 特殊限制说明

无

### 2.4 使用方法

示例1：实现了Expert Mode中将一个二维张量（Tensor） 的绝对值存到dst中

```
import tilelang
import tilelang.language as T
tilelang.cache.clear_cache()
dtype = "float16"          # Input data type
out_dtype = "float16"      # Output data type (same as input for abs)
def vabs_kernel(M, N):
    @T.prim_func
    def main(src: T.Tensor((M, N), dtype),
             dst: T.Tensor((M, N), out_dtype)):
        # Use a single block to process the whole tensor
        with T.Kernel(1, is_npu=True) as (bid, _):
            # Allocate UB memory for input and output
            src_ub = T.alloc_ub((M, N), dtype)
            dst_ub = T.alloc_ub((M, N), out_dtype)
            # Copy data from GM to UB
            T.copy(src, src_ub)
            T.vabs(src_ub, dst_ub)
            # Copy results back to GM
            T.copy(dst_ub, dst)
    return main
```

## 3. Tilelang Op到Ascend NPU IR Op的转换

**tilelang::vabsOp**将被下降为hivm::VAbsOp
