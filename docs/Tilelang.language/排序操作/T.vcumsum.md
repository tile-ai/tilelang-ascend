# Tilelang.language.cumsum

## 1. OP概述

简介：`tilelang.language.cumsum`该算子返回对输入张量 `src` 沿指定维度 `dim` 进行**​累积和（cumulative sum）​**操作。

```
T.vcumsum(src, dst, dim, reverse) [Developer Op]
```

## 2. OP规格

### 2.1 参数说明

| 参数名    | 类型         | 说明       |
| ----------- | -------------- | ------------ |
| `src` | `tensor` | 输入tensor |
| `dst` | `tensor` | 输出tensor |
| ``dim`` | ``int`` | (可选)  指定在输入张量的哪个维度上执行累积和，默认为0 |
| ``reverse`` | ``bool`` | (可选)  如果 reverse 为 True，则执行反向累积和（即从末尾向前累加） 目前只支持False|

### 2.2 支持规格

#### 2.2.1 DataType支持

|        | uint8 | int8 | uint16 | int16 | uint32 | int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool |
| -------- | ------- | ------ | -------- | ------- | -------- | ------- | -------- | ------- | ------ | ------ | ------ | ----------- |
| Ascend | √    | ×   |×     | ×    | ×     | √    | ×    | √   | √   |√   | ×   | ×        |

#### 2.2.2 Shape支持

结论：输入（input）与输出（output）shape要一致。

### 2.3 特殊限制说明

无

### 2.4 使用方法

示例：实现了第0维的cumsum

```
import tilelang
import tilelang.language as T
dtype = "float16"
accum_dtype = "float16"
def cumsum_kernel(M, N, dim, reverse):
    BLOCK_SIZE = 1
    @T.prim_func
    def main(src: T.Tensor((M, N), dtype),
             dst: T.Tensor((M, N), accum_dtype)):

        with T.Kernel(BLOCK_SIZE, is_npu=True) as (cid, _):
            # Allocate UB memory
            src_ub = T.alloc_ub((M, N), dtype)
            dst_ub = T.alloc_ub((M, N), accum_dtype)

            # Copy data from GM to UB
            T.copy(src, src_ub)

            # Perform cumulative sum
            T.cumsum(src_ub, dst_ub, dim=dim, reverse=reverse)

            # Copy results back to GM
            T.copy(dst_ub, dst)

    return main
```

## 3. Tilelang Op到Ascend NPU IR Op的转换

**tilelang::cumsumOp**将被下降为hivm::VCumsumOp
