# Tilelang.language.vsqrt

## 1. OP概述

简介：`tilelang.language.vsqrt`返回输入向量/标量基于输出形状的sqrt计算结果

sqrt计算公式：x^0.5

```
T.vsqrt(src, dst)
```

## 2. 规格

### 2.1 参数说明

| 参数名   | 类型       | 描述   |
|-------|----------|------|
| `src` | `tensor` | 输入tensor  |
| `dst` | `tensor` | 输出tensor |

### 2.2 OP 规格

#### 2.2.1 DataType 支持

|              | int8 | int16 | int32 | uint8 | uint16 | uint32 | uint64 | int64 | fp16 | fp32 | fp64 | bf16 | bool |
|:-------------|:----:|:-----:|:-----:|:-----:|:------:|:------:|:------:|:-----:|:----:|:----:|:----:|:----:|:----:|
| Ascend A2/A3 |  ×   |   ×   |   ×   |   ×   |   ×    |   ×    |   ×    |   ×   |  √   |  √   |  ×   |  ×   |  ×

#### 2.2.2 Shape 支持

无特殊要求

### 2.3 使用方法

```python
def vsqrt_kernel(M, N, dtype):
    BLOCK_SIZE = 1

    @T.prim_func
    def main(
            A: T.Tensor((M, N), dtype),
            B: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(BLOCK_SIZE, is_npu=True) as (cid, _):
            src_ub = T.alloc_shared((M, N), dtype)
            dst_ub = T.alloc_shared((M, N), dtype)

            T.copy(src, src_ub)
            T.vsqrt(src_ub, dst_ub)
            T.copy(dst_ub, dst)
    return main
```


## 3. Tilelang Op到Ascend NPU IR Op的转换

**tilelang::VSqrtOp**将被转换为`hivm::VSqrtOp`