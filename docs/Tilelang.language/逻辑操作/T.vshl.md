# Tilelang.language.vshl

## 1. OP概述

简介：`tilelang.language.vshl`对输入张量`A`执行按元素左移（bitwise left shift），移位位数由另一个输入`B`指定，结果写入输出`C`

```
T.vshl(A, B, C)
```

## 2. 规格

### 2.1 参数说明

| 参数名   | 类型       | 描述   |
|-------|----------|------|
| `A` | `tensor` | 输入tensor  |
| `B` | `tensor`,`scalar` | 输入tensor |
| `C` | `tensor` | 输出tensor  |

### 2.2 OP 规格

#### 2.2.1 DataType 支持

|              | int8 | int16 | int32 | uint8 | uint16 | uint32 | uint64 | int64 | fp16 | fp32 | fp64 | bf16 | bool |
|:-------------|:----:|:-----:|:-----:|:-----:|:------:|:------:|:------:|:-----:|:----:|:----:|:----:|:----:|:----:|
| Ascend A2/A3 |  ×   |   √   |   √   |   ×   |   ×    |   ×    |   ×    |   √   |  ×   |  ×   |  ×   |  ×   |  ×

#### 2.2.2 Shape 支持

C和A的shape必须保持一致
B的shape需要和A的一致或为标量

### 2.3 使用方法
示例1：B与A有相同的shape
```python
def vshl_kernel(N, dtype):
    BLOCK_SIZE = 1

    @T.prim_func
    def main(
            A: T.Tensor((N,), dtype),
            B: T.Tensor((N,), dtype),
            Out: T.Tensor((N,), dtype),
    ):
        with T.Kernel(BLOCK_SIZE, is_npu=True) as (cid, _):
            acc_A = T.alloc_shared((N,), dtype)
            acc_B = T.alloc_shared((N,), dtype)
            out_ub = T.alloc_shared((N,), dtype)

            T.copy(A, acc_A)
            T.copy(B, acc_B)
            T.vshl(acc_A, acc_B, out_ub)
            T.copy(out_ub, Out)
    return main
```

示例2：B是标量
```python
def vshl_kernel(N, dtype):
    BLOCK_SIZE = 1

    @T.prim_func
    def main(
            A: T.Tensor((N,), dtype),
            B: T.Tensor((1,), dtype),
            Out: T.Tensor((N,), dtype),
    ):
        with T.Kernel(BLOCK_SIZE, is_npu=True) as (cid, _):
            acc_A = T.alloc_shared((N,), dtype)
            acc_B = T.alloc_shared((1,), dtype)
            out_ub = T.alloc_shared((N,), dtype)

            T.copy(A, acc_A)
            T.copy(B, acc_B)
            T.vshl(acc_A, acc_B, out_ub)
            T.copy(out_ub, Out)
    return main
```


## 3. Tilelang Op到Ascend NPU IR Op的转换

**tilelang::VShlOp**将被转换为`hivm::VShLOp`