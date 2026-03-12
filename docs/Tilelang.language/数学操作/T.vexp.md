# tilelang.language.vexp

## 1. 概述

简介： `tilelang.language.vexp` 用于计算张量以欧拉数 `e` 为底的幂

```python
T.vexp(src, dst)
```

## 2. 规格

### 2.1 参数说明

| 参数名   | 类型       | 描述   |
|-------|----------|------|
| `src` | `tensor` | 源张量  |
| `dst` | `tensor` | 目的张量 |

约束： `src` 和 `dst` 应具有相同的形状和数据类型

### 2.2 OP 规格

#### 2.2.1 DataType 支持

|              | int8 | int16 | int32 | uint8 | uint16 | uint32 | uint64 | int64 | fp16 | fp32 | fp64 | bf16 | bool |
|:-------------|:----:|:-----:|:-----:|:-----:|:------:|:------:|:------:|:-----:|:----:|:----:|:----:|:----:|:----:|
| Ascend A2/A3 |  ×   |   ×   |   ×   |   ×   |   ×    |   ×    |   ×    |   ×   |  √   |  √   |  ×   |  ×   |  ×   |

#### 2.2.2 Shape 支持

仅支持 1-5D tensor

### 2.3 特殊限制说明

无

### 2.4 使用方法

以下示例实现了计算输入张量 `input` 中每个元素以 `e` 为底的幂并输出到张量 `output` 中：

```python
@tilelang.jit(target='npuir')
def vec_exp(M, N, block_M, block_N, dtype="float16"):
    m_num = M // block_M
    n_num = N // block_N

    @T.prim_func
    def main(
            Input: T.Tensor((M, N), dtype),
            Output: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, _):
            bx = cid // n_num
            by = cid % n_num

            ub_input = T.alloc_ub((block_M, block_N), dtype)
            ub_output = T.alloc_ub((block_M, block_N), dtype)

            T.copy(Input[bx * block_M, by * block_N], ub_input)
            T.vexp(ub_input, ub_output)
            T.copy(ub_output, Output[bx * block_M, by * block_N])
            
    return main
```

## 3. Tilelang Op 到 Ascend NPU IR Op 的转换

`tilelang.language.vexp` 被下降为 `hivm.hir.vexp`
