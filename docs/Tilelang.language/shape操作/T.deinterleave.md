# tilelang.language.deinterleave

## 1. 概述

简介： `tilelang.language.deinterleave` 用于对张量执行解交织，根据下标将源向量分解为多个目标向量

```python
T.deinterleave(src, *dsts, channel_nums=2, index_mode="ALL_CHANNELS", size=[])
```

## 2. 规格

### 2.1 参数说明

| 参数名            | 类型          | 缺省               | 描述                 |
|----------------|-------------|------------------|--------------------|
| `src`          | `tensor`    | 必需               | 源张量                |
| `*dsts`        | `tensor`    | 必需               | 可变数量的目的张量，可传入一个或多个 |
| `channel_nums` | `int`       | `2`              | 交织通道数              |
| `index_mode`   | `str`       | `"ALL_CHANNELS"` | 索引模式               |
| `size`         | `List[int]` | `[]`             | 源张量参与运算区域形状        |

约束：
- `src` 和 `dst` 应具有相同的数据类型
- `src` 的最后一维必须能被 `channel_nums` 整除
- `index_mode` 参数候选列表如下： `"CHANNEL_0", "CHANNEL_1", "ALL_CHANNELS"`

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

以下示例实现了将源张量 `Input` 解交织到两个目的张量中：

```python
@tilelang.jit(target='npuir')
def deinterleave_c2(M, N, block_M, dtype="float16"):
    assert N % 2 == 0
    m_num = M // block_M
    N_half = N // 2

    @T.prim_func
    def main(
            Input: T.Tensor((M, N), dtype),
            Output0: T.Tensor((M, N_half), dtype),
            Output1: T.Tensor((M, N_half), dtype),
    ):
        with T.Kernel(m_num, is_npu=True) as (cid, _):
            offset = cid * block_M

            ub_input = T.alloc_ub((block_M, N), dtype)
            ub_output0 = T.alloc_ub((block_M, N_half), dtype)
            ub_output1 = T.alloc_ub((block_M, N_half), dtype)

            T.copy(Input[offset:offset + block_M, :], ub_input)
            T.deinterleave(ub_input, ub_output0, ub_output1, channel_nums=2)
            T.copy(ub_output0, Output0[offset:offset + block_M, :])
            T.copy(ub_output1, Output1[offset:offset + block_M, :])

    return main
```

## 3. Tilelang Op 到 Ascend NPU IR Op 的转换

`tilelang.language.deinterleave` 被下降为 `hivm.hir.vdeinterleave`
