from gc import enable

# tilelang.language.store_fixpipe

## 1. 概述

简介： `tilelang.language.store_fixpipe` 用于在 `Expert` 模式中将 `L0C` 中的数据通过 `fixpipe` 数据通路搬运到 `GM` 中

```python
T.store_fixpipe(src, dst, size=[], enable_nz2nd=False, channel_split=False, pre_relu_mode="")
```

## 2. 规格

### 2.1 参数说明

| 参数名             | 类型          | 缺省      | 描述                   |
|-----------------|-------------|---------|----------------------|
| `src`           | `tensor`    | 必需      | 源张量（必须来自 `L0C` 地址空间） |
| `dst`           | `tensor`    | 必需      | 目的张量（必须来自 `GM` 地址空间） |
| `size`          | `List[int]` | `[]`    | 搬运的张量大小        |
| `enable_nz2nd`  | `bool`      | `False` | 是否在数据搬运同时做`nz2nd`变换  |
| `channel_split` | `bool`      | `False` | 是否在数据搬运同时切分通道        |
| `pre_relu_mode` | `str`       | `""`    | 在数据搬运同时应用何种relu操作    |

约束：
- `src` 和 `dst` 应具有相同的数据类型。
- 在 `src` 和 `dst` 维数不一致的情况下 `size` 参数会对齐最后的维度
- `pre_relu_mode` 参数候选列表如下：`"", "relu", "leaky_relu", "prelu"` ，其中默认值 `""` 代表不应用 `relu` 。

### 2.2 OP 规格

#### 2.2.1 DataType 支持

|              | int8 | int16 | int32 | uint8 | uint16 | uint32 | uint64 | int64 | fp16 | fp32 | fp64 | bf16 | bool |
|:-------------|:----:|:-----:|:-----:|:-----:|:------:|:------:|:------:|:-----:|:----:|:----:|:----:|:----:|:----:|
| Ascend A2/A3 |  ×   |   ×   |   ×   |   ×   |   ×    |   ×    |   ×    |   ×   |  ×   |  √   |  ×   |  ×   |  ×   |

#### 2.2.2 Shape 支持

- 参数 `src` 仅支持 2D `tensor`
- 参数 `dst` 支持 2-5D `tensor`

### 2.3 特殊限制说明

无

### 2.4 使用方法

以下示例实现了在矩阵乘法完成之后，使用 `tilelang.language.store_fixpipe` 将数据从 `L0C` 搬运到 `GM` 中：

```python
@tilelang.jit(target="npuir")
def kernel_mha_qk_matmul(b, n, s, d, block_d, dtype="float16", accum_dtype="float32"):

    @T.prim_func
    def main(
            Q: T.Tensor((b, n, s, d), dtype),
            K: T.Tensor((b, n, s, d), dtype),
            A: T.Tensor((b, n, s, s), accum_dtype)
    ):
        with T.Kernel(b * n, is_npu=True) as (cid, _):
            with T.Scope("Cube"):
                b_id = cid // n
                n_id = cid % n

                Q_BUF = T.alloc_L1([s, block_d], dtype)
                K_BUF = T.alloc_L1([s, block_d], dtype)
                A_BUF = T.alloc_L0C([s, s], accum_dtype)

                for i in T.serial(T.ceildiv(d, block_d)):
                    real_block_d = d - i * block_d
                    real_block_d = T.min(real_block_d, block_d)
                    T.load_nd2nz(Q[b_id, n_id, 0, i * block_d], Q_BUF, [s, real_block_d])
                    T.load_nd2nz(K[b_id, n_id, 0, i * block_d], K_BUF, [s, real_block_d])

                    T.gemm(Q_BUF, K_BUF, A_BUF, initC=(i == 0), b_transpose=True, size=[s, real_block_d, s])

                T.store_fixpipe(A_BUF, A[b_id, n_id, 0, 0], size=[s, s], enable_nz2nd=True)

    return main
```

## 3. Tilelang Op 到 Ascend NPU IR Op 的转换

`tilelang.language.store_fixpipe` 被下降为 `hivm.hir.fixpipe`
