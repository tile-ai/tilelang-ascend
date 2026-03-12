# Tilelang.language.pad

## 1. OP概述

简介：`tilelang.language.pad` 用于对张量按指定维度在首尾添加元素。

```
T.pad(src, dst, pad_value, low, high, size)
```

## 2. 规格

### 2.1 参数说明

| 参数名   | 类型       | 描述   |
|-------|----------|------|
| `src` | `tensor` | 源张量  |
| `dst` | `tensor` | 目的张量 |
| `pad_value` | Python 标量 | 填充值 |
| `low` | `List[Union[int, tir.Var]]` 或 `Tuple[...]` | 沿各维度「起始端」的 padding 长度 |
| `high` | `List[Union[int, tir.Var]]` 或 `Tuple[...]` | 沿各维度「结束端」的 padding 长度 |
| `size`(可选) | `List[int]`，默认 `[]` | 手动指定 src 的逻辑 shape

### 2.2 OP 规格

#### 2.2.1 DataType 支持

|              | int8 | int16 | int32 | uint8 | uint16 | uint32 | uint64 | int64 | fp16 | fp32 | fp64 | bf16 | bool |
|:-------------|:----:|:-----:|:-----:|:-----:|:------:|:------:|:------:|:-----:|:----:|:----:|:----:|:----:|:----:|
| Ascend A2/A3 |  ×   |   ×   |   ×   |   ×   |   ×    |   ×    |   ×    |   ×   |  √   |  √   |  ×   |  ×   |  ×

#### 2.2.2 Shape 支持

仅支持 1-5D tensor

### 2.3 使用方法

**示例 ：二维 tile 上对行维做对称 padding**
来自 `unittest/npuir/test_vec_pad.py`（简化）：

```python
@T.prim_func
def main(
    A: T.Tensor((M, N), src_dtype),
    B: T.Tensor((2*M, 2*N), dst_dtype),
    C: T.Tensor((M, N), dst_dtype),
):
    m_num = M // block_M
    n_num = N // block_N

    with T.Kernel(m_num * n_num, is_npu=True) as (cid, _):
        bx_ = cid // n_num
        bx = bx_ * block_M
        by_ = cid % n_num
        by = by_ * block_N

        A_VEC = T.alloc_ub((block_M, block_N), src_dtype)
        B_VEC = T.alloc_ub((2 * block_M, block_N), dst_dtype)
        C_VEC = T.alloc_ub((block_M + 2 * m_num * n_num, block_N), dst_dtype)

        T.copy(A[bx, by], A_VEC)

        # 1）在第 0 维两端各 pad block_M/2 行，pad_value = 0.0
        T.pad(A_VEC, B_VEC, 0.0, [block_M / 2, 0], [block_M / 2, 0])

        # 2）在第 0 维前后各 pad cid 行，作为示例
        T.pad(A_VEC, C_VEC, 0.0, [cid, 0], [cid, 0])

        T.copy(B_VEC, B[2 * bx, by])
        T.copy(C_VEC, C[bx, by])
```

## 3. Tilelang Op到Ascend NPU IR Op的转换

**tilelang::pad**将被下降为`hivm::VPadOp`