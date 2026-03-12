# Tilelang.language.infinity

## 1. OP概述

简介：`tilelang.language.infinity`根据给定的数据类型创建一个表示无穷大的值。

```
T.infinity(dtype)
```

## 2. 规格

### 2.1 参数说明

| 参数名   | 类型       | 描述   |
|-------|----------|------|
| `dtype` | `str` | 所需无穷大值的数据类型  |

### 2.2 OP 规格

#### 2.2.1 DataType 支持

|              | int8 | int16 | int32 | uint8 | uint16 | uint32 | uint64 | int64 | fp16 | fp32 | fp64 | bf16 | bool |
|:-------------|:----:|:-----:|:-----:|:-----:|:------:|:------:|:------:|:-----:|:----:|:----:|:----:|:----:|:----:|
| Ascend A2/A3 |  ×   |   ×   |   ×   |   ×   |   ×    |   ×    |   ×    |   ×   |  √   |  √   |  ×   |  ×   |  ×

#### 2.2.2 Shape 支持

返回标量

### 2.3 使用方法
以下示例实现了将一个二维张量B的所有元素都设置为负无穷大(-inf)
```python
    @T.prim_func
    def main(
            A: T.Tensor((M, 1), dtype),
            B: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(BLOCK_SIZE, is_npu=True) as (cid, _):
            bx_ = cid // n_num
            bx = bx_ * block_M
            by_ = cid % n_num
            by = by_ * block_N

            A_SCALAR = -T.infinity("float32")
            B_VEC = T.alloc_ub((block_M, block_N), dtype)
            for i in T.serial(T.ceildiv(m_num * n_num, BLOCK_SIZE)):
                block_id_base = i * BLOCK_SIZE
                block_id = block_id_base + cid
                block_id_m = block_id // n_num
                block_id_n = block_id % n_num
                bx = block_id_m * block_M
                by = block_id_n * block_N
                T.npuir_brc(A_SCALAR, B_VEC)
                T.copy(B_VEC, B[bx, by])
```


## 3. Tilelang Op到Ascend NPU IR Op的转换

**tilelang::infinityOp**将被转换为`arith::ConstantOp`