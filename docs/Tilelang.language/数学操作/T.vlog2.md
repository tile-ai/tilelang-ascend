# Tilelang.language.vlog2

## 1. OP概述

简介：`tilelang.language.vlog2`

* ​**表达式级 `T.log2(x)`**​：对标 `tvm.tir.log2`，对单个 `PrimExpr` 做逐元素 log⁡2(x)。
* ​**NPU tile 级 `T.vlog2(src, dst, tmp)`**​：在 UB 等 on-chip buffer 上，对张量 tile 做逐元素 log⁡2 运算，底层用 **`ln` + `mul(1/ln2)`** 组合实现，编译到 NPUIR 算子。

```
T.vlog2(src, dst, temp)
```

## 2. 规格

### 2.1 参数说明

| 参数名   | 类型       | 描述   |
|-------|----------|------|
| `src` | `tensor` | 源张量  |
| `dst` | `tensor` | 目的张量 |
| `tmp` | `tensor` | 中间缓存，用于保存 ln(src) |

### 2.2 OP 规格

#### 2.2.1 DataType 支持

|              | int8 | int16 | int32 | uint8 | uint16 | uint32 | uint64 | int64 | fp16 | fp32 | fp64 | bf16 | bool |
|:-------------|:----:|:-----:|:-----:|:-----:|:------:|:------:|:------:|:-----:|:----:|:----:|:----:|:----:|:----:|
| Ascend A2/A3 |  ×   |   ×   |   ×   |   ×   |   ×    |   ×    |   ×    |   ×   |  √   |  √   |  ×   |  ×   |  ×

#### 2.2.2 Shape 支持

仅支持 1-5D tensor

### 2.3 使用方法

​**NPU tile 级示例（`examples/log2.py`）**​：

```python
@T.prim_func
def main(
        A: T.Tensor((M, N), dtype),
        B: T.Tensor((M, N), dtype),
):
    with T.Kernel(BLOCK_SIZE, is_npu=True) as (cid, _):
        A_VEC = T.alloc_ub((block_M, block_N), dtype)
        B_VEC = T.alloc_ub((block_M, block_N), dtype)
        tmp = T.alloc_ub((block_M, block_N), dtype)

        for i in T.serial(T.ceildiv(m_num * n_num, BLOCK_SIZE)):
            block_id = i * BLOCK_SIZE + cid
            if block_id < m_num * n_num:
                block_id_m = block_id // n_num
                block_id_n = block_id % n_num
                bx = block_id_m * block_M
                by = block_id_n * block_N

                T.copy(A[bx, by], A_VEC)
                T.vlog2(A_VEC, B_VEC, tmp)  # 逐元素 log2
                T.copy(B_VEC, B[bx, by])
```

**表达式级 T.log2 在 TIR 中的用法（`test_tilelang_kernel_mha_bwd.py`）**：

```python
# logsum 是一维 Fragment/Buffer，标量级 log2
for i in T.Parallel(block_M):
    logsum[i] = T.log2(logsum[i]) + scores_max[i] * scale
```

## 3. Tilelang Op到Ascend NPU IR Op的转换

**tilelang::vlog2**将被下降为`hivm::VLnOp` + `hivm::VMulOp`