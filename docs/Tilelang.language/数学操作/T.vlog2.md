# Tilelang.language.vlog2

## 1. OP概述

简介：`tilelang.language.vlog2`

* ​**表达式级 `T.log2(x)`**​：对标 `tvm.tir.log2`，对单个 `PrimExpr` 做逐元素 log⁡2(x)。
* ​**NPU tile 级 `T.vlog2(src, dst, tmp)`**​：在 UB 等 on-chip buffer 上，对张量 tile 做逐元素 log⁡2 运算，底层用 **`ln` + `mul(1/ln2)`** 组合实现，编译到 NPUIR 算子。

```python
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
@tilelang.jit(target="npuir")
def vlog2_kernel(M, N, block_M, block_N, dtype="float16"):
    m_num = M // block_M
    n_num = N // block_N
    block_size = 8

    @T.prim_func
    def main(
        src: T.Tensor((M, N), dtype),
        dst: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(block_size, is_npu=True) as (cid, _):
            src_ub = T.alloc_ub((block_M, block_N), dtype)
            dst_ub = T.alloc_ub((block_M, block_N), dtype)
            tmp_ub = T.alloc_ub((block_M, block_N), dtype)

            for i in T.serial(T.ceildiv(m_num * n_num, block_size)):
                block_id = i * block_size + cid
                if block_id < m_num * n_num:
                    block_id_m = block_id // n_num
                    block_id_n = block_id % n_num
                    bx = block_id_m * block_M
                    by = block_id_n * block_N

                    T.copy(src[bx, by], src_ub)
                    T.vlog2(src_ub, dst_ub, tmp_ub)
                    T.copy(dst_ub, dst[bx, by])

    return main
```

**表达式级 T.log2 在 TIR 中的用法（`test_tilelang_kernel_mha_bwd.py`）**：

```python
@tilelang.jit(target="npuir")
def log2_expr_example(block_M):
    @T.prim_func
    def update_logsum(
        logsum: T.Tensor((block_M,), "float32"),
        scores_max: T.Tensor((block_M,), "float32"),
    ):
        scale = 0.5
        for i in T.Parallel(block_M):
            logsum[i] = T.log2(logsum[i]) + scores_max[i] * scale

    return update_logsum
```

## 3. Tilelang Op到Ascend NPU IR Op的转换

**tilelang::vlog2**将被下降为`hivm::VLnOp` + `hivm::VMulOp`
