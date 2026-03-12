# Tilelang.language.verf

## 1. OP概述

简介：`tilelang.language.verf`返回输入tensor src基于输出形状的误差函数（error function）计算结果。误差函数定义为
$$
\operatorname{verf}(x) = \frac{2}{\sqrt{\pi}} \int_0^x e^x{-t^2} \, dt
$$
其值域为 \((-1, 1)\)。

```
T.verf(src, dst)
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
def vec_erf(M, N, block_M, block_N, dtype="float16"):
    m_num = M // block_M
    n_num = N // block_N
    BLOCK_SIZE = 8

    @T.prim_func
    def main(
            A: T.Tensor((M, N), dtype),
            B: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(BLOCK_SIZE, is_npu=True) as (cid, _):
            A_VEC = T.alloc_shared((block_M, block_N), dtype)
            B_VEC = T.alloc_shared((block_M, block_N), dtype)

            for i in T.serial(T.ceildiv(m_num * n_num, BLOCK_SIZE)):
                block_id = i * BLOCK_SIZE + cid
                if block_id < m_num * n_num:
                    block_id_m = block_id // n_num
                    block_id_n = block_id % n_num
                    bx = block_id_m * block_M
                    by = block_id_n * block_N

                    T.copy(A[bx, by], A_VEC)
                    T.npuir_verf(A_VEC, B_VEC)
                    T.copy(B_VEC, B[bx, by])
    return main
```


## 3. Tilelang Op到Ascend NPU IR Op的转换

**tilelang::verfOp**将被转换为`arith::ConstantOp`, `arith::DivFOp`, `hivm::VMulOp`, `hivm::VAddOp`