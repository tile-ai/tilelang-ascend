# Tilelang.language.concat

## 1. OP概述

简介：`tilelang.language.concat` 将输入tensor在指定维度进行拼接

```
T.concat(src_1, src_2, ..., src_N, dst, concat_dim)
```

## 2. OP规格

### 2.1 参数说明

| 参数名 | 类型 | 说明 |
| - | - | - |
| `src_i` | `tensor` | 参与拼接的源张量 |
| `dst`                                                                | `tensor` | 用于写入拼接结果的张量 |
| `concat_dim`                                                              | `scalar` | 用于拼接的维度 |

### 2.2 支持规格

#### 2.2.1 DataType支持

无tensor类型限制

#### 2.2.2 Shape支持

无

### 2.3 特殊限制说明

无

### 2.4 使用方法

以下示例展示了concat的使用，在第1维进行拼接：

```python
def concat(M, N, block_M, block_N, dtype="float16"):
    m_num = M // block_M
    n_num = N // block_N
    VEC_NUM = 2
    BLOCK_SIZE = 8
    @T.prim_func
    def main(
            A: T.Tensor((M, N), dtype),
            B: T.Tensor((M, N), dtype),
            C: T.Tensor((M, 2*N), dtype),
    ):
        with T.Kernel(BLOCK_SIZE, is_npu=True) as (cid, _):
            bx_ = cid // n_num
            bx = bx_ * block_M
            by_ = cid % n_num
            by = by_* block_N
            A_VEC = T.alloc_ub((block_M, block_N), dtype)
            B_VEC = T.alloc_ub((block_M, block_N), dtype)
            C_VEC = T.alloc_ub((block_M, 2*block_N), dtype)
            for i in T.serial(T.ceildiv(m_num*n_num, BLOCK_SIZE)):
                block_id_base = i * BLOCK_SIZE
                block_id = block_id_base + cid
                block_id_m = block_id // n_num
                block_id_n = block_id % n_num
                bx = block_id_m * block_M
                by = block_id_n * block_N
                dim = 1
                T.copy(A[bx, by], A_VEC)
                T.copy(B[bx, by], B_VEC)
                T.concat(A_VEC, B_VEC, C_VEC, dim)
                T.copy(C_VEC, C[bx, 2*by])
    return main
```

## 3. Tilelang Op到Ascend NPU IR Op的转换

tilelang::concatOp将被下降为hivm::VConcatOp
