# Tilelang.language.vbitcast

## 1. OP概述

简介：`tilelang.language.vbitcast` 不改变底层数据的情况下重新解释已定型值（shaped value）的二进制位

```
T.vbitcast(src, dtype, size = [])
```

## 2. OP规格

### 2.1 参数说明

| 参数名 | 类型 | 说明 |
| - | - | - |
| `src` | `tensor` | 源向量 |
| `dtype`                       | `fp16、fp32` | 结果向量的数据类型 |
| `size`                        | `list` | 手动指定维度（比如`[32, 32]`），覆盖自动推导的维度，适配非标准形状的数据源 |

### 2.2 支持规格

#### 2.2.1 DataType支持

|  | uint8 | int8 | uint16 | int16 | uint32 | int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool/int1 |
| - | - | - | - | - | - | - | - | - | - | - | - | - |
| Ascend | × | × | × | × | × | × | × | × | √ | √ | × | × |

#### 2.2.2 Shape支持

结论：在shape方面，vbitcast无特殊要求；

### 2.3 特殊限制说明

要求**源和目标类型的总位宽必须相等**

### 2.4 使用方法

以下示例实现了一个形状为(M,N)的tensor的vbitcast功能

```
def vec_bitcast(M, N, block_M, block_N, src_dtype="float16"):
    m_num = M // block_M
    n_num = N // block_N
    @T.prim_func
    def main(
            A: T.Tensor((M, N), src_dtype),
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, _):
            bx_ = cid // n_num
            bx = bx_ * block_M
            by_ = cid % n_num
            by = by_ * block_N
            A_VEC = T.alloc_ub((block_M, block_N), src_dtype)
            T.copy(A[bx, by], A_VEC)
            T.vbitcast(A_VEC, "int16")
    return main</code></pre>
```

## 3. Tilelang Op到Ascend NPU IR Op的转换

**tilelang::vbitcastOp**将被下降为hivm::BitcastOp