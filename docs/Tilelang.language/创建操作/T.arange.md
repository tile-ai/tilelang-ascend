# Tilelang.language.arange

# 1. OP概述

简介：`tilelang.language.arange` 根据步长（strides）和偏移量（offset），向向量中填充从 0、1、2…… 开始的连续序列。

```markup
T.arange(dst, strides: Union[list, tuple], offset=0)
```

## 2. OP规格

### 2.1 参数说明

| 参数名 | 类型 | 说明 |
| - | - | - |
| `dst` | `tensor` | 输出tensor |
| `strides`                         | `Union[list,tuple]` | 输入步长 |
| `offset`                          | `int` | 输入偏移量 |

### 2.2 支持规格

#### 2.2.1 DataType支持

|   | uint8 | int8 | uint16 | int16 | uint32 | int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool/int1 |
| - | - | - | - | - | - | - | - | - | - | - | - | - |
| Ascend | × | × | × | × | × | × | × | × | √ | √ | × | × |

#### 2.2.2 Shape支持

结论：在shape方面，arange无特殊要求；

### 2.3 特殊限制说明

无

### 2.4 使用方法

以下示例实现了一个形状为(M,N)的tensor的arange功能

```markup
def vec_arange(M, N, block_M, block_N, src_dtype="float32", dst_dtype="float16"):
    m_num = M // block_M
    n_num = N // block_N

    @T.prim_func
    def main(
            A: T.Tensor((M, N), dst_dtype),
            B: T.Tensor((M, N), dst_dtype),
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, _):
            bx_ = cid // n_num
            bx = bx_ * block_M
            by_ = cid % n_num
            by = by_ * block_N

            A_VEC = T.alloc_ub((block_M, block_N), dst_dtype)
            B_VEC = T.alloc_ub((block_M, block_N), dst_dtype)
            strides= [1, 2]
            T.arange(A_VEC, strides, offset=1)
            T.arange(B_VEC, strides)
            T.copy(A_VEC, A[bx, by])
            T.copy(B_VEC, B[bx, by])

    return main
```

## 3. Tilelang Op到Ascend NPU IR Op的转换

**tilelang::arangeOp**将被下降为hivm::VArangeOp