# Tilelang.language.tanh

## 1. OP概述

简介：`tilelang.language.tanh` 返回输入向量/标量基于输出形状的tanh计算结果

```
T.vtanh(src, dst)
```

## 2. OP规格

### 2.1 参数说明

| 参数名  | 类型  | 说明  |
| ------------ | ------------ | ------------ |
| `src` | `tensor` | 输入tensor  |
| `dst` | `tensor` | 输出tensor  |

### 2.2 支持规格

#### 2.2.1 DataType支持

|   | uint8 | int8 | uint16 | int16 | uint32 | int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool/int1 |
| ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ |
| Ascend | ×  | × |  × |  × | ×  | ×  | ×  | ×  | √  | √ |  ×  | ×  |

#### 2.2.2 Shape支持

结论：无特殊要求

### 2.3 特殊限制说明

无

### 2.4 使用方法

```
def vec_tanh(M, N, dtype):
    BLOCK_SIZE = 8

    @T.prim_func
    def main(
            src: T.Tensor((M, N), dtype),
            dst: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(BLOCK_SIZE, is_npu=True) as (cid, _):
            src_ub = T.alloc_ub ((M, N), dtype)
            dst_ub = T.alloc_ub ((M, N), dtype)

            T.copy(src, src_ub)
            T.vtanh(src_ub, dst_ub)
            T.copy(src_ub, dst)

    return main

```

## 3. Tilelang Op到Ascend NPU IR Op的转换

**tilelang::vtanhOp**将被下降为arith::ConstantOp, arith::DivFOp, hivm::VMulOp, hivm::VAddOp