# tilelang.language.vadd

## 1. OP概述

简介：`tilelang.language.vadd` 对两个向量/tensor执行逐元素加法运算，支持自动广播

```markup
T.vadd(src0, src1, dst)
```

计算公式：`dst = src0 + src1`

## 2. OP规格

### 2.1 参数说明

| 参数名 | 类型 | 说明 |
| - | - | - |
| `src0` | `tensor` | 加数tensor（左操作数） |
| `src1`                                         | `tensor`或`scalar` | 加数tensor或标量（右操作数） |
| `dst`                                          | `tensor` | 输出tensor，存储加法结果 |

### 2.2 支持规格

#### 2.2.1 DataType支持

|   | uint8 | int8 | uint16 | int16 | uint32 | int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool/int1 |
| - | - | - | - | - | - | - | - | - | - | - | - | - |
| Ascend | × | × | × | × | × | × | × | × | √ | √ | × | × |

#### 2.2.2 Shape支持

结论：支持自动广播，具体规则如下：

- 相同shape：`[M, N] + [M, N]` → `[M, N]`
- 行广播：`[M, N] + [M, 1]` → `[M, N]`
- 列广播：`[1, N] + [1, 1]` → `[1, N]`
- 标量广播：`[M, N] + scalar` → `[M, N]`

注意：参与运算的tensor必须具有相同的rank（维度数），推荐统一使用2D buffer `[M, N]`。

### 2.3 特殊限制说明

1. 操作数buffer必须分配在 **UB（Unified Buffer）** 上
2. 标量操作数必须在 `T.Kernel` 内、`T.Scope` 外定义
3. `dst` 可以与 `src0` 或 `src1` 为同一buffer（支持原地操作）
4. 仅支持`src1`为标量，不支持`src0`为标量

### 2.4 使用方法

以下示例实现了一个形状为(M,N)的tensor的逐元素加法计算

```
def sub_kernel(M, N, dtype):
    BLOCK_SIZE = 1

    @T.prim_func
    def main(src0: T.Tensor((M, N), dtype),
             src1: T.Tensor((M, N), dtype),
             dst: T.Tensor((M, N), dtype)):

        with T.Kernel(BLOCK_SIZE, is_npu=True) as (cid, _):

            src0_ub = T.alloc_shared((M, N), dtype)
            src1_ub = T.alloc_shared((M, N), dtype)
            dst_ub = T.alloc_shared((M, N), dtype)

            T.copy(src0, src0_ub)
            T.copy(src1, src1_ub)
            T.vadd(src0_ub, src1_ub, dst_ub)
            T.copy(dst_ub, dst)

    return main
```

### 3. Tilelang Op到Ascend NPU IR Op的转换

**tilelang::vaddOp**将被下降为hivm::VAddOp