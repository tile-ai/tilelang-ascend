# Tilelang.language.interleave

## 1. OP概述

简介：`tilelang.language.interleave` 将多个tensor的值在最后一个维度上交错排列合并为一个tensor

```
T.interleave(src1, src2, ..., dst, channel_nums=2, size=[])
```

## 2. OP规格

### 2.1 参数说明

| 参数名            | 类型     | 说明                                              |
| ----------------- | -------- | ------------------------------------------------- |
| `src1, src2, ...` | `tensor` | 输入tensor列表，所有输入tensor必须具有相同的shape |
| `dst`             | `tensor` | 输出tensor                                        |
| `channel_nums`    | `int`    | 每个输入在每次交错中参与的通道数，默认为2         |
| `size`            | `list`   | 可选参数，指定buffer的extent，默认为空列表        |

### 2.2 支持规格

#### 2.2.1 DataType支持

|        | uint8 | int8 | uint16 | int16 | uint32 | int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool/int1 |
| ------ | ----- | ---- | ------ | ----- | ------ | ----- | ------ | ----- | ---- | ---- | ---- | --------- |
| Ascend | √     | √    | √      | √     | √      | √     | √      | √     | √    | √    | √    | √         |

#### 2.2.2 Shape支持

结论：所有输入tensor必须具有相同的shape；输出tensor的最后一个维度大小为输入tensor最后一个维度大小的2倍（当有2个输入时）

### 2.3 特殊限制说明

- 由于硬件限制，目前仅支持两个tensor的交错操作
- 输入tensor的shape必须相同
- 输出tensor的shape会根据输入tensor的shape和channel_nums自动计算

### 2.4 使用方法

以下示例实现了将两个形状为(M,N)的tensor在最后一个维度上交错排列

```
import torch
import torch_npu
import tilelang
import tilelang.language as T

def vinterleave_kernel(M, N, dtype):
    BLOCK_SIZE = 1

    @T.prim_func
    def main(A: T.Tensor((M, N), dtype),
             B: T.Tensor((M, N), dtype),
             C: T.Tensor((M, N * 2), dtype)):

        with T.Kernel(BLOCK_SIZE, is_npu=True) as (cid, _):

            A_ub = T.alloc_shared((M, N), dtype)
            B_ub = T.alloc_shared((M, N), dtype)
            C_ub = T.alloc_shared((M, N * 2), dtype)

            T.copy(A, A_ub)
            T.copy(B, B_ub)
            T.interleave(A_ub, B_ub, C_ub, channel_nums=2)
            T.copy(C_ub, C)

    return main
```

## 3. Tilelang Op到Ascend NPU IR Op的转换

**tilelang::interleaveOp**将被转换为mlir::hivm::VInterleaveOp