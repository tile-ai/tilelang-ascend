# Tilelang.language.reduce_sum

## 1. OP概述

简介：`tilelang.language.reduce_sum` 对输入tensor在指定维度上进行求和归约

```
T.reduce_sum(buffer, out, dim=-1, clear=True)
```

## 2. OP规格

### 2.1 参数说明

| 参数名   | 类型     | 说明                                                         |
| -------- | -------- | ------------------------------------------------------------ |
| `buffer` | `tensor` | 输入tensor                                                   |
| `out`    | `tensor` | 输出tensor                                                   |
| `dim`    | `int`    | 进行归约的维度，默认为-1（最后一个维度）                     |
| `clear`  | `bool`   | 是否在归约前清空输出tensor，默认为True。若为False，则在现有值上累加 |

### 2.2 支持规格

#### 2.2.1 DataType支持

|        | uint8 | int8 | uint16 | int16 | uint32 | int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool/int1 |
| ------ | ----- | ---- | ------ | ----- | ------ | ----- | ------ | ----- | ---- | ---- | ---- | --------- |
| Ascend | ×     | ×    | ×      | ×     | ×      | ×     | ×      | ×     | √    | √    | ×    | ×         |

#### 2.2.2 Shape支持

结论：输出tensor的shape为输入tensor的shape在指定维度上归约后的结果

### 2.3 特殊限制说明

- dim参数必须在输入tensor的维度范围内

### 2.4 使用方法

以下示例实现了对形状为(M,N)的tensor在最后一个维度上进行求和归约

```
def reduce_sum_kernel(M, N, dtype):
    BLOCK_SIZE = 1

    @T.prim_func
    def main(src: T.Tensor((M, N), dtype),
             dst: T.Tensor((M,), dtype)):

        with T.Kernel(BLOCK_SIZE, is_npu=True) as (cid, _):

            src_ub = T.alloc_shared((M, N), dtype)
            dst_ub = T.alloc_shared((M,), dtype)

            T.copy(src, src_ub)
            T.reduce_sum(src_ub, dst_ub, dim=-1, clear=True)
            T.copy(dst_ub, dst)

    return main
```

## 3. Tilelang Op到Ascend NPU IR Op的转换

**tilelang::reduce_sumOp**将被转换为`mlir::hivm::VReduceOp`