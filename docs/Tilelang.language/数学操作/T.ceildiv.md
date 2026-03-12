# tilelang.language.ceildiv

## 1.概述

简介:  `tilelang.language.ceildiv` 用于对两个整数做向上整除

```python
T.ceildiv(lhs, rhs)
```

## 2.OP概述

### 2.1 参数说明

| 参数名 | 类型     | 说明   |
| ------ | -------- | ------ |
| `lhs`  | `scalar` | 被除数 |
| `rhs`  | `scalar` | 除数   |

### 2.2 支持规格

#### 2.2.1 DataType支持

|              | int8 | int16 | int32 | uint8 | uint16 | uint32 | uint64 | int64 | fp16 | fp32 | fp64 | bf16 | bool |
| :----------- | :--: | :---: | :---: | :---: | :----: | :----: | :----: | :---: | :--: | :--: | :--: | :--: | :--: |
| Ascend A2/A3 |  √   |    √   |   √   |   ×   |   ×    |   ×    |   ×    |   √   |  ×   |  ×   |  ×   |  ×   |  ×   |

#### 2.2.2 Shape支持

在shape方面，ceildiv无特殊要求；

### 2.3 特殊限制说明

无

### 2.3 使用方法

一般在分核的核数计算及定义坐标时使用

```
def vec_add(block_M, block_N, dtype="float32"):
    M = T.symbolic("M")
    N = T.symbolic("N")

    @T.prim_func
    def main(
            A: T.Tensor((M, N), dtype),
            B: T.Tensor((M, N), dtype),
            C: T.Tensor((M, N), dtype)
    ):
        with T.Kernel(T.ceildiv(N, block_N) * T.ceildiv(M, block_M), is_npu=True) as (cid, _):
            #代码段省略

    return main
```

## 3. Tilelang Op到Ascend NPU IR Op的转换

`tilelang::ceildiv` Op 将被下降为 `arith.addi` Op, `arith.divsi` Op