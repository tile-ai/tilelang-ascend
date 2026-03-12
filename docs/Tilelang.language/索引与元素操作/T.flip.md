# tilelang.language.flip

## 1. 概述

简介: `tilelang.language.flip` 对一个 tensor 沿指定方向进行翻转

```python
T.flip(src, dst, axis: int)
```

## 2. 规格

### 2.1 参数说明

| 参数名 | 类型     | 描述     |
| ------ | -------- | -------- |
| `src`  | `tensor` | 输入向量 |
| `dst`  | `tensor` | 输出向量 |
| `axis` | `int`    | 翻转维度 |

### 2.2 DataType 支持

|              | int8 | int16 | int32 | uint8 | uint16 | uint32 | uint64 | int64 | fp16 | fp32 | fp64 | bf16 | bool |
| :----------- | :--: | :---: | :---: | :---: | :----: | :----: | :----: | :---: | :--: | :--: | :--: | :--: | :--: |
| Ascend A2/A3 |   ×   |    ×   |    ×  |    ×   |   ×    |   ×    |   ×    |    ×  |  √   |  √   |  ×   |  ×  |   ×  |

### 2.3 Shape支持

仅支持 1~5 维 tensor

### 2.4 使用方法

```python
@tilelang.jit(target="npuir")
def vec_flip(block_M, block_N, dtype="float16"):

    BLOCK_SIZE = 1

    @T.prim_func
    def main(
            A: T.Tensor((block_M, block_N), dtype),
            C: T.Tensor((block_M, block_N), dtype),
    ):
        with T.Kernel(BLOCK_SIZE, is_npu=True) as (cid, _):

            A_VEC = T.alloc_ub((block_M, block_N), dtype)
            C_VEC = T.alloc_ub((block_M, block_N), dtype)

            T.copy(A, A_VEC)
            T.flip(A_VEC[:block_M, :block_N], C_VEC[:block_M, :block_N], 1)
            T.copy(C_VEC, C)

    return main
```

## 3. Tilelang Op到Ascend NPU IR Op的转换

`tilelang::flip` Op 将被下降为 `hivm.hir.vfip` Op