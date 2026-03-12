# tilelang.language.vand

## 1.概述

简介:  `tilelang.language.vand` 用于对两个向量做按位与操作

```python
T.vand(A, B, C)
```

## 2.规格

### 2.1 参数说明

| 参数名 | 类型     | 描述       |
| ------ | -------- | ---------- |
| `A`    | `tensor` | 左输入向量 |
| `B`    | `tensor` | 右输入向量 |
| `C`    | `tensor` | 输出向量   |

### 2.2 支持规格

|              | int8 | int16 | int32 | uint8 | uint16 | uint32 | uint64 | int64 | fp16 | fp32 | fp64 | bf16 | bool |
| :----------- | :--: | :---: | :---: | :---: | :----: | :----: | :----: | :---: | :--: | :--: | :--: | :--: | :--: |
| Ascend A2/A3 |   ×   |   √   |    ×  |    ×   |   ×    |   ×    |   ×    |   √   |  ×   |  ×   |  ×   |  ×   |  √   |

### 2.3 Shape支持

支持自动广播

### 2.4 使用方法

```python
@tilelang.jit(target="npuir")
def vec_and(block_M, block_N, dtype="float16"):

    BLOCK_SIZE = 1

    @T.prim_func
    def main(
            A: T.Tensor((block_M, block_N), dtype),
			B: T.Tensor((block_M, block_N), dtype),
            C: T.Tensor((block_M, block_N), dtype),
    ):
        with T.Kernel(BLOCK_SIZE, is_npu=True) as (cid, _):
            A_VEC = T.alloc_ub((block_M, block_N), dtype)
	        B_VEC = T.alloc_ub((block_M, block_N), dtype)
            C_VEC = T.alloc_ub((block_M, block_N), dtype)

            T.copy(A, A_VEC)
	        T.copy(B, B_VEC)
            T.vand(A_VEC, B_VEC, C_VEC)
            T.copy(C_VEC, C)

    return main
```

## 3. Tilelang Op到Ascend NPU IR Op的转换

`tilelang::vandOp`将被下降为 `hivm.hir.VAndOp`