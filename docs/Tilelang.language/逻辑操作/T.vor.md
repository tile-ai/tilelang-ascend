# Tilelang.language.vor

## 1. OP概述

简介：`tilelang.language.vor` 用于在 NPU ​tile 级上做逐元素按位或运算.

```
T.vor(src0, src1, dst)
```

## 2. 规格

### 2.1 参数说明

| 参数名   | 类型       | 描述   |
|-------|----------|------|
| `src0` | `tensor` | 源张量0  |
| `src1` | `tensor` | 源张量1 (或 Python 标量 int / float：会被提升为与 C 相同 dtype 的常量并广播)|
| `dst` | `tensor` | 目的张量 |

约束：`src0`, `src1`(张量时) 和`dst`应具有相同的形状和数据类型

### 2.2 OP 规格

#### 2.2.1 DataType 支持

|              | int8 | int16 | int32 | uint8 | uint16 | uint32 | uint64 | int64 | fp16 | fp32 | fp64 | bf16 | bool |
|:-------------|:----:|:-----:|:-----:|:-----:|:------:|:------:|:------:|:-----:|:----:|:----:|:----:|:----:|:----:|
| Ascend A2/A3 |  ×   |   ×   |   √   |   ×   |   ×    |   ×    |   ×    |   ×   |  ×   |  ×   |  ×   |  ×   |  ×   |

​**规范使用场景是整型和 bool 类型的按位 OR**​，其它 dtype 为实现细节，不作为公开承诺。

#### 2.2.2 Shape 支持

仅支持 1-5D tensor

### 2.3 使用方法

以下示例实现了计算输入张量`input1`, `input2` 中逐元素按位或运算并输出到张量`output`中：

```python
@tilelang.jit(target='npuir')
def vec_exp(M, N, block_M, block_N, dtype="float16"):
    m_num = M // block_M
    n_num = N // block_N

    @T.prim_func
    def main(
            Input1: T.Tensor((M, N), dtype),
			Input2: T.Tensor((M, N), dtype),
            Output: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, _):
            bx = cid // n_num
            by = cid % n_num
			ub_input1 = T.alloc_ub((block_M, block_N), dtype)
	    	ub_input2 = T.alloc_ub((block_M, block_N), dtype)
            ub_output = T.alloc_ub((block_M, block_N), dtype)

            T.copy(Input1[bx * block_M, by * block_N], ub_input1)
	    	T.copy(Input2[bx * block_M, by * block_N], ub_input2)
            T.vor(ub_input1, ub_input2, ub_output)
            T.copy(ub_output, Output[bx * block_M, by * block_N])

    return main
```

## 3. Tilelang Op到Ascend NPU IR Op的转换

**tilelang::vor**将被下降为`hivm::VOrOp`