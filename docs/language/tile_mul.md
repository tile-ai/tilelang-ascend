# T.tile.mul

## 1. 功能说明

对两个操作数逐元素执行乘法运算：`dst[i] = src0[i] * src1[i]`

## 2. 函数原型

### 2.1 函数定义

```python
def mul(
    dst: Buffer | BufferRegion,
    src0: Buffer | BufferRegion,
    src1: Buffer | BufferRegion | BufferLoad | PrimExpr,
)
```

### 2.2 参数说明

| 参数名 | 输入/输出 | 描述 | 类型 | 必填/可选 |
|--------|----------|------|------|----------|
| dst | 输出 | 存放乘法运算结果 | 张量（tensor） | 必填 |
| src0 | 输入 | 第一个源操作数 | 张量（tensor） | 必填 |
| src1 | 输入 | 第二个源操作数，支持 tensor 或 scalar | 张量（tensor）/ 标量（scalar） | 必填 |

> **类型说明**：
> - **tensor**：通过 `T.alloc_ub`、`T.alloc_shared` 等分配的缓冲区（Buffer），或其切片（BufferRegion）
> - **scalar**：单个元素值，可以是 buffer 元素访问（BufferLoad）或 Python 标量/表达式（PrimExpr）

### 2.3 参数规格

#### 2.3.1 DataType 支持

| 平台 | dst | src0 | src1 |
|------|:---:|:----:|:----:|
| Ascend A2 / A3 | int16, float16, int32, float32 | int16, float16, int32, float32 | int16, float16, int32, float32 |
| Ascend A5 | int16, uint16, float16, bfloat16, int32, uint32, float32, int64, uint64 | int16, uint16, float16, bfloat16, int32, uint32, float32, int64, uint64 | int16, uint16, float16, bfloat16, int32, uint32, float32, int64, uint64 |

> **注意**：A5 平台不支持 int8 / uint8 的乘法运算。
> **后端差异说明**：A5 平台上 int64/uint64 仅 Ascend C 后端支持，PTO 后端不支持。

#### 2.3.2 Shape 支持

- 支持 1D 和 2D

### 2.4 约束条件

1. dst 与 src0 的 shape 必须相同
2. src1 为 tensor 时，shape 也必须与 dst 相同
3. src0 和 src1 的 dtype 必须与 dst 一致
4. 操作数地址需 32 字节对齐（硬件约束）

## 3. 示例代码

**示例 1：tensor-tensor 乘法**

```python
src0 = T.alloc_ub((256,), "float16")
src1 = T.alloc_ub((256,), "float16")
dst = T.alloc_ub((256,), "float16")
T.tile.mul(dst, src0, src1)
```

**示例 2：tensor-scalar 乘法**

```python
src0 = T.alloc_ub((256,), "float16")
dst = T.alloc_ub((256,), "float16")
T.tile.mul(dst, src0, 0.5)  # src1 = 0.5，每个元素乘 0.5
```
