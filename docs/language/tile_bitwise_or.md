# T.tile.bitwise_or

## 1. 功能说明

对两个操作数逐元素执行按位或运算：`dst[i] = src0[i] | src1[i]`

## 2. 函数原型

### 2.1 函数定义

```python
def bitwise_or(
    dst: Buffer | BufferRegion,
    src0: Buffer | BufferRegion,
    src1: Buffer | BufferRegion | BufferLoad | PrimExpr,
)
```

### 2.2 参数说明

| 参数名 | 输入/输出 | 描述 | 类型 | 必填/可选 |
|--------|----------|------|------|----------|
| dst | 输出 | 存放按位或运算结果 | 张量（tensor） | 必填 |
| src0 | 输入 | 第一个源操作数 | 张量（tensor） | 必填 |
| src1 | 输入 | 第二个源操作数，支持 tensor 或 scalar | 张量（tensor）/ 标量（scalar） | 必填 |

> **类型说明**：
> - **tensor**：通过 `T.alloc_ub`、`T.alloc_shared` 等分配的缓冲区（Buffer），或其切片（BufferRegion）
> - **scalar**：单个元素值，可以是 buffer 元素访问（BufferLoad）或 Python 标量/表达式（PrimExpr）

### 2.3 参数规格

#### 2.3.1 DataType 支持

| 平台 | dst | src0 | src1 |
|------|:---:|:----:|:----:|
| Ascend A2 / A3 | int16, uint16 | int16, uint16 | int16, uint16 |
| Ascend A5 | int8, uint8, int16, uint16, int32, uint32, int64, uint64 | int8, uint8, int16, uint16, int32, uint32, int64, uint64 | int8, uint8, int16, uint16, int32, uint32, int64, uint64 |

> **说明**：A2/A3 上 int32/uint32 需通过 ReinterpretCast 转为 int16/uint16 后调用。

#### 2.3.2 Shape 支持

- 支持 1D 和 2D

### 2.4 约束条件

1. dst 与 src0 的 shape 必须相同
2. src1 为 tensor 时，shape 也必须与 dst 相同
3. src0 和 src1 的 dtype 必须与 dst 一致
4. 仅支持整数类型，不支持浮点类型
5. 操作数地址需 32 字节对齐（硬件约束）

## 3. 示例代码

**示例 1：tensor-tensor 按位或**

```python
src0 = T.alloc_ub((256,), "int16")
src1 = T.alloc_ub((256,), "int16")
dst = T.alloc_ub((256,), "int16")
T.tile.bitwise_or(dst, src0, src1)
```

**示例 2：tensor-scalar 按位或**

```python
src0 = T.alloc_ub((256,), "int16")
dst = T.alloc_ub((256,), "int16")
T.tile.bitwise_or(dst, src0, 0x0F)  # src1 = 0x0F，每个元素与 0x0F 做按位或
```
