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
| src1 | 输入 | 第二个源操作数 | 张量（tensor）/ 标量（scalar） | 必填 |

> **类型说明**：
> - **tensor**：通过 `T.alloc_ub`、`T.alloc_shared` 等分配的缓冲区（Buffer），或其切片（BufferRegion）
> - **scalar**：单个元素值，可以是 buffer 元素访问（BufferLoad）或 Python 标量/表达式（PrimExpr）

### 2.3 参数规格

#### 2.3.1 DataType 支持

| 平台 | dst | src0 | src1 |
|------|:---:|:----:|:----:|
| Ascend A2 / A3 | int8, uint8, int16, uint16 | 同 dst | 同 dst |

#### 2.3.2 Shape 支持

- 支持 1D 和 2D

### 2.4 约束条件

1. 输入和输出张量必须位于 UB 内存
2. dst 与 src0 的元素个数必须相同
3. 当 src1 为切片时，其元素个数也必须与 dst 相同
4. src0 和 src1 的 dtype 必须与 dst 一致
5. 仅支持 int8、uint8、int16 和 uint16
6. 操作数地址需 32 字节对齐（硬件约束）
7. 当前 src1 仅支持张量，标量暂不可用（参见 [Issue #177](https://github.com/tile-ai/tilelang-ascend/issues/177)）

## 3. 示例代码

**示例 1：tensor-tensor 按位或**

```python
src0 = T.alloc_ub((256,), "int16")
src1 = T.alloc_ub((256,), "int16")
dst = T.alloc_ub((256,), "int16")
T.tile.bitwise_or(dst, src0, src1)
```

**示例 2：张量切片按位或**

```python
src0 = T.alloc_ub((128, 256), "int16")
src1 = T.alloc_ub((128, 256), "int16")
dst = T.alloc_ub((128, 256), "int16")
T.tile.bitwise_or(dst[0:128, 0:256], src0[0:128, 0:256], src1[0:128, 0:256])
```

**示例 3：int8 按位或**

```python
src0 = T.alloc_ub((256,), "int8")
src1 = T.alloc_ub((256,), "int8")
dst = T.alloc_ub((256,), "int8")
T.tile.bitwise_or(dst, src0, src1)
```
