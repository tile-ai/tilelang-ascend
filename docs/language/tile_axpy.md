# T.tile.axpy

## 1. 功能说明

将源 buffer `src0` 与标量 `scalar_value` 相乘后逐元素加到目标 buffer `dst` 上：`dst[i] = scalar_value * src0[i] + dst[i]`

dst 同时作为输入和输出，原地更新 dst 的内容。

## 2. 函数原型

### 2.1 函数定义

```python
def axpy(
    dst: Buffer | BufferRegion,
    src0: Buffer | BufferRegion,
    scalar_value: PrimExpr,
)
```

### 2.2 参数说明

| 参数名 | 输入/输出 | 描述 | 类型 | 必填/可选 |
|--------|----------|------|------|----------|
| dst | 输入/输出 | 目标 buffer，同时作为累加器输入 Y 和输出 | 张量（tensor） | 必填 |
| src0 | 输入 | 源 buffer X | 张量（tensor） | 必填 |
| scalar_value | 输入 | 标量系数 alpha | 标量（scalar） | 必填 |

> **类型说明**：
> - **tensor**：通过 `T.alloc_ub` 分配的缓冲区（Buffer），或其切片（BufferRegion）
> - **scalar**：Python 标量或标量表达式（PrimExpr），调用时会按目标数据类型进行转换

### 2.3 参数规格

#### 2.3.1 DataType 支持

| 平台 | dst | src0 | scalar_value |
|------|:---:|:----:|:------------:|
| Ascend A2 / A3 | float16, float32 | float16, float32 | 与 dst 数据类型兼容的标量表达式 |

> **数据类型约束**：当前 Ascend C 和 PTO 后端均要求 `dst` 与 `src0` 的数据类型相同。

#### 2.3.2 Shape 支持

- 支持 Buffer 和 BufferRegion；已验证二维 Buffer 和一维切片
- `dst` 与 `src0` 的元素总数必须相同

### 2.4 约束条件

1. `dst` 为 read-write 语义，调用后其内容被原地更新
2. `dst` 与 `src0` 的元素总数必须相同
3. 当前 Ascend C 和 PTO 后端要求 `dst` 与 `src0` 的数据类型相同
4. 操作数应位于 Unified Buffer，起始地址需满足 32 字节对齐要求

## 3. 示例代码

**示例 1：标量乘加**

```python
dst = T.alloc_ub((128,), "float16")
src = T.alloc_ub((128,), "float16")
T.tile.axpy(dst, src, 2.0)  # dst = 2.0 * src + dst
```

**示例 2：累加缩放**

```python
acc = T.alloc_ub((64, 128), "float16")
grad = T.alloc_ub((64, 128), "float16")
T.tile.axpy(acc, grad, 0.1)  # acc = 0.1 * grad + acc（梯度累加）
```