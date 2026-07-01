# T.tile.axpy

## 1. 功能说明

将源 buffer 与标量相乘后逐元素加到目标 buffer 上：`dst[i] = scalar * src[i] + dst[i]`

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
> - **tensor**：通过 `T.alloc_ub`、`T.alloc_shared` 等分配的缓冲区（Buffer），或其切片（BufferRegion）
> - **scalar**：单个元素值，可以是 Python 标量或表达式（PrimExpr）

### 2.3 参数规格

#### 2.3.1 DataType 支持

| 平台 | dst | src0 | scalar_value |
|------|:---:|:----:|:----:|
| Ascend A2 / A3 | float16, float32 | float16, float32 | float16, float32 |
| Ascend A5 | float16, float32, bfloat16, int64, uint64 | float16, float32, bfloat16, int64, uint64 | float16, float32, bfloat16, int64, uint64 |

> **混合精度说明**：PTO 后端支持 dst=float32 + src0=float16 的混合精度组合；Ascend C 后端强制要求 dst 与 src0 的 dtype 必须相同。

> **后端差异说明**：A5 平台上 int64/uint64 仅 Ascend C 后端支持。

#### 2.3.2 Shape 支持

- 支持 1D 和 2D
- dst 与 src0 的元素总数必须相同

### 2.4 约束条件

1. dst 为 read-write 语义：调用后 dst 内容被原地修改
2. dst 与 src0 的 shape 必须兼容（元素总数一致）
3. Ascend C 后端要求 dst 与 src0 的 dtype 相同
4. src=float16 + dst=float32 时，Ascend C 不支持地址重叠
5. 操作数地址需 32 字节对齐（硬件约束）

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
