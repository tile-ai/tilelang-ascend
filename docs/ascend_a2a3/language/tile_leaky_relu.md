# T.tile.leaky_relu

## 1. 功能说明

对源操作数逐元素执行 Leaky ReLU 激活运算：`dst[i] = src0[i] if src0[i] >= 0 else alpha * src0[i]`，其中 alpha 为负斜率系数，控制负值区域的斜率。

## 2. 函数原型

### 2.1 函数定义

```python
def leaky_relu(
    dst: Buffer | BufferRegion,
    src0: Buffer | BufferRegion,
    scalar_value: PrimExpr,
)
```

### 2.2 参数说明

| 参数名 | 输入/输出 | 描述 | 类型 | 必填/可选 |
|--------|----------|------|------|----------|
| dst | 输出 | 存放 Leaky ReLU 运算结果 | 张量（tensor） | 必填 |
| src0 | 输入 | 源操作数 | 张量（tensor） | 必填 |
| scalar_value | 输入 | 负斜率系数（negative slope） | 标量（scalar） | 必填 |

> **类型说明**：
> - **tensor**：通过 `T.alloc_ub`、`T.alloc_shared` 等分配的缓冲区（Buffer），或其切片（BufferRegion）
> - **scalar**：单个元素值，可以是 Python 标量或表达式（PrimExpr）

### 2.3 参数规格

#### 2.3.1 DataType 支持

| 平台 | dst | src0 | scalar_value |
|------|:---:|:----:|:---:|
| Ascend A2 / A3 | float16, float32 | float16, float32 | float16, float32 |

> **注意**：scalar_value 会自动按 dst 的 dtype 进行转换。

#### 2.3.2 Shape 支持

- 支持 1D 和 2D
- 支持整行切片（如 `buf[0:32, :]`）；仅计算切片区域内元素，区域外内容未定义
- 不支持 2D 列偏移切片（如 `buf[:, 8:40]`）

### 2.4 约束条件

1. dst 与 src0 的元素总数必须相同
2. dst 与 src0 的 dtype 必须一致（Ascend C 约束）
3. dst 可与 src0 为同一 buffer（如 `T.tile.leaky_relu(a_ub, a_ub, 0.01)`）
4. 操作数地址需 32 字节对齐（硬件约束）

## 3. 示例代码

**示例 1：1D Leaky ReLU**

```python
src0 = T.alloc_ub((256,), "float16")
dst = T.alloc_ub((256,), "float16")
T.tile.leaky_relu(dst, src0, 0.01)  # negative slope = 0.01
```

**示例 2：2D Leaky ReLU**

```python
src0 = T.alloc_ub((128, 64), "float16")
dst = T.alloc_ub((128, 64), "float16")
T.tile.leaky_relu(dst, src0, 0.01)  # negative slope = 0.01
```