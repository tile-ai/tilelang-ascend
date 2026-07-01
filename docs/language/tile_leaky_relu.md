# T.tile.leaky_relu

## 1. 功能说明

对操作数逐元素执行 Leaky ReLU 激活运算：`dst[i] = src[i] if src[i] >= 0 else alpha * src[i]`，其中 alpha 为负斜率系数，控制负值区域的斜率。

## 2. 函数原型

### 2.1 函数定义

```python
def leaky_relu(
    dst: Buffer | BufferRegion,
    src: Buffer | BufferRegion,
    alpha: PrimExpr,
)
```

### 2.2 参数说明

| 参数名 | 输入/输出 | 描述 | 类型 | 必填/可选 |
|--------|----------|------|------|----------|
| dst | 输出 | 存放 Leaky ReLU 运算结果 | 张量（tensor） | 必填 |
| src | 输入 | 源操作数 | 张量（tensor） | 必填 |
| alpha | 输入 | 负斜率系数（negative slope） | 标量（scalar） | 必填 |

> **类型说明**：
> - **tensor**：通过 `T.alloc_ub`、`T.alloc_shared` 等分配的缓冲区（Buffer），或其切片（BufferRegion）
> - **scalar**：单个元素值，可以是 Python 标量或表达式（PrimExpr）

### 2.3 参数规格

#### 2.3.1 DataType 支持

| 平台 | dst | src | alpha |
|------|:---:|:---:|:---:|
| Ascend A2 / A3 | float16, float32 | float16, float32 | float16, float32 |
| Ascend A5 | float16, float32 | float16, float32 | float16, float32 |

> **注意**：alpha 的 dtype 若与 dst 不同，codegen 会自动插入类型转换（如 `half(0.01)`）。

#### 2.3.2 Shape 支持

- 支持 1D 和 2D

#### 2.3.3 alpha 参数

- alpha 为必填参数，无默认值，需显式指定负斜率系数
- 常用取值范围：0.01 ~ 0.1 的小正数
- alpha 的数据类型需与 dst 元素类型一致，若不一致 codegen 会自动转换

### 2.4 约束条件

1. dst 与 src 的元素总数必须相同
2. dst 与 src 的 dtype 必须一致（Ascend C 硬件约束）
3. alpha 的数据类型需与 dst 元素类型一致（Ascend C 约束）
4. 操作数地址需 32 字节对齐（硬件约束）
5. 不支持 BF16 数据类型

## 3. 示例代码

**示例 1：1D Leaky ReLU**

```python
src = T.alloc_ub((256,), "float16")
dst = T.alloc_ub((256,), "float16")
T.tile.leaky_relu(dst, src, 0.01)  # negative slope = 0.01
```

**示例 2：2D Leaky ReLU**

```python
src = T.alloc_ub((128, 64), "float16")
dst = T.alloc_ub((128, 64), "float16")
T.tile.leaky_relu(dst, src, 0.01)  # negative slope = 0.01
```
