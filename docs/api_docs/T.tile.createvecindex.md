# T.tile.createvecindex

## 1. 功能说明

生成递增向量索引序列：`dst[i] = firstValue + i`（i = 0, 1, ..., count-1，count 由 dst 的总元素数自动推导）

## 2. 函数原型

### 2.1 函数定义

```python
def createvecindex(
    dst: Buffer,
    firstValue: PrimExpr,
)
```

### 2.2 参数说明

| 参数名 | 输入/输出 | 描述 | 类型 | 必填/可选 |
|--------|----------|------|------|----------|
| dst | 输出 | 存放索引序列，count 由 `math.prod(dst.shape)` 自动计算 | 张量（tensor） | 必填 |
| firstValue | 输入 | 索引序列的起始值，数据类型須与 dst 的元素类型一致 | 标量（scalar） | 必填 |

> **类型说明**：
> - **tensor**：通过 `T.alloc_ub`、`T.alloc_shared` 等分配的缓冲区（Buffer），或其切片（BufferRegion）
> - **scalar**：单个元素值，可以是 buffer 元素访问（BufferLoad）或 Python 标量/表达式（PrimExpr）

### 2.3 参数规格

#### 2.3.1 DataType 支持

| 平台 | dst | firstValue |
|------|:---:|:----------:|
| Ascend A2 / A3 | float32, int16, int32 | 同 dst |

- float16 仅 ascendc 后端支持（pto 后端编译失败）
- uint16、uint32 仅 pto 后端支持（ascendc 后端编译失败）

#### 2.3.2 Shape 支持

- 无维度限制（1D / 2D / 多维均可），`math.prod(dst.shape)` 自动计算总元素数

### 2.4 约束条件

1. firstValue 的数据类型須与 dst 的元素类型保持一致（编译期不强制校验，C++ 层会做隐式类型转换）
2. firstValue 不能超出 dst 元素数据类型的取值范围（硬件约束）
3. dst 的起始地址必须 32 字节对齐（硬件约束）

## 3. 示例代码

**示例 1：1D int32 索引序列**

```python
idx_ub = T.alloc_ub((128,), "int32")
T.tile.createvecindex(idx_ub, 0)  # idx_ub = [0, 1, 2, ..., 127]
```

**示例 2：2D float 索引序列**

```python
idx_2d = T.alloc_ub((16, 8), "float32")
T.tile.createvecindex(idx_2d, 0)  # 按行优先展开，idx_2d = [0, 1, 2, ..., 127]
```

**示例 3：以非零值起始**

```python
idx_ub = T.alloc_ub((64,), "int16")
T.tile.createvecindex(idx_ub, 10)  # idx_ub = [10, 11, 12, ..., 73]
```
