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

- 整数 dtype（int16/int32 等）会在编译期报错，不支持

> **注意**：scalar_value 的 dtype 若与 dst 不同，codegen 会自动插入类型转换（如 `half(0.01)`）。

#### 2.3.2 Shape 支持

- 支持 1D 和 2D
- 支持整行切片（如 `buf[0:32, :]`）；仅计算切片区域内元素，区域外内容未定义
- 不支持 2D 列偏移切片（如 `buf[:, 8:40]`）：真机实测两端均触发 aicore 异常（507015）

### 2.4 约束条件

1. dst 与 src0 的元素总数必须相同（Python 断言，报错信息 "size must be same"）
2. dst 与 src0 的 dtype 必须一致（Ascend C 约束）
3. 操作数地址需 32 字节对齐（硬件约束）
4. 不支持 BF16 数据类型（编译期报错）
5. 特殊值遵循 IEEE 语义：`leaky_relu(0)=0`、`leaky_relu(-inf)=-inf × alpha`、`leaky_relu(nan)=nan`

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