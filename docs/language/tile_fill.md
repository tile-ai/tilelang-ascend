# T.tile.fill

## 1. 功能说明

将 buffer 中的所有元素填充为指定标量值：`buffer[i] = value`

## 2. 函数原型

### 2.1 函数定义

```python
def fill(
    buffer: Buffer | BufferRegion,
    value: PrimExpr,
)
```

### 2.2 参数说明

| 参数名 | 输入/输出 | 描述 | 类型 | 必填/可选 |
|--------|----------|------|------|----------|
| buffer | 输入/输出 | 待填充的 buffer | 张量（tensor） | 必填 |
| value | 输入 | 填充的标量值 | 标量（scalar） | 必填 |

> **类型说明**：
> - **tensor**：通过 `T.alloc_ub` 分配的 Buffer，或其连续 BufferRegion。
> - **scalar**：单个元素值，可以是 Python 标量或表达式（PrimExpr），dtype 需可转换到 buffer 的 dtype。

### 2.3 参数规格

#### 2.3.1 DataType 支持

以下 dtype 基于 Ascend A2 / A3（910B）真机验证：

| dtype | Ascend C | PTO |
|-------|----------|-----|
| float16 | 支持 | 支持 |
| float32 | 支持 | 支持 |
| bfloat16 | 支持 | 支持 |
| int16 | 支持 | 支持 |
| uint16 | 支持 | 支持 |
| int32 | 支持 | 支持 |
| uint32 | 支持 | 支持 |
| int8 | 不支持 | 支持 |
| uint8 | 不支持 | 支持 |

#### 2.3.2 Shape 支持

- 支持 1D 和 2D Buffer，以及其中的连续 BufferRegion。
- size 由 buffer shape 自动推断（BufferRegion 时取 region extent 的乘积，Buffer 时取 shape 的乘积）。

### 2.4 约束条件

1. value 的 dtype 需可转换到 buffer 的 dtype；不一致时前端自动 Cast。
2. buffer 地址需 32 字节对齐（硬件约束）。
3. size 由 buffer shape 自动推断，无需显式传入 count 参数。
4. 仅支持 UB（Unified Buffer）内存；其他内存级别的 fill 行为未经验证。
5. 仅支持片上 buffer fill，GM 级别 fill 需用 T.copy。

## 3. 示例代码

**示例 1：填充零值**

```python
acc_s_ub = T.alloc_ub((block_M, block_N), "float16")
T.tile.fill(acc_s_ub, 0.0)  # 将 acc_s_ub 所有元素填充为 0.0
```

**示例 2：填充常量值**

```python
scale_ub = T.alloc_ub((128,), "float32")
T.tile.fill(scale_ub, 0.125)  # 将 scale_ub 所有元素填充为 0.125
```

**示例 3：填充 BufferRegion 切片**

```python
a_ub = T.alloc_ub((block_M, block_N), "float32")
T.tile.fill(a_ub, 10.0)                        # 全部填充为 10.0
T.tile.fill(a_ub[0:block_M // 2, 0:block_N], 5.0)  # 仅前半部分填充为 5.0
```
