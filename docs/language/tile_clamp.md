# T.tile.clamp

## 1. 功能说明

将源 buffer 中的元素钳位到闭区间 `[min_val, max_val]`，并写入目标 buffer：
`out[i] = min(max(buffer[i], min_val), max_val)`。

小于下界的元素替换为 `min_val`，大于上界的元素替换为 `max_val`，区间内的元素保持不变。
支持输入和输出使用同一个 buffer 的原地计算。

`min_val` 和 `max_val` 既可以是标量表达式，也可以是 UB buffer（张量形式）。
当使用张量形式时，逐元素钳位：`out[i] = min(max(buffer[i], min_val[i]), max_val[i])`。

> **注意区分**：`T.clamp` 是标量级运算，用于 `T.Parallel` 循环内的逐元素表达式；
> `T.tile.clamp` 是 buffer 级 intrinsic，对 Unified Buffer 中的一段连续区域执行计算。

## 2. 函数原型

### 2.1 函数定义

```python
def clamp(
    out: Buffer | BufferRegion,
    buffer: Buffer | BufferRegion,
    min_val: PrimExpr | Buffer | BufferRegion,
    max_val: PrimExpr | Buffer | BufferRegion,
    count: PrimExpr | None = None,
    *,
    tmp: Buffer | BufferRegion | None = None,
)
```

### 2.2 参数说明

| 参数名 | 输入/输出 | 描述 | 类型 | 必填/可选 |
|--------|----------|------|------|----------|
| out | 输出 | 存放钳位结果的 UB buffer，可与 `buffer` 相同 | 张量（tensor） | 必填 |
| buffer | 输入 | 源 UB buffer | 张量（tensor） | 必填 |
| min_val | 输入 | 闭区间下界，可为标量或张量 | 标量（scalar）或张量（tensor） | 必填 |
| max_val | 输入 | 闭区间上界，可为标量或张量 | 标量（scalar）或张量（tensor） | 必填 |
| count | 输入 | 从所选区域起始位置开始参与计算的元素个数，省略时使用 buffer 的总元素数 | 整数标量（scalar） | 可选，默认从 buffer 推导 |
| tmp | 输入/输出 | 可选的 UB 临时空间，当前无实际语义，仅为 API 兼容保留 | 张量（tensor）或 `None` | 可选，默认 `None` |

> **类型说明**：
> - **tensor**：通过 `T.alloc_ub` 分配的 Buffer，或其连续切片（BufferRegion）。
> - **scalar**：TileLang 编译期可表示的标量或整数表达式。

### 2.3 参数规格

#### 2.3.1 DataType 支持

| 平台 | out / buffer | min_val / max_val（标量） | min_val / max_val（张量） |
|------|:---:|:---:|:---:|
| Ascend A2 / A3 | float16, float32, int16, int32 | 可转换为 buffer dtype 的标量 | 与 buffer 相同的 dtype |

`out` 与 `buffer` 的 dtype 必须相同。当 `min_val` / `max_val` 为张量时，其 dtype 和元素数须与 `buffer` 一致。AscendC 与 PTO 后端的支持范围一致。

#### 2.3.2 Shape 支持

- 支持一维、二维 Buffer，以及其中的连续切片。
- `out` 与 `buffer` 的 shape 和元素总数必须相同。
- 当 `min_val` / `max_val` 为张量时，其 shape 和元素总数也须与 `buffer` 一致。

#### 2.3.3 count 说明

- AscendC 按 `count` 处理所选区域的前缀元素，未处理的后缀保持不变。
- PTO 当前要求 `count` 等于所选 tile 的完整元素数，不支持 partial count。

### 2.4 约束条件

1. `out` 与 `buffer` 必须具有相同的 dtype、shape 和可访问元素数。
2. `min_val` 必须小于或等于 `max_val`（标量形式时）；张量形式时逐元素满足。
3. `count` 不得超过源区域或目标区域可访问的元素数；PTO 还要求 `count` 等于完整 tile 元素数。
4. 操作数位于 Unified Buffer，起始地址需满足 32 字节对齐要求（硬件约束）。
5. `out` 可以与 `buffer` 完全相同以执行原地计算；其他部分重叠方式不属于支持范围。

## 3. 示例代码

**示例 1：标量边界**

```python
src = T.alloc_ub((64,), "float16")
dst = T.alloc_ub((64,), "float16")
T.tile.clamp(dst, src, 0.0, 6.0, 17)  # 只钳位前 17 个元素
```

**示例 2：原地计算（count 省略）**

```python
values = T.alloc_ub((4, 16), "float32")
T.tile.clamp(values, values, -1.0, 1.0)  # 原地计算，count 自动推导
```

**示例 3：张量边界**

```python
src = T.alloc_ub((64,), "float16")
dst = T.alloc_ub((64,), "float16")
min_ub = T.alloc_ub((64,), "float16")
max_ub = T.alloc_ub((64,), "float16")
T.tile.clamp(dst, src, min_ub, max_ub)  # 逐元素钳位
```

**示例 4：混合标量与张量边界**

```python
src = T.alloc_ub((64,), "float16")
dst = T.alloc_ub((64,), "float16")
max_ub = T.alloc_ub((64,), "float16")
T.tile.clamp(dst, src, -2.0, max_ub)  # 标量下界 + 张量上界
```
