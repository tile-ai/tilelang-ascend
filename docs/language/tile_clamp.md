# T.tile.clamp

## 1. 功能说明

将源 buffer 中的元素钳位到闭区间 `[min_scalar, max_scalar]`，并写入目标 buffer：
`out[i] = min(max(buffer[i], min_scalar), max_scalar)`。

小于下界的元素替换为 `min_scalar`，大于上界的元素替换为 `max_scalar`，区间内的元素保持不变。
支持输入和输出使用同一个 buffer 的原地计算。

> **注意区分**：`T.clamp` 是标量级运算，用于 `T.Parallel` 循环内的逐元素表达式；
> `T.tile.clamp` 是 buffer 级 intrinsic，对 Unified Buffer 中的一段连续区域执行计算。

## 2. 函数原型

### 2.1 函数定义

```python
def clamp(
    out: Buffer | BufferRegion,
    buffer: Buffer | BufferRegion,
    min_scalar: PrimExpr,
    max_scalar: PrimExpr,
    count: PrimExpr,
    *,
    tmp: Buffer | BufferRegion | None = None,
)
```

### 2.2 参数说明

| 参数名 | 输入/输出 | 描述 | 类型 | 必填/可选 |
|--------|----------|------|------|----------|
| out | 输出 | 存放钳位结果的 UB buffer，可与 `buffer` 相同 | 张量（tensor） | 必填 |
| buffer | 输入 | 源 UB buffer | 张量（tensor） | 必填 |
| min_scalar | 输入 | 闭区间下界，可转换为 `buffer` 的 dtype | 标量表达式（PrimExpr） | 必填 |
| max_scalar | 输入 | 闭区间上界，可转换为 `buffer` 的 dtype | 标量表达式（PrimExpr） | 必填 |
| count | 输入 | 从所选区域起始位置开始参与计算的元素个数 | 整数表达式（PrimExpr） | 必填 |
| tmp | 输入/输出 | PTO 路径可选的显式 UB 临时空间，仅能以关键字参数传入 | 张量（tensor）或 `None` | 可选，默认 `None` |

> **类型说明**：
> - **tensor**：通过 `T.alloc_ub` 分配的 Buffer，或其连续 BufferRegion。
> - **PrimExpr**：TileLang 编译期可表示的标量或整数表达式。

### 2.3 参数规格

#### 2.3.1 DataType 支持

| 平台 / 后端 | out | buffer | min_scalar / max_scalar |
|-------------|:---:|:------:|:-----------------------:|
| Ascend A2 / A3，AscendC | float16, float32, int16, int32 | float16, float32, int16, int32 | 可转换为 buffer dtype 的标量 |
| Ascend A2 / A3，PTO | float16, float32, int16, int32 | float16, float32, int16, int32 | 可转换为 buffer dtype 的标量 |

`out` 与 `buffer` 的 dtype 必须相同。

#### 2.3.2 Shape 支持

- 支持一维、二维 Buffer，以及其中的连续 BufferRegion。
- `out` 与 `buffer` 的 shape 和元素总数必须相同。

#### 2.3.3 count 与临时空间说明

- AscendC 按 `count` 处理所选区域的前缀元素，未处理的后缀保持不变。
- PTO 当前要求 `count` 等于所选 tile 的完整元素数，不支持 partial count。
- AscendC 使用基础 `Maxs` 和 `Mins` 指令组合，不需要临时空间。
- PTO 使用临时 tile 保存下界。`tmp=None` 时框架自动申请与所选源区域字节数相同的空间；
  显式传入 `tmp` 时，其 dtype 不参与计算，后端按 `buffer.dtype` 重新解释存储。

### 2.4 约束条件

1. `out` 与 `buffer` 必须具有相同的 dtype、shape 和可访问元素数。
2. `min_scalar` 必须小于或等于 `max_scalar`。
3. `count` 不得超过源区域或目标区域可访问的元素数；PTO 还要求 `count` 等于完整 tile 元素数。
4. 操作数位于 Unified Buffer，起始地址需满足 32 字节对齐要求（硬件约束）。
5. `out` 可以与 `buffer` 完全相同以执行原地计算；其他部分重叠方式不属于支持范围。
6. PTO 显式 `tmp` 必须是一维、静态、连续、固定宽度标量 dtype 的 `shared.ub` Buffer，
   或起始字节地址满足 32 字节对齐的 BufferRegion，容量不得小于所选源区域的字节数。
7. `tmp` 不得与 `out` 或 `buffer` 重叠。当前前端校验 `tmp` 的布局和起始地址，
   但不校验调用者提供的显式临时空间容量是否充足。

## 3. 示例代码

**示例 1：AscendC partial count**

```python
src = T.alloc_ub((64,), "float16")
dst = T.alloc_ub((64,), "float16")
T.tile.clamp(dst, src, 0.0, 6.0, 17)  # 只钳位前 17 个元素
```

**示例 2：PTO 完整 tile 与自动临时空间**

```python
values = T.alloc_ub((4, 16), "float32")
T.tile.clamp(values, values, -1.0, 1.0, 64)  # 原地计算，框架自动申请 PTO 临时空间
```

**示例 3：BufferRegion 与显式临时空间**

```python
src = T.alloc_ub((2, 64), "int16")
dst = T.alloc_ub((2, 64), "int16")
workspace = T.alloc_ub((64,), "int16")
T.tile.clamp(dst[1, :], src[0, :], -8, 7, 64, tmp=workspace)
```
