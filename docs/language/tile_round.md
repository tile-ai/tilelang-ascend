# T.tile.round

## 1. 功能说明

将源 buffer 中的浮点数逐元素舍入到最接近的整数值，并以原浮点数据类型写入目标 buffer：`out[i] = round(buffer[i])`。

当数值恰好位于两个整数中间时采用向偶数舍入（ties-to-even），例如 `2.5 → 2`、`3.5 → 4`、`-2.5 → -2`。支持输入和输出使用同一个 buffer 的原地计算。

## 2. 函数原型

### 2.1 函数定义

```python
def round(
    out: Buffer | BufferRegion,
    buffer: Buffer | BufferRegion,
    count: PrimExpr,
    *,
    tmp: Buffer | BufferRegion | None = None,
)
```

### 2.2 参数说明

| 参数名 | 输入/输出 | 描述 | 类型 | 必填/可选 |
|--------|----------|------|------|----------|
| out | 输出 | 存放舍入结果的 UB buffer，可与 `buffer` 相同 | 张量（tensor） | 必填 |
| buffer | 输入 | 源 UB buffer | 张量（tensor） | 必填 |
| count | 输入 | 从所选区域起始位置开始参与计算的元素个数 | 整数表达式（PrimExpr） | 必填 |
| tmp | 输入/输出 | AscendC float16 路径可选的显式 UB 临时空间，仅能以关键字参数传入 | 张量（tensor）或 `None` | 可选，默认 `None` |

> **类型说明**：
> - **tensor**：通过 `T.alloc_ub` 分配的 Buffer，或其连续 BufferRegion。
> - **PrimExpr**：正整数表达式，用于指定参与计算的元素个数。

### 2.3 参数规格

#### 2.3.1 DataType 支持

| 平台 / 后端 | out | buffer |
|-------------|:---:|:------:|
| Ascend A2 / A3，AscendC | float16, float32 | float16, float32 |
| Ascend A2 / A3，PTO | float32 | float32 |

`out` 与 `buffer` 的 dtype 必须相同。PTO 当前不支持 float16 round。

#### 2.3.2 Shape 支持

- 支持一维、二维 Buffer，以及其中的连续 BufferRegion。
- `out` 与 `buffer` 的 shape 和元素总数必须相同。
- `count` 不得超过源区域或目标区域可访问的元素数。
- AscendC 可用 `count` 处理所选区域的前缀元素；PTO 当前要求 `count` 等于所选 tile 的完整元素数。

### 2.4 约束条件

1. 只执行舍入到整数值，不支持指定小数位；结果仍使用输入浮点 dtype 保存。
2. 中间值采用向偶数舍入，而不是统一远离零舍入。
3. 操作数位于 Unified Buffer，起始地址需满足 32 字节对齐要求。
4. `out` 可以与 `buffer` 完全相同以执行原地计算；其他部分重叠方式不属于支持范围。
5. AscendC float16 使用临时空间。`tmp=None` 时框架自动申请 `max(256, source_access_bytes)` 字节；显式传入时，调用者应提供不少于该大小的空间。
6. 显式 `tmp` 必须是一维、静态、连续、固定宽度标量 dtype 的 `shared.ub` Buffer，或起始字节地址满足 32 字节对齐的 BufferRegion。其 dtype 不参与计算，后端按字节地址重新解释存储。
7. `tmp` 不得与 `out` 或 `buffer` 重叠。当前前端校验 `tmp` 的布局和起始地址，但不校验其容量是否充足。
8. AscendC float32 和 PTO 不需要临时空间；这些路径会移除 `tmp` 操作数。

## 3. 示例代码

**示例 1：自动临时空间与原地舍入**

```python
values = T.alloc_ub((64,), "float16")
T.tile.round(values, values, 64)
```

**示例 2：BufferRegion 与显式临时空间**

```python
src = T.alloc_ub((2, 64), "float16")
dst = T.alloc_ub((2, 64), "float16")
workspace = T.alloc_ub((256,), "uint8")
T.tile.round(dst[1, :], src[0, :], 64, tmp=workspace)
```

**示例 3：PTO float32**

```python
src = T.alloc_ub((4, 16), "float32")
dst = T.alloc_ub((4, 16), "float32")
T.tile.round(dst, src, 64)
```
