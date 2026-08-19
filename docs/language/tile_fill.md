# T.tile.fill

## 1. 功能说明

将 buffer 中的所有元素填充为指定标量值：`buffer[i] = value`

生成的底层指令取决于后端：

- **Ascend C 后端**：`AscendC::Duplicate<T>(dst, value, count)`（通过 `tl::ascend::Fill<T>` 模板调用）
- **PTO 后端**：`TEXPANDS(dst, value)`

`T.tile.clear(buf)` 内部委托给 `fill(buf, 0)` 实现。

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
| buffer | 输入/输出 | 待填充的 buffer | Buffer / BufferRegion | 必填 |
| value | 输入 | 填充的标量值 | Python 标量或 PrimExpr | 必填 |

> **类型说明**：
>
> - **Buffer**：通过 `T.alloc_ub`、`T.alloc_shared` 等分配的片上缓冲区
> - **BufferRegion**：Buffer 的切片，fill 将仅填充切片覆盖的区域
> - **value**：可以是 Python 标量（如 `10.0`、`5`）或 PrimExpr（如 `T.cast(10, "float32")`）。当 value 的 dtype 与 buffer 不一致时，前端会自动将其 Cast 到 buffer 的 dtype

### 2.3 参数规格

#### 2.3.1 Buffer Scope

`fill` 支持 **UB（Unified Buffer）** 和 **shared（L1）** 内存。底层 `AscendC::Duplicate` 和 PTO `TEXPANDS` 均操作 `LocalTensor<T>`。

> **说明**：fill 直接操作片上 buffer 的 access_ptr，不区分 UB 和 shared scope。`T.tile.clear` 同样通过 `fill` 实现，因此 clear 也支持 UB 和 shared。

#### 2.3.2 DataType 支持

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

> **说明**：Ascend C 后端 dtype 支持范围由 `AscendC::Duplicate` 的 `static_assert` 约束（dav_c220 平台），不支持 int8/uint8。PTO 后端使用 `TEXPANDS`，通过 `B82B16Trait` 将 B8 类型转换为 B16，支持 int8/uint8。
>
> **A5 平台**：int64/uint64 在当前环境（A2/A3）无法验证。`AscendC::Duplicate` 的 dav_c220 `static_assert` 不包含 int64/uint64，Ascend C 后端预计不支持。

#### 2.3.3 Shape 支持

- 支持 1D 和 2D
- size 由 buffer shape 自动推断（BufferRegion 时取 region extent 的乘积，Buffer 时取 shape 的乘积）
- 无需显式传入 count 参数；Ascend C 后端将推断的 size 作为 count 传入 `Duplicate`，PTO 后端根据 tile 的 valid_row/valid_col 自行推断填充范围

### 2.4 约束条件

1. value 的 dtype 不需要与 buffer 的 dtype 一致：当两者不同时，前端自动将 value Cast 到 buffer 的 dtype
2. buffer 地址需 32 字节对齐（硬件约束）
3. size 由 buffer shape 自动推断，无需显式传入 count 参数
4. fill 支持 UB 和 shared 内存；fragment（L0C/寄存器）级别的 fill 行为未经完整验证
5. 仅支持片上 buffer fill，GM 级别 fill 需用 T.copy

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

**示例 3：与 clear 的等价关系**

```python
buf = T.alloc_ub((256,), "float16")
T.tile.fill(buf, 0.0)   # 填充为 0.0
T.tile.clear(buf)        # 等价写法：清零（内部调用 fill(buf, 0)）
```

**示例 4：填充 BufferRegion 切片**

```python
a_ub = T.alloc_ub((block_M, block_N), "float32")
T.tile.fill(a_ub, 10.0)                        # 全部填充为 10.0
T.tile.fill(a_ub[0:block_M // 2, 0:block_N], 5.0)  # 仅前半部分填充为 5.0
```
