# T.tile.bitwise_lshift

## 1. 功能说明

对 buffer 中每个元素执行标量左移运算：`dst[i] = src0[i] << scalarValue`

底层委托给 `tir.call_intrin("tl.ascend_bitwise_lshift", ...)` 实现。生成的底层指令取决于后端：

- **Ascend C 后端**：`AscendC::ShiftLeft<T>(dst, src, scalarValue, count)`
- **PTO 后端**：`TSHLS(dst, src, scalar)`

> **移位语义说明**：无论有符号还是无符号类型，左移均为**逻辑左移**——高位丢弃，低位补 0。底层 `vshls` / `vshl` 指令不区分符号类型，符号位不会被特殊保留。

## 2. 函数原型

### 2.1 函数定义

```python
def bitwise_lshift(
    dst: Buffer | BufferRegion,
    src0: Buffer | BufferRegion,
    scalarValue: PrimExpr,
)
```

### 2.2 参数说明

| 参数名 | 输入/输出 | 描述 | 类型 | 必填/可选 |
|--------|----------|------|------|----------|
| dst | 输出 | 存放左移运算结果 | Buffer / BufferRegion | 必填 |
| src0 | 输入 | 源操作数 | Buffer / BufferRegion | 必填 |
| scalarValue | 输入 | 位移位数（标量） | PrimExpr / Python 标量 | 必填 |

> **类型说明**：
>
> - **Buffer**：通过 `T.alloc_ub`、`T.alloc_shared` 等分配的缓冲区
> - **BufferRegion**：缓冲区切片，通过 `buf[0:M, 0:N]` 语法指定
> - **scalarValue**：Python 标量或 PrimExpr，表示位移位数。codegen 会在 scalarValue 的 dtype 与 src0 不一致时自动插入类型转换（如 `int16(scalarValue)`），因此 Python 层面不强制要求 scalarValue 的 dtype 与 dst 一致。

### 2.3 参数规格

#### 2.3.1 Buffer Scope

`bitwise_lshift` 仅支持 **UB（Unified Buffer）** 内存。底层 `AscendC::ShiftLeft` 和 PTO `TSHLS` 均操作 `LocalTensor<T>`（UB）。

#### 2.3.2 DataType 支持

以下 dtype 基于 Ascend A2 / A3（910B3）真机验证：

| dtype | Ascend C | PTO |
|-------|----------|-----|
| int8 | 不支持（编译失败） | 不支持（编译失败） |
| uint8 | 不支持（编译失败） | 不支持（编译失败） |
| int16 | 支持 | 支持 |
| uint16 | 支持 | 支持 |
| int32 | 支持 | 支持 |
| uint32 | 支持 | 支持 |
| int64 | 不支持（编译失败） | 不支持（编译失败） |
| uint64 | 不支持（编译失败） | 不支持（编译失败） |
| float16 | 不支持（编译失败） | 不支持（编译失败） |
| float32 | 不支持（编译失败） | 不支持（编译失败） |

> **说明**：
>
> - **Ascend C 后端**：`ShiftLeftImpl` 仅有 `int16_t`、`uint16_t`、`int32_t`、`uint32_t` 的特化版本。其余类型命中通用模板，触发 `ASCENDC_ASSERT(false, ...)` 编译失败。
> - **PTO 后端（A2/A3）**：`TShiftCheck` 的 `static_assert` 仅允许 `int32_t`、`int`、`int16_t`、`uint32_t`、`uint16_t`、`unsigned int`。其余类型触发编译失败。
> - **A5 平台**：PTO a5 `TSHLS_IMPL` 的 `static_assert` 允许 `int8_t`、`uint8_t`、`int16_t`、`uint16_t`、`int32_t`、`uint32_t`。但 **不包含 `int64_t` / `uint64_t`**。当前环境（A2/A3）无法验证 A5 行为。

#### 2.3.3 Shape 支持

- 支持 1D 和 2D
- size 由 buffer shape 自动推断（BufferRegion 时取 region extent 的乘积，Buffer 时取 shape 的乘积）

### 2.4 约束条件

1. `bitwise_lshift` 委托给 `tir.call_intrin("tl.ascend_bitwise_lshift", ...)` 实现
2. buffer 必须位于 UB 内存（通过 `T.alloc_ub` 分配）
3. dst 与 src0 的 size（元素总数）必须相同（源码内含 `assert size_0 == size_2`）
4. src0 的 dtype 必须与 dst 一致（C++ 模板参数 `T` 必须统一）
5. 仅支持整数类型（int16/uint16/int32/uint32），不支持浮点类型
6. scalarValue 为标量（PrimExpr），不支持 tensor-tensor 位移；其 dtype 不要求与 dst 一致（codegen 自动转换）
7. 操作数地址需 32 字节对齐（硬件约束）

## 3. 示例代码

**示例 1：标量左移（2D）**

```python
src0 = T.alloc_ub((64, 256), "int16")
dst = T.alloc_ub((64, 256), "int16")
T.tile.bitwise_lshift(dst, src0, 2)  # dst[i] = src0[i] << 2
```

**示例 2：1D 标量左移**

```python
src0 = T.alloc_ub((1024,), "int32")
dst = T.alloc_ub((1024,), "int32")
T.tile.bitwise_lshift(dst, src0, 4)  # dst[i] = src0[i] << 4
```

**示例 3：BufferRegion 切片左移**

```python
src0 = T.alloc_ub((64, 256), "int32")
dst = T.alloc_ub((64, 256), "int32")
for i in range(64):
    T.tile.bitwise_lshift(dst[i, :], src0[i, :], 1)
```

## 4. 已知限制

### 4.1 非对齐 shape 精度问题

当 tile 的元素总数不满足 32 字节对齐时（例如 shape=(1024, 100)），Ascend C 后端产生精度错误（约 3.8%~10.7% 元素不正确）。PTO 后端不受此影响。

### 4.2 int8 / uint8 / int64 / uint64 不支持

原始文档声称 A5 支持 int64 / uint64。经核查，PTO a5 `TSHLS_IMPL` 的 `static_assert` 仅允许 int8/uint8/int16/uint16/int32/uint32，**不包含 int64 / uint64**。A2/A3 的 PTO `TShiftCheck` 和 Ascend C `ShiftLeftImpl` 均不支持 int8/uint8/int64/uint64。

### 4.3 移位语义

原始文档声称"有符号类型执行算术左移（次高位丢弃，低位补 0）"。经真机验证，左移为**逻辑左移**：无论有符号或无符号类型，高位丢弃、低位补 0，符号位不被特殊保留。底层 `vshls` / `vshl` 指令不区分符号类型。

### 4.4 A5 平台支持范围

原始文档声称 A5 支持 int8/uint8/int16/uint16/int32/uint32/int64/uint64。经核查 PTO a5 `static_assert`，实际允许 int8/uint8/int16/uint16/int32/uint32，**不包含 int64/uint64**。该声明未经真机验证。
