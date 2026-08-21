# T.tile.bitwise_or

## 1. 功能说明

对两个操作数逐元素执行按位或运算：`dst[i] = src0[i] | src1[i]`

内部委托给 `binary_op(dst, src0, src1, "bitwise_or")` 实现。生成的底层指令取决于后端：

- **Ascend C 后端**：`AscendC::Or<T>(dst, src0, src1, count)`
- **PTO 后端**：`pto::TOR(dst, src0, src1)`

> **scalar 路径不可用**：当 `src1` 为标量（`PrimExpr` / `int` / `float`）时，`binary_op` 会生成 `tl.ascend_bitwise_ors` 调用，但该算子在 C++ 层未注册（`src/op/ascend.cc` 中无 `TIR_DEFINE_TL_BUILTIN(ascend_bitwise_ors)`），编译期会报 `Operator tl.ascend_bitwise_ors is not registered`。因此 `src1` 目前仅支持 Buffer / BufferRegion / BufferLoad。

## 2. 函数原型

### 2.1 函数定义

```python
def bitwise_or(
    dst: Buffer | BufferRegion,
    src0: Buffer | BufferRegion,
    src1: Buffer | BufferRegion | BufferLoad | PrimExpr,
)
```

### 2.2 参数说明

| 参数名 | 输入/输出 | 描述 | 类型 | 必填/可选 |
|--------|----------|------|------|----------|
| dst | 输出 | 存放按位或运算结果 | Buffer / BufferRegion | 必填 |
| src0 | 输入 | 第一个源操作数 | Buffer / BufferRegion | 必填 |
| src1 | 输入 | 第二个源操作数 | Buffer / BufferRegion / BufferLoad | 必填 |

> **类型说明**：
>
> - **Buffer**：通过 `T.alloc_ub`、`T.alloc_shared` 等分配的缓冲区，或其切片（BufferRegion）
> - **BufferRegion**：缓冲区切片，通过 `buf[0:M, 0:N]` 语法指定
> - **BufferLoad**：buffer 元素访问（如 `buf[i]`），走 `tl.ascend_bitwise_ors` 路径
> - **PrimExpr / 标量**：函数签名中声明了 `PrimExpr`，但标量路径（`tl.ascend_bitwise_ors`）当前未注册，实际不可用。详见第 4 节约束条件。

### 2.3 参数规格

#### 2.3.1 Buffer Scope

`bitwise_or` 仅支持 **UB（Unified Buffer）** 内存。底层 `AscendC::Or` 和 PTO `TOR` 均操作 `LocalTensor<T>`（UB）。

#### 2.3.2 DataType 支持

以下 dtype 基于 Ascend A2 / A3（910B3）真机验证：

| dtype | Ascend C | PTO |
|-------|----------|-----|
| int8 | 支持 | 支持 |
| uint8 | 支持 | 支持 |
| int16 | 支持 | 支持 |
| uint16 | 支持 | 支持 |
| int32 | 编译通过但结果错误（50% 数据未处理） | 不支持（static_assert） |
| uint32 | 编译通过但结果错误（50% 数据未处理） | 不支持（static_assert） |
| int64 | 编译通过但结果错误（75% 数据未处理） | 不支持 |
| float16 | 编译通过但结果无意义 | 不支持 |
| float32 | 编译通过但结果无意义 | 不支持 |

> **说明**：
>
> - **int8 / uint8**：Ascend C 后端的 `AscendC::Or` count 模式通过 `OrIntrinsicsImpl` 仅有 `int16_t / uint16_t` 特化，但 `vor` 指令按 int16 重解释后仍能正确处理 int8/uint8 数据（已通过真机精度验证）。PTO 后端的 `TOR` 通过 `static_assert(sizeof(T)==2 || sizeof(T)==1)` 允许 int8/uint8。
> - **int32 / uint32**：Ascend C 后端 `OrImpl` count 模式会将 int32 数据按 int16 重解释，但 `set_vector_mask(0, count)` 中的 `count` 是 int32 元素个数，导致 mask 只覆盖一半数据，产生 50% 的错误结果。PTO 后端在编译期通过 `static_assert(sizeof(T)==2||sizeof(T)==1)` 直接拒绝。
> - **int64**：同理，int64 按 int16 重解释后 mask 只覆盖 1/4 数据，产生 75% 的错误结果。
> - **float 类型**：按位或对浮点数据无意义，会产生垃圾结果。
> - **A5 平台**：CANN 头文件（dav_m300）`OrImpl` 仅有 `int16_t / uint16_t` 特化，其他类型触发 `ASCENDC_ASSERT(false)`。当前环境（A2/A3）无法验证 A5 行为。

#### 2.3.3 Shape 支持

- 支持 1D 和 2D
- size 由 buffer shape 自动推断（BufferRegion 时取 region extent 的乘积，Buffer 时取 shape 的乘积）

### 2.4 约束条件

1. `bitwise_or` 委托给 `binary_op(dst, src0, src1, "bitwise_or")` 实现
2. buffer 必须位于 UB 内存（通过 `T.alloc_ub` 分配）
3. dst 与 src0 的 size（元素总数）必须相同（`binary_op` 内含 `assert size_0 == size_1`）
4. 当 src1 为 BufferRegion 时，其 size 也必须与 dst 相同
5. src0 和 src1 的 dtype 必须与 dst 一致（C++ 模板参数 `T` 必须统一）
6. 仅支持整数类型（int8/uint8/int16/uint16），不支持浮点类型
7. 操作数地址需 32 字节对齐（硬件约束）
8. **标量 src1 不可用**：当 `src1` 为 `PrimExpr` / `int` / `float` 时，`binary_op` 生成 `tl.ascend_bitwise_ors`，但该算子未在 C++ 层注册，编译会报 `Operator tl.ascend_bitwise_ors is not registered`

## 3. 示例代码

**示例 1：tensor-tensor 按位或**

```python
src0 = T.alloc_ub((256,), "int16")
src1 = T.alloc_ub((256,), "int16")
dst = T.alloc_ub((256,), "int16")
T.tile.bitwise_or(dst, src0, src1)
```

**示例 2：BufferRegion 切片按位或**

```python
src0 = T.alloc_ub((128, 256), "int16")
src1 = T.alloc_ub((128, 256), "int16")
dst = T.alloc_ub((128, 256), "int16")
T.tile.bitwise_or(dst[0:128, 0:256], src0[0:128, 0:256], src1[0:128, 0:256])
```

**示例 3：int8 按位或**

```python
src0 = T.alloc_ub((256,), "int8")
src1 = T.alloc_ub((256,), "int8")
dst = T.alloc_ub((256,), "int8")
T.tile.bitwise_or(dst, src0, src1)
```

## 4. 已知限制

### 4.1 标量 src1 不可用

原始文档示例中包含 `T.tile.bitwise_or(dst, src0, 0x0F)` 的 tensor-scalar 用法，但该路径在 C++ 层未注册（`src/op/ascend.cc` 中无 `TIR_DEFINE_TL_BUILTIN(ascend_bitwise_ors)`），编译期会报错。如需标量按位或，需先通过 `T.tile.fill` 将标量填入 buffer，再调用 tensor-tensor 版本。

### 4.2 int32/uint32 结果错误

Ascend C 后端对 int32/uint32 可编译但产生 50% 错误结果（CANN `OrImpl` count 模式的 mask 计算基于 `sizeof(T)`，将 int32 按 int16 重解释后 mask 只覆盖一半数据）。PTO 后端通过 `static_assert` 在编译期拒绝。

### 4.3 A5 平台支持范围

原始文档声称 A5 支持 int8/uint8/int16/uint16/int32/uint32/int64/uint64，但 CANN 头文件（dav_m300）`OrImpl` 仅有 `int16_t / uint16_t` 特化，其他类型触发 `ASCENDC_ASSERT(false)`。该声明未经真机验证且与头文件矛盾。
