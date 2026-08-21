# T.tile.bitwise_not

## 1. 功能说明

对操作数逐元素执行按位取反运算：`dst[i] = ~src[i]`

内部委托给 `unary_op(dst, src0, "bitwise_not")` 实现。生成的底层指令取决于后端：

- **Ascend C 后端**：`AscendC::Not<T>(dst, src, count)`
- **PTO 后端**：`pto::TNOT(dst, src)`

## 2. 函数原型

### 2.1 函数定义

```python
def bitwise_not(
    dst: Buffer | BufferRegion,
    src0: Buffer | BufferRegion,
)
```

### 2.2 参数说明

| 参数名 | 输入/输出 | 描述 | 类型 | 必填/可选 |
|--------|----------|------|------|----------|
| dst | 输出 | 存放按位取反运算结果 | Buffer / BufferRegion | 必填 |
| src0 | 输入 | 源操作数 | Buffer / BufferRegion | 必填 |

> **类型说明**：
>
> - **Buffer**：通过 `T.alloc_ub`、`T.alloc_shared` 等分配的缓冲区
> - **BufferRegion**：缓冲区切片，通过 `buf[0:M, 0:N]` 语法指定

### 2.3 参数规格

#### 2.3.1 Buffer Scope

`bitwise_not` 仅支持 **UB（Unified Buffer）** 内存。底层 `AscendC::Not` 和 PTO `TNOT` 均操作 `LocalTensor<T>`（UB）。

#### 2.3.2 DataType 支持

以下 dtype 基于 Ascend A2 / A3（910B3）真机验证：

| dtype | Ascend C | PTO |
|-------|----------|-----|
| int8 | 不支持（编译失败） | 支持 |
| uint8 | 不支持（编译失败） | 支持 |
| int16 | 支持 | 支持 |
| uint16 | 支持 | 支持 |
| int32 | 不支持（编译失败） | 不支持（编译失败） |
| uint32 | 不支持（编译失败） | 不支持（编译失败） |
| float16 | 不支持（编译失败） | 不支持（编译失败） |
| float32 | 不支持（编译失败） | 不支持（编译失败） |

> **说明**：
>
> - **int8 / uint8**：Ascend C 后端的 `NotImpl` 直接将原始类型传给 `vnot` 指令，而 `vnot` 仅接受 `__ubuf__ short *`（int16_t），因此编译失败。PTO 后端的 `TNOT_IMPL` 通过 `B82B16Trait` 将 int8/uint8 宽展为 int16 后再调用 `vnot`，因此能正确处理（已通过真机精度验证）。
> - **int32 / uint32**：两个后端均编译失败。Ascend C 的 `vnot` 不接受 int32 类型指针；PTO 的 `B82B16Trait` 仅对 1 字节类型做宽展，int32 不做转换，同样导致 `vnot` 拒绝。
> - **float 类型**：`vnot` 指令不支持浮点类型，两个后端均编译失败。
> - **A5 平台**：PTO a5 `TNOT_IMPL` 包含 `static_assert` 允许 int8/uint8/int16/uint16/int32/uint32。CANN 头文件中 `NotImpl` 为无约束模板，但实际 dtype 支持取决于 `vnot` 指令。当前环境（A2/A3）无法验证 A5 行为。

#### 2.3.3 Shape 支持

- 支持 2D shape
- size 由 buffer shape 自动推断（BufferRegion 时取 region extent 的乘积，Buffer 时取 shape 的乘积）

### 2.4 约束条件

1. `bitwise_not` 委托给 `unary_op(dst, src0, "bitwise_not")` 实现
2. buffer 必须位于 UB 内存（通过 `T.alloc_ub` 分配）
3. dst 与 src0 的 size（元素总数）必须相同（`unary_op` 内含 `assert size_0 == size_1`）
4. src0 的 dtype 必须与 dst 一致（C++ 模板参数 `T` 必须统一）
5. 仅支持整数类型（int8/uint8/int16/uint16），不支持浮点类型
6. 操作数地址需 32 字节对齐（硬件约束）
7. **非对齐 shape 精度问题**：当 tile 的元素总数不是 32 字节对齐时，Ascend C 后端可能产生精度错误（部分元素未被正确处理）。PTO 后端不受此影响。

## 3. 示例代码

**示例 1：按位取反**

```python
src0 = T.alloc_ub((256,), "int16")
dst = T.alloc_ub((256,), "int16")
T.tile.bitwise_not(dst, src0)  # dst[i] = ~src0[i]
```

**示例 2：BufferRegion 切片按位取反**

```python
src0 = T.alloc_ub((128, 256), "int16")
dst = T.alloc_ub((128, 256), "int16")
T.tile.bitwise_not(dst[0:128, 0:256], src0[0:128, 0:256])
```

**示例 3：int8 按位取反（仅 PTO 后端）**

```python
src0 = T.alloc_ub((256,), "int8")
dst = T.alloc_ub((256,), "int8")
T.tile.bitwise_not(dst, src0)  # 仅 PTO 后端支持
```

## 4. 已知限制

### 4.1 int8/uint8 仅 PTO 支持

Ascend C 后端的 `NotImpl` 直接将模板类型 `T` 传给 `vnot` 指令，而 `vnot` 仅接受 `__ubuf__ short *`（int16_t）。因此 int8/uint8 在 Ascend C 后端编译失败。PTO 后端通过 `B82B16Trait` 将 int8/uint8 宽展为 int16 后再调用 `vnot`，可正确处理。

### 4.2 int32/uint32 不支持

原始文档声称"A2/A3 上 int32/uint32 需通过 ReinterpretCast 转为 int16/uint16 后调用"，但实际两个后端均无法编译 int32/uint32。Ascend C 的 `vnot` 不接受 int32 类型指针；PTO 的 `B82B16Trait` 不对 4 字节类型做转换。如需对 int32 数据执行按位取反，需先将数据加载为 int16 buffer，再调用 `bitwise_not`。

### 4.3 非对齐 shape 精度问题

当 tile 的元素总数不满足 32 字节对齐时（例如 shape=(1024, 100)），Ascend C 后端可能产生精度错误（约 10.7% 元素不正确）。PTO 后端不受此影响。

### 4.4 A5 平台支持范围

原始文档声称 A5 支持 int8/uint8/int16/uint16/int32/uint32。PTO a5 `TNOT_IMPL` 的 `static_assert` 确实允许这些类型。但 CANN 头文件中 `NotImpl` 为无约束模板，实际 dtype 支持取决于 `vnot` 指令对 `__ubuf__ short *` 的要求。该声明未经真机验证。
