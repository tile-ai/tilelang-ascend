# T.tile.bitwise_xor

## 1. 功能说明

对两个操作数逐元素执行按位异或运算：`dst[i] = src0[i] ^ src1[i]`

内部委托给 `_call_intrin_with_optional_tmp("bitwise_xor", ...)` 实现。生成的底层指令取决于后端：

- **Ascend C 后端**：`AscendC::Xor<T>(dst, src0, src1, sharedTmpBuffer)`
- **PTO 后端**：`pto::TXOR(dst, src0, src1, tmp)`

> **复合实现**：CANN `AscendC::Xor` 在 A2/A3（`dav_c100`/`dav_m200`）和 A5（`dav_m300`）上均为复合实现，内部调用顺序为 `And → Or → Not → And`，即 `(x | y) & ~(x & y)`。不存在原生 `vxor` 指令。

## 2. 函数原型

### 2.1 函数定义

```python
def bitwise_xor(
    dst: Buffer | BufferRegion,
    src0: Buffer | BufferRegion,
    src1: Buffer | BufferRegion,
    *,
    tmp: Buffer | BufferRegion | None = None,
)
```

### 2.2 参数说明

| 参数名 | 输入/输出 | 描述 | 类型 | 必填/可选 |
|--------|----------|------|------|----------|
| dst | 输出 | 存放按位异或运算结果 | Buffer / BufferRegion | 必填 |
| src0 | 输入 | 第一个源操作数 | Buffer / BufferRegion | 必填 |
| src1 | 输入 | 第二个源操作数 | Buffer / BufferRegion | 必填 |
| tmp | 输入 | 临时缓冲区，1D UB Buffer（uint8） | Buffer / BufferRegion / None | 可选 |

> **类型说明**：
>
> - **Buffer**：通过 `T.alloc_ub`、`T.alloc_shared` 等分配的缓冲区，或其切片（BufferRegion）
> - **BufferRegion**：缓冲区切片，通过 `buf[0:M, 0:N]` 语法指定
> - **tmp**：可选的 1D UB 临时缓冲区。省略时由框架自动分配（大小为 `max(src_bytes, 64)` 字节）。传入时必须为一维缓冲区，dtype 由 lowering 重解释，语义上无意义
> - **src1 不支持标量**：`src1` 参数类型仅接受 `Buffer` / `BufferRegion`，不接受 `PrimExpr` / `int` / `float`。传入标量会触发 `AttributeError`

### 2.3 参数规格

#### 2.3.1 Buffer Scope

`bitwise_xor` 的 `dst`、`src0`、`src1` 必须位于 **UB（Unified Buffer）** 内存（通过 `T.alloc_ub` 分配）。`tmp` 同样必须为 UB 缓冲区。底层 `AscendC::Xor` 和 PTO `TXOR` 均操作 `LocalTensor<T>`（UB）。

#### 2.3.2 DataType 支持

以下 dtype 基于 Ascend A2 / A3（910B3）真机验证：

| dtype | Ascend C | PTO |
|-------|----------|-----|
| int8 | 不支持（`static_assert`） | 支持 |
| uint8 | 不支持（`static_assert`） | 编译通过但运行时报 "ub address out of bounds" |
| int16 | 支持 | 支持 |
| uint16 | 支持 | 支持 |
| int32 | segfault（TVM `OptimizeForTarget`） | segfault（TVM `OptimizeForTarget`） |
| uint32 | segfault（TVM `OptimizeForTarget`） | segfault（TVM `OptimizeForTarget`） |
| float16 | 不支持（`static_assert`） | 未验证 |
| float32 | 不支持（`static_assert`） | 未验证 |

> **说明**：
>
> - **int16 / uint16**：CANN `AscendC::Xor` 的 `static_assert` 仅允许 `int16_t` / `uint16_t`（见 `xor.h` 第 48 行），这是唯一在 Ascend C 后端上完全可用的 dtype。PTO 后端的 `TXOR` 通过 `ElementOpCal<DType, OP_XOR>::apply` 执行 `dst = src0 ^ src1`，无类型限制，int16/uint16 均通过真机验证。
> - **int8（PTO）**：PTO 后端 `TXOR` 无 `static_assert`，int8 编译并运行通过。Ascend C 后端因 `static_assert` 在编译期拒绝。
> - **uint8（PTO）**：PTO 后端编译通过，但运行时报 "VEC instruction error: the ub address out of bounds"。原因是 `XorCodegen` 将 tmp 的 row/col 继承自 dst，但 `sizeof(uint8)=1` 与 `sizeof(int16)=2` 的差异导致 tmp 字节数不足。
> - **int32 / uint32**：在 TVM `OptimizeForTarget` pass 中触发 segfault（非可捕获异常），Ascend C 和 PTO 后端均如此。CANN `xor.h` 的 `static_assert` 也会在 C++ 编译期拒绝 int32/uint32（若 TVM pass 不先崩溃）。
> - **float 类型**：按位异或对浮点数据无意义，CANN `static_assert` 会拒绝。
> - **A5 平台**：CANN `xor.h` 的 `static_assert` 在所有架构（`__NPU_ARCH__ == 2201 || 2002 || 3002`）上均仅允许 `int16_t` / `uint16_t`。A5（v300）实现（`xor_v300_impl.h`）同样为复合实现，不存在原生 `vxor` 指令。当前环境（A2/A3）无法验证 A5 真机行为。

#### 2.3.3 Shape 支持

- 支持 1D 和 2D
- size 由 buffer shape 自动推断（BufferRegion 时取 region extent 的乘积，Buffer 时取 shape 的乘积）
- 支持非对齐 size（如 200 个 int16 元素），内部按 `stackSize` 分块处理

### 2.4 约束条件

1. `bitwise_xor` 委托给 `_call_intrin_with_optional_tmp("bitwise_xor", [dst_ptr, src0_ptr, src1_ptr], 3, tmp)` 实现
2. `dst`、`src0`、`src1` 必须位于 UB 内存（通过 `T.alloc_ub` 分配）
3. `dst` 与 `src0` 的 size（元素总数）必须相同
4. `src1` 的 size 也必须与 `dst` 相同
5. `src0` 和 `src1` 的 dtype 必须与 `dst` 一致（C++ 模板参数 `T` 必须统一）
6. 仅支持整数类型（int16/uint16），不支持浮点类型（CANN `static_assert`）
7. `src1` 不支持标量（scalar），仅支持 Buffer / BufferRegion
8. `tmp` 为可选参数：省略时框架自动分配临时缓冲区（Ascend C: `max(src_bytes, 64)` 字节；PTO: `src_bytes` 字节）；传入时必须为一维 UB Buffer
9. A2/A3 和 A5 上均为复合实现（`And → Or → Not → And`），`dst` / `src0` / `src1` / `tmp` 四个操作数地址不得重叠
10. 操作数地址需 32 字节对齐（硬件约束）

## 3. 示例代码

**示例 1：tensor-tensor 按位异或**

```python
src0 = T.alloc_ub((256,), "int16")
src1 = T.alloc_ub((256,), "int16")
dst = T.alloc_ub((256,), "int16")
T.tile.bitwise_xor(dst, src0, src1)
```

**示例 2：BufferRegion 切片按位异或**

```python
src0 = T.alloc_ub((128, 256), "int16")
src1 = T.alloc_ub((128, 256), "int16")
dst = T.alloc_ub((128, 256), "int16")
T.tile.bitwise_xor(dst[0:128, 0:256], src0[0:128, 0:256], src1[0:128, 0:256])
```

**示例 3：显式 tmp 缓冲区**

```python
src0 = T.alloc_ub((128, 256), "int16")
src1 = T.alloc_ub((128, 256), "int16")
dst = T.alloc_ub((128, 256), "int16")
tmp = T.alloc_ub((128 * 256 * 2,), "uint8")  # 1D, uint8, >= src_bytes
T.tile.bitwise_xor(dst, src0, src1, tmp=tmp)
```

## 4. 已知限制

### 4.1 Ascend C 仅支持 int16/uint16

CANN `AscendC::Xor`（`xor.h`）在所有架构上均有 `static_assert` 限制 T 为 `int16_t` / `uint16_t`。int8/uint8/int32/uint32/float 等类型在 Ascend C 后端编译期被拒绝。

### 4.2 int32/uint32 触发 TVM segfault

int32/uint32 在 TVM `OptimizeForTarget` pass 中触发 segfault（非可捕获异常），Ascend C 和 PTO 后端均如此。这不是 CANN `static_assert` 的行为，而是 TVM 内部 pass 的缺陷。

### 4.3 uint8 PTO 运行时越界

uint8 在 PTO 后端编译通过，但运行时报 "VEC instruction error: ub address out of bounds"。原因是 PTO `XorCodegen` 将 tmp 的 row/col 继承自 dst，但 `sizeof(uint8)=1` 与 `sizeof(int16)=2` 的差异导致 tmp 字节数不足。

### 4.4 A5 平台声明更正

原始文档声称 A5 支持 int8/uint8/int16/uint16/int32/uint32 且有原生 vxor 指令。实际 CANN 头文件（`xor.h`、`xor_v300_impl.h`）显示：
- `static_assert` 在所有架构上仅允许 `int16_t` / `uint16_t`
- A5（v300）实现同样为复合实现（`And → Or → Not → And`），不存在原生 `vxor` 指令

该声明未经真机验证且与头文件矛盾，已更正。
