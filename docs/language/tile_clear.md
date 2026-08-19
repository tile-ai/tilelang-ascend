# T.tile.clear

## 1. 功能说明

将 buffer 中的所有元素清零：`buffer[i] = 0`

内部委托给 `fill(buffer, 0)` 实现。生成的底层指令取决于后端：

- **Ascend C 后端**：`AscendC::Duplicate<T>(dst, 0, count)`
- **PTO 后端**：`TEXPANDS(dst, 0)`

## 2. 函数原型

### 2.1 函数定义

```python
def clear(
    buffer: Buffer | tir.Var,
)
```

### 2.2 参数说明

| 参数名 | 输入/输出 | 描述 | 类型 | 必填/可选 |
|--------|----------|------|------|---------|
| buffer | 输入/输出 | 待清零的 buffer | Buffer / BufferRegion / tir.Var | 必填 |

> **类型说明**：
>
> - **Buffer**：通过 `T.alloc_ub` 分配的 UB 缓冲区。`fill` 也接受 `BufferRegion`（切片），因此传入 `BufferRegion` 同样可用，但函数签名中的类型标注为 `Buffer | tir.Var`。
> - **tir.Var**：当传入 tir.Var 时，`clear` 会通过 `T.has_let_value` / `T.get_let_value` 解析为 BufferRegion 再调用 `fill`。

### 2.3 参数规格

#### 2.3.1 Buffer Scope

`clear` 仅支持 **UB（Unified Buffer）** 内存。底层 `AscendC::Duplicate` 和 PTO `TEXPANDS` 均操作 `LocalTensor<T>`（UB）。

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

> **说明**：Ascend C 后端 dtype 支持范围由 `AscendC::Duplicate` 的 `static_assert` 约束，不支持 int8/uint8。PTO 后端使用 `TEXPANDS`，支持 int8/uint8。
>
> **A5 平台**：int64/uint64 在当前环境（A2/A3）无法验证。`IsDuplicateSupported_v` 不包含 int64/uint64，Ascend C 后端预计不支持。

#### 2.3.3 Shape 支持

- 支持 1D 和 2D
- size 由 buffer shape 自动推断（BufferRegion 时取 region extent 的乘积，Buffer 时取 shape 的乘积）

### 2.4 约束条件

1. `clear` 委托给 `fill(buffer, 0)` 实现
2. buffer 必须位于 UB 内存（通过 `T.alloc_ub` 分配）
3. buffer 地址需 32 字节对齐（硬件约束）
4. size 由 buffer shape 自动推断，无需显式传入 count 参数
5. 当传入 tir.Var 时，需确保该变量已通过 `T.has_let_value` 绑定到有效的 BufferRegion

## 3. 示例代码

**示例 1：清零 UB 缓冲区**

```python
a_ub = T.alloc_ub((block_M, block_N), "float16")
T.tile.fill(a_ub, 10.0)
T.tile.clear(a_ub)  # 将 a_ub 所有元素清零
```

**示例 2：清零 BufferRegion 切片**

```python
a_ub = T.alloc_ub((block_M, block_N), "float32")
T.tile.fill(a_ub, 10.0)
T.tile.clear(a_ub[0:block_M, 0:block_N])  # 清零指定区域
```
