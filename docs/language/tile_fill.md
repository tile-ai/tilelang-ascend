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
> - **tensor**：通过 `T.alloc_ub`、`T.alloc_shared` 等分配的缓冲区（Buffer），或其切片（BufferRegion）
> - **scalar**：单个元素值，可以是 Python 标量或表达式（PrimExpr），dtype 需与 buffer 一致

### 2.3 参数规格

#### 2.3.1 DataType 支持

| 平台 | buffer | value |
|------|:------:|:-----:|
| Ascend A2 / A3 | float16, float32, bfloat16, int8, uint8, int16, uint16, int32, uint32 | float16, float32, bfloat16, int8, uint8, int16, uint16, int32, uint32 |
| Ascend A5 | float16, float32, bfloat16, int8, uint8, int16, uint16, int32, uint32, int64, uint64 | float16, float32, bfloat16, int8, uint8, int16, uint16, int32, uint32, int64, uint64 |

> **说明**：fill 的底层指令（Ascend C 后端使用 Duplicate / Fill，PTO 后端使用 TEXPANDS）支持所有常见 dtype。value 的 dtype 需与 buffer 的 dtype 一致。

> **后端差异说明**：A2/A3 平台上 bfloat16 仅 Ascend C 后端支持；int8/uint8 仅 PTO 后端支持。

#### 2.3.2 Shape 支持

- 支持 1D 和 2D
- size 由 buffer shape 自动推断（BufferRegion 时取 region extent 的乘积，Buffer 时取 shape 的乘积）

### 2.4 约束条件

1. value 的 dtype 需与 buffer 的 dtype 一致（Ascend C Duplicate 约束）
2. buffer 地址需 32 字节对齐（硬件约束）
3. size 由 buffer shape 自动推断，无需显式传入 count 参数
4. fill 不区分 UB 级别和 L1/L0 级别，统一使用 `tl.ascend_fill` intrinsic
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
