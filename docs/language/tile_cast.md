# T.tile.cast

## 1. 功能说明

将 buffer 中的元素从源数据类型转换为目标数据类型：`dst[i] = (target_dtype)src[i]`

支持 7 种舍入模式控制精度损失行为。

## 2. 函数原型

### 2.1 函数定义

```python
def cast(
    dst: Buffer | BufferRegion,
    src: Buffer | BufferRegion,
    mode: str,
    count: PrimExpr,
)
```

### 2.2 参数说明

| 参数名 | 输入/输出 | 描述 | 类型 | 必填/可选 |
|--------|----------|------|------|----------|
| dst | 输出 | 存放类型转换结果，dtype 为目标类型 | 张量（tensor） | 必填 |
| src | 输入 | 源操作数，dtype 为源类型 | 张量（tensor） | 必填 |
| mode | 输入 | 舍入模式 | 字符串（string） | 必填 |
| count | 输入 | 参与计算的元素个数 | 整数（integer） | 必填 |

> **类型说明**：
> - **tensor**：通过 `T.alloc_ub`、`T.alloc_shared` 等分配的缓冲区（Buffer），或其切片（BufferRegion）
> - **string**：7 种合法舍入模式字符串之一
> - **integer**：正整数表达式（PrimExpr），指定参与计算的元素个数

### 2.3 参数规格

#### 2.3.1 DataType 支持

dst 和 src 可以具有不同的数据类型，形成转换对。以下列出各平台支持的转换对（src → dst）：

| 平台 | 支持的转换对 |
|------|-------------|
| Ascend A2 / A3 | float32 ↔ float16, float32 ↔ bfloat16, float32 ↔ int16/int32/int64, float16 ↔ int8/int16/int32, int8/uint8 → float16, bfloat16 → int32/float32, int32 → int16/int64, int64 → int32/float32, int4b_t ↔ float16 |
| Ascend A5 | float32 ↔ float16/bfloat16/int16/int32/int64/float8/hifloat8, float16 ↔ int4b_t/int8/int16/int32/hifloat8, bfloat16 ↔ float32/int32/float16/float4, int8 → int16/int32, uint8 → uint16/float16/uint32, int16 → uint8/uint32/int32/float16/float32, uint16 → uint8/uint32, uint32 → uint8/uint16/int16, int4b_t → int16/float16/bfloat16, float8_e4m3/e5m2 → float32, hifloat8 ↔ float16/float32, float4 → bfloat16, double ↔ bfloat16/int32/int64/float32 |

> **平台说明**：A5 的转换对远多于 A2/A3，包含 float8、hifloat8、float4、double 等新增类型。具体转换对取决于使用的后端（Ascend C 或 PTO），详见 AscendC:Cast.md 和 pto-isa:TCVT.md。

#### 2.3.2 Shape 支持

- 支持 1D 和 2D
- dst 与 src 的元素总数必须相同，count 不超过 dst 和 src 的元素总数

#### 2.3.3 mode 参数说明

mode 参数控制类型转换时的舍入行为，必须为以下 7 种合法值之一：

| mode 值 | 舍入行为 | 适用场景 |
|---------|---------|---------|
| `CAST_NONE` | 无特定舍入，由硬件默认行为决定 | 无精度损失的转换（如 float16 → float32） |
| `CAST_RINT` | 向最近整数舍入 | 通用舍入，最常用 |
| `CAST_FLOOR` | 向负无穷方向舍入（向下取整） | 需要保证结果不大于原值 |
| `CAST_CEIL` | 向正无穷方向舍入（向上取整） | 需要保证结果不小于原值 |
| `CAST_ROUND` | 向最近整数舍入，0.5 远离零舍入 | 与 CAST_RINT 的区别在于 0.5 的处理 |
| `CAST_TRUNC` | 向零方向截断 | 直接丢弃小数部分 |
| `CAST_ODD` | 向最近奇数舍入 | 特定量化场景 |

> **特殊说明**：
> - 当转换无精度损失时（如 float16 → float32 向上转换），mode 参数不生效，可使用 `CAST_NONE`
> - int32 → float16 转换时 roundMode 不生效，需配合 SetDeqScale 使用
> - TileLang-Ascend 不支持 Ascend C 的 CAST_HYBRID 模式（仅 hifloat8 专用随机舍入）

### 2.4 约束条件

1. mode 必须为 7 种合法舍入模式之一，否则 assert 失败
2. count 为必填参数，需显式指定参与计算的元素个数（不同于多数 tile API 自动推断 size）
3. dst 与 src 的 shape 必须兼容（count 不超过 dst 和 src 的元素总数）
4. dst 与 src 的 dtype 转换对必须在硬件支持的范围内
5. 操作数地址需 32 字节对齐（硬件约束）
6. 当 src 和 dst 位宽不同时，stride 参数以较大位宽为准
7. int32 → float16 转换时 roundMode 不生效，需配合 SetDeqScale 使用

## 3. 示例代码

**示例 1：float16 → float32 上转换**

```python
src = T.alloc_ub((1024,), "float16")
dst = T.alloc_ub((1024,), "float32")
T.tile.cast(dst, src, "CAST_NONE", 1024)  # fp16 → fp32，无精度损失，mode 不生效
```

**示例 2：float32 → float16 下转换（带舍入）**

```python
src = T.alloc_ub((512,), "float32")
dst = T.alloc_ub((512,), "float16")
T.tile.cast(dst, src, "CAST_RINT", 512)  # fp32 → fp16，向最近整数舍入
```

**示例 3：float32 → int32 截断**

```python
src = T.alloc_ub((256,), "float32")
dst = T.alloc_ub((256,), "int32")
T.tile.cast(dst, src, "CAST_TRUNC", 256)  # fp32 → int32，截断小数部分
```
