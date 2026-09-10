# T.tile.sort

## 1. 描述

对输入数据进行全量排序，以降序输出交错的（值, 索引）对：`dst = [val0, idx0, val1, idx1, ...]`，其中 `idx` 是排序前在 aligned buffer 中的 0-based 位置（包含 -inf padding 位置）。内部实现：每个 32 元素块通过 sort32 排序，然后所有排序后的块通过 merge_sort 合并。

## 2. 函数原型

### 2.1 函数定义

```python
def sort(
    dst: Buffer,
    src: Buffer,
    actual_num: PrimExpr,
)
```

### 2.2 参数

| 参数 | 方向 | 说明 | 类型 | 必须/可选 |
|------|------|------|------|-----------|
| dst | 输出 | 存储排序结果，以交错的（值, 索引）对形式存放。必须至少包含 `2 × aligned_count` 个元素（`aligned_count = ((actual_num + 31) // 32) × 32`） | tensor | 必须 |
| src | 输入/输出 | 待排序的源数据。float32 时，`actual_num` 到 `aligned_count` 之间的位置会被原地填充 -inf；float16 时，`src` 不会被修改（内部会转换为 float32 并在临时 buffer 中操作） | tensor | 必须 |
| actual_num | 输入 | `src` 中有效元素的个数。当小于 `aligned_count` 时，剩余位置会自动填充 -inf 后再排序 | 整数表达式 (PrimExpr) | 必须 |

> **类型说明**：
> - **tensor**：通过 `T.alloc_ub`、`T.alloc_shared` 等分配的 buffer。本 API 不支持 BufferRegion 切片。

### 2.3 参数规格

#### 2.3.1 数据类型支持

| 平台 | dst | src |
|------|:---:|:---:|
| Ascend A2 / A3 | float16, float32 | float16, float32 |

#### 2.3.2 Shape 支持

- 支持 1D 和 2D
- 2D buffer 按行主序**展平为一维数组**整体排序，不会逐行独立排序
- 本 API 仅接受 Buffer 类型。更高维度的 buffer 需要先通过 `T.copy` 复制到 1D/2D Buffer 后再传入

#### 2.3.3 actual_num 说明

| 值 | 含义 | 使用场景 |
|----|------|----------|
| 等于 aligned_count | 所有元素参与排序，无需 padding | 数据恰好填满 32 对齐的 buffer |
| 小于 aligned_count | 仅前 actual_num 个元素有效，剩余位置自动填充 -inf 参与排序 | 有效数据不足一个 32 对齐块 |

> `aligned_count = ((actual_num + 31) // 32) × 32`，即 actual_num 向上取整到 32 的倍数。

#### 2.3.4 输出数据格式

dst 以交错的（值, 索引）对存储，值和索引均使用 dst 的 dtype，布局如下：

| dst dtype | 存储布局 | 每对字节数 |
|-----------|----------|:----------:|
| float32 | `[value(float32), index(float32)]`。index 的位模式与 uint32 相同（内部以 uint32 存储，输出时按 float 读取） | 8 Bytes |
| float16 | `[value(float16), index(float16)]`。内部以 float32 排序，然后通过 CAST_RINT 舍入回 float16 | 4 Bytes |

> float16 index 是内部生成的 0-based 序列（0.0, 1.0, 2.0, ...），以 float32 排序后转回 float16。索引值超过 2048 时可能因 half 精度损失导致不完全精确。

### 2.4 约束

1. dst 和 src 的 dtype 必须相同
2. `aligned_count = ((actual_num + 31) // 32) × 32`；dst 必须至少有 `2 × aligned_count` 个元素（用于存储值-索引交错对）
3. src 必须至少有 `aligned_count` 个元素
4. actual_num 必须满足 `1 ≤ actual_num ≤ min(src buffer 大小, 8160)`
5. `repeatTimes = (actual_num + 31) // 32`，repeatTimes ∈ [1, 255]，即 actual_num 上限为 255 × 32 = 8160（硬件约束）
6. 大 actual_num 受 UB 容量限制：dst 需要 `2 × aligned_count` 个元素，src 需要 `aligned_count` 个元素，内部临时 buffer 量级相当，三者之和不能超过 UB 容量（实际可用的 actual_num 远小于 8160）
7. src 是否被原地修改取决于 dtype：float32 时，`actual_num` 到 `aligned_count` 之间的位置会被填充 -inf，src 被修改；float16 时，src 不会被修改（内部转换为 float32 后在临时 buffer 中操作）。如需保留原始数据，请先拷贝 src
8. src 和 dst 地址不能重叠（dst 被写入，内部 merge 过程在 dst 和 tmp 之间 ping-pong；float32 时 src 会被读取/修改）
9. 排序方向固定为降序
10. 所有 buffer 地址必须 32 字节对齐（硬件约束）

## 3. 示例代码

**示例 1：1D 排序（actual_num 等于 buffer 大小）**

```python
src = T.alloc_ub((256,), "float16")
dst = T.alloc_ub((512,), "float16")
T.tile.sort(dst, src, 256)
```

**示例 2：1D 排序（actual_num 小于 buffer 大小）**

```python
ub_N = ((131 + 31) // 32) * 32  # 160
src = T.alloc_ub((ub_N,), "float16")
dst = T.alloc_ub((ub_N * 2,), "float16")
T.tile.sort(dst, src, 131)
```

**示例 3：2D 排序（展平后整体排序）**

```python
M = 4
per_row_N = 128
ub_N = per_row_N  # 128 已经是 32 的倍数
src = T.alloc_ub((M, ub_N), "float16")    # 共 512 个元素
dst = T.alloc_ub((M, ub_N * 2), "float16") # 共 1024 个元素
T.tile.sort(dst, src, M * per_row_N)       # actual_num = 512，展平后整体排序
```