# T.tile.topk

## 1. 描述

对源数据进行降序排序，提取前 K 个交错的（值, 索引）对：`dst = [val0, idx0, val1, idx1, ..., val(K-1), idx(K-1)]`，其中 `idx` 是排序前在 aligned buffer 中的 0-based 位置（包含 -inf padding 位置）。内部实现：先执行全量排序（通过 sort32 + merge），然后将前 K 对复制到 `dst`。

## 2. 函数原型

### 2.1 函数定义

```python
def topk(
    dst: Buffer,
    src: Buffer,
    K: PrimExpr,
    actual_num: PrimExpr,
    *,
    tmp: Buffer | BufferRegion | None = None,
)
```

### 2.2 参数

| 参数 | 方向 | 说明 | 类型 | 必须/可选 |
|------|------|------|------|-----------|
| dst | 输出 | 存储前 K 个交错的（值, 索引）对。必须至少包含 `aligned_topk` 个元素（`aligned_topk = ((2*K + elems_per_block - 1) / elems_per_block) * elems_per_block`，其中 `elems_per_block = 32 / sizeof(T)`，即 float16 为 16，float32 为 8） | tensor | 必须 |
| src | 输入/输出 | 待提取 topk 的源数据。float32 时，`actual_num` 到 `aligned_count` 之间的位置会被原地填充 -inf；float16 时，`src` 不会被修改（内部会转换为 float32 并在临时 buffer 中操作） | tensor | 必须 |
| K | 输入 | 要提取的 top 元素个数，`1 <= K <= actual_num` | 整数表达式 (PrimExpr) | 必须 |
| actual_num | 输入 | `src` 中有效元素的个数。当小于 `aligned_count` 时，剩余位置会自动填充 -inf 后再排序 | 整数表达式 (PrimExpr) | 必须 |
| tmp | 输入 | 可选的完整 UB 临时存储空间。其标量 dtype 由 lowering 阶段重解释，无语义含义；省略时由编译器自动分配 | tensor | 可选（默认 `None`） |

> **类型说明**：
> - **tensor**：通过 `T.alloc_ub`、`T.alloc_shared` 等分配的 buffer。本 API 不支持 BufferRegion 切片。

### 2.3 参数规格

#### 2.3.1 数据类型支持

| 平台 | dst | src |
|------|:---:|:---:|
| Ascend A2 / A3 | float16, float32 | float16, float32 |

#### 2.3.2 Shape 支持

- 支持 1D 和 2D
- 2D buffer 在内部按行主序视为扁平的一维数组，但实际参与排序的元素数等于 shape 各维度之和（行 + 列），而非总元素数。推荐使用 1D buffer（所有内置示例均使用 1D）
- `src` 必须具有编译期静态 shape

#### 2.3.3 K 说明

| 值 | 含义 | 使用场景 |
|----|------|----------|
| `1 <= K <= actual_num` | 从 src 中提取前 K 个最大值 | 通用 TopK 场景 |

#### 2.3.4 actual_num 说明

| 值 | 含义 |
|----|------|
| 等于 src buffer 大小 | buffer 中所有元素参与排序 |
| 小于 src buffer 大小 | 仅前 actual_num 个元素有效，剩余位置自动填充 -inf 参与排序 |

> `aligned_count = ((buffer_size + 31) // 32) * 32`，由编译期 buffer 大小推导得出。

#### 2.3.5 输出数据格式

dst 以交错的（值, 索引）对存储，值和索引均使用 dst 的 dtype，布局如下：

| dst dtype | 存储布局 | 每对字节数 |
|-----------|----------|:----------:|
| float32 | `[value(float32), index(float32)]`。index 的位模式与 uint32 相同（内部以 uint32 存储，输出时按 float 读取） | 8 Bytes |
| float16 | `[value(float16), index(float16)]`。内部以 float32 排序，然后通过 CAST_RINT 舍入回 float16 | 4 Bytes |

> float16 index 是内部生成的 0-based 序列（0.0, 1.0, 2.0, ...），以 float32 排序后转回 float16。索引值超过 2048 时可能因 half 精度损失导致不完全精确。

### 2.4 约束

1. dst 和 src 的 dtype 必须相同
2. `elems_per_block = 32 / sizeof(T)`；dst 必须至少有 `aligned_topk = ((2*K + elems_per_block - 1) / elems_per_block) * elems_per_block` 个元素。有效结果占据前 `2*K` 个元素；`2*K` 之后的元素（至 `aligned_topk`）内容未定义
3. src 必须具有编译期静态 shape；`buffer_size = sum(src.shape)`，`aligned_count = ((buffer_size + 31) // 32) * 32`
4. src 的 buffer 大小应为 32 的倍数（使 `aligned_count == buffer_size`）
5. actual_num 必须满足 `1 <= actual_num <= src buffer 大小`
6. K 必须满足 `1 <= K <= actual_num`
7. `repeatTimes = (buffer_size + 31) // 32`，repeatTimes ∈ [1, 255]，即 src buffer 大小不超过 255 × 32 = 8160（硬件约束）
8. 大 buffer 受 UB 容量限制：dst 需要 `aligned_topk` 个元素，src 需要 `aligned_count` 个元素，内部临时 buffer 量级相当，三者之和不能超过 UB 容量
9. src 是否被原地修改取决于 dtype：float32 时，`actual_num` 到 `aligned_count` 之间的位置会被填充 -inf，src 被修改；float16 时，src 不会被修改（内部转换为 float32 后在临时 buffer 中操作）。如需保留原始数据，请先拷贝 src
10. src 和 dst 地址不能重叠
11. 排序方向固定为降序（提取最大值）
12. inf 被视为非常大的值；nan 总是排在首位（被视为最大值）
13. 所有 buffer 地址必须 32 字节对齐（硬件约束）

## 3. 示例代码

**示例 1：1D topk（buffer 大小是 32 的倍数）**

```python
src = T.alloc_ub((128,), "float16")
dst = T.alloc_ub((32,), "float16")  # aligned_topk = ceil(2*10/16)*16 = 32（float16）
T.tile.topk(dst, src, 10, 128)      # K = 10, actual_num = 128
```

**示例 2：1D topk（actual_num 小于 buffer 大小）**

```python
ub_N = ((131 + 31) // 32) * 32  # 160
src = T.alloc_ub((ub_N,), "float16")
dst = T.alloc_ub((32,), "float16")
T.tile.topk(dst, src, 10, 131)   # 仅前 131 个元素有效
```