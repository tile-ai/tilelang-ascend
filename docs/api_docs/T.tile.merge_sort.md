# T.tile.merge_sort

## 1. 功能说明

对 2/3/4 条已按降序排好的队列执行多路归并，合并为 1 条降序队列：`dst = MrgSort(src0, src1, [src2], [src3])`。输入输出均采用 (value, index) 交错对格式（每个元素占 2 个 buffer 位置）：`src = [val0, idx0, val1, idx1, ...]`，`dst[i]` 为全局第 i 大的 (value, index) 对。每条 src 的 blockLen（有效元素数）由 buffer 大小自动推算。

## 2. 函数原型

### 2.1 函数定义

```python
def merge_sort(
    dst: Buffer | BufferRegion,
    src0: Buffer | BufferRegion,
    src1: Buffer | BufferRegion,
    src2: Buffer | BufferRegion | None = None,
    src3: Buffer | BufferRegion | None = None,
    *,
    tmp: Buffer | BufferRegion | None = None,
)
```

### 2.2 参数说明

| 参数名 | 输入/输出 | 描述 | 类型 | 必填/可选 |
|--------|----------|------|------|----------|
| dst | 输出 | 存放归并排序后的 (value, index) 交错对，大小至少为所有 src 大小之和 | 张量（tensor） | 必填 |
| src0 | 输入 | 第一条已按降序排序的源队列 | 张量（tensor） | 必填 |
| src1 | 输入 | 第二条已按降序排序的源队列 | 张量（tensor） | 必填 |
| src2 | 输入 | 第三条已按降序排序的源队列（3 路或 4 路归并时使用） | 张量（tensor）/ None | 可选（默认 `None`） |
| src3 | 输入 | 第四条已按降序排序的源队列（仅 4 路归并时使用） | 张量（tensor）/ None | 可选（默认 `None`） |
| tmp | 输入 | 可选临时缓冲区。ascendc 后端不使用（显式传入也会在 codegen 中省略）；pto 后端需要 workspace，不传（None）时由编译器自动分配，显式传入时必须是足以容纳所需大小的非空 buffer | 张量（tensor）/ None | 可选（默认 `None`） |

> **类型说明**：
> - **tensor**：通过 `T.alloc_ub`、`T.alloc_shared` 等分配的缓冲区（Buffer），或其切片（BufferRegion）

### 2.3 参数规格

#### 2.3.1 DataType 支持

| 平台 | dst | src0 | src1 | src2 | src3 |
|------|:---:|:----:|:----:|:----:|:----:|
| Ascend A2 / A3 | float32 | float32 | float32 | float32 | float32 |

> **注意**：仅支持 `float32`，不支持 `float16`。如需对 half 数据排序，请先用 `T.cast` 转为 float32。

#### 2.3.2 Shape 支持

- 支持 1D buffer，以及 1D buffer 的切片（slice），切片起始需满足 32 字节对齐
- 支持 2D buffer 的整行切片（如 `buf[0, :]`、`buf[1, :]`）
- **不支持** 2D buffer 的列偏移切片（如 `buf[0, 8:136]`）；多行区域切片（如 `buf[0:2, :]`）仅 ascendc 支持
- 归并路数由 src2、src3 是否为 None 自动确定：

| src2 | src3 | 归并路数 | 可用性 |
|------|------|---------|--------|
| None | None | 2 路 | ascendc / pto 均支持 |
| 非 None | None | 3 路 | ascendc / pto 均支持 |
| 非 None | 非 None | 4 路 | ascendc / pto 均支持 |

#### 2.3.3 blockLen 说明

每条源队列的 blockLen（有效元素数）由 buffer 大小自动推算：`blockLen = buffer_size // 2`（value-index 对格式，每个元素占 2 个 buffer 位置）。

- **ascendc**：blockLen ∈ [1, 4095]，各 src 的 blockLen 可以不同（不等长归并）
- **pto**：blockLen ∈ [4, 4088]，且各 src 的 blockLen 必须相同（等长归并）

### 2.4 约束条件

1. src 数量必须为 2、3 或 4（由 src2、src3 是否为 None 判断），`num_ways < 2` 或 `> 4` 时抛 `ValueError`
2. 所有 src 与 dst 的 dtype 必须同为 float32
3. dst 大小必须**至少**为所有 src 大小之和（2 路：`dst >= src0 + src1`）
4. 每条源队列必须已经按降序排好（通常是 `T.tile.sort32` / `T.tile.sort` 的输出）
5. blockLen = src 大小 / 2；ascendc 支持 blockLen ∈ [1, 4095]（AscendC MrgSort elementLengths 上限），pto 支持 blockLen ∈ [4, 4088]（硬件约束）。src buffer 大小不能为 0
6. dst 与 src 地址需 32 字节对齐（硬件约束；`T.alloc_ub` / `T.alloc_shared` 分配的完整 buffer 天然满足）
7. 稳定排序：score 相同时，跨队列按 src0 → src1 → src2 → src3 顺序，队列内保持原始顺序
8. 不等长归并仅 ascendc 后端支持；pto 后端要求各 src 大小相同
9. NaN 输入无确定顺序语义，输出位置不受保证（硬件行为）
10. tmp 参数：ascendc 后端不使用；pto 后端需要 workspace——不传（None）时自动分配，显式传入时必须是足以容纳所需大小的非空 buffer

## 3. 示例代码

**示例 1：2 路归并**

```python
src0 = T.alloc_ub((64,), "float32")
src1 = T.alloc_ub((64,), "float32")
dst  = T.alloc_ub((128,), "float32")
T.tile.merge_sort(dst, src0, src1)
```

**示例 2：3 路归并**

```python
src0 = T.alloc_ub((64,), "float32")
src1 = T.alloc_ub((64,), "float32")
src2 = T.alloc_ub((64,), "float32")
dst  = T.alloc_ub((192,), "float32")
T.tile.merge_sort(dst, src0, src1, src2)
```