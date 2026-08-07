# T.tile.sort

## 1. Description

Performs a full sort on the input data and outputs the results in descending order as interleaved (value, index) pairs: `dst = [val0, idx0, val1, idx1, ...]`, where `idx` is the 0-based position in the aligned buffer before sorting (including -inf padding positions). Internally, each 32-element block is sorted via sort32, then all sorted blocks are merged via merge_sort.

## 2. Function Prototype

### 2.1 Function Definition

```python
def sort(
    dst: Buffer,
    src: Buffer,
    actual_num: PrimExpr,
)
```

### 2.2 Parameters

| Parameter | Direction | Description | Type | Required/Optional |
|-----------|-----------|-------------|------|-------------------|
| dst | Output | Stores the sort result as interleaved (value, index) pairs. Must have at least `2 × aligned_count` elements (`aligned_count = ((actual_num + 31) // 32) × 32`) | tensor | Required |
| src | Input/Output | Source data to be sorted. For float32, the positions from `actual_num` to `aligned_count` are padded with -inf in-place; for float16, `src` is not modified (it is internally cast to float32 and padded in a temporary buffer) | tensor | Required |
| actual_num | Input | Number of valid elements in `src`. When less than `aligned_count`, the remaining positions are automatically padded with -inf before sorting | integer expression (PrimExpr) | Required |

> **Type notes**:
> - **tensor**: A buffer allocated via `T.alloc_ub`, `T.alloc_shared`, etc. This API does not accept BufferRegion slices.

### 2.3 Parameter Specifications

#### 2.3.1 DataType Support

| Platform | dst | src |
|----------|:---:|:---:|
| Ascend A2 / A3 | float16, float32 | float16, float32 |

#### 2.3.2 Shape Support

- Supports 1D and 2D
- A 2D buffer is **flattened into a single 1D array** in row-major order and sorted as a whole; it is **not** sorted row by row
- This API only accepts Buffer type. Higher-dimensional buffers must first be copied into a 1D/2D Buffer via `T.copy` before being passed in

#### 2.3.3 actual_num Description

| Value | Meaning | Use Case |
|-------|---------|----------|
| Equal to aligned_count | All elements participate in sorting, no padding needed | Data exactly fills a 32-aligned buffer |
| Less than aligned_count | Only the first actual_num elements are valid; the remaining positions are automatically padded with -inf and participate in sorting | Valid data is smaller than one 32-aligned block |

> `aligned_count = ((actual_num + 31) // 32) × 32`, i.e., actual_num rounded up to a multiple of 32.

#### 2.3.4 Output Data Format

dst stores interleaved (value, index) pairs, where both value and index use the dtype of dst, laid out as follows:

| dst dtype | Storage Layout | Bytes per Pair |
|-----------|----------------|:--------------:|
| float32 | `[value(float32), index(float32)]`. The bit pattern of index is identical to uint32 (stored as uint32 internally, read as float in the output) | 8 Bytes |
| float16 | `[value(float16), index(float16)]`. Sorted internally as float32, then rounded back to float16 via CAST_RINT | 4 Bytes |

> The float16 index is an internally generated 0-based sequence (0.0, 1.0, 2.0, ...), sorted as float32 and then cast back to float16. Index values beyond 2048 may lose exactness due to half-precision rounding.

### 2.4 Constraints

1. dst and src must have the same dtype
2. `aligned_count = ((actual_num + 31) // 32) × 32`; dst must have at least `2 × aligned_count` elements (for storing value-index interleaved pairs)
3. src must have at least `aligned_count` elements
4. actual_num must satisfy `1 ≤ actual_num ≤ min(src buffer size, 8160)` (actual_num=0 triggers a hardware exception; when actual_num exceeds the buffer size, the hardware does not report an error but the result is unpredictable)
5. `repeatTimes = (actual_num + 31) // 32`, repeatTimes ∈ [1, 255], i.e., the upper limit of actual_num is 255 × 32 = 8160 (hardware constraint; repeatTimes=0 triggers an aicore exception)
6. Large actual_num is limited by UB capacity: dst requires `2 × aligned_count` elements, src requires `aligned_count` elements, and the internal temporary buffer is of the same order of magnitude; the sum of all three must not exceed UB capacity (the practically usable actual_num is far smaller than 8160)
7. Whether src is modified in-place depends on the dtype: for float32, positions from `actual_num` to `aligned_count` are padded with -inf, so src is modified; for float16, src is not modified (it is internally cast to float32 and operated on in a temporary buffer). For float32, copy src beforehand if you need to preserve the original data
8. src and dst addresses must not overlap (dst is written to, and the internal merge process ping-pongs between dst and tmp; src is read/modified for float32)
9. The sort direction is fixed to descending order
10. All buffer addresses must be 32-byte aligned (hardware constraint)

## 3. Example Code

**Example 1: 1D sort (actual_num equals buffer size)**

```python
src = T.alloc_ub((256,), "float16")
dst = T.alloc_ub((512,), "float16")
T.tile.sort(dst, src, 256)
```

**Example 2: 1D sort (actual_num smaller than buffer size)**

```python
ub_N = ((131 + 31) // 32) * 32  # 160
src = T.alloc_ub((ub_N,), "float16")
dst = T.alloc_ub((ub_N * 2,), "float16")
T.tile.sort(dst, src, 131)
```

**Example 3: 2D sort (flattened and sorted as a whole)**

```python
M = 4
per_row_N = 128
ub_N = per_row_N  # 128 is already a multiple of 32
src = T.alloc_ub((M, ub_N), "float16")    # 512 elements in total
dst = T.alloc_ub((M, ub_N * 2), "float16") # 1024 elements in total
T.tile.sort(dst, src, M * per_row_N)       # actual_num = 512, flattened single sort
```