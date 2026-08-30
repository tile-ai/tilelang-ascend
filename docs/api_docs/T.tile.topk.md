# T.tile.topk

## 1. Description

Sorts the source data in descending order and extracts the top K interleaved (value, index) pairs: `dst = [val0, idx0, val1, idx1, ..., val(K-1), idx(K-1)]`, where `idx` is the 0-based position in the aligned buffer before sorting (including -inf padding positions). Internally, the full sort is performed first (via sort32 + merge), then the top K pairs are copied to `dst`.

## 2. Function Prototype

### 2.1 Function Definition

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

### 2.2 Parameters

| Parameter | Direction | Description | Type | Required/Optional |
|-----------|-----------|-------------|------|-------------------|
| dst | Output | Stores the top K interleaved (value, index) pairs. Must have at least `aligned_topk` elements (`aligned_topk = ((2*K + elems_per_block - 1) / elems_per_block) * elems_per_block`, where `elems_per_block = 32 / sizeof(T)`, i.e. 16 for float16, 8 for float32) | tensor | Required |
| src | Input/Output | Source data to find the top K from. For float32, the positions from `actual_num` to `aligned_count` are padded with -inf in-place; for float16, `src` is not modified (it is internally cast to float32 and operated on in a temporary buffer) | tensor | Required |
| K | Input | Number of top elements to extract, `1 <= K <= actual_num` | integer expression (PrimExpr) | Required |
| actual_num | Input | Number of valid elements in `src`. When less than `aligned_count`, the remaining positions are automatically padded with -inf before sorting | integer expression (PrimExpr) | Required |
| tmp | Input | Optional complete UB scratch storage. Its scalar dtype is reinterpreted by lowering and has no semantic meaning; when omitted, the compiler allocates it automatically | tensor | Optional (default `None`) |

> **Type notes**:
> - **tensor**: A buffer allocated via `T.alloc_ub`, `T.alloc_shared`, etc. This API does not accept BufferRegion slices.

### 2.3 Parameter Specifications

#### 2.3.1 DataType Support

| Platform | dst | src |
|----------|:---:|:---:|
| Ascend A2 / A3 | float16, float32 | float16, float32 |

#### 2.3.2 Shape Support

- Supports 1D and 2D
- A 2D buffer is internally treated as a flat 1D array in row-major order; however, the number of elements actually involved in sorting equals the **sum** of the shape dimensions (rows + columns), not the total element count. Using 1D buffers is the intended usage (all built-in examples use 1D)
- `src` must have a compile-time static shape

#### 2.3.3 K Description

| Value | Meaning | Use Case |
|-------|---------|----------|
| `1 <= K <= actual_num` | Extract the top K maximum values from src | General TopK scenario |

#### 2.3.4 actual_num Description

| Value | Meaning |
|-------|---------|
| Equal to src buffer size | All elements in the buffer participate in sorting |
| Less than src buffer size | Only the first actual_num elements are valid; the remaining positions are automatically padded with -inf and participate in sorting |

> `aligned_count = ((buffer_size + 31) // 32) * 32`, derived from the compile-time buffer size.

#### 2.3.5 Output Data Format

dst stores interleaved (value, index) pairs, where both value and index use the dtype of dst, laid out as follows:

| dst dtype | Storage Layout | Bytes per Pair |
|-----------|----------------|:--------------:|
| float32 | `[value(float32), index(float32)]`. The bit pattern of index is identical to uint32 (stored as uint32 internally, read as float in the output) | 8 Bytes |
| float16 | `[value(float16), index(float16)]`. Sorted internally as float32, then rounded back to float16 via CAST_RINT | 4 Bytes |

> The float16 index is an internally generated 0-based sequence (0.0, 1.0, 2.0, ...), sorted as float32 and then cast back to float16. Index values beyond 2048 may lose exactness due to half-precision rounding.

### 2.4 Constraints

1. dst and src must have the same dtype
2. `elems_per_block = 32 / sizeof(T)`; dst must have at least `aligned_topk = ((2*K + elems_per_block - 1) / elems_per_block) * elems_per_block` elements. The valid result occupies the first `2*K` elements; the elements beyond `2*K` (up to `aligned_topk`) are unspecified
3. src must have a compile-time static shape; `buffer_size = sum(src.shape)` and `aligned_count = ((buffer_size + 31) // 32) * 32`
4. src's buffer size should be a multiple of 32 (so that `aligned_count == buffer_size`); otherwise the -inf padding overflows the buffer and can corrupt the output indices
5. actual_num must satisfy `1 <= actual_num <= src buffer size`
6. K must satisfy `1 <= K <= actual_num`
7. `repeatTimes = (buffer_size + 31) // 32`, repeatTimes in [1, 255], i.e. the src buffer size does not exceed 255 * 32 = 8160 (hardware constraint)
8. Large buffer sizes are limited by UB capacity: dst requires `aligned_topk` elements, src requires `aligned_count` elements, and the internal temporary buffer is of the same order of magnitude; the practically usable size is far smaller than 8160
9. Whether src is modified in-place depends on the dtype: for float32, positions from `actual_num` to `aligned_count` are padded with -inf, so src is modified; for float16, src is not modified (it is internally cast to float32 and operated on in a temporary buffer). For float32, copy src beforehand if you need to preserve the original data
10. src and dst addresses must not overlap
11. The sort direction is fixed to descending order (extracts the maximum values)
12. inf is treated as a very large value; nan always sorts first (treated as the largest value)
13. All buffer addresses must be 32-byte aligned (hardware constraint)

## 3. Example Code

**Example 1: 1D topk (buffer size is a multiple of 32)**

```python
src = T.alloc_ub((128,), "float16")
dst = T.alloc_ub((32,), "float16")  # aligned_topk = ceil(2*10/16)*16 = 32 for float16
T.tile.topk(dst, src, 10, 128)      # K = 10, actual_num = 128
```

**Example 2: 1D topk (actual_num smaller than buffer size)**

```python
ub_N = ((131 + 31) // 32) * 32  # 160
src = T.alloc_ub((ub_N,), "float16")
dst = T.alloc_ub((32,), "float16")
T.tile.topk(dst, src, 10, 131)   # only the first 131 elements are valid
```