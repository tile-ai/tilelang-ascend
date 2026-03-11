# tilelang-ascend Ops

[toc]

## Allocation Ops

### `alloc_L1`

*This operation allocates buffer in Ascend memory scope L1.*

**TileLang API:**

```
tilelang.language.alloc_L1(shape, dtype)
```

**Operands:**

|  Operand | Description  |
| ------------ | ------------ |
| `shape`  |List or Tuple, buffer shape|
| `dtype`  |String, buffer data type|

### `alloc_L0A`

*This operation allocates buffer in Ascend memory scope L0A.*

**TileLang API:**

```
tilelang.language.alloc_L0A(shape, dtype)
```

**Operands:**

|  Operand | Description  |
| ------------ | ------------ |
| `shape`  |List or Tuple, buffer shape|
| `dtype`  |String, buffer data type|

### `alloc_L0B`

*This operation allocates buffer in Ascend memory scope L0B.*

**TileLang API:**

```
tilelang.language.alloc_L0B(shape, dtype)
```

**Operands:**

|  Operand | Description  |
| ------------ | ------------ |
| `shape`  |List or Tuple, buffer shape|
| `dtype`  |String, buffer data type|

### `alloc_L0C`

*This operation allocates buffer in Ascend memory scope L0C.*

**TileLang API:**

```
tilelang.language.alloc_L0C(shape, dtype)
```

**Operands:**

|  Operand | Description  |
| ------------ | ------------ |
| `shape`  |List or Tuple, buffer shape|
| `dtype`  |String, buffer data type|

### `alloc_ub`

*This operation allocates buffer in Ascend memory scope ub.*

**TileLang API:**

```
tilelang.language.alloc_ub(shape, dtype)
```

**Operands:**

|  Operand | Description  |
| ------------ | ------------ |
| `shape`  |List or Tuple, buffer shape|
| `dtype`  |String, buffer data type|

## Math Ops

### `npuir_add`

*This operation performs element-wise addition.*

**TileLang API:**

```
tilelang.language.npuir_add(src1, src2, dst)
```

**Computation logic:**

```
dst = src1 + src2
```

**Operands:**

|  Operand | Description  |
| ------------ | ------------ |
| `src1`, `src2`  |Scalar or Buffer|
| `dst`  |Scalar or Buffer|

### `npuir_sub`

*This operation performs element-wise substraction.*

**TileLang API:**

```
tilelang.language.npuir_sub(src1, src2, dst)
```

**Computation logic:**

```
dst = src1 - src2
```

**Operands:**

|  Operand | Description  |
| ------------ | ------------ |
| `src1`, `src2`  |Scalar or Buffer|
| `dst`  |Scalar or Buffer|

### `npuir_mul`

*This operation performs element-wise multiplication.*

**TileLang API:**

```
tilelang.language.npuir_mul(src1, src2, dst)
```

**Computation logic:**

```
dst = src1 × src2
```

**Operands:**

|  Operand | Description  |
| ------------ | ------------ |
| `src1`, `src2`  |Scalar or Buffer|
| `dst`  |Scalar or Buffer|

### `npuir_div`

*This operation performs element-wise division.*

**TileLang API:**

```
tilelang.language.npuir_div(src1, src2, dst)
```

**Computation logic:**

```
dst = src1 / src2
```

**Operands:**

|  Operand | Description  |
| ------------ | ------------ |
| `src1`, `src2`  |Scalar or Buffer|
| `dst`  |Scalar or Buffer|

**TileLang API:**

```
tilelang.language.npuir_max(src1, src2, dst)
```

**Computation logic:**

```
dst[i] = max(src1[i], src2[i])
```

**Operands:**

|  Operand | Description  |
| ------------ | ------------ |
| `src1`, `src2`  |Scalar or Buffer|
| `dst`  |Scalar or Buffer|

### `npuir_min`

*This operation performs element-wise minimum.*

**TileLang API:**

```
tilelang.language.npuir_min(src1, src2, dst)
```

**Computation logic:**

```
dst[i] = min(src1[i], src2[i])
```

**Operands:**

|  Operand | Description  |
| ------------ | ------------ |
| `src1`, `src2`  |Scalar or Buffer|
| `dst`  |Scalar or Buffer|

**TileLang API:**

```
tilelang.language.npuir_max(src1, src2, dst)
```

**Computation logic:**

```
dst[i] = max(src1[i], src2[i])
```

**Operands:**

|  Operand | Description  |
| ------------ | ------------ |
| `src1`, `src2`  |Scalar or Buffer|
| `dst`  |Scalar or Buffer|

### `npuir_exp`

*This operation performs element-wise exponentiation.*

**TileLang API:**

```
tilelang.language.npuir_exp(src, dst)
```

**Computation logic:**

```
dst[i] = exp(src[i])
```

**Operands:**

|  Operand | Description  |
| ------------ | ------------ |
| `src`  |Scalar or Buffer|
| `dst`  |Scalar or Buffer|

### `npuir_dot`

*This operation performs matrix multiplication and addition.*

**TileLang API:**

```
tilelang.language.npuir_dot(A, B, C[, size, initC, a_transpose, b_transpose])
```

**Computation logic:**

```
C = C + A × B
```

**Operands:**

|  Operand | Description  |
| ------------ | ------------ |
| `A`  |Scalar or Buffer|
| `B`  |Scalar or Buffer|
| `C`  |Scalar or Buffer|
| `size`(Optional)  |List (default: empty), manually specify the matrix size [m, n, k] (A: m×k, B: k×n, C: m×n)|
| `initC`(Optional)  |Bool (default: false), whether to initialize C value to zero|
| `a_transpose`(Optional)  |Bool (default: false), whether to transpose matrix A before load|
| `b_transpose`(Optional)  |Bool (default: false), whether to transpose matrix B before load|

## Memory Ops

### `copy` (T.copy)

*This operation performs data copy between different memory regions. Use `T.copy(src, dst)` with Buffer or BufferRegion (slice) to specify the copy range; extent is derived from the arguments or inferred by the compiler. The legacy `npuir_copy` API is deprecated; use `T.copy` instead.*

**TileLang API:**

```
tilelang.language.copy(src, dst[, coalesced_width])
```

**Operands:**

|  Operand | Description  |
| ------------ | ------------ |
| `src`  |Buffer, BufferRegion, or BufferLoad (source region; use slice e.g. `A[lo:hi, ...]` for partial copy)|
| `dst`  |Buffer, BufferRegion, or BufferLoad (destination region)|
| `coalesced_width`(Optional)  |Int (default: None), width for coalesced memory access|

### `npuir_load_nd2nz`

*This operation performs data copy with on-the-fly ND to NZ layout transformation.*

**TileLang API:**

```
tilelang.language.npuir_load_nd2nz(src, dst[, size])
```

**Operands:**

|  Operand | Description  |
| ------------ | ------------ |
| `src`  |Buffer|
| `dst`  |Buffer|
| `size`(Optional)  |List (default: empty), manually specify the buffer size|

### `npuir_store_fixpipe`

*This operation performs data copy from L0C to L1 or Global Memory.*

**TileLang API:**

```
tilelang.language.npuir_store_fixpipe(src, dst[, size, enable_nz2nd, channel_split, pre_relu_mode])
```

**Operands:**

|  Operand | Description  |
| ------------ | ------------ |
| `src`  |Buffer|
| `dst`  |Buffer|
| `size`(Optional)  |List (default: empty), manually specify the buffer size|
| `enable_nz2nd`(Optional)  |Bool (default: false), whether enable nz2nd when store|
| `channel_split`(Optional)  |Bool (default: false), whether split channel when store|
| `pre_relu_mode`(Optional)  |String (default: empty), ""/"relu"/"leaky_relu"/"prelu"|

## Shape Manipulation Ops

### `npuir_brc`

*This operation broadcasts a vector or a scalar according to the broadcast axes array.*

**TileLang API:**

```
tilelang.language.npuir_brc(src, dst)
```

**Operands:**

|  Operand | Description  |
| ------------ | ------------ |
| `src`  |Scalar or Buffer|
| `dst`  |Buffer|

**Constraints:**

If both `src` and `dst` are buffers:

src_shape[i] = 1 or dst_shape[i]

## Reudction Ops

### `npuir_reduce`

*This operation reduces one or more axes of the source vector according to the reduction axes array, starting from an init value.*

**TileLang API:**

```
tilelang.language.npuir_reduce(src, dst, dims, reduce_mode)
```

**Operands:**

|  Operand | Description  |
| ------------ | ------------ |
| `src`  |Buffer|
| `dst`  |Buffer|
| `dims`  |List, specify the reduction dimensions|
| `reduce_mode`  |String, specify the reduction mode (sum/prod/max/min/max_with_index/min_with_index/any/all/xori/ori/none)|

**Constraints:**

`dims` and `reduce_mode` can not be empty.

## Type conversion Ops

### `npuir_cast`

*This operation performs element-wise vector type conversion.*

**TileLang API:**

```
tilelang.language.npuir_cast(src, dst, round_mode)
```

**Operands:**

|  Operand | Description  |
| ------------ | ------------ |
| `src`  |Buffer|
| `dst`  |Buffer|
| `round_mode`  |String, spcify the round mode (round/rint/floor/ceil/trunc/odd)|

**Constraints:**

src_shape[i] = 1 or dst_shape[i]

(This operation supports broadcast according to shapes of `src` and `dst`)

## Synchronization Ops

### `set_flag`

*This operation sets flag.*

**TileLang API:**

```
tilelang.language.set_flag(wait_pipe[, event_id])
```

**Operands:**

|  Operand | Description  |
| ------------ | ------------ |
| `wait_pipe`  |String|
| `exent_id`(Optional)  |Int (default: 0)|

### `wait_flag`

*This operation waits flag.*

**TileLang API:**

```
tilelang.language.wait_flag(set_pipe[, event_id])
```

**Operands:**

|  Operand | Description  |
| ------------ | ------------ |
| `set_pipe`  |String|
| `exent_id`(Optional)  |Int (default: 0)|

### `pipe_barrier`

*This operation sets pipe barrier.*

**TileLang API:**

```
tilelang.language.pipe_barrier(pipe)
```

**Operands:**

|  Operand | Description  |
| ------------ | ------------ |
| `pipe`  |String|

### `block_barrier`

*This operation sets block barrier.*

**TileLang API:**

```
tilelang.language.block_barrier(id)
```

**Operands:**

|  Operand | Description  |
| ------------ | ------------ |
| `id`  |Int, flag id|

### `subblock_barrier`

*This operation sets subblock barrier.*

**TileLang API:**

```
tilelang.language.subblock_barrier(id)
```

**Operands:**

|  Operand | Description  |
| ------------ | ------------ |
| `id`  |Int, flag id|

### `sync_block_set`

*This operation sets intra-block synchronization.*

**TileLang API:**

```
tilelang.language.sync_block_set(id)
```

**Operands:**

|  Operand | Description  |
| ------------ | ------------ |
| `id`  |Int, flag id|

### `sync_block_wait`

*This operation waits intra-block synchronization.*

**TileLang API:**

```
tilelang.language.sync_block_wait(id)
```

**Operands:**

|  Operand | Description  |
| ------------ | ------------ |
| `id`  |Int, flag id|

## Frames construction Ops

### `rs` (full name: `ResourceSpecialize`)

*This operation constructs a ResourceSpecializeFrame.*

**TileLang API:**

```
tilelang.language.rs(resource)
```

**Operands:**

|  Operand | Description  |
| ------------ | ------------ |
| `resource`  |String|

### `Scope`

*This operation constructs a scope frame.*

**TileLang API:**

```
tilelang.language.scope(scope)
```

**Operands:**

|  Operand | Description  |
| ------------ | ------------ |
| `scope`  |String, a string representing cube-core or vector-core ("Cube"/"Vector")|
