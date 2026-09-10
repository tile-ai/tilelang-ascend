Tile Language: TileLibrary
=========================

T.Kernel
--------

On Ascend, `T.Kernel` takes one block dimension. Fold a multidimensional task
space into that value and recover the logical indices from `cid` inside the
kernel. Developer-mode `threads` may be `1` or `2`; the Expert launch form
omits `threads` and binds `(cid, vid)`.

T.alloc_shared
--------------

args: shape, dtype

returns: Buffer

Allocate buffer on shared memory, It must be used within T.Kernel scope
and should be allocated at the top of the scope.

Dynamic shared memory is used.

T.alloc_fragment
----------------

args: shape, dtype

returns: Buffer

Allocate buffer on register memory, It must be used within T.Kernel
scope and should be allocated at the top of the scope.

The shape represents the whole shape of the buffer. Each element in the
buffer is distributed stored on each threads, this storage partition
will be inferred by the compiler.

T.copy
------

args: src, dst

Copies data from src to dst, src and dst can be one of (Buffer,
BufferLoad, BufferRegion). If you use BufferLoad that represents a
single starting point, the other params should not be BufferLoad, since
we need to know the copy region.

Zero will be padded if we detect the load is out of boundary.

T.gemm
------

args: A, B, C, transpose_A, transpose_B, policy

Performs gemm operation on A, B and C. C must be a fragment, B must be
on shared memory, A can be either a fragment or shared.

Note that the current implementation has some shape and dtype
constraints, for example, the length of reduction axis must be a
multiple of 32 for fp16 multiplicand case, we will update this later.

Temporary workspace arenas
--------------------------

The public APIs using the generic arena contract expose keyword-only
`tmp=None`. They include the three reduce APIs and `T.tile.broadcast`, `sort`,
`merge_sort`, `topk`, `gather_mask`, `select`, `gather`, `sigmoid`, `sin`,
`cos`, `pow`, `bitwise_xor`, `clamp`, `clamp_max`, `clamp_min`, `round`, the
deprecated `bilinear_interpolation`, `reduce_sum_experiment`, and
`reduce_sum_mask_experiment`. PTO does not support `bilinear_interpolation`,
`sin`, `cos`, or either experimental ReduceSum API.

Omitting `tmp` requests compiler-managed allocation for that call.

An explicit `tmp` supplies the complete target-specific byte arena for one
call. Its backing Buffer must be one-dimensional, static, contiguous, use a
fixed-width scalar dtype, and be in `shared.ub`; a BufferRegion must meet the
same requirements, lie within that Buffer, and start at a 32-byte-aligned byte
address. The dtype defines byte geometry only. Lowering creates the target's
typed views over the same bytes without numeric conversion.

The frontend validates structure and alignment, not nonzero capacity. The
caller must provide enough storage for the selected target implementation;
there is no public size-query API. A zero-extent arena is valid only when that
target path consumes no workspace and lowering removes the operand.

This UB scratch arena is unrelated to `workspace_idx`, which declares
runtime-allocated GM tensors in a JIT function signature.

T.reduce_sum / T.reduce_max / T.reduce_min
-----------------------------------------

args: src, dst, dim=-1, *args, clear=True, real_shape=None, tmp=None

Performs an Ascend fast-path reduce operation from src to dst on
dimension dim.

- `clear=True` initializes the destination before writing the reduce
  result.
- `clear=False` merges the reduce result into the existing destination
  (`sum` adds, `max` takes elementwise maximum, `min` takes elementwise
  minimum).
- `real_shape` is optional and describes the logical valid region of a
  sliced 2D UB tile.
- `dst` may use either the reduced output shape or the keepdim form,
  where the reduced axis is retained with extent `1` (for example,
  `[M, N] -> [M]` or `[M, 1]` for `dim=-1`, and `[N]` or `[1, N]` for
  `dim=0`).
- For sliced 2D buffers with `real_shape`, the current frontend also
  accepts compatible physical-layout output forms such as
  `[physical_cols]` or `[1, physical_cols]`.
- The frontend rejects invalid axes, invalid `real_shape`, and invalid
  output shapes before lowering to the backend.
- `tmp` follows the generic arena contract above. Omit it unless the required
  capacity is known for the selected target path.

T.tile.broadcast
----------------

args: dst, src, axis=None, *, tmp=None

Broadcasts a one- or two-dimensional UB source into a compatible UB
destination. `axis` may be `0`, `1`, or omitted for static shape inference.
The optional keyword-only `tmp` uses the same explicit-arena structural and
alignment rules as the reduce APIs; nonzero capacity remains the caller's
responsibility. PTO broadcast needs no workspace, so an explicit zero-length
arena is accepted and omitted during lowering.

T.tile.row_expand_*_experiment
------------------------------

`row_expand_mul_experiment`, `row_expand_sub_experiment`, and
`row_expand_div_experiment` apply one scalar per row. They are experimental
APIs with a specialized `tmp` contract, not the generic byte-arena contract
above.

```text
dst[i, j] = src0[i, j] * src1[i]  # mul
dst[i, j] = src0[i, j] - src1[i]  # sub
dst[i, j] = src0[i, j] / src1[i]  # div
```

- `dst`, `src0`, and `src1` have matching `float16` or `float32` dtype.
- Leading dimensions of `dst` and `src0` fold into a static row count in
  `8..248`, divisible by 8. Each logical row is 256 bytes: 128 fp16 or 64 fp32
  elements.
- The backing last dimensions of `dst` and `src0` are static 32-byte multiples
  that encode the same `1..255`-block row stride. Every operand access offset
  is 32-byte aligned.
- Without `tmp`, `src1` is `[rows, lanes_per_32B]` and each row already contains
  a replicated scalar.
- With `tmp`, `src1` is `[rows]`, `[rows, 1]`, or `[1, rows]`; `tmp` has the
  operand dtype and exactly `rows * lanes_per_32B` contiguous elements.

For BufferRegion operands, the frontend rejects outer-dimension folds it
cannot prove contiguous. Use dense row-major backing buffers; explicit Buffer
strides are outside this experimental contract.

T.Parallel
----------

You can use T.Parallel to write a loop. The loop will be partitioned to
all the threads by the compiler (The compiler will consider vectorize
size, the fragment’s thread mapping … ). Note that this is the only way
you can perform arbitrary operation on fragments.

T.Pipelined
-----------

args: start, stop, num_stages

Pipeline the loop, copy from the global memory will be converted to
async operations and reordered to the point after it is consumed.
num_stages is the number of buffer between producer-consumer.
(e.g.&nbsp;Double buffer when num_stages=2)

T.clear T.fill
--------------

nothing special, they will be converted to T.Parallel

T.use_swizzle
-------------

Optimization for L2 cache. The launch of blockIdx.x and blockIdx.y will
be serpentined.

You need to add it in a kernel after buffer is all allocated.
