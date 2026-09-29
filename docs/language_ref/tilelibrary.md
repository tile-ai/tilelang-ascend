Tile Language: TileLibrary
=========================

T.Kernel
--------

args: the grid size (0-3 dimension) and the num_threads.

returns: the blockIdx variables

launch a kernel, it must be used in a with statement. There can be
multiple kernels launched sequentially inside a prim function.

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

The workspace-consuming public APIs expose a keyword-only `tmp=None`.
They include the three reduce APIs and `T.tile.broadcast`, `sort`,
`merge_sort`, `topk`, `gather_mask`, `select`, `gather`, `sigmoid`, `sin`,
`cos`, `pow`, `bitwise_xor`, `clamp`, `clamp_max`, `clamp_min`, `round`, the
deprecated `bilinear_interpolation`, `reduce_sum_experiment`, and
`reduce_sum_mask_experiment`. PTO does not support `bilinear_interpolation`,
`sin`, `cos`, or either experimental ReduceSum API.

Omitting `tmp` requests compiler-managed allocation. Implicit calls with
different backend workspace dtypes share one maximum-sized `uint8` main
arena; lowering creates typed views over the same data variable. The
separate PTO reduce-output allocation used by implicit `clear=False`
reduction is unchanged.

An explicit `tmp` supplies the complete target-specific byte arena for one
call. Its backing Buffer must be one-dimensional, static, contiguous, use a
fixed-width scalar dtype, and be in `shared.ub`; a BufferRegion must itself be
one-dimensional and static, lie within that Buffer, and start at a 32-byte
aligned byte address. The dtype defines only the arena's byte geometry
(`extent * sizeof(dtype)`); it does not describe workspace values. Lowering
creates the target-required typed view over the same bytes without numeric
conversion and preserves a region's byte address. The frontend validates the
arena's structure and alignment. A2/A3 AscendC static FP32 last-axis reductions
use the same planner for allocation and the device helper; lowering rejects
undersized explicit arenas. See the [FP32 row-reduction contract](#static-fp32-row-reductions).
Other paths retain their target-specific allocation and internal-view heuristics;
these do not establish a lower bound for nonempty explicit arenas, whose capacity
remains the caller's responsibility. A path that needs no workspace removes the
operand and accepts a zero-extent arena. There is no public size-query API.

For the current fixed `dav-2201` AscendC target, compiler-managed sizing uses
the following policy. Let `S` be source bytes, `N = repeat_times * 32`, and `d`
be the source element width. Except for the exact FP32 row-reduction planner,
the figures are conservative allocation heuristics derived from CANN source and
targeted sampling, not capacity checks for nonempty explicit arenas.

| API | typed view | implicit bytes | basis |
| --- | --- | --- | --- |
| reduce | path-dependent | See [FP32 row-reduction workspace](#static-fp32-row-reductions) | Shared planner for static FP32 rows; existing backend sizing otherwise |
| sort | source dtype | half: `8*N*d`; float: `2*N*d` | CANN-source/sampling heuristic |
| topk | source dtype | half: `10*N*d`; float: `4*N*d` | CANN-source/sampling heuristic |
| bilinear interpolation | `uint8` | `(src0_elements + src1_elements) * 32` | CANN-source/sampling heuristic |
| sin/cos | `uint8` | half: `max(2*S, 512)`; float: `max(2*S, 384)` | CANN-source/sampling heuristic |
| tensor-tensor pow | `uint8` | half: `max(2*S, 1152)`; float/int32: `max(2*S, 768)` | CANN-source/sampling heuristic |
| bitwise xor | `uint8` | `max(S, 64)` | CANN-source/sampling heuristic |
| round half | `uint8` | `max(S, 256)` | CANN-source/sampling heuristic |
| sigmoid | `uint8` | `S` | CANN-source/sampling heuristic |
| experimental ReduceSum APIs | source dtype | `S` | CANN-source/sampling heuristic |

AscendC clamp variants and float round use basic intrinsics and consume no
workspace. AscendC merge sort, select, gather, and gather mask also remove the
optional operand. For b16/b32 broadcast, equal-shape and scalar cases use 0 B,
axis 0 uses 32 B, and axis 1 uses `q*q*d` plus
`q*align(dst_shape[1], q)*d` when the destination inner row is not 32-byte
aligned, where `q = 32/d`. For b8 broadcast, workspace is always present and
uses `2*(align(src_elements, 16) + align(dst_elements, 16) +
inner_half_elements)`, with `inner_half_elements` obtained from the same axis
policy using half elements. These broadcast formulas follow the CANN 2201
staging layout.

The current AscendC code generator preserves a BufferRegion's starting byte
address and workspace dtype, but does not apply its extent with
`LocalTensor::SetSize`. A region extent is therefore not a strict AscendC
`LocalTensor` upper bound; callers must provide storage for all backend
accesses. PTO typed views preserve both byte offsets and typed extents.

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
- `tmp` is keyword-only and supplies the complete target-specific scratch
  arena. It must be a one-dimensional, static, contiguous fixed-width scalar
  Buffer in `shared.ub`, or a one-dimensional static contiguous BufferRegion
  whose starting byte address is 32-byte aligned. Its dtype is ignored by the
  operation and lowering reinterprets the same byte storage. Zero extent is
  valid when the selected backend path needs no workspace.
- Omitting `tmp` requests compiler-managed allocation. Supplying it affects
  only that call and prevents hidden workspace allocations for it.
- Workspace capacity follows the [arena rules](#temporary-workspace-arenas).
  Static FP32 row reductions have the additional requirements below.
- For PTO row reduction with `clear=False`, lowering splits the arena into
  non-overlapping main-scratch and reduce-output views over the same data
  variable. The output begins at `align_up(primary_tmp_bytes, 32)`. PTO
  column reduction needs no main scratch; with `clear=False`, its sole view is
  the reduce output at offset zero.

<a id="static-fp32-row-reductions"></a>

### Static FP32 row reductions

These requirements apply to A2/A3 AscendC static FP32 `reduce_sum`, `reduce_max`,
and `reduce_min` over the last dimension. Other platforms, backends, dtypes, and
axes retain their existing implementations.

#### Logical width and physical pitch

For an input stored in UB, `M` is the logical row count, `N` is the number of
participating values per row, and `S` is the physical distance between row
starts, in FP32 elements. All three must be compile-time constants:

```text
M > 0, N > 0, S >= N, M == 1 or S % 8 == 0
```

Unaligned multirow inputs fail during lowering. `N` need not be divisible by
eight: use `real_shape` to exclude padding already present in the source:

```python
src = T.alloc_ub((31, 96), "float32")
dst = T.alloc_ub((31,), "float32")
T.reduce_max(src, dst, dim=-1, real_shape=[31, 95], clear=True)
```

Only the first 95 elements of each row contribute. `clear=True` assigns the
result; `clear=False` combines it with an initialized destination using the
same operation (max, min, or addition).

#### Buffers and temporary space

Prefer complete UB buffers. Source, destination, and scratch must have
32-byte-aligned, nonoverlapping bases and own these physical spans:

| Buffer | Required FP32 elements | Access |
| --- | ---: | --- |
| Source | `AlignUp(M * S, 8)` | Read-only |
| Destination | `AlignUp(M, 8)` | Writable, including padding |
| Scratch | Shared planner's reported requirement | Fully clobberable |

The logical destination contains only `M` results. A source padding block may
be read to form an intermediate result, but that result cannot contribute to
the logical reduction. Invalid lanes of the last logical block are masked.

Memory planning aligns allocation bases and total spans; it does not pad
individual source rows. Embedded regions require the caller to prove base
alignment, ownership of destination padding, and absence of aliases.

Omit `tmp` for automatic allocation. An explicit `tmp` must be a static,
contiguous, one-dimensional UB arena with enough bytes and an aligned start.
Lowering creates the required FP32 view and rejects insufficient capacity.
There is no public Python scratch-size query.

The shared planner checks instruction-field representability and the
helper-local UB footprint. The kernel must also fit every other live UB
allocation within the 196352-byte planner budget. Unsupported static plans
fail during lowering instead of selecting a runtime fallback.

See the [compiler design](../ascend/row_reduce.md) for instruction selection,
intermediate layouts, and synchronization within the helper.

T.tile.broadcast
----------------

args: dst, src, axis=None, *, tmp=None

Broadcasts a one- or two-dimensional UB source into a compatible UB
destination. `axis` may be `0`, `1`, or omitted for static shape inference.
The optional keyword-only `tmp` uses the same explicit-arena structural and
alignment rules as the reduce APIs; nonzero capacity remains the caller's
responsibility. PTO broadcast needs no workspace, so an explicit zero-length
arena is accepted and omitted during lowering.

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
