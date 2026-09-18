# Static FP32 row reductions

On A2/A3, AscendC uses a header-only C++17 implementation for static FP32
`T.reduce_sum`, `T.reduce_max`, and `T.reduce_min` over the last dimension.
These operations produce one result per row; they do not reduce the row count
to produce one result per column. A5, PTO, FP16, and other reduction axes retain
their existing implementations.

## Logical width and physical pitch

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
with T.Scope("V"):
    T.reduce_max(src, dst, dim=-1, real_shape=[31, 95], clear=True)
```

Only the first 95 elements of each row contribute. `clear=True` assigns the
result; `clear=False` combines it with an initialized destination using the
same operation (max, min, or addition).

## Buffers and temporary space

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

## Compiler integration

`InjectTmpBuffer` and the device template use the same `constexpr` planner and
placement calculation. Instruction selection chooses a dedicated FP32 row
terminal; code generation emits one typed `tl::ascend::reduce_2d` call whose
Vector-pipe declaration enables BiSheng to synchronize its caller dependencies.

Single-row inputs use a fixed width-based policy. Multiple rows use recursive
compression, with logical width and intermediate physical pitch represented
separately. The planner selects an intermediate pitch and a tail strategy
together with the emitted instructions.

The helper requires NORMAL mode with a full low mask word and restores both
mask words to full. Compiler-managed mask legalization establishes that entry
state and records the exit state. A later UB-to-GM copy still needs its normal
Vector-to-MTE3 synchronization.

The dependency calculator is shared by max/min/sum. A3 enables its
schedule-specific repeat-zero spacing; A2 disables it and uses Vector
barriers. A calculated distance is not an ISA-wide timing guarantee: changing
the instruction schedule, compiler, or platform requires renewed validation.
Dependencies without a numerical rule keep a Vector barrier.

The three implementation headers are package build inputs under
`src/tl_templates/ascend/`: `reduce_2d_v2.h`, `reduce_2d_m1.h`, and
`reduce_2d_vector_delay.h`. Host workspace sizing and device emission must
continue to use these same headers.
