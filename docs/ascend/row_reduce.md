# Static FP32 row reductions

On A2/A3, AscendC uses a header-only C++17 implementation for static FP32
`T.reduce_sum`, `T.reduce_max`, and `T.reduce_min` over the last dimension.
These operations produce one result per row; they do not reduce the row count
to produce one result per column. A5, PTO, FP16, and other reduction axes retain
their existing implementations.

The [language reference](../language_ref/tilelibrary.md#static-fp32-row-reductions)
defines the public shape, layout, and temporary-storage requirements. This page
describes the compiler integration and implementation design.

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
