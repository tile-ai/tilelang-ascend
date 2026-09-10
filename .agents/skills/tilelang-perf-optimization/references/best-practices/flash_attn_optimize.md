# Flash Attention optimization on TileLang-Ascend

This reference is a review checklist and a worked-example map. Its concrete
buffer shapes and schedule values come from
[`flash_attn_bhsd_expert_h16_d128.py`](../../../../../examples/flash_attention/fa_opt/flash_attn_bhsd_expert_h16_d128.py);
they are not universal recommendations for another shape, dtype, target, or
compiler revision.

## Start with the actual bottleneck

Before changing the kernel:

1. Run the existing correctness check for every supported shape and dtype.
2. Measure the complete operator and `main_kernel` with `msprof op`.
3. Inspect `get_kernel_source()` for the real Cube, Vector, DMA, and
   synchronization schedule.
4. Change one scheduling decision at a time, then repeat correctness and
   performance measurement.

Do not infer the bottleneck from the two GEMMs alone. Depending on the shape,
the exposed path may be Cube work, Vector softmax, GM workspace traffic, or a
buffer-reuse recurrence.

## Worked-example structure

The cited Expert kernel uses this dataflow:

```text
Cube:   Q/K -> QK^T -> score workspace
Vector: score workspace -> online softmax -> probability workspace
Cube:   probability/V -> partial O workspace
Vector: partial O workspace -> running O update -> Output
```

Its `num_stages`, `cross_interval`, tile sizes, layouts, and active-core count
form one jointly capacity- and shape-constrained schedule. When adapting it,
derive candidates from the new iteration count and L1/L0/UB budgets; benchmark
the feasible candidates. A copied value such as `num_stages=14`, `block_M=128`,
or 24 active Cube cores is only an experiment seed.

## Manual buffer ownership

A reused physical region's ownership spans its producer, consumer, first and
last accesses, and next reuse. The synchronization contract is in the
[Programming Guide](../../../../../docs/TileLang-Ascend%20Programming%20Guide.md).

Flash Attention needs particular care at these boundaries:

- **Persistent Q in L1:** MTE2 writes Q, while MTE1 may read it in several
  inner K/V iterations. Return the region only after the final MTE1 read, not
  after the first L1-to-L0 copy.
- **Vector store buffers:** V produces fp16 probability or output data and MTE3
  reads it for a GM store. V may reuse the region only after that store returns
  ownership; a barrier before the store does not protect the later reuse.
- **Load buffers:** MTE2 may overwrite a UB load region only after V completes
  its last read.
- **Aliases:** buffers that overlap through `T.annotate_address` share one
  ownership domain. Separate flags do not order two names backed by the same
  bytes.

TileLang local events and cross-core flags require complete Set/Wait pairing,
including initial tokens and final returns. Unmatched local events can leave
one call apparently correct yet disrupt later launches or other processes on
the same chip. Passing tests does not waive pairing. Only redundant Set/Wait
pairs may be removed.

Following one slot from initialization through acquire/return to cleanup
makes leftover tokens visible, including initial tokens in unused slots.

Local event IDs are allocated per directed `(src_pipe, dst_pipe)` pair. Reuse a
numeric ID across different pairs only after auditing the live events in each
pair; names such as `READY` and `FREE` should still express direction and role.

## Cube path decisions

`T.gemm_v0` and `T.mma` are both valid interfaces. Use `T.mma` when the design
requires explicit L0A/L0B/L0C placement and ownership; do not replace
`T.gemm_v0` solely because an Expert example uses the lower-level primitive.

For an explicit L0 schedule, verify:

- L0A, L0B, and L0C capacity for every live slot;
- the complete K accumulation and `init` condition;
- layout annotations against the current operand declarations;
- MTE1-to-M and M-to-FIX ownership for each physical slot; and
- generated source for the intended copy and MMA forms.

Double buffering is a candidate, not a guarantee of overlap. Compare feasible
slot depths under the complete producer-consumer lifecycle.

## Online softmax and Vector path

Across score blocks, preserve the online-softmax recurrence:

```text
m_new = max(m_old, rowmax(score))
r     = exp(m_old - m_new)
l_new = l_old * r + rowsum(exp(score - m_new))
O_new = O_old * r + P @ V
```

Batching several score blocks can amortize cross-core synchronization, but it
also increases live workspace and UB state. Validate the last partial batch,
the first-block initialization, the final normalization, and every running-O
rescale before treating a batching change as an optimization.

## Workspace and task mapping

A bounded per-core stage ring can use less GM than one workspace slot per
logical task, but only when both directions of stage ownership are complete.
Check the maximum live stages, wraparound reuse, and terminal partial batch.

Derive the active core count from the selected platform/runtime instead of
hard-coding the worked example. For static task assignment, distribute the
remainder explicitly so that task counts differ by at most one; confirm that
the smallest shape does not launch idle work that dominates execution.

## Shape and layout dispatch

Before reusing an existing kernel for a new shape, audit all of these dimensions:

- input/output layout and whether host `permute().contiguous()` adds a real
  kernel;
- query and KV tail tiles;
- head dimension and the L0 footprint of both GEMMs;
- GQA/MQA head mapping;
- causal or additive-mask semantics; and
- compile-time assumptions in task-count and workspace indexing.

Padding KV data with a numeric sentinel is not a general softmax mask. Mask
invalid score columns before exponentiation, and verify the tail with adversarial
input signs. A decode-specific narrow tile is a separate candidate that still
needs minimum-tile legality, capacity, and measured performance.

## Pass configuration

Use the programming mode selected for the operator. An Expert kernel that owns
all placement and synchronization normally disables the four automatic Ascend
passes; Developer/Hybrid code enables the relevant passes instead. Read
[`tilelang-programming-model-guide`](../../../tilelang-custom-skill/tilelang-programming-model-guide/SKILL.md)
rather than copying pass settings from this worked example.

## Acceptance gates

An optimization is acceptable only when:

- every intended shape/dtype passes the numerical test on every checked run;
- repeated or concurrent stress exposes no intermittent ownership failure;
- generated source matches the intended scopes, addresses, events, and flags;
- on-chip and GM workspace sizes remain within the selected platform limits;
- the same profiling method shows a repeatable improvement over the retained
  baseline; and
- unsupported shapes still dispatch to a correct fallback.

Warmup stabilizes timing. It never excuses a wrong first result.
