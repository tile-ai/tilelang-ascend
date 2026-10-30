Tile Language: Primitives
=========================

## Ascend compilation options and C/V scopes

Ascend kernels contain work for Cube and Vector execution units. By default, the
compiler identifies which unit each operation needs and places it in the matching
scope. This also applies to kernels that use only Cube or only Vector operations.

The following `tilelang.PassConfigKey` options can be passed through `pass_configs`
to `@tilelang.jit` or `tilelang.compile`:

| Option | Default | Effect |
| --- | --- | --- |
| `TL_ASCEND_AUTO_CV_COMBINE` | `True` | Identify and separate Cube and Vector work |
| `TL_ASCEND_AUTO_SYNC` | `False` | Insert synchronization within each execution unit |
| `TL_ASCEND_MEMORY_PLANNING` | `False` | Reuse on-chip memory according to buffer lifetimes |
| `TL_ASCEND_AUTO_CV_SYNC` | `False` | Insert synchronization for recognized Cube/Vector GM workspace handoffs; requires CombineCV |

Omit `TL_ASCEND_AUTO_CV_COMBINE` to use automatic scope placement. You may also write
`with T.Scope("C")` or `with T.Scope("V")`: valid explicit scopes are preserved, so
handwritten scopes do not require disabling this option. The operations inside a
scope must belong to that unit; nesting C inside V, or V inside C, is rejected.

Set `TL_ASCEND_AUTO_CV_COMBINE=False` when you want to supply all scopes yourself.
Every operation tied to Cube or Vector must then be inside its matching scope.
Operations whose unit cannot be inferred, including opaque external calls and raw
source code, require an explicit scope even with automatic placement enabled.

GM scalar assignments such as `Output[i] = value` also require an explicit
`T.Scope("C")` or `T.Scope("V")`. Choose the side that owns the assignment's
state and indices; the compiler neither infers this owner from surrounding work
nor defaults the write to Cube.

`T.barrier_all()` and `T.pipe_barrier` with `"ALL"`, `"MTE2"`, or `"MTE3"` can
use automatic placement when the surrounding region's other resource-specific
work belongs to one side, or the nearest concrete operations on both sides have
the same owner. Otherwise, supply an explicit scope for the barrier.

Automatic scope placement does not enable the other three options. In particular,
placing a producer and consumer on the correct units does not synchronize them;
their data dependencies still need automatic or explicit synchronization.

## Ascend loop metadata and Persistent domains

The loop interfaces retain their existing positional arguments:

```python
T.Parallel(*extents, coalesced_width=None, annotations=None)
T.Pipelined(start, stop=None, num_stages=0, order=None, stage=None,
            sync=None, group=None, cross_interval=1, *, annotations=None)
T.Persistent(domain, wave_size, index, group_size=8, num_stages=0, *, annotations=None)
```

`annotations` adds non-reserved loop metadata without modifying the supplied map.
New Parallel metadata is attached only to the outermost loop. The existing
`coalesced_width` behavior is retained on every generated Parallel dimension;
supplying conflicting widths through the keyword and metadata is an error.
Use the dedicated arguments for pipeline controls: generated stage/order/sync
keys, task/stage markers and internal layout/version controls cannot be injected
through this map. Pipelined alone accepts a boolean `enable_offset` consumed by
constraint planning. Existing Vector lowering and dimensional limits
remain in force.

`Persistent` requires positive integer domain extents, wave size and group size,
with `0 <= index < wave_size`. The last domain extent may be non-divisible by the
effective group size: the compiler pads the scheduling domain and guards the
body so that only valid tile coordinates execute. Internal wave/index arithmetic
uses int64; returned coordinates retain the domain's integer dtype. Symbolic
arguments must satisfy the positive-domain, worker-range and representability
contracts at runtime.

`num_stages=0` keeps Persistent wave execution serial. Legacy positive stages
require a provably unguarded, exactly one-full-wave domain with divisible
grouping. The calibrated constraint subset below additionally supports static
guarded wave loops. It predicates each operation with its logical tile's
validity and verifies actual per-worker coverage after pipeline expansion.

## Ascend Task and Stage markers

`with T.Task():` marks one complete operation; it does not launch a core or add
internal synchronization. The current A3 AscendC subset permits a single audited
float16/float32 Vector call: a supported single-row GM/UB copy, managed fill or
managed add. Multi-call bodies, internal control flow, nested tasks, manual
synchronization, Cube tasks and compound helpers are rejected. `Task` accepts
optional `latency` and `ii` keywords for signature compatibility, but values other
than `None` remain unsupported; supply calibrated costs through the model below.

`with T.Stage(n):` surrounds complete explicit tasks. The builder requires a
nonnegative constant signed int32 value and rejects booleans. It cannot be
placed inside Task or Parallel. Legacy retains its Stage(0)-only contract.
Nonzero stages require a supported constraint candidate and explicit
`annotations={"enable_offset": True}`; outside a positive candidate they are
rejected. Source reads retain their imperative meaning, so conflicting manual
stages report an unsatisfiable plan. With any explicit Stage in a candidate,
unmarked units are stage zero; Task alone does not freeze automatic staging.

The compiler checks marker structure before frontend rewrites, then validates and
consumes supported markers after lowering operations and before scheduling or
C/V splitting. The explicit read-only schedule analysis retains task/group and
stage metadata. Kernels without markers retain their previous behavior.

## Opt-in A3 constraint planning

The default pipeline planner remains `legacy`. Select the first constraint
subset explicitly and enable the existing synchronization and memory planner:

```python
pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_PIPELINE_PLANNER: "constraint",
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC_VS: False,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: False,
}
```

The single-stage subset requires A3 AscendC/auto, static serial loops, and at least one
`T.Pipelined(..., num_stages=1)` candidate. It supports audited float16/float32
full, aligned GM/UB copies and Vector fill/add. A static aligned tail can use a
separate full short tile; dynamic padding DMA is outside this subset.
Manual flag/barrier calls, mixed Cube/Vector, raw code, explicit stage/order
arrays and multiple stages or buffer versions are rejected in this single-stage
subset. Existing address hints are checked as described below. UB allocations must
be static float16/float32 and fit the native
memory budget, including row padding.

GM aliases remain conservative. When the caller guarantees distinct input and
output storage, declare `T.func_attr({"tir.noalias": True})` in the PrimFunc.
Without that contract, a loop can require cross-iteration GM dependencies that
the existing synchronization does not cover; such a kernel is rejected.

The planner orders operations within one iteration and preserves the source
loop iteration order. The existing injector and synchronization passes still generate
the executable code; a final verifier checks logical dependencies, physical
storage reuse and event lifetimes. Unsupported or unproved cases report an
error. This option requires the optional `ascend-scheduler` solver dependency
when used; legacy does not load it. The strategy participates in the existing
compilation cache key.
Verification expands at most 4096 hardware task instances and 65536 control
visits per trace; larger traces report a limit error.

## Calibrated multi-stage constraint pipelines

Load a measured A3 cost artifact and pass its complete content to the compiler:

```python
cost_model = tilelang.transform.load_ascend_pipeline_cost_model("a3-costs.json")
pass_configs[tilelang.PassConfigKey.TL_ASCEND_PIPELINE_COST_MODEL] = cost_model
```

`T.Pipelined(..., num_stages=2)` or `3` enables the supported modulo path.
The number is a buffer-version search cap. Actual buffer versions can differ;
the compiler validates and records the materialized versions, strides and bytes.
Single-row, contiguous float16/float32 UB buffers have 32-byte aligned version
sizes. Renamed storage must be loop-owned, with every read footprint completely
defined in that iteration and no carry from another iteration. Multiple dtype or
offset views share one storage version count and byte pitch. A full in-place
update after the required definition must pass the same producer and physical-slot
checks. Initialized readonly or carried storage can remain at one version under
the lifecycle contract below.

The profile matches exact operation/dtype/count/geometry and contains measured
effective nanosecond costs including issue and synchronization overhead. It is
not a raw instruction timing guarantee. The loader checks artifact/schema/hash;
planning rejects missing operation geometry. Use the loader's `expected_identity`
argument to check a deployment's SoC/CANN/bisheng/flags. The S5 NPU entries check
those fields against their actual worker before compilation; ordinary planning
does not discover the runtime device. The full content and digest participate in
the existing cache key; a filename alone cannot identify a model.

Automatic multi-stage planning enables offsets by default. Explicit
`enable_offset=False` restricts stages to zero. Manual stages are bounded by a
separate staging horizon of 64 and the expansion limits, rather than by the
requested version cap. A zero-trip candidate is emitted as a verified no-op;
it does not prove a physical multi-buffer allocation was executed.

Static guarded `Persistent` waves use the same audited Vector subset and cost
model. Group padding and the last partial wave remain guarded independently
for every operation. This does not enable Cube or arbitrary stateful helpers.
Managed Vector mask requirements are checked on the actual selected instruction
sequence, with one mask state per Vector domain rather than per buffer version.

Use `scripts/ascend_migration/s5_run_pipeline_npu.py --calibrate-profile` to
generate an artifact with `msprof op`. The migration plan records the complete
command, measurement policy, actual verification results and supported scope.

## Storage owners and source lifetimes

Constraint planning proves complete byte coverage before reads. A zero-trip or
skipped writer does not initialize storage. Multiple partial writes can jointly
define a later read; in-place operations consume their old inputs before defining
their outputs. On-chip allocations begin undefined, while actual external
parameters provide caller-entry values. Internal workspace is not assumed to
contain a caller-provided value.

The proof identifies storage by data Var and known alias bindings. Different
names do not establish independence, and different dtype views do not create
separate allocations. Known pointer aliases are normalized to a root plus one
byte offset. Every view uses the same physical version pitch, converted to its
element dtype. Unknown origins, escapes, unprovable offsets and out-of-bounds
views report an error.
Within a task, overlapping read/write regions must be exactly the same region;
shifted partial overlap is not supported. This rule also applies to final physical
storage reuse. Separate tasks may still contribute partial writes to a complete
definition. Byte arithmetic is checked before simplification or case substitution
can hide an int64 overflow.

Complete initialization outside a pipeline can support readonly UB or initialized
carry with one physical version. Outer allocations retain state across nested or
consecutive serial loops; an allocation inside a loop has a fresh activation on
each entry. Renamed multi-version storage cannot borrow definitions from another
iteration. Nested positive pipelines and general data-dependent control remain
unsupported. Bounded Boolean predicates are exhaustively checked; this compiler
proof does not itself establish scalar argument support in a device adapter.

The final verifier reconstructs source producers and checks each byte segment
against the nearest actual physical writer, then validates production, consumption,
overwrite dependencies and drain of outstanding event tokens using the existing
generated synchronization. Token drain does not independently prove completion
of every final call with no successor; kernel return retains the existing runtime contract.
Scope exit does not imply a barrier. Existing `annotate_address` byte hints retain
their upward 32-byte alignment and may permit proven lifetime reuse. Constraint
mode checks complete physical version size, capacity and aliases; this adds no
exact-placement API.

Compiler tooling can explicitly inspect already lowered, audited calls through
`tilelang.transform.AnalyzeAscendStorageOwners(func, platform="A3")` and
`tilelang.transform.AnalyzeAscendLifetimes(func, platform="A3")`. Both return typed
legacy TVM Maps without changing the input IR or invoking a scheduler. A source
readiness report is separate from the final dependency and event validation.

An internal constraint compiler ABI participates in the memory/disk cache key so
older artifacts cannot bypass the lifecycle checks during a compiler upgrade.
Legacy cache keys retain their existing format. See the migration plan's S6
record and its owner, lifecycle and A3 reuse validation entries for evidence.

## Opt-in A3 synchronization

`PassConfigKey.TL_ASCEND_SYNC_STRATEGY` selects `"legacy"` (default) or `"a3"`.
The A3 option requires the AscendC backend and A3 platform. With the constraint
configuration above, add:

```python
pass_configs[tilelang.PassConfigKey.TL_ASCEND_SYNC_STRATEGY] = "a3"
```

This selects the new local event generator for the audited Vector subset.
It reads final physical addresses and sizes, accounts for storage reuse, and
preserves the planner's task order. The independent final dependency and
lifetime verifier still runs. Unknown effects, scalar/VS effects and manual
event ambiguity remain outside this Vector contract.
The first generator places an all-pipeline barrier after each cross-pipeline
set/wait pair. An IR-adjacent wait alone does not establish that the hardware
flag was cleared before reuse. The final verifier checks fence-based lease
return separately from token pairing. These conservative fences serialize
synchronization boundaries; this option makes no throughput improvement claim.

With the legacy planner, the A3 strategy instead selects fixed explicit C/V
workspace ready/release handoffs. That path retains established local and VS
protection and requires static serial control, one GM channel, one workspace slot and the
audited C-to-two-V mapping. Positive mixed pipelines require a later migration
stage. A consumer release must complete before the next workspace overwrite;
production notification alone is insufficient.
Nonempty handoffs require `T.func_attr({"tir.noalias": True})`; the caller must
provide distinct storage for different GM parameters.
Vector GM/UB handoff copies use full dense rectangles with at most 4095 rows,
a conservative subset of the A3 DataCopyPad descriptor range. Cube transfers
retain their separately audited geometry requirements.

Local event slots are allocated within an A3 directed PIPE domain, with eight
slots per admitted domain. Cross-core handoffs use group-local mode2 and the
conservative user pool 0–10. The first cross-core implementation uses READY0
and RELEASE1 and rejects existing manual cross-core protocols or unknown
implicit flag users. Unproved participation, unsupported domains and budget
exhaustion report an error. No event spill or mode4 mapping is introduced.
The synchronization contract has a separate internal cache ABI for A3 artifacts.
See the migration plan's S7 record for the exact validated scope and commands.

## Finite A3 automatic scheduling

`PassConfigKey.TL_ASCEND_PIPELINE_PLANNER = "auto"` selects the complete backend
for its audited finite kernel subset. Legacy remains the default; `"constraint"`
retains the established bridge. The auto route requires the A3 AscendC backend:

```python
pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_PIPELINE_PLANNER: "auto",
    tilelang.PassConfigKey.TL_ASCEND_SYNC_STRATEGY: "a3",
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: False,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC_VS: False,
}
```

Keep automatic C/V ownership enabled, which is its default. The new backend
owns task ordering, finite expansion, UB/L1 version materialization and all
synchronization for the selected kernel. Its route bypasses the legacy planner,
injector and synchronization passes. Unsupported operations or control flow
produce an error; the backend does not silently mix ownership or fall back.

The first contract admits bounded static control, audited dense DMA, fill/add,
aligned float16-input/float32-output GEMM with independent initialization, and
audited clear-true float32 row reductions. GEMM is an indivisible multi-pipeline
task, including its internal scratch and event use. Unknown calls, manual
events/configuration, HF32, atomic effects and unsupported guards are rejected.
Finite expansion admits at most 256 tasks and 65536 control visits per function.
The numeric protocol admits at most 62 used physical resource domains; this is
a scheduling representation limit, distinct from the hardware event budget.
Read producers, actual version slots and final physical aliases are checked
independently. Vector mask checking includes a separate auto contract.

Loop-private UB/L1 storage can use one or two physical versions. Carry storage
retains one version. A caller-provided GM workspace retains its declared shape,
one slot and final visible contents; the initial mixed route uses one channel
with complete C-to-two-V participation and a ready/release protocol. It does
not infer permission to resize a caller argument or to create hardware-local
C/V direct transfers.

Scheduling uses the uniform `abstract_issue_step_v1` model to choose a legal
finite order. Its units are not hardware nanoseconds. Do not supply
`TL_ASCEND_PIPELINE_COST_MODEL` with this route; that calibrated profile belongs
to the constraint bridge. Actual compilation, host dispatch and device event
times are reported separately by the S8 validation entries. See the migration
plan's S8 record for the precise supported boundaries and measured results.
