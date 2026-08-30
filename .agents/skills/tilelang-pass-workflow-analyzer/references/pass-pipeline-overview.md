# TileLang-Ascend pass pipeline

This reference owns the current execution order in `tilelang/engine/phase.py`. Do not copy fixed
pass counts or source line numbers into other skills; target and config gates make them unstable.

## Two phases

```text
Python DSL
  -> Phase 1: LowerAndLegalize
       normalize the public DSL and lower tile operations
  -> Phase 2: OptimizeForTarget
       schedule, split resources, plan memory, insert synchronization,
       verify ownership, and select final AscendC Vector terminals
  -> Codegen
```

Phase 1 establishes analyzable hardware-oriented TIR. Phase 2 is target-specific and may contain
target/config-gated passes.

## Phase 1: LowerAndLegalize

Current order:

```text
InjectTmpBuffer
  -> AscendInferBufferScope
  -> AscendVidReduction
  -> BufferShapeCollector
  -> BindTarget
  -> HostProcesser
  -> Simplify
  -> AscendLowerParallelToVector
  -> LayoutInference
  -> CollectBufferShapes
  -> LowerTileOp
  -> AscendTailMaskPropagation
  -> AscendWorkspaceReduction
  -> LegalizeVectorizedLoop
  -> LegalizeSafeMemoryAccess
  -> Simplify
```

Important outputs:

| Output | Producer | Later consumer |
| --- | --- | --- |
| local buffer storage scopes | AscendInferBufferScope | resource classification, copy lowering, memory passes |
| `initial_buffer_shapes` attr | BufferShapeCollector | Flatten2DBuffer preserves the original inner dimension |
| `buffer_shapess` attr | CollectBufferShapes | WorkspaceReduction/PTO and MemoryPlanning fallback |
| `logic_buffer_shapes` attr | Flatten2DBuffer | AscendMemoryPlanning primary shape input |
| lowered semantic hardware calls | LowerTileOp and related lowering | Phase 2 scheduling/sync; final managed Selection |
| optional tail-aware calls | AscendTailMaskPropagation | existing backend and managed Selection |

## Phase 2: OptimizeForTarget

### Scheduling and structural lowering

```text
PlanAndUpdateBufferAllocationLocation
  -> CrossCorePipeline
  -> CombineCV
  -> PipelinePlanning
  -> InjectSoftwarePipeline
  -> AscendLowerOpaqueBlock
  -> NarrowDataType(32)
  -> ConfigIndexBitwidth
  -> Flatten2DBuffer
  -> FlattenBuffer
  -> Simplify
  -> VectorizeLoop
  -> AscendStorageRewrite
  -> UnrollLoop
  -> RenormalizeSplitPattern
  -> Simplify
  -> RemoveNoOp
  -> RewriteUnsafeSelect
  -> HoistIfThenElse
```

`CombineCV` is config-gated internally. When enabled, it uses the shared C/V resource classifier
to produce `resource_scope=0/1` branches. When disabled, Expert input must already provide explicit
scopes; the strict verifier later enforces the same contract.

### Memory, synchronization, and final backend boundary

The tail order is a correctness invariant:

```text
AscendMemoryPlanning
  -> AscendSyncInsert
  -> AscendSyncInsertVS
  -> [A2/A3 ascendc/auto] final Simplify
  -> AscendResourceScopeVerify
  -> [A2/A3 ascendc/auto] AscendVectorInstructionSelection
  -> [A2/A3 ascendc/auto] AscendVectorMaskLegalize
  -> Codegen
```

| Pass | Why it is here |
| --- | --- |
| AscendMemoryPlanning | Produces final `address_map` / `size_map` before dependency-based sync |
| AscendSyncInsert | Uses planned addresses to insert ordinary pipeline synchronization |
| AscendSyncInsertVS | Materializes final V->V and S-related synchronization calls |
| final Simplify | Cleans sync-generated conditions before the immutable managed terminal boundary |
| AscendResourceScopeVerify | Sees every final hardware call and verifies its C/V owner |
| InstructionSelection | Rewrites managed semantic Vector operations into typed physical terminals |
| MaskLegalize | Inserts required mask repairs and is the last TIR transformation |

The verifier runs even when compiler-managed mask lowering is not active. Selection and
MaskLegalize run only for A2/A3 with target model `ascendc` or `auto`.

## Non-crossable placement boundaries

1. A pass that changes buffer count, size, address, aliasing, or access relations belongs before
   AscendMemoryPlanning.
2. A pass that creates memory/pipeline-affecting hardware calls belongs before AscendSyncInsert
   and must update `OperationConfig` when dependency analysis needs the call.
3. Any pass that creates or changes hardware calls, barriers, events, or scopes belongs before
   AscendResourceScopeVerify.
4. A pass that changes public managed Vector semantics or arguments belongs before
   AscendVectorInstructionSelection.
5. Only InstructionSelection creates selected terminals; MaskLegalize follows it immediately.
6. No TIR-transforming pass may run after MaskLegalize.

Read `new-pass-placement-guide.md` before inserting a pass near this tail.

## Relevant configuration

| Key | Default semantics | Effect |
| --- | --- | --- |
| `tl.ascend_auto_cv_combine` | false | Enable automatic C/V splitting; otherwise explicit scopes are required |
| `tl.ascend_auto_cross_core_sync` | false | Enable CombineCV's existing automatic workspace cross-core sync |
| `tl.ascend_memory_planning` | false strategy flag | Control automatic planning behavior; the pass still runs to publish maps |
| `tl.ascend_auto_sync` | false | Enable AscendSyncInsert's automatic synchronization |
| `tl.ascend_auto_sync_vs` | target-dependent | Enable the V/S synchronization supplement |
| `tl.ascend_vector_mask_reuse` | true | Reuse compatible mask facts; false forces conservative per-terminal repair |

Do not infer a default from the fact that a pass is Ascend-specific. Verify its C++ `GetConfig`
default and any target-level defaults.

## Authoritative files

| Concern | File |
| --- | --- |
| exact order and target gates | `tilelang/engine/phase.py` |
| Python wrappers | `tilelang/transform/__init__.py` |
| public config keys | `tilelang/transform/pass_config.py` |
| C/V classifier and verifier | `src/transform/ascend_combinecv.cc` |
| managed mask contract | `docs/ascend/compiler_managed_vector_mask.md` |
