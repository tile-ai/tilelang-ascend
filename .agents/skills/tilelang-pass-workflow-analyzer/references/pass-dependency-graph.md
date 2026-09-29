# TileLang-Ascend pass dependencies

This reference owns data and structural dependencies between passes. The complete current order
belongs to `pass-pipeline-overview.md`.

## Two kinds of dependency

Passes do not communicate only through `PrimFunc` attrs.

| Kind | Examples | Consequence |
| --- | --- | --- |
| Attr data | shape attrs, `address_map`, `size_map` | Consumer must run after producer and preserve the attr |
| Structural IR contract | storage scopes, `resource_scope`, lowered calls, loop annotations, selected terminals, inserted sync | A rewrite can invalidate a later analysis even if no attr changes |

## Main attr data flow

```text
BufferShapeCollector -> initial_buffer_shapes -> Flatten2DBuffer
CollectBufferShapes  -> buffer_shapess -> WorkspaceReduction/PTO
Flatten2DBuffer      -> logic_buffer_shapes -- primary --\
buffer_shapess ------------------------ fallback --> AscendMemoryPlanning
                                                       |
                                               address_map + size_map
                                                       |
                                      AscendSyncInsert -> AscendSyncInsertVS
```

| Attr | Producer | Consumer | Meaning |
| --- | --- | --- | --- |
| `initial_buffer_shapes` | BufferShapeCollector | Flatten2DBuffer | pre-LowerTileOp shapes used to recover the logical inner dimension |
| `buffer_shapess` | CollectBufferShapes | WorkspaceReduction/PTO; MemoryPlanning fallback | earlier logical shapes retained for legacy consumers |
| `logic_buffer_shapes` | Flatten2DBuffer | AscendMemoryPlanning | final aligned 2D logical shapes |
| `address_map` | AscendMemoryPlanning | AscendSyncInsert, AscendSyncInsertVS | planned local-memory start addresses |
| `size_map` | AscendMemoryPlanning | AscendSyncInsert, AscendSyncInsertVS | planned local-memory spans |

Changing a buffer's shape, allocation, alias, address expression, or access relation after
MemoryPlanning can make these maps stale. “The new pass does not modify the attrs directly” is not
sufficient evidence that it is safe after planning.

## Main structural dependencies

### Buffer scope -> resource and copy behavior

```text
AscendInferBufferScope
  -> storage scopes on buffers
  -> LowerTileOp copy route
  -> CrossCorePipeline / CombineCV resource classification
  -> AscendResourceScopeVerify
```

Storage scope is part of the IR, not a separate attr table. The shared C/V classifier also
cross-checks an operation's semantic resource with local operand scopes.

### Cross-core and software pipeline structure

```text
CrossCorePipeline
  -> loop annotations / rewritten cross-core structure
  -> CombineCV
  -> PipelinePlanning
  -> InjectSoftwarePipeline
```

These dependencies are structural. A pass inserted in the middle must preserve the relevant loop,
scope, and workspace relationships.

### Memory planning -> synchronization

```text
final buffer/access structure
  -> AscendMemoryPlanning
  -> address_map / size_map
  -> AscendSyncInsert
  -> AscendSyncInsertVS
```

Both synchronization passes may create final hardware calls. Resource ownership therefore cannot
be authoritatively verified before they run.

### Resource ownership -> managed Vector terminals

```text
CombineCV or explicit T.Scope
  -> AscendSyncInsert / AscendSyncInsertVS
  -> final Simplify
  -> AscendResourceScopeVerify
  -> AscendVectorInstructionSelection
  -> AscendVectorMaskLegalize
  -> Codegen
```

The dependencies are:

1. the verifier must see sync-generated calls;
2. Selection consumes only semantic managed Vector calls already proven to be in V scope;
3. Legalize consumes selected terminals and establishes their required mask state;
4. Codegen assumes the selected ABI and setter placement are final.

No TIR rewrite may follow MaskLegalize.

## Dependency checklist for a new pass

Before placing a pass, answer all of these:

| Question | If yes |
| --- | --- |
| Does it change buffer count, size, scope, aliasing, address, or access relation? | Place before MemoryPlanning; re-evaluate collected shape attrs |
| Does it need `address_map` / `size_map`? | Place after MemoryPlanning but before the sync consumer only if it does not invalidate those maps |
| Does it create a memory/pipeline-affecting hardware call? | Place before SyncInsert; update OperationConfig when needed |
| Does it create or modify any hardware call, barrier, event, BufferLoad/Store, or scope? | Place before ResourceScopeVerify |
| Does it modify a public managed Vector call or its arguments? | Place before InstructionSelection |
| Does it create a selected terminal? | It belongs in InstructionSelection, not a separate pass |
| Does it move or rewrite a selected terminal/setter? | It cannot run after Selection/Legalize; redesign placement |

## Common incorrect placements

- **Address rewrite after MemoryPlanning:** maps no longer describe actual accesses.
- **Resource verifier in Phase 1:** misses hardware calls inserted by synchronization.
- **Simplify after MaskLegalize:** can move/remove selected calls or separate them from repairs.
- **Independent C/V classification in a new pass:** can disagree with CombineCV/verifier.
- **Opaque extern treated as neutral by name substring:** creates a fail-open ownership/effect rule.

## Source-of-truth routing

| Need | Read |
| --- | --- |
| exact pass sequence | `pass-pipeline-overview.md` and then `tilelang/engine/phase.py` |
| placement decision | `new-pass-placement-guide.md` |
| pass registration/signature | `../../tilelang-pass-analyzer/references/pass-registry-ascend.md` |
| C/V classifier internals | `../../tilelang-pass-analyzer/references/pass-designs/design_ascend_combinecv.md` |
| managed Vector state contract | `docs/ascend/compiler_managed_vector_mask.md` |
