# Placing a new TileLang-Ascend pass

This guide decides *where* a pass belongs. Mechanical registration and wrapper templates belong to
`tilelang-pass-generate/references/integration-points.md`.

Always confirm the current sequence in `tilelang/engine/phase.py` before editing it.

## 1. Describe the pass contract

Write down:

| Question | Required answer |
| --- | --- |
| Does the pass mutate TIR or only inspect it? | mutation / analysis / verification |
| Which exact IR form does it consume? | public DSL, lowered calls, scoped calls, planned addresses, selected terminals |
| Which attrs or structural facts does it require? | names, types, and producers |
| Which attrs or structural facts does it produce or invalidate? | names, types, and consumers |
| Is it target-independent or Ascend-specific? | scope and target gate |
| Can it change hardware calls, buffers, scopes, or synchronization? | explicit yes/no |

Do not place a pass from a label such as “legalization” or “optimization” alone. Two verifiers can
need different positions because they validate different final forms.

## 2. First choose the phase

Use Phase 1 when the pass normalizes public DSL semantics, performs target-independent lowering, or
collects information required before target optimization.

Use Phase 2 when it depends on Ascend resources, memory hierarchy, synchronization, backend
instruction forms, or target-specific code generation.

An Ascend-specific verifier is not automatically a Phase 1 legalization pass. It must run after
the last pass that can create the thing it verifies.

## 3. Respect the Phase 2 tail boundary

The current correctness boundary is:

```text
all call/scope/buffer/sync rewrites
  -> AscendMemoryPlanning
  -> AscendSyncInsert
  -> AscendSyncInsertVS
  -> [managed] final Simplify
  -> AscendResourceScopeVerify
  -> [managed] AscendVectorInstructionSelection
  -> [managed] AscendVectorMaskLegalize
  -> Codegen
```

Rules:

1. **Buffer/address/access mutation -> before MemoryPlanning.** This includes buffer count, size,
   storage scope, aliasing, pointer arithmetic, and access relations.
2. **Memory/pipeline-affecting call creation -> before SyncInsert.** Add its dependency metadata to
   `OperationConfig` when automatic synchronization must analyze it.
3. **Any hardware-call/scope/sync mutation -> before ResourceScopeVerify.** The verifier must see
   the final operation set.
4. **Managed semantic Vector mutation -> before InstructionSelection.**
5. **Selected-terminal creation -> inside InstructionSelection.** Do not create selected terminals
   in an unrelated pass.
6. **MaskLegalize immediately follows Selection; no TIR mutation may follow it.**

Do not place an “address reorder” after MemoryPlanning merely because it leaves
`address_map`/`size_map` attrs untouched; that is precisely how the attrs become stale.

## 4. Placement by dependency

| Required input | Earliest safe point |
| --- | --- |
| inferred buffer storage scope | after AscendInferBufferScope |
| final planned shape facts | after Flatten2DBuffer (`logic_buffer_shapes`); use `buffer_shapess` only for its legacy consumers |
| lowered tile calls | after LowerTileOp |
| final software-pipeline structure | after InjectSoftwarePipeline |
| final local addresses | after AscendMemoryPlanning |
| sync-generated hardware calls | after AscendSyncInsertVS |
| verified C/V ownership | after AscendResourceScopeVerify |
| selected Vector terminal ABI | after InstructionSelection, but only MaskLegalize may mutate it |

Then place the pass before the earliest consumer or boundary it could invalidate.

## 5. Common scenarios

### New public Tile operation lowering

- Public semantic lowering normally belongs in Phase 1 near `LowerTileOp`.
- If the A2/A3 AscendC physical form is compiler-managed Vector work, keep the semantic operation
  through existing Phase 2 and add its physical variant to InstructionSelection; do not add a late
  one-off lowering pass.

### Memory optimization

- If it changes allocation size, lifetime, aliasing, or accesses, place it before
  AscendMemoryPlanning.
- If it consumes planned addresses without changing them, it may sit between MemoryPlanning and
  SyncInsert only when its output is explicitly required by SyncInsert.
- Never change addresses after planning without recomputing all dependent maps.

### Synchronization optimization

- A pre-analysis used by SyncInsert belongs before that pass and after its required memory facts.
- A rewrite of inserted sync must finish before AscendResourceScopeVerify.
- A verifier of final hardware sync belongs after AscendSyncInsertVS, not generically at Phase 1
  end.

### Resource-scope verification or classification

- Extend the shared `ResourceForCall` classifier in `ascend_combinecv.cc`.
- Automatic separation remains in CombineCV.
- Authoritative validation remains after both sync passes.
- Do not add a second C/V inference pass around managed mask lowering.

### Managed Vector mask or terminal work

- Semantic capability and payload selection belong in AscendVectorInstructionSelection.
- State repair and reuse belong in AscendVectorMaskLegalize.
- Helper behavior belongs in the operation catalog's audited contract.
- A new pass after Legalize is not an extension point.

### Pure analysis

A read-only analysis can run later than a mutating pass only if it does not cause a downstream
consumer to trust facts that a later rewrite can invalidate. If it publishes attrs, identify every
consumer and invalidator explicitly.

## 6. Reject these rationales

- “It is a verifier, so put it at the end of Phase 1.”
- “It is Ascend-specific, so anywhere in Phase 2 is fine.”
- “It does not edit attrs, so it cannot invalidate MemoryPlanning.”
- “A final Simplify is always harmless.”
- “The operation name looks like copy/MMA, so ownership is obvious.”
- “Legalize can repair any selected call after another rewrite.”

Each statement ignores a concrete structural dependency.

## 7. Placement report

Before implementation, report:

```markdown
## Proposed placement

- Phase: Phase 1 / Phase 2
- Exact position: after X, before Y
- Consumed attrs/IR form:
- Produced or invalidated attrs/IR form:
- Target/config gate:
- Why it does not cross MemoryPlanning, ResourceScopeVerify,
  InstructionSelection, or MaskLegalize:
- Closest existing pass:
```

If the last line cannot be justified from the current `phase.py` and source, return to pass design
instead of guessing.

## References

- exact current order: `pass-pipeline-overview.md`
- attr and structural dependencies: `pass-dependency-graph.md`
- pass signatures: `../../tilelang-pass-analyzer/references/pass-registry-ascend.md`
- C/V classifier design:
  `../../tilelang-pass-analyzer/references/pass-designs/design_ascend_combinecv.md`
