# CombineCV and Ascend resource-scope verification

This reference owns the implementation design of the shared C/V resource classifier, `CombineCV`,
and `AscendResourceScopeVerify`. For user-facing programming rules and compiler-managed Vector
mask semantics, use `docs/ascend/compiler_managed_vector_mask.md`.

## 1. Contract

Ascend TIR has two execution resources:

- C (`resource_scope=0`): Cube compute and its local L1/L0/MTE1/FIX work;
- V (`resource_scope=1`): Vector compute and its local UB work.

Developer and Hybrid kernels may omit explicit scopes only when
`tl.ascend_auto_cv_combine=true`; `CombineCV` then creates the two branches. If automatic
separation is disabled, every resource-specific operation must already be inside the matching
`T.Scope("C")` or `T.Scope("V")`.

The outer region is not a third hardware resource. It may contain scalar/global structure and
operations classified as common, but not unowned local hardware work.

## 2. One shared classifier

`CombineCV` and `AscendResourceScopeVerify` both call `ResourceForCall()`. Never add a second
substring table. Unknown or opaque work cannot inherit an adjacent statement's owner; only the
known context-dependent synchronization described below may use surrounding statements.

```cpp
enum class AscendResource {
  kNone,      // no resource-specific hardware effect
  kCommon,    // retained in both branches; legal in outer
  kExplicit,  // resource cannot be inferred; explicit C/V scope required
  kCube,
  kVector,
};
```

### 2.1 Classification inputs

Classification combines operation semantics with concrete operands:

| Evidence | Use |
| --- | --- |
| Exact known operation | MMA, set-deq-scale, selected Vector terminal, mask setter, common control |
| `OperationConfig.default_pipeline` | Fixed V, M, MTE1, or FIX owner |
| Pipe or directed pipe pair | Barrier/event owner |
| `tvm_access_ptr` storage scope | Resolve copy/helper operands and verify local-buffer consistency |
| BufferLoad/BufferStore buffer scope | Verify local scalar accesses |

Operation names and buffer scopes are not interchangeable:

- an Add or MMA has an intrinsic execution resource;
- a generic copy is resolved from its concrete src/dst path;
- MTE2, MTE3, S, and ALL are not sufficient by themselves to choose C or V, so local operands or
  an explicit scope must disambiguate them.

### 2.2 Storage-scope mapping

```text
shared.ub                                      -> Vector
shared.l1 / wmma.matrix_a / wmma.matrix_b
          / wmma.accumulator                  -> Cube
global / scalar / unknown non-local storage   -> None
```

`MergeResources()` combines the operation and operand evidence. Cube semantics with a UB local
operand, or Vector semantics with an L1/L0 local operand, is a compiler error rather than a
last-writer-wins classification.

### 2.3 Important special cases

- `printf`, `sync_all`, and `use_swizzle` are common.
- A global-memory `dump_tensor` is common; a UB or L1/L0 dump follows that local buffer's owner.
- `_src_code` and an opaque extern with no classifiable local operand are explicit-only.
- Vector-mask setters, selected managed Vector terminals, and set-deq-scale are Vector.
- `set_flag` / `wait_flag` use their directed pipe pair.
- pipe barriers, cross flags, and auto barriers use their pipe argument.
- `barrier_all` and a local event whose MTE2/MTE3 pair has no unique owner are context-dependent;
  with CombineCV enabled, they are assigned only inside an otherwise pure C/V region or between
  two nearest concrete statements with the same owner.
- A normalized `call_extern` name is looked up in `GetOperationConfig()`; access pointers then
  cross-check or disambiguate the result.
- An unknown `tl.ascend_*` call may be classified from classifiable access pointers; without
  such evidence it is explicit-only.

Unknown/opaque calls fail closed. Do not add naming heuristics such as “contains `copy_`” or
“contains `mma`”.

## 3. CombineCV

When `tl.ascend_auto_cv_combine=false`, `CombineCV` returns the input unchanged. The late
verifier still enforces explicit ownership.

When enabled:

```text
input tilelang_root
    |
    +-- resolve known context-dependent synchronization
    |     pure region or equal two-sided owner -> internal matching scope
    |
    +-- pre-verify with require_explicit_scope=false
    |     reject conflicting nested scopes
    |     reject unscoped Explicit/opaque hardware calls
    |
    +-- Cube emitter
    |     keep Cube + Common + explicit C scope
    |
    +-- Vector emitter
          keep Vector + Common + explicit V scope
    |
    +-- optional existing cross-core sync insertion
    |
    `-- resource_scope=0 body ; resource_scope=1 body
```

Common operations are intentionally retained in both branches. Explicit C/V scope bodies enter
only their matching emitter and are not reclassified statement by statement.

For historical scalar/global stores in the outer Developer form, the emitter keeps the established
Cube-side convention. Local UB/L1/L0 stores are resource-specific and classified from storage.
The strict late verifier checks both BufferStore and BufferLoad, including accesses in conditions
or other expressions.

## 4. AscendResourceScopeVerify

The authoritative verifier runs after `AscendSyncInsert` and `AscendSyncInsertVS`, because
those passes can materialize new hardware calls. It checks the final TIR before managed Vector
instruction selection.

Rules:

1. Cube operations and local L1/L0 accesses require C scope.
2. Vector operations, UB accesses, vectorized loops, selected terminals, and mask setters require
   V scope.
3. `kExplicit` operations require some explicit C/V scope; the author chooses the owner.
4. `kCommon` and `kNone` are legal in outer.
5. C->C and V->V nesting are legal.
6. C->V and V->C nesting are rejected because the nested generated guards can never execute.

The verifier is read-only: it does not repair, split, or wrap operations. A failure means the
source must enable CombineCV, add the correct explicit scope, or teach the shared classifier a
provable resource rule.

## 5. Relationship to managed Vector mask lowering

The relevant pipeline tail is:

```text
AscendMemoryPlanning
  -> AscendSyncInsert
  -> AscendSyncInsertVS
  -> [managed A2/A3 AscendC] final Simplify
  -> AscendResourceScopeVerify
  -> [managed] AscendVectorInstructionSelection
  -> [managed] AscendVectorMaskLegalize
  -> Codegen
```

Selection and Legalize do not perform their own C/V split. They consume only operations already
proven to be in V scope. No TIR-transforming pass may run after MaskLegalize.

## 6. Extending the classifier

For a new hardware operation:

1. Identify the actual execution resource from the operation contract; do not infer it solely
   from dst/src storage.
2. If the resource depends on a copy route or pipe argument, classify from those typed operands.
3. Add or update `OperationConfig` when that is the existing semantic source of truth.
4. Make `MergeResources` detect inconsistent local operands.
5. Keep unknown forms explicit-only.
6. Add tests for auto separation, correct explicit scope, wrong scope, unscoped rejection, and
   nested-scope behavior.

If the operation is compiler-managed Vector work, also update the mask catalog and its
Selection/Legalize tests; do not duplicate mask-effect rules here.

## 7. Required invariants and tests

Permanent tests should protect behavior, not private class names:

- one operation whose name determines Cube or Vector ownership;
- one copy whose src/dst scopes determine the route;
- one operation whose semantic resource conflicts with a local operand;
- one pipe/event classification in each relevant resource;
- unknown outer extern rejected, same extern accepted in an explicit scope;
- local BufferLoad and BufferStore rejected in the wrong or outer scope;
- common control retained in both CombineCV branches;
- same-kind nesting accepted and conflicting nesting rejected;
- `CombineCV` output accepted by the late verifier;
- selected Vector terminal appears only after scope verification and only inside V scope.

The current cross-core workspace matching algorithm remains a separate responsibility in the same
source file. Do not copy its synchronization details into the resource-classifier contract.
