# Compiler-managed Ascend Vector mask state

On Ascend A2/A3, many ordinary count-form Vector APIs program the same mask before every call and
restore it afterwards. TileLang avoids that repeated work by selecting an Ascend C overload that
does not set the mask itself, then tracking the shared hardware mask state across adjacent
operations.

For example, two fp32 length-64 additions change from two self-contained count-form calls to one
NORMAL/full mask setup followed by two raw additions:

```text
Before
  Add(dst0, src0, src1, 64)
  Add(dst1, src2, src3, 64)

After
  SetMaskNorm()
  SetVectorMask(full)
  Add<float, false>(dst0, src0, src1, repeat=1)
  Add<float, false>(dst1, src2, src3, repeat=1)
```

Users still write `T.tile.*`. The compiler chooses the physical form, establishes the required
mask, and reuses that state only while it can prove the state is unchanged.

Here, a count-form overload receives a total element count and manages mask state internally. A
raw selected terminal uses `isSetMask=false` and consumes the state established by its caller.
NORMAL/full enables every dtype lane in each repeat; COUNTER stores the total element count in the
mask payload.

## Scope and safe fallback

The feature is enabled for `target="ascendc"` and `target="auto"` on A2/A3. PTO keeps its existing
lowering. The feature is not enabled on A5.

Mask reuse is enabled by default. To keep the same instruction selection while preventing facts
from flowing across selected operations, disable reuse:

Add the fallback to the pass configuration already required by the chosen programming mode:

```python
pass_configs[tilelang.PassConfigKey.TL_ASCEND_VECTOR_MASK_REUSE] = False
```

This is a conservative no-reuse fallback, not a return to semantic or count-form lowering. It
clears facts before and after every selected operation; an operation with a mask requirement then
rebuilds that complete required state. The fallback still relies on the operation's ABI,
`requires`, and emitter being correct.

Strict C/V resource-scope validation is an independent Ascend pipeline invariant. It remains
enabled for PTO and other Ascend paths even when compiler-managed mask lowering is inactive.

> **Temporary `uint8` And/Or compatibility:** direct `uint8` And/Or is deliberately excluded from
> managed raw lowering and temporarily keeps the pre-existing count-form call. On dav-c220 that
> unsupported API executes 16-bit lanes, so logical count `N` can access `2N` bytes. New code
> must not add direct `uint8` And/Or calls. Managed raw lowering supports only `int16`/`uint16`;
> reinterpreting logical bytes as `uint16` is valid only when byte pairing, alignment, and odd-byte
> tails are handled explicitly. The compatibility path exists only until typed views can express
> that physical view and count.

## Compilation flow

Selection runs after the existing scheduling, memory, pipeline, and synchronization passes have
finished. Legalization is the final TIR transformation before Ascend C code generation:

```text
T.tile.* semantic operation
        |
        v
existing Phase 2 transforms
        |
        v
AscendMemoryPlanning -> AscendSyncInsert -> AscendSyncInsertVS
        |
        v
final Simplify for the managed path
        |
        v
AscendResourceScopeVerify
        |
        v
AscendVectorInstructionSelection
        |
        v
selected internal operation + typed physical parameters
        |
        v
AscendVectorMaskLegalize
        |
        v
internal setter call(s) + selected terminal
        |
        v
Ascend C Codegen -> raw isSetMask=false overload or audited existing helper
```

No TIR-transforming pass may run after mask legalization. A later rewrite could otherwise move a
selected operation away from the setter or invalidate its typed mask parameters.

## Resource-scope contract

Compiler-managed Vector operations are valid only in `T.Scope("V")`. There are two supported ways
to establish that scope:

- Developer and Hybrid kernels enable `TL_ASCEND_AUTO_CV_COMBINE`; `CombineCV` classifies the
  resource-specific operations and creates C/V scopes.
- Expert kernels that disable automatic C/V separation write `T.Scope("C")` and `T.Scope("V")`
  explicitly.

Resource-specific work may not remain in the outer mixed region. The late
`AscendResourceScopeVerify` pass checks the final hardware calls after automatic synchronization
has run. It accepts same-kind nested scopes, rejects C-inside-V or V-inside-C nesting, and rejects
an unscoped opaque hardware call whose resource cannot be classified.

`CombineCV` and the verifier share one classifier. Known operations can be classified from their
operation contract; copies are resolved from the concrete source/destination memory scopes; pipe
barriers and events with a unique owner are resolved from normalized, case-insensitive pipe
arguments. Local buffer loads, stores, and dumps are resolved from their storage scope. An opaque
call is allowed only inside an explicit C/V scope; an unscoped unknown call fails closed instead
of inheriting the classification of an adjacent operation. An opaque call inside V scope
invalidates known mask facts.

Only known context-dependent synchronization may use surrounding statements as evidence. For
`barrier_all` or a local event whose MTE2/MTE3 pipes do not identify one owner, `CombineCV`
recursively summarizes the resource-specific work in each sequence, branch, loop, and block. It
assigns the synchronization to C or V only when the containing region is otherwise pure, or when
the nearest concrete statements on both sides have the same exact owner. A C/V boundary, mixed or
opaque region, missing two-sided evidence, or disabled `TL_ASCEND_AUTO_CV_COMBINE` remains an
error and requires an explicit scope.

The outer region is reserved for resource-independent or genuinely shared control such as
`printf`, `sync_all`, `use_swizzle`, and global-memory dumps. It is not a third execution resource.
Every outermost V scope starts mask analysis from unknown state, so its first selected terminal
repairs every field it needs; facts are never reused across sibling V scopes.

## Instruction selection

For ordinary unary, binary, and scalar Vector operations, let `N` be the element count and `S` the
element size in bytes. Selection uses these rules:

| Geometry | Mask mode | Repeat count | Payload |
| --- | --- | ---: | --- |
| `0 < N * S < 256` | NORMAL | 1 | first `N` element lanes enabled |
| `N * S > 0`, is a multiple of 256, and `N * S / 256 <= 255` | NORMAL | `N * S / 256` | all `256 / S` lanes enabled per repeat |
| otherwise, including symbolic `N` | COUNTER | 1 | element count `N` |

The selected operation carries typed physical parameters: mode, repeat count, and the two 64-bit
mask words. NORMAL and COUNTER are therefore two values of the same selected operation, not two
different public operations.

Selection validates the semantic ABI, exact dtype and operand relationship, and the count/mask
payload needed to choose a physical form. Emitters validate layout-derived repeat, stride, and
shape bounds before emitting the CANN call. Equal bit width alone is not sufficient. An
unsupported managed form is a compile-time error; it does not silently fall back to an
unvalidated overload.

Reductions, casts, broadcasts, clamp operations, copies, and composite helpers use operation-
specific selection recipes because their physical ABIs differ from the ordinary elementwise
families.

## Mask-state legalization

The hardware state has three independently tracked fields:

- mode: NORMAL or COUNTER;
- low 64-bit mask word;
- high 64-bit mask word.

For every selected operation, the legalizer derives two contracts:

- `requires`: facts that must hold before the operation executes;
- `ensures`: facts that are guaranteed after it finishes.

Each required field is either exact or unconstrained. Each post-state field is exact, preserved,
or unknown. Before an operation, the legalizer compares its requirements with the facts established
by earlier operations and inserts only the missing repair:

| Known state versus requirement | Repair |
| --- | --- |
| all required fields already match | none |
| mode differs | mode setter |
| either payload word differs | one payload setter that writes both words |
| mode and payload differ | both setters |

Facts are retained only when every possible runtime path proves them. `if` branches use a must-fact
merge; mask-affecting loops are entered and exited with unknown state; facts that mention a local
binding are dropped when the binding ends. Unknown calls and opaque source injection invalidate
the state.

Runtime-dependent helpers use the same rule. A GM-to-UB copy preserves the mask on its pure MTE2
path but leaves NORMAL/full when padding executes a `Duplicate`; only the facts common to both
outcomes survive. A runtime-strided dtype-converting UB copy similarly merges the zero-row preserve
path with the positive-row Cast path.

The contract covers entry state as well as exit state. For example, dav-c220
`Gather(..., count)` writes a NORMAL payload without switching the mode, so it requires NORMAL on
entry. Explicit-mask `FillExperiment` has the same requirement. The analysis also tracks the whole
architectural payload: a 32-bit operation that writes the high word to zero must publish that zero
because a later 16-bit operation observes it.

## Operation catalog and code generation

`src/op/ascend_vector_mask_ops.inc` is the declarative catalog for managed Vector operations. A
semantic group records:

- its public operation and source ABI;
- physical variants, dtype/operand constraints, and typed parameter layout;
- the mask-effect recipe used to derive `requires` and `ensures`;
- the Codegen emitter or existing helper.

The catalog drives internal operation registration, Selection, selected-call validation,
legalization, and Codegen dispatch. `SelectedCallView` is the shared typed reader for semantic
operands and physical parameters, so positional decoding is not repeated in every stage.

Codegen consumes the already selected operation. Raw families emit Ascend C overloads with
`isSetMask=false`; COUNTER and NORMAL both use this path after the legalizer establishes their
state. Codegen does not choose the mode again or decide whether a setter is needed. A remaining
managed semantic operation at this boundary is an error.

Composite helpers can remain helper calls when their ABI does not match a raw family, but their
mask effects still need an explicit audited contract. A helper with no proven post-state publishes
unknown rather than an optimistic value.

## Extending managed Vector lowering

When adding or changing a managed operation:

1. Add its semantic group and physical/helper variant to
   `src/op/ascend_vector_mask_ops.inc`.
2. Define the exact source ABI, dtype/operand constraints, selection recipe, mask-effect recipe,
   and emitter.
3. Verify every entry requirement and every exit path against the exact helper or CANN
   implementation. Audit mode and both payload words, including zero-work and runtime branches.
4. Reject unsupported counts, masks, repeats, strides, dtypes, and shapes before the CANN call is
   emitted; do not rely on debug-only CANN checks or integer narrowing.
5. Confirm the shared CombineCV/verifier classifier identifies the semantic and selected forms as
   Vector work; do not add a second ownership rule inside Selection or Legalize.
6. Add focused tests in `testing/python/language/test_tilelang_ascend_vector_mask.py` for selection,
   repair, control-flow facts, scope ownership, selected ABI validation, and generated Ascend C.
   Cover both reuse modes when changing a mask-effect contract, and preserve target/platform
   isolation.

Keep exhaustive dtype matrices, device sweeps, and performance experiments outside permanent CI
unless they protect a compact and stable compiler contract.
