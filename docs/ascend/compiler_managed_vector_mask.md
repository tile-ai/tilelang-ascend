# Compiler-managed Ascend Vector mask state

Ascend Vector instructions share mutable mask registers. Count-form Ascend C APIs update those
registers around every call, even when several adjacent instructions need the same mask. On A2 and
A3, TileLang instead tracks the required state in the compiler and emits a setter only when the
known state does not satisfy the next instruction.

This is an internal lowering feature. It does not add a public mask argument to `T.tile.*`, change
PTO lowering, initialize a function-wide default mask, or restore mask state when a function exits.
The pass is enabled for `target="ascendc"` and `target="auto"` on A2/A3. It is not enabled for A5,
whose TileLang backend does not use Ascend C.

## Compilation flow

The public TileLang call remains the semantic operation while existing optimization, memory, and
synchronization passes run. Physical Vector instruction selection happens only after those passes:

```text
T.tile.* semantic call
        |
        v
existing Phase 2 and synchronization passes
        |
        v
AscendVectorInstructionSelection
        |
        v
internal selected operation + typed mask parameters
        |
        v
AscendVectorMaskLegalize
        |
        v
mask setter(s), when required + selected operation
        |
        v
Ascend C explicit-mask overload with isSetMask=false
```

No TIR-transforming pass runs after mask legalization. This keeps the selected operation and its
mask parameters stable until code generation.

## Instruction selection

For ordinary unary, binary, and scalar Vector operations, let `N` be the element count and `S` the
element size in bytes. Selection uses the following exact rules:

| Geometry | Selected mask mode | Repeat count | Mask value |
| --- | --- | ---: | --- |
| `0 < N * S < 256` | NORMAL | 1 | first `N` lanes enabled |
| `N * S` is a multiple of 256 and `N * S / 256 <= 255` | NORMAL | `N * S / 256` | one full repeat |
| otherwise, including symbolic `N` | COUNTER | 1 | element count `N` |

The internal selected call carries a typed `MaskSpec`: mode, repeat count, and the two 64-bit mask
words. NORMAL and COUNTER are therefore values of one selected operation rather than separate
`add_normal` and `add_counter` operation identities.

Selection also checks the exact dtype set supported by each Ascend C overload. A dtype with the
same bit width is not accepted merely because its storage size matches. If the operation, dtype,
shape, or ABI is unsupported, compilation reports an error; it does not fall back to a count-form
call whose behavior has not been validated.

Operations with a different physical ABI, such as reductions, casts, broadcast, clamp, and
workspace-backed helpers, have their own selection recipes in the same catalog.

## Mask-state legalization

The legalizer models three pieces of hardware state independently:

- mask mode: NORMAL or COUNTER;
- low 64-bit mask word;
- high 64-bit mask word.

For each selected call, the legalizer derives two contracts from the operation catalog and its
typed parameters:

- `requires`: the state that must hold before the call;
- `ensures`: the state known after the call.

These contracts are computed by the legalizer; they are not duplicated as mutable attributes on
every internal operation. Each field can require an exact value or accept any value. After the
call, a field can become an exact value, preserve its previous fact, or become unknown.

Before a call, the pass compares its required values with the facts established by earlier calls.
It emits the minimum local repair:

- no setter when every requirement is already known to hold;
- a mode setter when only NORMAL/COUNTER differs;
- one payload setter when either mask word differs;
- both setters when both parts of the state need repair.

The analysis is deliberately conservative across control flow. An `if` keeps a fact only when both
successors prove the same value. A loop that may affect Vector mask state invalidates facts before
and after the loop. Opaque source injection and unknown mask-affecting calls also invalidate the
state. Facts that refer to a local `let` or block variable are removed when that binding ends.

Some composite helpers set or restore mask state internally. Their post-state is described by an
audited helper contract. Helpers without a proven post-state make the affected facts unknown, so a
later instruction repairs them before use.

Contracts include entry requirements as well as post-state. For example, dav-c220
`Gather(..., count)` programs a NORMAL payload but does not switch from COUNTER to NORMAL, so it
requires NORMAL on entry. Explicit-mask Fill has the same mode requirement. Runtime-dependent
helpers require a must-fact merge: a GM-to-UB copy either preserves mask state on its pure MTE2
path or leaves NORMAL/full after its padding `Duplicate`, and only facts true on both paths may
survive. A runtime-strided dtype-converting UB copy similarly preserves state when it executes
zero rows and leaves NORMAL/full after an executed Cast.

The facts describe the complete architectural register, not only the words consumed by the
current dtype. If a b32 helper writes the high mask word to zero, that zero is recorded because a
later b16 operation observes the same high word.

Selection and legalization operate only inside explicit AIV resource scopes. Developer and Hybrid
kernels obtain those scopes from `CombineCV`; Expert kernels that disable automatic C/V separation
must write `T.Scope("C")` and `T.Scope("V")` explicitly. A late verifier rejects resource-specific
hardware work in the mixed outer region, including opaque source or external calls whose owner
cannot be inferred. This keeps C/V separation, Selection, legalization, and Codegen on one region
contract instead of letting mask lowering perform a second implicit split.

Mask reuse is enabled by default. Setting
`tilelang.PassConfigKey.TL_ASCEND_VECTOR_MASK_REUSE` to `False` keeps the same selected raw
terminals, but clears tracked facts before and after every terminal. Each terminal therefore
rebuilds its complete required mask; strict resource-scope validation remains enabled.

## Internal operation catalog

`src/op/ascend_vector_mask_ops.inc` is the declarative source for managed Vector operations. Each
semantic operation groups the information consumed by the individual compiler stages:

- semantic identity and source-call ABI;
- selection rule, dtype domain, operand relations, and typed parameter layout;
- mask-effect recipe used by legalization;
- emitter family and Ascend C intrinsic used by code generation.

The catalog generates the internal TIR operation registrations and drives Selection, validation,
Legalizer contract lookup, and Codegen dispatch. Each pass interprets only its own columns. The
`SelectedCallView` checks the selected call once and exposes named semantic operands and mask
parameters, avoiding positional decoding in every consumer.

Selected helper operations are also explicit internal operations. They preserve semantic operands
and program order, but Codegen can continue to call an existing self-contained helper when its ABI
does not match a raw Ascend C family.

## Code generation

Codegen consumes the selected operation directly. Raw families emit Ascend C overloads with
`isSetMask=false`, using the mask state already established by the legalizer. COUNTER selection
does not use the count-form convenience overload: the legalizer sets COUNTER mode and its count,
then Codegen emits the same explicit-mask family as the NORMAL case.

Codegen does not reselect NORMAL versus COUNTER or infer whether a setter is needed. A remaining
managed semantic call at this boundary is an error, because it indicates that Selection was
skipped or lost an operation.

The same region rule is enforced at Codegen: selected terminals and compiler-inserted setters must
already be inside an explicit AIV scope. Codegen does not synthesize an outer `ASCEND_IS_AIV`
guard or guess ownership for an unscoped operation.

## Extending managed Vector lowering

When adding an operation:

1. Add its semantic group and physical/helper variant to
   `src/op/ascend_vector_mask_ops.inc`.
2. Choose or add a readable ABI, selection, contract, and emitter recipe.
3. Use an exact dtype and shape capability rule; do not add an unvalidated fallback.
4. Verify helper mask effects against the implementation before declaring a precise post-state.
   Check both mode and payload on entry and exit, including runtime-dependent helper branches.
5. Add focused coverage to
   `testing/python/language/test_tilelang_ascend_vector_mask.py` for selection, contract repair,
   resource-scope parity, generated Ascend C arguments, and target isolation.

Keep broad dtype matrices, device sweeps, and performance experiments as local validation unless
they protect a compact, stable compiler contract suitable for continuous integration.
