# A2/A3 AscendC compiler-managed Vector-mask inventory

This inventory closes the compatibility domain used by the mandatory A2/A3
AscendC mask pipeline. The machine-readable source of truth is
`src/op/ascend_vector_mask_ops.inc`; this document records the evidence and the
selection/contract/emission grouping used to construct it.

## Evidence baseline

- Repository baseline: live parent `272c0ab3928df30f84d7ac644456856366ff60c4`
  (the plan's historical audit commit `38e6b38f` is an ancestor).
- Toolkit: CANN 9.1.0-beta.1, `dav_c220` (`__NPU_ARCH__ == 2201`).
- Public interfaces: `kernel_operator_vec_*_intf.h` and their matching
  `kernel_operator_vec_*_intf_impl.h` files.
- Product implementations: `impl/basic_api/dav_c220/kernel_operator_vec_*.h`.
- TileLang emission: `src/target/codegen_ascend.cc` and
  `src/tl_templates/ascend/common.h`.
- Validation: generated AscendC source, Bisheng compilation, and the existing
  A2/A3 device tests under `testing/python/language/`.

`Raw` means the selected emitter reaches a Level-0 instruction with internal
mask setting disabled. `Composite` retains a Level-2/helper implementation and
models its observed pre/post state. `Neutral` does not read or modify the
Vector mask. `Self-contained` establishes the listed post-state without a
precondition.

## Selection, contract, and emission groups

| Semantic operations | Selected behavior | Contract | Emission |
|---|---|---|---|
| `add`, `sub`, `mul`, `div`, `max`, `min`, `bitwise_and`, `bitwise_or` | raw counter | require/ensure COUNTER + exact count | Level-0 binary, `isSetMask=false` |
| `adds`, `subs`, `muls`, `divs`, `maxs`, `mins`, `leaky_relu`, `axpy`, `bitwise_lshift`, `bitwise_rshift` | raw counter | require/ensure COUNTER + exact count | Level-0 scalar, `isSetMask=false` |
| `exp`, `ln`, `abs`, `reciprocal`, `sqrt`, `rsqrt`, `relu`, `bitwise_not` | raw counter | require/ensure COUNTER + exact count | Level-0 unary, `isSetMask=false` |
| `mul_add_dst`, `fill` | raw counter | require/ensure COUNTER + exact count | Level-0 binary/duplicate, `isSetMask=false` |
| `cast`, simple `round` | raw counter | require/ensure COUNTER + exact count | Level-0 cast, `isSetMask=false`; repeat strides preserve Level-2 dtype-width rules |
| `clamp_max`, `clamp_min`, `clamp` | raw counter | require/ensure COUNTER + exact count | one or two Level-0 scalar calls, `isSetMask=false` |
| no-scratch `broadcast` | raw counter | require/ensure COUNTER + exact destination count | TileLang helper with `isSetMask=false` |
| `sub_experiment`, `abs_experiment`, `mins_experiment` | raw counter | require/ensure COUNTER + exact count | corresponding Level-0 operation, `isSetMask=false` |
| narrow `reduce`, fp16 clear `reduce_sum` | raw normal | require/ensure NORMAL + exact materialized bit mask | WholeReduce/narrow helper, `isSetMask=false` |
| `block_reduce_max/min/sum`, `wholereducemax/min/sum` | raw normal | require/ensure NORMAL + exact materialized bit mask | Level-0 reduce, `isSetMask=false` |
| `fill_experiment` | raw normal | require/ensure NORMAL + `(mask0, 0)` | scalar-mask Duplicate, `isSetMask=false` |
| row-expand mul/sub/div experiments, `exp_experiment` | raw normal full | require/ensure NORMAL/FULL | existing explicit-mask helpers (already raw) |
| same-dtype UB-to-UB copy, `sort32`, `transpose`, `gatherb`, `merge_sort`, DCCI experiment, BRCB experiment | neutral | all fields preserve | existing mechanical emitter |
| `compare`, `compare_scalar` | neutral | all fields preserve | CANN Level-2 compare loop |
| `createvecindex`, `gather` | unknown | all fields become unknown | existing composite emitter |
| `arith_progression`, `init_sort_buf` | composite normal full | require NORMAL/FULL; mode remains NORMAL, payload becomes unknown | existing composite emitter |
| `sin`, `cos`, advanced `reduce`, `pow`, `bitwise_xor`, scratch `broadcast`, `sort`, `topk`, custom `gather_mask`, advanced `round`, ReduceSum/Sum experiments, tail unary/binary/scalar/reduce | composite normal full | require NORMAL/FULL; post-state unknown | existing composite emitter |
| dtype-converting UB-to-UB copy, `silu`, `sigmoid` | self-contained normal full | no requirement; ensure NORMAL/FULL | existing Level-2/helper emitter |
| fixed-pattern `gather_mask` | self-contained normal zero | no requirement; ensure NORMAL/zero payload | existing fixed-pattern emitter |
| bilinear interpolation, gather-mask experiment | self-contained normal dynamic | no requirement; ensure materialized NORMAL payload | existing emitter; selected operands carry the payload |
| `select` (tensor or scalar) | self-contained normal full | no requirement; ensure NORMAL/FULL | CANN Level-2 Select |

### Product-specific corrections found during validation

- `Compare`/`CompareScalar` on `dav_c220` use `CompareCompute` loops and do not
  touch the Vector mask. Treating them as raw counter consumers produced an
  invalid repeat schedule; they are neutral.
- `Select` exposes an `isSetMask` template parameter publicly, but the c220
  `VselImpl` reached by the interface does not propagate it and performs its own
  cmpmask/mask setup and restore. It is self-contained, not a raw consumer.
- Cross-dtype raw Cast must reproduce the Level-2 repeat-stride adjustment
  (`4/8` or `8/4` 32-byte blocks); default `8/8` silently skips half the data.
- The `Fill_experiment` helper's one-word mask array is not a valid two-word
  NORMAL mask operand. Selected emission uses the scalar overload with the
  compiler-established state instead.

## Non-semantic registrations

| Name | Classification |
|---|---|
| `tl.ascend_duplicate` | stale config-only name; no registered semantic op |
| `tl.arith_progression` | stale config-only spelling; public op is `tl.ascend_arith_progression` |
| `tl.ascend_copy_vc_experiment` | PTO-only config entry; not an AscendC semantic terminal |
| `tl.ascend_row_expand_mul` | unsupported on AscendC; Selection emits a user-facing diagnostic |

`T._src_code` is the only intentional mask black box. It is a full facts
barrier. New semantic Vector operations must add an explicit selected identity,
contract, base projection, emitter, and tests; no prefix-based fallback exists
on managed A2/A3.

## Audited non-terminal effect table

The Selection input grammar classifies non-terminal calls by exact identity or
an explicit predicate. A miss is a compile error rather than an implicit
neutral operation.

| Call identity/predicate | Effect | Evidence boundary |
|---|---|---|
| `tl.ascend_src_code` | barrier | arbitrary injected source may read or write mask state |
| TIR calls whose registered `TCallEffectKind` is expression annotation, pure, read-state, special-call-argument, or embedded-info | neutral | expression construction/addressing only; opaque, update-state, and control calls are rejected |
| exact `call_extern` helpers `copy_*` except `copy_ub_to_ub`, A5-only `copy_ub_to_ub_Nz`, and A5-only `copy_pipe_to_ub_V`; plus `atomic_add_*`, `mma`, `gemm_v0`, `gemm_v1` | neutral | execute on A2/A3 DMA/Cube paths and do not issue Vector-mask instructions; ordinary UB-to-UB copy is selected separately |
| `tl.ascend_copy`, `tl.ascend_atomic_add`, `tl.region`, `tl.ascend_set_deq_scale`, `tl.ascend_reinterpretcast` | neutral | audited data movement, annotation, and scalar/Cube configuration calls |
| flag, cross-flag, pipe barrier, free-pipe, global-sync, and compiler auto-sync identities | neutral | synchronization only; they neither read nor write Vector mask registers |
| GEMM/MMA, printf/dump, swizzle, SHMEM, and CV/VC-copy identities listed in `ClassifyNonTerminalMaskEffect` | neutral | audited infrastructure calls outside the Vector ALU mask interface |

Mask payload expressions use a deliberately smaller grammar: integer
arithmetic over lexical variables, with no `BufferLoad` or general call. The
only call node admitted there is TVM's internal `tir.large_uint_imm` encoding
of a `uint64` literal; it is not an executable helper.
