# MLIR Test Guide

## Validation order

Validate a new pass from the smallest scope to the largest scope:

1. `tilelangir-opt` or a lit test under `testing/mlir/Transforms/`
2. Python `Pipeline` smoke test through `tilelang/tladapter`
3. End-to-end regression under `testing/npuir/` or a minimal reproducer under `examples/`

Do not start with a full kernel regression if the pass itself has not been proven in isolation.

## Primary test locations

- `testing/mlir/Transforms/`
- `testing/npuir/`
- `examples/`

Treat `unittest/npuir/mlir_files` as deprecated and not the primary correctness baseline.

## 1. Lit test for the pass itself

Add a focused `.mlir` test under `testing/mlir/Transforms/`.

Example pattern:

```mlir
// RUN: tilelangir-opt --tilelangir-my-pass %s | FileCheck %s

// CHECK-LABEL: func.func @example
// CHECK: expected rewritten op
func.func @example(...) {
  ...
}
```

Why this matters:

- it proves that the textual pass name is registered;
- it proves the transform itself independent of Python and full codegen;
- it gives a stable regression target for IR shape changes.

## 2. Python smoke test for the adapter layer

Use Python when the risk is in wrapper text, pass options, or pipeline ordering.

Minimal shape:

```python
from tilelang.tladapter.utils import Pipeline
from tilelang.tladapter import transforms

pipeline = Pipeline()
pipeline.add(transforms.tilelangir.my_pass)
result = pipeline.run(mlir_str)
```

Good uses:

- verifying that `pass_fn(...)` uses the right textual pass name
- verifying anchors such as `func.func(...)`
- verifying pass order inside a mixed MLIR/BishengIR/TileLangIR pipeline

## 3. End-to-end `npuir` regression

If the pass changes operator semantics, scheduling, memory placement, sync structure, or final lowering behavior, add a regression in `testing/npuir/` or reuse the nearest existing test as the baseline.

Prefer modifying the closest existing case before creating a brand-new kernel.

## Debug workflow

Use these tools in this order:

1. `tilelangir-opt --tilelangir-my-pass test.mlir`
2. `Pipeline.enable_ir_printing()`
3. `Pipeline.enable_ir_printing_to_file_tree(...)`
4. `TILELANG_DUMP_IR=TRUE`

Useful commands:

```bash
cmake --build build --target tilelangir tilelangir-opt check-tilelangir
```

```bash
TILELANG_DUMP_IR=TRUE python testing/npuir/<your_test>.py
```

## What to inspect

- whether the pass name parses at all
- whether the pass runs on the operation type it was declared for
- the first IR stage that diverges from expectation
- structural invariants: op ordering, region nesting, memref/tensor boundaries, sync placement

## When to use `T.print`

Use `docs/Tilelang.language/调试操作/T.print.md` only when the problem has escaped the MLIR stage and you need kernel-side runtime evidence. It is not the first tool for pass authoring.
