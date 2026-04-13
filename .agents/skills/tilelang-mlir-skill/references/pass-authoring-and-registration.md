# End-to-End: Write a Pass and Register It into the Compile Flow

This is the concrete workflow for adding a real TileLangIR pass in this repo.

## Goal split

Be precise about which of these goals you need:

1. The pass exists and compiles.
2. `tilelangir-opt` can parse and run it.
3. Python `Pipeline` can parse and run it.
4. The default `npuir` compile flow runs it automatically.

Each goal requires one more integration step.

## Step 0: Start from the nearest existing pattern

Check these first:

- `tilelangir/lib/Transforms/CVSplit.cpp`
- `tilelangir/lib/Transforms/Vectorize.cpp`
- `testing/mlir/Transforms/cv-split.mlir`
- `tilelang/tladapter/transforms/tilelangir.py`
- `tilelang/engine/lower.py`

If none of them is close, say so explicitly and keep the new pass minimal.

## Step 1: Declare the pass in `Passes.td`

Edit:

- `tilelangir/include/tilelangir/Transforms/Passes.td`

Example:

```tablegen
def TileLangIRMyPass : Pass<"tilelangir-my-pass", "::mlir::ModuleOp"> {
  let summary = "Rewrite X before Y";
  let description = [{
    Use this pass to normalize X before downstream pass Y consumes the IR.
  }];
}
```

This is the textual name that both `tilelangir-opt` and Python pipelines will use.

## Step 2: Implement the pass body

Create:

- `tilelangir/lib/Transforms/MyPass.cpp`

Skeleton:

```cpp
#include "tilelangir/Transforms/Passes.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace tilelangir {

#define GEN_PASS_DEF_TILELANGIRMYPASS
#include "tilelangir/Transforms/Passes.h.inc"

namespace {
struct TileLangIRMyPass : impl::TileLangIRMyPassBase<TileLangIRMyPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();

    module.walk([&](Operation *op) {
      // Rewrite or collect here.
    });
  }
};
} // namespace

} // namespace tilelangir
} // namespace mlir
```

If the pass depends on a strict precondition, emit a diagnostic and call `signalPassFailure()`.

## Step 3: Link the implementation into the pass library

Edit:

- `tilelangir/lib/Transforms/CMakeLists.txt`

Add your file to `tilelangir_transforms`.

Without this step, the pass may appear declared but still never link into the loaded library.

## Step 4: Rebuild and confirm registration

Rebuild:

```bash
cmake --build build --target tilelangir_transforms tilelangir tilelangir-opt
```

Why registration works after that:

- `Passes.td` feeds TableGen
- `Passes.h` exposes generated registration hooks
- `tilelangir/tools/tilelangir-opt/tilelangir-opt.cpp` calls `::tilelangir::registerAllPasses()`
- `tilelangir/python/TileLangIRPasses.cpp` calls `tilelangir::registerAllPasses()` at module load

There is no separate handwritten registry list to update for those two entry points.

## Step 5: Prove the pass in isolation

Add:

- `testing/mlir/Transforms/my-pass.mlir`

Pattern:

```mlir
// RUN: tilelangir-opt --tilelangir-my-pass %s | FileCheck %s
```

This confirms:

- the pass name parses
- the pass is registered
- the transform does what you think it does

## Step 6: Expose the pass to Python

Edit:

- `tilelang/tladapter/transforms/tilelangir.py`

Example:

```python
my_pass = pass_fn("tilelangir-my-pass")
```

If the pass is not a module pass, add the correct anchor:

```python
my_func_pass = pass_fn("tilelangir-my-pass", anchor="func.func")
```

## Step 7: Register it into the real compile flow

Edit:

- `tilelang/engine/lower.py`

Current ground truth:

```python
pipeline = Pipeline()
pipeline.add(transforms.mlir.canonicalize, top_down=True)
pipeline.add(transforms.bishengir.adapt_triton_kernel)
```

If you want the pass to run automatically for `target.kind.name == "npuir"`, add it here.

Example:

```python
pipeline = Pipeline()
pipeline.add(transforms.mlir.canonicalize, top_down=True)
pipeline.add(transforms.bishengir.adapt_triton_kernel)
pipeline.add(transforms.tilelangir.my_pass)
```

Registration is not enough. This step is what makes it part of the default compile flow.

## Step 8: Add a smoke test for the Python path

Use a Python test when the risk is in wrapper name, pass order, or options.

Typical checks:

- the wrapper name maps to the right textual pass name
- the anchor is correct
- the pass order in the mixed pipeline is correct

## Step 9: Add or reuse an end-to-end regression

If the pass changes the final `npuir` output in a semantically important way, add or update the closest existing test in `testing/npuir/`.

Good examples:

- operator legality
- memory placement or workspace behavior
- sync insertion
- mixed-kernel splitting
- host wrapper structure

## Debugging checklist

If the pass does not seem to run:

1. Confirm the source `.td` and `.cpp` were edited, not generated `build/` artifacts.
2. Confirm the new `.cpp` is listed in `tilelangir/lib/Transforms/CMakeLists.txt`.
3. Confirm `tilelangir-opt --tilelangir-my-pass test.mlir` parses.
4. Confirm the Python wrapper uses the exact textual pass name.
5. Confirm `tilelang/engine/lower.py` actually adds the pass if you expect default execution.

If the pass runs but produces bad IR:

1. Run it alone with `tilelangir-opt`.
2. Use `Pipeline.enable_ir_printing()` or `enable_ir_printing_to_file_tree(...)`.
3. Use `TILELANG_DUMP_IR=TRUE` for the end-to-end path.
4. Add a narrower lit test that isolates the first broken invariant.

## Important current limitation

`tilelangir/include/tilelangir/InitAllPasses.h` declares `buildTileLangIRCompilePipeline(mlir::OpPassManager &pm)`, but the current source tree does not implement it.

So today:

- `tilelangir-opt` registration is real,
- Python `PassPipeline` registration is real,
- default compile-flow integration is controlled by `tilelang/engine/lower.py`.
