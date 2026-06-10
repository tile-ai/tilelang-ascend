# TileLangIR Pass Authoring Notes

## Source of truth

Use the checked-in source tree as the truth:

- `tilelangir/include/tilelangir/Transforms/Passes.td`
- `tilelangir/include/tilelangir/Transforms/Passes.h`
- `tilelangir/include/tilelangir/Transforms/CMakeLists.txt`
- `tilelangir/lib/Transforms/CMakeLists.txt`
- `tilelangir/lib/Transforms/*.cpp`

Do not edit generated files under `build/`. They may be stale relative to the source tree.

## Real pass anatomy in this repo

Today the repo uses the standard MLIR TableGen flow:

1. Declare the pass in `Passes.td`.
2. Generate declarations and registration hooks through `tilelangir_transforms_incgen`.
3. Include those generated hooks from `Passes.h`.
4. Implement the pass body in `tilelangir/lib/Transforms/<PassName>.cpp` using `GEN_PASS_DEF_*`.
5. Link the `.cpp` file into `tilelangir_transforms`.
6. Let `registerAllPasses()` expose it to the CLI and Python.

The existing examples are:

- `tilelangir/lib/Transforms/CVSplit.cpp`
- `tilelangir/lib/Transforms/Vectorize.cpp`

## Step 1: Declare the pass in TableGen

Add a new definition to `tilelangir/include/tilelangir/Transforms/Passes.td`.

Example:

```tablegen
def TileLangIRMyPass : Pass<"tilelangir-my-pass", "::mlir::ModuleOp"> {
  let summary = "Short one-line summary";
  let description = [{
    Describe what the pass rewrites, what invariants it expects,
    and what invariants it guarantees afterwards.
  }];
}
```

Notes:

- Keep the textual pass name stable. This is the string used by `tilelangir-opt` and Python textual pipelines.
- Choose the operation anchor carefully. In this repo, the checked-in source currently anchors passes on `::mlir::ModuleOp`.
- If the transform is naturally function-local, you may choose a narrower anchor, but then your Python wrapper and textual pipeline must use the matching anchor.

## Step 2: Implement the pass body

Create `tilelangir/lib/Transforms/MyPass.cpp`.

Minimal skeleton:

```cpp
#include "tilelangir/Transforms/Passes.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/Debug.h"

namespace mlir {
namespace tilelangir {

#define GEN_PASS_DEF_TILELANGIRMYPASS
#include "tilelangir/Transforms/Passes.h.inc"

namespace {
#define DEBUG_TYPE "tilelangir-my-pass"

struct TileLangIRMyPass : impl::TileLangIRMyPassBase<TileLangIRMyPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();

    // Walk or pattern-rewrite the IR here.
    // If you detect an invalid precondition, emit an error and fail the pass.
    if (false) {
      module.emitError() << "reason";
      signalPassFailure();
      return;
    }
  }
};

#undef DEBUG_TYPE
} // namespace

} // namespace tilelangir
} // namespace mlir
```

Implementation guidance:

- Prefer a pass that makes one structural change well over a pass that does many unrelated rewrites.
- Use `signalPassFailure()` when an invariant is violated and continuing would produce misleading IR.
- Emit diagnostics on the most local operation you can, not only on the module.
- Keep debug output behind MLIR diagnostics or `LLVM_DEBUG`, not unconditional `llvm::errs()`.

## Step 3: Link the pass into the build

Add the new source file to `tilelangir/lib/Transforms/CMakeLists.txt`.

Example:

```cmake
add_library(tilelangir_transforms STATIC
  CVSplit.cpp
  MyPass.cpp
  Vectorize.cpp
)
```

You do not need a manual registration list in C++ as long as:

- the pass is declared in `Passes.td`,
- the generated headers are included through `Passes.h`, and
- the implementation object file is linked into `tilelangir_transforms`.

## Step 4: Understand the real registration path

In this repo, registration becomes effective through these files:

- `tilelangir/include/tilelangir/Transforms/Passes.h`
  exposes generated declarations and `registerTileLangIRPasses()`
- `tilelangir/include/tilelangir/InitAllPasses.h`
  exposes `registerAllPasses()`
- `tilelangir/tools/tilelangir-opt/tilelangir-opt.cpp`
  calls `::tilelangir::registerAllPasses()`
- `tilelangir/python/TileLangIRPasses.cpp`
  calls `tilelangir::registerAllPasses()` during module load

That means:

- after rebuild, `tilelangir-opt --tilelangir-my-pass` should parse the pass name;
- after rebuild, Python `PassPipeline` should also parse the same textual pass name.

## Step 5: Rebuild the right targets

At minimum, rebuild the generated headers, the pass library, and the tools that load it.

Typical targets:

```bash
cmake --build build --target tilelangir_transforms tilelangir tilelangir-opt
```

If you added or changed lit tests:

```bash
cmake --build build --target check-tilelangir
```

## Common mistakes

- Editing generated files in `build/` instead of `tilelangir/include/.../Passes.td`
- Registering the pass but forgetting to link the new `.cpp` into `tilelangir_transforms`
- Exposing the pass in Python but never adding it to the default pipeline
- Assuming a pass runs by default just because `tilelangir-opt` can parse it
- Testing only end-to-end and never proving the pass in isolation
