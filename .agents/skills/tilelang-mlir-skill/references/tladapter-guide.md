# TileLang Adapter Guide

## What this layer does

The `tilelang/tladapter` layer is the Python-facing adapter around the native TileLangIR pass runner.

Relevant files:

- `tilelang/tladapter/__init__.py`
- `tilelang/tladapter/utils.py`
- `tilelang/tladapter/transforms/mlir.py`
- `tilelang/tladapter/transforms/bishengir.py`
- `tilelang/tladapter/transforms/tilelangir.py`
- `tilelang/engine/lower.py`
- `tilelangir/python/TileLangIRPasses.cpp`

## Real execution path

Today the execution path is:

1. `tilelang/tladapter/__init__.py` loads `libtilelangir`.
2. `tilelangir/python/TileLangIRPasses.cpp` registers dialects and passes on module load.
3. `tilelang/tladapter/utils.py` wraps the native `PassPipeline`.
4. `tilelang/engine/lower.py` builds a Python `Pipeline` and runs it on the MLIR string produced by NPU codegen.

The important consequence is:

- registration in C++ makes a pass visible to parsing,
- a wrapper in `tilelang/tladapter/transforms/*.py` makes the pass convenient to call from Python,
- adding it in `tilelang/engine/lower.py` makes it part of the default `npuir` compile flow.

These are three different steps.

## How to expose a pass to Python

If your pass name is `tilelangir-my-pass`, add a wrapper in
`tilelang/tladapter/transforms/tilelangir.py`.

Example:

```python
from tilelang.tladapter.utils import pass_fn

cv_split = pass_fn("tilelangir-cv-split")
vectorize = pass_fn("tilelangir-vectorize")
my_pass = pass_fn("tilelangir-my-pass")
```

If the pass is anchored on a nested operation instead of `builtin.module`, add the anchor:

```python
my_func_pass = pass_fn("tilelangir-my-func-pass", anchor="func.func")
```

The wrapper name should stay short and Pythonic. The textual pass name must exactly match `Passes.td`.

## How `Pipeline` composes passes

`tilelang/tladapter/utils.py` builds textual MLIR pipeline fragments. The native runner then parses them with `mlir::parsePassPipeline(...)` inside `tilelangir/python/TileLangIRPasses.cpp`.

Examples:

```python
from tilelang.tladapter.utils import Pipeline
from tilelang.tladapter import transforms

pipeline = Pipeline()
pipeline.add(transforms.mlir.canonicalize, top_down=True)
pipeline.add(transforms.tilelangir.my_pass)
result = pipeline.run(mlir_str)
```

If you pass a raw string, it must already be valid MLIR textual pipeline syntax.

## How to wire a pass into the real compile flow

The current default integration point is `tilelang/engine/lower.py`.

As of the current source tree, the `npuir` branch builds this post-codegen pipeline:

```python
pipeline = Pipeline()
pipeline.add(transforms.mlir.canonicalize, top_down=True)
pipeline.add(transforms.bishengir.adapt_triton_kernel)
```

If you want your TileLangIR pass to run by default, insert it here in the correct order.

Example:

```python
pipeline = Pipeline()
pipeline.add(transforms.mlir.canonicalize, top_down=True)
pipeline.add(transforms.bishengir.adapt_triton_kernel)
pipeline.add(transforms.tilelangir.my_pass)
```

Order matters. Place the pass next to the stage whose invariants it depends on.

## About `buildTileLangIRCompilePipeline(...)`

`tilelangir/include/tilelangir/InitAllPasses.h` declares:

```cpp
void buildTileLangIRCompilePipeline(mlir::OpPassManager &pm);
```

There is no implementation in the current source tree.

So for current work:

- use `tilelang/engine/lower.py` when the task is "make the pass run in the actual compile flow";
- only introduce a C++ pipeline builder if the task explicitly wants that refactor.

## Debugging helpers

Useful hooks already exist:

- `Pipeline.enable_ir_printing()`
- `Pipeline.enable_ir_printing_to_file_tree(dir)`
- `TILELANG_DUMP_IR=TRUE`

These are often enough to find the first bad pass without adding temporary logging to the pass itself.
