# npuir Architecture

## Core modules

- tilelang/language: DSL and API surface, including v-prefix aliases
- tilelang/jit/jit_npu.py: target=npuir JIT compile flow
- tilelang/engine/lower.py: high-level lowering pipeline
- tilelang/tladapter: adapter for transformation pipelines
- tilelangir: MLIR dialect, pass definitions, and opt tool
- src/target/codegen_npuir_api.cc and src/target/codegen_npuir_api.h: Expert mode NPUIR codegen implementation
- src/target/codegen_npuir_dev.cc and src/target/codegen_npuir_dev.h: Developer mode NPUIR codegen implementation
- src/target/codegen_npuir.cc: deprecated backend file

## Key directories

- tilelangir/include/tilelangir/Transforms/Passes.td
- tilelangir/lib/Transforms/CVSplit.cpp
- tilelangir/lib/Transforms/Vectorize.cpp
- tilelang/tladapter/transforms/tilelangir.py
