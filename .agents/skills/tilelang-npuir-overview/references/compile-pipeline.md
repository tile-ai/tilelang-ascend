# NPUIR Compile Pipeline

## End-to-end flow

1. Python DSL kernel definition with @tilelang.jit(target="npuir")
2. Lowering through tilelang/engine/lower.py
3. Optional pass orchestration through tilelang/tladapter
4. TileLangIR and MLIR pass application
5. Backend codegen through:
   - Expert mode: src/target/codegen_npuir_api.cc and src/target/codegen_npuir_api.h
   - Developer mode: src/target/codegen_npuir_dev.cc and src/target/codegen_npuir_dev.h
   - Deprecated: src/target/codegen_npuir.cc
6. Runtime launch integration via jit_npu workflow

## Practical checks

- Confirm target uses npuir
- Confirm vector ops prefer v-prefix aliases in generated examples
- Confirm pass failures with MLIR dump and pass isolation
