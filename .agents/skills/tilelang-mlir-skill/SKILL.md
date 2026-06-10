---
name: tilelang-mlir-skill
description: Use for TileLangIR and MLIR pass work on the npuir branch, especially pass authoring, pass registration, pipeline integration, tilelangir-opt, IR dumps, per-pass debugging, and TileLangIR/BishengIR transform failures.
---

# TileLang MLIR Skill

Use this skill when the task involves `tilelangir`, MLIR passes, textual pass pipelines, `tilelangir-opt`, Python `PassPipeline`, pass-by-pass IR comparison, or registration of a new pass into the `npuir` flow.

## Docs-first routing

Before reading examples or code, route through repo docs in this order:

1. `docs/developer/TileLangIR架构设计.md`
   Use for the architecture, ownership boundaries, and the intended TileLangIR position in the `npuir` compile stack.
2. `docs/Tilelang算子调试指南.md`
   Use for IR dump strategy, pass-stage isolation, and compile/runtime debugging workflow.
3. `docs/developer/EnvironmentVariables.md`
   Use for `TILELANG_DUMP_IR`, mode selection, and environment-dependent behavior.
4. `docs/Tilelang.language/调试操作/T.print.md`
   Use only when kernel-side debug instrumentation is relevant.

Keep at most 3 primary doc references in an answer. If docs and examples disagree, docs win.

## What this skill covers

- Writing a new TileLangIR pass
- Registering a pass so that `tilelangir-opt` and Python can see it
- Exposing a pass through `tilelang/tladapter/transforms/`
- Wiring a pass into the real `npuir` compile flow
- Running and debugging passes one by one
- Validating a pass with lit tests, Python smoke tests, and `testing/npuir/`

## Repo ground truth

- Pass definitions live in `tilelangir/include/tilelangir/Transforms/Passes.td`.
- Pass declarations and registration hooks are generated into `Passes.h.inc` and surfaced through `tilelangir/include/tilelangir/Transforms/Passes.h`.
- Pass implementations live in `tilelangir/lib/Transforms/*.cpp`.
- `tilelangir/tools/tilelangir-opt/tilelangir-opt.cpp` calls `::tilelangir::registerAllPasses()`, which is why registered passes become visible to the CLI.
- `tilelangir/python/TileLangIRPasses.cpp` also calls `tilelangir::registerAllPasses()`, which is why registered passes become visible to Python `PassPipeline`.
- Python pass wrappers live in `tilelang/tladapter/transforms/`.
- The current default `npuir` post-codegen pipeline is assembled in `tilelang/engine/lower.py`.

Important: in the current source tree, `tilelang/engine/lower.py` only adds `canonicalize` and `adapt-triton-kernel` after NPU codegen. A TileLangIR pass being registered does not mean it already runs in the default compile flow.

Also important: `tilelangir/include/tilelangir/InitAllPasses.h` declares `buildTileLangIRCompilePipeline(mlir::OpPassManager &pm)`, but there is no implementation in the source tree today. If the task is "register the pass into the real compile flow", use `tilelang/engine/lower.py` as the current integration point unless the task explicitly asks you to introduce that missing C++ pipeline builder.

## Required workflow

1. Start from the closest existing pattern in `examples/`, `testing/npuir/`, and `testing/mlir/Transforms/`.
2. Treat the checked-in source tree as the source of truth. Do not edit generated files under `build/`.
3. If the task is authoring a pass, read `references/tilelangir-pass.md`.
4. If the task is exposing or composing passes from Python, read `references/tladapter-guide.md`.
5. If the task is about validation or debugging, read `references/mlir-test-guide.md`.
6. If the task is end-to-end "write a pass and register it into the compile flow", read `references/pass-authoring-and-registration.md`.
7. Validate from smallest scope to largest scope:
   - `tilelangir-opt` / lit test
   - Python `Pipeline` smoke test
   - `testing/npuir/` regression or an `examples/` reproducer

## Test baseline

- Prioritize `examples/`, `testing/npuir/`, and `testing/mlir/` as the correctness baseline.
- Treat `unittest/npuir/mlir_files` as deprecated and not the primary validation target.

## References

- references/mlir-test-guide.md
- references/tilelangir-pass.md
- references/tladapter-guide.md
- references/pass-authoring-and-registration.md

## Working rules

- Be explicit about whether a change is only "registered", "Python-exposed", or actually "in the default compile flow".
- Prefer a minimal pass that proves the transform works before threading it into the full `npuir` pipeline.
- When debugging, isolate the first bad pass instead of reasoning from the final broken IR.
- When a pass affects user-visible operator behavior, pair the MLIR-level test with at least one `testing/npuir/` regression.

## Related skills

- `tilelang-debug-helper` for pass-stage debugging and IR dump workflows
- `tilelang-error-fixer` for compile/runtime failures after a pass change
