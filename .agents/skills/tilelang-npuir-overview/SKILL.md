---
name: tilelang-npuir-overview
description: TileLang npuir 分支总体架构与编译链路指南。用户提及 npuir 分支结构、target=npuir、编译流程、lower/codegen 链路、Developer/Expert 模式、tladapter、tilelangir、bishengir-compile、环境变量时必须使用本技能。
---

# TileLang NPUIR Overview

## What this skill provides

- npuir branch architecture map
- compilation pipeline from Python DSL to NPUIR codegen
- mode selection guidance for Developer and Expert
- module role mapping for tilelangir and tladapter

## Mandatory routing rule

Before answering, follow AGENTS.md section "Docs Auto Routing Rules (Mandatory)".

## Architecture map

- Frontend DSL: tilelang/language
- JIT entry: tilelang/jit/jit_npu.py
- Lowering entry: tilelang/engine/lower.py
- Adapter layer: tilelang/tladapter
- MLIR dialect and passes: tilelangir
- Backend codegen (Expert mode): src/target/codegen_npuir_api.cc and src/target/codegen_npuir_api.h
- Backend codegen (Developer mode): src/target/codegen_npuir_dev.cc and src/target/codegen_npuir_dev.h
- Deprecated backend file: src/target/codegen_npuir.cc

## Mode selection

- Developer mode: concise implementation, compiler-managed behavior
- Expert mode: explicit Scope control and fine-grained memory/sync

Common mode switch:
- os.environ["TILELANG_ASCEND_MODE"] = "Developer"

## References to read on demand

- references/arch.md
- references/compile-pipeline.md
- references/modes.md
- references/env-setup.md

## Official docs to consult

- docs/快速入门.md
- docs/开发指南.md
- docs/developer/EnvironmentVariables.md
- docs/developer/npu runtime.md

## Related skills

- tilelang-vector-skill
- tilelang-cube-skill
- tilelang-mixcv-skill
- tilelang-mlir-skill
