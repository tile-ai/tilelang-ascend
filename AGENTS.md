# TileLang npuir Agent Guide

This repository uses AGENTS skills for TileLang NPUIR development.

## Scope

The skills in .agents/skills are designed for target="npuir" workflows.
They prioritize v-prefix vector APIs such as vadd, vmul, vexp, vcast, vbrc.
Legacy npuir_xxx APIs remain valid as compatibility aliases.

## API Convention (Mandatory)

- Prefer v-prefix APIs in new examples and generated code.
- Keep compatibility with npuir_xxx when reading existing code.
- If both forms are available, output should default to v-prefix.

Examples:
- Prefer: T.vmul(A, B, C)
- Compatible: T.npuir_mul(A, B, C)

## Skill Index

1. tilelang-npuir-overview
Purpose: architecture and compile pipeline for npuir branch.

2. tilelang-vector-skill
Purpose: vector operator generation with v-prefix API style.

3. tilelang-cube-skill
Purpose: cube operator generation with load_nd2nz and store_fixpipe.

4. tilelang-mixcv-skill
Purpose: mixed Cube+Vector kernels such as flash attention pipelines.

5. tilelang-mlir-skill
Purpose: TileLangIR and MLIR pass workflow and debugging.

6. tilelang-debug-helper
Purpose: GDB + IR dump + pass-level debug workflow for npuir.

7. tilelang-error-fixer
Purpose: diagnosis and repair workflow for compile/runtime/pass failures.

8. tilelang-review-skill
Purpose: risk-first code review and CI-aligned format checks.

9. tilelang-github-operations
Purpose: npuir branch commit/rebase/PR/issue workflow.

## Trigger Guidance

Use the matching skill whenever the user asks for:
- npuir kernel writing, performance tuning, vector math, cube gemm, mixed kernels
- pass debugging, IR dump, MLIR transform troubleshooting
- compile/runtime error analysis on npuir branch
- code review, lint/format checks, PR readiness
- commit/push/rebase/upstream sync, PR or issue workflow

Developer-mode MixCV trigger rule:
- If one kernel contains Cube-side T.gemm and Vector-side at least one v-prefix op (such as T.vadd/T.vmul/T.vexp/T.vcast/T.vbrc), treat it as MixCV and use tilelang-mixcv-skill.

## Operator Implementation Baseline (Mandatory)

For operator-writing tasks, always start from existing examples and tests:

- First consult examples/ and testing/npuir/ for the closest existing pattern.
- Prefer modifying an existing operator case instead of generating a brand-new kernel from scratch.
- If no close template exists, explicitly state that and then build the minimal new kernel.

## Pre-PR Formatting Rule (Mandatory)

Before creating or updating a PR, run format validation for changed files from repository root:

- bash format.sh --files changed_files

Notes:
- This is a required self-check for clean code and style consistency.
- The changed_files placeholder represents the file list modified in the current branch.

## Docs Auto Routing Rules (Mandatory)

When any skill answers technical questions, it must route references by docs directory first.

### Routing Priority

1. docs/Tilelang.language/ (API semantics and signatures)
2. docs/Tilelang算子调试指南.md (debug and issue localization)
3. docs/developer/ (runtime and environment variables)
4. docs/开发指南.md and docs/快速入门.md (workflow and onboarding)
5. docs/Tilelang-Ascend贡献指南.md (PR, issue, contribution process)

### Keyword to Docs Mapping

- Vector ops (vadd/vmul/vexp/vcast/vbrc/reduce/sigmoid/rmsnorm):
    docs/Tilelang.language/数学操作/
    docs/Tilelang.language/数据类型转换操作/
    docs/Tilelang.language/shape操作/
    docs/Tilelang.language/规约操作/

- Cube ops (gemm/load_nd2nz/store_fixpipe/L1/L0C/NZ):
    docs/Tilelang.language/线性代数操作/
    docs/Tilelang.language/内存操作/

- Pipeline and sync (sync_block_set/wait/pipe_barrier/set_flag/wait_flag):
    docs/Tilelang.language/同步管道操作/

- Debug, compile failure, runtime failure, precision issue:
    docs/Tilelang算子调试指南.md
    docs/Tilelang.language/调试操作/

- MLIR, pass, tilelangir, bishengir-compile:
    docs/Tilelang算子调试指南.md
    docs/developer/EnvironmentVariables.md

- Runtime target, mode switch, env setup:
    docs/developer/npu runtime.md
    docs/developer/EnvironmentVariables.md
    docs/安装指南.md

- PR, rebase, commit, issue, CI workflow:
    docs/Tilelang-Ascend贡献指南.md

### Conflict Resolution

- If multiple mappings match, select by priority and keep at most 3 primary doc references.
- Always include at least 1 concrete API doc under docs/Tilelang.language/ when the question is API-related.
- If API docs and examples differ, API docs are source of truth and examples are secondary.
