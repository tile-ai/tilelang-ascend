---
name: tilelang-mlir-skill
description: TileLang npuir 的 TileLangIR 和 MLIR pass 工作流技能。用户提及 tilelangir、mlir、pass pipeline、cv_split、vectorize、IR dump、pass 前后对比、transform 调试、tilelangir-opt 或 BishengIR pass 失败时必须使用本技能。
---

# TileLang MLIR Skill

## Mandatory routing rule

Before answering, follow AGENTS.md section "Docs Auto Routing Rules (Mandatory)".

## What this skill handles

- tilelangir pass understanding and usage
- pass pipeline composition and isolation
- mlir file level troubleshooting

## Test baseline (Mandatory)

- Prioritize examples/ and testing/npuir/ as the primary correctness baseline.
- Do not treat unittest/npuir/mlir_files as the primary validation source.

## Known pass entry points

- tilelang/tladapter/transforms/tilelangir.py
- pass names: tilelangir-cv-split and tilelangir-vectorize

## References

- references/mlir-test-guide.md
- references/tilelangir-pass.md
- references/tladapter-guide.md

## Official docs to consult

- docs/Tilelang算子调试指南.md
- docs/developer/EnvironmentVariables.md

## Related skills

- tilelang-debug-helper
- tilelang-error-fixer
