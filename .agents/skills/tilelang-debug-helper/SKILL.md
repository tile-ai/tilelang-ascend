---
name: tilelang-debug-helper
description: TileLang npuir 调试辅助技能。用户提及调试 npuir kernel、GDB 附加、IR dump、精度异常定位、编译失败定位、pass 阶段定位、T.print 调试、最小复现缩减时必须使用本技能。
---

# TileLang Debug Helper (npuir)

## Mandatory routing rule

Before answering, follow AGENTS.md section "Docs Auto Routing Rules (Mandatory)".

## Debug workflow

1. Reproduce with minimal script
2. Add process attach window if native debug is needed
3. Capture IR snapshots around transformation boundaries
4. Narrow down failing pass or API misuse

## For API debugging

- First verify v-prefix API usage
- Then verify alias compatibility if legacy npuir_xxx appears

## References

- references/mlir-dump-guide.md

## Official docs to consult

- docs/Tilelang算子调试指南.md
- docs/Tilelang.language/调试操作/T.print.md
- docs/developer/EnvironmentVariables.md

## Related skills

- tilelang-mlir-skill
- tilelang-error-fixer
