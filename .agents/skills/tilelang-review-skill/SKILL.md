---
name: tilelang-review-skill
description: TileLang npuir 代码审查与格式校验技能。用户提及 review、代码审查、PR 前检查、lint、format、ruff、clang-format、规范检查、CI 不通过时必须使用本技能。优先识别行为回归、数值风险、同步风险与测试缺口，其次才是风格问题。
---

# TileLang Review Skill

## Mandatory routing rule

Before answering, follow AGENTS.md section "Docs Auto Routing Rules (Mandatory)".

## Scope

- pre-PR code review for npuir branch
- format and lint checks aligned with CI
- risk-focused review for correctness, performance, and synchronization

## Review priorities

1. Behavior regressions
2. Precision and dtype risks
3. Synchronization and pipeline hazards
4. Missing tests
5. Style and format consistency

## Docs to consult first

- docs/Tilelang-Ascend贡献指南.md
- docs/Tilelang算子调试指南.md
- docs/开发指南.md

## References

- references/checklist.txt

## Related skills

- tilelang-error-fixer
- tilelang-debug-helper
