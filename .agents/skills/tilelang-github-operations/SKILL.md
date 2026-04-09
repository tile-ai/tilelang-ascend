---
name: tilelang-github-operations
description: TileLang npuir 分支 GitHub 工作流技能。用户提及 commit、push、PR、rebase、upstream、issue、GitHub Actions、gh CLI、分支同步时必须使用本技能。默认遵循 npuir 分支协作规范并提示 Issue 标题使用 [AscendNPU-IR] 或 [npuir] 前缀。
---

# TileLang GitHub Operations Skill

## Mandatory routing rule

Before answering, follow AGENTS.md section "Docs Auto Routing Rules (Mandatory)".

## Scope

- branch sync and rebase workflow for npuir
- commit and push sequence
- pull request creation and readiness checks
- issue and PR metadata conventions

## Workflow baseline

1. Sync with upstream npuir
2. Run pre-PR format validation from repo root: bash format.sh --files changed_files
3. Commit focused changes
4. Push branch and create PR
5. Verify CI status and address feedback

## Docs to consult first

- docs/Tilelang-Ascend贡献指南.md
- docs/developer/EnvironmentVariables.md

## References

- references/pr-workflow.txt
- references/issue-template.txt

## Related skills

- tilelang-review-skill
- tilelang-error-fixer
