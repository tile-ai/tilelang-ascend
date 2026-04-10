---
name: tilelang-mixcv-skill
description: TileLang npuir 混合 Cube+Vector 算子开发技能。用户提及 flash attention、mixcv、online softmax、流水并行、sync_block_set/wait、Scope("Cube")+Scope("Vector")、PIPE_FIX、跨核 workspace 协同或融合算子性能调优时必须使用本技能。Developer 模式下，只要同一 kernel 同时包含 Cube 中的 T.gemm 与 Vector 中任意一个 v 前缀算子（如 vadd/vmul/vexp/vcast/vbrc），也必须触发本技能。
---

# TileLang MixCV Skill

## Mandatory routing rule

Before answering, follow AGENTS.md section "Docs Auto Routing Rules (Mandatory)".

## Operator baseline rule (Mandatory)

- Before writing a new MixCV operator, first check examples/ and testing/npuir/.
- Prefer adapting an existing operator case rather than writing from scratch.

## Focus

- mixed kernels with both Cube and Vector stages
- staged producer-consumer synchronization
- flash-attention-like patterns

## Developer mode identification rule (Mandatory)

- In Developer mode, classify as MixCV when both conditions are true in the same kernel:
    1) Cube-side compute contains T.gemm.
    2) Vector-side compute contains at least one v-prefix op (for example T.vmul, T.vadd, T.vexp, T.vcast, T.vbrc).
- If both conditions hold, route to this skill even if the user does not explicitly say "mixcv".

## Key primitives

- T.Scope("Cube") and T.Scope("Vector")
- T.rs("PIPE_FIX") and other pipe regions
- T.sync_block_set and T.sync_block_wait
- Pipelined loops where suitable

## References

- references/pipeline.md
- references/flash-attn-pattern.md
- references/flash-attn-dev.md

## Official docs to consult

- docs/Tilelang.language/同步管道操作/T.sync_block_set.md
- docs/Tilelang.language/同步管道操作/T.sync_block_wait.md
- docs/Tilelang.language/同步管道操作/T.pipe_barrier.md
- docs/Tilelang.language/线性代数操作/T.gemm.md
- docs/Tilelang.language/数学操作/T.vexp.md

## Related skills

- tilelang-cube-skill
- tilelang-vector-skill
- tilelang-debug-helper
