---
name: tilelang-cube-skill
description: TileLang npuir Cube 算子开发指南。用户提及 GEMM、matmul、batch gemm、L1/L0C、load_nd2nz、store_fixpipe、NZ 格式、Cube scope、矩阵分块与流水优化时必须使用本技能。
---

# TileLang Cube Skill

## Mandatory routing rule

Before answering, follow AGENTS.md section "Docs Auto Routing Rules (Mandatory)".

## Operator baseline rule (Mandatory)

- Before writing a new cube operator, first check examples/ and testing/npuir/.
- Prefer adapting an existing operator case rather than writing from scratch.

## Primary use cases

- matmul and batched matmul kernels
- cube-heavy stages in mixed kernels
- explicit L1 and L0C memory usage

## Core APIs

- T.alloc_shared (Developer mode)
- T.alloc_L1 (Expert mode only)
- T.alloc_L0C (Expert mode only)
- T.load_nd2nz (Expert mode only)
- T.gemm
- T.store_fixpipe (Expert mode only)

## Minimal flow

1. Partition blocks for M and N
2. Load global tiles with load_nd2nz in Expert mode or T.copy in Developer mode
3. Accumulate with T.gemm(initC controlled by k-loop)
4. Store outputs with store_fixpipe in Expert mode or T.copy in Developer mode

## NZ format rule

- NZ format path is Expert mode only.
- In Developer mode kernels, keep ND layout and use T.copy-based data movement.

## References

- references/api-cube.md
- references/examples-matmul.md
- references/nz-format.md

## Official docs to consult

- docs/Tilelang.language/内存操作/T.alloc_shared.md
- docs/Tilelang.language/线性代数操作/T.gemm.md
- docs/Tilelang.language/内存操作/T.load_nd2nz.md
- docs/Tilelang.language/内存操作/T.store_fixpipe.md
- docs/Tilelang.language/内存操作/T.alloc_L1.md
- docs/Tilelang.language/内存操作/T.alloc_L0C.md

## Related skills

- tilelang-vector-skill
- tilelang-mixcv-skill
- tilelang-debug-helper
