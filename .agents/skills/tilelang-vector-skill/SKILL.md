---
name: tilelang-vector-skill
description: TileLang npuir Vector 算子开发指南。用户提及逐元素、激活函数、归约、广播、sigmoid、rmsnorm、softmax 子流程、vadd/vmul/vexp/vcast/vbrc、向量精度或向量性能优化时必须使用本技能。默认输出必须优先采用 v 前缀 API，而非 npuir_xxx 形式。
---

# TileLang Vector Skill (npuir)

## Mandatory routing rule

Before answering, follow AGENTS.md section "Docs Auto Routing Rules (Mandatory)".

## Operator baseline rule (Mandatory)

- Before writing a new vector operator, first check examples/ and testing/npuir/.
- Prefer adapting an existing operator case rather than writing from scratch.

## API style policy

Mandatory default style:
- Prefer T.vadd, T.vsub, T.vmul, T.vdiv
- Prefer T.vexp, T.vln, T.vsqrt, T.vrsqrt, T.vrelu, T.vsigmoid
- Prefer T.vcast, T.vbrc, T.vcmp, T.vselect

Compatibility:
- T.npuir_add and friends are allowed only for compatibility with legacy code.

## Core workflow

1. Define shape and block strategy
2. Allocate UB or shared buffers based on mode
3. Copy in, compute with v-prefix APIs, copy out
4. Validate against torch reference

## References

- references/api-quickref.md
- references/examples.md
- references/troubleshooting.md

## Official docs to consult

- docs/Tilelang.language/数学操作/T.vadd.md
- docs/Tilelang.language/数学操作/T.vmul.md
- docs/Tilelang.language/数学操作/T.vexp.md
- docs/Tilelang.language/数据类型转换操作/T.vcast.md
- docs/Tilelang.language/shape操作/T.vbrc.md
- docs/Tilelang.language/规约操作/T.reduce.md

## Related skills

- tilelang-cube-skill
- tilelang-mixcv-skill
- tilelang-debug-helper
