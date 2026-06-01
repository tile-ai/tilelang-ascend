# 信息收集与信息源优先级

## 目录

- [1. 强制步骤 0：搜索本项目同类实现](#1-强制步骤-0搜索本项目同类实现)
- [2. 信息收集步骤](#2-信息收集步骤)
- [3. 禁止行为](#3-禁止行为)
- [4. 信息源优先级](#4-信息源优先级)
- [5. 冲突处理原则](#5-冲突处理原则)

---

## 1. 强制步骤 0：搜索本项目同类实现

在生成 design.md 前，**必须**执行以下工具调用：

```bash
# 1. 搜索同类算子（根据算子名称）
glob examples/**/*{算子名称}*.py
glob examples/**/*{算子类别}*.py  # 如 gemm, softmax, reduce

# 2. 如果找到同类实现，完整阅读
read examples/{找到的同类实现路径}

# 3. 检查关键技术点
grep "T.Kernel" examples/{同类实现}     # Kernel 维度
grep "T.gemm\|T.gemm_v0" examples/{同类实现}  # GEMM API
grep "T.alloc" examples/{同类实现}      # 内存分配方式
grep "T.Scope\|T.barrier" examples/{同类实现}  # 同步方式
```

## 2. 信息收集步骤

1. 查阅 `examples/` 中同类算子实现（**强制步骤 0**）
2. 查阅 `.agents/skills/tilelang-custom-skill/tilelang-api-best-practices/SKILL.md` 确认 API 可用性和用法
3. 查阅 `.agents/skills/tilelang-custom-skill/tilelang-expert-to-developer/SKILL.md` 确认编程模式和 pass_configs 配置
4. 如有参考实现，分析其计算步骤（**仅用于理解数学逻辑，不可直接使用 API**）

## 3. 禁止行为

- ❌ 在没有执行强制步骤 0 的情况下，直接使用外部参考实现的 API
- ❌ 凭记忆猜测 API 名称或参数
- ❌ 使用 `threads > 2` 的参数配置

## 4. 信息源优先级

| 优先级 | 信息源 | 用途 | 说明 |
|--------|--------|------|------|
| **0** | **本项目 `examples/` 同类实现** | **主要参考：API、编程模式、Kernel 结构** | **最权威**，直接可用 |
| 1 | `docs/TileLang-Ascend Programming Guide.md` | API 完整说明 | 补充细节 |
| 2 | `.agents/skills/tilelang-custom-skill/tilelang-api-best-practices/SKILL.md` | API 速查 | 快速确认 |
| 3 | `testing/python/language/` | 边界用法和测试模式参考 | 测试参考 |
| 4 | **外部参考实现** | **仅用于理解数学逻辑** | **不可直接使用 API** |
| 5 | `.agents/skills/tilelang-custom-skill/tilelang-expert-to-developer/SKILL.md` | 模式选择 | 辅助决策 |
| 6| `tilelang/language/__init__.py` + `tilelang/language/*.py` | 公开 API 导出关系与前端定义 | API 定义 |
| 7 | `src/op/` + `src/target/` | lowering 与后端实现状态 | 实现验证 |


**规则**：当信息源之间矛盾时，以 `examples/` 为准。若 `examples/` 未覆盖，以 `docs/` 为准。若 `docs/` 未覆盖，以 `tilelang/language/` 源码实际实现为准。
