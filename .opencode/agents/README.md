# TileLang-Ascend 算子开发代理体系

基于 OpenCode 多代理框架的端到端算子开发流程：从需求 → 设计 → 实现 → 精度修复 → （可选）性能调优，全程由 Orchestrator 编排，Subagent 在隔离上下文中执行各阶段。

## 适用场景

适合在 `examples/{op}/` 下新增**自定义 TileLang-Ascend 算子**的完整开发流程。**不适用于** Pass 开发、编译器内部修改、bug 修复等场景。

## 代理拓扑

```
┌─────────────────────────────────────────────────┐
│         tilelang-op-orchestrator (Primary)      │
│   流程编排 · 状态机 · 工件门禁 · 设计回退控制    │
└──┬─────────────┬──────────────┬─────────────────┘
   │ Stage 1     │ Stage 2-3    │ Stage 4 (opt)
   ▼             ▼              ▼
┌─────────┐ ┌──────────┐ ┌─────────────┐
│ analyst │ │developer │ │ perf-tuner  │
│  设计   │ │实现+精度 │ │  性能调优   │
└─────────┘ └──────────┘ └─────────────┘
   │             │              │
   ▼             ▼              ▼
[op-design] [op-generate]  [perf-optimization]
                              ← Skill 层
```

| 代理 | 角色 | 职责 |
|------|------|------|
| [tilelang-op-orchestrator](tilelang-op-orchestrator.md) | Primary | 4 阶段状态机、工件门禁、Subagent 调度、设计回退、环境预检 |
| [tilelang-op-analyst](tilelang-op-analyst.md) | Subagent | Stage 1 算子设计（含需求理解、设计回退） |
| [tilelang-op-developer](tilelang-op-developer.md) | Subagent | Stage 2 实现 + Stage 3 精度修复（含 `[DESIGN_ERROR]` 识别） |
| [tilelang-op-perf-tuner](tilelang-op-perf-tuner.md) | Subagent | Stage 4 性能调优（可选，用户确认后触发） |

## 四阶段流程

```
[env-check]              环境预检（一次性）
    ↓
[Stage 1] 算子设计       analyst + tilelang-op-design
    ↓                    产物：DESIGN.md
[Stage 2] 代码实现       developer + tilelang-op-generate
    ↓                    产物：example_{op}.py（含 kernel + golden + test 用例）
    ├─ PRECISION_PASS ─┐
    ├─ PRECISION_FAIL ─┐
    │                  ↓
[Stage 3] 精度修复       developer（无专属 skill，依赖方法学）
    ↓                    产物：history_version/{op}_impl_s3_attempt{N}.py + 修复后 impl
    └─→ PRECISION_PASS ─┐
                        ↓
              ┌──── 询问用户 ────┐
              │ 是否需要性能调优？│
              └─────┬────────┬───┘
              不需要│        │需要 + 提供调优信息
                   ↓        ↓
                SUCCESS   [Stage 4] 性能调优
                            ↓        产物：perf_tuning/
                          SUCCESS
```

### 阶段对照表

| Stage | 名称 | 必填？ | 复用 Skill | 主要产物 |
|-------|------|--------|-----------|---------|
| 0 | 环境预检 | ✅ 一次性 | `tilelang-env-check` | `env_check_passed=true` |
| 1 | 算子设计 | ✅ | `tilelang-op-design` | `DESIGN.md` |
| 2 | 代码实现 | ✅ | `tilelang-op-generate` | `example_{op}.py`（单一交付文件） |
| 3 | 精度修复 | ⭕ 仅当 PRECISION_FAIL | — | 修复后 impl + 备份 |
| 4 | 性能调优 | ⭕ 用户确认 | `tilelang-perf-optimization` | `perf_tuning/` |

## 关键机制

### 环境预检（Stage 1 前置）

Orchestrator 在进入 Stage 1 之前自动调用 `tilelang-env-check`，**一次性通过**后状态字段 `env_check_passed=true`，续跑/设计回退不重复执行。

- 自动可修复：子模块、编译产物、`source set_env.sh`
- 手动才能修复：torch / torch_npu / CANN 版本不达标（置 `BLOCKED_ENVIRONMENT`）
- 后续阶段若报 `ImportError` 等环境错误，可重置标志重新触发一次预检（逃生口）

### 设计回退（`[DESIGN_ERROR]`）

`DESIGN.md` **不视为不可质疑的硬性约束**。Developer 在 Stage 2 / Stage 3 中如发现以下情形之一，可在输出加 `[DESIGN_ERROR]` 标记触发回退：

| 触发情形 | 例子 |
|---------|------|
| API 不存在 | design 用了 `tilelang/language/` 中无导出的 API |
| L0C 容量溢出 | `block_M × block_N × sizeof(accum) > 128KB` |
| 内存层级不可实现 | design 要求 GM → L0 直接搬运跳过 L1/UB |
| 同步策略冲突 | Developer 模式 design 中要求手动同步 |
| 动态边界违规 | `T.Pipelined(batch_sizes[bz])` |
| Kernel 维度违规 | 三维 Kernel 或 threads > 2 |
| 多次精度修复后定位到设计 | Stage 3 反复失败指向 design |

回退处理：备份旧 design 到 `history_version/design_rev{N}.md` → analyst 用 `revision` 模式重做 → Stage 2 从头实现。**不设次数上限**，死循环风险由 Stage 2/3 自身的 5 次重试上限兜底。

### 性能调优用户确认

精度通过后 Orchestrator **主动询问用户**：是否需要性能调优？
- 不需要 → 直接置 `SUCCESS`
- 需要 → 询问调优必要信息（性能目标类型、目标数值、baseline、shape、噪声阈值、迭代上限），追加写入 DESIGN.md → 进入 Stage 4

### 重试与中止规则

| Stage | 重试上限 | 超限后状态 |
|-------|---------|-----------|
| 1 设计 | 3 次 | `BLOCKED_DESIGN` |
| 2 实现 | 5 次（仅运行失败累计） | `BLOCKED_IMPL` |
| 3 精度修复 | 5 次 | `BLOCKED_ACCURACY` |
| 4 性能调优 | 10 轮 / 连续 3 轮无提升 / 达成目标 | `SUCCESS` |
| 设计回退 | 无上限（由下游 retry 兜底） | — |
| 环境预检 | 一次重试 | `BLOCKED_ENVIRONMENT` |

## 使用方式

### 新建算子（最常用）

直接对 `@tilelang-op-orchestrator` 描述需求即可，Orchestrator 自动驱动全流程：

```
@tilelang-op-orchestrator 帮我开发一个 softmax 算子，公式是
softmax(x_i) = exp(x_i) / sum(exp(x_j))，输入是 [B, N] 的 float16
张量，沿最后一维做归一化。
```

Orchestrator 唯一会反向询问的关键信息：
1. **编程模式**（Developer / Expert / 混合）—— 根 [AGENTS.md](../../AGENTS.md) 强制要求
2. **精度通过后**：是否需要性能调优（可选）

其他字段（精度容忍度、动态轴范围等）有合理默认值，不主动提就用默认。

### 中断后续跑

再次对 `@tilelang-op-orchestrator` 提及算子名，它会从 `examples/{op}/.orchestrator_state.json` 自动续跑：

```
@tilelang-op-orchestrator 继续 softmax 的开发
```

环境预检不会重复执行（已通过状态保留）。

### 单独调用 Skill（跳过编排层）

如果只想用某个 skill 的能力，不走完整编排流程，可以直接调：

| 命令 | 作用 |
|------|------|
| `/tilelang-op-design` | 只生成 DESIGN.md |
| `/tilelang-op-generate` | 只生成 kernel 代码 |
| `/tilelang-perf-optimization` | 只做性能优化分析 |
| `/tilelang-env-check` | 单独跑环境检查 |

绕过 orchestrator 时，状态文件不会被写入。

## 产物目录约定

每个算子在 `examples/{op}/` 下有完整产物：

```
examples/{op}/
├── DESIGN.md                              # Stage 1
├── example_{op}.py                        # Stage 2 / 3 / 4 单一交付文件（@tilelang.jit kernel + 内嵌 golden + 几个常用 shape 的 test 用例 + main）
├── README.md                              # Stage 2（可选）
├── debug_log.md                           # Stage 2 / 3 每次调度追加一条记录
├── perf_tuning/                           # Stage 4（仅当用户启用）
│   ├── baseline_iter{N}.json
│   ├── {op}_impl_iter{N}_before.py
│   └── perf_log.md
├── history_version/                       # 两类备份
│   ├── design_rev{N}.md                   # 设计回退前的 DESIGN 备份
│   └── {op}_impl_s3_attempt{N}.py         # 精度修复前的 impl 备份
└── .orchestrator_state.json               # 状态文件（仅 orchestrator 可写）
```

### 单文件交付原则

| 文件 | 内容 |
|------|------|
| `example_{op}.py` | **单一交付文件**：`@tilelang.jit` kernel + 内嵌 PyTorch golden + 几个常用 shape 的 test 用例 + main 块（含三态标记输出） |
| `DESIGN.md` | 设计文档（11 章节：编程模式 / API 映射 / 内存规划 / Tiling / Loop / 同步 / 验证方案 / 风险点 等） |

> Golden 函数和 test 用例**全部内嵌**在 `example_{op}.py` 内，不单独成文件（符合 TileLang-Ascend 现有 examples 惯例）。Test 用例覆盖基础 / 典型 / 边界三类常用 shape。

## 状态文件

`examples/{op}/.orchestrator_state.json` 是 Orchestrator 的唯一权威状态：

```json
{
  "operator_name": "softmax",
  "env_check_passed": true,
  "current_stage": 2,
  "stage_status": {"1": "completed", "2": "in_progress"},
  "stage_retry_count": {"1": 0, "2": 0, "3": 0},
  "design_revision_count": 0,
  "perf_tuning_requested": null,
  "perf_iteration": {
    "count": 0,
    "last_improvement": 0.0,
    "consecutive_no_improvement": 0
  },
  "last_updated": "2026-05-19T00:00:00Z"
}
```

### 写入规则

- **仅 Orchestrator 可写**，Subagent 一律不得读写该文件
- 当前环境**无专用 `state_transition` 工具**，由 Orchestrator 通过 Read/Write 手动操作 JSON（详见 [orchestrator.md §状态写入接口](tilelang-op-orchestrator.md)）
- 写入时整文件覆盖（Write），不要用 Edit

## 调试与排错（FAQ）

| 现象 | 可能原因 | 处理 |
|------|---------|------|
| 流程卡在 Stage 1，反复问编程模式 | 用户输入不被识别 | 用 `Developer` / `Expert` / `混合` 三个明确词之一 |
| 反复出现 `[DESIGN_ERROR]` 回退 | analyst 收不到历史回退信息 | 检查状态文件 `design_revision_count` 是否累加；查看 `history_version/design_rev*.md` 是否齐全 |
| Stage 3 反复精度失败 | 实现层无解，应为 design 错误 | Developer 应主动返回 `[DESIGN_ERROR]`；若它没主动报，可手动跑 `/tilelang-op-design` 修正 design |
| `BLOCKED_ENVIRONMENT` | torch/torch_npu/CANN 版本不达标 | 按报告中的修复命令处理；自动可修复的（子模块/编译）已尝试 |
| 中断后想从头开始 | 不想续跑 | 删除 `examples/{op}/.orchestrator_state.json`，重新触发 orchestrator |
| 状态文件结构异常 | 字段缺失或旧格式 | Orchestrator 会自动迁移；严重情况下删除状态文件重跑 |

## 与项目根 AGENTS.md 的关系

代理体系遵循根 [AGENTS.md](../../AGENTS.md) 的 6 项核心原则：

1. 不要凭记忆猜 API
2. 从示例入手
3. 注意双层修改（Python + C++）
4. 遵循硬件内存层级
5. 优先复用、定位问题而非重写
6. 新增算子必须创建独立目录

Orchestrator 在调度 Subagent 时会在 prompt 中明确提醒这些原则，Subagent 内部的 skill 调用也会进一步落实（例如 `tilelang-op-design` 的"强制步骤 0：搜索本项目同类实现"）。

## 相关文档

- [AGENTS.md](../../AGENTS.md) — 项目级开发规范与代理体系入口
- [tilelang-op-orchestrator.md](tilelang-op-orchestrator.md) — Primary 代理完整规范
- [tilelang-op-analyst.md](tilelang-op-analyst.md) — Stage 1 设计代理
- [tilelang-op-developer.md](tilelang-op-developer.md) — Stage 2/3 开发代理
- [tilelang-op-perf-tuner.md](tilelang-op-perf-tuner.md) — Stage 4 性能调优代理
- [.agents/skills/](../../.agents/skills/) — Skill 库
