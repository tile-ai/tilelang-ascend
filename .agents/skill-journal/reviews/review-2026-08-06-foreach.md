# Skill Review Snapshot — 2026-08-06 (Updated after mish sync)

> 评审模式生成。已将 `.opencode/skills/`（mish 优化后的最新版）同步到 `.agents/skills/`。
> 基于同步后的版本重新核对，10 条 entry 中 1 条已落地（e8），9 条仍 pending 待 apply。

## 同步记录

- 同步源：`/home/developer/.config/opencode/skills/`（软链到 `/home/developer/.cannbot/repo/ops/`，mish 改动落点）
- 同步目标：`/mnt/workspace/gitCode/cann/tilelang-ascend/.agents/skills/`
- 同步的 5 个 skill：tilelang-api-best-practices（新建）、tilelang-op-design、tilelang-op-develop、tilelang-op-test-design、tilelang-perf-optimization
- 旧版备份已清理（无本地独有改动）

## 落地状态核对

| entry | target_skill | 状态 | 说明 |
|-------|--------------|------|------|
| e8 | tilelang-api-best-practices | ✅ 已落地 | api-kernel-memory.md §4.1/4.2 已含"默认 vid=0/1 算力浪费"+"按行切分模式" |
| e1 | tilelang-op-develop | ❌ pending | Subagent 返回格式约束（三态标记 + 禁乱码）未落地 |
| e7 | tilelang-op-develop | ❌ pending | cann_bench 接口签名匹配 proto.yaml 未落地 |
| e3 | tilelang-op-test-design | ❌ pending | float32 累加误差预估未落地 |
| e6 | tilelang-op-design | ❌ pending | Stage 3 优化方向优先级标注未落地 |
| e2 | tilelang-perf-optimization | ❌ pending | launch overhead 185us 框架约束未落地 |
| e9 | tilelang-perf-optimization | ❌ pending | V-compute-bound vs MTE2-bound 判定未落地（仅有"memory bound 未做流水"反模式，未提归约 num_stages=3 反效果） |
| e4 | tilelang-perf-optimization | ❌ pending | DATA COPIES vs COMPUTE ops 区分未落地 |
| e5 | tilelang-perf-optimization | ❌ pending | T.Pipelined 最小循环次数要求未落地 |
| e10 | tilelang-perf-optimization | ❌ pending | cann-bench op_times 隔离 vs 端到端未落地 |

## 评审表（9 项 pending，已排除 e8）

### tilelang-op-develop（2 项）

| # | sev | type | section | entry |
|---|-----|------|---------|-------|
| 3 | 🔴 high | missing_constraint | 工作流/返回格式约束（Subagent 返回乱码防护） | e1 |
| 4 | 🔴 high | missing_constraint | cann_bench 包打包 / 接口签名约束 | e7 |

### tilelang-op-design（1 项）

| # | sev | type | section | entry |
|---|-----|------|---------|-------|
| 2 | 🟡 low | other | §12.3 优化方向 / Stage 3 性能优化建议 | e6 |

### tilelang-op-test-design（1 项）

| # | sev | type | section | entry |
|---|-----|------|---------|-------|
| 5 | 🟠 medium | missing_constraint | L0/L1 测试用例 shape 选择 / float32 累加误差 | e3 |

### tilelang-perf-optimization（5 项）

| # | sev | type | section | entry |
|---|-----|------|---------|-------|
| 6 | 🔴 high | missing_constraint | 性能反模式排查清单 / TileLang launch overhead 185us 框架约束 | e2 |
| 7 | 🔴 high | mode_misjudgment | T.Pipelined / V-compute-bound vs MTE2-bound 算子类型判定 | e9 |
| 8 | 🟠 medium | mode_misjudgment | P0 host 反模式未区分 DATA COPIES vs COMPUTE ops | e4 |
| 9 | 🟠 medium | missing_constraint | T.Pipelined 未标注最小循环次数要求 | e5 |
| 10 | 🟠 medium | missing_constraint | cann-bench op_times 隔离测量 ≠ 端到端可恢复时间 | e10 |

## 汇总

- **已落地**: 1（e8，mish 同步带来）
- **仍 pending**: 9
- **按严重度**: 🔴 high=4, 🟠 medium=4, 🟡 low=1
- **按 skill**: tilelang-perf-optimization=5, tilelang-op-develop=2, tilelang-op-design=1, tilelang-op-test-design=1
