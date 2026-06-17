---
name: tilelang-op-generate
description: "基于设计文档生成 TileLang-Ascend 算子实现代码与测试。从 design.md 中提取关键信息，结合 examples/ 中的参考实现生成可运行代码。触发：实现算子、写 kernel、生成代码、算子编码、根据设计文档实现。"
---

# TileLang-Ascend 算子代码生成

基于设计文档（`design.md`）和已有示例，生成可运行的算子实现与测试。

---

## 1. 从 design.md 中提取的信息（只取这些）

design.md 可能很长，**只提取以下字段，忽略其余内容**：

| 提取字段 | 所在章节 | 用途 |
|---------|---------|------|
| 数学公式 | §1 概述 | 理解计算逻辑 |
| 算法步骤分解 | §1 算法描述 | 确定计算顺序 |
| API 映射表 | §3 API 映射设计 | **核心**：每步用哪个 TileLang API |
| 伪代码 | §3 计算伪代码 | **核心**：代码骨架 |
| 输入输出 shape 和 dtype | §4 数据规格 | 函数签名和测试数据 |
| block 大小 | §5 Tiling 策略 | 分块参数 |
| pass_configs | §7 同步策略 | JIT 配置 |
| Golden 函数 | §8 验证方案 | 测试对比基准 |
| 测试用例表 | §8 验证方案 | 测试配置 |
| 精度标准 | §8 验证方案 | atol / rtol |

**明确忽略的内容**（这些容易误导）：
- 模式选型的分析推理过程
- 内存预算的计算过程和多轮优化迭代
- 风险点与注意事项（过于笼统）
- 交付清单（仅是文件列表）
- 任何标注为"待确认"的内容

---

## 2. 参考来源（优先级高于 design.md 伪代码）

**当 design.md 伪代码与 examples/ 中同类实现有冲突时，以 examples/ 为准。**

### 2.1 API 用法和模式选择

- **API 用法**：查阅 [tilelang-api-best-practices SKILL.md](../tilelang-custom-skill/tilelang-api-best-practices/SKILL.md) 及其 references 目录
- **编程模式和 pass_configs**：查阅 [tilelang-expert-to-developer SKILL.md](../tilelang-custom-skill/tilelang-expert-to-developer/SKILL.md) 及其 references 目录

### 2.2 同类算子示例

生成代码前，必须查阅 `examples/` 中的同类算子：

| 算子类型 | 参考示例 |
|---------|---------|
| 逐元素运算（add/mul/sigmoid/relu） | `examples/elementwise/`、`examples/activation/` |
| 归约运算（reduce_sum/max/min） | `examples/reduce/` |
| 归一化（softmax/layernorm/rmsnorm） | `examples/softmax/`、`examples/normalization/` |
| GEMM | `examples/gemm/`、`examples/developer_mode/gemm_developer.py` |
| 融合算子 | `examples/flash_attention/`、`examples/pipeline/`、`examples/developer_mode/matmul_add_developer.py` |
| Developer 模式 | `examples/developer_mode/` |

查阅示例时关注：
1. **Kernel 结构**：`T.Kernel` 参数、`cid`/`vid` 用法
2. **Buffer 分配方式**：shape 和 dtype
3. **pass_configs 配置**：该类算子实际使用哪些开关
4. **数据搬运**：`T.copy` 的索引写法
5. **CV 交互**（融合算子，按模式）：Developer 默认 `threads=2` + 片上直连（无 workspace_idx）；Expert/混合或回退才看 workspace_idx、数量、shape

---

## 3. 代码生成流程

### 步骤 1：读取设计文档

读取 `design.md`，按 §1 的表格提取字段。

### 步骤 2：查找参考示例

在 `examples/` 中找到最相似的算子实现，**完整阅读其代码并记录技术决策**：

**必须记录的技术决策**（从参考实现中提取）：

| 决策项 | 示例值 | 说明 |
|--------|--------|------|
| 内存层级 API | `alloc_L1/L0C/ub`（显式）或 `alloc_shared/fragment`（自动） | 决定内存分配方式 |
| 同步策略 | 手动 `barrier_all/set_flag` 或自动同步 | 决定同步代码 |
| pass_configs | `AUTO_SYNC: True`，融合算子需 `AUTO_CV_COMBINE: True + AUTO_CV_SYNC: True` | 决定 JIT 配置 |
| 核分离方式 | `T.Scope("C"/"V")` 或无显式分离 | 决定核间协作方式 |
| CV 交互（融合算子，按模式） | Developer：`threads=2` + 单 `cid` 轴 + 片上直连（无 workspace_idx）；Expert/混合/回退：`{数量: 3, shape: [block_num, block_M, block_N], idx: [4,5,6]}` | Developer 默认消除 workspace/vid，见 mode-examples.md §6 |

**对比差异分析**（如有 design.md）：

| 项目 | design.md 方案 | 参考实现方案 | 选择理由 |
|------|---------------|-------------|---------|
| 内存层级 API | | | |
| 同步策略 | | | |
| pass_configs | | | |
| CV 交互 ⭐（Developer 默认 threads=2 片上直连 / 回退 workspace+vid） | | | |

**冲突处理**：当 design.md 与参考实现冲突时：
- **优先参考实现**：参考实现已验证通过，可信度高
- **记录差异**：在代码注释中说明为何偏离 design.md
- **询问用户**：重大差异需确认

### 步骤 3：生成实现代码

基于 design.md 的 API 映射 + 参考示例的代码风格，生成 `example_{op}.py`。完整文件结构骨架与融合算子注意事项见 [examples/code-skeleton.md](examples/code-skeleton.md)。

> **写代码时遇到**具体编码规范问题（Buffer 分配 / 索引一致性 / 同步 / 广播 / 测试模板）查 [references/coding-conventions.md](references/coding-conventions.md)。
>
> **V 核并行化**（按行切分、中间 buffer 索引一致性、CV 融合 V 核切分）查 [references/vector-parallelism.md](references/vector-parallelism.md)。
>
> **含 GEMM 或 CV 融合**时查 [references/gemm-cv-fusion.md](references/gemm-cv-fusion.md)（gemm_v0 初始化、NPU 分形限制、CV 融合必开的 4 个 pass_configs）。

### 步骤 4：运行验证

本 skill 只负责 L0（精度收敛）。先只跑 L0：

```bash
python examples/{op}/example_{op}.py --level l0
```

> L0 通过后，由 `tilelang-op-test-design`（场景 B）填充 L1/L2/Boundary 桩体，再 `--level all` 跑全量。
> main 分发器与 `--level` 接口由本 skill 生成并保持稳定（模板见 code-skeleton.md），扩展时不改动。

如果报错，查阅 [references/troubleshooting.md](references/troubleshooting.md) 进行排查：

| 错误类型 | 排查方向 | 详细参考 |
|---------|---------|---------|
| 编译错误 | buffer 大小、API 参数、对齐 | troubleshooting.md §编译时错误 |
| 运行错误 | 索引越界、同步缺失 | troubleshooting.md §运行时错误 |
| 精度错误 | Golden 实现、输出形状 | troubleshooting.md §精度问题 |

> **遇到具体错误信息时**，先查 [references/troubleshooting.md](references/troubleshooting.md) ——本 skill 配套的疑难解答手册，覆盖编译错误（UB 内存不足 / threads / 动态循环边界）、运行错误（index OOB / valid_shape）、精度错误（dtype / atol 阈值）等常见场景的具体解决方案。

### 步骤 5：上库前检查清单

运行通过后，必须按 [references/checklist.md](references/checklist.md) 全部 22 项检查。**最容易踩坑的 4 项重点提醒**：

| 关键项 | 说明 | checklist 编号 |
|--------|------|---------|
| **Golden 实现一致** | 迁移算子必须使用原算子的 golden 实现 | #9 |
| **tilelang.disable_cache()** | 放在 `__main__` 下方或 `main()` 内部 | #11 |
| **分层标记 + --level** | L0/L1 打 `[PRECISION_PASS/FAIL]`、L2/Boundary 打 `[BOUNDARY_PASS/WARN]`；main 支持 `--level`；L0/L1 全过才 `"Test Passed!"`+exit 0 | #14-17 |
| **代码格式** | `ruff check` + `ruff format --check` 通过 | #18 |

---

## 4. Skill 反馈采集

**算子开发流程跑完后**触发，把"哪些 skill 没讲清楚 / 被现实打脸 / 凭经验补的内容"写到 `.agents/skill-journal/`。

⚠️ **触发权归属取决于调用模式**（orchestrator 编排时不主动触发，单独调用时手动触发）。完整触发规则、枚举 skill、反思四问、写 journal schema、自检、完成报告见 [references/skill-feedback.md](references/skill-feedback.md)。

---

## 子目录索引

- [references/coding-conventions.md](references/coding-conventions.md) — Buffer 分配 / 索引 / 同步 / 广播 / 测试模板（写代码遇到具体规范时查）
- [references/vector-parallelism.md](references/vector-parallelism.md) — V 核并行化（用到 vid 切分时查）
- [references/gemm-cv-fusion.md](references/gemm-cv-fusion.md) — GEMM 与 CV 融合 pass_configs（含 GEMM 或融合算子时查）
- [references/checklist.md](references/checklist.md) — 22 项上库前检查清单（生成代码后逐项过）
- [references/troubleshooting.md](references/troubleshooting.md) — 编译 / 运行 / 精度错误排查手册（遇到具体错误时查）
- [references/skill-feedback.md](references/skill-feedback.md) — Skill 反馈采集流程（流程结束时查，orchestrator 模式跳过）
- [examples/code-skeleton.md](examples/code-skeleton.md) — example_{op}.py 文件结构骨架
