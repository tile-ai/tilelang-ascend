---
name: tilelang-op-develop
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
| Golden 函数 | §9.1 Golden 函数 | 测试对比基准 |
| 测试用例表 | §9.2 L0 门槛测试计划 | 测试配置 |
| 精度标准 | §9.3 精度标准 | 混合容差：atol / rtol / max_abs_error_limit / required_matched_ratio（按 dtype） |
| 路径性能可行性表 | §5/§6 | GM pass、DMA transaction、GM 标量访问、地址计算和并行度 |
| 性能可行性哨兵 | §9 | 每条路径最坏 dtype/最大任务数 case 与单 case 超时预算 |

**明确忽略的内容**（这些容易误导）：
- 模式选型的分析推理过程
- 内存预算的计算过程和多轮优化迭代
- 仅忽略没有量化证据的笼统风险；凡是包含具体 shape、dtype、超时、GM/DMA
  成本或回退路径的风险必须提取并作为验收约束
- 交付清单（仅是文件列表）
- 任何标注为"待确认"的内容

---

## 2. 参考来源（优先级高于 design.md 伪代码）

**当 design.md 伪代码与 examples/ 中同类实现有冲突时，以 examples/ 为准。**

### 2.1 API 用法和模式选择

- **API 用法**：查阅 [tilelang-api-best-practices SKILL.md](../tilelang-custom-skill/tilelang-api-best-practices/SKILL.md) 及其 references 目录
- **编程模式和 pass_configs**：查阅 [tilelang-programming-model-guide SKILL.md](../tilelang-custom-skill/tilelang-programming-model-guide/SKILL.md) 及其 references 目录

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
| transpose / layout transform | `examples/transpose/transpose.py`（提取结构谓词、
连续 suffix-record 聚合搬运和通用 fallback；不得照抄具体 perm/shape 分支） |

查阅示例时关注：
1. **Kernel 结构**：`T.Kernel` 参数、`cid`/`vid` 用法
2. **Buffer 分配方式**：shape 和 dtype
3. **pass_configs 配置**：该类算子实际使用哪些开关
4. **数据搬运**：`T.copy` 的索引写法
5. **CV 交互**（融合算子，按模式）：Developer 默认 `threads=2` + 片上直连（无 workspace_idx）；Expert/混合或回退才看 workspace_idx、数量、shape

---

## 3. 代码生成流程

**开始生成代码前，必须先读取并遵从 [references/ascend-constraints.md](references/ascend-constraints.md)**——其中定义了两条强制规则：

- §1 算子 kernel 划分原则：支持域完整 + fallback 审计，未覆盖输入立即返回 `[DESIGN_ERROR]`
- §2 Host 侧 Buffer 操作约束：算子核心逻辑全部在 kernel 内实现，host 侧禁止改数据指针 / 真实重排 / 改写 buffer / 用新 buffer 作弊 / 隐式触发 aclnn 调用

以下流程步骤均建立在这些约束之上；生成代码后会在步骤 5 上库前检查中逐项复核。

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

> **⚠️ 生成代码时必须遵守 [references/ascend-constraints.md](references/ascend-constraints.md) §2「Host 侧 Buffer 操作约束」**：算子的核心计算逻辑全部在 kernel 内实现。host 侧对 NPU 张量只能做「只改元数据」的视图操作（`reshape`/`view`/`transpose`/`permute`/`expand`）以及数据准备 / kernel 调用 / 结果验证；禁止改数据指针、禁止 `.contiguous()` 等真实重排、禁止改写 buffer 内容、禁止用新 buffer 作弊。拿不准时，一律放入 kernel。

> **⚠️ dtype 特化检查（支持多 dtype 的算子必须执行）**：生成代码时必须检查每个支持的 dtype 是否走硬件加速路径。如果 dtype 回退标量路径，必须比较同宽 reinterpret、kernel 内 cast、record-aware DMA、块 DMA + UB-local 标量重排等候选。**禁止把大张量降级成逐元素 strided GM load/store，也不能把“逐行 T.copy”误当成可完成任意转置。** UB 内局部标量 lowering 可以作为经最大 case 验证的 fallback；不得在 host 侧用 `.to(dtype)` / `.contiguous()` / `torch.stack` 绕过。详见 [references/coding-conventions.md §7](references/coding-conventions.md#7-dtype-性能特化)。

> **⚠️ stride/shape 参数传递（kernel 需要地址偏移时）**：优先作为 `@tilelang.jit` 函数的 Python 参数传入（JIT 编译期常量），kernel 内用 `T.alloc_var` 累加偏移。**不要打包成 int32 GM tensor 在运行时传入**——会增加一次 GM→UB 搬运。详见 [tilelang-api-best-practices/references/api-compute.md §4.10 Stride 参数作为 JIT 编译期常量](../tilelang-custom-skill/tilelang-api-best-practices/references/api-compute.md#stride-参数作为-jit-编译期常量传入-kernel避免创建-gm-tensor)。

> **⚠️ 数据重排实现门禁**：GM↔UB 应使用尽可能大的块/二维 `T.copy`。标量循环只能用于
> UB-local reorder，不能在大张量主路径中写成 `ub[i] = gm[base + i * stride]`。
> 对连续 suffix record，必须按通用结构谓词聚合多条 record，而不是每条 record 发两次短 DMA。
> 生成后用 `get_kernel_source()` 检查每条 dtype/路径；若 `GetValue/SetValue` 对 GM
> 按 numel 展开，或 DMA transaction 估算达到数十万/百万级，必须重新设计。
>
> 不要只执行上述门禁。凡是涉及数据布局变化，必须按
> [coding-conventions.md §6.1](references/coding-conventions.md#61-数据重排的正向实现配方)
> 的六步配方生成候选实现：识别连续 record → 按成本选路径 → 聚合搬运 →
> UB-local reorder → 必要时分阶段 → 数字验收。

完成候选后必须写 `[REORDER-COST-AUDIT]`，包含 GM pass、DMA transaction/平均字节、
GM 标量访问、逐元素 div/mod、active cores、每核串行任务和最大 case timeout。任一
指标无法估算时先读取 §6.1 补齐；大张量路径出现 numel 级 GM 标量访问或海量短 DMA
时不得进入精度验收。

基于 design.md 的 API 映射 + 参考示例的代码风格，生成**两个文件**：`{op}.py`（纯 kernel）与 `test_{op}.py`（golden + L0 + main，L1/L2/Boundary 留桩，从 `{op}.py` import kernel）。完整文件结构骨架与融合算子注意事项见 [examples/code-skeleton.md](examples/code-skeleton.md)。

> **写代码时遇到**具体编码规范问题（Buffer 分配 / 索引一致性 / 同步 / 广播 / 测试模板）查 [references/coding-conventions.md](references/coding-conventions.md)。
>
> **V 核并行化**（按行切分、中间 buffer 索引一致性、CV 融合 V 核切分）查 [references/vector-parallelism.md](references/vector-parallelism.md)。
>
> **含 GEMM 或 CV 融合**时查 [references/gemm-cv-fusion.md](references/gemm-cv-fusion.md)（gemm_v0 初始化、NPU 分形限制、CV 融合必开的 4 个 pass_configs）。

### 步骤 4：运行验证

本 skill 负责 L0 精度收敛，同时负责实现的最低性能可行性。先跑 L0：

```bash
python examples/{op}/test_{op}.py --level l0
```

随后必须运行 design.md 中的性能可行性哨兵（即使它被标为 large/L1），为每个 case
设置明确 timeout。用户明确给出的失败或超时 case 必须全部实际运行。任一哨兵超时，
不得宣称生成完成：应修复搬运路径；若现有 API 无法满足，则返回 `[DESIGN_ERROR]`
并附 GM/DMA 成本证据。

测试数据准备同样属于 aclnn 审计范围：随机数、特殊值注入、dtype 转换和 golden
物理重排全部在 CPU 完成，然后只做一次 H2D；验证时只做 D2H。不得在 NPU 上调用
`torch.rand/randint`、in-place random、`.contiguous()` 或 golden 计算。报错中若出现
`aclnnInplaceRandom`，说明失败发生在测试输入准备，不是 kernel 内存不足或精度问题。

若测试规格包含 NaN/Inf，生成的测试必须采用位置敏感验证：

1. 在 CPU 上用固定 seed 生成有限基础值和稀疏特殊值 mask，并保证至少一个特殊值、
   一个有限值；`[nan, nan]` 不得直接退化为全 NaN。
2. 数值容差判断前，分别要求 actual/golden 的 NaN、正 Inf、负 Inf mask 完全相等。
3. mask 一致后，只在双方有限的位置计算 atol/rtol、matched ratio 和 max absolute error。
4. 全 NaN/全 Inf 只能作为补充用例，不能作为唯一特殊值门禁；
   `torch.allclose(..., equal_nan=True)` 不能替代显式 mask 比较。

具体生成骨架见 [references/coding-conventions.md §5](references/coding-conventions.md#5-测试模板)。

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

运行通过后，必须按 [references/checklist.md](references/checklist.md) 逐项检查。

**⚠️ 首要检查：算子主要操作是否全部在 kernel 内实现（违反则立即修改，不得继续）**

逐项检查前，先回顾生成的代码，按 [references/ascend-constraints.md](references/ascend-constraints.md) §2「Host 侧 Buffer 操作约束」的五条禁令逐条核对 host 侧代码（**含 kernel 调用后的输出后处理路径**）：

1. 是否把输入/输出 tensor 重新绑定到别的 tensor（改了 `data_ptr`）后传入 kernel？
2. 是否存在真实数据拷贝/重排？对 reshape 应验证 storage/stride 兼容性，不能把
   `is_contiguous() == False` 直接等同于一定复制；`permute`/`transpose` 通常只改 metadata。
3. 是否在 host 侧直接改写了 tensor 数据（`x[:] =`、in-place `_()`、`out=` 等）？
4. 是否用「新建 buffer → host 侧处理 → 替换原 tensor」的方式作弊？
5. 是否在 host 侧隐式触发了 aclnn 调用？重点检查：`torch.nn.functional.pad`/`cat`/`interpolate`、`torch.cat`/`stack`、`.to(dtype)` dtype 转换、`.clone()`、以及**输出侧切片+reshape**（如 `y = y[:,:,:,:S]; y.reshape(shape)`——切片后非 contiguous，reshape 隐式 `.contiguous()` → `aclnnCopy`）。若需要从 padded 输出裁剪有效部分，应改为让 kernel 直接输出到与原始 shape 一致的 buffer（通过 `T.copy` + `pad_value` 处理尾块），host 侧无需切片。
6. 是否对所有支持的 dtype 做了特化检查？用 `get_kernel_source()` 确认每个 dtype 是否走硬件加速路径。标量回退的 dtype 是否在 kernel 内处理（cast 或逐行搬运），**而不是在 host 侧用 `.to()` / `.contiguous()` / `torch.stack` 绕过**？

任何一条命中，**必须立即修改**——把这些操作移入 kernel 内部，直到满足要求后才能继续后续检查。允许的 host 侧操作仅限：`reshape`/`view`/`transpose`/`permute`/`expand` 等只改元数据的视图操作，以及数据准备、kernel 调用、结果验证。

**最容易踩坑的 4 项重点提醒**：

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

- [references/ascend-constraints.md](references/ascend-constraints.md) — 算子 kernel 划分原则 + Host 侧 Buffer 操作约束（生成代码前/时必须遵守的强制规则）
- [references/coding-conventions.md](references/coding-conventions.md) — Buffer 分配 / 索引 / 同步 / 广播 / 测试模板（写代码遇到具体规范时查）
- [references/vector-parallelism.md](references/vector-parallelism.md) — V 核并行化（用到 vid 切分时查）
- [references/gemm-cv-fusion.md](references/gemm-cv-fusion.md) — GEMM 与 CV 融合 pass_configs（含 GEMM 或融合算子时查）
- [references/checklist.md](references/checklist.md) — 上库前检查清单（生成代码后逐项过）
- [references/troubleshooting.md](references/troubleshooting.md) — 编译 / 运行 / 精度错误排查手册（遇到具体错误时查）
- [references/skill-feedback.md](references/skill-feedback.md) — Skill 反馈采集流程（流程结束时查，orchestrator 模式跳过）
- [examples/code-skeleton.md](examples/code-skeleton.md) — `{op}.py`（kernel）+ `test_{op}.py`（测试）文件结构骨架
