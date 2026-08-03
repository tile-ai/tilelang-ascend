---
name: tilelang-op-design
description: "根据算子需求生成 TileLang-Ascend 算子设计文档（design.md）。涵盖编程模式选型（Developer/Expert/混合）、API 映射、内存层级规划、Tiling 策略、循环结构、同步策略、验证方案等。触发：设计算子、生成 design.md、算子方案设计、新算子开发、算子实现方案。"
---

# TileLang-Ascend 算子设计文档生成

## 1. 目标

根据算子需求信息，生成一份完整的 TileLang-Ascend 算子设计文档（`design.md`），涵盖以下核心决策：

- **编程模式选型**：Developer / Expert / 混合模式
- **API 映射**：将数学公式拆解为 TileLang DSL 原语组合
- **内存层级规划**：GM → L1/UB → L0 的数据搬运路径
- **Tiling 策略**：Block 划分与 Tile Shape 设计
- **循环结构**：T.Parallel / T.serial / T.Pipelined / T.Persistent 的选择
- **同步策略**：自动同步 vs 手动同步标志
- **验证方案**：Golden 函数与 L0 门槛测试计划（完整分层套件 L1/L2/Boundary 由 tilelang-op-test-design 生成）

---

## 2. 输入要求

### 必需信息

| 字段 | 说明 |
|------|------|
| 算子名称 | 如 `softmax`、`layer_norm`、`flash_attention` |
| 数学公式 | 算子的数学表达，如 $\text{softmax}(x_i) = e^{x_i} / \sum e^{x_j}$ |
| 输入张量规格 | shape、dtype |
| 输出张量规格 | shape、dtype |
| 编程模式偏好 | Developer / Expert / 混合 |
| **迁移算子路径** ⭐ | 原算子文件路径（迁移时必需），用于获取 golden 实现 |
| **输出形状** ⭐ | 原算子输出 shape（迁移时必需），如 `(N, M)` 或 `(M, N)` |

**迁移算子时必须提供原算子路径和输出形状**，否则无法证明迁移正确性。Golden 实现一致性要求详见 [tilelang-op-develop checklist.md #9 Golden 实现一致 / #10 输出形状匹配](../tilelang-op-develop/references/checklist.md)。

**提问规则（必须严格遵守）**：
1. **优先使用调用方传入的字段**：若调用方（如 `@tilelang-op-orchestrator` 通过 analyst 传入 `op_requirements` 结构）已经提供了字段值，**全部跳过提问**，直接进入技术约束检测和 design 生成
2. **每次只询问一个字段**：使用 `question` 工具时，`questions` 数组中只包含一个元素
3. **按表格顺序依次询问**：算子名称 → 数学公式 → 输入张量规格 → 输出张量规格 → 编程模式偏好
4. **已提供的字段跳过**：如果用户在初始请求中已提供某个字段的值，跳过该字段继续下一个
5. **示例**：
   - 第 1 次询问：只问"数学公式"
   - 用户回答后，第 2 次询问：只问"输入张量规格"
   - 以此类推

**⚠️ 当被 orchestrator → analyst Subagent 链路调度时**：
- analyst 会把 orchestrator 在 Primary 上下文预检收集到的 `op_requirements` 完整传入
- 此时 5 个必需字段应当全部已 provided，跳过整个提问环节
- 若 skill 仍发现字段歧义或缺漏，**不要**在当前 Subagent 上下文调用 `AskUserQuestion`（透传不到真实用户），而是让 analyst 返回 `partial_input` + 缺失字段名给 orchestrator，由 orchestrator 在 Primary 上下文追问

### 推荐信息

| 字段 | 说明 |
|------|------|
| 典型配置 | 常用的 shape 组合与优先级 |
| 参考实现 | PyTorch / NumPy 参考代码 |
| 性能目标 | 目标吞吐量或延迟 |
| 动态轴说明 | 哪些维度在运行时变化 |

若用户未提供**必需信息**中的任一项，通过提问补全后再继续。

---

## 3. 技术约束（必须遵守）

本项目为 TileLang-Ascend（华为昇腾 NPU），与 GPU 版 TileLang 有显著差异。外部参考实现不可直接使用，必须转换为 Ascend 兼容方案。

**生成 design.md 前必须执行强制检测**：三维 Kernel、threads 参数、动态循环边界、GPU 专用 API、GEMM 非整除、L0C 溢出等。

详细已知限制清单、强制检测规则、警告输出模板见 [references/ascend-constraints.md](references/ascend-constraints.md)。

### 3.1 算子 kernel 划分原则（强制规则）⭐

多 kernel 方案必须保证支持域完整：先提供覆盖全部声明输入域的通用路径，再按 dtype、
shape、对齐性或输入拓扑增加有限快路径。具体取值可以用于性能特化；只要未命中的输入
可靠回落到通用路径，就不要求特化条件本身“封闭可穷举”。

设计时记录每条路径的适用谓词、fallback、语义等价依据和验证用例。禁止仅枚举若干
ndim、perm 或 shape 而没有 fallback，也禁止新增一个取值就让算子变成“不支持”。

生成 design.md 前必须写出以下审计；任一项无法回答时，不得进入实现阶段：

```text
[DISPATCH-COVERAGE]
supported_domain: <design 声明的 shape/dtype/attr 范围>
generic_fallback: <kernel/path；没有则写 none>
specializations:
  - predicate: <纯 metadata 可判定条件>
    fallback_on_miss: <path>
    equivalence_evidence: <索引映射或参考实现>
unsupported_inputs: <必须与 supported_domain 不冲突>
result: pass/fail
```

判定规则：

1. `generic_fallback == none` 时，所有分支谓词的并集必须覆盖 `supported_domain`，否则 fail。
2. 有 fallback 时允许具体 dtype/shape/perm 快路径，但每条未命中输入必须落入 fallback。
3. 分派只能读取 shape、stride、dtype、attr 等 metadata，不能读取或改动 tensor 数据。
4. 每条 specialization 和 fallback 都必须在验证计划中至少有一个命中 case。

### 3.1.1 基于输入结构特征的多路径分派（性能设计指导）

> 当算子输入具有多种结构特征（如 perm 拓扑类型、数据连续性、对齐性等）时，可设计多条快路径按结构特征分派。这不同于 §3.1 的"通用+特化"——特化是按 dtype 等封闭条件划分 kernel，多路径分派是按**输入的结构拓扑**选择不同的搬运/计算策略。

**约束**：
- 分派谓词必须可判定，且不能读取或改动 tensor 数据
- 必须保留**通用 fallback**路径：无法归入任何快路径的输入走通用实现
- 快路径的判断逻辑在 host 侧完成（纯 Python 整数/元数据运算），不触碰数据

**思路**：
1. 分析算子输入可能的结构形态，找出可利用的结构特征（如"哪些轴连续""最内轴是否移动"等）
2. 每种结构形态设计一条快路径——核心是利用结构特征减少搬运次数或避免非连续访问
3. 保留通用 fallback 处理无法归类的输入

> **⚠️ 避免示例覆盖**：上述思路适用于所有输入具有结构差异的算子，**不限于 transpose 的 perm 拓扑**。判断准则是有没有可利用的结构特征让某些输入走更快的路径，有就分派，没有就统一处理。

参考思路：host 侧入口按输入结构特征（如 perm 拓扑）分派多条快路径，通用 fallback 兜底

### 3.1.2 数据重排的性能可行性（强制）

涉及物理布局变化时，必须读取
[coding-conventions.md §6.1](../tilelang-op-develop/references/coding-conventions.md#61-数据重排的正向实现配方)，
为每条结构路径和最大/关键用例完成 GM/DMA/地址解码/并行度成本验收。具体 record、
UB-local reorder 和 dtype fallback 配方以该 reference 为准，不在主流程重复。

design.md 必须为每条路径输出：

```text
[REORDER-COST]
path: <name>
gm_passes: <完整读写次数>
dma_transactions: <估算值>
average_dma_bytes: <估算值>
gm_scalar_accesses: <估算值>
address_div_mod_per_element: <估算值>
active_cores / serial_tasks_per_core: <估算值>
largest_case_timeout_gate: <case + timeout>
result: pass/fail
```

若大张量主路径的 GM 标量访问接近 numel、短 DMA 达到数十万次，或没有最大 case
timeout 门禁，结果必须为 fail，并按 reference 重新选择候选实现。

### 3.2 Host 侧 Buffer 操作约束（设计阶段必须遵守）⭐

> **⚠️ 核心原则：host 侧禁止改动 NPU 张量 buffer 内的真实内容，禁止触发任何 aclnn 调用**
>
> 算子的所有核心计算逻辑（数据搬运、数学运算、归约、维度重排、padding 等）必须在 `@tilelang.jit` 装饰的 kernel 函数内部完成。**host 侧（kernel 外的 Python 代码）对 NPU 侧张量 buffer 内的真实内容（数据值、物理排布、数据指针）一律不得改动**——只允许做只改 stride/shape 元数据的视图操作。**约束范围覆盖 kernel 调用前（输入预处理）和 kernel 调用后（输出后处理）的完整 host 代码路径。**
>
> **违规示例**（都属于"改动 buffer 真实内容"或"触发 aclnn"，一律禁止）：
> - `.contiguous()` / `.reshape(...).contiguous()` / `.permute(...).contiguous()` / `.transpose(...).contiguous()` —— 触发真实数据拷贝/重排
> - host 侧 padding：`x_padded = torch.zeros(...); x_padded[:, :M] = x; x = x_padded` —— 创建新 buffer + 写入数据 + 顶替原输入
> - **`torch.nn.functional.pad(x, ...)` / `torch.cat` / `torch.stack`** —— 隐蔽违规：表面是函数调用，实质是创建新 buffer + 数据拷贝，在 NPU 上会调用 `aclnnPad`/`aclnnCat` 等 aclnn 算子。等同禁止行为"用新 buffer 作弊"
> - 直接改写 buffer 内容：`x[:] = ...`、`x.add_(1)`、`torch.mul(x, 2, out=x)`
> - 用另一个经过 host 计算或物理化的 tensor 替代原输入后传入 kernel
> - `reshape` 无法保持原 storage/stride 时发生的隐式物理化；不能仅凭
>   `is_contiguous()` 判断，需证明目标 shape 与当前 stride 兼容，或比较操作前后的
>   storage/data pointer
> - **输出侧切片 + reshape**（隐蔽违规，group_norm 案例）：kernel 输出后对输出张量做切片（如 `y[:, :, :, :S]`）使 tensor 变为非 contiguous，随后 `reshape` 会隐式调用 `.contiguous()` → `aclnnCopy`。**约束范围不仅限于输入侧，kernel 调用后的输出后处理同样适用**。解法：让 kernel 直接输出到与原始 shape 一致的 buffer（通过 `T.copy` 的 `pad_value` 处理尾块），host 侧无需切片+reshape
>
> **允许**的 host 侧操作：经证明只改 stride/shape 元数据、不触碰 storage 的
> `reshape`/`view`/`transpose`/`permute`/`expand`，以及数据准备、kernel 调用和结果验证。
>
> **判定准则**：host 侧任何会改变 NPU 张量「数据指针」或「物理存储内容/排布」的操作均禁止；只改 metadata（stride/shape）的允许。拿不准时，一律放入 kernel。
>
> **`reshape` 语义判定**：`is_contiguous()` 为 True 是常见充分条件，不是必要条件。
> 对非 contiguous 输入先证明该目标 reshape 可返回共享 storage 的 view；不能证明时，
> 使用 stride-aware kernel 路径。
>
> **非整除处理**：输入、输出 GM 两侧必须显式使用 valid extent/BufferRegion；前端按
> 动态切片裁剪搬运，但不会替设计补齐错误的完整 tile 区域。无需 host padding，
> design.md 中不得出现 host padding + crop。
>
> **aclnn 依赖约束**（评测环境兼容性）：cann-bench 评测环境中 aclnn 编译产物可能被裁剪，host 侧任何会隐式触发 aclnn 调用的操作都会导致运行时失败。以下操作在 NPU tensor 上会触发 aclnn，一律禁止：
> - `torch.nn.functional.pad` / `torch.nn.functional.interpolate` / `torch.nn.functional.cat` 等 `torch.nn.functional.*` 计算 API
> - `torch.cat` / `torch.stack` 等会创建并填充新 buffer 的操作；`split`/切片本身可能
>   只是 view，需审计其后是否发生物理化
> - 对非 contiguous 张量的 `reshape`（隐式 `.contiguous()` → `aclnnCopy`），**包括输出侧切片后的 reshape**
> - `.to(another_dtype)` dtype 转换（触发 `aclnnCast`）；如需 dtype 转换应在 kernel 内用 `T.tile.cast` 完成
> - `.clone()` / `.copy_()` 等显式拷贝
>
> **判定方法**：逐项证明 host tensor 操作只改变 metadata；不能从 API 名称或
> `is_contiguous()` 单一布尔值直接下结论。
>
> **设计自检**（Phase 4 质量自检时核对）：审计 kernel 前后完整 host 路径。下游
> [tilelang-op-develop SKILL.md §3](../tilelang-op-develop/SKILL.md) 会再次校验，
> 违规设计在 Stage 2 返回 `[DESIGN_ERROR]`。

审计必须记录：

```text
[HOST-METADATA-AUDIT]
operation: <host tensor operation>
input_stride -> output_stride: <...>
shares_storage / same_data_ptr: true/false/unknown
aclnn_or_physical_copy: true/false/unknown
result: allow/reject
```

任何 `unknown` 按 reject 处理；改为 kernel 内实现或用可证明的 metadata-only 路径。

---

## 4. 工作流程

### Phase 1：输入解析与算子特征分析

1. 解析算子名称与数学公式
2. 验证必需字段是否完整
3. 分析算子特征：
   - **计算类型判定**：
     - 纯 Vector（element-wise / reduction）→ 仅需 UB
     - 纯 Cube（仅 matmul）→ 需要 L1 + L0A/L0B/L0C
     - 混合（matmul + element-wise 后处理）→ 核间流水线，需要 CV 融合
     - **Host 预处理**：如 im2col 等 Python 侧预处理步骤，标明在 design 的 §1 和 §4 中
   - **复杂度级别**：
     - 单步（如 element-wise add）→ 无循环、单次搬运
     - 多步（如 softmax = max + sub + exp + sum + div）→ 多次计算、可能需要中间缓冲
     - 融合（如 flash attention = GEMM + softmax + GEMM）→ 核间协作、流水线
   - **动态 shape 判定**：是否存在运行时才确定的维度
4. **非整除场景预判**：检查输入 shape 是否可能不被 block size 整除。`T.ceildiv(M, block_M)` 对非整除或 `M < block_M` 返回 ≥1（非零），`T.copy` 已支持动态 shape 切片自动处理尾块，**不需要 host padding**。用 `T.ceildiv` + 动态切片 `T.copy(A[m:m+valid, ...])`，参考 `examples/chunk_gated_delta_rule/expert_chunk_gated_delta_rule.py:107-108`。仅当多个 group 共享同一输出 buffer 时需注意尾块写入竞态——用 metadata 的 valid_m 字段限制写入范围 `T.copy(C_L0, Y[m:m+valid_m, ...])`
5. **多 group 输出竞态约束**（grouped 类算子）：当多个 group 共享同一输出 buffer（紧凑排列，不 padding）时，尾块按 block_M 整块写会溢出到隔壁 group 的区域，导致竞态条件（执行顺序不确定→结果不确定）。解法：metadata 记录 valid_m，kernel 用 `T.copy(C_L0, Y[m_start : m_start + valid_m, ...])` 只写有效行。参考 `examples/grouped_gemm/example_grouped_gemm_fwd.py` 的 block_metadata[2]（valid_m 字段，当前未使用，应启用）。
6. **数据重排成本建模**：按 §3.1.2 为每条路径和最大/关键 case 计算 DMA/GM
   标量访问/地址解码/并行度；不能只给 tile shape，不计算 transaction 数量。

### Phase 2：信息收集

**必须执行强制步骤 0：搜索本项目同类实现**。详细工具调用、信息收集步骤、禁止行为见 [references/info-sources.md](references/info-sources.md)。

### Phase 3：生成 design.md

> **⚠️ 生成 design.md 时必须遵守 §3.2「Host 侧 Buffer 操作约束」**。下游
> `tilelang-op-develop` 会再次校验；违规设计在 Stage 2 返回 `[DESIGN_ERROR]`。

基于 [examples/design-template.md](examples/design-template.md) 模板，填充所有章节：

1. 概述
2. 编程模式选型
3. API 映射设计
4. 数据规格与内存规划
5. Tiling 策略（非整除时用输入、输出两侧显式 valid extent/BufferRegion；前端负责
   按这些动态切片裁剪搬运，**不代表尾块无需设计**；host 侧不允许 padding + crop）
6. 循环与调度结构
7. 同步策略
8. CV 融合设计（**按模式分支**：Developer 默认消除 workspace/vid——`threads=2` + 片上直连，不产出 workspace 规格；仅 Expert/混合或复杂场景回退才设计 workspace + `workspace_idx`。详见 design-template.md §8.2）
9. 验证方案（Golden + **L0 门槛测试计划** + **性能可行性哨兵**；除规则
   shape 外，至少包含每条路径最坏 dtype/最大任务数的用户关键 case，并给出单 case 超时预算；
   完整分层套件 L1/L2/Boundary 交由 `tilelang-op-test-design`）。若算子支持
   NaN/Inf，精度方案必须声明位置敏感比较：特殊值用“有限值 + 稀疏特殊值”的
   混合输入，先严格比较 NaN/正 Inf/负 Inf mask，再只对有限值应用数值容差；
   禁止使用全 NaN/全 Inf 输入作为唯一特殊值用例，因为数据重排或索引错误可能被掩盖
10. 风险点与注意事项
11. 交付清单

### Phase 4：质量自检

**⚠️ 首要检查：host 侧 Buffer 操作合规性（违反则立即修订，不得继续）**

核对 design.md 的完整 host 路径是否只含经证明不物理化的 metadata view、kernel
调用与验证；命中真实拷贝或 aclnn 调用必须修订（详见 §3.2）。

按照 [references/quality-checklist.md](references/quality-checklist.md) 中的自检清单逐项检查，确保文档质量。

### Phase 5：针对性修订

仅修正未通过自检的项目。信息确实不足的标注为「待确认」并说明原因。

### Phase 6：输出

- 将 `design.md` 输出到当前目录或用户指定路径。若文件已存在，询问是否覆盖。
- **同时产出 `proto.yaml`**（算子接口规格，模板见 [examples/design-template.md](examples/design-template.md) §11.5）：**dtype 全集取自 §9.3 精度表**（每个支持的 dtype 一行；§4.1 只给代表性 dtype，不作 dtype 全集来源）、attr 取自 §1/§4，**机械派生**写到同目录（`examples/{op}/proto.yaml`）。这是覆盖门禁 `coverage_check.py --proto` 的权威 dtype/attr 来源，**每个算子都必须产出**；`inputs[].dtype` 须与 §9.3 精度表的 dtype 行一致。

---

## 5. 算子特征分析决策树

详细决策树（Ascend 版）、平台识别、API 映射规则、NPU 硬件约束（分形限制 / 对齐要求 / 存储大小上限）见 [references/decision-tree.md](references/decision-tree.md)。

---

## 6. 信息源优先级

信息源优先级表与冲突处理原则见 [references/info-sources.md](references/info-sources.md)。

---

## 7. 错误处理

| 场景 | 处理方式 |
|------|----------|
| 用户未提供数学公式 | 提问补全，给出常见算子公式作为参考 |
| 必需字段缺失 | 列出缺失项，逐一提问 |
| API 查询无结果 | 标注为「需扩展」，在风险点中说明 |
| 目标文件已存在 | 询问用户是否覆盖或另存 |
| 算子过于复杂 | 建议拆分为多个子算子分别设计 |

---

## 8. 完成报告

文档生成完成后，按 [examples/completion-report-template.md](examples/completion-report-template.md) 输出报告。

---

## 9. 生成算子

完成报告后，询问用户是否根据此报告生成对应算子代码。

---

## 子目录索引

- [references/ascend-constraints.md](references/ascend-constraints.md) — 技术约束清单、强制检测规则、警告输出格式
- [references/decision-tree.md](references/decision-tree.md) — 算子特征分析决策树、平台识别、NPU 硬件约束、API 映射规则
- [references/quality-checklist.md](references/quality-checklist.md) — 质量自检清单
- [references/info-sources.md](references/info-sources.md) — 信息收集步骤、信息源优先级、冲突处理原则
- [examples/design-template.md](examples/design-template.md) — design.md 完整模板
- [examples/completion-report-template.md](examples/completion-report-template.md) — 完成报告输出模板
