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

> **⚠️ 核心原则：禁止按"不可穷举的条件"划分 kernel**
>
> 一个算子可以有多个 kernel，但划分 kernel 的条件必须是**封闭可穷举**的。如果划分条件的取值集合无法列全或可无限扩展，则禁止按此条件划分——否则没覆盖到的取值就不支持了，违背算子通用性。

#### 判断方法（设计多 kernel 方案前必须执行，不能跳过）

按以下 3 步逐一判断，任何一步不通过则禁止按该条件划分：

1. **列出划分条件的所有可能取值**：把条件写成集合，如 `dtype ∈ {float16, float32, bfloat16, int8, int16, int32, int64}` 或 `ndim ∈ {2, 3, 4, 5, 6, 7, 8, ...}`
2. **判断集合是否封闭可穷举**：
   - 集合元素有限且固定 → 可穷举 → ✅ 允许划分
   - 集合元素可扩展 / 连续值 / 阶乘或指数增长 → 不可穷举 → ❌ 禁止划分
3. **新增取值测试**：假设集合新增一个取值（如 ndim 从 8 扩展到 9），是否需要写新 kernel？
   - 需要 → ❌ 禁止划分，改为通用实现
   - 不需要（通用 kernel 自动覆盖） → ✅ 允许划分

#### 反例（多种"不可穷举"实例化，禁止照搬其中任何一个当成规则全集）

> ⚠️ 以下每个反例都是"不可穷举"的**不同实例化**，规则是针对**所有不可穷举的情况**，不只是下列具体例子。判断时必须用上面的 3 步判断方法，不能只记反例。

| ❌ 反例 | 为什么不可穷举 |
|---------|---------------|
| 按维度数划分（2D/3D/4D... 各写一个 kernel） | 维度数可无限增长，永远写不完，没写的维度就不支持 |
| 按 (i,j) 轴组合划分（3D 有 3 种、4D 有 6 种...） | 组合数随维度数爆炸，C(n,2) 增长 |
| 按 shape 大小档位划分（小/中/大/超大 tensor 各一个 kernel） | shape 是连续值，档位无法穷举 |
| 按 perm 具体值划分（[1,0]、[0,2,1,3]、[4,3,2,1,0]... 各一个 kernel） | 排列数阶乘增长 n! |
| 按输出 numel 划分（< 1M / 1M~16M / > 16M 各一个 kernel） | numel 是连续值 |

#### 正例（封闭可穷举的划分，允许）

| ✅ 正例 | 为什么可穷举 |
|---------|-------------|
| 按 dtype 划分（7 种，集合封闭） | dtype 是有限集合，硬件支持也是有限的 |
| 按对齐性划分（对齐 / 非对齐，二分） | 二值集合，封闭 |
| 按硬件计算路径划分（Cube / Vector） | 硬件路径有限，封闭 |
| 按 GEMM 的 K 是否整除 block_K 划分（整除 / 有尾块，二分） | 二值集合，封闭 |

#### 默认通用 + 有限特化原则

当不确定划分条件是否可穷举时，**默认写一个通用 kernel 处理所有情况**，通过 host 侧 `reshape`/`permute`/`view` 等 view 操作把输入归一化到统一形态（如把任意 ndim 的转置降维到 3D `(batch, M, N)`），让一个 kernel 覆盖所有情况。性能优化阶段再按**封闭可穷举**的条件（如 dtype、对齐性）做有限特化。

**反例参考**：transpose 算子曾因按维度数特化（2D/3D/4D 各写一个 kernel）被否决，后改为一个通用 3D kernel + host reshape 降维处理 2D~8D。

#### 设计自检（Phase 4 质量自检时核对）

design.md 中如果出现多个 kernel，必须回答：
- 划分条件是什么？取值集合是否封闭可穷举？
- 如果新增一个取值，是否需要写新 kernel？
- 如果答案是"需要写新 kernel"且条件不可穷举 → **设计违规**，必须改为通用 kernel + host 归一化方案

**⚠️ host 侧分派逻辑同样适用此规则**：host 侧的多路径分派逻辑（如 `if perm == [0,2,1,3]: ...`）也不得硬编码不可穷举的取值。如果分派条件是 perm 具体值、shape 具体值等不可穷举集合，必须改为通用算法（如基于 perm 拓扑特征的分类器、基于 shape 的连续性判断等）。判断方法同上：假设新增一个 perm 取值，是否需要加新的 `if` 分支？如果需要 → **设计违规**，必须改为通用算法。

### 3.1.1 基于输入结构特征的多路径分派（性能设计指导）

> 当算子输入具有多种结构特征（如 perm 拓扑类型、数据连续性、对齐性等）时，可设计多条快路径按结构特征分派。这不同于 §3.1 的"通用+特化"——特化是按 dtype 等封闭条件划分 kernel，多路径分派是按**输入的结构拓扑**选择不同的搬运/计算策略。

**约束**：
- 分派条件必须是**封闭可穷举**的（同 §3.1 判断方法）。典型封闭分派条件：数据连续性（连续/非连续，二值）、输入拓扑类型（有限分类）、硬件路径（有加速/回退，有限集合）
- 必须保留**通用 fallback**路径：无法归入任何快路径的输入走通用实现
- 快路径的判断逻辑在 host 侧完成（纯 Python 整数/元数据运算），不触碰数据

**思路**：
1. 分析算子输入可能的结构形态，找出可利用的结构特征（如"哪些轴连续""最内轴是否移动"等）
2. 每种结构形态设计一条快路径——核心是利用结构特征减少搬运次数或避免非连续访问
3. 保留通用 fallback 处理无法归类的输入

> **⚠️ 避免示例覆盖**：上述思路适用于所有输入具有结构差异的算子，**不限于 transpose 的 perm 拓扑**。判断准则是有没有可利用的结构特征让某些输入走更快的路径，有就分派，没有就统一处理。

参考思路：host 侧入口按输入结构特征（如 perm 拓扑）分派多条快路径，通用 fallback 兜底

### 3.1.2 数据重排的性能可行性（强制）

对涉及物理布局变化的算子，先按
[coding-conventions.md §6.1](../tilelang-op-generate/references/coding-conventions.md#61-数据重排的正向实现配方)
生成至少一个正向候选方案。该配方基于“连续 record + 聚合 DMA + UB-local reorder”
这一通用数据流，不绑定 transpose、固定 perm 或固定维数，可迁移到 layout
transform、pack/unpack、blocked layout 与 gather/scatter。

对 transpose、layout transform、gather/scatter 等纯搬运算子，精度正确不等于设计可交付。
design.md 必须对每条结构路径和用户给出的最大/关键用例量化以下成本：

- GM 完整读写次数、DMA transaction 数量及平均每次搬运字节数
- GM 标量 load/store 数量，以及地址计算中的整除/取模次数
- 实际使用的 AIV core 数、每个 core 内的串行任务数

以下实现不得作为大张量主路径：

- 按元素从 GM 做 strided scalar load/store
- 对数十万或数百万个短 record 各发一次 GM→UB 和 UB→GM `T.copy`
- 每个元素重复执行多维 `//`、`%` 地址解码

若输入和输出都保留一个物理连续 suffix record，应按结构谓词识别
`prefix + A + B + suffix -> prefix + B + A + suffix` 一类拓扑，设计 record-aware
快路径：用二维/成组 `T.copy` 聚合多条 suffix record，在 UB 内完成必要的局部重排，再连续写回。
不得写死某个 ndim、perm 或 shape；未匹配的输入仍走通用 fallback。

无硬件 tile 指令的 dtype（如 int64）也必须遵守上述 GM 搬运规则。允许 UB 内局部标量
重排，但大张量主路径禁止逐元素 strided GM 访问。若最大官方用例无法在超时预算内完成，
这是设计不可行，必须调整方案或明确返回设计错误，不能留给后续性能阶段。

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
> - 改数据指针顶替：`x = y`（y 是另一个 tensor）后传入 kernel
> - **`reshape` 对非 contiguous 张量**（隐蔽违规）：`reshape` 仅在张量 contiguous 时是零拷贝 view；对非 contiguous 张量（如 `permute`/`transpose` 后），`reshape`（尤其是 `reshape(-1)`）会触发物理拷贝，等价于 `.contiguous()`。代码中没有 `.contiguous()` 字样但行为相同——一律禁止
> - **输出侧切片 + reshape**（隐蔽违规，group_norm 案例）：kernel 输出后对输出张量做切片（如 `y[:, :, :, :S]`）使 tensor 变为非 contiguous，随后 `reshape` 会隐式调用 `.contiguous()` → `aclnnCopy`。**约束范围不仅限于输入侧，kernel 调用后的输出后处理同样适用**。解法：让 kernel 直接输出到与原始 shape 一致的 buffer（通过 `T.copy` 的 `pad_value` 处理尾块），host 侧无需切片+reshape
>
> **允许**的 host 侧操作：`reshape`/`view`/`transpose`/`permute`/`expand` 等**只改 stride/shape 元数据、不触碰真实数据**的视图操作；以及数据准备（输入 tensor 创建）、kernel 调用、结果验证。**⚠️ `reshape` 的零拷贝性质以 `x.is_contiguous()` 为前提**——非 contiguous 张量的 `reshape` 触发物理拷贝，属违规。
>
> **判定准则**：host 侧任何会改变 NPU 张量「数据指针」或「物理存储内容/排布」的操作均禁止；只改 metadata（stride/shape）的允许。拿不准时，一律放入 kernel。
>
> **`reshape` 语义判定**：判断 `reshape` 是否安全——检查 `x.is_contiguous()`：为 `True` 则 `reshape` 是零拷贝 view（允许）；为 `False` 则 `reshape` 触发物理拷贝（禁止）。常见非 contiguous 场景包括 `permute`/`transpose`/`movedim` 后的张量，**但不限于这些**——任何改变了 stride 但未拷贝数据的操作都会产生非 contiguous 张量。**替代方案**：当需要对非 contiguous 张量做维度归一化时，改用 stride buffer 方案（host 侧只算 stride 参数传入 kernel，kernel 内逐行搬运），不在此做 `reshape`。
>
> **非整除处理**：前端框架已支持自动尾块搬运（`T.copy` 动态 shape 切片，非整除时尾块无需特殊处理，详见 `tilelang-api-best-practices/references/api-kernel-memory.md` §T.copy 动态 shape 切片），**不需要 host padding**。design.md 中不得出现 host padding + crop 的设计描述。
>
> **aclnn 依赖约束**（评测环境兼容性）：cann-bench 评测环境中 aclnn 编译产物可能被裁剪，host 侧任何会隐式触发 aclnn 调用的操作都会导致运行时失败。以下操作在 NPU tensor 上会触发 aclnn，一律禁止：
> - `torch.nn.functional.pad` / `torch.nn.functional.interpolate` / `torch.nn.functional.cat` 等 `torch.nn.functional.*` 计算 API
> - `torch.cat` / `torch.stack` / `torch.split` 等会创建新 buffer 的操作
> - 对非 contiguous 张量的 `reshape`（隐式 `.contiguous()` → `aclnnCopy`），**包括输出侧切片后的 reshape**
> - `.to(another_dtype)` dtype 转换（触发 `aclnnCast`）；如需 dtype 转换应在 kernel 内用 `T.tile.cast` 完成
> - `.clone()` / `.copy_()` 等显式拷贝
>
> **判定方法**：host 侧代码只允许出现 `reshape`/`view`/`transpose`/`permute`/`expand`（仅限 contiguous 张量）和 kernel 调用。任何其他对 NPU tensor 的操作都应视为可疑，检查是否触发 aclnn。
>
> **设计自检**（Phase 4 质量自检时核对）：design.md 的 host 侧步骤（**含 kernel 调用后的输出后处理**）是否仅限视图操作（`reshape`/`view`/`transpose`/`permute`/`expand`）+ kernel 调用 + 结果 reshape？是否出现 `.contiguous()` / host padding / `torch.nn.functional.*` / `torch.cat` / 新建 buffer 切片赋值 / `x = 新tensor` 顶替 / 输出切片+reshape 等描述？命中即违规，必须修订。下游 `tilelang-op-generate` skill 会按相同规则再次校验，违规设计会在 Stage 2 触发 `[DESIGN_ERROR]` 设计回退（详见 [tilelang-op-generate SKILL.md §3](../tilelang-op-generate/SKILL.md)）。

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

> **⚠️ 生成 design.md 时必须遵守 §3.1「Host 侧 Buffer 操作约束」**：设计文档中 host 侧（kernel 外的 Python 代码）对 NPU 张量只能做「只改元数据」的视图操作（`reshape`/`view`/`transpose`/`permute`/`expand`）以及数据准备 / kernel 调用 / 结果验证；禁止改数据指针、禁止 `.contiguous()` 等真实重排、禁止改写 buffer 内容、禁止用新 buffer 作弊。**所有数据搬运、padding、维度重排、非整除处理等核心计算逻辑必须落入 `@tilelang.jit` kernel 内部**。拿不准时，一律放入 kernel。下游 `tilelang-op-generate` skill 会按相同规则再次校验，违规设计会在 Stage 2 触发 `[DESIGN_ERROR]` 设计回退。

基于 [examples/design-template.md](examples/design-template.md) 模板，填充所有章节：

1. 概述
2. 编程模式选型
3. API 映射设计
4. 数据规格与内存规划
5. Tiling 策略（**含非整除处理说明**：前端框架已支持自动尾块搬运，非整除时尾块无需特殊处理；**host 侧不允许 padding + crop**）
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

核对 design.md 的 host 侧步骤描述：是否仅限视图操作（`reshape`/`view`/`transpose`/`permute`/`expand`，只改 stride/shape 元数据）+ kernel 调用 + 结果 reshape？是否出现 `.contiguous()` / host padding / 新建 buffer 切片赋值 / `x = 新tensor` 顶替 等改动 buffer 真实内容的描述？命中即违规，必须修订后再继续后续检查（详见 §3.1）。

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
- [references/quality-checklist.md](references/quality-checklist.md) — 18 项质量自检清单
- [references/info-sources.md](references/info-sources.md) — 信息收集步骤、信息源优先级、冲突处理原则
- [examples/design-template.md](examples/design-template.md) — design.md 完整模板
- [examples/completion-report-template.md](examples/completion-report-template.md) — 完成报告输出模板
