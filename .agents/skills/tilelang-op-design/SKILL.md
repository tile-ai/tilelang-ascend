---
name: tilelang-op-design
description: "根据算子需求生成 TileLang-Ascend 算子设计文档（design.md）。涵盖编程模式选型（Developer/Expert/混合）、API 映射、内存层级规划、Tiling 策略、循环结构、同步策略、验证方案等。触发：设计算子、生成 design.md、算子方案设计、新算子开发、算子实现方案。"
---

# TileLang-Ascend 算子设计文档生成

---

## 1. 目标

根据算子需求信息，生成一份完整的 TileLang-Ascend 算子设计文档（`design.md`），涵盖以下核心决策：

- **编程模式选型**：Developer / Expert / 混合模式
- **API 映射**：将数学公式拆解为 TileLang DSL 原语组合
- **内存层级规划**：GM → L1/UB → L0 的数据搬运路径
- **Tiling 策略**：Block 划分与 Tile Shape 设计
- **循环结构**：T.Parallel / T.serial / T.Pipelined / T.Persistent 的选择
- **同步策略**：自动同步 vs 手动同步标志
- **验证方案**：Golden 函数与多级测试计划

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

**迁移算子时必须提供原算子路径和输出形状**，否则无法证明迁移正确性。详见 [tilelang-op-generate/references/pr-ready-guide.md §1](../tilelang-op-generate/references/pr-ready-guide.md)。

**提问规则（必须严格遵守）**：
1. **每次只询问一个字段**：使用 `question` 工具时，`questions` 数组中只包含一个元素
2. **按表格顺序依次询问**：算子名称 → 数学公式 → 输入张量规格 → 输出张量规格 → 编程模式偏好
3. **已提供的字段跳过**：如果用户在初始请求中已提供某个字段的值，跳过该字段继续下一个
4. **示例**：
   - 第 1 次询问：只问"数学公式"
   - 用户回答后，第 2 次询问：只问"输入张量规格"
   - 以此类推

### 推荐信息

| 字段 | 说明 |
|------|------|
| 典型配置 | 常用的 shape 组合与优先级 |
| 参考实现 | PyTorch / NumPy 参考代码 |
| 性能目标 | 目标吞吐量或延迟 |
| 动态轴说明 | 哪些维度在运行时变化 |

若用户未提供**必需信息**中的任一项，通过提问补全后再继续。

---

## 2.5 技术约束清单（必须遵守）

本项目为 TileLang-Ascend（华为昇腾 NPU），与 GPU 版 TileLang 有显著差异。
**外部参考实现不可直接使用，必须转换为 Ascend 兼容方案。**

### 2.5.1 本项目已知限制

| 约束 | 说明 | 影响 | 替代方案 |
|------|------|------|----------|
| **不支持三维 Kernel** | `T.Kernel` 只接受一维 block 数 | 三维并行设计无法实现 | 使用 `block_metadata` 预计算机制（参考 `examples/grouped_gemm/`） |
| **threads 参数限制** | 只支持 1 或 2，不支持大值 | `threads=128` 等设计报错 | 默认不指定 threads 或设为 2 |
| **动态循环边界不支持** | 循环次数不能依赖 tensor 值（如 `batch_sizes[bz]`） | `T.Pipelined(batch_sizes[bz])` 报错 | 预计算最大循环次数，用 `T.serial(max_iters)` + 条件判断 |
| **流水线不支持动态边界** | `T.Pipelined` 的循环次数必须静态 | 动态批次无法流水线 | 改用 `T.serial` 或预计算固定迭代次数 |
| **部分 GPU API 不可用** | CUDA 专用 API 在 Ascend 不存在 | 直接移植 GPU 代码失败 | 查阅本项目 `examples/` 确认 Ascend API |
| **GEMM 要求 M,N 为 block 整数倍** | `M // block_M` 整除依赖；`M < block_M` 时零 block 启动 | 输出全零或除零编译崩溃 | 设计文档 §4/§5 必须明确处理策略：host 侧 padding+crop 或 Kernel 动态 block |
| **L0C 容量上限** | A2/A3 设备 L0C = 128KB | `block_M × block_N × sizeof(accum) > 128KB` 导致 segfault | 设计 block 时满足 `block_M × block_N ≤ 16384`（float32 accum） |

### 2.5.2 强制检测规则

在设计文档生成前，**必须**执行以下检测：

| 检测项 | 触发条件 | 处理方式 |
|--------|----------|----------|
| 三维 Kernel | 参考实现包含 `T.Kernel(..., batch_count)` 或 3 个维度参数 | **立即警告**，提出 `block_metadata` 方案 |
| threads 参数 | 参考实现 threads > 2 | **立即警告**，建议 threads=2 或移除 |
| 动态循环边界 | 循环边界依赖 tensor 值 | **立即警告**，提出静态边界 + 条件判断方案 |
| GPU 专用 API | CUDA 相关 API（如 `T.gemm` 通用版） | **立即警告**，查阅本项目确认 Ascend API |
| GEMM 非整除风险 | `M` 或 `N` 不被 block size 整除（即 `M % block_M ≠ 0` 或 `N % block_N ≠ 0`） | **立即警告**，要求 design 中明确 padding 策略 |
| L0C 溢出风险 | block_M × block_N × sizeof(accum_dtype) > 131072 (128KB) | **立即警告**，建议减小 block 或拆分 |

### 2.5.3 警告输出格式

```
⚠️ 技术限制检测警告

检测到参考实现包含本项目不支持的功能：

1. 三维 Kernel（本项目只支持一维 Kernel）
   - 参考实现：T.Kernel(m_num, n_num, batch_count)
   - 本项目方案：T.Kernel(total_blocks) + block_metadata 预计算表
   - 参考：examples/grouped_gemm/example_grouped_gemm_fwd.py

2. 动态循环边界（本项目不支持 tensor 值作为循环边界）
   - 参考实现：T.Pipelined(batch_sizes[bz])
   - 本项目方案：T.serial(max_k_iters) + if k < k_iters 条件判断
   - 参考：examples/grouped_gemm/example_grouped_gemm_fwd.py

建议：
- 先查阅本项目 examples/ 中的同类实现
- 确认 Ascend API 用法后再生成设计文档

是否继续生成设计文档？
```

---

## 3. 工作流程

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
4. **非整除场景预判**：检查输入 shape 是否可能不被 block size 整除。GEMM 类算子的 `M // block_M` 和 `N // block_N` 在 `M < block_M` 或 `N < block_N` 时产生零 block 或不完整 tile，必须在设计中明确处理策略（host 侧 zero-padding + crop，或 Kernel 内动态 block size）

### Phase 2：信息收集

#### 强制步骤 0：搜索本项目同类实现

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

#### 信息收集步骤

1. 查阅 `examples/` 中同类算子实现（**强制步骤 0**）
2. 查阅 [tilelang-api-best-practices SKILL.md](../tilelang-custom-skill/tilelang-api-best-practices/SKILL.md) 确认 API 可用性和用法
3. 查阅 [tilelang-expert-to-developer SKILL.md](../tilelang-custom-skill/tilelang-expert-to-developer/SKILL.md) 确认编程模式和 pass_configs 配置
4. 判断 API 可用性时，必须同时核对公开导出路径（如 `tilelang/language/__init__.py`）与 lowering / codegen 实现（如 `src/op/`、`src/target/`），不能仅凭 `_ascend.py` / `_cuda.py` 文件名推断
5. 如有参考实现，分析其计算步骤（**仅用于理解数学逻辑，不可直接使用 API**）

#### 禁止行为

- ❌ 在没有执行强制步骤 0 的情况下，直接使用外部参考实现的 API
- ❌ 凭记忆猜测 API 名称或参数
- ❌ 使用 GPU 版 TileLang 的三维 Kernel 设计
- ❌ 使用 `threads > 2` 的参数配置

### Phase 3：生成 design.md

基于 `templates/design-template.md` 模板，填充所有章节：

1. 概述
2. 编程模式选型
3. API 映射设计
4. 数据规格与内存规划
5. Tiling 策略（**必含：非整除时 padding+crop 策略，或 Kernel 内动态 block 方案**）
6. 循环与调度结构
7. 同步策略
8. CV 融合设计
9. 验证方案
10. 风险点与注意事项
11. 交付清单

### Phase 4：质量自检

按照 §5 中的自检清单逐项检查，确保文档质量。

### Phase 5：针对性修订

仅修正未通过自检的项目。信息确实不足的标注为「待确认」并说明原因。

### Phase 6：输出

将 `design.md` 输出到当前目录或用户指定路径。若文件已存在，询问是否覆盖。

---

## 4. 算子特征分析决策树（修订版）

### 4.0 函数设计原则

1. **维度参数自推导**：算子调用函数（如 `conv_im2col_gemm`）应从输入 tensor shape 提取 B/C/H/W 等维度，不依赖模块级全局变量。这保证多场景顺序测试时不发生变量污染。
2. **Host 预处理显式声明**：若计算的一部分在 Python 侧完成（如 im2col），必须在 §1 算法描述和 §4 数据流中明确标注。

### 4.1 平台识别

**本项目为 TileLang-Ascend（昇腾 NPU）**，与 GPU 版 TileLang 有差异：

| 差异项 | GPU 版 TileLang | 本项目（Ascend） |
|--------|----------------|-----------------|
| Kernel 维度 | 支持三维 | **只支持一维** |
| threads 参数 | 支持 128+ | **只支持 1 或 2** |
| 循环边界 | 支持动态 | **只支持静态** |
| GEMM API | `T.gemm` | **`T.gemm_v0` / `T.gemm_v1`** |
| 内存分配 | 自动映射 | **Expert 模式需显式层级** |

**规则**：外部参考实现仅用于理解数学逻辑，API 映射必须查阅本项目。

### 4.1 决策树（Ascend 版）

**重要**：`T.reduce_sum/max/min` 和 `T.tile.*` 在 Developer 和 Expert 模式下**都可使用**。模式选择取决于是否需要手动控制内存层级和同步，而非使用了哪个 API。

```
算子数学公式
├─ 含 matmul / @ / 矩阵乘
│   ├─ 仅 matmul → 纯 Cube
│   │   模式: Developer (推荐) 或 Expert
│   │   API: T.gemm_v0 / T.mma
│   │   内存: GM→L1→L0A/L0B→L0C→GM
│   │   pass_configs: 全开启（Developer）
│   │   Kernel: T.Kernel(任务数, is_npu=True) as (cid, _)
│   │
│   └─ matmul + element-wise 前处理/后处理 → CV 融合算子
│       ├─ Developer 模式（推荐）
│       │   模式: Developer + AUTO_CV_COMBINE
│       │   API: T.tile.* (Vector) + T.gemm_v0 (Cube)
│       │   内存: GM→L1→L0C→workspace→UB→GM
│       │   pass_configs: AUTO_SYNC + AUTO_CV_COMBINE + AUTO_CV_SYNC
│       │   同步: AUTO_SYNC + AUTO_CV_SYNC 自动处理
│       │   V 核: 可用 vid 并行化（每个 V 核处理 block_N // VEC_NUM 行）
│       │
│       ├─ Expert 模式（极致性能）
│       │   模式: Expert + T.Scope("C"/"V") + T.set_cross_flag
│       │   同步: 手动核间同步（T.set_cross_flag / T.wait_cross_flag）
│       │
│    典型算子: W4A8 GEMM, Flash Attention, 量化 GEMM
│
├─ 纯 element-wise（逐元素运算）
│   参考: examples/elementwise/*.py, examples/activation/*.py
│   ├─ 单步运算 → Developer 模式
│   │   API: T.Parallel + 算术符号
│   │   内存: T.alloc_shared（编译器映射到 UB）
│   │
│   └─ 多步运算（如 softmax、layer_norm）
│       参考: examples/softmax/*.py, examples/normalization/*.py
│       ├─ 需精细 buffer 控制 → Expert 模式
│       └─ 无需精细控制 → Developer 模式
│
├─ 含归约（reduce_sum / reduce_max / reduce_min）
│   参考: examples/reduce/*.py
│   API: T.reduce_sum / T.reduce_max / T.reduce_min
│   内存: T.alloc_shared → UB
│
├─ 含分组/动态批次
│   参考: examples/grouped_gemm/*.py（重要！）
│   关键技术:
│   - block_metadata 预计算表（替代三维 Kernel）
│   - 静态循环边界 + 条件判断（替代动态边界）
│   Kernel: T.Kernel(total_blocks) + 手动索引分解
│
└─ 其他复杂算子
    强制步骤: 先搜索本项目 examples/
```

**⚠️ NPU 硬件约束（必查）**：

设计 Tiling 策略时，必须考虑：
1. **分形限制**（Fractal Limits）：
   - L0A: M ≥ 16, K ≥ 32
   - L0B: K ≥ 32, N ≥ 16
   - L0C: M ≥ 16, N ≥ 16
2. **对齐要求**：
   - UB/L1: 32 Byte
   - L0A/L0B: 512 Byte
   - L0C: 64 Byte
3. **存储大小上限**：
   - L0A/L0B: 64KB
   - L0C: 128KB
   - L1: 512KB
   - UB: 192KB

违反约束会导致编译错误或运行时错误。详见 [tilelang-api-best-practices](../tilelang-custom-skill/tilelang-api-best-practices/references/api-kernel-memory.md)。

### 4.2 API 映射规则

| 类别 | Ascend 专用 API（推荐） | 通用 API（本项目不推荐/不支持） |
|------|------------------------|-------------------------------|
| GEMM | `T.gemm_v0` | `T.gemm`（可能不支持） |
| 内存分配（Expert）| `T.alloc_L1`, `T.alloc_L0C`, `T.alloc_ub` | - |
| 内存分配（Developer）| `T.alloc_shared`, `T.alloc_fragment` | - |
| Kernel | `T.Kernel(一维, is_npu=True)` | `T.Kernel(三维)` ❌ |
| 同步 | `T.barrier_all()`, `T.Scope("C")` | 自动同步（Developer 模式） |
| 循环 | `T.serial`, `T.unroll` | `T.Pipelined(动态边界)` ❌ |

---

## 5. 质量自检清单

生成 `design.md` 后，逐项检查：

| # | 检查项 | 是否必须通过 |
|---|--------|-------------|
| 1 | **编程模式有明确结论和理由**：不是笼统的「视情况而定」 | ✅ 必须 |
| 2 | **API 映射具体到函数名和参数**：不是「使用相关 API」 | ✅ 必须 |
| 3 | **内存搬运路径完整**：从 GM 到计算再到 GM 的每一步都有说明 | ✅ 必须 |
| 4 | **Tiling 策略有约束分析**：解释了为什么选择该 Block/Tile 大小 | ⭕ 推荐 |
| 5 | **同步策略与编程模式匹配**：Developer 用自动同步、Expert 标明手动同步点 | ⭕ 推荐 |
| 6 | **验证方案覆盖 4 类典型配置**：完美对齐 + 单维 padding + 全维 padding + 多 block（GEMM 类必含），不是「待补充」 | ⭕ 推荐 |
| 7 | **无占位符或模糊描述**：无 `{placeholder}`、TODO、「待补充」（已确认的除外） | ✅ 必须 |
| 8 | **技术约束已确认**：三维 Kernel、threads、动态边界等问题已处理 | ✅ 必须 |
| 9 | **含 GEMM 场景**：Tiling 策略满足 NPU 分形限制（block_M ≥ 16, block_N ≥ 16） | ✅ 必须 |
| 10 | **含 GEMM 场景 L0C 容量约束验证**：`block_M × block_N × sizeof(accum_dtype) ≤ L0C_capacity (128KB)` | ⭕ 推荐 |
| 11 | **含 GEMM 场景非整除处理策略明确**：主机侧 padding+crop 或 Kernel 内动态 block，说明溢出 / 下溢处理 | ✅ 必须 |
| 12 | **含 CV 融合场景**：workspace 规格、数据流、pass_configs 设计完整| ✅ 必须 |
| 13 | **含 CV 融合场景 workspace_idx 配置正确**：与 workspace 参数位置一致 | ✅ 必须 |
| 14 | **本项目同类实现已列出**：有具体的 examples/ 文件路径参考 | ✅ 必须 |
| 15 | **参考实现差异已说明**：如有外部参考，列出 API/结构差异 | ⭕ 推荐 |
| 16 | **参考实现分析完整**：如有外部参考，记录内存层级 API、同步策略、pass_configs 等技术决策 | ⭕ 推荐 |
| 17 | **参考实现标注原算子路径**：如有外部参考，标注文件路径，用于获取 golden 实现 | ⭕ 推荐 |
| 18 | **参考实现标注输出形状**：如有外部参考，说明输出形状是否需要 transpose | ⭕ 推荐 |
| 19 | **函数无全局变量依赖**：维度参数从 tensor shape 或函数参数获取，支持多场景顺序测试 | ⭕ 推荐 |

**通过条件**：必须项（1, 2, 3, 7, 8, 9, 14）全部通过，推荐项至少通过 4/9。

---

## 6. 信息源优先级（修订版）

| 优先级 | 信息源 | 用途 | 说明 |
|--------|--------|------|------|
| **0** | **本项目 `examples/` 同类实现** | **主要参考：API、编程模式、Kernel 结构** | **最权威**，直接可用 |
| 1 | `docs/TileLang-Ascend Programming Guide.md` | API 完整说明 | 补充细节 |
| 2 | [tilelang-api-best-practices SKILL.md](../tilelang-custom-skill/tilelang-api-best-practices/SKILL.md) | API 速查 | 快速确认 |
| 3 | **外部参考实现** | **仅用于理解数学逻辑** | **不可直接使用 API** |
| 4 | [tilelang-expert-to-developer SKILL.md](../tilelang-custom-skill/tilelang-expert-to-developer/SKILL.md) | 模式选择 | 辅助决策 |
| 5 | `tilelang/language/__init__.py` + `tilelang/language/*.py` | 公开 API 导出关系与前端定义 | API 定义 |
| 6 | `src/op/` + `src/target/` | lowering 与后端实现状态 | 实现验证 |
| 7 | `testing/python/language/` | 边界用法和测试模式参考 | 测试参考 |

### 冲突处理原则

| 冲突类型 | 处理方式 |
|----------|----------|
| 外部参考实现 API 与本项目 examples 不同 | **以本项目 examples/ 为准** |
| 外部参考实现使用三维 Kernel | **改用本项目的一维 + block_metadata 方案** |
| 外部参考实现使用动态循环边界 | **改用静态边界 + 条件判断方案** |
| 本项目无同类实现 | 使用 tilelang-api-best-practices 中的示例代码 |

**规则**：当信息源之间矛盾时，以 `examples/` 为准。若 `examples/` 未覆盖，以 `docs/` 为准。若 `docs/` 未覆盖，以 `tilelang/language/` 源码实际实现为准。

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

文档生成完成后，输出以下格式的报告：

```
## 设计文档生成报告

- 算子: {算子名称}
- 编程模式: {Developer / Expert / 混合}
- 计算类型: {纯 Vector / 纯 Cube / 混合}
- 输出路径: {文件路径}

### 自检结果
1. 编程模式选型: ✅ / ❌
2. API 映射具体性: ✅ / ❌
3. 内存搬运完整性: ✅ / ❌
4. Tiling 约束分析: ✅ / ❌
5. 同步策略匹配: ✅ / ❌
6. 验证方案覆盖（4 类）: ✅ / ❌
7. 无占位符: ✅ / ❌
8. 技术约束确认: ✅ / ❌
9. 本项目同类实现列出: ✅ / ❌
10. 参考实现差异说明: ✅ / ❌ / N/A
11. 非整除处理策略: ✅ / ❌ / N/A
12. L0C 容量约束: ✅ / ❌ / N/A
13. 无全局变量依赖: ✅ / ❌

### 待确认项
- {列出需要用户进一步确认的内容}
```

## 9. 生成算子
完成报告后，询问用户是否根据此报告生成对应算子代码
