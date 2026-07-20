---
name: tilelang-op-test-design
description: "TileLang-Ascend 算子测试设计技能。支持多种场景：(1) 从 design.md 设计测试配置 (2) 从 examples/{op}/*.py 补充测试 (3) 手动提供算子信息生成测试 (4) 测试覆盖率分析。理解算子实现逻辑后智能判断测试策略。触发：设计算子测试、生成测试用例、补充测试、测试覆盖率不足。"
---

# TileLang-Ascend 算子测试设计

---

## 1. 技能定位与支持场景

### 1.1 支持的多种场景

本技能支持 **4 种主要场景**：

| 场景 | 输入来源 | 适用时机 | 工作流程 |
|------|---------|---------|---------|
| **场景 A** | design.md | 算子设计阶段 | 从设计文档提取信息 → 智能判断测试策略 → 生成测试配置建议 |
| **场景 B** | examples/{op}/*.py | 算子已实现 | 从实现代码提取信息 → 分析现有测试 → 补充缺失用例 |
| **场景 C** | 用户口头描述 | 早期讨论阶段 | 用户交互收集信息 → 智能判断测试策略 → 生成测试模板 |
| **场景 D** | 现有测试分析 | 测试完善阶段 | 分析现有测试覆盖率 → 智能判断缺失场景 → 补充测试用例 |

---

### 1.2 场景触发关键词

| 场景 | 触发关键词示例 |
|------|---------------|
| **场景 A** | "为这个算子设计测试"、"根据 design.md 生成测试配置" |
| **场景 B** | "补充这个算子的测试"、"完善现有测试" |
| **场景 C** | "我想开发一个 softmax 算子，帮我设计测试"、"算子是 xxx，数学公式是 yyy" |
| **场景 D** | "分析测试覆盖率"、"现有测试不够全面，需要补充" |

---

## 2. 算子类别划分依据

### 2.1 划分依据来源

算子类别划分参考 **tilelang-op-design skill §4 算子特征分析决策树**，基于以下三个维度：

1. **计算类型（硬件特性）**
2. **复杂度级别（计算步骤）**
3. **数学公式特征（数学运算）**

---

### 2.2 计算类型划分（硬件特性）

**依据**：算子主要使用哪类硬件单元

| 计算类型 | 使用的硬件单元 | 数学公式特征 | 测试重点 |
|---------|--------------|-------------|---------|
| **纯 Cube** | Cube 核（矩阵乘单元） | 仅含 matmul / @ | 矩阵维度组合、block size |
| **纯 Vector** | Vector 核（向量单元） | 无 matmul，仅 element-wise/reduction | dtype 组合、shape 组合 |
| **混合（CV 融合）** | Cube + Vector 核 | matmul + element-wise 后处理 | 核间协作正确性。Developer 模式默认消除显式 workspace（`threads=2` + 片上直连），测试聚焦 CV 交互结果；Expert/混合或回退才涉及显式 workspace（GM 中转）+ 跨核同步 |

**判断方法**：理解算子实现逻辑后判断

---

### 2.3 复杂度级别划分（计算步骤）

**依据**：算子有多少个计算步骤

| 复杂度级别 | 计算步骤数 | 典型算子 | 测试特点 |
|-----------|----------|---------|---------|
| **单步（Single）** | 1 步 | Add, Mul, ReLU | 简单配置，快速验证 |
| **多步（Multi）** | 2~5 步 | Softmax, LayerNorm | 详细配置，多 dtype |
| **融合（Fusion）** | 多算子组合 | FlashAttention | 复杂配置，测试完整数据流 |

**判断方法**：分析数学公式或算法描述，理解计算步骤后判断

---

### 2.4 数学公式特征划分

**依据**：数学公式中的关键运算

| 数学公式特征 | 算子类别 | 测试策略 |
|-------------|---------|---------|
| 含 `matmul` / `@` / 矩阵乘 | **GEMM 类** | 三维参数（M/N/K），block size 组合多 |
| 含 `exp + sum + div` 组合 | **Softmax 类** | 多 dtype，精度按 dtype 不同 |
| 含 `mean + var + sqrt` 组合 | **Normalization 类** | eps 参数重要，多 dtype |
| 含 `sigmoid` / `relu` / `gelu` | **Activation 类** | 简单配置，逐元素验证 |
| 含 `sum(dim)` / `max(dim)` | **Reduction 类** | 归约维度重要 |

---

### 2.5 综合分类示例

| 算子 | 计算类型 | 复杂度 | 数学特征 | 综合类别 |
|------|---------|--------|---------|---------|
| **MatMul** | 纯 Cube | Single | matmul | GEMM（纯矩阵乘） |
| **Softmax** | 纯 Vector | Multi | exp+sum+div | Softmax（多步归一化） |
| **LayerNorm** | 纯 Vector | Multi | mean+var+sqrt | Normalization（多步归一化） |
| **SiLU** | 纯 Vector | Single | sigmoid | Activation（单步激活） |
| **FlashAttention** | 混合（CV） | Fusion | matmul+softmax+matmul | Fusion（融合算子） |

---

### 2.6 参数约束关系

**C-001：dtype 一致性约束**
大多数 TileLang 算子要求输入输出 tensor dtype 一致。

---

## 3. 算子类别识别方法

### 3.1 核心原则

**算子类别识别方法**：阅读设计文档/代码，理解算子实现逻辑后给出判断。

---

### 3.2 判断流程

```
步骤 1：阅读算子信息
    ├─ 场景 A：阅读 design.md §1.3 数学公式 + §1.4 算法描述
    ├─ 场景 B：阅读 examples/{op}/ 算子实现代码
    ├─ 场景 C：理解用户口头描述的数学公式
    └─ 场景 D：阅读现有测试代码，分析覆盖情况

步骤 2：理解实现逻辑
    ├─ 分析数学公式中的关键运算（matmul/exp/sum/reduce 等）
    ├─ 分析计算步骤数（单步/多步/融合）
    ├─ 分析硬件需求（Cube/Vector/混合）
    └─ 分析参数维度（M/N/K/dim 等）

步骤 3：给出判断
    ├─ 计算类型：纯 Cube / 纯 Vector / 混合
    ├─ 复杂度级别：Single / Multi / Fusion
    ├─ 数学特征：GEMM / Softmax / Activation / Reduction 等
    └─ 综合类别：GEMM（纯） / Softmax / Fusion 等

步骤 4：基于判断生成测试策略
    └─ 不同类别有不同的测试配置生成策略
```

---

### 3.3 判断示例

#### 示例 1：GEMM 算子判断

**阅读信息**：
```
数学公式：C = A @ B
算法描述：矩阵乘法，分块计算
```

**理解逻辑**：
- 公式中只有 `@`（矩阵乘）运算 → 纯 Cube 计算
- 只有 1 个计算步骤 → Single 复杂度
- 没有其他运算 → 纯 GEMM

**判断结果**：
```python
{
    "计算类型": "纯 Cube",
    "复杂度": "Single",
    "数学特征": "matmul",
    "综合类别": "GEMM（纯矩阵乘）",
    "测试策略": {
        "dtype_count": 2,
        "shape_count": 5,  # 多种 M/N/K 组合
        "block_count": 3,  # 多种 block size
        "三维参数": True,   # M/N/K
    }
}
```

---

#### 示例 2：Softmax 算子判断

**阅读信息**：
```
数学公式：softmax(x_i) = exp(x_i) / sum_j(exp(x_j))
算法描述：先计算 max，再 exp，再 sum，最后 div
```

**理解逻辑**：
- 公式中有 exp、sum、div，无 matmul → 纯 Vector 计算
- 有 4 个计算步骤（max → exp → sum → div） → Multi 复杂度
- 是典型的 softmax 公式 → Softmax 类

**判断结果**：
```python
{
    "计算类型": "纯 Vector",
    "复杂度": "Multi",
    "数学特征": "exp+sum+div",
    "综合类别": "Softmax（多步归一化）",
    "测试策略": {
        "dtype_count": 3,  # FP16/FP32/BF16
        "shape_count": 4,
        "block_count": 2,
        "精度按 dtype": True,  # 不同 dtype 精度不同
    }
}
```

---

#### 示例 3：FlashAttention 算子判断

**阅读信息**：
```
数学公式：Attention = softmax(Q @ K^T / sqrt(d)) @ V
算法描述：先 GEMM（Q,K），再 softmax，再 GEMM（attn,V）
```

**理解逻辑**：
- 公式中有两次 matmul + softmax → Cube + Vector 混合计算
- 有 3 个算子组合（GEMM + softmax + GEMM） → Fusion 复杂度
- 是典型的融合算子 → Fusion 类

**判断结果**：
```python
{
    "计算类型": "混合（CV 融合）",
    "复杂度": "Fusion",
    "数学特征": "matmul+softmax+matmul",
    "综合类别": "Fusion（融合算子）",
    "测试策略": {
        "dtype_count": 2,
        "shape_count": 3,
        "block_count": 2,
        "workspace配置": True,  # 仅 Expert/混合或回退写法；Developer 模式默认消除 workspace，此项为 False
    }
}
```

---

## 4. 多场景工作流程

### 4.1 场景 A：从 design.md 输入

**触发**："根据 design.md 设计测试"

**工作流程**：

```
Phase 1：信息提取（强制步骤）
    ├─ 定位 design.md 文件（examples/{op}/design.md）
    ├─ 提取 §1.3 数学公式
    ├─ 提取 §1.4 算法描述（计算步骤）
    ├─ 提取 §2 编程模式
    ├─ 提取 §4 输入输出规格（shape/dtype）
    ├─ 提取 §5 block size
    └─ 提取 §9.3 精度标准（未定义则阻塞，回 Stage 1 补齐）

Phase 2：理解判断
    ├─ 阅读数学公式，理解计算逻辑
    ├─ 判断计算类型（纯 Cube/Vector/混合）
    ├─ 判断复杂度（Single/Multi/Fusion）
    ├─ 判断数学特征（GEMM/Softmax/Activation等）
    └─ 给出综合类别判断

Phase 3：用户交互（补充决策）
    ├─ 询问测试重点（功能验证/全面测试/异常测试）
    ├─ 询问用例数量（快速冒烟/标准测试/全面测试）
    ├─ 询问不规则 shape（自然包含/重点测试/不需要）
    ├─ 询问特殊场景（空 tensor/极值/INF/NAN等）
    └─ 询问精度标准（如 §9.3 未定义）

Phase 3.5：闸门确认（可选，防止错误扩散）
    ├─ 展示测试配置摘要（dtype 组合、shape 组合、特殊场景）
    ├─ 询问用户："测试配置是否正确？是否继续生成测试代码？"
    ├─ 用户确认 → 进入 Phase 4
    └─ 用户否决 → 返回 Phase 2 重新判断
    └─ 适用场景：融合算子（Fusion 类）、多步复杂算子（Multi 复杂度）

Phase 4：生成测试配置
    ├─ 基于判断生成 L0 配置
    ├─ 基于判断生成 L1 配置（含不规则 shape）
    ├─ 基于用户交互生成 L2 配置
    └─ 基于用户交互生成 Boundary 配置

Phase 5：输出测试代码
    └─ 根据算子类别选择对应模板，生成测试代码
```

---

### 4.2 场景 B：从 examples/{op}/ 算子文件输入

**触发**："补充这个算子的测试"

**工作流程**：

```
Phase 1：信息提取（强制步骤）
    ├─ 定位 examples/{op}/ 算子文件（如 silu.py, flash_attn_bhsd.py）
    ├─ 阅读 Kernel 实现代码
    ├─ 提取函数签名（参数列表）
    ├─ 分析已有测试配置
    └─ 分析 pass_configs 配置

Phase 2：理解判断
    ├─ 阅读实现代码，理解计算逻辑
    ├─ 判断算子类别（直接判断）
    └─ 分析现有测试覆盖情况

Phase 3：用户交互（补充决策）
    ├─ 询问测试重点（补充功能测试/补充异常测试）
    ├─ 询问缺失场景（现有测试缺少哪些）
    └─ 询问用例数量

Phase 4：分析测试空白
    ├─ 对比现有测试 vs 理应有测试
    ├─ 识别缺失场景（dtype组合/shape组合/异常场景）
    └─ 生成补充配置

Phase 5：输出补充测试（填充现有 test_{op}.py 的桩体，不新建文件、不碰 kernel）
    ├─ 定位 generate 已生成的 test_{op}.py 中 test_{op}_l1/l2/boundary 三个桩函数
    ├─ 用真实分层用例替换桩体（参 §9.1）：
    │     L1 → 规则+不规则 shape（含尾块）；L2 → 非法输入（应被拒绝）；Boundary → INF/NAN/极值（合法）
    ├─ L1 用 _run_precision（[PRECISION_*]，阻塞）；L2 用 _run_exception（期望拒绝）、Boundary 用 _run_boundary（比精度，不过 WARN）（均 [BOUNDARY_*]，非阻塞）
    └─ 保持 main 分发器与 --level 接口不变（不改 generate 已写好的 main），kernel 仍从 {op}.py import
```

> **场景 B 输出形式（强约束）**：**只填充现有 `test_{op}.py`** 的三个桩函数体，不新建独立 test 文件、**不改 `{op}.py`（kernel）**。generate 在 first_impl 已生成 `test_{op}.py`（含 `from {op} import {op}`、`test_{op}_l1/l2/boundary` 桩 + 稳定的 `main` 分发器）；场景 B 只替换这三个桩函数的函数体，**不改 main、不改 `--level` 接口、不动 L0 与 kernel 文件**。替换后用 `python examples/{op}/test_{op}.py --level all` 跑全量验证。

---

### 4.3 场景 C：用户口头描述

**触发**："我想开发一个 softmax 算子，帮我设计测试"

**工作流程**：

```
Phase 1：用户交互收集信息
    ├─ 询问算子名称
    ├─ 询问数学公式（参考 tilelang-op-design 的交互方式）
    ├─ 询问输入输出规格
    ├─ 询问编程模式偏好
    └─ 询问其他信息（典型配置、性能目标等）

Phase 2：理解判断
    ├─ 基于数学公式理解计算逻辑
    ├─ 判断算子类别（直接判断）
    └─ 给出测试策略建议

Phase 3：生成测试配置
    └─ 基于判断和用户需求生成测试配置

Phase 4：输出测试模板
    └─ 生成测试代码模板（或输出到文件）
```

---

### 4.4 场景 D：测试覆盖率分析

**触发**："分析测试覆盖率，补充缺失用例"

**工作流程**：

```
Phase 1：分析现有测试
    ├─ 阅读现有测试代码
    ├─ 统计已覆盖的 dtype 组合
    ├─ 统计已覆盖的 shape 组合
    ├─ 统计已覆盖的异常场景
    └─ 统计已覆盖的边界场景

Phase 2：判断缺失场景
    ├─ 基于算子类别判断应覆盖的场景
    ├─ 对比现有测试 vs 应覆盖场景
    ├─ 识别缺失的 dtype 组合
    ├─ 识别缺失的 shape 组合（含不规则 shape）
    ├─ 识别缺失的异常场景
    └─ 识别缺失的边界场景

Phase 3：用户交互（确认补充）
    ├─ 展示缺失场景清单
    ├─ 询问是否全部补充或选择性补充
    └─ 询问用例数量

Phase 4：生成补充配置
    └─ 为缺失场景生成测试配置

Phase 5：输出补充测试代码
    └─ 输出补充测试函数
```

---

## 5. 测试分层体系（四层）

| 层级 | 名称 | 用例数 | 测试目标 | Shape 特点 |
|------|------|--------|---------|-----------|
| **L0** | 门槛测试 | ≤50 | 核心功能验证 | 规则 shape（快速冒烟） |
| **L1** | 功能测试 | 100-200 | 参数组合覆盖 | **规则 + 不规则 shape**（自然包含） |
| **L2** | 异常测试 | ≤20 | 非法输入拒绝验证（负向，不比精度） | 非法 shape / 不支持 dtype |
| **Boundary** | 边界测试 | ≤10 | 合法特殊值精度验证（比精度，不过报 WARN） | INF/NAN/极值/空 tensor |

---

## 6. 确定性 Shape 生成（强制非对齐必出）

**规则**：L1 的 shape 集合由 block 反推**确定性生成**，非对齐 / 尾块 / 质数 shape **默认必出**——不再用"是否需要不规则 shape"的软问法。用户只能在此基线上**加量**，不能减到 0。覆盖维度 ID 与判定见 `references/coverage-matrix.md`。

**生成公式**（给定主分块 `block=(bM, bN[, bK])`，`k` 为倍数）：
```python
def gen_l1_shapes(bM, bN, k=4):
    return {
        "D-SHAPE-ALIGNED":  (bM*k,            bN*k),             # block 整除（规则）
        "D-SHAPE-TAIL-1":   (bM*k + 1,        bN*k),             # 余数 1（最易暴露边界 bug）
        "D-SHAPE-TAIL-MID": (bM*k + bM//2,    bN*k + bN//2),     # 中间余数
        "D-SHAPE-PRIME":    (nearest_prime(bM*k), nearest_prime(bN*k)),  # 完全非对齐
        "D-SHAPE-EDGE":     (1, bN*k),                           # 退化（另配 (bM*k,1)/单元素）
    }
```

**类别特例**：
- **GEMM / 含 matmul**：上式扩到 K 轴——至少一条 `K=bK*k+1` 或质数 K，保证 M/N/K **三轴都出现过非对齐**（不能只非对齐 M）。
- **多维算子**：对 proto 支持的每个 rank（2D/3D/4D/5D…）重复「ALIGNED + 一条非对齐」，命中 `D-SHAPE-RANK-<r>`。
- **逐元素激活（无 tiling 边界）**：尾块与对齐路径等价，可对 `D-SHAPE-TAIL-*`/`PRIME` 走豁免（写入 `COVERAGE_NA`，见 §9）。

`nearest_prime(n)` 取 ≤ n 的最近质数，避免超出 proto 支持范围。

---

## 7. 用户交互流程（参考 tilelang-op-design）

### 交互规则（严格遵守）

参考 tilelang-op-design skill §2：

1. **每次只询问一个问题**
2. **按顺序依次询问**
3. **已提供的跳过**

---

### 交互示例

**步骤 1：测试重点**
```
请选择本次测试的重点：
[1] 功能验证（L0+L1） - 快速验证基本功能和参数组合
[2] 全面测试（L0+L1+L2+Boundary） - 完整测试，含异常和边界
[3] 仅补充异常测试（L2） - 现有测试已完善，仅补充异常场景
[4] 精度专项测试 - 重点验证不同 dtype 的精度标准
```

**步骤 2：用例数量**
```
请选择测试用例数量规模：
[1] 快速冒烟（L0≤10, L1≤50）
[2] 标准测试（L0≤50, L1=100-200）
[3] 全面测试（L0≤50, L1=200-300）
```

**步骤 3：不规则 shape 加量（非对齐为强制基线，不可关闭）**
```
不规则 shape（尾块/质数）已由 §6 确定性生成强制包含。请选择是否额外加量：
[1] 标准（推荐） - 仅 §6 强制基线（ALIGNED/TAIL-1/TAIL-MID/PRIME/EDGE 各 1）
[2] 重点加量 - 在强制基线上额外生成更多尾块/质数配置
```
> 说明：不再提供"不需要不规则 shape"选项——非对齐覆盖为覆盖矩阵强制维度，关闭会被 checker 判 MISS。

---

## 8. 完成报告

生成完成后输出报告：

```
## 测试代码生成报告

### 算子信息
- 算子名称: {op_name}
- 输入来源: design.md / examples/{op}/*.py / 用户描述 / 测试分析
- 输入场景: {场景 A/B/C/D}

### 判断结果
1. 计算类型: {纯 Cube / 纯 Vector / 混合} - 基于 {数学公式分析}
2. 复杂度级别: {Single / Multi / Fusion} - 基于 {计算步骤分析}
3. 数学特征: {GEMM / Softmax / Activation 等} - 基于 {关键运算分析}
4. 综合类别: {最终类别判断}

### 用户交互决策
- 测试重点: {用户选择}
- 用例数量: {用户选择}
- 不规则 shape: {用户选择}
- 特殊场景: {用户选择}

### 测试配置统计
- L0: {n} 个用例
- L1: {n} 个用例(规则={n}, 不规则={n})
- L2: {n} 个用例
- Boundary: {n} 个用例

### 覆盖矩阵（逐维度，来自 coverage_check.py）
| 维度 ID | 应覆盖 | 实际数量 | 状态 |
|---|---|---|---|
| D-DTYPE-fp16 | ≥1 | {n} | PASS |
| D-SHAPE-PRIME | ≥1 | {n} | PASS / MISS |
| D-VALRANGE-L | ≥1 | {n} | PASS / MISS |
| D-SPECIAL-INF | ≥1 | {n} | PASS / N/A（理由） |
| ... | ... | ... | ... |

**覆盖结论**：{x} PASS / {y} MISS / {z} N/A → {PASS 全绿 / FAIL 有未豁免 MISS}
> 有未豁免 MISS 时必须先补齐用例再交付，不得直接报 `[PRECISION_PASS]`。

### 输出文件
- 路径: {output_file}
```

---

## 9. 测试代码结构示例

### 9.1 标准测试结构

> 这是 **`test_{op}.py`** 的结构（kernel 在同目录 `{op}.py`，此处从中 import）。测试文件不含 kernel 定义。

```python
import argparse
import os
import sys

import tilelang
import torch

# 从同目录 kernel 文件导入被测 kernel
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from {op} import {op}   # noqa: E402

# ========== 精度标准定义（混合容差，详见 references/precision-standard.md）==========
def get_precision(dtype):
    """返回 (atol, rtol, max_abs_error_limit, required_matched_ratio)。
    浮点：混合容差；整型：精确匹配（0 误差）。"""
    fp_table = {
        # dtype       : (atol,   rtol,   max_abs_error_limit, required_matched_ratio)
        "float16":     (2**-14, 2**-9,  1e-1, 0.99),   # atol 6.10e-5, rtol 1.95e-3
        "bfloat16":    (2**-10, 2**-6,  1e0,  0.99),   # atol 9.77e-4, rtol 1.56e-2
        "float32":     (2**-16, 2**-10, 1e-2, 0.99),   # atol 1.53e-5, rtol 9.77e-4
        "hifloat32":   (2**-16, 2**-10, 1e-2, 0.99),
        "float8_e4m3": (2**-4,  2**-2,  1e0,  0.99),   # atol 0.0625, rtol 0.25
        "float8_e5m2": (2**-3,  2**-1,  1e-1, 0.99),   # atol 0.125,  rtol 0.5
    }
    int_types = {"int8", "int16", "int32", "int64", "uint8"}
    if dtype in int_types:
        return (0.0, 0.0, 0.0, 1.0)          # 整型：精确匹配，一个元素不符即 FAIL
    return fp_table.get(dtype, (2**-14, 2**-9, 1e-1, 0.99))

def check_precision(actual, golden, dtype):
    """精度判定：返回 (passed, matched_ratio, max_abs_error)。
    浮点双门限：matched_ratio ≥ required 且 max_abs_error ≤ max_abs_error_limit；
    整型逐元素精确相等；inf/nan 位置做结构比对，不计入数值容差。"""
    atol, rtol, max_abs_limit, required_ratio = get_precision(dtype)
    a = actual.detach().cpu()
    g = golden.detach().cpu()
    if atol == 0.0 and rtol == 0.0:                      # 整型精确匹配
        mism = (a != g).sum().item()
        total = max(a.numel(), 1)
        return mism == 0, 1.0 - mism / total, (0.0 if mism == 0 else float("inf"))
    a = a.float()
    g = g.float()
    special = ~torch.isfinite(g)                         # inf/nan 位置结构比对
    if special.any():
        if not torch.equal(torch.isnan(a[special]), torch.isnan(g[special])) or \
           not torch.equal(torch.isinf(a[special]), torch.isinf(g[special])):
            return False, 0.0, float("inf")
    m = torch.isfinite(g)                                # golden 有限值位置全比：actual 若为 inf/nan 则计为不达标
    if m.sum().item() == 0:
        return True, 1.0, 0.0
    abs_err = (a[m] - g[m]).abs()                        # actual 为 inf/nan 处 abs_err=inf/nan → 逐元素判 False 且拉高 max_abs
    matched_ratio = (abs_err <= (atol + rtol * g[m].abs())).float().mean().item()
    max_abs_error = abs_err.max().item()
    passed = (matched_ratio >= required_ratio) and (max_abs_error <= max_abs_limit)
    return passed, matched_ratio, max_abs_error

# ========== Golden 函数定义 ==========
def golden_{op}(input_data):
    # 根据算子数学公式实现
    pass

# ========== L0/L1：阻塞层（精度），失败打 [PRECISION_FAIL] 计入退出码 ==========
def _run_precision(level, shape, dtype, block):
    """L0/L1 单用例：通过打 [PRECISION_PASS]，失败打 [PRECISION_FAIL] 并返回 False。"""
    try:
        # 运行 kernel + golden 对比 → out, ref
        passed, ratio, max_abs = check_precision(out, ref, dtype)
        tag = "PASS" if passed else "FAIL"
        print(f"[PRECISION_{tag}] {level} shape={shape} dtype={dtype} "
              f"matched_ratio={ratio:.4f} max_abs={max_abs:.3e}")
        return passed
    except Exception as e:
        print(f"[PRECISION_FAIL] {level} shape={shape} dtype={dtype}: {e}")
        return False

# ========== L2：异常测试（负向，非阻塞）——非法输入应被拒绝 ==========
def _run_exception(name, fn):
    """L2 单用例：fn() 喂非法输入，期望被算子拒绝。
    抛异常 → [BOUNDARY_PASS]（正确拒绝）；未抛 → [BOUNDARY_WARN]（应拒绝却静默接受）。均非阻塞。"""
    try:
        fn()
    except Exception as e:
        print(f"[BOUNDARY_PASS] l2 {name}: 正确拒绝 ({type(e).__name__})")
        return
    print(f"[BOUNDARY_WARN] l2 {name}: 非法输入未被拒绝（静默接受）")

# ========== Boundary：边界/特殊值（精度，非阻塞）——合法极值需满足精度验收标准 ==========
def _run_boundary(name, dtype, fn):
    """Boundary 单用例：合法特殊值（INF/NAN/极值/空 tensor），fn() 返回 (out, ref)。
    按精度验收标准比对（check_precision，与 L0/L1 同一套 dtype 阈值）：
    精度过 → [BOUNDARY_PASS]；精度不过或抛异常 → [BOUNDARY_WARN]。均非阻塞，不计入退出码。"""
    try:
        out, ref = fn()
        passed, ratio, max_abs = check_precision(out, ref, dtype)
        tag = "PASS" if passed else "WARN"
        print(f"[BOUNDARY_{tag}] boundary {name} dtype={dtype} "
              f"matched_ratio={ratio:.4f} max_abs={max_abs:.3e}")
    except Exception as e:
        print(f"[BOUNDARY_WARN] boundary {name} dtype={dtype}: {e}")

# ========== L0 测试：门槛测试（规则 shape，block 整除）==========
def test_{op}_l0():
    """L0 门槛测试：快速冒烟（来自 DESIGN.md §9.2 L0 计划）。返回是否全过。"""
    test_configs = [
        ("float16", {shape}, {block}),
        ("float32", {shape}, {block}),
    ]
    ok = True
    for dtype, shape, block in test_configs:
        ok &= _run_precision("l0", shape, dtype, block)
    return ok

# ========== 覆盖标注（机器可校验，详见 references/coverage-matrix.md）==========
# 每条 L1 用例带 tags=命中的覆盖维度 ID；coverage_check.py 反查命中集合。
# (shape, dtype, block, value_range, tags)
L1_CASES = [
    ((512, 512),        "float16",  {block}, (-1, 1),   ["D-DTYPE-fp16","D-SHAPE-ALIGNED","D-VALRANGE-S"]),
    ((512, 512),        "float32",  {block}, (-10, 10), ["D-DTYPE-fp32","D-SHAPE-ALIGNED","D-VALRANGE-M"]),
    ((512, 512),        "bfloat16", {block}, (-1, 1),   ["D-DTYPE-bf16","D-SHAPE-ALIGNED"]),
    ((513, 512),        "float16",  {block}, (-1, 1),   ["D-SHAPE-TAIL-1"]),           # 余数1
    ((512+64, 512+64),  "float16",  {block}, (-1, 1),   ["D-SHAPE-TAIL-MID"]),         # 中间余数
    ((509, 503),        "bfloat16", {block}, (-1, 1),   ["D-SHAPE-PRIME"]),            # 质数非对齐
    ((1, 512),          "float16",  {block}, (-1, 1),   ["D-SHAPE-EDGE"]),             # 退化
    ((512, 512),        "float16",  {block}, (-50, 50), ["D-VALRANGE-L"]),             # 大值域
    ((512, 512),        "float16",  {block}, (-5, 10),  ["D-VALRANGE-ASYM"]),          # 非对称
]
# 覆盖汇总（coverage_check.py 与上面 tags 二选一，建议同时给出便于核对）
COVERAGE_MANIFEST = {}      # 由 tags 自动汇总，或手填各维度计数
COVERAGE_NA = {}            # 合理缺失的豁免：{"D-SPECIAL-INF": "纯整数算子无浮点特殊值"}

# ========== L1 测试：功能测试（确定性非对齐 shape，见 §6）==========
def test_{op}_l1():
    """L1 功能测试：参数组合覆盖，⭐ 强制含尾块/质数 shape。返回是否全过。"""
    ok = True
    for shape, dtype, block, vrange, tags in L1_CASES:
        ok &= _run_precision("l1", shape, dtype, block)  # 可用 vrange 控制输入分布
    return ok

# ========== L2 测试：异常测试（负向，非阻塞）——非法输入应被拒绝 ==========
def test_{op}_l2():
    """L2 异常测试：不支持的 dtype / 非法 shape 应被算子拒绝。
    正确抛异常 = PASS，静默接受 = WARN。仅记录，不阻塞。"""
    _run_exception("unsupported_dtype", lambda: ...)  # 喂不支持的 dtype，期望报错
    _run_exception("illegal_shape",     lambda: ...)  # 喂非法 shape，期望报错

# ========== Boundary 测试：边界/特殊值（精度，非阻塞）==========
def test_{op}_boundary():
    """Boundary 测试：INF/NAN/极值/空 tensor（合法输入），按精度验收标准比对；
    精度不过打 [BOUNDARY_WARN]，不阻塞。每个 lambda 造特殊值输入 → 跑 kernel + golden → 返回 (out, ref)。"""
    _run_boundary("inf",   {dtype}, lambda: ...)  # 造含 inf 输入 → (out, ref)
    _run_boundary("nan",   {dtype}, lambda: ...)  # 造含 nan 输入 → (out, ref)
    _run_boundary("empty", {dtype}, lambda: ...)  # 造空 tensor 输入 → (out, ref)

# ========== 主函数：--level 分发 + 退出码 ==========
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", default="all",
                        choices=["l0", "l1", "l2", "boundary", "all"])
    args = parser.parse_args()

    tilelang.disable_cache()
    torch.manual_seed(0)

    blocking_ok = True  # 仅 L0/L1 计入退出码
    if args.level in ("l0", "all"):
        blocking_ok &= test_{op}_l0()
    if args.level in ("l1", "all"):
        blocking_ok &= test_{op}_l1()
    if args.level in ("l2", "all"):
        test_{op}_l2()         # 非阻塞
    if args.level in ("boundary", "all"):
        test_{op}_boundary()   # 非阻塞

    if blocking_ok:
        print("Test Passed!")  # L0/L1 全过；bench_test.sh 据此判定
        sys.exit(0)
    sys.exit(1)

if __name__ == "__main__":
    main()
```

> **导入**：`test_{op}.py` 顶部需 `import argparse, os, sys`（连同 `tilelang` / `torch`），并 `sys.path.insert(0, ...)` + `from {op} import {op}` 导入 kernel。**kernel 定义在 `{op}.py`，测试文件不含 kernel。**
> **场景 B 扩展时**：保留 generate 生成的 `main` 分发器与 `--level` 接口不变，仅把 `test_{op}.py` 里 `test_{op}_l1/l2/boundary` 的桩体替换为上述真实实现（见 §4.2）；**不改 `{op}.py`**。

---

### 9.2 关键设计要点

| 要点 | 说明 |
|------|------|
| **精度标准** | 混合容差：按 **dtype** 取 (atol, rtol, max_abs_error_limit, required_matched_ratio)，与算子类别无关；整型 0 误差精确匹配（详见 references/precision-standard.md） |
| **Golden 函数** | 根据数学公式实现，可用 PyTorch 标准实现 |
| **L0 测试** | 规则 shape，快速冒烟（≤10 用例） |
| **L1 测试** | 规则 + 不规则 shape，自然包含尾块（100-200 用例） |
| **L2 测试** | 负向测试：非法 dtype / shape 应被拒绝——**正确抛异常 = PASS，静默接受 = WARN**；用 `_run_exception`，不比精度（无合法 golden），≤20 用例，非阻塞 |
| **Boundary 测试** | 合法特殊值（INF/NAN/极值/空 tensor）——用 `_run_boundary` 跑 kernel+golden，**按精度验收标准（check_precision）比对，精度不过 = WARN**，≤10 用例，非阻塞 |
| **分层标记** | L0/L1 → `[PRECISION_PASS]`/`[PRECISION_FAIL]`（阻塞，计入退出码）；L2/Boundary → `[BOUNDARY_PASS]`/`[BOUNDARY_WARN]`（非阻塞，不改退出码） |
| **退出码** | L0/L1 全过 → 打印 `"Test Passed!"` 且 `exit(0)`；L0/L1 任一失败 → `exit(1)`；L2/Boundary 失败不影响退出码 |
| **--level 分发** | main 支持 `--level {l0,l1,l2,boundary,all}`；精度收敛跑 `l0`，扩展后跑 `all` |
| **异常隔离** | L2 用 `_run_exception`（期望拒绝，抛异常 = PASS）、Boundary 用 `_run_boundary`（比精度，精度不过 = WARN）；两者都 `try/except` 包裹、非阻塞、失败后继续，不得中断后续用例 |
| **覆盖标注** | 每条 L1 用例带 `tags=`（命中的 `D-*` 维度 ID）；文件含 `COVERAGE_MANIFEST` / `COVERAGE_NA`。无标注 → checker 判 MISS |
| **覆盖门禁** | 扩展完成后必须跑 `scripts/coverage_check.py test_{op}.py`；任一**强制维度** MISS → 退出码 1，等同自检失败，须补齐用例后再判 `[PRECISION_PASS]`（见 §10.1） |

---

## 10. 覆盖门禁与总结

### 10.1 覆盖自检门禁（强制步骤）

生成 / 扩展用例后，**必须**执行覆盖自检，确保"skill 描述的每类场景"真正落进了 `test_{op}.py`：

```
步骤 1：判定应覆盖维度
    └─ 用 references/operator-category.md 判出算子类别
    └─ 查 references/coverage-matrix.md 第二节得到「强制维度集」
    └─ 结合 proto.yaml 的 dtype/attr/shape 范围实例化各维度最小数量

步骤 2：生成带标注的用例
    └─ 每条 L1 用例带 tags（命中的 D-* 维度 ID）
    └─ 写 COVERAGE_MANIFEST（计数）+ COVERAGE_NA（合理缺失 + 理由）
    └─ shape 集合遵循 §6 确定性生成（非对齐必出）

步骤 3：跑 checker
    └─ python scripts/coverage_check.py examples/{op}/test_{op}.py --proto examples/{op}/proto.yaml
    └─ 打印逐维度 PASS / MISS / N/A 覆盖矩阵

步骤 4：判定
    └─ 任一【强制维度】MISS（未豁免，或对强制项写了豁免）→ 退出码 1
       视为自检失败：补齐缺失维度的用例 → 重跑 checker，直至全 PASS/N/A
    └─ 全 PASS / N/A → 退出码 0，方可交付 / 报 [PRECISION_PASS]
```

> **与 orchestrator 衔接**：场景 B（developer agent Stage 2 扩展 L1/L2/Boundary）完成后，覆盖门禁与 `[PRECISION_PASS]` 并列为交付前置条件——覆盖矩阵有未豁免 MISS 时不得返回 `[PRECISION_PASS]`。详见 developer agent「分层测试与扩展流程」与 AGENTS.md Stage 2 门禁。

### 10.2 总结

#### 核心要点

1. **支持多种场景**：不只是 design.md，还支持 examples/{op}/*.py、用户描述、测试分析
2. **算子类别划分依据科学**：基于硬件特性、计算步骤、数学公式三个维度
3. **算子类别识别方法正确**：理解实现逻辑后判断
4. **覆盖从"描述性"转"契约式"**：非对齐等场景由 §6 确定性生成 + 覆盖矩阵强制 + checker 门禁三重保证，杜绝"漏场景"

#### 技能文件结构

```
tilelang-op-test-design/
├── SKILL.md                    # 主文档（多场景 + 判断 + 覆盖门禁）
├── scripts/
│   └── coverage_check.py       # 覆盖自检 checker（应覆盖 vs 实际覆盖）
└── references/
    ├── operator-category.md    # 算子类别划分依据（详细）
    ├── precision-standard.md   # 精度标准体系
    └── coverage-matrix.md      # 测试覆盖矩阵（强制契约：维度 ID / 谓词 / 最小数量）
```

**说明**：
- SKILL.md 包含完整方法论、测试代码结构示例（§9）、覆盖门禁（§10.1）
- references/operator-category.md 提供算子类别划分详细说明
- references/precision-standard.md 提供混合容差精度标准体系（按 dtype 的 atol/rtol/max_abs_error_limit/required_matched_ratio；整型 0 误差精确匹配）
- references/coverage-matrix.md 提供覆盖维度强制契约，是 coverage_check.py 的判定依据