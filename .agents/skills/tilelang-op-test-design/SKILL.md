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
    └─ 提取 §8 精度标准（如有）

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
    └─ 询问精度标准（如 §8 未定义）

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

Phase 5：输出补充测试（追加到现有 example_{op}.py，不新建文件）
    ├─ 定位 generate 已生成的 example_{op}.py 中 test_{op}_l1/l2/boundary 三个桩函数
    ├─ 用真实分层用例替换桩体（参 §9.1）：
    │     L1 → 规则+不规则 shape（含尾块）；L2 → 异常输入；Boundary → INF/NAN/极值
    ├─ L1 用 _run_precision（[PRECISION_*]，阻塞）；L2/Boundary 用 _run_boundary（[BOUNDARY_*]，非阻塞）
    └─ 保持 main 分发器与 --level 接口不变（不改 generate 已写好的 main）
```

> **场景 B 输出形式（强约束）**：**追加/填充到现有 `example_{op}.py`**，不新建独立 test 文件。generate 在 first_impl 已生成 `test_{op}_l1/l2/boundary` 桩 + 稳定的 `main` 分发器；场景 B 只替换这三个桩函数的函数体，**不改 main、不改 `--level` 接口、不动 L0 与 kernel**。替换后用 `--level all` 跑全量验证。

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
| **L2** | 异常测试 | ≤20 | 异常场景验证 | 任意 shape |
| **Boundary** | 边界测试 | ≤10 | 特殊值验证 | 特殊 shape |

---

## 6. 不规则 Shape 自然包含

**规则**：在 L1 测试中自然包含不规则 shape，根据用户需求调整数量。

**生成逻辑**：
- 规则 shape：block size 整除的典型配置
- 不规则 shape：有余数的配置（自然包含尾块场景）

**常见不规则配置**：
```python
# 自然包含在 L1 中
irregular_shapes = [
    (32*3+30, 32*2),   # 余数30
    (32*3+1, 32*2),    # 余数1
    (100, 100),         # 余数4
]
```

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

**步骤 3：不规则 shape**
```
是否需要测试不规则 shape（含尾块场景）？
[1] 自然包含（推荐） - 在 L1 中自然包含不规则 shape
[2] 重点测试 - 特别关注尾块场景，生成更多不规则配置
[3] 不需要 - 仅测试规则 shape
```

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

### 输出文件
- 路径: {output_file}
```

---

## 9. 测试代码结构示例

### 9.1 标准测试结构

```python
import argparse
import sys

import tilelang
import tilelang.language as T
import torch

# ========== 精度标准定义 ==========
def get_precision(dtype):
    precision_map = {
        "float16": (1e-3, 1e-3),
        "float32": (1e-5, 1e-5),
        "bfloat16": (1e-2, 5e-3),
    }
    return precision_map.get(dtype, (1e-3, 1e-3))

# ========== Golden 函数定义 ==========
def golden_{op}(input_data):
    # 根据算子数学公式实现
    pass

# ========== L0/L1：阻塞层（精度），失败打 [PRECISION_FAIL] 计入退出码 ==========
def _run_precision(level, shape, dtype, block):
    """L0/L1 单用例：通过打 [PRECISION_PASS]，失败打 [PRECISION_FAIL] 并返回 False。"""
    atol, rtol = get_precision(dtype)
    try:
        # 运行 kernel + golden 对比
        torch.testing.assert_close(out.cpu(), ref.cpu(), rtol=rtol, atol=atol)
        print(f"[PRECISION_PASS] {level} shape={shape} dtype={dtype}")
        return True
    except Exception as e:
        print(f"[PRECISION_FAIL] {level} shape={shape} dtype={dtype}: {e}")
        return False

# ========== L2/Boundary：非阻塞层，失败打 [BOUNDARY_WARN] 后继续，不改退出码 ==========
def _run_boundary(level, name, fn):
    """L2/Boundary 单用例：通过打 [BOUNDARY_PASS]，失败打 [BOUNDARY_WARN] 后继续。"""
    try:
        fn()
        print(f"[BOUNDARY_PASS] {level} {name}")
    except Exception as e:
        print(f"[BOUNDARY_WARN] {level} {name}: {e}")

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

# ========== L1 测试：功能测试（规则 + 不规则 shape）==========
def test_{op}_l1():
    """L1 功能测试：参数组合覆盖，⭐ 自然包含尾块。返回是否全过。"""
    dtypes = ["float16", "float32", "bfloat16"]
    shapes = [
        (128, 128),      # 规则 shape
        (512, 512),      # 规则 shape
        (100, 100),      # 不规则 shape（余数4）
        (32*3+30, 32*2), # 不规则 shape（余数30）
    ]
    ok = True
    for dtype in dtypes:
        for shape in shapes:
            ok &= _run_precision("l1", shape, dtype, {block})
    return ok

# ========== L2 测试：异常测试（非阻塞）==========
def test_{op}_l2():
    """L2 异常测试：不支持的 dtype / 不合法 shape。失败只记录，不阻塞。"""
    _run_boundary("l2", "unsupported_dtype", lambda: ...)
    _run_boundary("l2", "illegal_shape", lambda: ...)

# ========== Boundary 测试：边界/特殊值（非阻塞）==========
def test_{op}_boundary():
    """Boundary 测试：INF/NAN/极值/空 tensor。失败只记录，不阻塞。"""
    _run_boundary("boundary", "inf", lambda: ...)
    _run_boundary("boundary", "nan", lambda: ...)

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

> **导入**：文件顶部需 `import argparse, sys`（连同 `tilelang` / `torch`）。
> **场景 B 扩展时**：保留 generate 生成的 `main` 分发器与 `--level` 接口不变，仅把 `test_{op}_l1/l2/boundary` 的桩体替换为上述真实实现（见 §4.2）。

---

### 9.2 关键设计要点

| 要点 | 说明 |
|------|------|
| **精度标准** | 根据算子类别和 dtype 定义 rtol/atol |
| **Golden 函数** | 根据数学公式实现，可用 PyTorch 标准实现 |
| **L0 测试** | 规则 shape，快速冒烟（≤10 用例） |
| **L1 测试** | 规则 + 不规则 shape，自然包含尾块（100-200 用例） |
| **L2 测试** | 异常场景（≤20 用例） |
| **Boundary 测试** | 极值、空 tensor（≤10 用例） |
| **分层标记** | L0/L1 → `[PRECISION_PASS]`/`[PRECISION_FAIL]`（阻塞，计入退出码）；L2/Boundary → `[BOUNDARY_PASS]`/`[BOUNDARY_WARN]`（非阻塞，不改退出码） |
| **退出码** | L0/L1 全过 → 打印 `"Test Passed!"` 且 `exit(0)`；L0/L1 任一失败 → `exit(1)`；L2/Boundary 失败不影响退出码 |
| **--level 分发** | main 支持 `--level {l0,l1,l2,boundary,all}`；精度收敛跑 `l0`，扩展后跑 `all` |
| **异常隔离** | L2/Boundary 用例必须 `try/except` 包裹，失败打 `[BOUNDARY_WARN]` 后继续，不得中断后续用例 |

---

## 10. 总结

### 核心要点

1. **支持多种场景**：不只是 design.md，还支持 examples/{op}/*.py、用户描述、测试分析
2. **算子类别划分依据科学**：基于硬件特性、计算步骤、数学公式三个维度
3. **算子类别识别方法正确**：理解实现逻辑后判断
4. **不规则 shape 自然包含**：在 L1 中自然包含，不特殊强调

### 技能文件结构

```
tilelang-op-test-design/
├── SKILL.md                    # 主文档（多场景 + 判断）
└── references/
    ├── operator-category.md    # 算子类别划分依据（详细）
    └── precision-standard.md   # 精度标准体系
```

**说明**：
- SKILL.md 包含完整方法论和测试代码结构示例（§9）
- references/operator-category.md 提供算子类别划分详细说明
- references/precision-standard.md 提供精度标准体系（不同 dtype/算子类别的 rtol/atol）