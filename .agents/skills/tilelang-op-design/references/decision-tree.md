# 算子特征分析决策树（Ascend 版）

## 目录

- [1. 函数设计原则](#1-函数设计原则)
- [2. 决策树](#2-决策树)
- [3. NPU 硬件约束](#3-npu-硬件约束)
- [4. API 映射规则](#4-api-映射规则)

> 本项目为 TileLang-Ascend，与 GPU 版 TileLang 在 Kernel 维度、threads、循环边界、GEMM API、内存分配等方面有显著差异。完整对比与约束清单见 SKILL 子目录索引下的 `ascend-constraints.md`。外部参考实现仅用于理解数学逻辑，API 映射必须查阅本项目。

---

## 1. 函数设计原则

1. **维度参数自推导**：算子调用函数（如 `conv_im2col_gemm`）应从输入 tensor shape 提取 B/C/H/W 等维度，不依赖模块级全局变量。这保证多场景顺序测试时不发生变量污染。
2. **Host 预处理显式声明**：若计算的一部分在 Python 侧完成（如 im2col），必须在 §1 算法描述和 §4 数据流中明确标注。

## 2. 决策树

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

## 3. NPU 硬件约束

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

违反约束会导致编译错误或运行时错误。详见 `.agents/skills/tilelang-custom-skill/tilelang-api-best-practices/references/api-kernel-memory.md`。

## 4. API 映射规则

| 类别 | Ascend 专用 API（推荐） | 通用 API（本项目不推荐/不支持） |
|------|------------------------|-------------------------------|
| GEMM | `T.gemm_v0` | `T.gemm`（可能不支持） |
| 内存分配（Expert）| `T.alloc_L1`, `T.alloc_L0C`, `T.alloc_ub` | - |
| 内存分配（Developer）| `T.alloc_shared`, `T.alloc_fragment` | - |
| Kernel | `T.Kernel(一维, is_npu=True)` | `T.Kernel(三维)` ❌ |
| 同步 | `T.barrier_all()`, `T.Scope("C")` | 自动同步（Developer 模式） |
| 循环 | `T.serial`, `T.unroll` | `T.Pipelined(动态边界)` ❌ |
