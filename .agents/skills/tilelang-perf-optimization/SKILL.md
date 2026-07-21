---
name: tilelang-perf-optimization
description: TileLang 算子性能调优与潜在性能劣化模式检查。提供瓶颈预判、性能数据采集、多维度交叉优化、效果验证能力；也用于生成或评审算子时对照常见性能劣化模式示例检查当前 kernel 代码。触发：算子精度通过后需要优化性能、性能不及预期时。
---

# TileLang 性能优化

## 工作流程

```
  → Step 0: 瓶颈预判（优化前必做）
  → Step 1: 基线采集（性能 + 精度）
  → Step 2: 算子类型判断
  → Step 3: 阅读参考文档并识别优化点（输出到 optimization_log.md）
  → Step 3.5: 多维度交叉实验矩阵（禁止单维度递增）
  → Step 4: 逐项实施优化点
  → Step 4.5: 组合优化（将各轮最优配置合并验证）
  → Step 5: 效果验证（性能 + 精度）
```

## 核心约束

- **瓶颈先行**：未做瓶颈预判禁止开始优化（Step 0 是门禁）
- **交叉优先**：存在多个优化维度时，必须做交叉实验矩阵，禁止单维度递增
- **逐项实施**：每次 Edit 只改一个优化点，改完立即验证
- **精度优先**：精度未通过禁止性能优化
- **性能验证**：必须使用 `msprof op`，禁止用 Python/Torch 计时
- **Host 轻量化**：禁止 host 侧全量数据搬运（`F.pad`、`.contiguous()`、`.to(dtype)` 等），必须移入 kernel

## 参考文档

- **优化指南**：[optimization-guide.md](references/optimization-guide.md)
- **反模式清单**：[performance-antipatterns.md](references/performance-antipatterns.md)
- **编译器限制清单**：[compiler-limitations.md](references/compiler-limitations.md)
- **同步原语**：[api-schedule-sync.md](../tilelang-custom-skill/tilelang-api-best-practices/references/api-schedule-sync.md)（set_flag / wait_flag / barrier_all / cross_flag 语义与用法）
- **API 用法**：[tilelang-api-best-practices](../tilelang-custom-skill/tilelang-api-best-practices/SKILL.md)
- **编程模式**：[tilelang-expert-to-developer](../tilelang-custom-skill/tilelang-expert-to-developer/SKILL.md)
- **cube最佳实践**：[cube_optimization_path.md](references/best-practices/cube_optimization_path.md)
- **vector最佳实践**：[vector-practices/](references/vector-practices/)

---

## 执行步骤

### Step 0: 瓶颈预判（优化前必做，门禁）

**在任何优化开始前，必须先判断瓶颈类型，否则禁止后续步骤。**

#### 0.1 估算理论最小耗时

```python
# 计算理论最小 GM 带宽
min_read_bytes = sum(input_tensor.numel() * input_tensor.element_size() for input_tensor in inputs)
min_write_bytes = sum(output_tensor.numel() * output_tensor.element_size() for output_tensor in outputs)
total_min_bytes = min_read_bytes + min_write_bytes

# 910B3 HBM 带宽约 1.5 TB/s
hbm_bandwidth = 1.5e12  # bytes/s
theoretical_min_us = total_min_bytes / hbm_bandwidth * 1e6
```

#### 0.2 判断瓶颈类型

| 判断条件 | 瓶颈类型 | 优化方向 |
|---------|---------|---------|
| `当前耗时 / 理论最优 > 3x` | **GM bandwidth-bound** | 减少 pass 数、readback、workspace |
| `当前耗时 / 理论最优 < 2x` | **compute-bound** | 指令融合、AUTO_SYNC=False |
| `当前耗时 < 50μs` | **launch-bound** | Fixed Core、减少 block 数、消除分发开销 |

#### 0.3 输出瓶颈预判报告

在 `optimization_log.md` 中记录：

```
[BOTTLENECK-PREASSESS]
  理论最小耗时: {theoretical_min_us} μs
  当前耗时: {current_us} μs
  比值: {ratio}x
  瓶颈类型: {bandwidth/compute/launch}-bound
  推荐优化方向: {方向列表}
  禁止优化方向: {方向列表}
```

**门禁规则**：
- bandwidth-bound → 禁止指令融合、AUTO_SYNC=False 等 compute 层优化
- compute-bound → 禁止减少 pass 数等 bandwidth 层优化
- launch-bound → 禁止 Double Buffer 等 kernel 内优化

---

### Step 1: 基线采集

在 `examples/{op_name}/` 下查找含 `@tilelang.jit` 的脚本，运行：

```bash
msprof op --kernel-name="main_kernel" --output=./msprof_output python ./examples/{op_name}/<script_name>.py
```

精度未通过 → 禁止后续步骤。

### Step 2: 算子类型判断

**生成翻译后的 Ascend C 代码**：

在算子脚本中，JIT 编译返回的函数对象调用 `get_kernel_source()` 可获取翻译后的 Ascend C 代码：

```python
func = jit_func(batch=B, seq_len=S, ...)
print(func.get_kernel_source())
```

运行脚本后，从输出中搜索关键字判断算子类型：

| 判断依据 | 类型 | 典型算子 |
|---------|------|---------|
| `IS_ASCEND_AIC` 出现 | Cube 型 | GEMM、MatMul、Linear |
| `IS_ASCEND_AIV` 出现 | Vector 型 | RoPE、Softmax、Add |
| 两者均出现 | 混合型 | FlashAttention、SparseFlashAttention |

### Step 3: 识别优化点（强制，禁止与 Step 4 合并）

根据算子类型阅读 `optimization-guide.md` 对应章节 + `performance-antipatterns.md`，如果是 cube 核额外参考 `best-practices/cube_optimization_path.md`，如果是 vector 核额外参考 `vector-practices/` 目录下的文档，如果是多 pass Vector 融合算子额外参考 `best-practices/vector_fused_operator_optimization.md`，在 `optimization_log.md` 中输出：

> **优先级提醒**：多 pass 算子应首先审视算法层优化（pass 数量缩减、readback 模式、自适应 tiling），这些优化的收益通常远超核内优化。详见 `optimization-guide.md` §一.五。

**Part A 优化点清单**：逐条标注适用/不适用 + 原因 + 参考文件行号。`pass_configs` 不是独立优化点，是伴随修改。

```
[#1] [名称]（参考: optimization-guide.md L445-L650 §2.13）：[适用/不适用] — [原因]
```

**Part B `[ORDER-PLAN]`**：分析依赖关系，排出实施顺序链。依赖分析三条规则：
1. **布局依赖**：改变 layout 的优化排在依赖此 layout 的优化之前
2. **数量依赖**：涉及预算的优化排在改变 buffer 数量的优化之后
3. **配置依赖**：涉及 pass_configs 的优化在相关功能实施后才改动

```
[ORDER-PLAN] 实施顺序：
1. [#N] [名称] — 前置依赖: [无] — 理由: [...]
2. [#M] [名称] — 前置依赖: [#N] — 理由: [...]
```

### Step 3.5: 多维度交叉实验矩阵（禁止单维度递增）

**当存在多个独立优化维度时，必须做交叉实验，禁止逐轮单维度递增。**

#### 3.5.1 识别独立维度

常见独立维度（互不影响代码结构）：

| 维度 | 典型取值 |
|------|---------|
| Tiling 参数 | block_M=8/16/32, block_N=128/256/512 |
| Kernel 架构 | 单 kernel / 双 kernel / 混合 kernel |
| 同步模式 | AUTO_SYNC=True / False |
| Pass 数量 | 1-pass / 2-pass / 3-pass |

#### 3.5.2 构建交叉矩阵

选择 2 个最关键维度，构建 2×2 或 3×3 实验矩阵：

```
[CROSS-MATRIX] 维度 A: {tiling} × 维度 B: {架构}

|                | 单 kernel | 双 kernel |
|----------------|----------|----------|
| block_M=16     | 实验 1   | 实验 2   |
| block_M=32     | 实验 3   | 实验 4   |
```

#### 3.5.3 执行与选择

1. 快速实现 4 个实验（可简化代码，不要求完整优化）
2. 每个实验只测 3 个代表性 shape（小/中/大）
3. 选出最优组合，作为后续精细调优的起点

**门禁**：未完成交叉矩阵禁止进入 Step 4 逐项实施。

#### 3.5.4 分发开销评估（双 kernel 前必做）

引入双 kernel / 混合 kernel 前，必须评估分发开销：

```python
# 1. 估算分发开销
dispatch_overhead_us = 10  # .item() host 同步约 5~15μs

# 2. 估算 kernel 耗时
kernel_us = estimated_current_us

# 3. 判断占比
overhead_ratio = dispatch_overhead_us / kernel_us

# 4. 决策
if overhead_ratio > 0.10:
    # 分发开销 > 10%，不适合双 kernel
    use_single_kernel = True
elif overhead_ratio < 0.05:
    # 分发开销 < 5%，适合双 kernel
    use_dual_kernel = True
else:
    # 5~10% 灰色地带，需要混合策略
    use_hybrid = True
```

**日志格式**：
```
[DISPATCH-ASSESS]
  分发开销: {dispatch_overhead_us} μs
  kernel 耗时: {kernel_us} μs
  占比: {overhead_ratio}%
  决策: {single/dual/hybrid}
```

### Step 4: 逐项实施

**固定优先级**：先静态分析（对照 `performance-antipatterns.md`），再 P0 Host 侧优化（`optimization-guide.md` §2.12）。P0 完成后 Host 侧只允许零拷贝形状变换。

**后续优化点**按 `[ORDER-PLAN]` 逐个实施，每个走 6 子步骤：

```
0: ORDER-CHECK → A: Read 文档 → B: Edit 代码 → C: msprof op 验证 → D: 记录结果 → (失败) E: 重读文档修复
```

**门禁**：`[ORDER-CHECK]` 未写禁止 Read；`[IMPL-#N]` 未写禁止 Edit；`[RESULT-#N]` 未写禁止下一个。

**日志格式**：
```
[ORDER-CHECK] 准备实施: [#N] [名称] | 前置依赖: [#1 ✅ / #2 ❌] | 结论: [✅/❌]
[IMPL-#N] 已阅读 <文件> L行号（§X.X），关键约束: ...
[SELF-CHECK] 本次 Edit 只涉及 [#N]
[RESULT-#N] 优化点: [名称] | 精度: [pass/fail] | 性能: [X us] | 对比: [+/-X%]
```

**Double Buffer 特殊要求**：实施前必须完成 `[DB-ANALYSIS]`（Q1: 循环内有 MTE3？Q2: 有跨迭代累加器？Q3: 选同步方式），未完成禁止写代码。

**最佳实践参考**：

| 算子类型 | 文档 | 核心优化技术 |
|---------|------|-------------|
| Vector 型（简单） | [RoPE 优化](references/best-practices/rope-developer-mode.md)、[归约遍数融合](references/vector-practices/vector_reduce_pass_fusion.md) | NPU 内动态生成 Mask、Tile API 向量化、参数简化 |
| Vector 型（融合） | [AddRmsNormDynamicQuant 优化](references/best-practices/vector_fused_operator_optimization.md) | Pass 缩减、Readback、混合 kernel 策略、交替 buffer |
| Cube 型 | [GEMM Intrinsic](references/best-practices/gemm_intrinsic_optimize.md) | 多缓冲流水线、细粒度 Flag 同步、MMA intrinsic、L0 分块、负载均衡 |
| CV 融合型 | [Flash Attention](references/best-practices/flash_attn_optimize.md) | num_stages 流水线、批量 Softmax、Cross-core Semaphore、数据布局优化；**多 shape 适配**（BSND 免转置、Sq==1 decode 窄块、加性 mask 屏蔽变长 Skv） |

### Step 4.5: 组合优化（将各轮最优配置合并验证）

**在所有单维度优化完成后，必须做一轮组合验证，禁止直接结束。**

#### 4.5.1 汇总各轮最优配置

从 `optimization_log.md` 中提取每轮的最优配置：

```
[COMBO-SUMMARY]
  最优算法: {R1: 2-pass}
  最优内存: {R2: Double Buffer}
  最优 Tiling: {R3: block_M=16, block_N=adaptive}
  最优架构: {R4: 双 kernel readback/recompute}
  最优同步: {AUTO_SYNC=True}
```

#### 4.5.2 检查配置冲突

| 冲突类型 | 示例 | 解决方案 |
|---------|------|---------|
| Tiling 与架构冲突 | block_M=16 在双 kernel 下退化 | 混合策略（按 shape 分发） |
| 同步与指令冲突 | AUTO_SYNC=False 需要交替 buffer | 保持 AUTO_SYNC=True |
| 内存与 Tiling 冲突 | block_N=512 导致 UB 溢出 | 保持 block_N=256 |

#### 4.5.3 实现组合版本

将所有无冲突的最优配置合并到一个版本中，完整验证精度 + 性能。

**日志格式**：
```
[COMBO-IMPL] 组合配置: {配置列表}
[COMBO-RESULT] 精度: {pass/fail} | 性能: {X us} | 对比最优单轮: [+/-X%]
```

**门禁**：组合版本性能低于最优单轮版本 → 分析冲突原因，回退到最优单轮版本。

### Step 5: 效果验证

每个优化点后执行：精度验证 → `msprof op` → 记录 → 对比基线。精度失败时保持优化调试，不撤销。

调试手段：`T.printf`、`T.dump_tensor`、`get_kernel_source()`，详见 [Programming Guide](../../../docs/TileLang-Ascend%20Programming%20Guide.md)。

迭代终止：达到目标或连续 3 次无提升则中断上报。

---

## 优化记录

保存在 `examples/{op_name}/perf_tuning/`：
- `baseline.json` - 基线性能
- `optimization_log.md` - 优化记录
- `final_report.md` - 最终报告
