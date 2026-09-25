---
name: tilelang-perf-optimization
description: TileLang 算子性能调优与潜在性能劣化模式检查。提供性能数据采集、瓶颈诊断、优化实施、效果验证能力；也用于生成或评审算子时对照常见性能劣化模式示例检查当前 kernel 代码。触发：算子精度通过后需要优化性能、性能不及预期时。
---

# TileLang 性能优化

## 工作流程

```
Step 1: 基线采集（性能 + 精度）
  → Step 2: 算子类型判断
  → Step 3: 阅读参考文档并识别优化点（输出到 optimization_log.md）
  → Step 4: 逐项实施优化点
  → Step 5: 效果验证（性能 + 精度）
```

## 核心约束

- **逐项实施**：每次 Edit 只改一个优化点，改完立即验证
- **精度优先**：精度未通过禁止性能优化
- **性能验证**：必须使用 `msprof op`，禁止用 Python/Torch 计时
- **Host 轻量化**：禁止 host 侧全量数据搬运（`F.pad`、`.contiguous()`、`.to(dtype)` 等），必须移入 kernel

## 参考文档

- **优化指南**：[optimization-guide.md](references/optimization-guide.md)
- **反模式清单**：[performance-antipatterns.md](references/performance-antipatterns.md)
- **API 用法**：[tilelang-api-best-practices](../tilelang-custom-skill/tilelang-api-best-practices/SKILL.md)
- **编程模式**：[tilelang-programming-model-guide](../tilelang-custom-skill/tilelang-programming-model-guide/SKILL.md)
- **cube最佳实践**：[cube_optimization_path.md](references/best-practices/cube_optimization_path.md)
- **vector最佳实践**：[vector-practices/](references/vector-practices/)
---

## 执行步骤

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

### Step 2.5: Roofline 瓶颈分析（Vector 核强制，Cube 核建议）

算子类型的判断只回答了"算什么核"，但"优化方向选 GM 流量还是计算吞吐"需要通过 Arithmetic Intensity 判定。跳过这一步容易在 Memory-Bound 算子上做指令融合、或在 Compute-Bound 算子上做 Pass 融合——方向性浪费。

**计算步骤**：

1. **GM 流量统计**：统计 kernel 所有 pass 的 GM 读取/写入总字节数。多 pass 架构要算每遍读取的所有 tensor（含 weight、参数 tile）。
2. **有效 FLOPs 统计**：统计每次 kernel 执行的浮点操作数（包括 add、mul、cmp 等，cast 可近似算 0.5 FLOP）。
3. **Arithmetic Intensity**：`AI = FLOPs / GM_Bytes`（单位 FLOP/Byte）。
4. **对照硬件平衡点**：

   | 核类型 | 平衡点 (FLOP/Byte) | 说明 |
   |--------|-------------------|------|
   | AIV (Vector) | 10-20 | Vector pipe 单位带宽吞吐 |
   | AIC (Cube) | 30-50 | Matrix pipe 单位带宽吞吐 |
   | Mix | 取主路径 | 按实际耗时主导核判定 |

5. **判定结论**：

   | 判定 | 条件 | 主要优化方向 |
   |------|------|-------------|
   | 强 Memory-Bound | AI < 平衡点 / 2 | **减少 GM 流量**：Pass 融合（归约/输出）、block_N 扩展、数据复用 |
   | 弱 Memory-Bound | 平衡点/2 ≤ AI ≤ 平衡点 | GM 流量和计算吞吐并行 |
   | Compute-Bound | AI > 平衡点 × 2 | **提升计算吞吐**：指令融合、分块计算、Double Buffer 重叠 |

6. **理论最小时间**：`T_min = GM_Bytes / HBM_Bandwidth`（910B ≈ 1.6 TB/s）。实测时间 / T_min 即为当前效率指标。

**输出位置**：写入 `optimization_log.md` 的 Roofline 分析段落，包含 FLOPs、GM 字节、AI 值、判定结论、理论最小时间、理论最大加速比。Memory-Bound 算子的优化方向全部围绕"减少 GM 流量"展开，后续 Part A 只列出此类优化的适用项；Compute-Bound 算子反过来。

### Step 3: 识别优化点（强制，禁止与 Step 4 合并）

先读取 `optimization-guide.md` 的目录和各章节标题 + `performance-antipatterns.md` 的各条目标题，根据算子类型（Step 2 判定）和算子实现特征初步筛选出可能适用的优化点清单。再针对每个候选优化点，读取其"适用场景"、"约束"、"使用条件"等描述，确认是否真正适用。如果是 cube 核额外参考 `best-practices/cube_optimization_path.md`，如果是 vector 核额外参考 `vector-practices/` 目录下的文档。

在 `optimization_log.md` 中输出：

**Part A 优化点清单**：逐条标注适用/不适用 + 原因 + 参考文件行号。`pass_configs` 不是独立优化点，是伴随修改。

```
[#1] [名称]（参考: optimization-guide.md L445-L650 §2.13）：[适用/不适用] — [原因]
```

若候选会扩大现有 kernel 的调用域（新增 shape、batch、dtype、尾块或多 stage），
Part A 必须先写：

```text
[KERNEL-REUSE-PRECHECK]
candidate: <优化点及 reference 章节>
expanded_domain: <新增调用域>
compatibility_dimensions: <从该章节读取，不自行猜测>
unknown_or_false: <none/列表>
plan: direct-reuse / repair-kernel-first / new-kernel-first
```

存在 `unknown/false` 时，ORDER-PLAN 不得写 direct reuse。

> **重点**：每节都需读取其"适用场景"和"约束"描述确认是否适用。仅当章节标题明确标注了特定算子类型专属（如"Cube 核"）且当前算子不属于该类型时，才可初步排除；其余所有章节必须读约束确认，不得仅凭标题或算子类型跳过。

**Part B `[ORDER-PLAN]`**：先按**三层优先级框架**确定大跨层执行顺序，每层内按三条依赖规则确定细粒度顺序。

#### 三层优先级框架

改变 kernel 结构的优化必须先做完（因为上层改动会迫使下层返工），稳定代码上的微优化放到最后（避免被上层改动破坏）。如果某层所有优化点都被标记为"不适用"，直接跳到下一层。

| 层次 | 定义 | 典型优化项 | 执行理由 |
|------|------|-----------|---------|
| **L0 算法层** | 改变计算等价性、减少扫描遍数、改变数据流 | Pass 融合（归约遍数 + 输出遍数）、Online 修正公式、数学等价变换 | 改变 kernel 结构 → 后续所有预算和循环都需重算 |
| **L1 架构层** | 调整 tile 参数、减少 DMA 头开销 | block_N/block_M 扩展（UB 预算反推）、多行 Tile 粒度 | 改变循环次数，是 Double Buffer 的前置依赖 |
| **L2 流水层** | 重叠搬运与计算 | Double Buffer、T.Pipelined、num_stages | 依赖循环结构和 tile 参数已稳定 |
| **L3 细节层** | 不改变结构的指令/配置级微调 | 指令融合（mul_add_dst、rsqrt）、Fixed Core、pass_configs 微调、Host 侧优化 | 在稳定代码上做微优化，避免被上层改动破坏 |

#### 层内依赖规则

层内按以下三条依赖规则排序：
1. **布局依赖**：改变 layout 的优化排在依赖此 layout 的优化之前
2. **数量依赖**：涉及预算的优化排在改变 buffer 数量的优化之后
3. **配置依赖**：涉及 pass_configs 的优化在相关功能实施后才改动

#### 输出模板

每条输出前加层次标签，便于回溯优化策略的演进：

```
[ORDER-PLAN] 实施顺序：
1. [L0] [#1] Pass 融合 — 前置依赖: [无] — 理由: 最高收益 (33% GM 节省)，改变 kernel 架构
2. [L1] [#2] block_N 扩展 — 前置依赖: [#1] — 理由: 融合后 UB 占用变化，需重新计算预算
3. [L2] [#3] Double Buffer — 前置依赖: [#1, #2] — 理由: 需在确定循环结构和 tile 参数后实施
4. [L3] [#4] 指令融合 mul_add — 前置依赖: [#2] — 理由: 代码结构稳定后做微优化
5. [L3] [#6] Fixed Core — 前置依赖: [#1-4] — 理由: 不改变 kernel 结构
```

### Step 4: 逐项实施

**固定优先级**：先静态分析（对照 `performance-antipatterns.md`），再 P0 Host 侧优化（`optimization-guide.md` §2.12）。P0 完成后 Host 侧只允许零拷贝形状变换。

**后续优化点**按 `[ORDER-PLAN]` 逐个实施，每个走 6 子步骤：

```
0: ORDER-CHECK → A: Read 文档 → B: Edit 代码 → C: msprof op 验证 → D: 记录结果 → (失败) E: 重读文档修复
```

**门禁**：`[ORDER-CHECK]` 未写禁止 Read；`[IMPL-#N]` 未写禁止 Edit；`[RESULT-#N]` 未写禁止下一个。

Edit 调用方、dispatch 或 planner 前，再写 `[KERNEL-REUSE-AUDIT]` 复核候选章节规定的
兼容性维度。任一项仍为 unknown/false，先修复或新建 kernel，并单独验证后再扩大调用域。
不同优化点使用各自 reference 章节的维度，不套用其他优化的专属字段。

**日志格式**：
```
[ORDER-CHECK] 准备实施: [#N] [名称] | 前置依赖: [#1 ✅ / #2 ❌] | 结论: [✅/❌]
[IMPL-#N] 已阅读 <文件> L行号（§X.X），关键约束: ...
[KERNEL-REUSE-AUDIT] 依据: <章节> | 维度及结论: <...> | 结论: <reuse/repair/new>
[SELF-CHECK] 本次 Edit 只涉及 [#N]
[RESULT-#N] 优化点: [名称] | 精度: [pass/fail] | 性能: [X us] | 对比: [+/-X%]
```

**Double Buffer 特殊要求**：实施前必须完成 `[DB-ANALYSIS]`（Q1: 循环内有 MTE3？Q2: 有跨迭代累加器？Q3: 选同步方式），未完成禁止写代码。

**最佳实践参考**：

| 算子类型 | 文档 | 核心优化技术 |
|---------|------|-------------|
| Vector 型 | [RoPE 优化](references/best-practices/rope-developer-mode.md)、[归约遍数融合](references/vector-practices/vector_reduce_pass_fusion.md) | | NPU 内动态生成 Mask、Tile API 向量化、参数简化 |
| Cube 型 | [GEMM Intrinsic](references/best-practices/gemm_intrinsic_optimize.md) | 多缓冲流水线、细粒度 Flag 同步、MMA intrinsic、L0 分块、负载均衡 |
| CV 融合型 | [Flash Attention](references/best-practices/flash_attn_optimize.md) | num_stages 流水线、批量 Softmax、Cross-core Semaphore、数据布局优化；**多 shape 适配**（BSND 免转置、Sq==1 decode 窄块、加性 mask 屏蔽变长 Skv） |

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
