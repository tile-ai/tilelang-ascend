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

#### 1.1 性能数字来源核实（强制）⭐

> **背景**：perf-tuner 曾在 Stage 3 报告中把 skill 文档里的**另一个算子案例的历史数据**当作当前 kernel 的 baseline，导致 Orchestrator 误判"已达标"。本地 bench 与官方 cann-bench 存在系统性偏差（element-wise 算子偏差 +58%），不能混用。

**报告任何性能数字时必须标注来源**：

| 来源类型 | 标注格式 | 可信度 |
|---------|---------|--------|
| **当前 kernel 本地 bench 实测** | `[本地 bench, {timestamp}]` | 低（与官方偏差大） |
| **当前 kernel 官方 cann-bench 上传结果** | `[官方 cann-bench, job_id={xxx}]` | 高（达标判断基准） |
| **skill 文档历史数据**（另一算子案例） | `[skill 文档历史, {skill_path}]` | ❌ **禁止当作当前 kernel baseline** |

**强制规则**：
1. perf-tuner 报告 baseline speedup 时**必须**标注数字来源（以上三种之一）
2. **禁止用 skill 文档历史数据当作当前 kernel 的 baseline**——skill 文档里的 mish 0.6641 是另一案例的数据，不是当前 kernel 实测
3. Orchestrator 在采纳任何性能数字前**必须**核实来源（读 bench 脚本输出或官方 `results.json`，不能凭 perf-tuner 文字报告）
4. **达标判断必须以官方 cann-bench 测评结果为准**——本地 bench 仅用于验证优化方向有效（相对提升 > 3%），不用于判定"目标未达"或"已达标"
5. 当本地 bench 显示 speedup < 0.5 时，官方可能在 0.65-0.80 范围（element-wise 算子偏差 +58%），**不要基于本地数据过早中止**

**反面案例**：
- mish Stage 3 iter1 perf-tuner 报告 "Official cann-bench mean speedup was 0.6641"——实际是 skill 文档中另一 mish 案例的历史数据，当前 kernel 本地 bench 实际是 0.0159，Orchestrator 采信后误判"已达标"
- 正确做法：perf-tuner 应报告 `[本地 bench, 2026-08-07] speedup=0.0159`，Orchestrator 据此判断"未达标，需继续优化或上传官方确认"

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

**Part B `[ORDER-PLAN]`**：分析依赖关系，排出实施顺序链。依赖分析三条规则：
1. **布局依赖**：改变 layout 的优化排在依赖此 layout 的优化之前
2. **数量依赖**：涉及预算的优化排在改变 buffer 数量的优化之后
3. **配置依赖**：涉及 pass_configs 的优化在相关功能实施后才改动

```
[ORDER-PLAN] 实施顺序：
1. [#N] [名称] — 前置依赖: [无] — 理由: [...]
2. [#M] [名称] — 前置依赖: [#N] — 理由: [...]
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

### Step 5.5: element-wise 算子 host 侧 tiling 优化决策树 ⭐

> **背景**：element-wise 算子（如 mish/sigmoid/relu）的 kernel 时间通常已接近 baseline（大 shape 0.92-0.96x），性能瓶颈在 **host 侧 adapter 的 tiling 选择**导致的 `num_blocks` 过多。mish 官方 speedup 从 0.5932 提升到 0.7168（达标），**+20.5% 全部来自 host 侧优化，不是 kernel 优化**。

**决策树**（按输入 shape 类别逐项检查）：

| 输入 shape 类别 | 问题 | 优化方案 | 预期收益 |
|---------------|------|---------|---------|
| **1D shape**（M≤2，含质数如 `[1000003]`） | `block_N` cap 默认 512 → num_blocks 数千（如 1954） | **`block_N` cap 提到 8192**（rows_per_vec=1 时单 buffer 8192×4B=32KB < 192KB UB） | num_blocks 从数千降到数百，speedup +650%（mish case 12: 0.076→0.570） |
| **ND shape 且最后一维小**（如 `[11,13,17,67,67]` N=67） | 固定 "merge 除最后维度外到 M" → M 巨大 N 极小 → num_blocks 暴增 | **smart-flatten**：搜索所有 split_idx，选 `num_blocks` 最小的 (M, N) 切分 | num_blocks -73% → kernel -69%（mish case 13: 0.220→0.714） |
| **ND shape 已良好 tile** | 固定切分可能不是最优 | smart-flatten 在 `num_blocks` 平局时优先选更大 split_idx（接近原逻辑，避免回归） | 无回归，部分 case bonus +36%（mish case 18） |

**零拷贝前提**（必须满足）：
- 输入 contiguous 时 `reshape` 只改 stride/shape metadata，**不触发物理拷贝**（cann-bench 默认 contiguous）
- 非 contiguous 时 `.contiguous()` 兜底（会触发拷贝，应避免）

**判定指标**：
- `num_blocks = m_num * n_num = ceil(M/block_M) * ceil(N/block_N)` 是 compute-bound element-wise op 的 kernel 时间可靠代理
- 实测 mish case 13：num_blocks -73.1% → kernel -69.2%（线性相关）

**adapter 实现模板**（参考 `custom/mish/Mish/cann_bench/mish.py`）：

```python
def _select_tiling(tl_dtype, M, N):
    # 1D shape（M<=2）：block_N cap 提到 8192
    if M <= 2:
        max_bn = min(N, 8192)
    else:
        max_bn = min(N, 512)
    # 搜索 block_N in {128, 256, 512, 1024, ...} up to max_bn
    # block_M 由 UB 预算反推：block_M = (2 * effective_budget) // block_N
    # 选 (num_iters, -block_N) 最小的组合

def _estimate_num_blocks(tl_dtype, M, N):
    """用于 ND smart-flatten 选最优切分点"""
    block_M, block_N = _select_tiling(tl_dtype, M, N)
    return ceil(M/block_M) * ceil(N/block_N)

def mish(x):
    if x.ndim <= 1:
        # 1D：近平方 reshape 启用 VEC_NUM=2
        ...
    else:
        # ND：smart-flatten 搜索所有 split_idx
        for split_idx in range(len(dims) - 1):
            M = prod(dims[:split_idx+1]); N = total // M
            nb = _estimate_num_blocks(tl_dtype, M, N)
            # 选 nb 最小的切分
```

**反模式**（禁止）：
- ❌ host 侧 `F.pad` 补齐到整除 shape（mish iter2 测试：pad 增 20us device copy，净退化）
- ❌ kernel 内 stride indexing 处理 ND（需 kernel 重写 + lowering 改动，smart-flatten 零拷贝达到同样效果）

---

## 优化记录

保存在 `examples/{op_name}/perf_tuning/`：
- `baseline.json` - 基线性能
- `optimization_log.md` - 优化记录
- `final_report.md` - 最终报告
