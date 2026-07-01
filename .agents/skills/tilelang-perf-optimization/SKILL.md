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
- **同步原语**：[sync-primitives-guide.md](references/sync-primitives-guide.md)
- **API 用法**：[tilelang-api-best-practices](../tilelang-custom-skill/tilelang-api-best-practices/SKILL.md)
- **编程模式**：[tilelang-expert-to-developer](../tilelang-custom-skill/tilelang-expert-to-developer/SKILL.md)

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

### Step 3: 识别优化点（强制，禁止与 Step 4 合并）

根据算子类型阅读 `optimization-guide.md` 对应章节 + `performance-antipatterns.md`，在 `optimization_log.md` 中输出：

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

| 算子类型 | 文档 |
|---------|------|
| Vector 型 | [RoPE 优化](references/best-practices/rope-developer-mode.md) |
| Cube 型 | [GEMM Intrinsic](references/best-practices/gemm_intrinsic_optimize.md) |
| CV 融合型 | [Flash Attention](references/best-practices/flash_attn_optimize.md) |

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
