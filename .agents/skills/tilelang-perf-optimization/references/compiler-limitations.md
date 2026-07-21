# TileLang 编译器已知限制清单

> **用途**：在优化前快速检查计划使用的特性是否受编译器支持，避免无效尝试。
> 
> **更新频率**：每次遇到编译器限制时追加记录，标注发现日期和算子名称。

---

## 使用方式

1. 在 Step 0（瓶颈预判）或 Step 3（识别优化点）时，对照本清单检查计划使用的优化特性
2. 如果计划使用的特性在"已知限制"表中，直接跳过或选择替代方案
3. 如果遇到新的编译器限制，追加到本文件末尾

---

## 已知限制

### 1. AUTO_SYNC=False + V pipe 指令队列排空

| 项目 | 内容 |
|------|------|
| **特性** | `TL_ASCEND_AUTO_SYNC: False` |
| **限制** | `barrier_all()` 无法排空 V pipe 异步指令队列。当 `n_num > 1`（H 维度循环次数 > 1）时，V pipe 队列中残留未完成的指令，后续 scalar 操作（reduce_sum、rsqrt 等）读取到旧值，导致精度失败 |
| **触发条件** | Pass 1 的 V pipe 循环结束后，立即执行 reduce_sum/rsqrt 等 scalar 操作 |
| **表现** | 精度错误量级与 n_num 正相关（n_num=1 通过，n_num=32 误差 3.9，n_num=16 误差 3936） |
| **发现日期** | 2026-07-17 |
| **发现算子** | AddRmsNormDynamicQuant (R6) |
| **替代方案** | 保持 `AUTO_SYNC=True`，使用交替 buffer（hw_a/hw_b）消除 RAW 依赖，为未来编译器支持做准备 |
| **参考** | optimization-guide.md §2.2.0 同步模式决策表 |

### 2. Fixed Core 模式 + 空循环 range 约束

| 项目 | 内容 |
|------|------|
| **特性** | Fixed Core（按物理核数 launch，内层循环处理多个 block） |
| **限制** | 当 `m_num < CORE_NUM` 时，部分 core 的循环范围为空（如 `T.serial(16, 17)` 但 `if block_idx < 16` 永远为 False），TVM `StmtSimplifier` 无法处理 `block_idx` 的 range 约束冲突，报 `InternalError: Trying to update var with a different maximum value` |
| **触发条件** | `m_num < CORE_NUM`（如 M=256, block_M=32 → m_num=8 < 24 cores） |
| **表现** | JIT 编译阶段 InternalError，无法生成 kernel |
| **发现日期** | 2026-07-17 |
| **发现算子** | AddRmsNormDynamicQuant (R6) |
| **尝试的修复** | 将 `single_core_load` 作为 kernel 参数传入 → 仍失败；使用 `T.min(m_num, CORE_NUM)` 动态 launch → 仍失败 |
| **替代方案** | 使用标准 `m_num` launch，不做 Fixed Core |
| **参考** | optimization-guide.md §2.9 Fixed Core 模式 |

### 3. mul_add_dst 在 bandwidth-bound 算子上的收益

| 项目 | 内容 |
|------|------|
| **特性** | `T.tile.mul_add_dst(dst, src0, src1)` 融合乘加指令 |
| **限制** | 在 GM bandwidth-bound 算子上无性能收益（±1% 噪声范围）。瓶颈在 GM 带宽而非 V pipe 指令数时，减少 1 条 V pipe 指令不影响总耗时 |
| **触发条件** | 算子当前耗时 / 理论最优 > 3x（bandwidth-bound） |
| **表现** | 性能变化在测量噪声范围内，无统计显著差异 |
| **发现日期** | 2026-07-15 |
| **发现算子** | AddRmsNormDynamicQuant (R5) |
| **替代方案** | 优先优化 GM 带宽（减少 pass 数、readback），指令融合留到 compute-bound 场景 |
| **参考** | performance-antipatterns.md §基础指令拼接未融合 |

### 4. 双 kernel 分发开销对小 shape 的影响

| 项目 | 内容 |
|------|------|
| **特性** | 双 kernel / 混合 kernel（Python wrapper 按条件分发到不同 kernel） |
| **限制** | `.item()` host 同步开销约 5~15μs（NPU→CPU 数据传输），当 kernel 耗时 < 50μs 时，分发开销占比 > 10%，导致小 shape 性能退化 |
| **触发条件** | kernel 耗时 < 50μs（通常 M < 1024 且 H < 1024） |
| **表现** | 小 shape 用例退化 15~47%（Case 17: 4.7μs → 8.9μs） |
| **发现日期** | 2026-07-15 |
| **发现算子** | AddRmsNormDynamicQuant (R4) |
| **替代方案** | 混合策略：小 shape 用单 kernel，大 shape 用双 kernel |
| **参考** | SKILL.md Step 3.5.4 分发开销评估 |

### 5. AUTO_SYNC=True 下连续 tile 指令的 V pipe RAW hazard

| 项目 | 内容 |
|------|------|
| **特性** | `AUTO_SYNC=True` 下连续 tile 指令的 dst→src 链（如 `mul(hw, h, gamma)` → `abs(hw, hw)`） |
| **限制** | `AUTO_SYNC=True` 时编译器自动插入 `PipeBarrier<PIPE_V>`，V pipe 串行化，大多数 in-place 操作安全。但特定连续 tile 指令链中，后一条指令的 src 是前一条的 dst 时，V pipe 流水线可能读到旧值 |
| **触发条件** | 连续 tile 指令形成 RAW 依赖链，且中间 buffer 被复用 |
| **表现** | 计算结果不确定，精度测试偶发失败 |
| **发现日期** | 2026-07-17 |
| **发现算子** | AddRmsNormDynamicQuant (R6) |
| **替代方案** | 使用交替 buffer（hw_a/hw_b），让连续 tile 指令的 dst 和 src 使用不同 buffer。详见 optimization-guide.md §2.2.2 |
| **参考** | optimization-guide.md §2.2.2 交替 buffer 消除 V pipe RAW hazard |

---

## 安全特性清单（已验证可用）

以下特性在 AddRmsNormDynamicQuant 优化中已验证可用，无编译器限制：

| 特性 | 验证轮次 | 备注 |
|------|---------|------|
| `TL_ASCEND_AUTO_SYNC: True` | R0-R7 | 所有轮次均使用 |
| `TL_ASCEND_MEMORY_PLANNING: True` | R0-R7 | 自动复用已死亡 buffer |
| Double Buffer（三阶段流水） | R2-R7 | prefetch → main → epilogue |
| 交替 buffer（hw_a/hw_b） | R6-R7 | 消除 V pipe RAW 依赖 |
| `T.tile.broadcast` | R2-R7 | 向量化广播替代 scalar 循环 |
| `T.tile.mul_add_dst` | R5-R7 | 代码质量提升，性能中性 |
| 自适应 block_N（H<256→H） | R3-R7 | 消除尾块处理 |
| 混合 kernel 分发 | R7 | 按 M 维度动态选择 kernel |

---

## 追加记录模板

遇到新的编译器限制时，按以下格式追加：

```markdown
### N. {特性名称}

| 项目 | 内容 |
|------|------|
| **特性** | {特性描述} |
| **限制** | {限制描述} |
| **触发条件** | {触发条件} |
| **表现** | {错误表现} |
| **发现日期** | {YYYY-MM-DD} |
| **发现算子** | {算子名称} ({轮次}) |
| **尝试的修复** | {尝试过的修复方案} |
| **替代方案** | {替代方案} |
| **参考** | {相关文档} |
```
