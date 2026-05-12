---
name: tilelang-perf-optimization
description: TileLang 算子性能调优。提供性能数据采集、瓶颈诊断、优化实施、效果验证能力。触发：算子精度通过后需要优化性能时，或性能不及预期时。
---

# TileLang 性能优化

## 工作流程

```
Step 1: 精度校验（强制前置）→ Step 2: 性能数据采集 → Step 3: 算子类型判断
         ↓
Step 4: 优化实施 → Step 5: 精度再验证 → Step 6: 效果验证
```

## 核心约束

- **精度优先**：精度未通过禁止性能优化；每次优化后必须重新验证精度
- **迭代验证**：每次只修改一个参数/配置，修改后立即验证，性能回退则回退
- **记录可追溯**：中间文件保存在 `examples/{op_name}/perf_tuning/` 目录

## 执行步骤

### Step 1: 精度校验（强制前置）

- 运行命令：`python examples/{op_name}/<script_name>.py`
- **`<script_name>` 获取**：在 `examples/{op_name}/` 目录下查找包含 `@tilelang.jit` 装饰器的 Python 脚本。若存在多个，询问用户确认使用哪一个，后续步骤自动复用该选择。
- 算子必须包含与参考实现的精度对比逻辑
- 精度未通过 → 禁止进入后续步骤

### Step 2: 性能数据采集

```bash
msprof op --kernel-name="main_kernel" --output=./msprof_output ./examples/{op_name}/<script_name>.py
```

### Step 3: 算子类型判断

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

### Step 4: 优化实施

根据算子类型选择优化手段（详见 [optimization-guide](references/optimization-guide.md)）：

| 优化方向 | 说明 | 典型手段 |
|---------|------|---------|
| pass_configs 调优 | 调整编译器 pass 行为 | 关闭自动同步、关闭内存规划 |
| 核内优化 | 提升单核内指令并行度 | Double Buffer、L1 常驻、指令向量化、Split-K pipelined GEMM |
| 核间优化 | 优化 Cube/Vector 核间协作 | num_stages 调优、同步优化、Fixed Core 模式 |
| 流水线优化 | 计算与访存重叠 | T.Pipelined（核内/核间流水）、T.Persistent（数据块调度） |
| Fixed Core | 按物理核数 launch，减少冗余初始化和显存膨胀 | `T.Kernel(core_num, is_npu=True)`、Workspace 按物理核分配 |
| 指令融合 | 减少指令下发次数 | AXPY 融合指令、broadcast 向量化 |
| 稀疏访存优化 | 离散数据高效搬运 | 双 vector 核访存、Gather + 连续搬出、异步拷贝 |

**编程模式选择**：

优先使用 **Developer 模式**（自动内存规划、自动同步、编译器自动分离 Cube/Vector），参考 [tilelang-expert-to-developer](../tilelang-custom-skill/tilelang-expert-to-developer/SKILL.md)。

如无法满足性能要求，再使用 **Expert 模式**手动控制（显式指定 L1/UB/L0 层级、手动同步、细粒度调度），参考同一文档。

- 调用 `/tilelang-api-best-practices` 查阅相关 API 用法

### Step 5: 精度再验证

- 运行命令：`python examples/{op_name}/<script_name>.py`（`<script_name>` 为 `examples/{op_name}/` 目录下的 Python 脚本文件名）
- 精度失败时：在保持已实施优化手段前提下调试精度，不能为修复精度撤销优化

**调试手段**：

| 方法 | 用途 | 示例 |
|------|------|------|
| `T.printf(fmt, *args)` | 设备端打印中间值 | `T.printf("val=%d\n", val)` |
| `T.dump_tensor(tensor, desc, size, shape_info)` | 转储张量内容到文件 | `T.dump_tensor(acc_o, "acc_o", 128, (128, 128))` |
| 查看生成的 Ascend C 代码 | 分析编译器生成逻辑 | `print(func.get_kernel_source())` |

详细说明参考 [TileLang-Ascend Programming Guide](../../../docs/TileLang-Ascend%20Programming%20Guide.md)。

### Step 6: 效果验证

- 对比基线性能，记录优化结果
- 迭代终止条件：
  - 有明确目标：达到目标 OR 连续 3 次无提升 → 中断并上报用户
  - 无明确目标：做完所有可优化手段后输出最终性能值

**性能目标要求**：
- 用户提出性能需求时，必须明确指定对应的 **shape** 和 **dtype**
- 性能标准可以是：
  - 具体的耗时时间（如 `< 100 us`）
  - 对应 AscendC 接口的调用脚本（用于对比性能）

## pass_configs 参数参考

| 参数 | 默认值 | 调优建议 |
|-----|-------|---------|
| `TL_ASCEND_AUTO_SYNC` | True | 关闭可提升性能，手动插入同步避免冗余 |

## 优化记录

中间文件统一保存在 `examples/{op_name}/perf_tuning/` 目录：
- `baseline.json` - 基线性能数据
- `optimization_log.md` - 优化记录（每轮详情）
- `final_report.md` - 最终优化报告（必须包含验收 shape、dtype 及性能标准对比）

## 与其他 Skill 的协作

| 调用时机 | 调用 Skill | 用途 |
|---------|-----------|------|
| Step 4 优化实施 | `/tilelang-api-best-practices` | 查阅内存、计算、调度 API 用法 |
