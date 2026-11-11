# 离散重排 fallback 的多阶段连续化

## 适用条件与先后顺序

通用 permutation 的共同连续 record 很短、导致大量离散 GM 访问，且能分解成有限个相邻轴组 rotation 时，可评估每 stage 使用块 DMA + UB-local transpose。full reversal 只是一个例子，不是唯一启用条件。额外 stage 增加全量流量与 launch，必须比较完整路径。

**先证明 stage kernel，再写 planner、再改 dispatch。** 原 path1 对单 stage 的测试不能证明其在 planner 产生的多 batch、双尾块和 `M/N<tile` 下安全。候选分析先记录 `[KERNEL-REUSE-PRECHECK]`；接入 dispatch 前用如下具体维度复核：

```text
[KERNEL-REUSE-AUDIT]
stage table: [(batch, M, N, dtype, block_M, block_N), ...]
tile: true/false + 各stage资源/静态指令对齐推导
core: true/false + launch容量、逻辑task、每核load与vid owner
tail: true/false + 输入/输出有效BufferRegion
multibatch: true/false + 两侧copy不跨batch证明
specialization: true/false + dtype静态UB变体与合法lowering
```

任一项 false/unknown，方案必须明确“先修复/新建 stage kernel”，不能直接复用接 planner。通用候选状态由调优流程管理，本页保留布局复用所需的具体证明。

## 标量 GM 起点为何危险

对于 `x[batch,M,N]`，`T.copy(x[b,m_start,n_start], full_ub)` 没有表达当前 batch 的二维有效边界。若 M=11、N=13、tile=128×128，一个 batch 143 元素，完整 tile 16384 元素，后续访问会跨 batch。输出若 `T.copy(full_ub,y[b,n_start,m_start])` 同样污染后续 batch。

全 NaN 输入可能隐藏错位，有限值+稀疏 NaN 更容易暴露。减少 core 数或修正 rotation 轴序不能修复非法单 task copy。

输入、输出必须同时携带二维有界 region。只有当前 lowering 已验证自动裁剪 tensor 边界时，才使用 block 端点；否则显式计算：

```text
valid_m = min(block_M, M-m_start)
valid_n = min(block_N, N-n_start)
GM x[b, m_start:m_start+valid_m, n_start:n_start+valid_n]
  → UB a[:valid_m,:valid_n]（其余完整tile初始化）
完整静态UB transpose a[block_M,block_N] → b[block_N,block_M]
UB b[:valid_n,:valid_m]
  → GM y[b, n_start:n_start+valid_n, m_start:m_start+valid_m]
```

此处是边界契约，实际切片 rank/pitch/pad 按当前 API 实现，不是未经验证的通用可执行模板。禁止只修输入或输出一侧。

## 每个 stage 独立配置

按本 stage M/N、dtype、scratch 与版本重新选 tile；不能统一固定 128×128。tile 可大于有效尾块，物理 footprint 仍须满足指令与容量约束。需要 cast 的 dtype 使用各自静态 JIT 变体，不在 TIR 条件分支分配可能未定义或动态 shape 的 UB。

```text
m_blocks = ceildiv(M,block_M)
n_blocks = ceildiv(N,block_N)
logical_tasks = batch*m_blocks*n_blocks
core_num = min(logical_tasks, 已确认的该Kernel映射launch容量)
single_core_load = ceildiv(logical_tasks,core_num)
task = cid*single_core_load + local_id
有效task解码：
    n_blk = task % n_blocks
    m_blk = (task//n_blocks) % m_blocks
    batch_id = task//(n_blocks*m_blocks)
```

空域单独处理；有效 task 恰好一次，vid 所有权明确。Fixed Core 摊薄初始化，不改变合法搬运范围。输出 workspace 的生命周期覆盖下一 stage 消费；复用两个大 workspace 时必须等待相应 consumer 完成。

## Planner 与逐步验证

每轮从**更新后** `cur_shape/cur_axes` 计算 batch/M/N，保存 stage 表并在最终断言 axes 等于目标 permutation；不能反复用原始 shape 推导。每 stage 同时记录 input/output shape、stride、buffer、静态 tile 和 workspace bytes。

接入 planner 前独立验证 stage kernel：batch>1 且 M/N 均非整除、M/N 分别/同时小于 tile、输入输出双尾、最大 batch/逻辑任务数、各 dtype 变体。接入后逐 stage 检查唯一编号映射，再跑所有受影响 canonical cases；浮点使用有限值+稀疏 NaN 并严格比较 NaN mask，int64 使用高位非零位模式。

精度失败先停止该优化并定位 stage，不继续叠加其他优化。fallback 通过、其他候选通过或最终 shape 正确都不能当本候选通过；静默回退后的 performance 也不能归给多阶段方案。

比较基线、各 stage 及完整链 latency、额外 GM 流量、workspace 和 launch；无端到端收益保留正确 fallback，记录“不适用/无收益”。
