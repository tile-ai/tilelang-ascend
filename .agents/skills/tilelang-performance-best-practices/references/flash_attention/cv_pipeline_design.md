# Flash Attention C/V 流水与所有权

## 适用证据

生成代码同时含 Cube 和 Vector 工作，分数、概率、partial O 通过 workspace
交接，已有时间线/统计或源码显示生产消费等待。瓶颈可能是 Cube、Vector、GM
workspace 流量、slot 复用递推或 scalar 发射，不能仅凭有两次 GEMM 判为计算瓶颈。

当前示例使用普通 GM workspace 的交接形式；优化时须按实际数据通路、
存储层级与指令约束设计。参考 [当前 FA 索引](guide.md) 选择具体实现。

## 四个关键 ownership 边界

| 区域 | 生命周期 |
|---|---|
| Q 常驻 L1 | MTE2 写入→多轮 MTE1 读取→最后一次读取后归还 |
| Vector 输入 UB slot | MTE2 写入→V 最后读取→才允许下次 MTE2 覆盖 |
| probability/output UB store slot | V 生产→MTE3 完成读取/写回→才允许 V 重用 |
| 核间 workspace slot | 生产核写→消费核读/计算→消费完成通知生产者重用 |

store 前的 barrier 只能约束此前生产完成，不能保护 store 后的下一次覆盖。
通过 `annotate_address` 重叠的两个名字仍共享所有权。事件要覆盖首个 token、每轮
Set/Wait、最后归还和未使用槽 drain；单次精度通过不证明残留 token 安全。

local event ID 的核对单位是有向 `(src_pipe,dst_pipe)` 对。不同方向可出现同一数字，
但须分别审计活跃 token，不能根据 `READY/FREE` 名字推断它们天然不冲突。
新 Expert 源遵循 [模式指南](../../../tilelang-custom-skill/tilelang-programming-model-guide/SKILL.md)
的 LOCK DSL；见 [v3 源](../../../../../examples/flash_attention/fa_opt/expert_v3/kernel.tl)。

## num_stages、槽深与 cross_interval

`T.Pipelined` 可作为 Developer/混合模式的候选；手动 Expert ring 则显式管理
ownership。两者都需计算真实同时活跃的 L1/L0/UB 和 workspace，而不是把 `num_stages`
与某个 buffer 第一维机械等同。

- 从小的可行槽深开始比较；旧 `num_stages=14`、`block_M=128` 等只是历史案例值。
- 浅流水的候选起点为 `num_stages=2`，不超过可利用的循环次数；短循环时要判断
  pipeline 启停成本，不能把“至少2”强加给只需一个 stage 的合法实现。
- 增大槽深能解除部分复用等待，也会扩大 live state；瓶颈在必要计算时不会消掉工作量。
- `cross_interval=2` 是旧候选，不是任意同步可以每两次才通知。只有批内消费者不需要
  中间结果、空间容纳整批、终批也会通知时才可合并同步。
- 不机械嵌套 `T.Pipelined`；涉及核内手动 buffering 时依据当前
  [pipeline 教程](../../../../../docs/tutorials/t_pipelined.md) 检查组合可表达性。

## Workspace ring 与跨任务流水

按 active core × live stages 分配 ring，可避免按全部逻辑任务分配 workspace。
需要证明 wraparound、终批、不同 batch/head/query 任务切换都完成双向所有权。
不在每个 query 边界排空可减少启动成本，但运行统计量与 O 累加必须归属正确 query。
v3 的常驻 FP32 O 和独立输入/输出槽是这种路线的具体案例。

固定核映射同时约束 Q 复用、ring 下标和 workspace 规模。活动核数从选定平台取得，
任务数少时减少空核；余数分配应平衡。详见 [Cube 任务映射](../matmul/task_mapping_design.md)。

## 如何解释已有诊断结果

Cube/Vector busy 时间有重叠，不得直接相加推算总耗时。去掉某路径只快一点，也不
等于它原本成本很小。v2/v3 历史消融指出过搬运竞争和 Vector 暴露成本，但这些
删除计算的版本不正确，仅为瓶颈假设提供线索。

Vector 侧可考虑批量 softmax、广播复用、减少非必要临时状态或把必要计算放入依赖
间隔。地址错位、原地/非原地和短指令序列会影响当前 backend 的调度与冲突，不能
只按指令数多少决定。计划中应列出数值、地址和同步三类验证；具体候选仍需定向精度与性能验证。
