# A2/A3 Flash Attention 方案指南

用于分析 AIC/AIV 融合 attention 的数据流、容量、online softmax 和调度候选。
不在此技能中采集数据或实施优化。本次迁移核对当前源码与既有实验说明，未重新
上板验证；历史数据不作为当前 kernel 的性能承诺。

## 按瓶颈与形态路由

| 已有证据/需求 | 优先阅读 |
|---|---|
| L1/L0、两次 GEMM 的块大小或 MMA 控制 | [Matmul](../matmul/guide.md) |
| Cube/Vector 等待、workspace ring、num_stages/cross_interval | [C/V 流水与 ownership](cv_pipeline_design.md) |
| Vector softmax、O rescale、归约或广播占用 | 本页递推及实现版本；按需查当前 Vector API |
| 新增 decode、GQA、mask、非整除 shape | 本页调用域审计，先核对 kernel 再规划 dispatch |

## 数据流与不能丢失的递推

```text
Cube:   Q,K → QKᵀ → score workspace
Vector: score → online softmax → probability workspace
Cube:   probability,V → partial O workspace
Vector: partial O + running state → final Output
```

对每个有效 score tile 保持：

```text
m_new = max(m_old, rowmax(score))
r     = exp(m_old - m_new)
l_new = l_old*r + rowsum(exp(score - m_new))
O_new = O_old*r + P@V
final = O/l
```

分批 softmax 或延迟累加可以摊薄同步和搬运，但每份 partial O 必须匹配其对应的
缩放系数。扩大批次时检查首块初始化、最后不足批次、running O 每次 rescale 和
最终归一化。空/全 masked 行、NaN 等特殊输入的语义以算子规格为准，不能用普通
有限输入的测试替代。

## 当前可读的实现及边界

旧路径 `fa_opt/flash_attn_bhsd_expert_h16_d128.py` 已迁移；通过
[fa_opt 索引](../../../../../examples/flash_attention/fa_opt/README.md) 选择具体路线。

| 路线 | 有区别的技术 | 适用/证据边界 |
|---|---|---|
| [Expert v1](../../../../../examples/flash_attention/fa_opt/expert_v1/kernel.py) | 旧手动同步、显式 workspace 路线 | 历史实现，改 shape 前审计本版本源码 |
| [Expert v2](../../../../../examples/flash_attention/fa_opt/expert_v2/README.md) | 普通 FP32 归约、三槽、延迟输出累加；保留共享 KV 的 GQA | D=128，序列长度为1024倍数，query heads可整除KV heads；默认24逻辑核是示例设置 |
| [Expert v3](../../../../../examples/flash_attention/fa_opt/expert_v3/README.md) | O 累加常驻 UB、输入/输出双槽、三槽核间 ring、跨 query 不排空、必要计算填充依赖间隔 | FP16输入输出/FP32中间态，D=128，Q为256倍数，KV为512倍数，非因果MHA/GQA |
| [自动 pipeline H16/D128](../../../../../examples/flash_attention/fa_opt/auto_pipeline/h16_d128.py) | 编译器管理的流水路线 | 依当前代码的 shape/pass 条件适配，不复制 Expert 的全部开关 |

v2 的历史两槽→三槽比较为 2.04%–3.20% 用时下降，来源注明 Atlas A3、编译器
`77a444b2`，原始内部日志未发布。v3 记录 `Ascend910_9392`、CANN 9.1、提交
`4c7267018af02284ff28f1cd090a1492c7985bed`，三个长序列用时下降约6.36%–6.77%。
两条路线的比较对象不同，不能相加或当成 v3 对 v2 的收益。具体参数与验证范围保留
在链接的版本文档中。

v3 的可编辑源是 [kernel.tl](../../../../../examples/flash_attention/fa_opt/expert_v3/kernel.tl)，
`kernel.py` 是生成结果。其显式等待间隔、地址错位和 UnitFlag 调度只在记录环境中
验证，不把它概括成通用延迟公式。本技能只引用这些来源，不触发预处理或修改。

## Shape / layout / dispatch 审计

扩大 kernel 调用域前逐项记录：输入输出 BHSD/BSND 与真实 stride、Q/KV 尾块、
D 与两次 GEMM 的 L0 足迹、GQA/MQA head 映射、causal/additive mask、任务数量和
workspace 下标的编译期假设。

- 原布局直接访问可能避免 `permute().contiguous()` 的真实搬运；仅改变 Python
  view 不证明原 kernel 的物理访问仍合法。
- `Sq==1` decode 窄 tile 可减少无效计算，仍要满足 Cube 最小分形和容量；不能只
  改 dispatch 就复用为大 query 设计的 kernel。
- KV padding 的某个数值不等于 softmax mask。非法 score 列须在 exp 前按语义
  屏蔽；用不同符号和极值输入检查尾块，而非依赖“填零碰巧没贡献”。
- GQA 不要求复制共享 K/V；核内映射必须依据 `q_heads/kv_heads`，并证明整除条件。
- 不满足特化前提时保留正确 fallback，报告具体未覆盖范围。

## 候选的验收证据

方案要求后续执行者验证受影响 shape/dtype、短流水、尾块、重复不同输入启动和
必要的并发场景；对照生成的 scope、地址和事件及同口径性能。预热用于计时稳定，
不能用来忽略第一次错误。只删除计算的消融版本是诊断资料，不能作为正确候选。
