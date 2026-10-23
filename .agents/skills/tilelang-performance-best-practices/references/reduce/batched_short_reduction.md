# 批量短归约与逐记录写回

> 迁移来源：Ascend950 同路径文档。保留可复用原文结构与门禁，仅改写 A2/A3 平台、后端、API、路径和证据范围。本页是设计/源码参考；本次未进行 NPU 编译、精度或性能验证，不继承950模板的验证状态。

适用于单次归约较短、外层记录较多，并在归约后继续逐记录归一化、缩放或其他逐元素 epilogue 的结构。每行先得到归约标量再写回整行时，同时读取 [行归约与逐元素写回](rowwise_reduce_epilogue.md)。

**证据等级：A2/A3 设计参考，待目标 case 复验。** 源文记录该结构已有 Ascend 实现与实测，但没有为本仓 A2/A3 目标提供可复验的硬件、版本、命令和原始结果；本次只迁移通用设计，不继承源文的已实现/已实测等级。

## 优化结构

- 记录按块处理，但 DMA 必须服从实际连续轴：record 连续的输入/输出可整块搬运，split-major 输入通常逐 record 搬连续的 `[split, feature]` 矩形。
- UB 特征轴按当前 A2/A3 指令的 dtype 对齐和 footprint padding；GM 只搬有效 record/feature，短轴归约和写回使用合法有效范围与幺元，标量结果由唯一 owner 写回。
- 向量 partial 沿短特征轴累加，标量 partial 用当前 broadcast 路径汇总；点积可用已验证 T.reduce_sum/T.tile 归约，并核对有效范围和归约顺序。源仓 masked `vcadd`/`vdupv` 不作为本仓 API。
- 显式 stage 维的 UB 用每个 core 的 wave 选择槽位并完整表达 ownership；自动流水 buffer 用当前已支持的 T.Pipelined/规划。源仓 `annotate_manual_multi_buffer`、`annotate_buffer_versions` 不在本仓对应 API 中，不复制这些调用；同一 buffer 不重复混用两种寻址方式。
- 当前 T.Persistent 的任务映射、buffer 槽位和真实地址偏移保持一致，并从生成地址与 event 核对实际重叠；常驻量保持单版本。当前 Persistent 无源仓 `num_stages`/`enable_offset` 参数，流水按独立的已验证机制表达。
- batching、Vector 指令 footprint、DMA 形态和 buffer 生命周期共享同一布局时，先作为一个数据流假设落地，再对可独立拆分的部分做 A/B；尾块只访问有效 GM，允许无副作用地计算完整 UB footprint。
- `rows_per_tile` 与 `num_stages` 是独立搜索轴：批量粒度先按连续布局、UB 预算、DMA 粒度和任务 wave 选择几何候选，再对可行 tile 实测流水级数；不能用单记录上的 stage 搜索代替多记录候选。
- producer 内合并 partial 与 `n_splits==1` 连续快路是独立候选，只在数据依赖和目标 case 匹配时评估。

复用前须核对批量粒度、归约顺序、partial 数、搬运形态、buffer 版本和 launcher 参数；上下文不同的负结果只否决当前组合。

## 源文结构组合（A2/A3 待验证）

| 路径 | 待在 A2/A3 验证的物理结构 |
|---|---|
| 归约后归一化 | 多记录批处理和按目标指令对齐的 UB；split-major 输入按 record 搬连续矩形，显式 stage 输入配合当前后端支持的输出版本化；broadcast 标量与向量 partial 累加后批量写回 |
| 归约后逐元素 epilogue | 连续记录使用矩形 UB 和批量 DMA；每条记录独立生成 fp32 归约标量，随后在 tile 内逐记录写回，尾 tile 只访问有效 GM |
| 带参数梯度归约 | 输入输出批量 DMA，流式 buffer 多版本；有效范围归约/broadcast 完成点积与回填，标量结果只由有效 lane 写回 |

## A2/A3 接口依据

任务接口见 [persistent.py](../../../../../tilelang/language/persistent.py)，签名为
`Persistent(domain, wave_size, index, group_size=8)`；其 group_size 还有整除约束。
流水入口见 [pipeline.py](../../../../../tilelang/language/pipeline.py)，实际使用前仍需核对
编译后的 buffer 版本与同步。两者不能混成源仓带 `num_stages` 的 Persistent 调用。
