# A2/A3 Reduction、Norm 与 Softmax

> 迁移来源：Ascend950 同路径文档。保留可复用原文结构与门禁，仅改写 A2/A3 平台、后端、API、路径和证据范围。本页是设计/源码参考；本次未进行 NPU 编译、精度或性能验证，不继承950模板的验证状态。

## 选型

根据归约轴是否能驻留 UB、输出行数是否足以占满向量核、是否需要输出每个元素，选择 full-load、recompute、online 或两阶段 split-axis。

## TileLang A2/A3 实现

先实现每个输出由唯一 owner 持有的 fp32 基线；一个核可处理多个输出，避免把 owner 与物理核一一绑定。Norm 常驻 weight；softmax 使用稳定 max-subtract。只有行数太少且轴极长时，才用 Kernel A 写 fp32 partial、Kernel B 确定顺序合并。

基础 reduction 代码参考 [当前 RMSNorm](../../../../../examples/normalization/rms_norm.py) 与 [Online Softmax](../../../../../examples/softmax/example_online_softmax.py)：GM 数据通过 T.copy 进入 UB，按当前 API 用 fp32 UB 工作区与 T.reduce_max/T.reduce_sum、T.tile 运算；任务遵循当前 T.Kernel 的 cid/vid 映射，单个输出只有一个 owner。当前 A2/A3 接口不沿用源仓 T.SimtVF、alloc_reducer/finalize_reducer 或一维网格限制。

短归约可在一个合法 Vector tile 内完成时，可用当前 T.reduce_sum 或已验证的 tile 归约路径，并把相邻输出批量搬入 UB；必须 A/B 验证运算顺序、临时量和收益。不直接复制源仓 `S.vcadd`/`S.vdupv` 或 SIMT reducer API。若 partial 只供下游归约，先评估在 producer 内完成 split-axis 合并并特化 `n_splits==1`，同时保留确定性与原子语义。匹配该结构时读取 [批量短归约](batched_short_reduction.md)。

固定小矩阵的归一化可让当前矩阵及反向梯度跨整轮计算常驻 UB，仅保存反向所需快照；寄存器驻留只有在当前 lowering 证明后才作为额外候选。每组先求一次倒数再乘各元素，须验证局部布局、同步、误差、临时量和展开代码量。匹配该结构时读取 [固定小状态](fixed_small_state.md)。

若每行先生成归约标量，再使用该标量逐元素写回整行，读取 [行归约与逐元素写回](rowwise_reduce_epilogue.md)。

## 精度门禁

Softmax 额外检查每行和、重复最大值、全负无穷和正无穷语义；Norm 的 eps 加在 fp32 统计量上。

低精度输入的 max、sum、平方和、方差、rsqrt、exp 与 online 状态保持 fp32。尾部 max lane 填负无穷、sum lane 填 0。覆盖 tile±1、极长归约轴、抵消、极值、NaN/Inf 契约以及 forward/backward 的公开范围。

## 性能门禁

依次测 tile、threads、resident weight、1/2 stage 和 split-axis；保持正确 fallback。

归约计算较轻、非归约连续轴较宽且小 tile 的调度或 DMA 开销明显时，按 [宽输出轴 Tiling](wide_output_tiling.md) 合并连续输出元素。

目标 A2/A3 后端的定向精度测试全通过后，比较端到端 latency、有效 GM bytes、Vector/MTE 时间、核利用率、UB 字节与 stage。多 kernel 方案必须计入 workspace 和全部 launch，不能只报告局部 kernel。

## A2/A3 专属细节与既有纠错

完整保留上一版 A2/A3 内容于 [A2/A3 补充](a2a3_guide.md)，包括当前接口、边界、历史记录及验证限制。

## 专题入口

- [宽输出轴](wide_output_tiling.md)
- [批量短归约](batched_short_reduction.md)
- [固定小状态](fixed_small_state.md)
- [行归约写回](rowwise_reduce_epilogue.md)
- [A2/A3 归约遍数融合](vector_reduce_pass_fusion.md)
- [策略与接入流程](templates/usage_guide.md)
