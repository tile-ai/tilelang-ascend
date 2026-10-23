# Reduction Kernel 使用流程

> 迁移来源：Ascend950 同路径文档。保留可复用原文结构与门禁，仅改写 A2/A3 平台、后端、API、路径和证据范围。本页是设计/源码参考；本次未进行 NPU 编译、精度或性能验证，不继承950模板的验证状态。

## 选型

新增 ReduceSum/Max、LayerNorm、RMSNorm、Softmax 时使用。

## TileLang A2/A3 实现

Python factory 接受静态 reduction width、dtype、threads、tile 与 stages，返回 T.prim_func。动态 rows 仅在当前后端和入口已有支持时表达；核数按 min(可用 launch 容量, 独立任务数) 选择，并区分 cid/vid 与行 owner，其中硬件核数由查询结果显式传入。先复用现有 kernel，再添加 shape specialization。

基础 reduction 代码参考 [当前 RMSNorm](../../../../../../examples/normalization/rms_norm.py) 与 [Online Softmax](../../../../../../examples/softmax/example_online_softmax.py)：GM 数据通过 T.copy 进入 UB，按当前 API 用 fp32 UB 工作区与 T.reduce_max/T.reduce_sum、T.tile 运算；任务遵循当前 T.Kernel 的 cid/vid 映射，单个输出只有一个 owner。当前 A2/A3 接口不沿用源仓 T.SimtVF、alloc_reducer/finalize_reducer 或一维网格限制。

## 精度门禁

测试文件与 kernel 同步提交，reference 使用 torch fp32 或更高精度稳定表达式。

低精度输入的 max、sum、平方和、方差、rsqrt、exp 与 online 状态保持 fp32。尾部 max lane 填负无穷、sum lane 填 0。覆盖 tile±1、极长归约轴、抵消、极值、NaN/Inf 契约以及 forward/backward 的公开范围。

## 性能门禁

定向精度测试后运行相关完整测试套，并记录每个 dispatch 分界。

目标 A2/A3 后端的定向精度测试全通过后，比较端到端 latency、有效 GM bytes、Vector/MTE 时间、核利用率、UB 字节与 stage。多 kernel 方案必须计入 workspace 和全部 launch，不能只报告局部 kernel。

## 专题入口

- [euclidean_norm_group_block_split_r](euclidean_norm_group_block_split_r.md)
- [euclidean_norm_interleaved_contiguous_tail_a](euclidean_norm_interleaved_contiguous_tail_a.md)
- [euclidean_norm_interleaved_contiguous_tailr](euclidean_norm_interleaved_contiguous_tailr.md)
- [fused_attention_online_softmax_design](fused_attention_online_softmax_design.md)
- [softmax_v2_ar_full_load](softmax_v2_ar_full_load.md)
- [softmax_v2_ar_recompute](softmax_v2_ar_recompute.md)
- [softmax_v2_ar_small_r](softmax_v2_ar_small_r.md)
- [softmax_v2_ara_full_load](softmax_v2_ara_full_load.md)
- [softmax_v2_ara_online](softmax_v2_ara_online.md)
- [softmax_v2_ara_recompute](softmax_v2_ara_recompute.md)
- [state_resident_design](state_resident_design.md)

## 实现状态

本文件保留算法流程和验收门禁，状态为 `DESIGN_ONLY`；链接的 A2/A3 示例只提供局部结构依据，不证明本策略已有端到端 kernel。源仓 dav310 Python 模板没有复制到本库。
