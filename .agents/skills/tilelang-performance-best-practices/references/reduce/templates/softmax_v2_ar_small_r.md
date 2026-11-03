# Softmax 小归约轴

## 选型

R 较小，一个 task 可批处理多行以提高向量利用率。

## TileLang 实现

UB 工作区形状为 blk_m×padded_r，T.reduce_max/sum 沿最后维；blk_m、当前线程/vid 分工与 padded_r 由 factory 静态选择。

基础 reduction 代码参考 [当前 RMSNorm](../../../../../../examples/normalization/rms_norm.py) 与 [Online Softmax](../../../../../../examples/softmax/example_online_softmax.py)：GM 数据通过 T.copy 进入 UB，按当前 API 用 fp32 UB 工作区与 T.reduce_max/T.reduce_sum、T.tile 运算；任务遵循当前 T.Kernel 的 cid/vid 映射，单个输出只有一个 owner。任务维度与归约输出形状须遵循当前 T.Kernel 和归约 API 的合同。

## 精度门禁

最后一批不足 blk_m 时只写有效行，padding 行不得访问 GM。

低精度输入的 max、sum、平方和、方差、rsqrt、exp 与 online 状态保持 fp32。尾部 max lane 填负无穷、sum lane 填 0。覆盖 tile±1、极长归约轴、抵消、极值、NaN/Inf 契约以及 forward/backward 的公开范围。

## 性能门禁

搜索 blk_m 与当前线程分工，重点测 launch 开销和小 copy 合并。

目标后端的定向精度测试全通过后，比较端到端 latency、有效 GM bytes、Vector/MTE 时间、核利用率、UB 字节与 stage。多 kernel 方案必须计入 workspace 和全部 launch，不能只报告局部 kernel。

## 实现状态

状态为 `DESIGN_ONLY`；链接示例只提供局部结构依据，不证明本策略已有端到端 kernel。实现后须完成上述精度与性能门禁。
