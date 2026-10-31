# Softmax AR Recompute

## 选型

整行不能驻留 UB，但输出每个元素，允许用额外读取换容量。

## TileLang 实现

第一遍分 tile 求全局 max；第二遍求 sum(exp(x-max))；第三遍重新读取并输出。每遍复用一个 fp32 tile，状态 max/sum 常驻。

基础 reduction 代码参考 [当前 RMSNorm](../../../../../../examples/normalization/rms_norm.py) 与 [Online Softmax](../../../../../../examples/softmax/example_online_softmax.py)：GM 数据通过 T.copy 进入 UB，按当前 API 用 fp32 UB 工作区与 T.reduce_max/T.reduce_sum、T.tile 运算；任务遵循当前 T.Kernel 的 cid/vid 映射，单个输出只有一个 owner。任务维度与归约输出形状须遵循当前 T.Kernel 和归约 API 的合同。

## 精度门禁

三遍的尾部幺元一致；第二、三遍 exp 表达式必须相同。

低精度输入的 max、sum、平方和、方差、rsqrt、exp 与 online 状态保持 fp32。尾部 max lane 填负无穷、sum lane 填 0。覆盖 tile±1、极长归约轴、抵消、极值、NaN/Inf 契约以及 forward/backward 的公开范围。

## 性能门禁

与两遍 online 统计方案比较 GM 读取和 exp 成本；recompute 作为稳健 fallback。

目标后端的定向精度测试全通过后，比较端到端 latency、有效 GM bytes、Vector/MTE 时间、核利用率、UB 字节与 stage。多 kernel 方案必须计入 workspace 和全部 launch，不能只报告局部 kernel。

## 实现状态

状态为 `DESIGN_ONLY`；链接示例只提供局部结构依据，不证明本策略已有端到端 kernel。实现后须完成上述精度与性能门禁。
