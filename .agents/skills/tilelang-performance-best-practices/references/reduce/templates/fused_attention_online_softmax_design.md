# Attention Online Softmax

> 迁移来源：Ascend950 同路径文档。保留可复用原文结构与门禁，仅改写 A2/A3 平台、后端、API、路径和证据范围。本页是设计/源码参考；本次未进行 NPU 编译、精度或性能验证，不继承950模板的验证状态。

## 选型

Softmax 结果立即与 V 相乘，无需输出完整 probability。

## TileLang A2/A3 实现

每个 KV block 计算 tile max/sum，使用 m_new=max(m,m_tile)、l_new=l*exp(m-m_new)+l_tile*exp(m_tile-m_new)，并用相同 alpha 重标定 fp32 O accumulator。实现从 [当前 FA 路线](../../../../../../examples/flash_attention/fa_opt/README.md) 派生，并按各版本限制核对。

基础 reduction 代码参考 [当前 RMSNorm](../../../../../../examples/normalization/rms_norm.py) 与 [Online Softmax](../../../../../../examples/softmax/example_online_softmax.py)：GM 数据通过 T.copy 进入 UB，按当前 API 用 fp32 UB 工作区与 T.reduce_max/T.reduce_sum、T.tile 运算；任务遵循当前 T.Kernel 的 cid/vid 映射，单个输出只有一个 owner。当前 A2/A3 接口不沿用源仓 T.SimtVF、alloc_reducer/finalize_reducer 或一维网格限制。

## 精度门禁

mask 在 max 前应用；全遮蔽行按算子契约处理，不能产生未定义除零。

低精度输入的 max、sum、平方和、方差、rsqrt、exp 与 online 状态保持 fp32。尾部 max lane 填负无穷、sum lane 填 0。覆盖 tile±1、极长归约轴、抵消、极值、NaN/Inf 契约以及 forward/backward 的公开范围。

## 性能门禁

比较 full score 与 online 两条路径的 GM bytes、exp 次数和端到端 latency。

目标 A2/A3 后端的定向精度测试全通过后，比较端到端 latency、有效 GM bytes、Vector/MTE 时间、核利用率、UB 字节与 stage。多 kernel 方案必须计入 workspace 和全部 launch，不能只报告局部 kernel。

## 实现状态

本文件保留算法流程和验收门禁，状态为 `DESIGN_ONLY`；链接的 A2/A3 示例只提供局部结构依据，不证明本策略已有端到端 kernel。源仓 dav310 Python 模板没有复制到本库。
