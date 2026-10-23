# Softmax AR Full-load

> 迁移来源：Ascend950 同路径文档。保留可复用原文结构与门禁，仅改写 A2/A3 平台、后端、API、路径和证据范围。本页是设计/源码参考；本次未进行 NPU 编译、精度或性能验证，不继承950模板的验证状态。

## 选型

整行输入、fp32 工作区和必要 buffer 可同时放入 UB。

## TileLang A2/A3 实现

一次 T.copy 整行到 fp32 UB 工作区，reduce_max，计算 exp(x-max)，reduce_sum，再归一化并写回。对齐 padding 参与 reduction 前设置幺元。

基础 reduction 代码参考 [当前 RMSNorm](../../../../../../examples/normalization/rms_norm.py) 与 [Online Softmax](../../../../../../examples/softmax/example_online_softmax.py)：GM 数据通过 T.copy 进入 UB，按当前 API 用 fp32 UB 工作区与 T.reduce_max/T.reduce_sum、T.tile 运算；任务遵循当前 T.Kernel 的 cid/vid 映射，单个输出只有一个 owner。当前 A2/A3 接口不沿用源仓 T.SimtVF、alloc_reducer/finalize_reducer 或一维网格限制。

## 精度门禁

稳定形式与 fp32 reference 比较，输出 dtype 只在最终 store 转换。

低精度输入的 max、sum、平方和、方差、rsqrt、exp 与 online 状态保持 fp32。尾部 max lane 填负无穷、sum lane 填 0。覆盖 tile±1、极长归约轴、抵消、极值、NaN/Inf 契约以及 forward/backward 的公开范围。

## 性能门禁

小中等 R 比较 single-stage 与 2-stage 行流水，避免为单行内部强行多版本。

目标 A2/A3 后端的定向精度测试全通过后，比较端到端 latency、有效 GM bytes、Vector/MTE 时间、核利用率、UB 字节与 stage。多 kernel 方案必须计入 workspace 和全部 launch，不能只报告局部 kernel。

## 实现状态

本文件保留算法流程和验收门禁，状态为 `DESIGN_ONLY`；链接的 A2/A3 示例只提供局部结构依据，不证明本策略已有端到端 kernel。源仓 dav310 Python 模板没有复制到本库。
