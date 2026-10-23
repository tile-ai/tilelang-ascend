# FlashAttention 标量热循环精简

> 迁移来源：Ascend950 同路径文档。保留可复用原文结构与门禁，仅改写 A2/A3 平台、后端、API、路径和证据范围。本页是设计/源码参考；本次未进行 NPU 编译、精度或性能验证，不继承950模板的验证状态。

## 优化目标

KV 循环的地址、循环和状态管理占用明显标量周期。

## A2/A3 实现

缩短局部变量生命周期，复用静态 offset，避免在循环内构造复杂 view；把固定 row/vector 循环用 range/T.unroll 表达，把动态 sequence 只保留在 block 数和尾块。

FlashAttention 参考 [当前 BHSD 实现](../../../../../examples/flash_attention/flash_attn_bhsd.py)、[Expert/auto_pipeline 路线](../../../../../examples/flash_attention/fa_opt/README.md) 及配套 [精度测试](../../../../../examples/flash_attention/test_flash_attn_bhsd.py)。按当前源码核对 QK/PV、online softmax、mask、buffer 布局和流水；示例存在不代表本轮后端已验证。修改 tile、stage、mask 或 forwarding 后运行对应精度用例，mask 必须在 row max 前应用。

## 精度门禁

精简不能删除真实的 mask、尾块或状态更新依赖。

必须与 torch scaled_dot_product_attention reference 比较，覆盖全遮蔽行、causal 对角、sequence/block 边界、极大正负 logits、重复最大值、NaN/Inf 契约及支持的 dtype。online 状态和输出累加保持 fp32。

## 性能门禁

通过生成源码与 timeline 统计标量 gap，只有端到端 latency 改善才保留。

记录完整 kernel latency、TFLOPS、Q/K/V/O GM bytes、Cube/Vector/MTE 时间、L1/L0/UB 占用和 stage。一次只改变一个优化变量，定向精度测试全通过后才能比较相关完整测试套。

## A2/A3 接口依据

当前展开入口为 [T.unroll](../../../../../tilelang/language/tir/ir.py)，不使用源文的
`T.Unroll` 拼写；展开后的实际代码量和依赖仍由上述门禁检查。
