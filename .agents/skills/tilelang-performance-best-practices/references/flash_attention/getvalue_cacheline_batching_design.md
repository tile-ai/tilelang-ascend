# KV 连续批量搬运

> 迁移来源：Ascend950 同路径文档。保留可复用原文结构与门禁，仅改写 A2/A3 平台、后端、API、路径和证据范围。本页是设计/源码参考；本次未进行 NPU 编译、精度或性能验证，不继承950模板的验证状态。

## 优化目标

K/V block 的小片段或不连续访问导致 copy transaction 过多。

## A2/A3 实现

选择 BC 使 K[BC,D] 和目标 PV 路径需要的 V tile 形成连续矩形 T.copy；V 的转置/分形布局通过当前已验证的 T.copy、layout 或 T.tile.transpose 路径表达，不编造 transpose=True 参数。相邻 KV block 由已验证的 Pipelined 或显式 stage 多版本承载，不编写标量逐元素 load。

FlashAttention 参考 [当前 BHSD 实现](../../../../../examples/flash_attention/flash_attn_bhsd.py)、[Expert/auto_pipeline 路线](../../../../../examples/flash_attention/fa_opt/README.md) 及配套 [精度测试](../../../../../examples/flash_attention/test_flash_attn_bhsd.py)。按当前源码核对 QK/PV、online softmax、mask、buffer 布局和流水；示例存在不代表本轮后端已验证。修改 tile、stage、mask 或 forwarding 后运行对应精度用例，mask 必须在 row max 前应用。

## 精度门禁

验证 K/V layout、head stride、GQA/MQA 共享规则和 KV 尾块。

必须与 torch scaled_dot_product_attention reference 比较，覆盖全遮蔽行、causal 对角、sequence/block 边界、极大正负 logits、重复最大值、NaN/Inf 契约及支持的 dtype。online 状态和输出累加保持 fp32。

## 性能门禁

比较 copy 数、burst、MTE2 时间和增大 BC 后的 L1/L0 压力。

记录完整 kernel latency、TFLOPS、Q/K/V/O GM bytes、Cube/Vector/MTE 时间、L1/L0/UB 占用和 stage。一次只改变一个优化变量，定向精度测试全通过后才能比较相关完整测试套。
