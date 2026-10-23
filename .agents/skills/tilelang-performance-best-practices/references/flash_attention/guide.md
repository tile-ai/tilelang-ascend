# A2/A3 FlashAttention 优化索引

> 迁移来源：Ascend950 同路径文档。保留可复用原文结构与门禁，仅改写 A2/A3 平台、后端、API、路径和证据范围。本页是设计/源码参考；本次未进行 NPU 编译、精度或性能验证，不继承950模板的验证状态。

## 优化目标

高性能实现 softmax(QK^T·scale)·V，避免显式写出完整 score/probability。

## A2/A3 实现

按 query block 分配 kernel task；Q 驻留，KV 分块流式。每个 KV block 更新 fp32 m、l 和 O accumulator，最后 O/l 写回。factory 按 D、sequence、causal 与 dtype 选择 BR/BC 和 stage。

FlashAttention 参考 [当前 BHSD 实现](../../../../../examples/flash_attention/flash_attn_bhsd.py)、[Expert/auto_pipeline 路线](../../../../../examples/flash_attention/fa_opt/README.md) 及配套 [精度测试](../../../../../examples/flash_attention/test_flash_attn_bhsd.py)。按当前源码核对 QK/PV、online softmax、mask、buffer 布局和流水；示例存在不代表本轮后端已验证。修改 tile、stage、mask 或 forwarding 后运行对应精度用例，mask 必须在 row max 前应用。

## 精度门禁

同时验证数学输出与每行 softmax 归一化；不以较低精度 reference 掩盖误差。

必须与 torch scaled_dot_product_attention reference 比较，覆盖全遮蔽行、causal 对角、sequence/block 边界、极大正负 logits、重复最大值、NaN/Inf 契约及支持的 dtype。online 状态和输出累加保持 fp32。

## 性能门禁

基线、tile、stage、mask specialization、Q resident 分开 A/B。

记录完整 kernel latency、TFLOPS、Q/K/V/O GM bytes、Cube/Vector/MTE 时间、L1/L0/UB 占用和 stage。一次只改变一个优化变量，定向精度测试全通过后才能比较相关完整测试套。

## A2/A3 专属细节与既有纠错

完整保留上一版 A2/A3 内容于 [A2/A3 补充](a2a3_guide.md)，包括当前接口、边界、历史记录及验证限制。

## 专题入口

- [编译期预计算](scalar_precompute_design.md)
- [标量热循环](scalar_general_techniques_design.md)
- [状态访存](scalarldst_subbound_design.md)
- [KV 批量搬运](getvalue_cacheline_batching_design.md)
- [C/V 转发目标与 A2/A3 通路](tilingdata_cv_ssbuf_forwarding_design.md)
- [A2/A3 workspace ownership](cv_pipeline_design.md)
