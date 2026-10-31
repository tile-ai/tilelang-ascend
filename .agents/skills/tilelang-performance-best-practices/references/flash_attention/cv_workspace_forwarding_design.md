# Cube-Vector Workspace 转发

## 优化目标

QK 的 fp32 score 要交给 Vector softmax，概率再交回 Cube PV，目标是减少不必要的 GM 与中间复制。

## 实现

当前路线通过有界 GM workspace 交接 L0C→GM→UB 的 score，以及 UB→GM→L1 的 probability，再执行对应 T.gemm_v0/T.mma 的 PV。BR/BC 与 buffer 大小由 Python factory 固化。减少冗余中间副本、复用 per-core stage ring 与 producer/consumer 重叠仍是候选。

FlashAttention 参考 [当前 BHSD 实现](../../../../../examples/flash_attention/flash_attn_bhsd.py)、[Expert/auto_pipeline 路线](../../../../../examples/flash_attention/fa_opt/README.md) 及配套 [精度测试](../../../../../examples/flash_attention/test_flash_attn_bhsd.py)。按当前源码核对 QK/PV、online softmax、mask、buffer 布局和流水；示例存在不代表改动后的候选已验证。修改 tile、stage、mask 或 forwarding 后运行对应精度用例，mask 必须在 row max 前应用。

## 精度门禁

bf16 probability 转换前在 fp32 中完成 max、exp 与 sum；评估转换误差对最终 O 的影响。

必须与 torch scaled_dot_product_attention reference 比较，覆盖全遮蔽行、causal 对角、sequence/block 边界、极大正负 logits、重复最大值、NaN/Inf 契约及支持的 dtype。online 状态和输出累加保持 fp32。

## 性能门禁

统计 L0C/UB/L1 搬运与同步 gap；增大 buffer 只有在重叠收益超过容量代价时启用。

记录完整 kernel latency、TFLOPS、Q/K/V/O GM bytes、Cube/Vector/MTE 时间、L1/L0/UB 占用和 stage。一次只改变一个优化变量，定向精度测试全通过后才能比较相关完整测试套。

## 通路依据

[当前 copy 源码](../../../../../tilelang/language/copy_op.py) 将 copy_cv_experiment/copy_vc_experiment 标为 PTO A5 only，不能仅因接口名存在就用于当前 kernel。普通 workspace 的完整生命周期见 [C/V 流水](cv_pipeline_design.md)。
