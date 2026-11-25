# FlashAttention 编译期预计算

## 优化目标

scale、block count、offset 和 mask 边界在热 KV 循环中重复计算。

## 实现

由 Python factory 计算 BR、BC、NUM_KV_BLOCKS、scale 与静态 stride；循环内只保留 task 相关 offset。causal/full 两种 kernel 分开生成，消除热路径运行时模式分支。

FlashAttention 参考 [当前 BHSD 实现](../../../../../examples/flash_attention/flash_attn_bhsd.py)、[Expert/auto_pipeline 路线](../../../../../examples/flash_attention/fa_opt/README.md) 及配套 [精度测试](../../../../../examples/flash_attention/test_flash_attn_bhsd.py)。按当前源码核对 QK/PV、online softmax、mask、buffer 布局和流水；示例存在不代表改动后的候选已验证。修改 tile、stage、mask 或 forwarding 后运行对应精度用例，mask 必须在 row max 前应用。

## 精度门禁

预计算值用足够精度，尤其 scale 与 log2/exp 转换常量。

必须与 torch scaled_dot_product_attention reference 比较，覆盖全遮蔽行、causal 对角、sequence/block 边界、极大正负 logits、重复最大值、NaN/Inf 契约及支持的 dtype。online 状态和输出累加保持 fp32。

## 性能门禁

检查生成代码的标量指令、寄存器与代码体积；过度 specialization 需计入编译缓存。

记录完整 kernel latency、TFLOPS、Q/K/V/O GM bytes、Cube/Vector/MTE 时间、L1/L0/UB 占用和 stage。一次只改变一个优化变量，定向精度测试全通过后才能比较相关完整测试套。
