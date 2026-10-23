# GEMM 提前发射与深流水

> 迁移来源：Ascend950 同路径文档。保留可复用原文结构与门禁，仅改写 A2/A3 平台、后端、API、路径和证据范围。本页是设计/源码参考；本次未进行 NPU 编译、精度或性能验证，不继承950模板的验证状态。

## 适用条件

基础 2-stage 已正确，但生成代码中下一 K tile 的 copy 启动晚于可用时机。

## TileLang A2/A3 实现流程

按当前模式通过可支持的 T.Pipelined、物理 buffer 槽数和循环嵌套表达预取。只有生成源码和 timeline 证明 copy 前移，才保留配置；Expert 手动路径按当前 LOCK DSL/生成事件的所有权合同实现，不沿用源仓“禁止手工 event”的后端规则。

基础实现参考 [当前 GEMM 示例](../../../../../examples/gemm/example_gemm.py) 与 [显式 MMA 示例](../../../../../examples/gemm/example_gemm_intrinsic.py)：按当前 B 的物理布局设置 transpose_B，使用 T.gemm_v0 或 T.mma；浮点 GEMM 的 L0C 使用 fp32，并且只在第一个有效 K tile 设置 init=True。输出 tile 的核数不超过独立任务数；M/N/K 尾块必须走已验证的 padded-copy 或专用 fallback。

## 精度门禁

预取版本不得覆盖仍被 T.gemm_v0/T.mma 路径消费的 L1 tile。

所有输入 dtype 与输出 dtype 分开测试。覆盖 M/N/K 的 tile±1、长 K 消除误差、正负抵消、大小量混合、0、NaN/Inf 契约。浮点 GEMM 的任何 K 分片的 partial 和最终归约保持 fp32；不得以放宽容差掩盖累加顺序或输出转换错误。

## 性能门禁

以生成代码和 timeline 验证实际前移；若仅增加内存没有缩短 gap，则回退。

对每个候选运行编译、目标 A2/A3 后端的定向精度测试和统一 benchmark。报告 latency、TFLOPS、Cube/MTE2 时间、GM 字节、L1/L0/UB 占用、核利用率和生成代码体积。只有完整算子端到端更快且没有精度回退时进入 dispatch。

## A2/A3 专属细节与既有纠错

完整保留上一版 A2/A3 内容于 [A2/A3 补充](a2a3_mte2_preload_design.md)，包括当前接口、边界、历史记录及验证限制。
