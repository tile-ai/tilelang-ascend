# Shape-aware 输出 Tile 调度

## 适用条件

M/N 输出 tile 足够多但尾部负载不均，或遍历顺序影响 A/B L2 复用。

## TileLang 实现流程

使用当前 cid/grid-stride 或已验证 T.Persistent 映射作为基础；需要 serpentine/尾块特化时在 Python factory 生成静态 scheduler 变体。每个 task 仍唯一拥有一个输出 tile，不依赖跨核同步。

基础实现参考 [当前 GEMM 示例](../../../../../examples/gemm/example_gemm.py) 与 [显式 MMA 示例](../../../../../examples/gemm/example_gemm_intrinsic.py)：按当前 B 的物理布局设置 transpose_B，使用 T.gemm_v0 或 T.mma；浮点 GEMM 的 L0C 使用 fp32，并且只在第一个有效 K tile 设置 init=True。输出 tile 的核数不超过独立任务数；M/N/K 尾块必须走已验证的 padded-copy 或专用 fallback。

## 精度门禁

任何重排必须覆盖每个逻辑 tile 恰好一次；用 tile-id bitmap 测试重复与遗漏。

所有输入 dtype 与输出 dtype 分开测试。覆盖 M/N/K 的 tile±1、长 K 消除误差、正负抵消、大小量混合、0、NaN/Inf 契约。浮点 GEMM 的任何 K 分片的 partial 和最终归约保持 fp32；不得以放宽容差掩盖累加顺序或输出转换错误。

## 性能门禁

分别测尾部 wave、整体 latency、L2 hit 与核间负载；没有 profile 证据不保留复杂调度。

对每个候选运行编译、目标后端的定向精度测试和统一 benchmark。报告 latency、TFLOPS、Cube/MTE2 时间、GM 字节、L1/L0/UB 占用、核利用率和生成代码体积。只有完整算子端到端更快且没有精度回退时进入 dispatch。
