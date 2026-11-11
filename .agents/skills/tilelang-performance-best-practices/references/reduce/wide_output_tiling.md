# Reduction 宽输出轴 Tiling

## 目标与适用条件

用于归约计算较轻、非归约连续轴较宽，且小 tile 导致 task、DMA 或地址准备开销过高的归约。它优化输出轴切分，不改变归约轴和数值顺序。

## TileLang 实现

- 增大非归约连续轴的 tile，使一个 Persistent task 连续处理多个 Vector chunk；按几何级数搜索候选，不固化经验值。
- 同时计算 `tasks = outer * ceil(output_inner / tile_inner)`、有效/填充元素、DMA 次数、每核 wave 和全部版本化 buffer 的 UB 字节。
- tile 增大后重新选择核数；避免并行度不足、尾块浪费、UB 溢出或多 stage 压力抵消收益。tile 与 stage 分开做 A/B 测试。
- 使用 Python factory 按静态 shape、dtype 和归约规模 dispatch，并保留正确 fallback。

## 验证门禁

保持低精度归约的 fp32 状态和单输出 owner。覆盖归约规模、tile±1、非整尾块、极小/极大输出轴及 forward/backward；GM 只访问 valid。比较 task/DMA 数、平均搬运粒度、Scalar/MTE/Vector 时间、核利用率、UB 占用和同口径 latency。
