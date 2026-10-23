# A2/A3 Convolution 优化索引

## 支持路径

先检查当前仓库的 Ascend lowering 是否提供满足语义的 convolution primitive。没有时，将卷积显式变换为 im2col/grouped GEMM，并复用 examples/gemm/example_gemm.py；不得调用不存在的 Load3D 或 layout API。

Python factory 固化 N/C/H/W、kernel、stride、padding、dilation、groups 与 layout。im2col tile 只在 UB/L1 中生成，不写完整中间矩阵到 GM。标准卷积按 output spatial×Cout 形成 GEMM M/N，Cin×kH×kW 形成 K；depthwise 按 group 批处理并为小 K 生成专用向量或 grouped GEMM 路径。

## 门禁

精度覆盖 NCHW/支持 layout、padding/stride/dilation、groups、bias、所有边缘窗口和尾块。fp16/bf16 使用 fp32 accumulator。性能比较端到端 latency、im2col 地址计算、GM bytes、Cube/MTE/Vector 时间和 workspace；未通过 lowering 微测的 primitive 不进入代码。

## 当前参考范围

上述通用 im2col/grouped GEMM 是 `DESIGN_ONLY`，没有在本次迁移中取得 A2/A3 编译或上板证据。`examples/causal_conv1d/causal_conv1d.py` 可提供核内卷积、常驻权重和流水的源码参考，但不是通用二维卷积模板。GEMM 示例只证明矩阵乘实现结构存在，不能代替 im2col、group 和边缘窗口验证。详细流程见 [Conv 实现流程](conv_common_perf_experience.md)。
