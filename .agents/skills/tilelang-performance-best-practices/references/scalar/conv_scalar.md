# Conv/GEMM 标量精简

卷积若 lowering 为 im2col/grouped GEMM，shape、stride、padding、dilation、groups 和 BM/BN/BK 由 Python factory 固化。热路径只保留 output tile 与 K tile 索引，并复用 examples/gemm/example_gemm_persistent.py 或 examples/gemm/example_gemm_intrinsic_persistent.py 的 T.Persistent 任务映射；grouped GEMM 需按实际 group 语义另行实现和验证。

对 depthwise/小 K，标量开销占比高：生成专用 group/layout kernel，合并地址计算，避免大范围 unroll。对标准大 GEMM，优先优化 Cube/MTE 流水，标量改动必须有 timeline 证据。

覆盖所有 layout、padding、group 尾部和 bias 顺序；比较生成代码体积、标量 gap 与端到端 latency。
