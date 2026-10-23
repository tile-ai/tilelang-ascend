# A2/A3 Conv 实现流程

1. 用 PyTorch reference 固定输出 shape、padding 和 bias 语义。
2. 在 Python factory 选择 direct primitive 或 tiled im2col+GEMM。
3. 每个 task 负责一个 output spatial/Cout tile；在 UB/L1 中按有效窗口生成 A tile，越界 padding 写 0。
4. Weight 按当前 GEMM 路径要求的布局进入 L1，调用 T.gemm_v0 或经 L0A/L0B 分块后调用 T.mma，并在 fp32 L0C 累加。布局和 transpose_B 参数按 examples/gemm/example_gemm.py、example_gemm_transpose_l1.py 及当前 API 核对，不能直接套用 [Cout,K]。
5. bias 与 activation 在输出 tile 转换前融合，保持 reference 运算顺序。
6. depthwise 对多个 group 合并 task，但禁止混合 group 的 K 累加。

tile、K 展开、weight resident 和 pipeline stage 必须满足 L1/L0/UB 预算。覆盖每个 output 边缘和 group 尾部；性能按实际 shape 搜索，不能沿用固定硬件经验阈值。

本流程为 `DESIGN_ONLY`；适用范围和当前源码依据见 [Convolution 优化索引](guide.md)。
