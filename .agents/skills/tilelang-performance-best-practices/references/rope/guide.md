# A2/A3 RoPE 优化索引

> 迁移来源：Ascend950 同路径文档。保留可复用原文结构与门禁，仅改写 A2/A3 平台、后端、API、路径和证据范围。本页是设计/源码参考；本次未进行 NPU 编译、精度或性能验证，不继承950模板的验证状态。

## 实现

以 [当前 rope.py](../../../../../examples/pos_embedding/rope.py) 和 [两种布局示例](../../../../../examples/pos_embedding/RotaryPositionEmbedding/rope_half_interleaved.py) 为源码参考。Python factory 固化 layout、dim、dtype 与 position offset；按当前 T.Kernel 的 cid/vid 合同以 grid-stride 分配 token×head task。输入、sin、cos 通过 T.copy 进入 UB，使用当前已支持的 T.tile/T.Parallel 路径以 fp32 完成成对旋转，最后转换输出 dtype。

half-split 与 interleaved 必须生成不同 kernel，不能在热循环中动态分支。任意 position tensor 需要先验证当前后端的标量/离散索引 lowering；编译期 position offset 的连续位置是另一独立候选，不能扩大既有示例的语义。生产路径使用可 lowering 的 Tile API、跨 head 复用或算子融合，并附带硬件、版本、命令和原始性能结果；不迁移源仓 SimtVF 的已编译声明。

## 精度

覆盖 position=0/最大值、offset、奇偶 token/head、dim 边界、两种 layout、原位/非原位、bf16/fp16/fp32。reference 的 sin/cos、乘加和拼接使用 fp32；输出只在最终边界转换。

## 性能

比较每 head 重载 sin/cos 与 token-grouped resident 版本，报告 GM bytes、Vector/MTE 时间、UB 占用和 latency。定向精度测试全通过后运行相关完整测试套。

## A2/A3 专属细节与既有纠错

完整保留上一版 A2/A3 内容于 [A2/A3 补充](a2a3_guide.md)，包括当前接口、边界、历史记录及验证限制。

## 专题入口

- [RoPE 融合](vector_fused_design.md)
