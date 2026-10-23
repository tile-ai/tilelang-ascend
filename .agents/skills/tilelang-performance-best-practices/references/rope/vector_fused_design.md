# RoPE SIMD 融合

> 迁移来源：Ascend950 同路径文档。保留可复用原文结构与门禁，仅改写 A2/A3 平台、后端、API、路径和证据范围。本页是设计/源码参考；本次未进行 NPU 编译、精度或性能验证，不继承950模板的验证状态。

## 实现流程

将 load、旋转与 store 放在同一 tile 中，不写中间 GM。half-split 读取 left/right，计算 left*cos-right*sin 与 right*cos+left*sin；interleaved 读取相邻偶奇元素并使用相同公式。代码从 [当前 rope.py](../../../../../examples/pos_embedding/rope.py) 或 [对应布局示例](../../../../../examples/pos_embedding/RotaryPositionEmbedding/rope_half_interleaved.py) 派生并审计其 shape、索引和 dtype 范围。

sin/cos 若由调用方按 half 维存储则直接 T.copy；若存储为 full dim，factory 明确索引映射。当前 A2/A3 使用实际可用的 T.tile/T.Parallel 表达，不引入源仓 T.SimtVF/T.SimdVF。新的向量化/融合版本只有在通过 lowering 与性能 A/B 后才能替换。

## 门禁

运算提升 fp32，覆盖极值角度、重复 position、非整 SIMD 尾部和 alias。比较融合前后 GM 字节、lane 利用率、寄存器压力和端到端 latency。
