# SIMD Lane-parallel Scan

> 迁移来源：Ascend950 同路径文档。保留可复用原文结构与门禁，仅改写 A2/A3 平台、后端、API、路径和证据范围。本页是设计/源码参考；本次未进行 NPU 编译、精度或性能验证，不继承950模板的验证状态。

## 适用条件

同一向量内部希望用 log2(lanes) 个 shift-add 层替代串行 lane 循环。

## TileLang A2/A3 实现

使用当前已验证的 T.tile gather/index 组合构造 offset=1,2,4... 的静态移位向量，每层只对 lane>=offset 更新；浮点向量间仍通过 fp32 carry 串联，整数值/索引保持精确 dtype。该路径必须先独立 lowering 微测，不能把源仓寄存器 shift 直接当成本仓接口。

当前结构参考为 [cummin 示例](../../../../../../examples/cann-bench/cummin/example_cummin.py) 与 [Tile API](../../../../../../tilelang/language/ascend_tile.py)；不把源仓 scan_base.py 视为本仓已验证基线。row-owned 方案由唯一 task 持有每行，沿最后连续轴分 tile，GM/UB 使用 T.copy；浮点累加 carry 使用 fp32，整数值/索引保持其精确 dtype。该分块设计不依赖 grid barrier，具体 scan 原语需按目标后端核验。

## 精度门禁

每一层 mask 必须阻止跨向量串行污染，尾向量只提取最后有效 lane。

覆盖 cols=1、63/64/65、127/128/129、tile±1、多 tile、最大长度，rows 在向量核数以下/等于/以上，以及正负抵消、大小量、NaN/Inf 和溢出契约。浮点 cumsum 与 torch.cumsum(x.float(), dim=-1) 比较并检查每个跨 tile carry；cummin/cummax、整数与索引变体使用对应的精确 reference，不把所有 scan 转为 fp32。

## 性能门禁

测 lane scan 指令数、寄存器压力和短行开销；失败或不增益时保持 streaming fallback。

先通过目标 A2/A3 后端 lowering 与定向精度测试。比较 serial correctness、已验证的向量 lane scan、1/2 stage DMA 和三 kernel split-row 的完整端到端 latency；报告有效带宽、Vector/MTE 时间、核利用率、workspace bytes 与所有 launch。

## 实现状态

本文件保留算法流程和验收门禁，状态为 `DESIGN_ONLY`；链接的 A2/A3 示例只提供局部结构依据，不证明本策略已有端到端 kernel。源仓 dav310 Python 模板没有复制到本库。
