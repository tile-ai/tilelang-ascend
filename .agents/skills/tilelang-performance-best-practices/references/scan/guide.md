# A2/A3 Prefix Scan

> 迁移来源：Ascend950 同路径文档。保留可复用原文结构与门禁，仅改写 A2/A3 平台、后端、API、路径和证据范围。本页是设计/源码参考；本次未进行 NPU 编译、精度或性能验证，不继承950模板的验证状态。

## 适用条件

inclusive scan、cummin/cummax 及带 index 变体。先固定 axis、方向、dtype、累加和输出语义。

## TileLang A2/A3 实现

行数足够时使用 row-owned streaming；短行可 full-load；极少超长行才使用 local-prefix、chunk-total scan、fixup 三 kernel 方案。lane-parallel 仅在对应 SIMD gather/shift 微测通过后启用。

当前结构参考为 [cummin 示例](../../../../../examples/cann-bench/cummin/example_cummin.py) 与 [Tile API](../../../../../tilelang/language/ascend_tile.py)；不把源仓 scan_base.py 视为本仓已验证基线。row-owned 方案由唯一 task 持有每行，沿最后连续轴分 tile，GM/UB 使用 T.copy；浮点累加 carry 使用 fp32，整数值/索引保持其精确 dtype。该分块设计不依赖 grid barrier，具体 scan 原语需按目标后端核验。源仓 serial inner scan 的精度回归定位不作为本仓性能结论；没有附带硬件、版本、命令和原始结果的性能数字不作为生产 dispatch 依据。

## 精度门禁

cummin/cummax 的 tie-break、NaN 和 index 传播需独立 reference。

覆盖 cols=1、63/64/65、127/128/129、tile±1、多 tile、最大长度，rows 在向量核数以下/等于/以上，以及正负抵消、大小量、NaN/Inf 和溢出契约。浮点 cumsum 与 torch.cumsum(x.float(), dim=-1) 比较并检查每个跨 tile carry；cummin/cummax、整数与索引变体使用对应的精确 reference，不把所有 scan 转为 fp32。

## 性能门禁

按短/中/长行建立 dispatch，不能以单一 shape 决定。向量 lane scan 或 split-row 只有通过当前 A2/A3 lowering、精度及端到端比较后才替换 serial inner scan；不适用时保留正确 streaming/UB-local fallback，尤其不能为消除标量路径而缩小整数值域。

先通过目标 A2/A3 后端 lowering 与定向精度测试。比较 serial correctness、已验证的向量 lane scan、1/2 stage DMA 和三 kernel split-row 的完整端到端 latency；报告有效带宽、Vector/MTE 时间、核利用率、workspace bytes 与所有 launch。

## A2/A3 专属细节与既有纠错

完整保留上一版 A2/A3 内容于 [A2/A3 补充](a2a3_guide.md)，包括当前接口、边界、历史记录及验证限制。

## 专题入口

- [A2/A3 正交轴与精确比较](orthogonal_axis_vectorization.md)
- [算法策略与接入](templates/usage_guide.md)
