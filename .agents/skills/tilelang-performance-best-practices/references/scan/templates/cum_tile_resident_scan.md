# UB Resident Scan

## 适用条件

整行及 fp32 工作区可放入 UB。

## TileLang 实现

一次搬入整行，在 UB 工作区完成 prefix 后一次写回。按 dtype lane 对齐分配 padding，但只处理 valid 元素。

当前结构参考为 [cummin 示例](../../../../../../examples/cann-bench/cummin/example_cummin.py) 与 [Tile API](../../../../../../tilelang/language/ascend_tile.py)。row-owned 方案由唯一 task 持有每行，沿最后连续轴分 tile，GM/UB 使用 T.copy；浮点累加 carry 使用 fp32，整数值/索引保持其精确 dtype。该分块设计不依赖 grid barrier，具体 scan 原语需按目标后端核验。

## 精度门禁

验证最后有效 lane 作为输出而非 padding。

覆盖 cols=1、63/64/65、127/128/129、tile±1、多 tile、最大长度，rows 在向量核数以下/等于/以上，以及正负抵消、大小量、NaN/Inf 和溢出契约。浮点 cumsum 与 torch.cumsum(x.float(), dim=-1) 比较并检查每个跨 tile carry；cummin/cummax、整数与索引变体使用对应的精确 reference，不把所有 scan 转为 fp32。

## 性能门禁

适合短中行；比较 full-load 与 streaming 的 launch、copy 和 UB 占用。

先通过目标后端 lowering 与定向精度测试。比较 serial correctness、已验证的向量 lane scan、1/2 stage DMA 和三 kernel split-row 的完整端到端 latency；报告有效带宽、Vector/MTE 时间、核利用率、workspace bytes 与所有 launch。

## 实现状态

状态为 `DESIGN_ONLY`；链接示例只提供局部结构依据，不证明本策略已有端到端 kernel。实现后须完成上述精度与性能门禁。
