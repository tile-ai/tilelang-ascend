# 超长行三阶段 Split Scan

## 适用条件

rows 很少、cols 极长，row-owned kernel 大量空核。

## TileLang 实现

浮点 cumsum 的 Kernel A 每核扫描一个 chunk 并写 local prefix 与 fp32 chunk total；Kernel B 每行扫描 chunk totals；Kernel C 给每个 chunk 加前一 chunk carry。host wrapper 顺序执行三个 kernel。其他 scan 的 combine/fixup 需按精确整数或 value/index/NaN/tie 语义独立证明，不能套用加法。

当前结构参考为 [cummin 示例](../../../../../../examples/cann-bench/cummin/example_cummin.py) 与 [Tile API](../../../../../../tilelang/language/ascend_tile.py)。row-owned 方案由唯一 task 持有每行，沿最后连续轴分 tile，GM/UB 使用 T.copy；浮点累加 carry 使用 fp32，整数值/索引保持其精确 dtype。该分块设计不依赖 grid barrier，具体 scan 原语需按目标后端核验。

## 精度门禁

chunk totals 与 fixup 使用 fp32，覆盖 chunk=1、最后短 chunk 和不同分区数。

覆盖 cols=1、63/64/65、127/128/129、tile±1、多 tile、最大长度，rows 在向量核数以下/等于/以上，以及正负抵消、大小量、NaN/Inf 和溢出契约。浮点 cumsum 与 torch.cumsum(x.float(), dim=-1) 比较并检查每个跨 tile carry；cummin/cummax、整数与索引变体使用对应的精确 reference，不把所有 scan 转为 fp32。

## 性能门禁

计入三次 launch、workspace 和额外两遍 GM；只在完整路径胜过单 owner 时启用。

先通过目标后端 lowering 与定向精度测试。比较 serial correctness、已验证的向量 lane scan、1/2 stage DMA 和三 kernel split-row 的完整端到端 latency；报告有效带宽、Vector/MTE 时间、核利用率、workspace bytes 与所有 launch。

## 实现状态

状态为 `DESIGN_ONLY`；链接示例只提供局部结构依据，不证明本策略已有端到端 kernel。实现后须完成上述精度与性能门禁。
