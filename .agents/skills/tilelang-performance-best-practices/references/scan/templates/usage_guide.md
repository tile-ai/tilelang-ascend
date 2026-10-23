# Scan Kernel 接入流程

> 迁移来源：Ascend950 同路径文档。保留可复用原文结构与门禁，仅改写 A2/A3 平台、后端、API、路径和证据范围。本页是设计/源码参考；本次未进行 NPU 编译、精度或性能验证，不继承950模板的验证状态。

## 适用条件

向当前仓库新增 cumulative operator 时使用。

## TileLang A2/A3 实现

Python wrapper 处理空 tensor 并选择 row-owned 或 split-row。factory 固化 tile_cols、dtype 和模式；动态 rows/cols 仅在后端已验证时启用。

当前结构参考为 [cummin 示例](../../../../../../examples/cann-bench/cummin/example_cummin.py) 与 [Tile API](../../../../../../tilelang/language/ascend_tile.py)；不把源仓 scan_base.py 视为本仓已验证基线。row-owned 方案由唯一 task 持有每行，沿最后连续轴分 tile，GM/UB 使用 T.copy；浮点累加 carry 使用 fp32，整数值/索引保持其精确 dtype。该分块设计不依赖 grid barrier，具体 scan 原语需按目标后端核验。

## 精度门禁

先声明 inclusive/exclusive、forward/reverse、axis 和输出 dtype。

覆盖 cols=1、63/64/65、127/128/129、tile±1、多 tile、最大长度，rows 在向量核数以下/等于/以上，以及正负抵消、大小量、NaN/Inf 和溢出契约。浮点 cumsum 与 torch.cumsum(x.float(), dim=-1) 比较并检查每个跨 tile carry；cummin/cummax、整数与索引变体使用对应的精确 reference，不把所有 scan 转为 fp32。

## 性能门禁

定向精度测试覆盖全部 dispatch，再运行同口径 benchmark。

先通过目标 A2/A3 后端 lowering 与定向精度测试。比较 serial correctness、已验证的向量 lane scan、1/2 stage DMA 和三 kernel split-row 的完整端到端 latency；报告有效带宽、Vector/MTE 时间、核利用率、workspace bytes 与所有 launch。

## 专题入口

- [cum_core_partition_scan](cum_core_partition_scan.md)
- [cum_lane_parallel_scan](cum_lane_parallel_scan.md)
- [cum_oneway_sklansky](cum_oneway_sklansky.md)
- [cum_row_kogge_stone](cum_row_kogge_stone.md)
- [cum_streaming_scan](cum_streaming_scan.md)
- [cum_tile_resident_scan](cum_tile_resident_scan.md)

## 实现状态

本文件保留算法流程和验收门禁，状态为 `DESIGN_ONLY`；链接的 A2/A3 示例只提供局部结构依据，不证明本策略已有端到端 kernel。源仓 dav310 Python 模板没有复制到本库。
