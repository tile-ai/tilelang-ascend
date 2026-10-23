# 多维 Broadcast 与矩形搬运

## 目标与适用条件

适用于可合并为连续矩形的多轴广播。

## TileLang Ascend 实现

先在 host/factory 合并相邻连续维，得到 outer、broadcast、inner 三段。T.copy 只负责连续矩形；不能表达的离散维用当前 DSL 显式索引。禁止假设任意 N 维广播都能 lowering 为单条 DMA。

实现沿用当前仓库 T.Kernel 的网格和 `(cid, vid)` 映射。纯向量任务从已确认的可用 AIV 核数及 launch 映射取核数，并以 min(核数, 独立任务数) 限制空核。GM 与 UB/L1 之间使用 T.copy；连续规则计算使用当前模式的 T.Parallel 或 T.tile 原语；跨 tile 任务使用当前 T.Persistent、Fixed Core 核内循环或已验证的 T.Pipelined，多版本方式沿用当前编译器支持的流水或显式槽位所有权。

## 精度门禁

验证合并前后逻辑 shape、stride 与 storage offset；非连续 view 使用当前仓库已验证的 stride/BufferRegion 表达。

必须覆盖最小 shape、常见 shape、最大 shape、tile-1/tile/tile+1、非 32B 尾块、0、正负极值以及接口规定的 NaN/Inf 语义。不得通过扩大容差、降低 reference 精度或跳过 case 获得通过。

## 性能门禁

比较维度合并前后的 copy 数、标量地址计算和总 GM 字节。只有生成代码确认矩形搬运成立时采用 DMA 路径。

先用目标仓库既有后端与命令运行定向精度测试，再运行相关完整测试套；全部通过后，在相同输入、dtype、warmup、repeat、设备和并发条件下测量性能。报告 kernel latency、有效 GM 带宽、UB 占用、stage 数和基线差异。

## 可执行代码与证据

动态维、张量视图与搬运限制按当前 [copy_op.py](../../../../../tilelang/language/copy_op.py)、[API 计算参考](../../../tilelang-custom-skill/tilelang-api-best-practices/references/api-compute.md) 的 stride 模式及 [common.h](../../../../../src/tl_templates/ascend/common.h) 核对。

本页讨论多维广播的矩形搬运，不引入 NDDMA 指令；A2/A3 具体 API 与参数加载见 [广播补充](guide.md)。
