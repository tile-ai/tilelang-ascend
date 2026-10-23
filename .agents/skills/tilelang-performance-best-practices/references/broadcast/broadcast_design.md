# Broadcast 总体实现

## 目标与适用条件

用于标量、单轴和多轴广播的 elementwise 算子。将输出空间展平为独立 tile，并从输出索引计算输入索引。

## TileLang Ascend 实现

连续广播轴优先把小输入搬入 UB 一次，然后在 Vector 计算中复用；标量使用当前已验证的 fill/broadcast 原语。复杂不连续索引按当前 DSL 显式计算，并通过 Python factory 固化 rank、axis 和 stride。输出任务由当前 T.Persistent 分配，并核对其 domain/group_size 约束与 Kernel 映射。

实现沿用当前仓库 T.Kernel 的网格和 `(cid, vid)` 映射。纯向量任务从已确认的可用 AIV 核数及 launch 映射取核数，并以 min(核数, 独立任务数) 限制空核。GM 与 UB/L1 之间使用 T.copy；连续规则计算使用当前模式的 T.Parallel 或 T.tile 原语；跨 tile 任务使用当前 T.Persistent、Fixed Core 核内循环或已验证的 T.Pipelined，多版本方式沿用当前编译器支持的流水或显式槽位所有权。

## 精度门禁

广播前后的 dtype 转换顺序必须与 reference 一致；低精度超越函数与累加提升 fp32。验证 size=1 轴、多个广播轴、零维标量和尾块。

必须覆盖最小 shape、常见 shape、最大 shape、tile-1/tile/tile+1、非 32B 尾块、0、正负极值以及接口规定的 NaN/Inf 语义。不得通过扩大容差、降低 reference 精度或跳过 case 获得通过。

## 性能门禁

重点测 GM 重复读取量、索引计算开销和向量利用率。小广播输入常驻 UB，避免每个输出 tile 重复搬运。

先用目标仓库既有后端与命令运行定向精度测试，再运行相关完整测试套；全部通过后，在相同输入、dtype、warmup、repeat、设备和并发条件下测量性能。报告 kernel latency、有效 GM 带宽、UB 占用、stage 数和基线差异。

## 可执行代码与证据

复用 [Vector Add](../../../../../examples/elementwise/elementwise_add.py) 的基础结构与 [流水版本](../../../../../examples/elementwise/elementwise_add_pipeline.py) 的槽位映射；广播 API 核对 [ascend_tile.py](../../../../../tilelang/language/ascend_tile.py)。

A2/A3 参数 staging、cast 与广播维度的补充约束见 [广播指南](guide.md)。

## 专题路由

- 一维任务与广播基线：[一维实现](onedim_design.md)。
- 小输入在 UB 内复用：[UB 广播](ub_broadcast_design.md)。
- 输入填充与尾块：[DataCopyPad](datacopypad_design.md)。
- 多轴合并与矩形搬运：[多维搬运](nddma_design.md)。
- 多轴 tiling、驻留及 shape 特化：[高级 Tiling](advanced_tiling.md)。
