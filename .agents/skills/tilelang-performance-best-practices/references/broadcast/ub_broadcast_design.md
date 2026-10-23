# UB 常驻 Broadcast

## 目标与适用条件

适用于同一小张量被多个输出 tile 或多个广播倍率重复读取。

## TileLang Ascend 实现

在每核任务循环外将小张量 T.copy 到单版本 UB；循环内只加载输出 tile 并从 resident buffer 读取。若小张量跨核共享，每核各自加载，避免依赖未验证的跨核共享。

实现沿用当前仓库 T.Kernel 的网格和 `(cid, vid)` 映射。纯向量任务从已确认的可用 AIV 核数及 launch 映射取核数，并以 min(核数, 独立任务数) 限制空核。GM 与 UB/L1 之间使用 T.copy；连续规则计算使用当前模式的 T.Parallel 或 T.tile 原语；跨 tile 任务使用当前 T.Persistent、Fixed Core 核内循环或已验证的 T.Pipelined，多版本方式沿用当前编译器支持的流水或显式槽位所有权。

## 精度门禁

每个 core 的 resident 数据必须完整初始化；动态长度尾部填充不得参与有效输出。

必须覆盖最小 shape、常见 shape、最大 shape、tile-1/tile/tile+1、非 32B 尾块、0、正负极值以及接口规定的 NaN/Inf 语义。不得通过扩大容差、降低 reference 精度或跳过 case 获得通过。

## 性能门禁

计算 resident 数据的每核复制成本与复用次数；只有节省的 GM 流量超过初始化成本且 UB 不挤压流水时启用。

先用目标仓库既有后端与命令运行定向精度测试，再运行相关完整测试套；全部通过后，在相同输入、dtype、warmup、repeat、设备和并发条件下测量性能。报告 kernel latency、有效 GM 带宽、UB 占用、stage 数和基线差异。

## 可执行代码与证据

权重常驻模式参考当前仓库同类算子；参数搬运、cast 和 broadcast 的 A2/A3 结构见 [广播指南](guide.md)。

A2/A3 参数 staging、cast 与广播维度的补充约束见 [广播指南](guide.md)。
