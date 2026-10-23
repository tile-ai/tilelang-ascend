# 单轴 Broadcast

## 目标与适用条件

适用于一个输入轴长度为 1，或一段连续数据沿相邻维复制。

## TileLang Ascend 实现

把输出线性索引分解为 outer、broadcast index、inner。inner 连续时，每个 task 搬一次源 slice 到 UB，并在 SIMD 循环中写多个目标位置；标量 inner 使用当前已验证的 fill/broadcast 原语。

实现沿用当前仓库 T.Kernel 的网格和 `(cid, vid)` 映射。纯向量任务从已确认的可用 AIV 核数及 launch 映射取核数，并以 min(核数, 独立任务数) 限制空核。GM 与 UB/L1 之间使用 T.copy；连续规则计算使用当前模式的 T.Parallel 或 T.tile 原语；跨 tile 任务使用当前 T.Persistent、Fixed Core 核内循环或已验证的 T.Pipelined，多版本方式沿用当前编译器支持的流水或显式槽位所有权。

## 精度门禁

覆盖 broadcast size 为 1、2、向量 lane±1 和大倍率；检查索引乘法使用足够位宽。

必须覆盖最小 shape、常见 shape、最大 shape、tile-1/tile/tile+1、非 32B 尾块、0、正负极值以及接口规定的 NaN/Inf 语义。不得通过扩大容差、降低 reference 精度或跳过 case 获得通过。

## 性能门禁

在广播倍率大时测源 GM 字节是否接近只读一次；倍率小时防止 UB staging 的固定开销超过直接访问。

先用目标仓库既有后端与命令运行定向精度测试，再运行相关完整测试套；全部通过后，在相同输入、dtype、warmup、repeat、设备和并发条件下测量性能。报告 kernel latency、有效 GM 带宽、UB 占用、stage 数和基线差异。

## 可执行代码与证据

基于 [Vector Add](../../../../../examples/elementwise/elementwise_add.py)；动态 stride 按 [API 计算参考](../../../tilelang-custom-skill/tilelang-api-best-practices/references/api-compute.md) 与 [copy_op.py](../../../../../tilelang/language/copy_op.py) 中实际接口核对。

A2/A3 参数 staging、cast 与广播维度的补充约束见 [广播指南](guide.md)。
