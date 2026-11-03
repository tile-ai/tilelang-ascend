# Broadcast 尾部填充

## 目标与适用条件

用于 broadcast 输入或输出的最后一块不满足对齐要求。

## TileLang Ascend 实现

T.npu_copy_v2(..., pad_value=value) 只在目标后端示例支持的形状上使用；max/topk 填负无穷，sum 填 0。输出始终只写 valid 区域。需要区分数据填充值和最终输出值。

实现沿用当前仓库 T.Kernel 的网格和 `(cid, vid)` 映射。纯向量任务从已确认的可用 AIV 核数及 launch 映射取核数，并以 min(核数, 独立任务数) 限制空核。GM 与 UB/L1 之间使用 T.copy；连续规则计算使用当前模式的 T.Parallel 或 T.tile 原语；跨 tile 任务使用当前 T.Persistent、Fixed Core 核内循环或已验证的 T.Pipelined，多版本方式沿用当前编译器支持的流水或显式槽位所有权。

## 精度门禁

覆盖最后一块有效元素为 1、lane-1、lane+1，并验证负无穷、NaN 和 dtype 转换。

必须覆盖最小 shape、常见 shape、最大 shape、tile-1/tile/tile+1、非 32B 尾块、0、正负极值以及接口规定的 NaN/Inf 语义。不得通过扩大容差、降低 reference 精度或跳过 case 获得通过。

## 性能门禁

pad copy 与显式 fill+copy 做 A/B；记录额外 UB 写入和 copy latency。

先用目标仓库既有后端与命令运行定向精度测试，再运行相关完整测试套；全部通过后，在相同输入、dtype、warmup、repeat、设备和并发条件下测量性能。报告 kernel latency、有效 GM 带宽、UB 占用、stage 数和基线差异。

## 可执行代码与证据

直接参考 [尾块测试](../../../../../testing/python/language/test_tilelang_ascend_language_tail_block.py) 与 [copy_op.py](../../../../../tilelang/language/copy_op.py)。

参数 staging、cast 与广播维度的约束见 [广播指南](broadcast_design.md)。
