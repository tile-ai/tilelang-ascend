# Elementwise SIMD 与融合

## 目标与适用条件

用于连续逐元素、量化转换、激活和简单复合表达式。

## TileLang Ascend 实现

连续规则路径使用当前模式的 T.Parallel/T.tile 原语；bf16/fp16 超越函数先转 fp32。把只使用一次的中间结果保存在寄存器或同一 UB tile，融合相邻表达式以消除 GM round-trip。复杂索引按当前 DSL 已验证的访问路径处理。

仿射链 `a*x+b` 将当前支持且数学等价的融合原语与 mul+add 作为不同 lowering 候选；不假设 fast-math 会自动融合，从当前生成代码确认指令，并分别验证精度和同口径性能。

短记录可把多个 token 打包到一个向量，按当前已验证的索引/Vector 重排路径处理，并将常量与索引模式移出热循环。常见整 tile 可由 Python wrapper 固化 shape 和展开数，其他 shape 保留动态尾块。完整结构见 [索引短记录变换](indexed_short_record.md)。

实现沿用当前仓库 T.Kernel 的网格和 `(cid, vid)` 映射。纯向量任务从已确认的可用 AIV 核数及 launch 映射取核数，并以 min(核数, 独立任务数) 限制空核。GM 与 UB/L1 之间使用 T.copy；连续规则计算使用当前模式的 T.Parallel 或 T.tile 原语；跨 tile 任务使用当前 T.Persistent、Fixed Core 核内循环或已验证的 T.Pipelined，多版本方式沿用当前编译器支持的流水或显式槽位所有权。

## 精度门禁

严格保持 reference 的运算顺序、饱和/舍入模式和输出 cast 边界。量化必须覆盖零 scale、极值与非有限输入契约。

必须覆盖最小 shape、常见 shape、最大 shape、tile-1/tile/tile+1、非 32B 尾块、0、正负极值以及接口规定的 NaN/Inf 语义。不得通过扩大容差、降低 reference 精度或跳过 case 获得通过。

## 性能门禁

测向量 lane 利用率、指令数量、GM 字节和 spill。融合导致寄存器压力或代码膨胀时拆分。

先用目标仓库既有后端与命令运行定向精度测试，再运行相关完整测试套；全部通过后，在相同输入、dtype、warmup、repeat、设备和并发条件下测量性能。报告 kernel latency、有效 GM 带宽、UB 占用、stage 数和基线差异。

## 可执行代码与证据

可执行结构见 [Vector Add](../../../../../examples/elementwise/elementwise_add.py) 和 [流水版本](../../../../../examples/elementwise/elementwise_add_pipeline.py)；精确融合语义见 [A2/A3 Elementwise 补充](guide.md)。
