# UB/L1 常驻与多版本缓冲

## 目标与适用条件

用于跨多个 tile 重复使用的权重、scale、查找表和 reduction 状态。

## TileLang Ascend 实现

常驻数据在 T.Persistent/T.Pipelined 外分配并只搬一次，不随流式 tile 扩展多版本。输入输出 tile 按 num_stages 多版本；reduction carry、online softmax 的 max/sum 和 scan carry 保持单版本。预算为所有 buffer 元素数乘 dtype 字节数与版本数，再加 padding 和安全余量。

实现沿用当前仓库 T.Kernel 的网格和 `(cid, vid)` 映射。纯向量任务从已确认的可用 AIV 核数及 launch 映射取核数，并以 min(核数, 独立任务数) 限制空核。GM 与 UB/L1 之间使用 T.copy；连续规则计算使用当前模式的 T.Parallel 或 T.tile 原语；跨 tile 任务使用当前 T.Persistent、Fixed Core 核内循环或已验证的 T.Pipelined，多版本方式沿用当前编译器支持的流水或显式槽位所有权。

## 精度门禁

常驻低精度权重可保持原 dtype；统计状态与长链累加保持 fp32。确认每个 task 不会读到上一 task 的残留状态。

必须覆盖最小 shape、常见 shape、最大 shape、tile-1/tile/tile+1、非 32B 尾块、0、正负极值以及接口规定的 NaN/Inf 语义。不得通过扩大容差、降低 reference 精度或跳过 case 获得通过。

## 性能门禁

A/B 测试重复 GM 流量、UB 占用、occupancy 和流水重叠。常驻导致 stage 数下降或 spill 时，比较总延迟后再取舍。

先用目标仓库既有后端与命令运行定向精度测试，再运行相关完整测试套；全部通过后，在相同输入、dtype、warmup、repeat、设备和并发条件下测量性能。报告 kernel latency、有效 GM 带宽、UB 占用、stage 数和基线差异。

## 可执行代码与证据

参考当前仓库同类实现中的权重常驻，以及 [online softmax](../../../../../examples/softmax/example_online_softmax.py) 中的 fp32 online 状态。

---

## A2/A3 实现补充与约束

## UB/L1 常驻与状态生命周期

### 适用条件

权重、scale、LUT、索引或统计状态跨多个 tile 复用，且常驻量与流式 buffer、scratch 和必要版本可同时放入有效容量。小侧 GEMM panel 可放 L1；Vector 参数和 scan/online-softmax 状态可放 UB。不要因“数据小”就默认常驻更快。

把常量加载和不变量计算移到真正的复用循环外；若每核都要初始化，统计 `used_cores × resident_bytes` 和预处理指令成本。较少核数、多 task/core 可能更合算，需与 [任务映射](task_mapping.md) 联合比较。不同 task 共享的是只读常量；可变累加状态必须在每个独立输出开始时重置。

### Buffer 版本

- 只读常驻量通常单份，不随 stage 复制。
- scan carry、online max/sum、固定小状态属于递归依赖，按依赖顺序更新，不因流水盲目双份。
- 只有同一物理 buffer 的当前值仍被异步 consumer 读取、后续迭代就要覆盖时才需要多个版本或额外等待。
- V pipe 中间量可单份的前提是所有 consumer 都在下一次覆盖前完成，且同步/指令 hazard 已处理；“只在 V pipe”本身不是普遍安全证明。
- L0C 是否多版本由输出生命周期决定；同一输出的 K 分块必须保留累加，不能因 K 奇偶切换 L0C 而遗漏 partial。

用 [资源账本](hardware_resource_discovery.md) 比较节省的 GM 流量、一次初始化成本、生命周期延长、stage 减少与寄存器/UB 压力。GEMM full-load 若复用次数很少，或占满 L1 后损失流水，可能无收益。

### 验证

常驻参数的加载/广播完成后再消费，避免第二遍输入搬运覆盖其 storage。先检查内存规划与定向依赖，不能将“广播后插入 barrier_all”当作通用修复。状态测试覆盖多个 task 连续运行、改变 case 顺序、多个 stage 和尾块；full-load 比较完整算子，不只计单侧省掉的 copy。
