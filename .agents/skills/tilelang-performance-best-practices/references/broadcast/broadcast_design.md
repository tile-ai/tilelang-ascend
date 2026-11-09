# Broadcast 总体实现

## 目标与适用条件

用于标量、单轴和多轴广播的 elementwise 算子。将输出空间展平为独立 tile，并从输出索引计算输入索引。

## TileLang Ascend 实现

按 [当前 broadcast DSL](../../../../../tilelang/language/ascend_tile.py) 和 [API 计算参考](../../../tilelang-custom-skill/tilelang-api-best-practices/references/api-compute.md) 核对支持的扩展轴、rank、dtype、scratch 与 slice。末轴扩展是归一化的常见候选结构，其他扩展轴仍按实际 API 与 lowering 核验。

连续小输入可在任务循环外加载一次；多轴广播先合并逻辑连续维，区分 outer / broadcast / inner。内层连续时可让一个 task 复用源 slice 写多个输出片段；不连续维需显式地址映射，不能假定任意 N 维广播都可一条 DMA。

连续广播轴优先把小输入搬入 UB 一次，然后在 Vector 计算中复用；标量使用当前已验证的 fill/broadcast 原语。复杂不连续索引按当前 DSL 显式计算，并通过 Python factory 固化 rank、axis 和 stride。输出任务由当前 T.Persistent 分配，并核对其 domain/group_size 约束与 Kernel 映射。

实现沿用当前仓库 T.Kernel 的网格和 `(cid, vid)` 映射。纯向量任务从已确认的可用 AIV 核数及 launch 映射取核数，并以 min(核数, 独立任务数) 限制空核。GM 与 UB/L1 之间使用 T.copy；连续规则计算使用当前模式的 T.Parallel 或 T.tile 原语；跨 tile 任务使用当前 T.Persistent、Fixed Core 核内循环或已验证的 T.Pipelined，多版本方式沿用当前编译器支持的流水或显式槽位所有权。

### 统计量与 GM 参数 staging

| 来源 | 候选数据流 | 注意 |
|---|---|---|
| 所有行共享的统计标量 | `[1] → fill [rows,1] → broadcast [rows,block_S]` | rows 为真实有效行数 |
| 每行独立统计量 | `[rows] / [rows,1] → 合法形状整理 → broadcast` | 按当前 keepdim/shape 规则，不默认 copy 跨 rank |
| GM per-row gamma/beta | 有效 1D slice → aligned staging → cast → padded broadcast → 有效行 | rows_padded 与 rows 区分 |

GroupNorm 参数可采用以下五步数据流：

1. host/签名将参数表达为 `[G, rows]`，对当前组搬 `param[g,0:rows]`；避免误用二维 slice 改变 DMA 解释。
2. staging 为 `[rows_padded,1]`，按实际 dtype/cast 对齐分配。需要预填充时 fill 在 copy 前；使用 `T.npu_copy_v2(..., pad_value=...)` 时，其填充值须与预填充值一致或具有明确覆盖范围。`T.copy` 不接受 `pad_value` 参数。
3. cast 使用完整、已初始化且满足指令约束的 `rows_padded` 范围，低精度统计/计算通常提升 fp32。
4. 广播为 `[rows_padded,block_S]`，预算包含完整临时 buffer。
5. 仅保留/消费有效 `[rows,block_S]` 区域，循环外常驻。`gamma_bc_full` 只有在最后 consumer 完成后才可复用。

`rows_padded=ceildiv(rows,16)*16` 是 fp16/bf16 参数路径的示例，不能代替所有 dtype 的约束。`T.tile.fill` 从 scalar buffer 传值与 `T.copy` 重新解释 shape 不等价；出现 `StructuralEqual` 或 broadcast mismatch 时查 API，不机械换成跨 rank copy。

常驻参数保持单版本；流式输入/输出按 [生命周期](../common/ub_resident_design.md) 决定版本。参数广播与后续输入 copy 的 storage 若发生重叠，先修规划/依赖，不以每轮全局 barrier 掩盖。

## 精度门禁

广播前后的 dtype 转换顺序必须与 reference 一致；低精度超越函数与累加提升 fp32。验证 size=1 轴、多个广播轴、零维标量和尾块。

必须覆盖最小 shape、常见 shape、最大 shape、tile-1/tile/tile+1、非 32B 尾块、0、正负极值以及接口规定的 NaN/Inf 语义。不得通过扩大容差、降低 reference 精度或跳过 case 获得通过。


覆盖 size=1 轴、多轴、标量、rows 对齐边界、最后一组、不同 broadcast 倍率、所有 dtype 和允许 stride。唯一编号参数可检查广播方向，非均匀 gamma/beta 可暴露错位；全 1 参数不足以验证映射。记录生成 DMA、重复读字节、指令数、UB 及完整 latency。

## 性能门禁

同时比较逐行指令、整 tile broadcast、多行批处理与参数驻留。较大 broadcast 倍率可减少重复 GM 读，较小时 staging 固定成本可能更高；每核各自加载参数的成本也要计入。显式 GM padding 与 kernel 内 fill+valid copy 按总流量比较。


重点测 GM 重复读取量、索引计算开销和向量利用率。小广播输入常驻 UB，避免每个输出 tile 重复搬运。

先用目标仓库既有后端与命令运行定向精度测试，再运行相关完整测试套；全部通过后，在相同输入、dtype、warmup、repeat、设备和并发条件下测量性能。报告 kernel latency、有效 GM 带宽、UB 占用、stage 数和基线差异。

## 可执行代码与证据

复用 [Vector Add](../../../../../examples/elementwise/elementwise_add.py) 的基础结构与 [流水版本](../../../../../examples/elementwise/elementwise_add_pipeline.py) 的槽位映射；广播 API 核对 [ascend_tile.py](../../../../../tilelang/language/ascend_tile.py)。

## 专题路由

- 一维任务与广播基线：[一维实现](onedim_design.md)。
- 小输入在 UB 内复用：[UB 广播](ub_broadcast_design.md)。
- 输入填充与尾块：[DataCopyPad](datacopypad_design.md)。
- 多轴合并与矩形搬运：[多维搬运](rectangular_copy_design.md)。
- 多轴 tiling、驻留及 shape 特化：[高级 Tiling](advanced_tiling.md)。
