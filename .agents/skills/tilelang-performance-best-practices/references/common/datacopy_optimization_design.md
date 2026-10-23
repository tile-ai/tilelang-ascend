# T.copy 搬运与访存合并

## 目标与适用条件

用于 GM 到 UB/L1、L1 到 L0 以及当前后端支持的反向写回。目标是减少搬运次数、扩大连续 burst，并让搬运与计算重叠。

## TileLang Ascend 实现

优先复制连续矩形切片；尾部使用精确 valid slice 或 T.npu_copy_v2 的 pad_value。重复读取的权重、scale 和查找表在 Persistent 循环外搬入单版本 buffer。对 strided 输入使用当前仓库支持的 stride/BufferRegion 表达；当前 copy API 不提供源平台的 l2_cache_ctrl 参数，不能直接传入；缓存控制只有当前 API/lowering 和目标示例已经证明时才能进入正式路径。

相邻短记录或标量应沿外层维合并为矩形/连续搬运，避免每个 token 产生 4B GM 访问；同时核对 nburst、burst_len、无效字节和 UB 占用。

实现沿用当前仓库 T.Kernel 的网格和 `(cid, vid)` 映射。纯向量任务从已确认的可用 AIV 核数及 launch 映射取核数，并以 min(核数, 独立任务数) 限制空核。GM 与 UB/L1 之间使用 T.copy；连续规则计算使用当前模式的 T.Parallel 或 T.tile 原语；跨 tile 任务使用当前 T.Persistent、Fixed Core 核内循环或已验证的 T.Pipelined，多版本方式沿用当前编译器支持的流水或显式槽位所有权。

## 精度门禁

低精度输入在 UB 中按算子语义提升为 fp32；pad_value 必须是该运算的幺元，例如 max 用负无穷、sum 用 0。禁止读取未初始化 padding lane。

必须覆盖最小 shape、常见 shape、最大 shape、tile-1/tile/tile+1、非 32B 尾块、0、正负极值以及接口规定的 NaN/Inf 语义。不得通过扩大容差、降低 reference 精度或跳过 case 获得通过。

## 性能门禁

比较 GM 总字节、copy 指令数、MTE2/MTE3 时间和有效带宽。合并搬运只有在没有增加无效数据量或 UB 压力时保留。

先用目标仓库既有后端与命令运行定向精度测试，再运行相关完整测试套；全部通过后，在相同输入、dtype、warmup、repeat、设备和并发条件下测量性能。报告 kernel latency、有效 GM 带宽、UB 占用、stage 数和基线差异。

## 可执行代码与证据

参考 [尾块测试](../../../../../testing/python/language/test_tilelang_ascend_language_tail_block.py)、[Vector Add](../../../../../examples/elementwise/elementwise_add.py) 和 [GEMM 示例目录](../../../../../examples/gemm/)。

---

## A2/A3 实现补充与约束

## 搬运合并、稀疏访问与流水层级

### 优化对象

从实际数据流统计 GM 读写、DMA 次数、连续 burst、stride/gap、无效字节、每核初始化与复用次数。只有 profiling 支持时才判定搬运受限。GM→L1/UB、L1→L0A/L0B、L0C 输出属于不同层级；L1→L0 是 MTE1 路径，不能将它误称为 MTE2 预取。

依据 [搬运 DSL](../../../../../tilelang/language/copy_op.py)、[Ascend 模板](../../../../../src/tl_templates/ascend/common.h) 与同类实现确认合法层级、地址、长度、gap、slice 和 padding；不从单条 copy 的简短写法推断真实访存量。

### 连续小块合并

相邻 scale、bias、LUT、小记录或多个输出行若在物理布局连续，可一次搬入多个迭代所需数据，按当前迭代偏移消费。收益来自减少 DMA 固定成本；代价是额外驻留、padding、可能降低 stage 数。

候选用 `merge_count × record_bytes` 与完整生命周期建模。旧“20 KB 以下必须合并”“带宽从 50%–70% 到 80%”仅是历史启发，不能作为通用阈值或收益承诺。合并倍率 2/4/8 只是搜索候选，必须覆盖末次不足倍率的有效范围。K 方向 scale 必须与正确 operand block 对应，普通 GEMM API 不因此获得量化 scale 语义。

### 离散 KV / Gather

Paged/Sparse Attention 中可先沿索引表把多个合法记录搬到连续 UB，再批量写 workspace，减少小写回与重复同步。输入仍可能是多次离散 DMA，应分别计算读端和写端收益。

- 保留 `index >= 0`、页表范围、页内 offset 和尾记录检查；非法索引对应的结果按接口填充，不能为去分支而访问无效地址。
- `gather_ub[records, feature]` 是批处理 buffer，**不等于双缓冲**。只有存在两个物理槽位且所有权交替，才称 ping-pong。
- `D` 与 `D_tail` 可独立搬运；每部分必须满足当前 DMA 约束，不能把短记录一律拆成 4B/8B GM 事务。
- workspace 连续写回前等待全部生产者完成；减少全局 barrier 时仍保留真正的 producer/consumer 依赖。
- 比较 record 聚合与块 DMA + UB-local 重排，纳入索引准备、无效读、scratch 和所有 launch。

### 核内 K 分块与预取

旧指南的“Split-K”主要指同一输出的 K 子块切分：解决 L0 容量并尝试 MTE1/Cube overlap，不是跨核 split-K。前者通常由同一 owner 累加 L0C；后者要产生 partial 并合并，必须计入 workspace、额外 launch 和累加顺序变化。Cube 具体策略见 [MatMul](../matmul/guide.md)。

提前搬入下一 K tile 要区分 GM→L1 预取与 L1→L0 预取；先验证首轮、稳态、尾轮及 buffer 释放，再用生成代码证明发射前移。不能只把循环改名或多分配 buffer 就报告 overlap。详见 [流水与 CV](pipeline_and_cv_overlap.md)。

### 验证

矩形搬运是否可用由真实 stride 和 lowering 决定；行之间有 gap 并不一律非法，reshape 也不会改变原物理 stride。用唯一编号输入检查位置对应、跨 batch 与双尾块；验证总有效字节、DMA 次数、MTE 时间和完整算子 latency。
