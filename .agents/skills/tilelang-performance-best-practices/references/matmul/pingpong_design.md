# GEMM 多级流水

## 适用条件

K tile 至少两轮，MTE2、MTE1 和 Cube 之间存在可重叠空间。

### 候选成立条件

已有证据表明搬运和计算未充分重叠；存在至少两份独立可推进的数据；同时活跃的槽
满足 [容量预算](tiling_design.md)。只有双份 buffer 而没有独立任务或完整归还顺序，
不会自动形成有效流水。

显式 MMA 路径的数据流是：

```text
GM --MTE2--> L1 --MTE1--> L0A/L0B --M--> L0C --FIX--> GM
```

## TileLang 实现流程

A/B L1 buffer 先设计两份物理槽位，依据当前模式使用已支持的 T.Pipelined 或 Expert 手动 ownership。若显式 L0A/L0B 子块，内层同样进行容量约束和生产/消费/复用证明；不机械嵌套未经验证的 Pipelined。Expert 新源按当前 LOCK DSL 生成同步，现有 low-level/legacy event 仍须完整配对。

基础实现参考 [当前 GEMM 示例](../../../../../examples/gemm/example_gemm.py) 与 [显式 MMA 示例](../../../../../examples/gemm/example_gemm_intrinsic.py)：按当前 B 的物理布局设置 transpose_B，使用 T.gemm_v0 或 T.mma；浮点 GEMM 的 L0C 使用 fp32，并且只在第一个有效 K tile 设置 init=True。输出 tile 的核数不超过独立任务数；M/N/K 尾块必须走已验证的 padded-copy 或专用 fallback。

### 按物理存储建立所有权

| 物理区域 | 生产者→消费者就绪 | 消费者→下一生产者归还 | 归还时点 |
|---|---|---|---|
| 每个 L1 slot | MTE2→MTE1 | MTE1→MTE2 | 最后一次读取该 L1 数据的 MTE1 完成 |
| 每个 L0A/B slot | MTE1→M | M→MTE1 | MMA 最后读取该操作数之后 |
| 每个 L0C slot | M→FIX | FIX→M | FIX 导出完成，下一输出 tile 才可覆盖 |

每个槽要有初始可用 token、生产、消费、归还、退出 drain；没有使用的初始槽也要
按同步合同收尾。`T.annotate_address` 造成重叠的别名属于同一所有权域，换 buffer
变量名或 flag ID 不会隔离同一物理地址。

新 Expert 手写入口按
[编程模式指南](../../../tilelang-custom-skill/tilelang-programming-model-guide/SKILL.md)
采用 LOCK DSL；底层 flag 用于阅读 legacy / generated 代码时核对。完整的生成源示例可见
[FA expert_v3/kernel.tl](../../../../../examples/flash_attention/fa_opt/expert_v3/kernel.tl)。
不要把下表方向机械变成缺少初始/终止配对的片段。

### 两级缓冲如何组合

- L1 的 `S1` 控制 MTE2 与 MTE1 重叠；L0 的 `S2` 控制 MTE1 与 MMA 重叠，独立预算。
- 旧常见候选为 `S1=2`、`S2=2`；L1 2–4 槽只可作搜索种子，不能承诺增加必然获益。
- `K_L1>block_K` 提供同一 L1 tile 内多个 L0 消费。若相等，内层没有多次轮转，
  应检查跨外层 K 迭代是否仍存在可重叠工作，而非断言全部双缓冲无效。
- 保持单 L0C 跨所有 K 块累加时，仅首个有效 K 子块 `init=True`；不能把每个
  ping-pong 槽当成独立输出后漏掉归并。
- 若 L0C 多槽承载独立输出，各槽分别等待 FIX 归还；容量中必须乘以槽数。

旧 [intrinsic 示例](../../../../../examples/gemm/example_gemm_intrinsic.py) 使用
`128×256` 输出、`block_K=64`、`K_L1=256` 和两级双槽。代码以 `kk==3` 释放 L1，
仅对应四次 L0 子块读取。更改 K 比例时，方案必须改为“最后一次 MTE1 读取后归还”，
同时考虑尾 K 块；不能原样搬用这个常量。

### 自动处理与手动处理的边界

gemm_v0 的内部同步不等于外围 MTE2↔MTE1 已自动重叠。自动同步是否插入了阻塞点，
要从当前生成代码判断。若选用手动 pipeline，应说明谁负责每条依赖及 buffer 复用，
按现有模式合同调整相关 pass；不得同时删去自动同步和缺少手动保护。

旧记录中“整体双缓冲优于提前一轮 L0 预取”是该组实验的结果，见
[历史案例](historical_cases.md)。同一次迭代的 copy→wait→mma 保证依赖，并不单凭
这一行顺序就能证明跨迭代没有 overlap；需结合不同物理槽和异步管线判断。

## 精度门禁

init 条件必须同时考虑外层 kt 与内层 sk 的首轮。

所有输入 dtype 与输出 dtype 分开测试。覆盖 M/N/K 的 tile±1、长 K 消除误差、正负抵消、大小量混合、0、NaN/Inf 契约。浮点 GEMM 的任何 K 分片的 partial 和最终归约保持 fp32；不得以放宽容差掩盖累加顺序或输出转换错误。

### 后续验证重点

验证首块、末块、奇数循环次数、仅一块、空闲槽、重复不同输入调用，以及所有受影响
的 dtype/尾块。比较总耗时和暴露等待，不以分配了两个槽或指令数减少作为成功标准。
该调度的具体实现与收益待验证。

## 性能门禁

先测 2 stage，再测 3 stage；增加 stage 后若 L1 压力降低 occupancy，则保留较浅流水。

对每个候选运行编译、目标后端的定向精度测试和统一 benchmark。报告 latency、TFLOPS、Cube/MTE2 时间、GM 字节、L1/L0/UB 占用、核利用率和生成代码体积。只有完整算子端到端更快且没有精度回退时进入 dispatch。
