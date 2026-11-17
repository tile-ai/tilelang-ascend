# TileLang msprof op CSV 字段完整参考

用于分析真实 NPU 上运行的 TileLang kernel。不同 CANN/MindStudio 版本、设备和
`--aic-metrics` 组合可能输出不同文件或字段；分析器必须先探测当前产物 schema，
不能假设所有指标始终存在。

分析前必须先按 `expected_kernel` 筛选当前入口中的目标 kernel，排除输入构造、数据转换和其他辅助 kernel。字段不存在、值为空或为 `N/A` 时跳过该字段，不得用其他 kernel 的值补齐。完整诊断至少要求 `OpBasicInfo.csv` 与 `PipeUtilization.csv`；其余 CSV 缺失时在摘要中标记，不得伪造。

> 字段名中的 `ai*` 表示 `aic`（Cube Core）或 `aiv`（Vector Core）。先根据目标 kernel 的实际运行记录判定 AIV-only、AIC-only 或混合 AIC/AIV。

## 目录

1. [OpBasicInfo.csv](#1-opbasicinfocsv)
2. [PipeUtilization.csv](#2-pipeutilizationcsv)
3. [ArithmeticUtilization.csv](#3-arithmeticutilizationcsv)
4. [Memory.csv](#4-memorycsv)
5. [MemoryL0.csv](#5-memoryl0csv)
6. [MemoryUB.csv](#6-memoryubcsv)
7. [L2Cache.csv](#7-l2cachecsv)
8. [ResourceConflictRatio.csv](#8-resourceconflictratiocsv)
9. [TileLang 联合诊断顺序](#9-tilelang-联合诊断顺序)

---

## 1. OpBasicInfo.csv

首先使用该文件确认目标 kernel、总体耗时、block 数和运行频率。

| 字段名 | 含义 | TileLang 关注点 |
|---|---|---|
| Op Name | 运行时 kernel 名称 | 必须与当前 TileLang 源码中的目标 kernel 唯一对应 |
| Op Type | kernel 类型 | 用于区分 AIV、AIC 或混合 kernel |
| Task Duration(us) | Task 总耗时，包含调度、执行和响应 | 用于判断启动/响应开销；逐方案统一 `kernel_time` 仍按目标 aiv/aic 关键路径计算 |
| Block Dim | Task 逻辑 block 数 | 对照 `T.Kernel(num_blocks)`、有效 Tile 数和实际设备核数 |
| Mix Block Dim | 混合 kernel 的从核 blockDim，非混合 kernel 为 N/A | 检查混合 AIC/AIV 的核配比 |
| Device ID | NPU 设备编号 | 保证 baseline 与各 variant 在同一设备口径下采集 |
| PID | 进程号 | 校验一次进程只运行一个 case |
| Current Freq | 当前运行频率 | 与 Rated Freq 对比，识别动态降频 |
| Rated Freq | 额定频率 | Current < Rated 时不同轮次可能不可比 |

### 对应 TileLang 结构

- `Block Dim`：检查 `with T.Kernel(num_blocks)` 的 block 数是否与输出 Tile 数匹配。
- 混合 kernel：同时保留 aic/aiv 原始时间，用关键路径而不是简单相加作为 `kernel_time`。
- 小 shape：同时比较 `Task Duration` 与核内 `aiv_time/aic_time`，二者差值较大时优先检查启动和控制开销。

---

## 2. PipeUtilization.csv

最重要的瓶颈定位文件。每一行是由 `block_id` 和 `sub_block_id` 标识的 profiler
核/子核记录。`ai*_time(us) > 0` 只表示该行有设备侧计时；启动、同步、flag 管理和
编译器生成框架也可能产生非零时间，不能据此认定该核承担了有效业务 Tile。

### 公共字段

| 字段名 | 含义 |
|---|---|
| block_id | 目标 Task 的逻辑 block 标识 |
| sub_block_id | block 内的 Vector/Cube 子核名称和序号 |
| aic_time(us) | Cube Core 执行时间 |
| aic_total_cycles | Cube Core 总 cycle 数 |
| aiv_time(us) | Vector Core 执行时间 |
| aiv_total_cycles | Vector Core 总 cycle 数 |

### 流水线单元耗时和占比

表中的比例用于定位候选瓶颈，必须同时检查对应流水的绝对耗时、实际工作量和源码路径。
小 workload、单 block 或尾块场景不能只按比例阈值下结论。

| 字段名 | 含义 | 瓶颈判定阈值 |
|---|---|---|
| aiv_vec_time(us) | Vector 指令耗时 | — |
| aiv_vec_ratio | Vector 指令 cycle 占比 | >50% 时重点检查 VEC Bound |
| aic_cube_time(us) | Cube 指令耗时 | — |
| aic_cube_ratio | Cube 指令 cycle 占比 | MatMul 中通常占主导；纯 Vector kernel 应接近 0 |
| ai*_scalar_time(us) | Scalar 指令耗时 | — |
| ai*_scalar_ratio | Scalar 指令 cycle 占比 | >30% 时重点检查 SCALAR Bound |
| aic_fixpipe_time(us) | FixPipe（L0C→GM/L1）耗时 | — |
| aic_fixpipe_ratio | FixPipe cycle 占比 | >15% 时检查输出 Tile、路径和地址对齐 |
| aic_mte1_time(us) | MTE1（L1→L0A/L0B）耗时，不含等待 | — |
| aic_mte1_ratio | MTE1 cycle 占比 | — |
| ai*_mte2_time(us) | MTE2 搬入耗时 | — |
| ai*_mte2_ratio | MTE2 cycle 占比 | >50% 时重点检查搬入瓶颈 |
| ai*_mte3_time(us) | MTE3 搬出耗时 | — |
| ai*_mte3_ratio | MTE3 cycle 占比 | 与 MTE2 一起判断双向 GM 搬运压力 |
| ai*_icache_miss_rate | ICache miss 率 | >15% 时检查分支和生成代码规模 |

### 活跃带宽（仅 A2/A3，是否输出取决于当前工具参数）

| 字段名 | 含义 |
|---|---|
| aiv_mte2_active_bw(GB/s) | Vector 核 MTE2 活跃带宽 |
| aiv_mte3_active_bw(GB/s) | Vector 核 MTE3 活跃带宽 |
| aic_mte1_active_bw(GB/s) | Cube 核 MTE1 活跃带宽，需要 MemoryDetail |
| aic_mte2_active_bw(GB/s) | Cube 核 MTE2 活跃带宽，需要 MemoryDetail |
| aic_mte3_active_bw(GB/s) | Cube 核 MTE3 活跃带宽 |
| aic_fixpipe_active_bw(GB/s) | Cube 核 FixPipe 活跃带宽 |

### 对应 TileLang 结构

| profiling 现象 | 优先检查的 TileLang 结构 |
|---|---|
| VEC 高 | `T.Parallel`、`T.tile.*`、UB 复用、Cast 和逐元素融合 |
| CUBE 高 | `T.gemm_v0`/`T.mma`、`T.alloc_L1`、`T.alloc_L0C`、矩阵 Tile shape |
| SCALAR 高 | 设备端动态分支、`T.serial`、内层索引计算和动态 Tensor 标量读取 |
| MTE2/MTE3 高 | `T.copy` 次数、连续 slice、Tile 大小和 GM 往返 |
| 流水串行 | `T.Pipelined(num_stages=...)`、当前编程模式的 buffer 规划和跨迭代依赖 |
| 承担可比任务的核时间差异大 | `T.Kernel` block 数、`T.ceildiv` 尾块切分和 `T.Persistent` 调度 |

---

## 3. ArithmeticUtilization.csv

查看 Cube 和 Vector 指令类型、指令数和计算量。

### Cube 指令字段

| 字段名 | 含义 | TileLang 关联 |
|---|---|---|
| aic_cube_ratio | Cube 指令 cycle 占比 | `T.gemm` 是否为主路径 |
| aic_cube_fp16_ratio | Cube fp16 指令占比 | 输入 dtype 与 `T.gemm` 计算路径 |
| aic_cube_int8_ratio | Cube int8 指令占比 | 量化矩阵计算路径 |
| aic_cube_fops | Cube 浮点运算总数 | 与理论 FLOPS 比较计算利用率 |
| aic_cube_total_instr_number | Cube 指令总条数 | Tile 太碎时指令数可能过高 |
| aic_cube_fp_instr_number | Cube 浮点指令条数 | — |
| aic_cube_int_instr_number | Cube 整数指令条数 | — |

### Vector 指令字段

| 字段名 | 含义 | TileLang 关联 |
|---|---|---|
| aiv_vec_ratio | Vector 指令 cycle 占比 | `T.Parallel` 主计算是否成为瓶颈 |
| aiv_vec_fp32_ratio | Vector fp32 指令占比 | 检查是否存在不必要的 fp32 Cast/计算 |
| aiv_vec_fp16_ratio | Vector fp16 指令占比 | 与预期计算 dtype 对比 |
| aiv_vec_int32_ratio | Vector int32 指令占比 | 检查索引和整数运算占比 |
| aiv_vec_int16_ratio | Vector int16 指令占比 | — |
| aiv_vec_misc_ratio | Vector misc 指令占比 | 高时检查特殊函数、类型转换和复杂控制 |
| aiv_vec_fops | Vector 浮点运算总数 | 与算法预期 FLOPS 对比，识别重复计算 |

### TileLang 诊断

- fp32 ratio 远高于预期：定位 `T.cast`、fragment dtype 和归约累加 dtype。
- 指令数高但有效 FLOPS 低：合并多次 `T.Parallel` 遍历，复用 `T.alloc_fragment` 中间值。
- Reduction scalar 高：优先检查 `T.reduce_sum/max/min` 是否适用，避免可替代的 `T.serial` 累加。
- CUBE 指令数过多：增大 K Tile 或输出 Tile，但必须同步检查 L1/L0 容量和并行度。

---

## 4. Memory.csv

查看内存带宽、搬运指令数和数据量。

### 带宽速率

| 字段名 | 含义 |
|---|---|
| aiv_gm_to_ub_bw(GB/s) | GM→UB 带宽 |
| aiv_ub_to_gm_bw(GB/s) | UB→GM 带宽 |
| aic_l1_read_bw(GB/s) | L1 读带宽 |
| aic_l1_write_bw(GB/s) | L1 写带宽 |
| ai*_main_mem_read_bw(GB/s) | 主存读带宽 |
| ai*_main_mem_write_bw(GB/s) | 主存写带宽 |

### 指令统计

| 字段名 | 含义 |
|---|---|
| aic_mte1_instructions | MTE1 指令数 |
| aic_mte1_ratio | MTE1 cycle 占比 |
| ai*_mte2_instructions | MTE2 指令数 |
| ai*_mte2_ratio | MTE2 cycle 占比 |
| ai*_mte3_instructions | MTE3 指令数 |
| ai*_mte3_ratio | MTE3 cycle 占比 |

### 数据搬运量

| 字段名 | 含义 |
|---|---|
| read_main_memory_datas(KB) | 读主存总量 |
| write_main_memory_datas(KB) | 写主存总量 |
| GM_to_L1_datas(KB) | GM→L1 搬运量 |
| L1_to_GM_datas(KB)(estimate) | L1→GM 搬运量（估算） |
| L0C_to_L1_datas(KB) | L0C→L1 搬运量 |
| L0C_to_GM_datas(KB) | L0C→GM 搬运量 |
| GM_to_UB_datas(KB) | GM→UB 搬运量 |
| UB_to_GM_datas(KB) | UB→GM 搬运量 |

### 带宽利用率

| 字段名 | 含义 | 参考标准 |
|---|---|---|
| GM_to_L1_bw_usage_rate(%) | GM→L1 带宽利用率 | 搬运量足够且 GM 位于关键路径时，>60% 通常较好 |
| L1_to_GM_bw_usage_rate(%)(estimate) | L1→GM 带宽利用率 | 搬运量足够且 GM 位于关键路径时，>60% 通常较好 |
| L0C_to_L1_bw_usage_rate(%) | L0C→L1 带宽利用率 | 与 FixPipe 路径联合分析 |
| L0C_to_GM_bw_usage_rate(%) | L0C→GM 带宽利用率 | 与输出路径联合分析 |
| GM_to_UB_bw_usage_rate(%) | GM→UB 带宽利用率 | 搬运量足够且 GM 位于关键路径时，>60% 通常较好 |
| UB_to_GM_bw_usage_rate(%) | UB→GM 带宽利用率 | 搬运量足够且 GM 位于关键路径时，>60% 通常较好 |

小数据量或固定开销占主导时，低带宽利用率不代表带宽实现存在问题。只有 MTE 绝对耗时
显著、搬运量足够且 GM 位于关键路径时，才用上述比例评价搬运效率。

### 对应 TileLang 结构

- GM↔UB：`T.copy` 与 `T.alloc_shared`。
- GM↔L1：`T.copy` 与 Developer 的 `T.alloc_shared` 或 Expert 的 `T.alloc_L1`。
- L1/L0：`T.alloc_L1`、`T.alloc_L0A/L0B/L0C` 和 `T.gemm_v0`/`T.mma`。
- GM 字节数高于算法必需量：检查中间结果是否可以保留在 shared/fragment 中并融合计算。
- 搬运指令很多但总数据量不大：增大连续 Tile，减少零碎 `T.copy`。

理论搬运耗时统一使用：

```text
理论耗时(us) = 搬运字节数 / 当前设备可用带宽(Byte/s) × 1e6
```

---

## 5. MemoryL0.csv

L0A/L0B/L0C 带宽，主要分析使用 `T.gemm` 的 TileLang kernel。

| 字段名 | 含义 | TileLang 关注点 |
|---|---|---|
| aic_l0a_read_bw(GB/s) | L0A 读带宽 | A Tile 的 L0 访问 |
| aic_l0a_write_bw(GB/s) | L0A 写带宽 | `T.alloc_L0A`/自动 L0A 路径 |
| aic_l0b_read_bw(GB/s) | L0B 读带宽 | B Tile 的 L0 访问 |
| aic_l0b_write_bw(GB/s) | L0B 写带宽 | `T.alloc_L0B`/自动 L0B 路径 |
| aic_l0c_read_bw_cube(GB/s) | Cube 从 L0C 读带宽 | 跨 K Tile 累加和输出路径 |
| aic_l0c_write_bw_cube(GB/s) | Cube 向 L0C 写带宽 | `T.alloc_L0C` 和 `T.gemm_v0`/`T.mma` 累加 |

L0 带宽异常时联合检查 `TILE_M/N/K`、`clear_accum`、K 循环 `T.Pipelined` 和 L1→L0 copy 次数。

---

## 6. MemoryUB.csv

Vector 和 Scalar 对 UB 的读写带宽。

| 字段名 | 含义 | TileLang 关注点 |
|---|---|---|
| aiv_ub_read_bw_vector(GB/s) | Vector 从 UB 读带宽 | `T.alloc_shared` 是否被重复完整读取 |
| aiv_ub_write_bw_vector(GB/s) | Vector 向 UB 写带宽 | 中间结果是否产生多次 UB 写入 |
| aiv_ub_read_bw_scalar(GB/s) | Scalar 从 UB 读带宽 | 内层动态标量访问是否过多 |
| aiv_ub_write_bw_scalar(GB/s) | Scalar 向 UB 写带宽 | 标量循环和小粒度写入是否过多 |

Vector UB 带宽高且 VEC 耗时高时，优先尝试 `T.alloc_fragment` 复用；Scalar UB 带宽高时，减少内层 Tensor 标量读取和 `T.serial` 循环。

---

## 7. L2Cache.csv

查看目标 kernel 的 L2 命中次数和命中率。

| 字段名 | 含义 | 参考标准 |
|---|---|---|
| ai*_write_cache_hit | 写 cache 命中次数 | — |
| ai*_write_cache_miss_allocate | 写 cache miss 后分配次数 | — |
| ai*_r*_read_cache_hit | 各读通道 cache 命中次数 | — |
| ai*_r*_read_cache_miss_allocate | 各读通道 miss 后分配次数 | — |
| ai*_write_hit_rate(%) | 写 cache 命中率 | 存在写复用预期时，>80% 通常较好 |
| ai*_read_hit_rate(%) | 读 cache 命中率 | 存在读复用预期时，>80% 通常较好 |
| ai*_total_hit_rate(%) | 总命中率 | 存在复用预期且 GM/MTE 位于关键路径时，<50% 才重点检查 |

### 对应 TileLang 结构

1. 先检查 `T.Persistent` 输出 Tile 顺序和数据局部性。
2. 再根据数据复用方式对 `T.copy(..., l2_cache_ctrl=...)` 做逐输入 A/B 测试。
3. 权重等复用数据与流式输入/输出可能需要不同策略。
4. 命中率必须与 GM 总字节数和 `kernel_time` 联合判断，不能单独作为优化目标。

---

## 8. ResourceConflictRatio.csv

查看 UB bank group、bank conflict、资源冲突和等待比例。该文件由上板采集产生，仿真结果可能不包含。

### 核心冲突指标

| 字段名 | 含义 | 参考标准 |
|---|---|---|
| aiv_vec_total_cflt_ratio | Vector 指令总阻塞占比 | <5% 通常良好，>15% 严重 |
| aiv_vec_bankgroup_cflt_ratio | bank group 冲突阻塞占比 | <3% |
| aiv_vec_bank_cflt_ratio | bank 冲突阻塞占比 | <3% |
| aiv_vec_resc_cflt_ratio | 计算单元资源冲突占比 | <5% |
| aiv_vec_mte_cflt_ratio | Vector/MTE 冲突占比 | <3% |

### 等待指标

| 字段名 | 含义 |
|---|---|
| aic_cube_wait_ratio | Cube 单元等待占比 |
| aiv_vec_wait_ratio | Vector 单元等待占比 |
| ai*_mte1_wait_ratio | MTE1 等待占比 |
| ai*_mte2_wait_ratio | MTE2 等待占比 |
| ai*_mte3_wait_ratio | MTE3 等待占比 |

### TileLang 优化映射

| 冲突类型 | 优先修改 |
|---|---|
| bankgroup_cflt 高 | 调整 `T.alloc_shared` shape、行 stride、padding 和 `T.Parallel` 索引映射 |
| bank_cflt 高 | 改变不同 UB operand 的起始偏移或行宽，避免同周期集中访问相同 bank |
| resc_cflt 高 | 调整融合表达式、Vector/Cube 顺序和 pipeline stage，减少同一执行资源争用 |
| mte_cflt 高 | 调整 `T.copy` 位置、`T.Pipelined` stage 和 buffer 版本，避免搬运与计算访问同一版本 |
| wait ratio 高 | 检查真实数据依赖、Tile 粒度、多缓冲版本和不必要的串行循环 |

修改布局后必须重新跑精度测试，防止 padding、slice 或边界索引改变语义。

---

## 9. TileLang 联合诊断顺序

1. 从 `OpBasicInfo.csv` 校验 kernel 名、频率、Task Duration 和 Block Dim。
2. 从 `PipeUtilization.csv` 取得 timed rows、主导流水和 `kernel_time`；结合源码任务分配
   区分启动 block、timed subcore 与 task-bearing block 后，再判断核间差异。
3. 从 `ArithmeticUtilization.csv` 判断有效计算、Cast、Vector/Cube 指令结构。
4. 从 `Memory.csv` 计算总搬运量、带宽利用率和理论搬运时间。
5. 对 CUBE kernel 读取 `MemoryL0.csv`；对 Vector kernel 读取 `MemoryUB.csv`。
6. 读取 `L2Cache.csv` 和 `ResourceConflictRatio.csv` 验证缓存与冲突假设。
7. 将结论映射到 `T.Kernel`、Tile shape、`T.copy`、buffer scope、`T.Pipelined`、`T.Persistent`、`T.Parallel` 或 `T.gemm` 的具体修改。
8. 完成 pytest 相关完整测试套精度回归后，按同一 case 和同一计时口径重新采集。
