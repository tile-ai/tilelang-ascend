# TileLang 瓶颈优化速查表

根据 msprof CSV 确认目标 TileLang kernel 的瓶颈后，按本表形成可验证、可操作的 TileLang 修改建议。每次只改变一个主要变量，并使用相同 case 集、相同 kernel 过滤规则和相同计时口径重新采集。

## 目录

1. [VEC Bound](#1-vec-boundvector-计算瓶颈)
2. [MTE2/MTE3 Bound](#2-mte2mte3-bound数据搬运瓶颈)
3. [CUBE Bound](#3-cube-bound矩阵计算瓶颈)
4. [SCALAR Bound 与头开销](#4-scalar-bound-与头开销)
5. [核间负载不均衡](#5-核间负载不均衡)
6. [Bank Conflict](#6-bank-conflict)
7. [流水重叠不足](#7-流水重叠不足)
8. [L2 Cache 命中率低](#8-l2-cache-命中率低)
9. [交叉关联诊断](#9-交叉关联诊断)
10. [仓库实现参考](#10-仓库实现参考)

## 使用门禁

- 只分析与当前算子入口唯一对应的运行时 kernel，排除输入构造和其他辅助 kernel。
- 先判定 kernel 是 AIV-only、AIC-only 还是混合 AIC/AIV，再选择对应字段。
- 修改 `T.Kernel` block 数、Tile shape、`num_stages`、buffer 版本或数据布局后，先运行原 pytest 相关完整测试套精度回归，再重新采集全部性能 case。
- Tile 大小、buffer 数量和 pipeline stage 必须满足目标设备 UB/L1/L0 容量；编译失败或资源超限时回退该参数。

---

## 1. VEC Bound（Vector 计算瓶颈）

**判定**：目标 kernel 的 `aiv_vec_ratio` 最高，且 Vector 实际耗时显著高于其他流水。

### TileLang 优化顺序

| # | 方法 | TileLang 实现 | 适用信号 |
|---|---|---|---|
| 1 | 融合中间结果 | 使用 `T.alloc_shared` 保留 UB 中间值，避免中间结果写回 GM 后再次 `T.copy` 读入 | 多步计算间存在 GM 往返 |
| 2 | UB/局部值复用 | Developer 模式复用 `T.alloc_shared`，Expert 模式复用 `T.alloc_ub`；合并相邻 `T.Parallel` 或 `T.tile.*` 计算 | 同一输入被多次从 UB 读取 |
| 3 | 减少 Cast | 合并 dtype 转换，在 `T.Parallel` 循环中批量转换并复用结果 | fp32/fp16/bf16 指令占比与预期不符 |
| 4 | 合并逐元素表达式 | 在同一个 `T.Parallel` 循环中完成相邻逐元素计算 | 存在多次完整 Tile 遍历 |
| 5 | 优化归约 | 使用 `T.reduce_sum/max/min`，避免可替代的串行标量累加 | Reduction 中 scalar ratio 高 |
| 6 | 调整并行粒度 | 调整 `T.Kernel(..., threads=...)`、`vid` 切分和 `T.Parallel` 的工作量 | 向量宽度或线程利用率不足 |

### 基本结构

```python
in_ub = T.alloc_shared((tile_elems,), dtype)
out_ub = T.alloc_shared((tile_elems,), dtype)

for tile in T.Pipelined(num_tiles, num_stages=2):
    T.copy(x[tile * tile_elems], in_ub)
    for i in T.Parallel(tile_elems):
        out_ub[i] = fused_compute(in_ub[i])
    T.copy(out_ub, y[tile * tile_elems])
```

不要为套用模板而强制分配额外局部 buffer；只有确实能消除重复 UB 读取或重复计算时才使用。模板中的 `fused_compute` 只是结构占位，实际表达式必须使用当前仓库已公开的符号 API 或 `T.tile.*`；涉及 dtype 转换时先核对 `tilelang-api-best-practices`。

---

## 2. MTE2/MTE3 Bound（数据搬运瓶颈）

**判定**：`ai*_mte2_ratio` 或 `ai*_mte3_ratio` 最高，并结合 `Memory.csv` 确认 GM 搬运量、指令数和带宽利用率。

### 先判断是否接近带宽上限

```text
理论搬运时间(us) = 搬运字节数 / 当前设备可用带宽(Byte/s) × 1e6
```

- 实际时间接近理论时间：优先用流水隐藏搬运，或减少总搬运量。
- 实际时间明显高于理论时间：检查连续性、对齐、单次 `T.copy` 粒度、尾块和 L2 行为。

### TileLang 优化顺序

| # | 方法 | TileLang 实现 |
|---|---|---|
| 1 | 减少 GM 往返 | 将可融合计算放在同一 kernel 和同一 UB Tile 中 |
| 2 | 增大连续搬运粒度 | 增大 Tile，并用连续 slice 执行 `T.copy` |
| 3 | 处理非对齐尾块 | 为目标 buffer 预留对齐空间，使用 `T.copy(..., pad_value=...)` 或显式有效区间 |
| 4 | 重用常量或权重 | Developer 模式使用 `T.alloc_shared`，Expert 模式使用 `T.alloc_L1` 保留跨迭代复用的数据 |
| 5 | 搬运计算重叠 | 使用 `T.Pipelined(..., num_stages=N)` 和多版本 buffer |
| 6 | 调整 L2 策略 | 在确认访问复用模式后，对 `T.copy` 设置并对比 `l2_cache_ctrl` |

不要只追求更大的 Tile；必须同时检查 UB/L1 占用、尾块浪费和可并行 Tile 数量。

---

## 3. CUBE Bound（矩阵计算瓶颈）

**判定**：`aic_cube_ratio` 最高，并结合 `aic_cube_fops`、L0/L1 带宽和实际 FLOPS 判断是否接近计算上限。

### TileLang 优化顺序

| # | 方法 | TileLang 实现 |
|---|---|---|
| 1 | 调整矩阵 Tile | 联合调整 `TILE_M/TILE_N/TILE_K`，平衡计算效率、L1/L0 容量和并行 Tile 数 |
| 2 | L1 数据复用 | Developer 模式使用 `T.alloc_shared`，Expert 模式使用 `T.alloc_L1` 保存 K Tile 或可跨输出 Tile 复用的数据 |
| 3 | L0 累加 | Expert 模式使用 `T.alloc_L0C` 和 `T.gemm_v0(..., init=(k == 0))` 跨 K Tile 累加 |
| 4 | K 流水 | 使用 `T.Pipelined(K_TILES, num_stages=N)` 重叠 GM→L1、L1→L0 和计算 |
| 5 | 持久化调度 | 使用 `T.Persistent` 遍历输出 Tile，减少启动和尾块不均衡 |
| 6 | 输出路径 | 根据输出 dtype 和混合 kernel 结构选择当前 API 支持的 `T.copy`/workspace 路径，避免额外中间搬运 |

### 基本结构

```python
a_l1 = T.alloc_L1((block_m, block_k), dtype)
b_l1 = T.alloc_L1((block_k, block_n), dtype)
acc_l0 = T.alloc_L0C((block_m, block_n), accum_dtype)

for k in T.Pipelined(num_k_tiles, num_stages=num_stages):
    T.copy(a_gm[..., k * block_k], a_l1)
    T.copy(b_gm[..., k * block_k], b_l1)
    T.gemm_v0(a_l1, b_l1, acc_l0, init=(k == 0))
```

---

## 4. SCALAR Bound 与头开销

**判定**：`ai*_scalar_ratio` 高，或小 shape 的 `Task Duration` 主要由启动、动态分支和标量循环组成。

### TileLang 优化顺序

| # | 方法 | TileLang 实现 |
|---|---|---|
| 1 | 编译期特化 | 将 dtype、固定 shape、模式开关放到 JIT 参数或 Python 分支中，减少设备端动态判断 |
| 2 | 移出循环不变量 | 将不随 Tile 变化的索引、尺度或常量计算移到外层 |
| 3 | 减少 `T.serial` | 能并行的元素处理改用 `T.Parallel`，归约使用 `T.reduce_sum/max/min` |
| 4 | 减少动态标量访问 | 避免在内层循环重复读取动态 Tensor 元素或重复构造 `T.alloc_var` |
| 5 | 调整 block 数 | 小数据量时降低 `T.Kernel(num_blocks)` 的 block 数，避免每核工作过少 |
| 6 | 合并小 Tile | 增大每个 block 的工作量，减少循环控制和启动占比 |

小 shape 头开销高不一定是实现错误；需要同时报告绝对耗时和理论可优化空间。

---

## 5. 核间负载不均衡

**判定**：`PipeUtilization.csv` 中目标 kernel 各核 `ai*_time(us)` 差异超过 10%。

```python
times = [row["aiv_time(us)"] for row in target_kernel_rows]
imbalance = (max(times) - min(times)) / max(times) * 100
```

### TileLang 优化顺序

1. 检查 `T.Kernel(num_blocks)` 是否远大于有效 Tile 数或与数据切分不匹配。
2. 使用 `T.ceildiv` 计算 Tile 数，并让每个 block 处理相近数量的完整 Tile。
3. 将尾块分散到多个 block，避免最后一个 block 独占大尾块。
4. 对不规则工作量使用 `T.Persistent`，让 block 持续领取输出 Tile。
5. 若单个 Tile 本身工作量差异很大，重新设计 Tile 维度或按 case 特化切分策略。

---

## 6. Bank Conflict

**判定**：`ResourceConflictRatio.csv` 中 `aiv_vec_total_cflt_ratio`、`aiv_vec_bankgroup_cflt_ratio` 或 `aiv_vec_bank_cflt_ratio` 超过阈值。

### TileLang 优化顺序

| 冲突类型 | TileLang 修改 |
|---|---|
| bankgroup 高 | 调整 `T.alloc_shared` 的二维 shape、行 stride 和 `T.Parallel` 索引映射，避免并行访问集中到相同 bank group |
| bank 高 | 为 UB 行或相邻 operand 添加 padding，改变起始偏移，避免多个 operand 同周期命中同一 bank |
| resource 高 | 拆分过长的融合表达式，调整 Vector/Cube 计算顺序和 pipeline stage |
| MTE 冲突高 | 调整 `T.Pipelined` stage、buffer 版本和 `T.copy` 放置位置，错开搬运与 Vector 访问同一 UB buffer |

每次只调整一个 padding、stride 或索引映射，并重新检查实际冲突比例；不要仅凭 UB 物理结构猜测最佳值。

---

## 7. 流水重叠不足

**判定**：流水图显示 MTE2、VEC、CUBE 或 MTE3 大量串行；`vec + scalar + mte2 + mte3` 的比例和接近 100% 时尤其需要检查。

### 流水结构

```python
buf = T.alloc_shared((tile_elems,), dtype)

for tile in T.Pipelined(num_tiles, num_stages=num_stages):
    T.copy(x[tile * tile_elems], buf)
    compute(buf)
```

### 检查清单

1. `num_stages` 是否至少为 2，并且当前编程模式/Pass 能正确规划多版本 buffer。
2. 不同迭代是否访问互相独立的数据；真实 RAW/WAR 依赖会阻止重叠。
3. `T.copy` 与计算是否位于同一个可流水循环中。
4. 是否存在不必要的同步、串行内层循环或跨迭代写后读。
5. 自动多缓冲无法表达手工 ring buffer 时，按当前编程模式文档设计显式版本维度与同步；不要猜测不存在的 annotation API。
6. 增加 stage 后是否因 UB/L1 占用上升导致编译失败、block 数下降或性能反而退化。

---

## 8. L2 Cache 命中率低

**判定**：`L2Cache.csv` 中目标 kernel 的 `ai*_total_hit_rate(%)` 低，并且 Memory/Pipe 数据表明 GM 访问是主要瓶颈。

### TileLang 优化顺序

1. 先通过 Tile 和持久化调度增加数据局部性，避免同一数据被不同 block 无序重复读取。
2. 对有明确复用的数据使用 `T.copy(..., l2_cache_ctrl="NORMAL_FV")` 等保留策略。
3. 对仅流式访问一次且会污染缓存的数据使用经当前 TileLang 版本验证的 `NOTALLOC_*` 策略。
4. 对输入、权重和输出分别做 A/B profiling，不能用同一 L2 策略覆盖所有方向。
5. 同时检查命中率和总 GM 字节数；命中率上升但总耗时不降时撤销修改。

仓库中的 `example_gemm_bypass_l2.py` 使用 `T.copy(..., l2_cache_ctrl=...)` 展示了输入权重和输出采用不同策略的方式。

---

## 9. 交叉关联诊断

| 现象组合 | 根因假设 | TileLang 优先检查 |
|---|---|---|
| 高 vec_ratio + 高 bank conflict | UB 布局放大 Vector 耗时 | `T.alloc_shared` shape/padding、并行索引映射 |
| 高 mte2_time + 低 L2 hit rate | 数据复用或 L2 策略不合理 | Tile 调度、`T.Persistent`、`T.copy(l2_cache_ctrl=...)` |
| 高 fixpipe_ratio | 输出路径或地址对齐低效 | 输出 Tile、有效区间和当前 API 支持的 `T.copy`/workspace 路径 |
| 高 mte2 + 高 mte3 | GM 双向搬运饱和 | 融合中间结果、增大 Tile、减少 GM 往返 |
| 低 Block Dim + 高 Duration | 并行 Tile 数或 block 设置不足 | `T.Kernel` block 数、Tile shape、`T.Persistent` |
| 各核耗时差异大 | Tile 或尾块切分不均 | `T.ceildiv`、尾块分散、持久化调度 |
| scalar 高 + 小 shape | 动态控制和启动占比高 | 编译期特化、减少 `T.serial`、降低 block 数 |
| MTE/VEC 串行 | 多缓冲未形成 | `T.Pipelined`、buffer 版本和依赖关系 |

所有根因都必须通过源码检查和修改后的重新 profiling 验证。

---

## 10. 仓库实现参考

优先从与目标算子结构接近的文件抽取模式：

| 优化模式 | 参考文件 |
|---|---|
| Vector Tile、UB 搬运、归约 | `examples/pos_embedding/rms_norm.py` |
| Expert GEMM 的 L1/L0C 数据流 | `examples/gemm/example_gemm.py` |
| GEMM intrinsic 与流水 | `examples/gemm/example_gemm_intrinsic.py` |
| 独立性能入口与多 case | `examples/gqa_fwd_varlen/perf_gqa_fwd_varlen.py` |
| `msprof op` 单 kernel 入口 | `examples/pos_embedding/RotaryPositionEmbedding/rope_half_interleaved.py` |
| `do_bench` 与 `msprof op` 双入口 | `examples/fusedmoe/test_fusedmoe.py` |
| 普通 `msprof` 结果解析 | `examples/flash_attention/fa_opt/run.py` |

抽取优化模式时必须保留目标算子的接口、数据布局、边界处理和精度语义，不能整文件替换目标 kernel。
