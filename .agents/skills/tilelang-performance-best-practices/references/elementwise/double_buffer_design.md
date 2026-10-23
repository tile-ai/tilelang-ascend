# Elementwise 多版本流水

## 目标与适用条件

多版本流水用于让第 `i+1` 个 tile 的搬入、第 `i` 个 tile 的计算和第 `i-1` 个 tile 的搬出在硬件允许时重叠。它适用于 GM 搬运与 Vector/Cube 计算均占有可见时间、每核有足够多独立 tile、UB 容量能容纳全部版本的算子。小任务、单迭代或某条流水已接近理论下界时，多版本可能只增加 UB、同步和头尾开销。

实施前画出每个逻辑迭代的依赖：

```text
GM --CopyIn--> input/local buffer --Compute--> output/local buffer --CopyOut--> GM
```

对每个会在下一迭代被写入、但上一迭代仍可能被 Compute 或 CopyOut 读取的 UB/L1 buffer 标记版本。输入、输出、索引、临时中间量都按真实生命周期判断；不能假设“只给输入双缓冲”就足够。只读且跨迭代不变的常驻数据通常保持单版本。

同时核算版本总 footprint、padding、常驻数据和每核有效迭代数。有效迭代少于 stage 数时回退单 stage；3 stage 只有在 2 stage 后仍有流水缺口且容量允许时才进入候选。

## 自动与手动多版本

访问关系简单、仿射且编译器能唯一识别生产者/消费者时，先试自动版本化。优先把无条件、固定 extent 的完整迭代从尾部处理中剥离，并用 `T.Pipelined` 暴露规范的 CopyIn → Compute → CopyOut 主体：

```python
input_ub = T.alloc_shared((tile_elems,), dtype)
output_ub = T.alloc_shared((tile_elems,), dtype)

for tile in T.Pipelined(full_tile_count, num_stages=2):
    T.copy(src[tile], input_ub)
    compute(input_ub, output_ub)
    T.copy(output_ub, dst[tile])

# 余数 tile 使用独立的单 stage 路径，不放进上面的稳态主体。
if has_tail:
    process_tail(...)
```

这是待核验 lowering 的结构示意，compute/process_tail 为占位逻辑，不是完整可执行 kernel。这个自动模板有四个必须同时满足的结构条件：

1. `input_ub/output_ub` 按**单份逻辑 tile** 分配；不得先写显式 `[2, ...]` stage 维度再要求编译器对同一 buffer 自动扩为两个版本，否则等于同时要求手动和自动扩版本，可能造成 extent 重写错误。
2. 只版本化稳态迭代中确实跨迭代存活的可变 buffer。只读 LUT/index、流水外 tail buffer 和未参与该主体的 fallback buffer保持单版本。
3. `T.Pipelined` 主体每次执行相同 rank、shape 和 copy extent 的 CopyIn/Compute/CopyOut。会改变访问范围的 full/tail 分支、零工作分支和动态余数处理放在主体外。
4. 首个自动代表优先直接访问版本化 buffer 的原始逻辑 shape；不要再通过带隐藏 extent 的 `view`、扁平别名或显式 stage 下标访问。若 intrinsic 需要 base pointer，仍须保证它可追溯到同一个逻辑 buffer 和固定范围。

当前 `T.Persistent` 不支持 `num_stages`；Persistent 或 Fixed Core 核内循环不能只因外层已有任务映射就视为自动流水的标准形态。只有它同样提供无条件、固定 extent、唯一生产者/消费者的稳态任务主体时才作为自动候选；否则先用上面的 `T.Pipelined(full_tile_count, ...)` 代表裁决自动版本化。CANN/编译器升级也不能代替这些结构条件。

`num_stages` 只是调度意图，不能代替 buffer versions 和依赖核验。view/别名、flattened index 或 gather/scatter 可能令自动分析错误认领 storage、扩大 extent 或失去版本关系；先最小化并确认失败位置，再改用显式 stage storage 和当前版本实际存在的 buffer ownership 表达：

```text
显式分配 input_ub/output_ub[2, tile_elems]
用当前 LOCK 源/预处理器建立每个 stage 的 free/ready/computed ownership
for wave in full_waves:
    stage = wave % 2
    acquire free[stage]
    copy_in(wave, stage); publish/wait ready[stage]
    compute(stage); publish/wait computed[stage]
    copy_out(wave, stage); release free[stage] after last consumer
```

这是所有权示意，不等于已经形成 overlap。A2/A3 的 prefetch/main/epilogue 和正确轮转见下文补充；不能将显式槽位再交给不匹配的自动版本化处理。

所有跨迭代存活的 buffer 使用同一 stage。这里要区分普通动态 UB 下标与 tile 原语的动态 base pointer：`ub[index]`、`T.copy` 的动态槽位不能因某次失败被概括为“不支持动态地址”；若 `ub[stage, ...]` 成为当前原语的 base pointer，需要分别核验版本分析、指令选择和地址合法化。

若失败证据落在这种动态 base 上，只淘汰“动态 stage addressing”组合，并在外层按 stage 调用编译期常量的通用 body：

```python
stage = wave % 2
copy_in(wave, stage)  # T.copy 能否使用动态槽位仍按当前 lowering 核验。
if stage == 0:
    compute(input_ub[0], output_ub[0])
    copy_out(output_ub[0], wave)
else:
    compute(input_ub[1], output_ub[1])
    copy_out(output_ub[1], wave)
```

这样 Vector body 内的 base 是编译期常量，但算法、tile 和 runtime kernel 仍保持通用；不能按测试 shape 特化。静态 stage 也不能只凭编译成功晋升，仍需与同算术、同任务映射的 stage-1 版本配对验证 latency 与 overlap。API 与 lowering 必须从实际导入的 TileLang/当前后端 或仓库已验证实现核对。

流水候选按四个维度记录，失败只否决准确组合：

1. **Storage**：自动 versions，或显式 `[stage, ...]` storage 加显式 ownership；
2. **Addressing**：动态 stage，或由外层 stage 分支/macro 形成的静态 stage 0/1 body；
3. **Control flow**：条件式循环，或剥离尾部后的无条件完整 full-wave；
4. **Scheduler**：`Persistent`/`Pipelined` 或 Fixed Core、stage 数和每核稳态迭代数；`Persistent` 本身不接收 stage 参数。

当每核有足够独立迭代且 profiling 表明 MTE 与 Compute 均有可重叠时间时，流水候选即已准入。准入后先试自动组合；只有全部跨迭代存活 buffer 均正确版本化、生成代码存在预期切换/同步，且 stage-1/2 配对的 latency/overlap 改善可复现，才算自动组合完整。

自动版本化的 `not eligible`、extent/别名错误、lowering 失败、只版本化部分存活 buffer或实测未生效，只淘汰自动组合。无论用户是否给出数值目标，在宣告已准入的流水方向失败或收敛前，都必须完成以下手动代表，或提供当前 lowering、容量、依赖或 API 的确定不可实施证据：

- 输入、输出及所有跨迭代存活临时 buffer 均为显式多版本；
- 显式 ownership，必要时配合静态 stage body；
- 当前可表达的无条件 full-wave 流水主体，尾部独立；
- 与同 tile、同算术、同任务映射的 stage-1 control 紧邻配对实测。

自动组合已经完整生效时，可以不再测试手动等价实现，但必须明确记录 `NOT_NEEDED_AUTO_COMPLETE`。某个工作粒度或数据流使自动版本化失效，不能作为另一工作粒度/数据流下手动流水不可行的证据。

手写 `set_flag`/`wait_flag` 或 event 次序不是默认替代方案；只有当前仓库或实际安装源码存在同访问模式的验证示例时才进入候选。死锁只否决对应 event 方案。

稳态流水主体必须让调度器可见；不要用运行时条件包住整段 CopyIn/Compute/CopyOut。用通用边界剥离尾部：

```text
full_waves = work_items // items_per_wave
pipeline(full_waves):
    无条件执行完整 CopyIn → Compute → CopyOut
if work_items % items_per_wave != 0:
    单独处理最后一个尾波
```

动态 shape、越界保护和尾块语义仍须保留。选择 `Pipelined`、`Persistent` 或手动 Fixed Core 流水取决于当前 API 的迭代映射，不能仅凭名字判断。

## 生效判定

编译成功、出现两个 buffer、`num_stages > 1`、运行正确或单条 pipe ratio 变化，都不能单独证明流水生效。必须在同口径下配对比较 stage 1/2：

- kernel latency 的改善超过测量噪声并可复现；
- 按源码实际 GM 字节数计算的有效带宽或实际算力改善；
- 与假设对应的 MTE/Vector/Cube overlap 改善，且总耗时同步下降；
- 必要时检查生成 IR/代码中的版本切换、同步和满波调度结构。

若 latency/overlap 没有超出噪声的改善，记录“编译通过但未生效”。stage-2 失败记录完整组合标识，不得外推到其他 tile、Vector 数据流或控制流。

流水不能改变运算、cast 和输出语义；尾 tile 的每个版本独立初始化且只搬 valid 范围。覆盖 stage/tile 边界、非对齐尾块和接口规定的极值。带多版本、同步或调度 warning 的候选必须在同一进程跑全部目标 case，并改变 case 顺序复测；出现顺序相关漂移、偶发错误或内存破坏时淘汰。不得放宽精度或跳过 case。

## 可执行代码与证据

先在当前仓库检索 `num_stages`、`T.Pipelined`、`T.Persistent`、显式 stage storage 和 LOCK/flag 的真实用例，再检查实际导入的 TileLang 源码与当前后端 lowering。可从 [Vector Add pipeline](../../../../../examples/elementwise/elementwise_add_pipeline.py)、[T.Pipelined 教程](../../../../../docs/tutorials/t_pipelined.md)、[LOCK 源](../../../../../examples/flash_attention/fa_opt/expert_v3/kernel.tl) 及仓库内匹配到的其他 Ascend 实现开始，但每次复用都必须重新核验适用的访问模式、版本和设备证据。

---

## A2/A3 实现补充与约束

## Vector 双缓冲与三阶段结构

### 参考与适用性

参考 [单缓冲 Add](../../../../../examples/elementwise/elementwise_add.py) 与 [流水 Add](../../../../../examples/elementwise/elementwise_add_pipeline.py)。它们是当前仓库代码证据，本次文档重构未重新上板验证。现有示例的整除 shape、尾部与配置必须在目标中重新核验；不将本文结构图当作可直接执行的通用 kernel。

适合每核有多个独立 tile、MTE2/Vector/MTE3 存在可重叠工作且 UB 足够的路径。单次迭代、小 payload、某 pipe 已接近下界时，双缓冲可能只增加启停/同步和容量压力；旧“Vector≥128、Cube≥256 元素”不是平台通用门槛。

### 工作量与地址

对采用两个 Vector 子核平分行的已有模式：

```text
rows_per_vec = sub_M / VEC_NUM
tiles_per_vec = block_M / sub_M
row(tile, vid) = bx*block_M + tile*sub_M + vid*rows_per_vec
```

上述完整块模式要求 `block_M % sub_M == 0`、`sub_M % VEC_NUM == 0`、各列地址/长度合法。`VEC_NUM=2` 是此映射参数，不是所有 kernel 的强制划分；尾行/尾列另行有效范围处理。不能把原 `vid*block_M/VEC_NUM` 偏移不变地搬入新子 tile。

输入 a/b 与输出 c 分配 `[stages, rows_per_vec, block_N]`；多行情况下不能退成 `[2,block_N]` 而丢失 rows。Add 的显式占用为 `3*stages*rows_per_vec*block_N*sizeof(dtype)`，另加 scratch、padding 与常驻量。计算链临时量是否单份按真实 live interval 决定。

### 正确三段式与槽位

下面是 ownership 伪代码；实际新 Expert kernel 用 LOCK DSL，已有 low-level 实现可以核对对应 flag。只展示两槽、非空 task：

```text
init: free[0] = free[1] = available
prefetch:
    acquire free[0]
    copy input tile 0 -> slot 0
    publish ready[0]
main, tile = 0 .. tile_count-2:
    cur = tile % 2
    nxt = (tile+1) % 2
    acquire free[nxt]
    copy input tile tile+1 -> slot nxt
    publish ready[nxt]
    wait ready[cur]
    compute slot cur
    publish computed[cur]
    wait computed[cur]
    copy output slot cur -> output tile tile
    release free[cur] after copy-out completion
epilogue:
    last = tile_count-1; slot = last % 2
    wait ready[slot]; compute slot; publish/wait computed[slot]
    write output tile last; release free[slot]
drain: wait all initialized free tokens before reusing resources/return
```

`tile_count==1` 只执行 prefetch+epilogue；`tile_count==0` 不 prefetch。每个 tile 恰好读、算、写一次；初始化 token 与 drain 平衡，包括没有实际使用的槽位。旧 `si=1; cur=si%2` 与预取 slot 0 不匹配：第一消费者必须是 tile 0 / slot 0。

低层事件对应 `mte3→mte2` 的槽位释放、`mte2→v` 的输入就绪、`v→mte3` 的最终输出就绪。cast/copy 到 output buffer 也属于生产链；不能提前发布 output ready。多输入必须全部就绪后再发布，最后输出 consumer 结束前不能覆写。

### 自动与手动方案

Developer 路径先沿用可识别的 `T.Pipelined`/同步结构，从生成代码确认是否真有多版本和 overlap。手工 ownership 与自动插入的同步若冲突，定位具体等待再调整；不因算子类型直接关闭全部自动功能。新手工路径不得省掉 `T.Scope("V")` 和 LOCK DSL 所有权描述。

优先让满 tile 主体保持固定范围和清晰 producer/consumer；尾部单独处理。需要显式槽位时核对动态 slot 索引、alias、物理 buffer 和事件 ID；当前 lowering 无法表达的组合记录最小复现，不能照搬其他平台的 annotation API。

### 验证矩阵

同 tile、同 arithmetic、同任务映射的单缓冲与双缓冲配对；再单独搜索 tile/stage。检查生成源码的三类事件、地址轮转和跨 stage 依赖，测 latency 与 overlap 的可复现改善。未超过噪声记为“未证实收益”。

覆盖 0/1/2/3 次迭代、奇偶 tile 数、最后一波、两轴尾块、dtype cast、多个 case 连续与逆序运行。出现偶发错误/顺序相关漂移时先排查生命周期，不能放宽容差或减少测试。核间/核内联合优化遵循 [flat pipeline](../common/pipeline_and_cv_overlap.md)。
