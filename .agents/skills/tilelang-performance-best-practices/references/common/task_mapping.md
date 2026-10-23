# Fixed Core 与任务映射

## 分离逻辑任务、launch block 与 Vector 子核

先从目标 [Kernel DSL](../../../../../tilelang/language/kernel.py)、同类样例与生成代码确认 `cid/vid`、launch block 与 AIC/AIV 数的关系。常见 `(cid, vid)` 和每 block 两个 Vector 子核不意味着直接把查询到的 AIV 总数填成 `T.Kernel` 的 block 数。记录硬件核数、当前映射和实际启用数，不能写死 24。

逻辑任务远多于可同时运行的核时，可固定 launch 数并让每核处理多个 task，摊薄初始化、地址准备和 workspace 分配。逻辑任务少时减少空核。对确实独立且非空的任务域，候选 `core_num <= min(可用launch容量, task数)`；空任务在接口层或合法 device 分支处理，不以 `max(1, tasks)` 后无条件读取输入。

连续分配：

```text
tasks_per_core = ceildiv(task_count, core_num)
task = cid * tasks_per_core + local_id
仅 task < task_count 时处理
```

跨步分配：`task = cid + wave*core_num`。两者应按局部复用、最长/最短核任务量、尾波、任务耗时差异及地址计算比较。workspace 可按 core 槽位复用，但生产/消费尚未结束时不能被下一 task 覆盖；按 task 保存的最终 partial 不能错误缩成 core 大小。

## 工作树与多轴域

列出 `outer item → independent inner chunks`。outer item 少于可用核时，评估 `(outer, chunk)` 展平；chunk 间有 scan、归约或重叠写依赖时先证明合法分解。展平后可能每 task 太小，再评估合并相邻 chunk。

二维或多维任务显式保存各轴含义，生成映射要覆盖笛卡尔积恰好一次。热循环 div/mod 成为实测开销时，比较静态循环嵌套、偏移增量与当前已有 scheduler；`T.Persistent` 必须按本仓 [实际签名与 domain 约束](persistent_task_mapping.md) 使用，不带源平台 `num_stages` 参数。没有生成代码差异的等价表达无需重复测试。

## owner 与 MIX

每个输出必须有唯一 owner 或已设计的归约。MIX 类型不应一律加 `if vid == 0`：目标可能本来按 `vid` 分割两半，统一 guard 会丢一半结果。若旧 kernel 确实让两个 AIV 重复处理同一 slice，可选择唯一 owner 或补充分片，但必须连同输入、输出、参数和 workspace 索引一起核验。

记录每 case 的 task 数、outer/flattened waves、payload、尾部比例、初始化复制次数；小 shape 与大 shape 分别比较。Fixed Core 不是所有 shape 的强制优胜方案。
