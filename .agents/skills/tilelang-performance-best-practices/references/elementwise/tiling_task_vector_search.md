# Elementwise / Gather 的 Tiling、任务与 Vector 数据流搜索

本指南用于输出布局变换、gather/scatter、逐元素转换或图像/序列打包一类 Vector kernel。目标是系统枚举高收益结构，避免用未经比较的保守常数过早固定 tiling，也避免只优化外层任务而漏掉内层可并行工作。

## 1. 先画工作分解树

把计算写成至少两层：

```text
外层逻辑 item（row / patch / token / segment）
└── 内层独立 chunk（pixel round / channel chunk / vector tile）
```

分别记录外层 item 数、内层 chunk 数、可用 Vector Core 数和每核迭代数。若外层 item 数少于核数，而单个 item 内存在多个互不依赖的 chunk，必须把 `(item, chunk)` 展平成候选任务空间，与“一个核串行完成整个 item”配对比较。输出仍可按 disjoint slice 写回；若 chunk 间存在归约、顺序依赖或重叠写，明确记录为何不能展平。

展平只证明任务数增加，不证明调度形式已经最优。当前串行循环中显式计算 `task_id = core_id + iteration * num_cores`（本仓 `T.serial` 不接收第三个步长位置参数） 表示每个核执行自己的普通跨步串行循环，适合短小的核内迭代；`T.Persistent([total_tasks], num_cores, core_id, group_size=...)` 向调度器显式暴露跨核持久任务空间，更适合需要统一任务映射或后续流水的外层循环，但也可能增加固定开销。若某个慢 case 依赖 inner flatten 才填满核，必须在相同 tile、算术、访存和尾部条件下配对比较这两种外层映射，或用当前 lowering/生成代码证明它们等价。不得把某个算子的实测结果写成“Persistent 永远快于 serial”；真正的门禁是不能只测试一种跨核映射就关闭任务调度候选。

大 item 的耗时明显高于其他 case 时，不能只因为它沿用 fallback 就忽略此检查。若展平后每个任务过小，再比较一个任务合并若干相邻 chunk 的粒度。

Fixed Core 的具体分配与 owner 约束见 [任务映射](../common/task_mapping.md)。

首次修改前为每个 case 记录工作树、outer/flattened waves、payload、实测时间、下界和主要 pipe。结构触发的待验证方向独立于当前候选池；候选排序变化不能使它们消失。具体落盘格式和状态由调用本指南的调优工作流规定。

## 2. 枚举有来源的 tile 边界

候选 tile/group 不从单个经验常数开始。至少列出以下边界中对当前布局有效的值：

1. **完整连续语义单元**：一整行、一个连续 segment、一个 channel plane，或不会跨元数据/边界的最大连续范围。
2. **SIMD 边界**：lane 数、单条 load/select/gather/scatter 的有效 payload，以及整数个 vector chunk。
3. **DMA 边界**：连续 burst、二维 copy 的整行宽度、对齐和 stride 限制。
4. **容量边界**：把所有常驻 buffer、索引、padding、安全余量和 1/2/3 个版本计入后的 UB/L1 最大值。
5. **并行边界**：至少一波、两波及稳态多波所需的任务数；同时记录最长核负载和尾 tile 比例。

对指令边界分别核对 dtype 对齐、compare/select、transpose 与 cast 的完整 footprint。最小合法粒度也超限时修改数据流，不返回伪合法最小值；候选可按几何步进补齐，但不能用任意 cap=512/1024 代替边界搜索。

将这些边界去重后形成小型候选集合，先用资源公式淘汰不合法值，再实测有不同数据流含义的值。可以设置安全余量，但必须说明它保护的具体 lowering 或临时存储；不得用任意 cap 代替完整连续单元或容量极限候选。

逐 case 记录这五类边界及其来源值。候选实测只能裁决与其实际粒度一致的边界；小 group、经验 cap 或某个并行粒度不能代替完整连续单元。停止前必须给出每类边界的实测结果或明确的容量/lowering 不可实施证据，不能静默丢弃。

## 3. 建立 Vector 数据流族与指令成本表

先按数据实际经过的存储层级和访问方式分类，而不是按某个算子的 API 拼写预设答案。常见但不穷尽的数据流族包括：

| 数据流族 | 代表性物理路径 | 主要成本 |
|---|---|---|
| 标量路径 | 每任务独立索引、分支、算术与读写 | 标量索引、分支和任务粒度 |
| memory-indexed Vector | 当前 API 支持的 UB 索引访问，或读取 table/LUT | 离散访存、索引寄存器、lane 利用率和表初始化 |
| register-resident Vector | 连续 Vector load 后在寄存器内 select/shuffle/pack/convert/算术，最后写出 | 连续 load/store、寄存器重排、转换和 live register 压力 |
| materialized transform | 先在 UB/L1 物化中间布局，再 transpose/permute/DMA 或连续写出 | 额外完整读写、临时容量、同步和对齐 |

这些名称描述物理数据路线，不规定固定 intrinsic；一个实现可以组合多个族。 register-resident 路线只有当前 API、架构和 lowering 均有证据时才可实施，否则记录 DESIGN_ONLY 或确定的不适用证据，不能强制实现。只有当前接口语义、实际 TileLang API/lowering 或数据依赖证明某一族不适用时，才能以逐 case 证据关闭。

当前可表达的路线还包括连续 DMA + tile 运算、块 DMA + UB-local 重排与标量 fallback。UB 内局部 `SetValue` 不必然差于逐元素 GM DMA，参见 [dtype 适配](../conversion/dtype_adaptation.md)。

物理路线按数据从源到结果经过的完整层级分类，不能只看最后一条算术指令。先把交错/离散数据写成完整 planar、transpose 或 scratch 布局，再连续 load 到寄存器，属于 `MATERIALIZED_TRANSFORM`；只有从原始或分段源窗口直接连续 load，并在寄存器内完成 select/shuffle/pack、没有完整中间 payload 写回，才属于 `CONTIGUOUS_LOAD_REGISTER_REORDER`。

### 3.1 同一索引重排的两条常见物理路线

对 `y[j] = f(x[index[j]])`，逐 lane gather 和“连续 load + 寄存器 select/shuffle”解决同一问题，只是索引作用层级不同：

```text
memory-indexed:
    生成/加载 index → 每个 lane 按 index 访问 UB/GM → 计算

register-resident:
    连续 load 源窗口 → 窗口内 lane index → register select/shuffle → 计算
```

第一次候选池前，逐 case 记录源窗口特征：

```text
source_span_bytes = 覆盖全部所需源元素的最小连续窗口字节数
source_density = 实际消费的不同源字节数 / source_span_bytes
window_vregs = 容纳源窗口所需的 Vector register 数
index_regularity = 固定/周期/仿射，还是运行时动态/数据相关（影响索引生成，不单独决定路线）
source_reuse = 同一窗口产生的有效输出或复用次数
```

| 源布局与访问特征 | 优先候选 | 原因 |
|---|---|---|
| 候选源范围可容纳于合理的寄存器 footprint；密度高或源可跨 lane/迭代复用 | 连续 load + register select/shuffle/pack | 用连续访问替代逐 lane 访存；index 可以规则也可以数据相关 |
| 宽跨度中只取少量元素；候选范围无法装入可选择的寄存器；完整窗口会过取或溢出 | gather / memory-indexed Vector | 只读所需元素，避免复杂跨寄存器重排 |
| 分段连续但段间离散 | 分段连续 load + 段内寄存器重排，与 gather 配对比较 | 不能把局部连续性或段间离散性单独当成全局结论 |

索引跨越 row、segment 或对象边界时分别计算合法窗口，不能用大包围区间虚构连续性。寄存器 payload、dtype、跨寄存器选择和 lowering 限制从当前 TileLang/所选后端与目标架构获取，不硬编码平台宽度。交错 record 到分字段布局（如 `[a0,b0,c0,a1,b1,c1,...]`）若最终消费窗口中大部分字段，属于高密度窗口，应尝试连续 load 后解交错；只取宽跨度中的少数字段则通常更适合 gather。

寄存器 select/shuffle 是否覆盖目标映射，要核对 dtype、单/多寄存器选择范围和当前后端 lowering；gather 的 index 单位、源 dtype、随路扩宽和 mask 语义也必须从实际源码确认。仓库示例只证明写法存在，不证明它在当前数据分布上占优。

对相同有效输出量统计：连续与 indexed GM/UB 访问、select/shuffle/pack、table/LUT 与初始化、索引、cast/算术、predicate、临时物化、lane 利用率和 live-register 压力。源码拼写不同但生成指令与访存相同的候选可合并；物理路线不同则不能因都属于 Vector 执行侧而合并。

路线对比必须尽量固定正交轴：相同 tile/group、任务映射、数值精度、输出落盘方式和 stage 数，只替换待验证的输入/重排路线。若一次候选同时改变了精度、cast 链、输出 scatter/store 或流水，它的结果只能裁决这条**完整指令链**，不能直接写成“连续 load + register reorder 慢于 gather”或反向结论。无法完全固定时，必须列出差异并为仍可能高收益的最低成本代表建立新候选。

初始候选记录包含上述窗口指标、两条路线的估算成本和状态。两条均可实施且可能影响主瓶颈时，各测试一个最低成本代表；只保留一条必须有容量、语义、lowering 或严格成本支配证据。若有界窗口可装入寄存器，关闭连续 load + register reorder 还必须列出实际搜索路径、symbol 和 lowering 结论。Vector-bound 后若仍有路线能减少 indexed 访问、完整物化或提高有效 lane，必须新增或重开，不能因语义 GM 字节接近下界就收敛。声称“最小序列”还须给出最低语义操作、生成指令/访存计数、API 搜索范围及其他路线的裁决。

### 3.2 连续窗口与寄存器重排的适用边界

这里的三个概念必须分清：

- **连续窗口**：源数据先通过当前可用的连续 load 路径读入；
- **register-resident**：后续重排只操作寄存器，没有把完整中间布局写回 UB/L1；
- **lane index**：寄存器内的 lane 编号，不是 UB 地址。它不要求固定步长，但每个索引都必须落在该次可选择的寄存器窗口内。

寄存器宽度和选择原语以实际架构、API 与 lowering 为依据。若当前 API 无法表达所需寄存器选择，就保留当前可执行的 Vector/UB-local 路线；不能把 UB 中物化的布局称为 register-resident。

若所需窗口跨多个寄存器，不得把全局索引直接交给单个寄存器选择。应将源范围切成可证明的局部窗口，为每段重映射 index，再合并/写出；若跨寄存器选择和 pack 成本高于逐 lane UB 访问，则与当前可执行的索引路线实测比较。索引单位、源 dtype、随路扩宽、lane 数和后续算术必须从当前实现核验。

### 3.2.1 最低成本类型与 lane 链门禁

连续 load 路线的首个代表必须是当前 API/lowering 下有依据的低成本链，不能把选数后的 lane 排布修复成本当成路线固有成本。实现前逐步记录每条 intrinsic 的输入/输出 dtype、有效 lane、转换分片语义和是否产生空槽，并检查：

1. gather 是否已经随路扩宽，而连续 load 路线是否需要显式整数扩宽；
2. 直接窄整数转浮点是否因转换分片产生需要多次交织修复的 lane 排布；
3. 是否存在先做一次整数扩宽、再转目标浮点的更短链；
4. 目标精度允许时，是否能在较窄浮点上用当前支持且数学等价的融合 affine，避免无必要的 FP32 扩宽、交织和最终 pack；
5. 当前映射是否匹配当前已支持的 load distribution、store distribution 或其他已 lowering 的原生排布操作，避免手工复制其功能。

对窄整数字段，先整数扩宽再转目标浮点可作为待核验的低成本形态。实际 cast/融合分别核对 [ascend_tile.py](../../../../../tilelang/language/ascend_tile.py) 与所选后端。

若直接 `u8 -> f16 -> f32` 的代表需要多个分片转换、数次交织和 pack，而上述整数扩宽或其他当前 lowering 路径尚未裁决，则在候选失败原因中注明该实现不具代表性：候选本身可以按工作流状态淘汰，但不能关闭直接寄存器重排路线，并须新建最低成本代表。反之，gather 的随路扩宽也必须计入其优势，不能只比较连续 load 与索引访问单条指令名称。

核验位置不要靠记忆：从实际导入的 TileLang 根目录查看 [ascend_tile.py](../../../../../tilelang/language/ascend_tile.py)、[copy_op.py](../../../../../tilelang/language/copy_op.py)、当前后端 codegen 与 [Ascend 模板](../../../../../src/tl_templates/ascend/common.h)。具体索引/select 组合仍须在当前 API、lowering 与测试中逐项核对，示例不能替代目标 shape 的精度与 profiling。

还要区分intrinsic 的 base 与其索引操作数：普通动态 `ub[index]` 并非统一禁止；但 `ub[stage, ...]` 若成为当前原语的 base pointer，需要分别核验地址或指令选择限制。一次动态 base 失败只否决该 addressing 组合；多版本流水必须继续测试外层 stage 分支形成的静态 base body。具体结论以当前版本 lowering、生成代码和最小复现为准，不能外推成硬件永久限制。

区分“只改变 task/group 并继承父版本数据流”与“引入或替换 gather、连续 load + register reorder、完整物化等物理路线”。若已确认寄存器窗口可实施，且直接寄存器重排尚未被基线或确定证据关闭，首个以物理路线为假设的候选应验证它；熟悉某个 gather 示例不构成改变顺序的证据。候选字段及状态由调用本指南的调优工作流规定。

### 3.3 按输出顺序索引与连续写出的对照路线

当所需输入跨越多个连续窗口、但能按最终输出顺序生成 UB 索引时，同时考虑当前已支持的索引收集与连续写出组合：

```text
output-order index → indexed load(raw UB) → Vector compute → contiguous store(output UB)
```

这条路线把布局转换成本放在输入侧：索引访问的 lane 按连续输出 lane 排列，计算后即可连续写入输出 UB。它应与“可实施的连续 load + 重排 + scatter 或分段 store”按相同有效输出 payload 比较，分别统计 indexed UB read、索引生成/加载、随路扩宽、跨寄存器重排和输出 store。某个输入路线失败或较慢，不能自动否决另一种输出落盘组合；反之亦然。实际实现的 dtype、mask 和 store 布局必须从当前 API/lowering 核对。

## 4. 正交轴组合与重开门禁

不要把“一个候选”只记成一个名字。对布局转换和 gather/scatter 算子，将它记成至少包含以下正交轴的组合：

```text
(tile/group 边界, 任务映射, Vector 数据流, 计算精度, buffer 版本/流水, 尾部策略)
```

一个轴的改动可能会改变另一个轴的前提：更大 group 会增加连续 payload、摊薄 LUT/元数据成本，但也可能把瓶颈从 DMA/调度转移到 Vector 重排；SIMD 数据流在小 tile 上可能不划算，在更长的连续 tile 上却可能成为关键。因此，`group + 原数据流` 失败和 `原 tile + SIMD` 失败，都不能推出 `group + SIMD` 失败。

六轴元组是候选身份而不是说明文字。tile/group、任务映射、物理数据流、精度、流水或尾部任一轴实质变化，都建立新候选并保留原候选结果；禁止复用旧 ID 表达另一组合。当前候选池只是执行队列，完整待裁决方向由调用本指南的调优工作流维护。

若 tiling/任务改动提高了连续性或降低了每任务固定开销，而 Vector 改动能消除该新 tile 的主计算/重排成本，则两者是有交互的高收益组合。同一个候选必须实际覆盖对应粒度与物理路线，才能裁决该组合。停止前至少完成下列裁决矩阵，或对未执行的象限给出当前 lowering、容量、数据依赖或指令成本证据：

| | 基线 Vector 数据流 | 候选 Vector 数据流 |
|---|---|---|
| 基线 tile/任务 | 基线 | 单独裁决 Vector 改动 |
| 候选 tile/任务 | 单独裁决 tiling 改动 | **裁决二者组合** |

对“输入顺序与输出布局不同”的算子，grouping 后必须重新写出布局映射。若原生 permute/transpose 不存在或 lowering 未验证，不得直接回退到逐元素标量 fallback 并宣告结构收敛；根据实际 API 比较 memory-indexed Vector、register-resident Vector、临时布局物化或其他可实施路线。能在生产者内直接写最终布局时，优先把转换与计算融合，并把省掉的 UB 读写与新增索引指令都纳入成本表。

若完整连续单元可以界定，`FULL_CONTIGUOUS_UNIT × CONTIGUOUS_LOAD_REGISTER_REORDER` 是必须单独裁决的通用交互项。测试一小段 grouping 后不得用“更大 group 只是同一路线”关闭它；完整单元会同时改变 DMA 长度、任务数、索引摊销和可用的寄存器重排窗口。

每次实测后都要判断瓶颈是否转移。若新瓶颈正好由候选池中另一个轴解决，立即把对应组合新增或重开为高收益候选；不得因初始列表没有写出该组合而静默跳过。

## 5. 把候选失败限定在准确范围

候选记录至少包含：tile/group、任务映射、执行域、关键指令序列、buffer 版本、流水控制流和尾部策略。一次失败只否决这个组合：

- 某个 gather 索引格式精度失败，不否决其他 SIMD select/shuffle 或索引格式；
- 某个 dtype、cast 链或 mask 格式失败，不否决其他精度合法的路线；
- planar/materialized Vector 失败，不否决无完整中间 payload 的连续 load + register reorder；
- 小 group 失败或收益不足，不否决来源值不同的完整连续语义单元；
- 某个 tile 的 stage 2 变慢，不否决其他 tile/控制流的 stage 2；
- 自动 buffer version 不 eligible，不否决显式 storage 的手动多版本；
- 动态 stage 地址 lowering 失败，不否决静态 stage body。

多阶段布局失败必须区分 planner、单 stage 边界与 dtype 路径，不能用 fallback 通过冒充优化通过。

只有生成代码证明候选实质等价时，才能合并失败结论。若两个改动位于不同正交轴，候选记录必须明确说明它们的组合是已实测、无交互，还是因确定证据不可实施。

若以“严格受支配”关闭未实施路线，必须比较同一有效 payload 下的访存层级、load/store/gather/scatter、转换、临时物化、同步和 active lanes，并给出当前 API/lowering 可实施依据。笼统的“指令更多”“代码更复杂”不构成关闭证据。

## 6. 停止前的结构覆盖审计

性能目标未达到或仍有明显慢 case 时，停止前逐项填写：

| 结构族 | 已测试的最高收益候选 | 结果/证据 | 未实施原因 |
|---|---|---|---|
| 连续搬运与 tile/group 边界 |  |  |  |
| 外层/内层任务映射 |  |  |  |
| 索引源窗口诊断（span/density/register footprint/regularity） |  |  |  |
| memory-indexed Vector 路线 |  |  |  |
| register-resident Vector 路线 |  |  |  |
| 临时物化或其他可实施 Vector 路线 |  |  |  |
| 有交互的 tile/task × Vector 组合 |  |  |  |
| 单 stage 与完整多版本流水 |  |  |  |
| 尾部与小任务低开销路径 |  |  |  |

初始及后续重开的高收益候选必须有实测结果或确定的不适用/淘汰证据；存在未裁决项时，不满足证据收敛停止条件。具体状态名由调用它的优化工作流定义。
