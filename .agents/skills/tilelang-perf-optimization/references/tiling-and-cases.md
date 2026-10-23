# Tiling 与 case 建模

## 先还原执行路径

对每个 case 从公开入口向下记录：dispatch 条件 → build 参数 → `@T.prim_func` → `T.Kernel` 任务映射 → tile 循环 → 执行域。相同算子名不代表相同执行路径；broadcast 轴、reduction 轴、dtype、对齐和尾块都可能改变分支。

建议建立下表：

| Case | 语义路径 | dtype/累加 | dispatch | 核数 | tile | 尾块 | 基线瓶颈 |
|---|---|---|---|---:|---|---|---|

只有执行路径和瓶颈均相近的 case 才合并建模。

## 通用模型

### 任务与核数

- `independent_tasks`：无需跨任务依赖即可并行的任务数。
- `used_cores <= independent_tasks`。
- `tasks_per_core = ceil(independent_tasks / used_cores)`。
- 同时记录最长核和最短核的任务量；平均值不能反映尾差。

核数来源及 vector/cube/mixed 选择必须由 `tilelang-performance-best-practices` 和当前仓库配置确认。

### tile 与循环

- `tiles_per_task = ceil(valid_extent / tile_extent)`。
- 每 tile 区分 `valid_extent` 与为 SIMD/DMA/layout 准备的 `padded_extent`。
- 估算固定开销时使用实际 tile 次数；估算 GM 字节时只统计合法读写；估算 UB 占用时使用完整 padded footprint。

对小 shape，增大 tile 或合并任务可能降低循环与调度开销；对大 shape，过大 tile 可能减少并行度或造成 buffer 压力。两者都必须由实测验证。

### buffer 预算

逐个 buffer 计算：

`bytes = product(padded_shape) * dtype_bytes * versions`

总预算还要加入常驻数据、临时结果、对齐 padding 和安全余量。`T.Pipelined` 和显式 stage storage 的版本关系以当前 TileLang 实现和已验证样例为准，不自行假设。当前 `T.Persistent` 不接收 `num_stages` 参数。

### 数据流

为每个 tile 列出：

- GM 读取/写回字节。
- UB/L1/L0 中间结果生命周期。
- 同一数据的重复读取次数。
- 运算量与 reduction/GEMM 累加状态。
- 搬入、计算、写回是否有可验证的重叠机会。

## 参数输出

方案中的参数必须同时给出：当前值、候选值、推导依据、适用 case、容量/对齐门禁和回退条件。示例：

| 参数 | 当前 | 候选 | 依据 | 适用 case | 门禁 |
|---|---:|---:|---|---|---|
| `tile_n` | 128 | 256 | 小 case 循环开销主导 | group-S | UB 含 versions 与 padding 后不超预算 |
| `used_cores` | 20 | 8 | 独立任务仅 8 个 | group-tail | 不超过任务数，逐 case latency 不退化 |

没有经过当前版本 API、lowering 和容量核验的候选值只能标为待验证，不能写成最终配置。

## 任务树与并行度

- 列出外层 batch/head/row 等独立任务，以及可独立切分的 inner chunks。
- 区分逻辑 block 数、实际执行核数和每核循环；逻辑网格可以大于物理核数，Fixed Core 只是候选映射之一。
- `waves = ceil(independent_tasks / used_cores)`，同时记录最长/最短核工作量、空任务和最后一波利用率。
- 扩大 tile 可能减少调度开销，也可能降低独立任务数。小任务应比较降低核数和扩大可并行维度；不能一律 launch 历史案例中的 20/24 核。
- 核内 K 分块、跨核 split-K、Stream-K 是不同策略；后两者需要额外考察 partial、归并、同步、精度和完整链路成本。

## 数据流与完整成本

逐 tile 记录读取/写回、中间结果和消费者：

- 原布局直接访问、连续搬运后重排、UB 中间物化、多阶段 GM workspace 分别建模。
- 统计所有阶段 launch、scratch 写回和再读取、dtype 转换、同步及尾块，不把中间物化当成零成本寄存器重排。
- 比较相同有效 payload，记录重复计算/搬运量、有效 lane、常驻数据每核复制成本。
- 寻找不依赖输出 channel/head/table 的公共计算，估算外提或复用可消除的工作量；该估算不是实际加速比。
- 对流水列出每个 buffer 的生产→消费→复用顺序、版本数、首轮/稳态/尾轮及跨迭代状态。跨迭代 accumulator 必须保持合法依赖，不能简单给所有对象套双缓冲。

## 候选组合与依赖

将任务映射、tile、布局、精度和流水分别作为候选轴。有交互时为实施阶段设计完整组合实验，例如增大连续单元后重算 UB，改变版本数后重算 tile。自动流水失败仅说明该表达/配置失败，是否可用手动多版本要按当前 API、容量和依赖另行判断。

实施顺序由三类依赖决定：布局先于依赖该布局的访问；buffer 份数和生命周期先于最终预算；功能结构与同步正确性先于伴随的 pass 配置。精度或硬件限制可以排除候选，但须记录明确依据。

## 扩大 kernel 调用域前检查

为复用记录实际所需维度，而非复制一套与候选无关的表：shape/batch、dtype、stride、完整 tile/尾块、输入输出有效范围、stage/padding、workspace、模式与同步合同。

| 候选 | 扩大调用域 | 兼容性依据 | 未知/不满足项 | 处理 |
|---|---|---|---|---|

在方案中将处理标记为已证明可直接复用、先修复 kernel 或先实现新 kernel。存在未知项时，不能先接通 planner/dispatch 再用端到端结果掩盖 stage 越界。
