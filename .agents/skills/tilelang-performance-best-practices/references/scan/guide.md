# Prefix Scan

## 适用条件

inclusive scan、cummin/cummax 及带 index 变体。先固定 axis、方向、dtype、累加和输出语义。

### 先固定依赖与合约

明确 inclusive/exclusive、forward/reverse、axis、value/index dtype、溢出、相等值取首/末和 NaN 传播。扫描轴存在前缀依赖；正交轴常有独立并行机会，见 [正交轴专题](orthogonal_axis_vectorization.md)。[cummin 示例](../../../../../examples/cann-bench/cummin/example_cummin.py) 可作为结构来源，但其 host NaN 探测、值域假设和特化参数均需审查，不继承性能结论。

候选包括 row-owned streaming、整行驻留、正交轴向量化、UB 中分块扫描，以及少数超长行的 local-prefix/chunk-total/fixup 三阶段。Sklansky/Kogge-Stone 树形网络是算法方向，需当前 API 对跨 lane 依赖及快照的可表达证据；不能直接搬其他平台线程/寄存器模板。

## TileLang 实现

行数足够时使用 row-owned streaming；短行可 full-load；极少超长行才使用 local-prefix、chunk-total scan、fixup 三 kernel 方案。lane-parallel 仅在对应 SIMD gather/shift 微测通过后启用。

当前结构参考为 [cummin 示例](../../../../../examples/cann-bench/cummin/example_cummin.py) 与 [Tile API](../../../../../tilelang/language/ascend_tile.py)。row-owned 方案由唯一 task 持有每行，沿最后连续轴分 tile，GM/UB 使用 T.copy；浮点累加 carry 使用 fp32，整数值/索引保持其精确 dtype。该分块设计不依赖 grid barrier，具体 scan 原语需按目标后端核验。没有附带硬件、版本、命令和原始结果的性能数字不作为生产 dispatch 依据。

### 扫描轴批量搬运

若每一步都 GM→UB、比较/更新、UB→GM，小 DMA 和等待可能主导。先选择连续正交轴宽度 N，再用剩余容量求扫描块 `block_L`：

```text
resident = N * 每lane运行状态字节 + 对齐/scratch/固定量
tile_bytes = block_L * N * 每元素所有活跃输入/输出/计算字节
tile_bytes 包含各自 versions
resident + tile_bytes <= 当前后端有效UB容量
```

先以任务 waves 和 compare/select 约束选择 N，再枚举 block_L；也比较不同 N/block_L 组合，不能先把一个轴吃满 UB。最小合法 tile 仍超限时调整 buffer，而非强行返回 1。

每块一次二维 GM→UB，UB 内按扫描位置顺序更新 `[1,N]` carry，最后批量写回 value/index。carry 跨块保持、每个独立 scan 初始化，不能随流式 buffer 盲目多版本。

### 首块、满块、尾块

对于静态 L 的 JIT factory，可用 Python 整数计算 `n_full=L//block_L` 与 `partial=L%block_L`：

1. L=0 按公开接口处理；L>0 首块先从位置 0 初始化 running value/index，再扫描该块其余位置。
2. 后续完整块从局部位置 0 开始消费前一块 carry；全局 index 为 `base+jj`。
3. 尾块只搬/算/写 `partial` 行，不能读未初始化 padding 并更新 carry。

若 block_L 大于 L，走只有首个部分块的路径，不能无条件搬完整首块。`block_L` 为外层记录数通常无需沿扫描轴增加向量对齐，但内层 N、每行 pitch 和二维搬运仍需合法。动态 L 按实际 DSL 写控制流，不将静态示意当动态支持证明。

历史实验记录过 `[1024,1024] fp16 dim=-1`、`block_L=170` 的约 `490→146 μs`、barrier 约 `1024→7` 观察；缺少完整环境与原始证据，该数字仅说明历史动机，不能用作当前收益或默认参数。应保留其结构原理：批量 DMA 摊薄 O(L) 次固定开销。

### 三阶段长行与树形方案

任务不足时可拆为 A：各 chunk 局部 prefix+total；B：按行 scan chunk totals；C：以先前 carry 修正每 chunk。三次 launch、workspace 和额外读写都计入；带 argmin/argmax 的 pair combine 必须保持 tie/NaN/index 语义。

Kogge-Stone 每层从**前一层快照**读取 `prev[i-d]`，不能原地串行污染；Sklansky 每组用下半组最后位置作 anchor。非二次幂 tail、跨 tile carry 与结合顺序均需验证。算法未实现时只作为设计候选。

## 精度门禁

cummin/cummax 的 tie-break、NaN 和 index 传播需独立 reference。

覆盖 cols=1、63/64/65、127/128/129、tile±1、多 tile、最大长度，rows 在向量核数以下/等于/以上，以及正负抵消、大小量、NaN/Inf 和溢出契约。浮点 cumsum 与 torch.cumsum(x.float(), dim=-1) 比较并检查每个跨 tile carry；cummin/cummax、整数与索引变体使用对应的精确 reference，不把所有 scan 转为 fp32。

### 验证

覆盖 L=1、二次幂±1、block_L±1、多块、最后短块、正交轴小于/等于/大于核容量、多个 batch、重复值和稀疏 NaN。value 和 index 分开精确/容差比较，index 不能因 value 正确而省略。端到端纳入必要输入/输出重排、host 探测和 dtype 转换。

## 性能门禁

按短/中/长行建立 dispatch，不能以单一 shape 决定。向量 lane scan 或 split-row 只有通过当前 lowering、精度及端到端比较后才替换 serial inner scan；不适用时保留正确 streaming/UB-local fallback，尤其不能为消除标量路径而缩小整数值域。

先通过目标后端 lowering 与定向精度测试。比较 serial correctness、已验证的向量 lane scan、1/2 stage DMA 和三 kernel split-row 的完整端到端 latency；报告有效带宽、Vector/MTE 时间、核利用率、workspace bytes 与所有 launch。

## 专题入口

- [正交轴与精确比较](orthogonal_axis_vectorization.md)
- [算法策略与接入](templates/usage_guide.md)
