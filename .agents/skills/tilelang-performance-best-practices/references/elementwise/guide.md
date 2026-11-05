# Elementwise 优化索引

## 选择实现

以 [单缓冲 Add](../../../../../examples/elementwise/elementwise_add.py) 和目标同类 kernel 为起点。保留既有模式：Developer 可用 `T.Parallel`，Expert/混合模式按 [Tile API](../../../../../tilelang/language/ascend_tile.py) 的 `T.tile.*`。示例存在仅证明参考结构，不代表目标 kernel 已通过设备测试或比当前 baseline 快。

逐 case 先记录 [任务树与数据流](tiling_task_vector_search.md)，再按连续搬运、任务粒度、向量化、融合和 [流水](double_buffer_design.md) 建候选。不用“elementwise 必然 memory-bound”代替 profiling。

## 技术专题

- [Tiling、任务与 Vector 数据流搜索](tiling_task_vector_search.md)：工作树、边界、物理路线与正交轴组合。
- [向量化与融合](vector_efficiency_design.md)：整 tile 计算、AXPY 等价性、dtype 与精度门禁。
- [索引短记录变换](indexed_short_record.md)：共享字段、累积布局、partial 合并与 JIT 配置。
- [多版本流水](double_buffer_design.md)：自动/手动版本、槽位所有权、首尾与生效判定。
