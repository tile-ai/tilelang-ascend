# Cube 任务映射、Fixed Core 与 Swizzle

## Fixed Core / persistent

当逻辑输出 tile 多于可用物理核时，可让每核在一次 launch 内循环多个 tile，复用
片上分配与 per-core workspace。任务数少时 launch 上限取有效任务数，避免空工作。
物理 Cube 核数来自选定设备，旧示例的 20 和 24 都不是通用常量。

```python
# Python 侧静态任务规划示意；kernel 内仍需合法尾块搬运。
logical_tasks = batch * ceildiv(M, block_M) * ceildiv(N, block_N)
active_cores = min(logical_tasks, available_cube_cores)
# core c 处理 c, c+active_cores, c+2*active_cores, ...
```

grid-stride 可让任务数差至多一；连续分配可能改善局部性，但需均匀处理余数。
若任务成本不同（稀疏、变长等），仅均分数量不等于均分耗时。每个逻辑任务仍必须
有边界 guard；fixed-core 不会修复 GM 越界。每次输出任务开始重置累加器，跨任务
复用的 L1/L0 槽遵守 [归还规则](pingpong_design.md)。

可读 [persistent 示例](../../../../../examples/gemm/example_gemm_intrinsic_persistent.py)
理解 `T.Persistent` 的结构；示例固定核数和 K 槽释放条件仍需审计。Fixed Core
节省多少取决于任务、编译器和调度，不把历史“约200us”作为固定收益模型。

## Swizzle

`T.use_swizzle` 改变逻辑 tile 的访问顺序，可改变 A/B 在 L2 的复用距离、并发访问和
地址计算成本。多个核读取相同 A tile 本身不是 bank conflict 的证明；它也可能形成
有益的 cache 复用。应比较映射，而不是默认“越分散越快”。

当前用法见 [intrinsic 示例](../../../../../examples/gemm/example_gemm_intrinsic.py)
中的 `T.use_swizzle(task, M, N, K, block_M, block_N, off=3)`。`off=3` 是该示例参数。
方案中先证明每个有效输出 tile 恰好覆盖一次，batch 和尾 tile 映射没有重复/遗漏，
再比较直接映射与 swizzle；保留不使用 swizzle 的合法回退。

## 与常驻数据的依赖

改变任务顺序可能缩短或打断 [Full-load](fullload_design.md) 的复用范围，因此规划
先明确 task→(batch,m,n) 和常驻地址，再安排常驻/预取。不要在常驻 kernel 上只换
索引映射却沿用旧输入缓存。

后续验证覆盖任务少于核数、不整除核数、大任务量、batch>1、尾块及重复调用；收益
看完整调用数据。旧 swizzle 正负案例见 [历史记录](historical_cases.md)，收益须在当前环境复验。
