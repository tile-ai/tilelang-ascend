---
name: tilelang-api-best-practices
description: TileLang Ascend API 使用最佳实践。提供内存分配、数据搬运、矩阵计算、归约、元素级运算、同步、调度原语等 API 的正确用法和最佳实践。触发：使用 TileLang API 编写 Ascend NPU kernel 时或遇到 API 相关问题时。
---

# TileLang Ascend API 最佳实践

---

## API 类别索引

| API 类别 | 涵盖 API | 核心文档 | 典型场景 |
|---------|---------|---------|---------|
| **Kernel 定义与启动** | T.prim_func, T.Kernel, T.Tensor, @jit | [api-kernel.md](references/api-kernel.md) | Kernel 定义、数据切分、JIT 编译 |
| **内存分配** | T.alloc_shared, T.alloc_fragment, T.alloc_ub, T.alloc_L1, T.alloc_L0A/L0B/L0C | [api-memory.md](references/api-memory.md) | 片上存储管理、Cube/Vector 内存 |
| **数据搬运** | T.copy | [api-datacopy.md](references/api-datacopy.md) | GM↔L1/UB、L1↔L0、L0C↔GM 数据搬运 |
| **矩阵计算** | T.gemm_v0, T.gemm_v1 | [api-gemm.md](references/api-gemm.md) | GEMM、矩阵乘累加 |
| **归约操作** | T.reduce_sum, T.reduce_max, T.reduce_min | [api-reduce.md](references/api-reduce.md) | Softmax、LayerNorm、ReduceMean |
| **Element-wise 运算** | T.Parallel, +/-/\*/÷, T.exp, T.log, T.abs 等 | [api-elementwise.md](references/api-elementwise.md) | 逐元素计算、激活函数 |
| **Tile 扩展原语** | T.tile.add/sub/mul/div/exp/ln/cast/compare/select 等 | [api-tile-ops.md](references/api-tile-ops.md) | Expert 模式向量操作 |
| **调度原语** | T.Pipelined, T.Persistent, T.serial, T.unroll | [api-schedule.md](references/api-schedule.md) | 流水线并行、负载均衡 |
| **同步原语** | T.set_flag, T.wait_flag, T.barrier_all, T.set_cross_flag 等 | [api-sync.md](references/api-sync.md) | 核内/核间同步 |
| **调试工具** | T.printf, T.dump_tensor | [api-debug.md](references/api-debug.md) | Kernel 调试、Tensor 转储 |

---

## 场景索引

| 使用场景 | 相关文档 | 关键技巧 |
|---------|---------|---------|
| **GEMM 矩阵乘** | [api-gemm.md](references/api-gemm.md), [api-memory.md](references/api-memory.md) | shared→fragment 层级搬运、init 初始化 |
| **Softmax/LayerNorm** | [api-reduce.md](references/api-reduce.md), [api-elementwise.md](references/api-elementwise.md) | reduce_max/reduce_sum、T.exp、广播 |
| **逐元素计算** | [api-elementwise.md](references/api-elementwise.md), [api-tile-ops.md](references/api-tile-ops.md) | T.Parallel 或 T.tile.xxx 两种范式 |
| **流水线优化** | [api-schedule.md](references/api-schedule.md), [api-sync.md](references/api-sync.md) | T.Pipelined num_stages、核间流水线 |
| **多核负载均衡** | [api-schedule.md](references/api-schedule.md) | T.Persistent 缓存友好调度 |
| **混合编程** | [api-tile-ops.md](references/api-tile-ops.md), [api-elementwise.md](references/api-elementwise.md) | Developer + Expert 混合模式 |
| **Kernel 调试** | [api-debug.md](references/api-debug.md) | T.printf 格式化输出、T.dump_tensor 转储 |
| **性能调优** | [api-schedule.md](references/api-schedule.md) | msProf 工具、Roofline 分析 |

---

## 快速参考

完整的 API 参数速查表：[api-quickref.md](references/api-quickref.md)
