---
name: tilelang-api-best-practices
description: TileLang Ascend API 使用最佳实践。提供内存分配、数据搬运、矩阵计算、归约、元素级运算、同步、调度原语等 API 的正确用法和最佳实践。触发：使用 TileLang API 编写 Ascend NPU kernel 时或遇到 API 相关问题时。
---

# TileLang Ascend API 最佳实践

---

## API 文档索引

| 文档 | 涵盖内容 | 典型场景 |
|------|---------|---------|
| [api-kernel-memory.md](references/api-kernel-memory.md) | Kernel 定义、JIT/`compile_flags`、内存分配、数据搬运 | Kernel 编写、编译配置、片上存储管理 |
| [api-compute.md](references/api-compute.md) | GEMM、归约、`tmp` arena、Element-wise、Tile 扩展原语 | GEMM、Softmax、逐元素计算、排序、显式 UB 规划 |
| [api-schedule-sync.md](references/api-schedule-sync.md) | 循环（T.serial, T.unroll）、流水线（T.Pipelined）、持久化调度（T.Persistent）、同步（T.set_flag/wait_flag, T.barrier_all, T.set_cross_flag）、调试（T.printf, T.dump_tensor） | 流水线优化、多核均衡、同步、调试 |

---

## 场景索引

| 使用场景 | 相关文档 | 关键技巧 |
|---------|---------|---------|
| **GEMM 矩阵乘** | [api-compute](references/api-compute.md), [api-kernel-memory](references/api-kernel-memory.md) | 内存层级、layout、`init` 与模式对应的同步 |
| **Softmax/LayerNorm** | [api-compute](references/api-compute.md) | T.reduce_max/sum、T.tile.exp/sub/div |
| **逐元素计算** | [api-compute](references/api-compute.md) | T.Parallel + 符号 API 或 T.tile.xxx 两种范式 |
| **多 block/core 累加到 GM** | [api-compute](references/api-compute.md) | T.tile.atomic_add(dst_gm, src_local)，调用前显式清零 GM |
| **CV 融合算子** | [api-kernel-memory](references/api-kernel-memory.md), [api-schedule-sync](references/api-schedule-sync.md) | Developer/Hybrid 用 `threads` + passes；Expert 用显式 scope/sync |
| **流水线优化** | [api-schedule-sync](references/api-schedule-sync.md) | T.Pipelined num_stages、核间/核内流水线 |
| **多核负载均衡** | [api-schedule-sync](references/api-schedule-sync.md) | T.Persistent 缓存友好调度 |
| **排序** | [api-compute](references/api-compute.md) | T.tile.sort → T.tile.merge_sort → T.tile.topk |
| **显式 UB scratch / row-expand** | [api-compute](references/api-compute.md) | 区分通用 `tmp` arena 与 row-expand 专用布局 |
| **JIT 编译选项** | [api-kernel-memory](references/api-kernel-memory.md) | 使用 kernel-scoped `compile_flags`，不修改进程环境 |
| **手动缓冲复用** | [api-schedule-sync](references/api-schedule-sync.md) | 同时审计 ready 与 free 两个 ownership 方向 |
| **Kernel 调试** | [api-schedule-sync](references/api-schedule-sync.md) | T.printf、T.dump_tensor、get_kernel_source() |
| **dtype 标量回退适配** | [api-compute](references/api-compute.md) | 先确认硬件支持；同宽 reinterpret / kernel 内 cast / record-aware DMA / 块 DMA + UB-local fallback；宽 dtype lane 拆分仅作已验证实验 |

---

## 使用规则

1. 先按场景读取一份 reference；不要一次加载全部文档。
2. Reference 中链接的 `docs/` 是公开契约，API 签名和支持范围以该文档及当前源码为准。
3. 写 kernel 前检查最接近的 `examples/`；文档未覆盖时再查
   `tilelang/language/ascend.py`、`ascend_tile.py` 和 `testing/python/language/`。
4. 不从旧代码或 PR 描述推断当前接口，也不使用 legacy `tilelang/language/pto.py`。
5. 更新本 skill 时只增加会改变 Agent 决策的规则；API 手册内容应在 canonical docs 中维护，
   这里通过链接引用，不复制第二份。
