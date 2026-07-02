---
name: tilelang-api-best-practices
description: TileLang Ascend API 使用最佳实践。提供内存分配、数据搬运、矩阵计算、归约、元素级运算、同步、调度原语等 API 的正确用法和最佳实践。触发：使用 TileLang API 编写 Ascend NPU kernel 时或遇到 API 相关问题时。
---

# TileLang Ascend API 最佳实践

---

## API 文档索引

| 文档 | 涵盖内容 | 典型场景 |
|------|---------|---------|
| [api-kernel-memory.md](references/api-kernel-memory.md) | Kernel 定义（T.prim_func, T.Kernel, @jit）、内存分配（Developer: T.alloc_shared/fragment/var, Expert: T.alloc_ub/L1/L0x）、数据搬运（T.copy） | Kernel 编写、片上存储管理、数据搬运 |
| [api-compute.md](references/api-compute.md) | 矩阵计算（T.gemm_v0, T.mma）、归约（T.reduce_sum/max/min）、Element-wise（T.Parallel + 符号 API）、Tile 扩展原语（T.tile.xxx，含 T.tile.atomic_add） | GEMM、Softmax、逐元素计算、排序、原子累加 |
| [api-schedule-sync.md](references/api-schedule-sync.md) | 循环（T.serial, T.unroll）、流水线（T.Pipelined）、持久化调度（T.Persistent）、同步（T.set_flag/wait_flag, T.barrier_all, T.set_cross_flag）、调试（T.printf, T.dump_tensor） | 流水线优化、多核均衡、同步、调试 |

---

## 场景索引

| 使用场景 | 相关文档 | 关键技巧 |
|---------|---------|---------|
| **GEMM 矩阵乘** | [api-compute](references/api-compute.md), [api-kernel-memory](references/api-kernel-memory.md) | shared→fragment 层级搬运、init 参数、T.barrier_all |
| **Softmax/LayerNorm** | [api-compute](references/api-compute.md) | T.reduce_max/sum、T.tile.exp/sub/div |
| **逐元素计算** | [api-compute](references/api-compute.md) | T.Parallel + 符号 API 或 T.tile.xxx 两种范式 |
| **多 block/core 累加到 GM** | [api-compute](references/api-compute.md) | T.tile.atomic_add(dst_gm, src_local)，调用前显式清零 GM |
| **CV 融合算子** | [api-kernel-memory](references/api-kernel-memory.md), [api-schedule-sync](references/api-schedule-sync.md) | workspace 索引一致性、AUTO_CV_COMBINE、vid 并行化 |
| **流水线优化** | [api-schedule-sync](references/api-schedule-sync.md) | T.Pipelined num_stages、核间/核内流水线 |
| **多核负载均衡** | [api-schedule-sync](references/api-schedule-sync.md) | T.Persistent 缓存友好调度 |
| **排序** | [api-compute](references/api-compute.md) | T.tile.sort → T.tile.merge_sort → T.tile.topk |
| **Kernel 调试** | [api-schedule-sync](references/api-schedule-sync.md) | T.printf、T.dump_tensor、get_kernel_source() |

---

## API 速查表

### Kernel 定义

| API | 说明 |
|-----|------|
| `@T.prim_func` | 定义 kernel 函数 |
| `T.Tensor((M, N), dtype)` | 声明张量参数 |
| `T.Kernel(block_num, is_npu=True) as (cid, vid)` | Kernel 启动上下文 |
| `@jit(out_idx=[-1], pass_configs={...})` | JIT 编译装饰器 |
| `T.dyn['K']` / `T.dynamic('K', 'int32')` | 动态 shape |

### 内存分配

| API | 说明 | 模式 |
|-----|------|------|
| `T.alloc_shared(shape, dtype)` | shared 层级（编译器自动判断 L1/UB） | Developer |
| `T.alloc_fragment(shape, dtype)` | fragment 层级（编译器自动判断 L0A/B/C） | Developer |
| `T.alloc_var(dtype, init=...)` | 标量变量 | Developer |
| `T.alloc_ub / T.alloc_L1 / T.alloc_L0A/L0B/L0C` | 显式指定存储层级 | Expert |

### 数据搬运与计算

| API | 说明 |
|-----|------|
| `T.copy(src, dst)` | GM/L1/UB/L0 之间搬运数据 |
| `T.tile.atomic_add(dst_gm, src_local)` | 将本地 tensor 原子累加到 GM；V1 支持 local/UB → GM |
| `T.gemm_v0(A, B, C, transpose_A, transpose_B, init)` | 标准 GEMM |
| `T.mma(A, B, C, init)` | NPU MMA 指令 |
| `T.reduce_sum/max/min(buffer, out, dim)` | 按维度归约 |

### 循环与调度

| API | 说明 |
|-----|------|
| `T.serial(N)` / `T.unroll(N)` | 普通循环 / 循环展开 |
| `T.Parallel(ext0, ext1, ...)` | 元素级并行循环 |
| `T.Pipelined(range, num_stages=N)` | 流水线并行 |
| `T.Persistent(domain, wave_size, index)` | 持久化调度 |

### 同步与调试

| API | 说明 |
|-----|------|
| `T.set_flag / T.wait_flag` | 核内流水线同步 |
| `T.barrier_all() / T.pipe_barrier(pipe)` | 管线屏障 |
| `T.set_cross_flag / T.wait_cross_flag` | 核间同步 |
| `T.sync_all()` | 全局同步 |
| `T.printf(fmt, *args)` | 设备端格式化打印 |
| `T.dump_tensor(tensor, desc, size, shape_info)` | Tensor 转储 |

### 常用 pass_configs

| 配置项 | 说明 |
|-------|------|
| `TL_ASCEND_AUTO_SYNC: True` | 自动同步插入 |
| `TL_ASCEND_MEMORY_PLANNING: True` | 自动内存规划 |
| `TL_ASCEND_AUTO_CV_COMBINE: True` | 自动 CV 分离（核间流水线） |
| `TL_ASCEND_AUTO_CV_SYNC: True` | 自动核间同步（核间流水线） |
| `TL_ASCEND_INJECT_TMP_BUFFER: True` | 自动注入 tmp_buffer（默认 True，关闭后需手动分配 `tmp_ub`） |
