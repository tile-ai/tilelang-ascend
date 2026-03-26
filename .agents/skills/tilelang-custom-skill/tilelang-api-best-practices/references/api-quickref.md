# TileLang Ascend API 速查表

## Kernel 定义

| API | 说明 |
|-----|------|
| `@T.prim_func` | 定义 kernel 函数 |
| `T.Tensor((M, N), dtype)` | 声明张量参数 |
| `T.Kernel(block_num, is_npu=True) as (cid, vid)` | Kernel 启动上下文 |
| `@jit(out_idx=[-1], pass_configs={...})` | JIT 编译装饰器 |
| `T.dyn['K']` | 动态 shape（通过 buffer.shape 获取） |
| `T.dynamic('K', 'int32')` | 动态 shape（直接使用） |

## 内存分配

| API | 存储层级 | 用途 |
|-----|---------|------|
| `T.alloc_shared(shape, dtype)` | L1/UB（自动） | Developer 模式 |
| `T.alloc_fragment(shape, dtype)` | L0A/L0B/L0C（自动） | Developer 模式 |
| `T.alloc_ub(shape, dtype)` | Unified Buffer | Expert 模式 |
| `T.alloc_L1(shape, dtype)` | L1 Buffer | Expert 模式 |
| `T.alloc_L0A(shape, dtype)` | L0A Buffer | Expert 模式 |
| `T.alloc_L0B(shape, dtype)` | L0B Buffer | Expert 模式 |
| `T.alloc_L0C(shape, dtype)` | L0C Buffer | Expert 模式 |

## 数据搬运

| API | 说明 |
|-----|------|
| `T.copy(src, dst)` | 在 GM/L1/UB/L0 之间搬运数据 |

## 矩阵计算

| API | 说明 |
|-----|------|
| `T.gemm_v0(A, B, C, transpose_A, transpose_B, init)` | 标准 GEMM |
| `T.gemm_v1(A, B, C, transpose_A, transpose_B, init)` | 分层 GEMM |

## 归约

| API | 说明 |
|-----|------|
| `T.reduce_sum(buf, out, tmp, dim)` | 按维度求和 |
| `T.reduce_max(buf, out, tmp, dim)` | 按维度求最大值 |
| `T.reduce_min(buf, out, tmp, dim)` | 按维度求最小值 |

## 循环与调度

| API | 说明 |
|-----|------|
| `T.serial(N)` | 普通 for 循环 |
| `T.unroll(N)` | 循环展开 |
| `T.Parallel(ext0, ext1, ...)` | 元素级并行循环 |
| `T.Pipelined(range, num_stages=N)` | 流水线并行 |
| `T.Persistent(domain, wave_size, index)` | 持久化调度 |

## Element-wise 运算（T.Parallel 内）

| 运算 | 符号/API |
|------|---------|
| 加/减/乘/除 | `+`, `-`, `*`, `/` |
| 指数/对数 | `T.exp(x)`, `T.log(x)` |
| 绝对值 | `T.abs(x)` |
| 平方根 | `T.sqrt(x)`, `T.rsqrt(x)` |
| 最大/最小 | `T.max(a, b)`, `T.min(a, b)` |
| 位运算 | `~`, `<<`, `>>`, `&`, `\|` |

## Tile 扩展原语（Expert 模式）

| API | 说明 |
|-----|------|
| `T.tile.add/sub/mul/div(dst, src0, src1)` | 双目算术 |
| `T.tile.exp/ln/abs/sqrt/rsqrt/relu(dst, src)` | 单目运算 |
| `T.tile.reciprocal(dst, src)` | 取倒数 |
| `T.tile.leaky_relu(dst, src, scalar)` | Leaky ReLU |
| `T.tile.axpy(dst, src, scalar)` | dst = scalar*src + dst |
| `T.tile.sin/cos(dst, src, tmp)` | 三角函数 |
| `T.tile.compare(dst, src0, src1, mode)` | 逐元素比较 |
| `T.tile.select(dst, mask, src0, src1, mode)` | 条件选择 |
| `T.tile.cast(dst, src, mode, count)` | 精度转换 |
| `T.tile.fill(buffer, value)` | 数据填充 |
| `T.tile.transpose(dst, src)` | 16×16 转置 |
| `T.tile.createvecindex(dst, first_value)` | 创建向量索引 |
| `T.tile.gather(dst, src, offset, base_addr)` | 数据收集 |
| `T.tile.arith_progression(buf, first, diff, count)` | 等差数列 |
| `T.tile.sort(dst, src, indices, tmp, repeat)` | 排序 |
| `T.tile.merge_sort(dst, src, block_size, block_num, is_copy)` | 合并排序 |
| `T.tile.topk(dst, src, tmp, block_size)` | Top-K |

## 同步原语

| API | 说明 |
|-----|------|
| `T.set_flag(src, dst, eventId)` | 设置核内同步标志 |
| `T.wait_flag(src, dst, eventId)` | 等待核内同步标志 |
| `T.barrier_all()` | 全管线屏障 |
| `T.pipe_barrier(pipe)` | 指定管线屏障 |
| `T.sync_all()` | 全局同步 |
| `T.set_cross_flag(pipe, flag, mode)` | 设置核间同步标志 |
| `T.wait_cross_flag(flag, pipe)` | 等待核间同步标志 |

## 调试

| API | 说明 |
|-----|------|
| `T.printf(format_str, *args)` | 格式化打印 |
| `T.dump_tensor(tensor, desc, dump_size, shape_info)` | Tensor 转储 |
| `func.get_kernel_source()` | 查看生成的 AscendC 代码 |

## 常用 pass_configs

| 配置项 | 说明 |
|-------|------|
| `TL_ASCEND_AUTO_SYNC: True` | 自动同步插入 |
| `TL_ASCEND_MEMORY_PLANNING: True` | 自动内存规划 |
| `tl.ascend_auto_cv_combine: True` | 自动 CV 分离（核间流水线需要） |
| `tl.ascend_auto_cross_core_sync: True` | 自动核间同步（核间流水线需要） |
