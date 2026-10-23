# Cross Entropy Loss - 性能测试与优化路径

**中文** | [English](benchmark_summary.md)

## 优化路径

### 基线（原实现）
- 逐行串行循环：`for n_idx: T.tile.sub(x[n_idx,:], max[n_idx])` — 每个块 64 次串行 vector op
- 无 `pad_value`，C 不整除 block_C 时存在越界风险
- 单一 dtype 路径（float32 输入也要经过 `x_ub` 中转）

### 第 1 步：pad_value 尾数处理
- 两遍扫描的 `T.copy` 均增加 `pad_value=-inf`
- 消除 `C % block_C != 0` 时的越界读
- 保证尾数 tile 上的归约正确

### 第 2 步：broadcast 替代逐行循环
- `T.tile.broadcast(max_2d, tile_max, axis=1)` + `T.tile.sub(x_32, x_32, max_2d)` — 1 次整体运算替代 64 次串行
- float16 block_C=128：8.1ms → 3.9ms（2.08x）

### 第 3 步：broadcast axis=1 修复（reviewer）
- 全部 6 处 broadcast 显式指定 `axis=1`
- 修复 `block_C == block_N_2` 时的维度推断错误（59/64 行结果不匹配）

### 第 4 步：logsum_2d 外提（reviewer）
- `prev_max`/`prev_sum` 的 broadcast 移出第二层 `bc` 循环
- C=131072、block_C=128 时：每次迭代 2 次 broadcast → 总共 2 次（节省 2048 次运算）
- 引入 `logsum_2d` buffer；两步相减保持 FP32 数值稳定性

### 第 5 步：FP32 专用 kernel
- float32 输入跳过 `x_ub`，直接搬入 `x_32`（节省 64KB UB）
- block_C 可开到 192
- 去掉多余的 `l_n` buffer，`l_n_32` 直接写入 `loss`
- float32 block_C=192：8.1ms → 2.9ms（2.79x）

### 第 6 步：数据竞争修复（reviewer）
- `bn = (cid * VEC_NUM + vid) % n_2_num` 改为线性 `bn = cid * VEC_NUM + vid` + `if bn < n_2_num` 守卫
- 消除多余任务回绕时对 GM 的并发写

### 第 7 步：y_dtype 清理（reviewer）
- 从 `y_dtype` Literal 中移除 `int64`（AscendC 不支持 int64 Adds）

### 第 8 步：CombineCV pass（reviewer）
- pass_configs 增加 `TL_ASCEND_AUTO_CV_COMBINE: True`
- 移除手动 `with T.Scope("V"):` — scope 由 pass 自动处理

## 内部性能对比（N=4, C=131072）

### float16

| 版本 | block_C | 耗时 | vs 基线 |
|---------|---------|---------|-------------|
| 基线（逐行） | 128 | 8.1ms | 1.0x |
| 基线（逐行） | 256 | 4.6ms | 1.76x |
| +broadcast | 128 | 3.9ms | 2.08x |
| +broadcast+axis+logsum_2d | 128 | 3.7ms | 2.19x |

### float32

| 版本 | block_C | 耗时 | vs 基线 |
|---------|---------|---------|-------------|
| 基线（逐行） | 128 | 8.1ms | 1.0x |
| +broadcast+FP32 专用 | 128 | 3.5ms | 2.31x |
| +broadcast+FP32 专用 | 192 | 2.9ms | 2.79x |

## 与 CANN 对比（block_N=128, block_C=128）

| Shape (N, C) | 本 kernel | CANN 算子 | 比值 |
|--------------|-------------|---------------|-------|
| (1024, 1024) | 35.00us | 44.64us | 0.78x（快于 CANN） |
| (4, 131072) | 3333.14us | 26.66us | 125x（慢于 CANN） |

- **大批量小词表（N=1024, C=1024）**：比 CANN 快 1.27x
- **小批量大词表（N=4, C=131072）**：明显慢于 CANN；属于 tiling 设计问题

## 已知限制

当前 tiling 使用 `block_N x block_C` 二维块。对于小 N/大 C 场景（如 N=4, C=131072）：
- `block_N=128` 造成大量 padding 浪费
- `c_num=1024` 串行循环使访存带宽成为瓶颈

后续计划：增加小 N/大 C 专用 kernel 分支，将并行度拆分到 C 维。

## 测试覆盖（23 用例）

- dtype：float16 / float32 / bfloat16
- block_C：16 / 32 / 64 / 128 / 192
- block_N：16 / 32 / 64 / 128
- 方形 block / c_num=1 / 尾数处理 / batch 4~1024
