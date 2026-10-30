[English](benchmark_fp8_mqa_logits.md) | **中文**

# GLM-5.3 K-Pool FP8 MQA Logits 性能基准（昇腾）

## 1. 算子说明

GLM-5.3-Flash 稀疏注意力 indexer 的分页池化历史打分。对每个 query 行，在
有效 pool 区间 `[pool_start, pool_end)` 内按 `pool_page_table` 从调用方持有
的分页 FP8 cache 中 gather 池化 key（E4M3 字节用 `uint8` 承载），用与压缩器
写入时相同的位级方案反量化，计算 32 个 indexer 头的打分（`keys @ query^T`），
乘缓存的 per-pool scale，经 ReLU gate 与按头加权后归约为每个 pool 一个 FP32
logit。范围外或无效页的输出为 `-inf`。

```
blk         = pool_page_table[page_table_rows[r], p // 64]
off         = p % 64
logits[r,p] = sum_h relu( dot(q[r,h,:], decode(k_cache[blk, off, :])) * scale_cache[blk, off] ) * w[r,h]
              当 p ∈ [pool_start[r], pool_end[r]) 且页有效
            = -inf 其他情况
```

- 输入：`query` [R, 32, 128] bf16（E4M3 值域无损承载；调用方把 per-head query
  scale 折入 `weights`）、`k_cache` [num_blocks, 64, 128] uint8（压缩器写出的
  E4M3 字节位模式）、`scale_cache` [num_blocks, 64] fp32、`weights` [R, 32]
  fp32、`pool_page_table` [num_page_rows, max_pages] int32、`page_table_rows`
  / `pool_starts` / `pool_ends` [R] int32
- 输出：`logits` [R, max_num_pools] fp32（kernel 写满 64-pool 对齐的 tile，
  宿主返回 `[:, :max_num_pools]` 视图）
- 参考实现：tile-ai/tilelang 主仓 `examples/kpool` 的
  `glm53_kpool_fp8_mqa_logits_reference`；判定口径见 §7

## 2. 软硬件环境

| 项 | 值 |
|------|-------|
| NPU | Ascend 910B3（Atlas A2） |
| CANN | 9.2.0 |
| 框架 | torch 2.7.1 + torch_npu 2.7.1.post4 |
| 基线 | 该算子在昇腾上没有已发布基线（下表为绝对耗时） |
| 测量口径 | NPUGraph 图捕获回放，kernel-only；每图 10/4/1 次调用（按 shape），20 次回放；三次运行取中位 |

## 3. 架构设计（每次调用两个 kernel）

| 组件 | 选择 | 理由 |
|-----------|--------|--------|
| Mask kernel | 每 64-pool tile 一个 AIV，按 `pool_starts`/`pool_ends` 与页有效性生成逐 lane 的 0 / -inf 掩码 | 64 lane 的 compare/select 与索引生成在单 AIV 上精确；小向量 compare 需 256B 对齐 |
| 主 kernel | 每 (row, 64-pool 页) 一个 block；两个 AIV 各负责 32-pool 半页：clamp 页表读、整半页 32x128B gather、整半页 E4M3 解码；Cube 每 block 一次 64x32x128 GEMM；Vector 做 scale、ReLU、按头权重与 32 头归约；一次静态写回 | 整半页搬运让 GM 标量访问保持 O(页数)；页级 GEMM 满足 L0 分形约束，L0C 累加器 8KB |
| CV 交互 | 显式 `workspace_keys` / `workspace_scores` GM 缓冲，各一对 producer/consumer；自动核间同步 | `threads=2` 的 vid-reduction 路径会 mis-split 本 kernel 的子区域拷贝与小 compare/broadcast/index（已用 probe 证实） |
| FP8 路径 | `k_cache` uint8 + kernel 内位级 E4M3 解码 | A2/A3 无 FP8 数据通路；字节与参考 cache 一致 |
| 宿主适配 | 仅 metadata 校验（无 `.item()`/device 读）；padded 输出 + 视图 | 热路径上不产生 D2H 往返 |

## 4. 性能（kernel-only，图捕获回放）

形状：page_size=64、head_dim=128、32 个 indexer 头、pool 区间全有效、页表覆盖
全部页（上游测试使用的形态）。

| R（行数） | max_num_pools | num_blocks | pool-tile 数（R x tiles） | kernel-only（us） | ns / pool-row |
|----------|---------------|------------|---------------------------|-------------------|---------------|
| 2 | 128 | 4 | 4 | 13.6 | 53.1 |
| 8 | 1,024 | 16 | 128 | 55.1 | 6.7 |
| 4 | 32,768 | 512 | 2,048 | 664.9 | 5.1 |
| 32 | 32,768 | 512 | 16,384 | 4,978.9 | 4.7 |

## 5. 性能分析

- 设备侧拆分（图捕获回放，R=4 / 32,768 pools）：mask kernel **163.9 us** +
  主 kernel **476.2 us**。mask kernel 的算术量很小却占约 1/4 设备时间：它按
  64-pool tile 每块只用一个 AIV，耗时主要来自逐块标量 metadata 读与小
  compare/select op。
- 主 kernel 为 CV 混合：每 block 两次 32-pool E4M3 解码（Vector）、一次
  64x32x128 GEMM（Cube），随后是 32x32 后处理（按行乘 scale、ReLU、按列乘
  头权重、32 头归约）与每个半页一次 128B 写回。
- 时间随 pool tile 数线性扩展：2,048 -> 16,384 blocks 为 665 -> 4,979 us
  （7.5x）。
- msprof 下同一次调用的两个 kernel 分别报告 270 us / 537 us（含 profiler
  开销）；Cube 管道只在主 kernel 中活跃。

## 6. 端到端（含宿主适配，墙钟）

| R（行数） | max_num_pools | E2E（us） |
|----------|---------------|----------|
| 2 | 128 | 1,196 |
| 8 | 1,024 | 1,224 |
| 4 | 32,768 | 1,225 |
| 32 | 32,768 | 5,312 |

小/中 shape 的 E2E 主要被本环境每次 kernel 调用约 0.45 ms 的宿主侧启动开销
主导（每次调用两个 kernel），不属于 kernel 交付本体。宿主适配只做 metadata
校验（不读 device 数值），墙钟里没有额外的 D2H 往返。

## 7. 精度

`test_example_glm53_kpool_fp8_mqa_logits.py` 全部用例通过（8/8）。比较是
位置敏感的：先严格比较 `-inf` 掩码，再对有限 logits 用 `rtol=2^-10`、
`atol=2^-16` 校验（冒烟用例 max diff 1.953e-3，容差内）。metadata（`uint8`
字节、`int32` 索引）按精确比较。用例覆盖全区间、部分/空/非页对齐区间、无效
页表项、非 64 倍数 `max_num_pools`、空输入 no-launch 路径与负向参数校验。
测试数据使用非负头权重（gated indexer 的实际形态）：此时 32 头求和没有相消，
fp32 容差成立；混合符号权重大数相消会超出该容差口径，属已知边界。
