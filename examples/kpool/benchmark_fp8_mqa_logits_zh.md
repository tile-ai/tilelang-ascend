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

## 3. 架构设计（每次调用一个 kernel）

| 组件 | 选择 | 理由 |
|-----------|--------|--------|
| 主 kernel | 每个 block 负责一行的 `pages_per_block` 个连续 64-pool 页；两个 AIV 各负责每页的 32-pool 半页：clamp 页表读、整半页 gather、整半页 E4M3 解码；Cube 每页一次 64x32x128 GEMM；Vector 做 scale、ReLU、按头权重、32 头归约与掩码；每个半页一次静态写回 | 多页合块摊薄每块的 query 加载、页表 metadata 与核间握手的开销；页级 GEMM 满足 L0 分形约束，L0C 累加器 8KB |
| 页批大小 | `pages_per_block` 默认取"每行刚好铺满 AI Core 数"的页数，上限 8 | 910B3 实测 8 页/块优于 1/2/4/16 页；自动核间同步按页发一对 flag，flag 计数器在 16 页以上会饱和挂死 |
| 有效性掩码 | kernel 内按 `[32 pools, 32 heads]` tile 生成 0 / -inf，归约后相加；全有效页走 fast path 直接跳过 | 去掉独立 mask kernel（原占约 25% 设备时间），且 compare/select 都高于 256B 向量指令下限 |
| CV 交互 | 显式 `workspace_keys` / `workspace_scores` GM 缓冲，各一对 producer/consumer；自动核间同步 | `threads=2` 的 vid-reduction 路径会 mis-split 本 kernel 的子区域拷贝与小 compare/broadcast/index（已用 probe 证实） |
| FP8 路径 | `k_cache` uint8 + kernel 内位级 E4M3 解码 | A2/A3 无 FP8 数据通路；字节与参考 cache 一致 |
| 宿主适配 | 仅 metadata 校验（无 `.item()`/device 读）；padded 输出 + 视图 | 热路径上不产生 D2H 往返 |

## 4. 性能（kernel-only，图捕获回放）

形状：page_size=64、head_dim=128、32 个 indexer 头、pool 区间全有效、页表覆盖
全部页（上游测试使用的形态）。

| R（行数） | max_num_pools | grid（block 数） | 页/块 | kernel-only（us） | ns / pool-row |
|----------|---------------|------------------|-------|-------------------|---------------|
| 2 | 128 | 4 | 1 | 8.9 | 34.8 |
| 8 | 1,024 | 16 | 8 | 28.3 | 3.5 |
| 4 | 32,768 | 256 | 8 | 283.5 | 2.2 |
| 32 | 32,768 | 2,048 | 8 | 2,166.6 | 2.1 |

当前数字包含 fast-path mask 与 bit-sign 解码（§8）；相对优化前
（330.5 / 2,557.5 us）分别快 14.2%（R=4）和 15.3%（R=32）。R=2/R=8 也在
同一构建上重新测过。

## 5. 性能分析

msprof（`op`，R=4 / 32,768 pools，**优化前基线**：Task Duration 338.6 us，
Block Dim 256）：

| 管道（AIV，按 40 AIV 归一） | 利用率 |
|-----------------------------|--------|
| scalar | 69.2% |
| vector | 52.8% |
| MTE2 | 16.3% |
| MTE3 | 8.3% |

- AIV 在基线上忙碌约占 block 时长的 94%；scalar 与 vector 都明显高于 50%，
  属于指令发射/计算受限，而不是带宽受限。当前 283.5 us 下有效带宽约
  120 MB / 283.5 us ≈ 423 GB/s，约为 1.2 TB/s HBM 峰值的 35%。
- E4M3 位级解码是最大的单项开销：把它换成一次平凡 cast 后，基线 R=4 /
  32,768 pools 从 330 us 降到 232 us，即解码链约占 kernel 的 30%。在
  fast-path mask 与 bit-sign 解码之后 kernel 为 283.5 us，解码链仍是最大
  单项开销。Cube 基本空闲（cube 1.8%、MTE 6.7%、FIX 5.4% 的 AIC 时间），
  因为每页的 64x32x128 GEMM 很小。
- 多页合块是去掉 per-block 开销的关键：同一 kernel 在 1 页/块时 543 us；
  8 页/块基线 330 us，当前构建 283.5 us。再往上受核间 flag 计数器限制
  （16 页会挂死，8 页安全）。

## 6. 端到端（含宿主适配，墙钟）

| R（行数） | max_num_pools | E2E（us） |
|----------|---------------|-----------|
| 2 | 128 | 387 |
| 8 | 1,024 | 401 |
| 4 | 32,768 | 354 |
| 32 | 32,768 | 2,565 |

小 shape 的 E2E 由宿主主导：kernel 本身只有 9.6~32 us，而本环境单次调用约
350~400 us（校验 + JIT 派发 + workspace 分配 + launch）。宿主适配只做
metadata 校验（不读 device 数值），墙钟里没有额外的 D2H 往返。该表是基线
宿主测量，受宿主负载影响且不随 kernel 改动逐次重测；性能门禁以 §4 的
kernel-only 数字为准。

## 7. 精度

`test_example_glm53_kpool_fp8_mqa_logits.py` 全部用例通过（10/10）。比较是
位置敏感的：先严格比较 `-inf` 掩码，再对有限 logits 用 `rtol=2^-10`、
`atol=2^-16` 校验（冒烟用例 max diff 1.953e-3，容差内）。metadata（`uint8`
字节、`int32` 索引）按精确比较。用例覆盖全区间、部分/空/非页对齐区间、无效
页表项、非 64 倍数 `max_num_pools`、多页合块（含非页对齐的合块尾块）、空输入
no-launch 路径与负向参数校验。测试数据使用非负头权重（gated indexer 的实际
形态）：此时 32 头求和没有相消，fp32 容差成立；混合符号权重大数相消会超出该
容差口径，属已知边界。

## 8. 优化记录

- 全有效页走 fast path，直接跳过 `-inf` 掩码；部分有效/无效页保留通用路径。
  R=4 / 32,768 pools 下 kernel 时间约降低 12%。
- E4M3 符号改为直接设置 FP32 符号位，去掉符号乘法，节省一条向量指令和
  16 KB UB。
- `T.Pipelined` 解码双缓冲目前被代码生成限制阻塞：流水化后的循环体会重复
  生成 `T.reinterpretcast` 的局部变量（`dfnorm` / `dmag` / `dfval`），在
  `pages_per_block > 1` 时编译失败。手动双缓冲（`AUTO_SYNC=False` + 手动
  flag）留作后续工作。
