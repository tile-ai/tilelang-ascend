[English](benchmark.md) | **中文**

# GLM-5.3 K-Pool Compress 性能基准（昇腾）

## 1. 算子说明

GLM-5.3-Flash 稀疏注意力 indexer 的融合压缩算子。对每个由 `pool_size` 个
token 组成的 pool：对 `slot_score + ape` 做逐维 softmax 池化、池化向量过
BF16 边界、归一化 Hadamard-128 旋转、每向量一个 FP32 2 的幂 absmax scale
的 E4M3 量化；E4M3 行与 scale 写入连续 scratch，由宿主适配器散射进分页
cache。

- 输入：`slot_k` [P, pool_size, 128] bf16、`slot_score` [P, pool_size, 128]
  bf16、`ape` [pool_size, 128] fp32、`loc` [P] int64、`write_mask` [P] bool
- Cache：`k_cache` [num_blocks, page_size, 128] uint8（E4M3 字节位模式）、
  `scale_cache` [num_blocks, page_size] fp32（原地写）
- 参考实现：tile-ai/tilelang 主仓 `examples/kpool` 的
  `glm53_kpool_reference`；判定口径：写入行每个 E4M3 字节与参考相差不超过
  1 ULP，scale 逐位一致

## 2. 软硬件环境

| 项 | 值 |
|------|-------|
| NPU | Ascend 910B3（Atlas A2） |
| CANN | 9.2.0 |
| 框架 | torch 2.7.1 + torch_npu 2.7.1.post4 |
| 基线 | 该算子在昇腾上没有已发布基线（下表为绝对耗时） |
| 测量口径 | NPUGraph 图捕获回放，kernel-only；预热 5 次，20 次回放 x 每次 10 个 kernel |

## 3. 架构设计（单 kernel、纯 Vector）

| 组件 | 选择 | 理由 |
|-----------|--------|--------|
| 并行/Tiling | 每 AIV 子核处理 32 个 pool（每 block 覆盖 64 个 pool，共 `ceil(P/64)` 个 block） | 32 是 UB 预算内最大的 tile（P=64 内存规划失败）；block 的两个子核独立并行 |
| FP8 路径 | 用 ASCEND cast 白名单 + 整数位运算模拟 E4M3，`k_cache` 用 uint8 承载 | A2/A3 无 FP8 数据通路；字节与参考完全一致 |
| Hadamard | 在 dim-major tile 的连续半区上做升序蝶形 | 与参考逐位一致，P<=128 保证编译时间正常 |
| 池化 | 按 slot 顺序串行累积 | 与参考求和顺序一致 |
| Scale | 位域法 `2^ceil(log2(absmax/448))`，`round_scale=True` 时逆用位构造精确求出 / `False` 时直接除法 | 与参考 scale 逐位一致 |
| 宿主适配 | `ape` 折入 scores（fp32）、slot-major 重排、按 64 补齐；`write_mask` 为隐含全真时跳过 mask gather | 与参考的 kernel 内算术数值完全一致 |
| Cache/scale 写回 | kernel 将 E4M3 行与 scale 连续写入两个 scratch，适配器各用一次设备侧索引操作散射进 cache | 设备侧逐行散射循环会被标量访存周围的保守 PIPE_ALL 屏障串行化（约占总 kernel 时间 30%）；且 4 字节标量 GM 写在跨核共享 cache line 时会丢更新 |

## 4. 性能（kernel-only，图捕获回放）

形状：pool_size=4、head_dim=128、page_size=64、`num_blocks = P / 64`（cache
恰好容纳全部 pool）、`round_scale=True`。

| P（pool 数） | kernel（us） | ns / pool |
|-----------|-------------|-----------|
| 1,024 | 16.5 | 16.1 |
| 4,096 | 49.0 | 12.0 |
| 16,384 | 149.2 | 9.1 |

随 block 占用率上升而改善（1,024 个 pool 为 16 个 block，每个 block 使用两
个 AIV 子核；该部件共 20 个 AI core）。

## 5. 端到端（含宿主适配，墙钟）

| P（pool 数） | E2E（us） | ns / pool |
|-----------|----------|-----------|
| 1,024 | 2,200 | 2,148 |
| 4,096 | 3,135 | 765 |
| 16,384 | 6,725 | 410 |

本环境下 E2E 主要被 torch_npu 的逐 op 宿主调度开销主导（预处理/散射 op 约
0.45 ms/op），并非设备计算；该部分不属于 kernel 交付本体。后续可优化方向：
`ape` 内化进 kernel、减少宿主预处理 op 数。

## 6. 精度

`test_example_glm53_kpool_compress.py` 全部用例通过。对标主仓的小用例保持上游
字节级口径：写入行的每个 E4M3 字节与 `glm53_kpool_reference` 相差不超过 1 ULP
（通常逐字节相同）。参数化的多 tile 扫描用例（300/1024/4096/16384 pools，
单用例最多 210 万元素）按"每个元素不超过 1 个 E4M3 码位或 1 个 per-pool
scale 单位"判定——在该元素规模下字节级 1 ULP 口径不具备统计稳定性：罕见的
bf16 池化边界穿越会被 Hadamard 对近零维度的相消放大。对 4 种 shape × 90 次
运行（约 4400 万元素）实测，多码位偏差不超过 0.25 个 scale 单位。
`round_scale=True` 时 scale 逐位一致，未写入的 cache 槽位在所有用例中完全
保持原值。
