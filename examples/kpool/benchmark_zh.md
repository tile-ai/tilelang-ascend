[English](benchmark.md) | **中文**

# GLM-5.3 K-Pool Compress 性能基准（昇腾）

## 1. 算子说明

GLM-5.3-Flash 稀疏注意力 indexer 的融合压缩算子。对每个由 `pool_size` 个
token 组成的 pool：对 `slot_score + ape` 做逐维 softmax 池化、池化向量过
BF16 边界、归一化 Hadamard-128 旋转、每向量一个 FP32 2 的幂 absmax scale
的 E4M3 量化，并写入分页 cache。

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
| 并行/Tiling | 每 block 处理 32 个 pool，共 `ceil(P/32)` 个 block | 32 是 UB 预算内最大的 tile（P=64 内存规划失败） |
| FP8 路径 | 用 ASCEND cast 白名单 + 整数位运算模拟 E4M3，`k_cache` 用 uint8 承载 | A2/A3 无 FP8 数据通路；字节与参考完全一致 |
| Hadamard | 在 dim-major tile 的连续半区上做升序蝶形 | 与参考逐位一致，P<=128 保证编译时间正常 |
| 池化 | 按 slot 顺序串行累积 | 与参考求和顺序一致 |
| Scale | 位域法 `2^ceil(log2(absmax/448))`，`round_scale=True` 时逆用位构造精确求出 / `False` 时直接除法 | 与参考 scale 逐位一致 |
| 宿主适配 | `ape` 折入 scores（fp32）、slot-major 重排、按 32 补齐并用失效行填充、scale 散射 | 与参考的 kernel 内算术数值完全一致 |
| Scale 写回 | kernel 将 scale 连续写入 scratch，适配器用一次设备侧索引操作散射到 cache | 4 字节标量 GM 写在跨核共享 cache line 时会丢更新（工具链关闭了标量访存自动 DCCI）；128 字节整行拷贝是跨核安全的 |

## 4. 性能（kernel-only，图捕获回放）

形状：pool_size=4、head_dim=128、page_size=64、`num_blocks = P / 64`（cache
恰好容纳全部 pool）、`round_scale=True`。

| P（pool 数） | kernel（us） | ns / pool |
|-----------|-------------|-----------|
| 1,024 | 22.5 | 22.0 |
| 4,096 | 70.0 | 17.1 |
| 16,384 | 230.2 | 14.0 |

随 block 占用率上升而改善（1,024 个 pool 只有 32 个 block，小于 40 个 AIV）。

## 5. 端到端（含宿主适配，墙钟）

| P（pool 数） | E2E（us） | ns / pool |
|-----------|----------|-----------|
| 1,024 | 3,220 | 3,145 |
| 4,096 | 3,920 | 957 |
| 16,384 | 6,770 | 413 |

本环境下 E2E 主要被 torch_npu 的逐 op 宿主调度开销主导（7-8 个预处理/散射
op，约 0.45 ms/op），并非设备计算；该部分不属于 kernel 交付本体。后续可优化
方向：`ape` 内化进 kernel、减少宿主 op 数、scale 在 kernel 内完成落位。

## 6. 精度

`test_example_glm53_kpool_compress.py` 全部用例通过：写入行的每个 E4M3 字节
与 `glm53_kpool_reference` 相差不超过 1 ULP（通常逐字节相同），
`round_scale=True` 时 scale 逐位一致，未写入的 cache 槽位完全保持原值。
