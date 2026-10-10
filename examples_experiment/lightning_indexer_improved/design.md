# Lightning Indexer 算子设计

## 1. 功能概述

Lightning Indexer 用于 NSA（Native Sparse Attention）的候选位置选择。算子计算 Query 与 Key 的分组相关性，对分数执行 ReLU 和分组权重归约，并沿 Key 序列维返回 Top-K 索引。

本实现使用 TileLang Expert 模式，在一个 MIX kernel 内完成矩阵乘、分数归约、分段排序和全局候选归并。实现文件同时提供 PyTorch golden 和单用例正确性检查函数；分级测试由独立测试文件组织。

## 2. 目录结构

```text
examples_experiment/lightning_indexer_improved/
├── design.md
├── example_lightning_indexer_dynamic_shape_improved.py  # 算子、golden、用例执行函数和 quick 入口
└── test_lightning_indexer_improved.py                    # L0/L1/L2/Boundary 分级测试
```

## 3. 接口定义

### 3.1 输入与输出

| 张量 | 形状 | 数据类型 | 说明 |
|---|---|---|---|
| `Query` | `[B, S1, N2, G * D]` | `float16` | 查询向量，G 个头在末维连续排列 |
| `KEY` | `[B, S2, N2, D]` | `float16` | Key 向量 |
| `WEIGHTS` | `[B, S1, N2, G]` | `float32` | 分组归约权重 |
| `OUT` | `[B, N2, S1, TOP_K]` | `int32` | 沿 S2 维选出的 Top-K 索引 |

数学定义如下：

```text
qk[b, s1, n2, g, s2] = relu(dot(Query[b, s1, n2, g, :], KEY[b, s2, n2, :]))
score[b, n2, s1, s2] = sum_g(qk[b, s1, n2, g, s2] * WEIGHTS[b, s1, n2, g])
OUT = topk_indices(score, TOP_K, dim=S2)
```

`index_golden` 使用 PyTorch 按上述定义计算参考结果。由于相同分数可能产生不同但等价的索引顺序，正确性检查按每行索引多重集合比较，并要求匹配率大于 0.99。

### 3.2 编译期参数

| 参数 | 支持范围 | 说明 |
|---|---|---|
| `BLOCK_M` | 64 | Query 行块；两个 AIV 各处理 32 行 |
| `BLOCK_N` | 64、128、256、512 | Key trunk 宽度；512 拆为两个 256 列 GEMM |
| `BLOCK_K` | 等于 D；D 为 64 或 128 | GEMM 归约维度 |
| `VECTOR_BASEN` | 等于 `BLOCK_N` | QK slot 搬运、归约和排序的列宽 |
| `VECTOR_BASEG` | G 的正因数；测试入口固定为 16 | G 维配置参数 |
| `S2_SPLITS` | 正整数且 `S2 % (S2_SPLITS * BLOCK_N) == 0` | S2 并行切分数 |

## 4. Kernel 设计

### 4.1 工作划分

Host 侧将一个工作项定义为 `(batch, n2, S1 tile, S2 split)`。grid 不超过设备 Cube 核数；当工作项多于 grid 时，每个 block 串行处理多个工作项。`auto_s2_splits` 在满足整除条件的候选中，根据工作项数量、trunk 数和归并行数选择切分数。

Device 侧采用 `MIX_AIC_1_2` 组织方式，每个 block 包含一个 AIC 和两个 AIV：

1. AIC 将 Query、Key 从 GM 搬运到 L1，执行 `gemm_v0`，并将 ReLU 后的矩阵乘结果写入双缓冲 `QK_SLOT`。
2. 两个 AIV 分担 `BLOCK_M` 行，读取对应 slot，完成分组权重乘、G 维归约、trunk 内排序，并将候选写入 `SORTED_WORKSPACE`。
3. 所有 trunk 完成后，AIV 对每行候选执行多路归并，输出全局 Top-K 索引。

### 4.2 在线 Top-K

每个 trunk 仅保留 `min(TOP_K, BLOCK_N)` 个候选。该截断满足：全局 Top-K 中属于某一 trunk 的元素，必然位于该 trunk 的局部 Top-K 内。因此，最终结果可由所有 trunk 的局部候选归并得到，无需保存完整 S2 分数矩阵。

Phase 2 每次将 history 与最多三个新 trunk 组成四路输入，通过 `merge_sort` 更新 history。最后仅将索引部分写入 `OUT`。

### 4.3 数据路径与缓冲

```text
Query/KEY (GM) -> L1 -> L0C -> QK_SLOT (GM)
QK_SLOT -> UB -> 加权归约 -> trunk 排序 -> SORTED_WORKSPACE (GM)
SORTED_WORKSPACE -> UB -> 多路归并 -> OUT (GM)
```

- `QK_SLOT[GRID_SIZE, 2, BLOCK_M, G, BLOCK_N]` 用于 AIC 与 AIV 间双缓冲传递。
- `SORTED_WORKSPACE[TASK_COUNT, BLOCK_M, TRUNKS_MAX, 2 * TRUNK_KEEP]` 保存分数与索引对。
- `BLOCK_N=512` 时，受单次 L0B 容量限制，AIC 使用两个 256 列 GEMM 串行覆盖一个 trunk。

### 4.4 同步策略

实现关闭自动内存规划和自动同步，显式管理以下依赖：

- 本地 `set_flag`/`wait_flag` 负责 MTE2、M、FIX、V 和 MTE3 管线之间的生产者/消费者关系。
- `set_cross_flag`/`wait_cross_flag` 维护 AIC 与 AIV 间两个 slot 的 READY/FREE 信用。
- 当 `S2_SPLITS > 1` 时，Phase 1 后执行 `sync_all`，确保 Phase 2 读取跨 block workspace 前所有写入已经完成。
- 事件 ID 在各有向管线的独立命名空间内使用，并保持生产与等待次数配对。

## 5. 支持域与约束

| 约束 | 原因 |
|---|---|
| `S1 % BLOCK_M == 0` | 未实现 S1 尾块处理 |
| `S2 % BLOCK_N == 0` | 未实现 trunk 尾块处理 |
| `S2 % (S2_SPLITS * BLOCK_N) == 0` | 每个 split 包含整数个 trunk |
| `0 < TOP_K <= S2 <= MAX_S2` | Top-K 定义域和 workspace 容量约束 |
| `G` 位于 `[8, 248]` 且为 8 的倍数 | `row_expand_mul` 行数约束 |
| `G % VECTOR_BASEG == 0` | G 维配置约束 |
| `VECTOR_BASEN == BLOCK_N` | 缓冲区列宽必须覆盖完整 trunk |
| `calc_dtype == "float"` | 列块乘按 fp32 实现 |
| `input_dtype == "float16"` | 当前 GEMM 和输入搬运路径仅按 fp16 验证 |
| `D == BLOCK_K` 且 D 为 64 或 128 | L1 缓冲和 GEMM K 维必须一致 |
| B、N2、S1、S2、块参数、切分数和 core 数为正整数 | 防止无效 shape、grid 和除零 |
| `MAX_S2 <= 2^24` | 索引在归并阶段以 fp32 保存，必须保持整数精确性 |

TOP_K 还受 Phase 2 UB 占用约束。当前支持上界为：`BLOCK_N=512` 时 1536，`BLOCK_N=256` 时 1920，`BLOCK_N=64/128` 时 2048。`validate_indexer_config` 在 kernel 构造前统一检查编译参数，并可同时检查 B、S1、S2 等运行形状。

## 6. 测试结构

算子文件的 `main()` 仅运行一个小形状 quick 用例。分级测试位于 `test_lightning_indexer_improved.py`：

- `l0`：最小阻塞精度门禁。
- `l1`：覆盖 D=64 的常规单 batch 配置和 D=128 的常规多 batch 配置。
- `l2`：覆盖数据类型、维度、块大小、整除关系、UB 上界和正整数约束的 Host 侧拦截。
- `boundary`：覆盖单 trunk 且 `TOP_K == S2`，以及当前支持的最大 TOP_K。
- `all`：依次执行全部级别。

```bash
python example_lightning_indexer_dynamic_shape_improved.py
python test_lightning_indexer_improved.py --level l0
python test_lightning_indexer_improved.py --level all
```
