# StreamK GEMM 算子设计文档（Ascend版本）

## 1. 概述

### 1.1 算子名称
gemm_streamk

### 1.2 功能描述
基于StreamK思想的负载均衡GEMM算子。使用T.Persistent进行tile级调度，实现多核负载均衡。

### 1.3 数学公式
$$
C = A \times B
$$

## 2. GPU StreamK vs Ascend适配

### 2.1 GPU StreamK核心思想

| 特性 | GPU实现 | 说明 |
|-----|--------|------|
| 负载均衡 | 动态调度 | 不同SM处理不同数量的tiles |
| K分割 | 可分割K维度 | 一个tile的K计算可由多个SM分担 |
| 原子累加 | `T.atomic_add` | partial tiles结果原子加 |

### 2.2 Ascend限制

| GPU特性 | Ascend支持 | 替代方案 |
|---------|-----------|---------|
| `T.atomic_add` | ❌ 不支持 | 使用workspace + 两阶段累加 |
| K维度分割 | ❌ 复杂 | 简化为tile级分割（不分割K） |
| 动态调度 | 部分支持 | 使用 `T.Persistent` |

### 2.3 Ascend适配方案

**方案：Persistent + Pipeline**

1. **T.Persistent**：进行tile级负载均衡
2. **T.Pipelined**：K维度流水线优化
3. **Workspace**：避免原子操作，使用显式累加

## 3. 算法设计

### 3.1 负载均衡策略

```
┌─────────────────────────────────────────────────┐
│                    M × N tiles                   │
├─────────────────────────────────────────────────┤
│  core_0: tiles[0], tiles[core_num], ...         │
│  core_1: tiles[1], tiles[core_num+1], ...       │
│  core_2: tiles[2], tiles[core_num+2], ...       │
│  ...                                             │
└─────────────────────────────────────────────────┘
```

每个core通过`T.Persistent`循环处理分配的tiles，实现负载均衡。

### 3.2 数据流

```
GM[A] → L1[A_L1] ──┐
                    │
GM[B] → L1[B_L1] ──┼──→ L0C[C_L0] → GM[C]
                    │
        T.Pipelined(num_stages=2)
```

### 3.3 核心代码结构

```python
with T.Kernel(core_num, is_npu=True) as (cid, _):
    A_L1 = T.alloc_L1((block_M, block_K), dtype)
    B_L1 = T.alloc_L1((block_K, block_N), dtype)
    C_L0 = T.alloc_L0C((block_M, block_N), accum_dtype)

    # Persistent循环：每个core处理多个tiles
    for bx, by in T.Persistent([m_num, n_num], core_num, cid):
        # Pipelined K维度迭代
        for k in T.Pipelined(k_num, num_stages=2):
            T.copy(A[bx * block_M, k * block_K], A_L1)
            T.copy(B[k * block_K, by * block_N], B_L1)
            T.gemm_v0(A_L1, B_L1, C_L0, init=(k == 0))

        T.copy(C_L0, C[bx * block_M, by * block_N])
```

## 4. 文件结构

```
examples/gemm_streamk/
├── example_gemm_streamk.py  # 算子实现
├── design_gemm_streamk.md   # 本设计文档
└── README.md                # 使用说明
```

## 5. 性能考虑

### 5.1 分块参数

| 参数 | 推荐值 | 说明 |
|-----|-------|------|
| block_M | 128 | M方向分块 |
| block_N | 256 | N方向分块 |
| block_K | 64 | K方向分块 |
| num_stages | 2-3 | 流水线阶段数 |

### 5.2 负载均衡效果

- 当 `tiles > core_num` 时，每个core处理多个tiles
- 相邻tiles在同一core处理，提高L1缓存利用率

## 6. 与GPU版本差异

| 方面 | GPU StreamK | Ascend版本 |
|-----|-------------|------------|
| K分割 | 支持 | 不支持 |
| 原子操作 | atomic_add | workspace累加 |
| 动态调度 | while循环 | T.Persistent |
| 流水线 | T.Pipelined | T.Pipelined |

## 7. 参考

- [tilelang/examples/gemm_streamk/](../../tilelang/examples/gemm_streamk/) - GPU版本
- [examples/gemm/example_gemm_persistent.py](../gemm/example_gemm_persistent.py) - Ascend Persistent
- [examples/pipeline/gemm_v0_pipeline.py](../pipeline/gemm_v0_pipeline.py) - Ascend Pipeline