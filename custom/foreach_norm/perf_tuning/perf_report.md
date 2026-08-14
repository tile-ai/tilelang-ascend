# ForeachNorm 性能调优最终报告

## 1. 结果概要

| 指标 | 值 |
|------|-----|
| 目标 | 平均加速比 ≥ 0.6× (baseline_compare vs torch.norm) |
| 最终结果 | 平均加速比 = **0.2095** |
| 是否达标 | **NO** |
| 迭代次数 | 7 (iter1~iter7) |
| 采纳迭代 | iter1 (基线), iter4 (batch +22%), iter6 (1D+smart batch), iter7 (torch.pow) |
| 回滚迭代 | iter2, iter3, iter5 |
| 精度 | PASS (L0/L1/L2/Boundary 全过) |
| 中止原因 | **[DESIGN_ERROR]** TileLang launch 开销使 0.6× 目标数学上不可达 |

## 2. 20 用例性能明细 (warmup=5, iters=20, median)

| case | shape | dtype | scalar | tl | baseline_us | ours_us | speedup |
|------|-------|-------|--------|----|-------------|---------|---------|
| 1 | [1024,1024] x2 | fp16 | 1.0 | 2 | 68.9 | 338.8 | 0.203 |
| 2 | [2048,2048] x3 | fp32 | 1.0 | 3 | 91.0 | 373.8 | 0.244 |
| 3 | [4096,4096] | bf16 | 1.0 | 1 | 73.6 | 353.5 | 0.208 |
| 4 | [2048,2048] | fp16 | 2.0 | 1 | 58.9 | 303.4 | 0.194 |
| 5 | [2048,4096] x3 | fp32 | 3.0 | 3 | 132.0 | 624.9 | 0.211 |
| 6 | [1023,1023] | bf16 | 1.5 | 1 | 56.3 | 317.6 | 0.177 |
| 7 | [1009,1021] | fp16 | 1.5 | 1 | 53.6 | 311.4 | 0.172 |
| 8 | [1537,769] | fp32 | 4.0 | 1 | 57.4 | 283.7 | 0.202 |
| 9 | [363,367,373] x2 | bf16 | 2.0 | 2 | 321.8 | 1261.2 | 0.255 |
| 10 | [2049,513] | fp16 | 1.0 | 1 | 52.2 | 285.9 | 0.183 |
| 11 | [3,7,13,4001] x3 | fp32 | 2.0 | 3 | 91.1 | 350.0 | 0.260 |
| 12 | [1000003] | bf16 | inf | 1 | 52.1 | 286.9 | 0.182 |
| 13 | [11,13,17,67,67] | fp32 | 5.0 | 1 | 90.5 | 372.3 | 0.243 |
| 14 | [3,7,11,13,1009] | fp16 | 2.0 | 1 | 56.7 | 304.7 | 0.186 |
| 15 | [512,2049] x2 | fp32 | 2.0 | 2 | 70.3 | 342.9 | 0.205 |
| 16 | [255,8193] x4 | bf16 | 1.0 | 4 | 109.3 | 375.5 | 0.291 |
| 17 | [4097,511] | fp16 | -1.0 | 1 | 58.6 | 320.3 | 0.183 |
| 18 | [2,511,2049] x2 | fp32 | 2.0 | 2 | 72.2 | 352.7 | 0.205 |
| 19 | [4,255,2049] x2 | bf16 | 3.0 | 2 | 73.5 | 378.6 | 0.194 |
| 20 | [2,3,17,1024,101] x4 | fp32 | 2.5 | 4 | 188.6 | 987.5 | 0.191 |
| **AVG** | | | | | **83.1** | **414.2** | **0.2095** |

## 3. 优化路径

| 迭代 | 策略 | avg_speedup | 改善 | 结果 |
|------|------|------------|------|------|
| iter1 | 多核 partial reduction + host finalize | 0.1713 | — | 基线 |
| iter2 | finalize kernel 内化到 NPU | 0.1360 | -20.7% | ROLLBACK (TileLang launch > CANN op) |
| iter3 | block_N=16384 for fp16/bf16 | 0.1707 | -0.3% | ROLLBACK (噪声) |
| iter4 | 多 tensor batch (1 launch for list_len tensors) | 0.2093 | +22.2% | **ADOPTED** |
| iter5 | T.Pipelined 双 buffer | 0.2040 | -2.5% | ROLLBACK (loop 太短) |
| iter6 | 1D 快速路径 + smart batch 阈值 | 0.2092 | -0.05% | KEPT (架构优化) |
| iter7 | torch.pow 替代 log/div/exp | 0.2095 | +0.14% | KEPT (代码简化) |

## 4. [DESIGN_ERROR] 分析

### 4.1 根因：TileLang launch 开销

通过 trivial kernel 实测：
- **TileLang kernel launch: 185us/call** (含 Python dispatch + argument marshal + CANN launch + sync)
- **CANN native op (torch.sum/norm): 44us/call**

每个 tensor 需要 1 次 TileLang launch (185us) + 2~5 次 CANN finalize op (88~220us) = 273~405us 固定开销。

### 4.2 数学不可达证明

设每 case 最低耗时 = 195us (1 TileLang launch + 0 compute + 10us Python):
- 13/20 case 的 baseline < 117us → 即使零计算，speedup = baseline/195 < 0.6
- 全部 20 case 的理论最高平均 speedup (零计算) = avg(83.1/195) ≈ **0.44**
- 目标 0.6 要求 avg_ours ≤ 83.1/0.6 = 138.5us，但 195us > 138.5us → **不可达**

### 4.3 6 种优化方案及结论

1. **finalize kernel 内化** (iter2): TileLang 1 launch (185us) 比 CANN 2~5 ops (88~220us) 更贵或相当 → 退化
2. **block_N 增大** (iter3): FP16/BF16 从 8192→16384，tile 数减半但 per-tile MTE2+V 时间翻倍 → 无改善
3. **多 tensor batch** (iter4): 同 shape tensor stack 成 (batch,N)，1 launch 处理全部 → **+22.2%**（唯一有效优化）
4. **T.Pipelined 双 buffer** (iter5): single_core_load=3~6 太短，pipeline fill/drain 占比过大 → 无改善
5. **1D 快速路径** (iter6): batch=1 用 1D kernel 避免 2D T.copy 开销 → 修复单 tensor 退化
6. **torch.pow 简化** (iter7): Lp finalize 从 5 CANN op 降到 3 → 噪声范围内

### 4.4 可能的解决方向（超出 Stage 3 范围）

- 降低 TileLang dispatch 开销（需框架层优化，如 AOT 编译或 C++ 直调）
- 混合策略：小 tensor 用 CANN 原生 torch.norm，大 tensor 用 TileLang（但这等于用 baseline）
- 更高 CORE_NUM 硬件（如 910C 的 48 核）

## 5. 精度复验

- L0: PASS (15 cases)
- L1: PASS (19 cases)
- L2/Boundary: PASS (8 cases)
- 覆盖门禁: PASS
- precision_degraded: false

## 6. 最终 kernel 架构

```
foreach_norm(x_list, scalar)
├── 按 flattened N 分组
├── 对每组:
│   ├── _should_batch(N, batch, dtype) == True:
│   │   └── 2D batched kernel (1 launch, outer T.serial(batch) + inner strided tiles)
│   │       → Partial (batch, launch_cores) → batched host finalize → (batch,)
│   └── _should_batch == False (batch=1 或大 N):
│       └── 1D kernel (per-tensor, 1 launch each)
│           → Partial (launch_cores,) → single host finalize → 0-dim
├── 7 kernel 特化: L0/L1/L2/Linf/Lneg-inf/Lp (1D + 2D batched = 14 JIT kernels)
└── Host finalize: sum/max/min + sqrt/pow + cast (2~5 CANN ops)
```
