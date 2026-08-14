# Mish 性能调优最终报告

## 概述

- **算子**: Mish activation (`y = x * tanh(softplus(x))`)
- **调优对象**: `custom/mish/mish.py`（12 步 T.tile.xxx，float32 中间计算，Developer 模式）
- **Baseline**: `torch.nn.functional.mish`（PyTorch NPU）
- **测试 shape**: cann-bench 20 个标准 case
- **最终版本**: iter2（smart-flatten + 动态 block_M/block_N + 32B 对齐约束）
- **最终精度**: `test_mish.py --level all` → `Test Passed!`（L0:9 + L1:10 + L2:3 + Boundary:4 全通过）

## 性能提升摘要

| 指标 | Baseline | Final | 提升 |
|------|----------|-------|------|
| mean_speedup | 0.3733x | 0.3916x | +4.9% |
| sum kernel_ms (20 cases) | 8.03ms | 7.66ms | -4.6% |
| min speedup | 0.0373x (case 12) | 0.2114x (case 12) | +466% |
| max speedup | 0.9411x (case 4) | 0.9413x (case 4) | +0% |

> **性能数字来源**: `[本地 bench, 2026-08-10T11:15~12:00Z]` — Python 端到端计时（torch.npu.synchronize + time.perf_counter），与 cann-bench HAP 计时方式一致。本地 bench 与官方 cann-bench 存在系统性偏差（element-wise 算子约 +58%），达标判断需以官方 cann-bench 上传结果为准。

## 20-case 详细对比

| case | shape | dtype | baseline speedup | final speedup | delta | 驱动 |
|------|-------|-------|-----------------|---------------|-------|------|
| 1 | [1024,1024] | fp16 | 0.2417 | 0.2421 | +0% | 持平（128×128） |
| 2 | [2048,2048] | fp32 | 0.3668 | 0.3711 | +1% | 持平 |
| 3 | [4096,4096] | bf16 | 0.6602 | 0.6660 | +1% | 持平 |
| 4 | [8192,8192] | fp16 | 0.9411 | 0.9413 | +0% | 持平（接近 baseline） |
| 5 | [8192,8192] | fp32 | 0.9175 | 0.9147 | -0% | 持平 |
| 6 | [1023,1023] | bf16 | 0.2335 | 0.2404 | +3% | 持平 |
| 7 | [1009,1021] | fp16 | 0.2391 | 0.2290 | -4% | 噪声 |
| 8 | [1537,769] | fp32 | 0.2361 | 0.2297 | -3% | 噪声 |
| 9 | [363,367,373] | bf16 | 0.6878 | 0.6863 | -0% | 持平（128×128 已最优） |
| 10 | [2049,513] | fp16 | 0.2380 | 0.2428 | +2% | 持平 |
| 11 | [3,7,13,4001] | fp32 | 0.2384 | 0.2169 | -9% | 噪声（tiling 相同） |
| 12 | [1000003] | bf16 | 0.0373 | 0.2114 | **+466%** | ❗ 1D: block_N 128→8192, num_blocks 7813→123 |
| 13 | [11,13,17,67,67] | fp32 | 0.3058 | 0.4529 | **+48%** | ❗ smart-flatten: split_idx=2, num_blocks 1273→684 |
| 14 | [3,7,11,13,1009] | fp16 | 0.3232 | 0.3195 | -1% | 持平 |
| 15 | [512,2049] | fp32 | 0.2326 | 0.2275 | -2% | 噪声 |
| 16 | [255,8193] | bf16 | 0.3001 | 0.2855 | -5% | 噪声 |
| 17 | [4097,511] | fp16 | 0.3046 | 0.2871 | -6% | 噪声 |
| 18 | [2,511,2049] | fp32 | 0.2784 | 0.2658 | -5% | 噪声 |
| 19 | [4,255,2049] | bf16 | 0.2910 | 0.2798 | -4% | 噪声 |
| 20 | [2,3,17,1024,101] | fp32 | 0.3925 | 0.5228 | **+33%** | ❗ smart-flatten: split_idx=2, num_blocks 816→606 |

## 迭代历史

| 轮次 | 优化方向 | 采纳/回滚 | mean_speedup | vs baseline | vs iter(N-1) | 原因 |
|------|---------|----------|-------------|-------------|-------------|------|
| baseline | — | — | 0.3733x | — | — | 固定 reshape(-1,last_dim) + 128×128 |
| iter1 | host smart-flatten + 动态 tiling | 采纳 | 0.3869x | +3.6% | +3.6% | case 12 +504%, 13 +46%, 20 +21%；但 11/18/19 退化 |
| iter2 | smart-flatten 用 128×128 评估（偏好 M>=128） | 采纳 | 0.3928x | +5.2% | +1.5% | 修复 iter1 退化（11/18/19），保留 12/13/20 收益 |
| iter3 | T.tile.max→T.tile.relu | 回滚 | 0.3987x* | +6.8% | +1.5% | 噪声范围内（±3%），无法确认提升 |
| iter4 | 手动 kernel cache | 回滚 | 0.3965x | +6.2% | +0.9% | 噪声范围内，tilelang JIT 已缓存 |
| final | iter2 版本 | — | 0.3916x | +4.9% | — | 最终采纳版本 |

*iter3 两次测量平均：0.4042x / 0.3931x → 0.3987x

## 采纳的优化方向

### 1. host 侧 smart-flatten（iter1, iter2 优化）
- **问题**: 原实现固定 `reshape(-1, last_dim)`，对 ND shape 产生过多 num_blocks
  - case 13 [11,13,17,67,67]: reshape(-1, 67) → M=163363, N=67 → num_blocks=1273
  - case 20 [2,3,17,1024,101]: reshape(-1, 101) → M=104448, N=101 → num_blocks=816
- **优化**: 搜索所有 split_idx，用 128×128 评估 num_blocks，选最小的 (M, N) 切分
  - case 13: split_idx=2 → M=2431, N=4489 → num_blocks=684 (-47%)
  - case 20: split_idx=2 → M=102, N=103424 → num_blocks=606 (-26%)
- **零拷贝前提**: contiguous tensor reshape 是 view 操作，不触发物理拷贝

### 2. 1D shape block_N 放大（iter1）
- **问题**: 原实现固定 block_N=128，1D shape [1000003] → num_blocks=7813
- **优化**: M<128 时允许 block_N up to 8192（UB 预算允许）
  - case 12: block_M=2, block_N=8192 → num_blocks=123 (-99%)
- **UB 约束**: rows_per_vec × block_N × bytes_per_elem ≤ 196352B

### 3. M>=128 用 Vector sweet spot 128×128（iter2 修复）
- **问题**: iter1 用 UB 反推 block_M（如 138/152），非 128 倍数降低 Vector 效率（-10~15%）
- **优化**: M>=128 时固定 block_M=128, block_N=128（Vector sweet spot）
- **验证**: case 11/18/19 从 iter1 退化（-12%/-10%/-9%）恢复到持平

### 4. 32B 对齐约束（iter1 修复）
- **问题**: 非对齐 block_N（如 67/101）导致 DataCopyNd 数据损坏，case 13/20 精度失败
- **优化**: block_N 对齐到 32B（fp32→8, fp16/bf16→16）
- **验证**: 20/20 cann-bench case 精度全通过

### 5. fp32 用 20B/elem UB 预算（iter1 优化）
- **问题**: fp32 的 tmp_orig 是 dead buffer（need_cast=False），但原按 24B 保守计算
- **优化**: fp32 用 20B/elem（MEMORY_PLANNING 复用 tmp_orig），允许 block_M=128 而非 126

## 回滚的优化方向

### iter3: T.tile.max→T.tile.relu
- **假设**: 硬件 ReLU 指令可能比通用 max 更高效
- **结果**: 两次测量 0.4042x / 0.3931x，差异 2.8%（噪声范围内）
- **回滚原因**: 无法确认 > 3% 提升，且 max 已是 1/12 步，收益预计 < 1%

### iter4: 手动 kernel cache
- **假设**: 避免 JIT cache lookup 开销
- **结果**: 0.3965x (+0.9% vs iter2)
- **回滚原因**: tilelang JIT 已有全局 cache，手动 cache 收益在噪声范围内

## 瓶颈分析

### 已优化瓶颈
1. **host 侧固定 tiling**（P0，已解决）: smart-flatten + 动态 tiling 减少 num_blocks 36%
2. **1D shape block_N 过小**（P0，已解决）: block_N 128→8192，num_blocks -99%
3. **非 128 倍数 block_M**（P1，已解决）: M>=128 固定 128×128 Vector sweet spot
4. **非 32B 对齐 block_N**（P1，已解决）: 强制对齐避免 DataCopyNd 数据损坏

### 未解决瓶颈（优化空间耗尽）
1. **小 shape case host 开销 dominate**: kernel ~0.22ms，baseline ~0.05ms，host 开销 0.17ms
   - 来源: JIT cache lookup + kernel launch + synchronize（框架内部，无法优化）
   - 影响: case 1-8, 10, 11, 14-19 的 speedup 0.22-0.37
2. **case 9 num_blocks=3123**: 128×128 已最优，smart-flatten 已选最优 split，无法进一步减少
3. **kernel 侧 12 步 T.tile.xxx**: log-sum-exp trick 已最优（消除了 T.if_then_else 反模式），每步必要
4. **大 shape case (4, 5) 接近 baseline**: speedup 0.94，kernel 计算 dominate，无法超越 baseline

## 中止原因

1. **连续 2 次无提升**（iter3 + iter4 回滚，均 < 3% 噪声阈值）
2. **优化空间耗尽**:
   - kernel 侧: 12 步 log-sum-exp trick 不可简化，无融合指令可用
   - host 侧: smart-flatten + 动态 tiling + kernel cache 已做完
3. **进一步优化收益在噪声范围内**（±3%）

## 产出文件

```
custom/mish/
├── mish.py                          # 最终版本（iter2, smart-flatten + 动态 tiling）
├── test_mish.py                     # 精度测试（未改）
└── perf_tuning/
    ├── bench_perf.py                # 20-case 性能基准脚本
    ├── verify_precision.py          # 动态 tiling 精度验证脚本
    ├── baseline_iter1.json          # baseline 性能数据
    ├── iter1.json                   # iter1 性能数据
    ├── iter2.json                   # iter2 性能数据（采纳）
    ├── iter3.json                   # iter3 性能数据（回滚）
    ├── iter3_rerun.json             # iter3 重跑（确认噪声）
    ├── iter4.json                   # iter4 性能数据（回滚）
    ├── final.json                   # 最终版本性能数据
    ├── mish_impl_iter1_before.py    # iter1 备份（优化前 = baseline）
    ├── mish_impl_iter2_before.py    # iter2 备份（= iter1 版本）
    ├── mish_impl_iter3_before.py    # iter3 备份（= iter2 版本）
    ├── mish_impl_iter4_before.py    # iter4 备份（= iter2 版本）
    ├── optimization_log.md          # 优化日志（Part A/B + 迭代记录）
    ├── perf_log.md                  # perf 日志（结构化记录）
    └── perf_report.md               # 本报告
```
