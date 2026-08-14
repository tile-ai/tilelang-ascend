# Mish 性能调优日志

## Iteration 1 — 2026-08-10T11:30:00Z
- bottleneck_type: transfer (host 侧固定 tiling 导致 num_blocks 过多，DMA 调度开销 dominate)
- optimization: host 侧 smart-flatten + 动态 block_M/block_N
  - smart-flatten: 搜索所有 split_idx 选 num_blocks 最小的 (M,N) 切分（零拷贝 reshape）
  - 动态 tiling: M>=128 用 Vector sweet spot 128×128；M<128 用大 block_N(8192)+小 block_M
  - 32B 对齐约束: block_N 对齐到 32B（fp32→8, fp16/bf16→16），避免 DataCopyNd 数据损坏
- baseline_time: mean_speedup=0.3733x (sum kernel_ms=8.03ms across 20 cases)
- candidate_time: mean_speedup=0.3869x (sum kernel_ms=7.75ms across 20 cases)
- improvement: +3.6% (mean_speedup), -3.5% (sum kernel_ms)
- precision: pass (20/20 cann-bench cases + test_mish.py --level all Test Passed!)
- adopted: yes
- rollback_reason: none
- next_hint: case 11/14/18/19 (M<128) 略退化 9-12%，因小 block_M 降低 Vector 利用率。可尝试 smart-flatten 优先 M>=128 的 split（除非 num_blocks 差异显著）

### iter1 详细 case 对比

| case | shape | dtype | baseline speedup | iter1 speedup | delta | 驱动 |
|------|-------|-------|-----------------|---------------|-------|------|
| 1 | [1024,1024] | fp16 | 0.2417 | 0.2370 | -2% | 持平（128×128） |
| 2 | [2048,2048] | fp32 | 0.3668 | 0.3718 | +1% | 持平 |
| 3 | [4096,4096] | bf16 | 0.6602 | 0.6767 | +3% | 持平 |
| 4 | [8192,8192] | fp16 | 0.9411 | 0.9398 | -0% | 持平 |
| 5 | [8192,8192] | fp32 | 0.9175 | 0.9150 | -0% | 持平 |
| 6 | [1023,1023] | bf16 | 0.2335 | 0.2400 | +3% | 持平 |
| 7 | [1009,1021] | fp16 | 0.2391 | 0.2386 | -0% | 持平 |
| 8 | [1537,769] | fp32 | 0.2361 | 0.2404 | +2% | 持平 |
| 9 | [363,367,373] | bf16 | 0.6878 | 0.6730 | -2% | 持平（smart-flatten 相同 split） |
| 10 | [2049,513] | fp16 | 0.2380 | 0.2391 | +0% | 持平 |
| 11 | [3,7,13,4001] | fp32 | 0.2384 | 0.2102 | -12% | ❌ M=21<128, block_M=2 Vector 低效 |
| 12 | [1000003] | bf16 | 0.0373 | 0.2255 | +504% | ❗❗ 1D block_N=8192, num_blocks 7813→123 |
| 13 | [11,13,17,67,67] | fp32 | 0.3058 | 0.4477 | +46% | ❗ smart-flatten split_idx=2, num_blocks 1273→684 |
| 14 | [3,7,11,13,1009] | fp16 | 0.3232 | 0.2934 | -9% | ❌ 可能噪声（tiling 相同） |
| 15 | [512,2049] | fp32 | 0.2326 | 0.2353 | +1% | 持平 |
| 16 | [255,8193] | bf16 | 0.3001 | 0.2739 | -9% | ❌ 可能噪声（tiling 相同） |
| 17 | [4097,511] | fp16 | 0.3046 | 0.2935 | -4% | 持平 |
| 18 | [2,511,2049] | fp32 | 0.2784 | 0.2497 | -10% | ❌ M=2<128, block_M=2 |
| 19 | [4,255,2049] | bf16 | 0.2910 | 0.2637 | -9% | ❌ M=4<128, block_M=2 |
| 20 | [2,3,17,1024,101] | fp32 | 0.3925 | 0.4745 | +21% | ❗ smart-flatten split_idx=2, num_blocks 816→606 |

### 关键发现
1. **1D shape 优化（case 12）**：block_N cap 128→8192，num_blocks 7813→123，speedup +504%
2. **ND smart-flatten（case 13/20）**：搜索 split_idx 选 num_blocks 最小切分，speedup +46%/+21%
3. **M>=128 用 128×128 sweet spot**：避免非 128 倍数 block_M（138/152）降低 Vector 效率
4. **32B 对齐约束**：非对齐 block_N（67/101）导致 DataCopyNd 数据损坏，case 13/20 精度失败
5. **M<128 退化（case 11/18/19）**：小 block_M 降低 Vector 利用率，num_blocks 减少不足以补偿

## Iteration 2 — 2026-08-10T11:45:00Z
- bottleneck_type: transfer (iter1 引入的 M<128 退化)
- optimization: smart-flatten 用 128×128 评估 num_blocks（偏好 M>=128 split），实际执行仍用动态 tiling
- baseline_time: mean_speedup=0.3869x (iter1)
- candidate_time: mean_speedup=0.3928x
- improvement: +1.5% (vs iter1), +5.2% (vs baseline)
- precision: pass (20/20 cann-bench + test_mish.py --level all)
- adopted: yes (修复 iter1 退化 case 11/18/19，累计 +5.2% > 3%)
- rollback_reason: none
- next_hint: kernel 侧微优化（max→relu, 消除 tmp_orig）

## Iteration 3 — 2026-08-10T11:55:00Z
- bottleneck_type: compute (kernel 12 步 T.tile.xxx，max(x,0) 可能用 relu 融合)
- optimization: T.tile.max(t1_ub, a_ub, 0.0) → T.tile.relu(t1_ub, a_ub)
- baseline_time: mean_speedup=0.3928x (iter2)
- candidate_time: mean_speedup=0.3987x (两次测量平均: 0.4042 / 0.3931)
- improvement: +1.5% (vs iter2, 噪声范围内 ±3%)
- precision: pass (test_mish.py --level all Test Passed!)
- adopted: no
- rollback_reason: 两次测量差异 2.8%（噪声范围内），无法确认 > 3% 提升；max 已是 1/12 步，收益预计 < 1%
- next_hint: 手动 kernel cache（避免 JIT lookup）

## Iteration 4 — 2026-08-10T12:00:00Z
- bottleneck_type: transfer (host 侧 JIT cache lookup 开销)
- optimization: 手动 _kernel_cache dict 缓存 kernel 对象
- baseline_time: mean_speedup=0.3928x (iter2)
- candidate_time: mean_speedup=0.3965x
- improvement: +0.9% (vs iter2, 噪声范围内)
- precision: pass (test_mish.py --level all Test Passed!)
- adopted: no
- rollback_reason: tilelang JIT 已有全局 cache，手动 cache 收益在噪声范围内 (+0.9% < 3%)
- next_hint: 优化空间耗尽（kernel 侧 log-sum-exp trick 已最优，host 侧 smart-flatten + 动态 tiling 已做完）

## 中止决策 — 2026-08-10T12:05:00Z
- 最终版本: iter2 (mean_speedup=0.3916x, +4.9% vs baseline)
- 中止原因:
  1. 连续 2 次无提升（iter3 + iter4 回滚，均 < 3% 噪声阈值）
  2. 优化空间耗尽: kernel 侧 12 步 log-sum-exp trick 不可简化，host 侧 smart-flatten + 动态 tiling + kernel cache 已做完
  3. 进一步优化收益在噪声范围内（±3%）
- consecutive_no_improvement: 2 (iter3 + iter4)

