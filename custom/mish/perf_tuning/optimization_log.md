# Mish 性能优化日志

## 基线信息

- **Kernel**: `custom/mish/mish.py`（12 步 T.tile.xxx，float32 中间计算，Developer 模式）
- **Test**: `custom/mish/test_mish.py`（L0:9 + L1:10 + L2:3 + Boundary:4，全量精度通过）
- **Baseline 采集**: `[本地 bench, 2026-08-10T11:15:00Z]` mean_speedup=0.3733x
- **算子类型**: 纯 Vector（element-wise，get_kernel_source 只有 IS_ASCEND_AIV 预期）
- **编程模式**: Developer（alloc_shared + AUTO_SYNC + MEMORY_PLANNING）

### Baseline 20-case 数据（baseline_iter1.json）

| case | shape | dtype | kernel_ms | baseline_ms | speedup | num_blocks(估) |
|------|-------|-------|-----------|-------------|---------|----------------|
| 1 | [1024,1024] | fp16 | 0.2129 | 0.0515 | 0.2417 | 64 |
| 2 | [2048,2048] | fp32 | 0.2370 | 0.0869 | 0.3668 | 256 |
| 3 | [4096,4096] | bf16 | 0.3720 | 0.2456 | 0.6602 | 1024 |
| 4 | [8192,8192] | fp16 | 0.9690 | 0.9119 | 0.9411 | 4096 |
| 5 | [8192,8192] | fp32 | 0.9415 | 0.8638 | 0.9175 | 4096 |
| 6 | [1023,1023] | bf16 | 0.2200 | 0.0514 | 0.2335 | 64 |
| 7 | [1009,1021] | fp16 | 0.2149 | 0.0514 | 0.2391 | 64 |
| 8 | [1537,769] | fp32 | 0.2182 | 0.0515 | 0.2361 | 84 |
| 9 | [363,367,373] | bf16 | 1.0030 | 0.6899 | 0.6878 | 3123 |
| 10 | [2049,513] | fp16 | 0.2238 | 0.0532 | 0.2380 | 85 |
| 11 | [3,7,13,4001] | fp32 | 0.2205 | 0.0526 | 0.2384 | 96 |
| 12 | [1000003] | bf16 | 1.4787 | 0.0551 | 0.0373 | 7813 |
| 13 | [11,13,17,67,67] | fp32 | 0.5387 | 0.1648 | 0.3058 | 1277 |
| 14 | [3,7,11,13,1009] | fp16 | 0.2382 | 0.0770 | 0.3232 | 192 |
| 15 | [512,2049] | fp32 | 0.2180 | 0.0507 | 0.2326 | 68 |
| 16 | [255,8193] | bf16 | 0.2212 | 0.0664 | 0.3001 | 130 |
| 17 | [4097,511] | fp16 | 0.2204 | 0.0671 | 0.3046 | 132 |
| 18 | [2,511,2049] | fp32 | 0.2241 | 0.0624 | 0.2784 | 136 |
| 19 | [4,255,2049] | bf16 | 0.2220 | 0.0646 | 0.2910 | 136 |
| 20 | [2,3,17,1024,101] | fp32 | 0.4056 | 0.1592 | 0.3925 | 816 |

### 瓶颈诊断

1. **host 侧固定 tiling 是主要瓶颈**（Step 5.5 决策树）：
   - case 12 (1D [1000003]): 固定 block_N=128 → num_blocks=7813，kernel 1.48ms（应为 ~0.05ms）
   - case 9 (3D [363,367,373]): 固定 reshape(-1,373) → num_blocks=3123
   - case 13 (5D [11,13,17,67,67]): 固定 reshape(-1,67) → num_blocks=1277
   - case 20 (5D [2,3,17,1024,101]): 固定 reshape(-1,101) → num_blocks=816
2. **小 shape case host 开销 dominate**：1M-2M 元素 case 延迟恒定 ~0.22ms，远超理论计算时间
3. **kernel 本身已用 log-sum-exp trick 消除条件分支**（避免了 T.if_then_else 反模式，speedup 0.0159→0.7168 的关键）

---

## Part A 优化点清单

### 静态检查（performance-antipatterns.md）

- [AP-1] Vector for loop 逐行计算：**不适用** — kernel 用 T.tile.xxx 整 tile 向量化，无 for loop
- [AP-2] T.if_then_else 逐元素条件分支：**不适用** — 已用 log-sum-exp trick 消除（当前实现 speedup 0.3733，非反模式的 0.0159）
- [AP-3] 冗余全局同步：**不适用** — Developer 模式 AUTO_SYNC=True，无手动 barrier
- [AP-4] 基础指令拼接未融合：**部分适用** — `max(x, 0)` 可用 `T.tile.relu` 替代，但收益小（见 #5）
- [AP-5] tile size 过小：**适用** — 当前固定 block_M=128, block_N=128，对 1D 和小 N case 浪费严重（见 #1）
- [AP-6] AIC/AIV 未开启 CV overlap：**不适用** — 纯 Vector 算子，无 AIC
- [AP-7] 纯 AIV memory bound 未做流水/双 buffer：**部分适用** — 但 element-wise 算子 host tiling 是更大瓶颈（见 #1）
- [AP-8] launch core 数过多：**适用** — case 12 num_blocks=7813 >> 24 核（见 #1）

### 动态优化点（optimization-guide.md + Step 5.5 决策树）

[#1] host 侧 smart-flatten + 动态 block_M/block_N（参考: cann-bench-elementwise-optimization.md §动态 tiling 自适应 + skill Step 5.5）：**适用** — **P0 优先**
- 原因：element-wise 算子主要瓶颈在 host 侧 tiling 选择导致 num_blocks 过多。case 12 num_blocks 7813→123（-98%），case 9/13/20 也有显著改善
- 零拷贝前提：contiguous tensor reshape 是 view 操作，cann-bench 默认 contiguous ✓
- 实施内容：
  - 1D shape (M≤2): block_N cap 提到 8192（rows_per_vec=1 时单 buffer 8192×22B=180KB < 192KB UB）
  - ND shape: smart-flatten 搜索所有 split_idx，选 num_blocks 最小的 (M,N) 切分
  - 动态 block_M/block_N: 根据 dtype 和 UB 预算自适应（block_M × block_N ≤ UB_budget / bytes_per_elem × VEC_NUM）

[#2] max(x, 0) → T.tile.relu 融合（参考: performance-antipatterns.md §基础指令拼接未融合）：**适用但低优先级**
- 原因：当前 `T.tile.max(t1_ub, a_ub, 0.0)` 可替换为 `T.tile.relu(t1_ub, a_ub)`，减少 1 条指令
- 风险：需验证 relu 对 NaN/Inf 的行为与 max(x,0) 一致（relu(NaN)=NaN, max(NaN,0)=NaN ✓；relu(-inf)=0, max(-inf,0)=0 ✓）
- 收益：小（1/12 步指令减少），但无风险

[#3] 消除 tmp_orig buffer（fp32 路径）（参考: cann-bench-elementwise-optimization.md §UB 预算校验）：**适用**
- 原因：kernel 无条件分配 tmp_orig，fp32 (need_cast=False) 时不用但仍占 UB 预算
- 实施：用条件分配或依赖 MEMORY_PLANNING 复用
- 收益：释放 UB 预算，允许更大 block_M/block_N

[#4] 调整 VEC_NUM（参考: DESIGN.md §5.3）：**不适用**
- 原因：threads 参数限制仅 1 或 2，当前 VEC_NUM=2 已是最优。改 VEC_NUM=1 会减半并行度

[#5] AUTO_CV_COMBINE 实验（参考: DESIGN.md §3.4）：**不适用**
- 原因：纯 Vector 算子，开启 AUTO_CV_COMBINE 会 spawn 空闲 AIC core（DESIGN.md 已论证）

---

## Part B [ORDER-PLAN] 实施顺序

```
[ORDER-PLAN] 实施顺序：
1. [#1] host 侧 smart-flatten + 动态 block_M/block_N — 前置依赖: [无] — 理由: P0，element-wise 算子最大收益点，预期 mean_speedup 0.37→0.6+
2. [#3] 消除 tmp_orig buffer (fp32 路径) — 前置依赖: [#1] — 理由: 释放 UB 预算，为 #1 的大 block_N 提供 headroom
3. [#2] max(x,0) → T.tile.relu 融合 — 前置依赖: [#1] — 理由: 低风险微优化，需精度复验
```

---

## 迭代记录

### [RESULT-#1a] iter1: host smart-flatten + 动态 tiling (初版)
- 优化点: smart-flatten 用 _select_tiling 评估 num_blocks + 动态 block_M/block_N
- 精度: pass (修复 N<128 和 0维 后)
- 性能: mean_speedup 0.3869x (baseline 0.3733x)
- 对比: +3.6% (采纳)
- 问题: case 11/18/19 退化 9-12%（M<128, block_M=138/152 非 128 倍数）

### [RESULT-#1b] iter2: smart-flatten 用 128×128 评估 (修复退化)
- 优化点: smart-flatten 评估 num_blocks 时用固定 128×128（偏好 M>=128 split），实际执行仍用动态 tiling
- 精度: pass
- 性能: mean_speedup 0.3928x (iter1 0.3869x)
- 对比: +1.5% vs iter1, +5.2% vs baseline (采纳，修复退化)
- 修复: case 11/18/19 从退化恢复到持平

### [RESULT-#2] iter3: max→relu 融合 (回滚)
- 优化点: T.tile.max(t1_ub, a_ub, 0.0) → T.tile.relu(t1_ub, a_ub)
- 精度: pass
- 性能: 两次测量 0.4042x / 0.3931x（差异 2.8%，噪声范围内）
- 对比: +1.5% vs iter2 (平均，< 3% 噪声阈值)
- 决策: 回滚（无法确认 > 3% 提升）

### [RESULT-#3] iter4: 手动 kernel cache (回滚)
- 优化点: _kernel_cache dict 缓存 kernel 对象，避免 JIT cache lookup
- 精度: pass
- 性能: mean_speedup 0.3965x
- 对比: +0.9% vs iter2 (< 3% 噪声阈值)
- 决策: 回滚（tilelang JIT 已有全局 cache，手动 cache 无额外收益）

### 最终决策
- 最终版本: iter2 (mean_speedup=0.3916x final rerun, +4.9% vs baseline)
- 中止原因: 连续 2 次无提升 (iter3 + iter4) + 优化空间耗尽
- 未实施的优化点:
  - [#3] 消除 tmp_orig: fp32 已用 20B/elem，无 UB headroom 可释放
  - [#4] VEC_NUM 调整: threads 限制 1 或 2，VEC_NUM=2 已最优
  - [#5] AUTO_CV_COMBINE: 纯 Vector 算子，开启会 spawn 空闲 AIC

