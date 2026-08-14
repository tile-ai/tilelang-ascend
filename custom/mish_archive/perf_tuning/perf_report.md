# Mish Performance Tuning Report (Stage 3)

## 1. 概述

- **算子**: mish (`y = x * tanh(softplus(x)) = x * tanh(ln(1 + e^x))`)
- **编程模式**: Developer (T.alloc_shared + AUTO_SYNC + MEMORY_PLANNING) + 中间 fp32 + cast 桥接
- **baseline**: torch.nn.functional.mish (PyTorch NPU 实现)
- **性能目标类型**: baseline_compare
- **目标**: 平均加速比 ≥ 0.6x
- **测试 shape**: DESIGN.md §12 perf_target shape set (9 configs: 3 dtype × S/M/L aligned + 非对齐 + 质数)
- **噪声阈值**: 3%
- **迭代轮数**: 1 (本轮)
- **最终状态**: 部分采纳（[#1] 保留，[#3] 回滚）

## 2. 优化措施摘要

### Iteration 1: [#1] 关闭 AUTO_CV_COMBINE（保留）+ [#3] Fixed Core（回滚）

**优化前 kernel** (baseline):
- `pass_configs = {AUTO_CV_COMBINE: True, AUTO_SYNC: True, MEMORY_PLANNING: True}`
- `T.Kernel(m_num * n_num)` — 按逻辑任务数 launch
- 每 block 处理 (128, 128) tile，VEC_NUM=2 vid 切分
- 12 步 fp32 中间计算 + cast 桥接（非 fp32 输入）

**优化后 kernel** (candidate, 最终采纳):
- `pass_configs = {AUTO_SYNC: True, MEMORY_PLANNING: True}` (移除 AUTO_CV_COMBINE)
- `T.Kernel(m_num * n_num)` — 保持按任务数 launch（[#3] Fixed Core 已回滚）
- 12 步 fp32 中间计算 + cast 桥接逻辑不变
- VEC_NUM=2 保留

**[#3] Fixed Core（已回滚）**:
- `launch_cores = min(block_num, CORE_NUM=24)` + `T.serial(single_core_load)` 每核串行处理
- 大 shape (8192,8192) 严重退化 +25-36%（T.serial 循环 171 tile/core 开销 > launch 减少）
- 回滚到 T.Kernel(m_num*n_num) launch

## 3. 性能对比

### 3.1 bench 端到端 (Python 计时, warmup=30, iters=100, torch.npu.synchronize)

| shape | dtype | torch.mish | tilelang (baseline) | tilelang (after [#1]) | speedup (after vs torch) |
|-------|-------|-----------|---------------------|----------------------|--------------------------|
| (1024, 1024) | float16 | 0.0505 ms | 0.1867 ms | 0.1873 ms | 0.270x |
| (1024, 1024) | float32 | 0.0511 ms | 0.1881 ms | 0.1841 ms | 0.277x |
| (1024, 1024) | bfloat16 | 0.0511 ms | 0.1879 ms | 0.1868 ms | 0.273x |
| (2048, 2048) | float16 | 0.0906 ms | 0.2177 ms | 0.2174 ms | 0.417x |
| (2048, 2048) | float32 | 0.0873 ms | 0.2177 ms | 0.2209 ms | 0.395x |
| (8192, 8192) | float16 | 0.9129 ms | 0.9500 ms | 0.9595 ms | 0.951x |
| (8192, 8192) | float32 | 0.8616 ms | 0.9381 ms | 0.9325 ms | 0.924x |
| (1023, 1023) | bfloat16 | 0.0525 ms | 0.1996 ms | 0.1917 ms | 0.274x |
| (1537, 769)  | float32 | 0.0536 ms | 0.1995 ms | 0.1967 ms | 0.272x |

**bench 端到端结论**: mean speedup 0.449x → 0.449x（无变化，[#1] 在噪声范围内）。0.6x 目标未达。

### 3.2 [#3] Fixed Core 回滚数据（已回滚，仅记录）

| shape | dtype | baseline | [#1]+[#3] | 变化 | 判定 |
|-------|-------|----------|-----------|------|------|
| (8192, 8192) | float16 | 0.9500 ms | 1.2882 ms | +35.6% | 严重退化 |
| (8192, 8192) | float32 | 0.9381 ms | 1.1764 ms | +25.5% | 严重退化 |
| (2048, 2048) | float16 | 0.2177 ms | 0.2384 ms | +9.5% | 退化 |
| (2048, 2048) | float32 | 0.2177 ms | 0.2312 ms | +6.2% | 退化 |
| **mean speedup** | | **0.449x** | **0.394x** | **-12.2%** | **回滚** |

### 3.3 瓶颈定位

```
bench 端到端时间分解 (mean across shapes):
  小 shape (1024,1024): tilelang ~187us = host ~137us + NPU ~50us
                        torch ~50us ≈ NPU 执行 (host 开销极小)
  大 shape (8192,8192): tilelang ~950us = host ~50us + NPU ~900us
                        torch ~910us ≈ NPU 执行

  host 开销 = tilelang runtime (Python→C++→ACL) + torch.npu.synchronize()
  NPU 执行 = kernel task duration

关键发现:
  1. 小 shape 瓶颈在 host 侧 tilelang runtime 开销 (~137us)，非 NPU kernel
  2. 大 shape NPU kernel 已接近 torch (0.92-0.96x)
  3. mish 12 步计算让 NPU kernel 时间占比比 sigmoid 大，bench 端到端 0.449x 优于 sigmoid 0.25x
```

## 4. 精度验证

| 级别 | 用例数 | 结果 | max_abs (fp16) | max_abs (fp32) | max_abs (bf16) |
|------|--------|------|----------------|----------------|----------------|
| L0 | 8 | 全 PASS | 4.883e-04 | 5.960e-07 | 0.000e+00 |
| L1 | 15 | 全 PASS | 4.883e-04 | 9.841e-07 | 1.526e-05 |
| L2 | 2 | 1 PASS + 1 WARN (与优化前一致) | — | — | — |
| Boundary | 4 | 全 PASS | 1.907e-06 | 0.000e+00 | — |

精度标准 (DESIGN.md §9.3): float16 atol=2⁻¹⁴/rtol=2⁻¹⁰/max_abs=1e2; float32 atol=2⁻¹⁷/rtol=2⁻¹³/max_abs=1e0; bfloat16 atol=2⁻¹¹/rtol=2⁻⁷/max_abs=1e3。全部满足。优化后 max_abs 与 baseline 完全一致（计算逻辑未变）。

## 5. 采纳判定

- **[#1] 关闭 AUTO_CV_COMBINE**: 采纳（基于反模式修复）
  - bench 端到端无变化（< 3% 噪声阈值）
  - 但 performance-antipatterns.md 明确指出纯 Vector + AUTO_CV_COMBINE 是反模式（AIC 空跑浪费 launch 与初始化开销）
  - sigmoid 先例保留（custom/sigmoid/sigmoid.py iter1 同样保留 [#1]）
  - 精度通过，无退化
- **[#3] Fixed Core**: 回滚
  - 大 shape (8192,8192) 严重退化 +25-36%
  - mish 12 步计算 per tile 比 sigmoid 1 步重得多，T.serial 循环开销超过 launch 数减少收益
  - mean speedup 下降 12.2% > 3% 噪声阈值
- **0.6x 目标**: 未达（0.449x）
  - 大 shape (8192,8192) 已接近 torch（0.92-0.96x）
  - 小 shape 瓶颈是 host runtime 开销（~137us），非 NPU kernel 问题
  - 优化空间有限：UB 压力限制 tile size，Expert 双缓冲不可行

## 6. 反模式检查 (performance-antipatterns.md)

| 反模式 | 命中 | 处理 |
|--------|------|------|
| launch core 数关注项 A (block_num >> 24) | HIT (8192,8192 → 4096 block) | 已尝试 [#3] Fixed Core，回滚（大 shape 退化） |
| 纯 Vector + AUTO_CV_COMBINE 误分核 | HIT | 已修复: [#1] 关闭 AUTO_CV_COMBINE |
| tile size 过小 (UB 占用高) | HIT (168-176KB / 192KB) | 无法修改: fp32 中间计算 6 buffer 已占满 UB，放大 tile 会超限 |
| 纯 AIV memory bound 未做流水/双 buffer | HIT | 无法修改: Expert 双缓冲 stages=2 × 6 buffer × fp32 超 UB 上限 |
| Vector for loop 逐行计算 | N/A | 已用 T.tile.xxx 整 tile SIMD |
| 冗余全局同步 | N/A | AUTO_SYNC 自动管理 |
| 基础指令拼接未融合 | PARTIAL | mish 必须 12 步分解（T.tile.tanh 不存在），无法融合 |

## 7. 下一轮建议

1. **Expert 双缓冲评估（高风险）**: mish 12 步 + 6 buffer，stages=2 下 UB 需 12 × rows_per_vec × block_N × 4B。block_M 最多 32（rows_per_vec=16），tile 太小可能退化。需仔细评估。
2. **host 侧优化（超出 kernel 范围）**: tilelang runtime launch 开销 ~137us 是小 shape 端到端瓶颈，建议与 tilelang runtime 团队沟通。
3. **接受当前性能**: 大 shape 已接近 torch（0.92-0.96x），mean speedup 0.449x 优于 sigmoid 0.25x。0.6x 目标未达但优化空间有限，建议中止。

## 8. 文件清单

| 文件 | 说明 |
|------|------|
| `custom/mish/mish.py` | 优化后 kernel ([#1] 关闭 AUTO_CV_COMBINE，保留 T.Kernel(m_num*n_num) launch) |
| `custom/mish/history_version/mish_perf_iter0_baseline.py` | baseline kernel 备份 (iter0 完整原版) |
| `custom/mish/history_version/mish_perf_iter1_before.py` | iter1 修改前备份 (回滚基线) |
| `custom/mish/perf_tuning/mish_impl_iter1_before.py` | iter1 修改前备份 (perf_tuning 目录副本) |
| `custom/mish/perf_tuning/baseline_iter1.json` | baseline 性能数据 (9 shape × 3 dtype) |
| `custom/mish/perf_tuning/candidate_iter1.json` | [#1]+[#3] 候选性能数据 (已回滚) |
| `custom/mish/perf_tuning/candidate_iter1_v2.json` | [#1] only 候选性能数据 (最终采纳) |
| `custom/mish/perf_tuning/bench_perf.py` | 性能基准脚本 |
| `custom/mish/perf_tuning/baseline_iter1.log` | baseline bench 日志 |
| `custom/mish/perf_tuning/candidate_iter1.log` | [#1]+[#3] bench 日志 |
| `custom/mish/perf_tuning/optimization_log.md` | 优化记录 (含 [ORDER-PLAN] 和每步 [RESULT]) |
| `custom/mish/perf_tuning/perf_log.md` | 迭代日志 (结构化) |
| `custom/mish/perf_tuning/perf_report.md` | 本报告 |
| `custom/mish/test_output_all_iter1.log` | iter1 全量精度测试日志 |
| `custom/mish/Mish/` | cann-bench 包目录 (build.sh + setup.py + cann_bench/) — **已补全** (Stage 3 finalize: __init__.py + mish.py adapter 生成, _common.py/_mish_kernel.py 确认与优化后 mish.py 一致) |
| `custom/mish/Mish/Mish.zip` | cann-bench 压缩包 — **已生成并验证** (7 files, extract+import+run test 全通过, fp16/fp32/bf16/1D/4D 五种 shape 精度通过, fp32 max_diff vs torch=7.15e-7) |
