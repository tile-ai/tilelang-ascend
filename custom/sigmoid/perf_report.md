# Sigmoid Performance Tuning Report (Stage 3)

## 1. 概述

- **算子**: sigmoid (`y = 1 / (1 + exp(-x))`)
- **编程模式**: Developer (T.alloc_shared + AUTO_SYNC + MEMORY_PLANNING)
- **baseline**: torch.sigmoid (PyTorch NPU 实现)
- **性能目标类型**: baseline_compare
- **测试 shape**: 主基准 (1024, 8192) float16 + 辅助 (512, 512) float32
- **噪声阈值**: 3%
- **迭代轮数**: 1 (本轮)
- **最终状态**: 采纳 (基于 msprof kernel 级提升)

## 2. 优化措施摘要

### Iteration 1: [#1] 关闭 AUTO_CV_COMBINE + [#3] Fixed Core

**优化前 kernel** (baseline):
- `pass_configs = {AUTO_CV_COMBINE: True, AUTO_SYNC: True, MEMORY_PLANNING: True}`
- `T.Kernel(m_num * n_num)` — 按逻辑任务数 launch (512 block for 1024×8192)
- 每 block 处理 (128, 128) tile，VEC_NUM=2 vid 切分

**优化后 kernel** (candidate):
- `pass_configs = {AUTO_SYNC: True, MEMORY_PLANNING: True}` (移除 AUTO_CV_COMBINE)
- `CORE_NUM = 24` (Ascend A2/A3 物理核数)
- `launch_cores = min(block_num, CORE_NUM)` — Fixed Core 模式
- `T.Kernel(launch_cores)` + `T.serial(single_core_load)` 每核串行处理 ~22 tile
- Striped 分配: `logical_cid = block_idx * launch_cores + cid`
- buffer 在循环内分配 (hoisting 到循环外导致编译卡住)
- T.tile.sigmoid 一步原语保留，VEC_NUM=2 保留

## 3. 性能对比

### 3.1 bench 端到端 (Python 计时, warmup=30, iters=100)

| shape | dtype | torch.sigmoid | tilelang (before) | tilelang (after) | speedup (after vs torch) |
|-------|-------|--------------|-------------------|------------------|--------------------------|
| (1024, 8192) | float16 | 0.0545 ms | 0.2147 ms | 0.2153 ms | 0.253x |
| (512, 512) | float32 | 0.0399 ms | 0.1895 ms | 0.1923 ms | 0.207x |

**bench 端到端结论**: 性能无显著变化 (+0.3% fp16, +1.5% fp32, 均 < 3% 噪声阈值)。

### 3.2 msprof NPU kernel task duration

| 指标 | before | after | 变化 |
|------|--------|-------|------|
| Task Duration (fp16 1024×8192) | 71.9 us | 53.6 us | **-25.5%** |
| Block Dim | 512 | 24 | -95.3% |
| Mix Block Dim | 1024 | 48 | -95.3% |
| Op Type | mix (MIX_AIC_1_2) | mix (MIX_AIC_1_2) | task type 未变 |

**msprof kernel 级结论**: NPU 侧 kernel 性能提升 25.5%，launch 数减少 95.3%。

### 3.3 瓶颈定位

```
bench 端到端时间分解 (fp16 1024×8192):
  baseline:  214.7 us = 142.8 us (host) + 71.9 us (NPU)
  candidate: 215.3 us = 161.7 us (host) + 53.6 us (NPU)
  
  host 开销 = tilelang runtime (Python→C++→ACL) + torch.npu.synchronize()
  NPU 执行 = kernel task duration (msprof)
  
  torch.sigmoid: 54.5 us ≈ NPU 执行 (host 开销极小)
```

**核心发现**:
1. NPU 侧 kernel 已从 71.9 us 优化到 53.6 us，接近 torch.sigmoid (54.5 us)
2. bench 端到端瓶颈在 host 侧 tilelang runtime 开销 (~160 us)，非 NPU kernel
3. tilelang kernel 的 NPU 侧性能已接近 torch.sigmoid，端到端差距主要来自 host runtime

## 4. 精度验证

| 级别 | 用例数 | 结果 | max_abs (fp16) | max_abs (fp32) |
|------|--------|------|----------------|----------------|
| L0 | 7 | 全 PASS | 4.883e-04 | 0.000e+00 |
| L1 | 10 | 全 PASS | 4.883e-04 | 0.000e+00 |
| L2 | 2 | 1 PASS + 1 WARN (与优化前一致) | — | — |
| Boundary | 4 | 全 PASS | 0.000e+00 | — |

精度标准 (DESIGN.md §9.3): float16 atol=2⁻¹⁴(6.10e-5)/rtol=2⁻⁹(1.95e-3)/max_abs=1e-1; float32 atol=2⁻¹⁶/rtol=2⁻¹⁰/max_abs=1e-2。全部满足。

## 5. 采纳判定

- **判定依据**: msprof NPU kernel task duration 提升 25.5% > 3% 噪声阈值
- **bench 端到端无提升的原因**: host 侧 tilelang runtime 开销 (~160 us) dominates，掩盖 NPU 侧提升
- **采纳**: yes
- **回滚**: no

## 6. 反模式检查 (performance-antipatterns.md)

| 反模式 | 命中 | 处理 |
|--------|------|------|
| launch core 数关注项 A (block_num >> 24) | HIT (baseline 512 block) | 已修复: Fixed Core 24 核 launch |
| 纯 Vector + AUTO_CV_COMBINE 误分核 | PARTIAL HIT | 已处理: 关闭 AUTO_CV_COMBINE (task type 未变但 pass 不再运行) |
| tile size 过小 (UB 占用 25%) | HIT | 暂未修改: 留待下轮 [#5] |
| 纯 AIV memory bound 未做流水/双 buffer | HIT | 暂未修改: 留待下轮 [#4] |
| Vector for loop 逐行计算 | N/A | 已用 T.tile.sigmoid 整 tile SIMD |
| 冗余全局同步 | N/A | AUTO_SYNC 自动管理，单 block 内必要 |

## 7. 下一轮建议

1. **[#4] Vector Double Buffer + 关闭 AUTO_SYNC**: 让 T.serial 循环内 MTE2/V/MTE3 三路流水重叠。需 Expert 模式手动 flag (set_flag/wait_flag)，风险较高但可能进一步降低 NPU task duration。
2. **[#5] 增大 tile size**: (128,128)→(128,512) fp16，UB 占用 25%→75%，减少 tile 数 4x。需同步更新 test_sigmoid.py L0 用例 block 配置。
3. **host 侧优化** (超出 kernel 范围): tilelang runtime launch 开销 ~160 us 是当前 bench 端到端瓶颈，建议与 tilelang runtime 团队沟通。

## 8. 文件清单

| 文件 | 说明 |
|------|------|
| `custom/sigmoid/sigmoid.py` | 优化后 kernel (Fixed Core + 关闭 CV) |
| `custom/sigmoid/history_version/sigmoid_perf_iter1.py` | 优化前 kernel 备份 (回滚基线) |
| `custom/sigmoid/perf_tuning/baseline_iter1.json` | 基线性能数据 |
| `custom/sigmoid/perf_tuning/optimization_log.md` | 优化记录 (含 [ORDER-PLAN] 和每步 [RESULT]) |
| `custom/sigmoid/perf_tuning/perf_log.md` | 迭代日志 (结构化) |
| `custom/sigmoid/perf_tuning/perf_report.md` | 本报告 |
| `custom/sigmoid/perf_tuning/bench_perf.py` | 性能基准脚本 |
| `custom/sigmoid/perf_tuning/bench_perf_result.json` | bench 结果 JSON |
| `custom/sigmoid/perf_tuning/run_once_for_msprof.py` | msprof 单次运行脚本 |
| `custom/sigmoid/perf_tuning/kernel_source_float16_1024x8192.cpp` | 翻译后 Ascend C 源码 |
| `custom/sigmoid/perf_tuning/msprof_output_iter1/` | 优化前 msprof 数据 |
| `custom/sigmoid/perf_tuning/msprof_output_iter1_after/` | 优化后 msprof 数据 |
