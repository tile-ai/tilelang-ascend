# ForeachNorm Best+List 优化日志

## 基线
- 版本: best (custom/foreach_norm/foreach_norm.py, 1740 行, 27 kernel)
- 备份: custom/foreach_norm/history_version/foreach_norm_best_baseline.py
- 已有优化: 多核 launch_cores + VEC_NUM=2 + 条件性 T.Pipelined + L1 list kernel + _direct_norm

## 瓶颈分析
- 5 个多 tensor case (L2/Lp + scl<20) 仍有 torch.stack 开销
- L1 已有 list kernel 消除 stack；L2/Lp 未覆盖

## [ORDER-PLAN]
1. [#1] L2 list kernel (l2_norm_kernel_list2/3/4) — 前置: 无 — abs→mul(x²)
2. [#2] Lp list kernel (lp_norm_kernel_list2/3/4) — 前置: 无 — abs+ln+mul+exp
3. [#3] dispatch 泛化 (_use_l1_list_kernel → _use_list_kernel) — 前置: #1, #2

## [KERNEL-REUSE-PRECHECK]
candidate: L2/Lp list kernel (复制 L1 list kernel 结构)
expanded_domain: L2 (scalar=2.0) 和 Lp (scalar=3/4/5/1.5 等) 的 batch=2/3/4 + scl<20 路径
compatibility_dimensions:
  - buffer 布局: 与 L1 list kernel 一致 (x_ub/x_cal/pow_ub or abs_ub/tile_sum_ub/acc_ub)
  - out_idx: list2=[2], list3=[3], list4=[4] (N 输入 + 1 Partial 输出)
  - Partial shape: (batch, launch_cores, VEC_NUM) — 与 L1 list kernel 一致
  - finalize: host 侧 _finalize_batched 不变 (sum/sqrt/pow/cast)
unknown_or_false: none
plan: direct-reuse (复制 L1 list kernel，仅改 compute op)

## [RESULT-#1] L2 list kernel (l2_norm_kernel_list2/3/4)
- 精度: pass (L0/L1 全部 L2 多 tensor case PASS)
- 性能: case 11 (batch=3) 747.91→316.63us (-58%), case 15 (batch=2) 521.27→303.09us (-42%), case 18 (batch=2) 520.46→308.72us (-41%)
- 对比: 大幅改善 (消除 batch-1 次 TileLang launch 开销)

## [RESULT-#2] Lp list kernel (lp_norm_kernel_list2/3/4)
- 精度: pass (L1 Lp 多 tensor case PASS)
- 性能: case 19 (scalar=3.0, batch=2) 573.13→338.94us (-41%)
- 对比: 大幅改善

## [RESULT-#3] dispatch 泛化 (_use_l1_list_kernel → _use_list_kernel)
- 精度: pass (全量复验，唯一失败 l1_c12_fp32_l5_5d 为 best baseline 已有 pre-existing 问题，batch=1 不走 list kernel)
- 性能: avg_speedup 0.2270 → 0.2486 (+9.5%)
- 对比: 改善
