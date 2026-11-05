# group_norm perf optimization log

## Baseline (msprof op, Ascend 910B3, task duration μs)

| case | shape | dtype | now_us | sep_us | 瓶颈信号 |
|------|-------|-------|--------|--------|---------|
| cb1  | [8,32,64,64] | fp16 | 15.5 | 13.7 | — |
| cb13 | [1023,257] | fp16 | **134.4** | 95.7 | 2D 路径每 block 1 行（1023 微 block，vec 2μs/块） |
| cb15 | [4,31,251,251] | bf16 | **332.5** | 325.3 | mte2 83% busy（275μs），block_S=160 → 320B 突发 + 126KB 跨距 |
| cb20 | [1,513,63,63] | fp32 | **271.2** | 273.7 | mte2 92% busy（250μs），block_S=16 → **64B 突发 + 15.9KB 跨距** |

诊断：重灾区为 MTE2 跨步小突发读（[cpg, block_S] 瓦片行距 S）与 2D 路径的行粒度。

## Part A 优化点清单

[#1] **2D kernel 多行 Tile（block_M 行/block）**（参考: optimization-guide.md §2.13 L921+，Layer 1-3）：**适用** — cb13 走 2D 路径（x.ndim==2），现每 block 处理 1 行 [cpg]，1023 个微 block；行维在 GM 连续（Layer 1 ✓ stride=cpg），合并为 [ROWS, cpg] tile 后 MTE2 突发从 514B/block 升至 ~17KB/block，block 数从 1023 降至 ~16。参照 rms_norm simple kernel 的已验证结构（ROWS = block_M // VEC_NUM，VEC_NUM=2 双核分行）。

[#2] **UB 预算 per_unit 修正**（参考: optimization-guide.md §2.11 L539+，"只计算同时活跃的 buffer，双缓冲 input/output 算双份；MEMORY_PLANNING 自动复用已死亡 buffer"）：**适用** — 现公式 per_unit = cpg×(6×cal + 6×dtype) 是不区分相位的保守界（12 单元）。实际峰值（Pass 2，DB 全活）：4×dtype + 3×cal = fp32 28B / fp16 20B（vs 48/36）→ block_S ×1.7~1.8，突发相应加长。MEMORY_PLANNING=True 已开启（kernel 现状）✓ 前提成立。

[#3] **cpg 维分块（[R_cpg, block_S] 瓦片）**（参考: §2.13 Layer 2）：**暂缓** — 可再放大 block_S ~4x，但需重构 Pass1 累加结构（sum_row 切片累加）与 scale/offset 分块广播，风险/工作量高；待 #1/#2 验证后评估。

[#4] Fixed Core 模式（§2.9）：**不适用** — PR body 已实测 -7.4% 回归。
[#5] Pass 1 Double Buffer（§2.2）：**不适用** — 无 MTE3 写回，v→mte2 flag 不可用（AIV 死锁），PR body 已论证。

## [ORDER-PLAN]

1. [#2] per_unit 修正 — 前置依赖: 无 — 理由: host 侧一行公式改动，零 kernel 风险，先锁定突发增益
2. [#1] 2D kernel 多行 Tile — 前置依赖: 无 — 理由: 独立代码路径（x.ndim==2 分支），与 #2 无耦合
3. [#3] 视 #1/#2 结果决定 — 前置依赖: [#2]

## 实施记录

### [#2] per_unit 修正

[IMPL-#2] 已阅读 optimization-guide.md L539-L600（§2.11），关键约束: 只计同时活跃 buffer；DB input/output 双份；MEMORY_PLANNING 复用死亡 buffer。
[SELF-CHECK] 本次 Edit 只涉及 _find_block_S 的 per_unit 公式。

- 旧: per_block = c × (6×cal + 6×dtype)
- 新: per_block = c × (4×dtype + 3×cal)  # Pass2 峰值: db(2)+out_db(2) input + cal/scale/offset 3 cal

[RESULT-#2] 精度: full suite pass（L0 3/3+cb7, L1 14/14, L2 3/3, Boundary 4/4）| 性能: cb1 15.5→15.2μs, cb20 271.2→**31.4μs (-88%!)**, cb15 332.5→**91.6μs (-72%)**, cb13 134.4→134.2（2D 路径不走该公式）

### [#1] 2D kernel 多行 Tile

[IMPL-#1] 已阅读 optimization-guide.md L921-L1010（§2.13 Layer 1-3），关键约束: R 维与 BS 维内存相邻（2D 输入行维 stride=cpg ✓）；block_S 不可硬编码须 UB 反推；get_kernel_source 含 KERNEL_TYPE_MIX 时包 if vid==0。
[KERNEL-REUSE-AUDIT] 依据: §2.13 | expanded_domain: 2D 输入路径的 M 维分块（原每 block 1 行）| 维度: layout 连续 ✓ / UB 预算反推 ✓ / MIX guard（kernel 现有 if vid==0 保留并按 ROWS 分行）| 结论: repair-kernel-first（重写 kernel_2d 主体）
[DB-ANALYSIS] Q1: 循环内有 MTE3？— 无（单遍，AUTO_SYNC=True 路径）Q2: 跨迭代累加器？— 有（sum_sq per row，行内归约后即写回，无跨迭代）Q3: 同步方式 — AUTO_SYNC=True 自动（现路径保持）
[SELF-CHECK] 本次 Edit 只涉及 group_norm_kernel_2d + wrapper 的 2D 分支 tiling。

[RESULT-#1] 精度: cb13 精确匹配 + full suite pass | 性能: cb13 134.2→**12.9μs (-90%!)**

## 最终对比（msprof task duration）

| case | baseline | final | 提升 |
|------|----------|-------|------|
| cb1  | 15.5 | 15.2 | -2% |
| cb13 | 134.4 | 12.9 | **-90%** |
| cb15 | 332.5 | 91.6 | **-72%** |
| cb20 | 271.2 | 31.4 | **-88%** |

全量 20 用例 msprof + 双后端精度全过（见 final_report.md）。
