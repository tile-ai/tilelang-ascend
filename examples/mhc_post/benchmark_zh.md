**中文** | [English](benchmark.md)

# MHC Post 性能基准与优化路径

## 1. 算子

```
output = x * post_layer_mix + comb_res_mix^T @ residual
```

- 输入：x [n, h] bf16, residual [n, hc, h] bf16, post_layer_mix [n, hc, 1] fp32, comb_res_mix [n, hc, hc] fp32
- 输出：[n, hc, h] bf16
- 约束：1 <= hc <= 8（JIT 参数，已验证范围）

## 2. 硬件与软件

| 项目 | 值 |
|------|-----|
| NPU | Ascend 910B |
| CANN | 9.0.0 |
| 工具 | do_bench (Python), msprof op (硬件级) |
| 数据类型 | bf16 输入, fp32 累加 |

## 3. 优化路径（V0 -> V10 泛化 HC + 尾部掩码）

| 版本 | 改动 | Kernel-only (n=4096, h=2560) | vs CANN | 关键突破 |
|------|------|------------------------------|---------|---------|
| V0 | Cube 双 kernel | 4.63 ms | 0.49x | 基线 |
| V1 | AIV 单 V 核 | 13.37 ms | 0.17x | 纯 Vector |
| V2 | 双 V 核 + h_blk=256 | 2.56 ms | 0.89x | 利用两个 V 核 |
| V3 | comb 常驻 UB + 跳过 pad | 1.70 ms | 1.32x | 消除冗余 GM 读取 |
| V4 | AXPY + h_blk=2048 | 0.70 ms | 3.17x | 结构性重构 |
| V5 | T.Pipelined(stage=2) | 0.67 ms | 3.37x | 流水线重叠 |
| V6 | cast 融合 + kernel 缓存 | 0.65 ms | 3.46x | 消除 host 开销 |
| V7 | adaptive h_blk + out 复用 + T.unroll | 0.42 ms | 5.39x | 消除 padding 浪费 |
| V8 混合 | 2D UB 快速路径 + 1D UB 回退 | 0.38 ms | 5.98x | 2D 路径 merged copy，3584 通过 1D 回退保留 |
| V9 统一 | 单 kernel：2D res + 1D out | 0.38 ms | 5.88x | 一个 kernel 覆盖所有 h_blk 含 3584 |
| V10 泛化 | 删除 host pad + 泛化 hc + merged bf16 store | 0.38 ms | 5.98x | kernel 内 tail（pad_value + TAIL_MASK），hc 1-8，2D bf16 merged MTE3 |

### 关键决策

**V0 -> V1：为什么不用 Cube？**
- hc=4 pad 到 16 浪费 93.75% Cube MAC 资源
- CV 同步 bug 阻止单 kernel 融合
- 纯 Vector 避免了这两个问题
- 但 V1 比 Cube 慢 3 倍：只用了一个 V 核，且用 `broadcast + mul + reduce_sum` 模拟 `[4,4]@[4,h]` 矩阵乘，远不如专用 Cube 硬件高效。这促使了 V2（双 V 核）和 V4（AXPY 替代 broadcast+reduce）。

**V1 -> V2：双 V 核**
- `bid = cid * 2 + vid`——两个 Vector 单元处理不同 token
- 2.9 倍加速（之前一个 V 核空闲）

**V3：循环不变量外提**
- comb 系数在 h 循环外加载一次（4 个独立 1D UB buffer）
- 按 reviewer 建议评估了 2D UB 布局；2D slice 操作（T.copy、T.tile.cast、T.tile.axpy 的 2D dst/src）正确，但 T.copy(GM→2D UB) 后做 2D 标量索引（comb[i,j]）读错地址——reviewer 的完整 2D 方案因使用 2D 标量读而失败（max_diff=25.59）。折中方案（2D slice 搬 res/out + 1D 存 comb）可行且 h=2560 时快 ~12%，但需要单独 h_blk sweep。最终用 4 个独立 1D buffer 绕过，可用更大 h_blk（2560/3584）。
  > V8 混合后来通过将 comb 对齐到 [4, 8]（32B/行）解决了 2D 标量读问题，启用 2D UB 快速路径。详见下方 V8 混合。
- 也评估了单个扁平 1D comb buffer `[16]` + 一次 `T.copy(comb[bid,0,0], comb_fp32)`；但读到错误数据（3D 标量偏移拷贝不跨行），所以保留 4 个独立 1D buffer。

**V4：AXPY（最大突破）**
- 用 `T.tile.mul(dst, src, scalar)` + `T.tile.axpy(dst, src, scalar)` 替代 `broadcast + mul + reduce_sum`
- 消除了 7 个大型 2D FP32 UB buffer（56KB -> 16KB）
- 更小的 UB 占用使 h_blk 从 512 提到 2048，循环次数从 5 减到 2

**V6：cast 融合**
- 删除 host 侧 FP32->BF16 cast（独立 kernel launch）
- kernel 直接接收 FP32，无 BF16 量化
- Golden reference 同步修改（删除 ref 的 bfloat16() 量化）
- max_diff 改善：0.125 -> 0.0625

**V7：adaptive h_blk + out 复用 + T.unroll**
- h_blk 从 [3584, 3072, 2560, 2048, 1024, 512] 中选择 h 的最大因数
  - h=2560 -> h_blk=2560（无 padding，1 tile）
  - h=7168 -> h_blk=3584（无 padding，2 tiles）
- 旧版 padded 路径多算 1.6 倍元素（h=2560：原来 4096，现在 2560）
- out0~3 合并为单个可复用 out_fp32（UB 占用 -24KB）
- T.unroll(4) 替代手动 4 倍代码复制
- F.pad 替代手写 _pad_3d/_pad_2d_1d 函数

**V8 混合：2D UB 快速路径 + 1D UB 回退**
- 两个 kernel 变体按 shape JIT 编译，由 `_select_path(h)` 分发：
  - **2D UB 路径**（h_blk ≤ 3072）：merged res/out copy（1 次 T.copy 替代 4 次），对齐 comb `comb_fp32[4, 8]`（32B/行，使 2D 标量读 `comb_fp32[res_idx, out_idx]` 正确）。无需 host pad（要求 h % h_blk == 0）。更少的 MTE2/MTE3 launch 开销。
  - **1D UB 路径**（h_blk = 3584 或非整除 h）：独立逐行 res/out buffer（更小 UB 占用）+ 对齐 comb [4,8]（同 2D 路径）。host 侧 F.pad 处理非整除 h。
- 分发规则：从合并候选 [3584, 3072, 2560, 2048, 1024, 512] 中找整除 h 的最大 h_blk；在 2D 列表（≤ 3072）中 -> 2D 路径，3584 -> 1D 路径，无整除 -> 1D 路径 + pad。
- 关键突破：`comb_fp32 = T.alloc_ub((HC, (HC+7)//8*8), accum_dtype)` = [4, 8] 将每行对齐到 32 字节，匹配 AlignInnerDim 的 padding。使 2D 标量读正确工作（之前 [4, 4] 未对齐 -> 地址错误 -> max_diff=25.59）。
- 结果（5 次平均，do_bench warmup=20 rep=100）：
  - h=4096×2560（2D 路径，h_blk=2560）：**5.98x**（V7：5.34x，+12%）
  - h=4096×7168（1D 路径，h_blk=3584）：**7.40x**（V7：7.27x，+2%）
- h_blk=3584 的 2D UB 测试失败（UB 溢出，kernel 挂起）。混合方案通过 1D UB 回退规避。

**V9 统一：单 kernel（2D res + 1D out）**
- 将 V8 的两个路径合并为一个 kernel：2D res（1 次 T.copy 搬全部 4 行）+ 1D out（流式写回，out buffer 跨 out_idx 复用）。
- 2D res 保持 MTE 效率（1 次 copy vs 4 次）；1D out 保持低 UB 占用（~126KB），使 h_blk=3584 可行——V8 的 2D 路径溢出是因为 2D `out_fp32`/`out_bf16`（`[4, h_blk]`，~189KB）。
- 删除 `_select_path` 分发；host 只选最大整除 h_blk，非整除时 pad。代码：2 个 kernel + 分发 -> 1 个 kernel（约 -97 行）。
- 精度 15/15。性能：h=7168 7.42x（持平/略快），h=2560 5.88x。
- 工程 trade-off vs V8：h=2560 慢 ~1.7%（5.88x vs 5.98x），但单 kernel 结构删除了一个 kernel 变体和分发逻辑（约 -97 行）；h=7168 略快。为代码简化接受。
- T.Persistent（reviewer 建议）已评估但未采纳：single-shape 快 4%~17%，但 full-shape 测试在小 n / pad / 单 tile 及多 shape 连续运行场景触发 vector-core 异常（UUB 地址未对齐，err 0x10）。留为已知方向待后端修复。
- AUTO_SYNC=False 手写流水也已评估但未采纳。作为纯 AIV kernel，设 `AUTO_SYNC=False` 可获得 +13%~21%（7168：7.44x->8.98x），因为 `AUTO_SYNC=True` 插入冗余同步压制 MTE2/V/MTE3 重叠。但实现该收益需要手写 `set_flag`/`wait_flag`，遇到三个障碍：
  1. `set_flag` event id 必须是编译期常量。传 `T.serial` 循环变量 `i % 2` 使 codegen 丢掉 `% 2`，生成运行时 event id（`i` = 0,1,2,...）无法配对 -> 死锁。修复：用 `stages` 变量 + `T.serial(0, h_num - 1)` + 显式 epilogue 使循环展开为常量 0/1。
  2. `T.tile.mul` 的标量读（`post_fp32[out_idx]`）触发 `PipeBarrier<PIPE_ALL>`，全局屏障与 flag 环形成死锁。修复：用 `T.tile.fill(dst, 0.0)` + `T.tile.axpy(dst, src, scalar)` 替代 `mul`（axpy 的标量读不触发 barrier）。
  3. 完整流水需要双缓冲输出（`out_bf16[2, 4, h_blk]`），但 h_blk=3584 时 UB 溢出（~210KB > 192KB）；单缓冲（`[4, h_blk]`）在 tile i 的 MTE3 写和 tile i+1 的 V cast 之间竞争。障碍 1 和 2 可解；障碍 3 是 h_blk=3584 的硬 UB 约束。留为已知方向。

**V10 泛化：删除 host pad + 泛化 hc + merged bf16 store**
- 回应 reviewer 反馈的三项改动：
  1. **删除 host F.pad**：tail tile 并入同一个 `T.Pipelined` 循环（`total_tiles = ceildiv(h, h_blk)`）。每个 tile copy 带 `pad_value=0.0`（full tile 不触发填零，tail tile 填零 gap）。开启 `TL_ASCEND_TAIL_MASK` pass 使 vector op 只计算 valid region。无需 host padding。
  2. **泛化 hc（1-8）**：`hc` 为 JIT 参数，不再硬编码 4。`comb_row_stride = (hc + 7) // 8 * 8` 按 hc 向上对齐到 32B，保证 2D 标量读正确。`_max_h_blk(hc)` 按 UB budget 限制 h_blk 上限。
  3. **2D bf16 merged store**：`out_bf16 = T.alloc_ub((hc, h_blk), dtype)`，逐行 cast 到 `out_bf16[out_idx, :]`，最后 1 次 merged `T.copy(out_bf16, output)` 替代 hc 次独立 MTE3 store。
- 分离的 tail block（V9 的 `if tail > 0` 在 pipeline 循环外）与 `T.Pipelined` 双缓冲和 tail mask pass 的 buffer 跟踪冲突。将 tail 并入主循环解决。
- 精度 22/22（15 个 hc=4 case + 7 个 hc=1/2/3/8 case 覆盖 tail path）。
- 性能：h=2560 与 V9 持平（5.98x）；h=7168 从 0.82ms 提升到 0.75ms（7.42x -> 8.10x），可能来自更简洁的循环结构。5 次平均。

## 4. 最终性能（V10 泛化，do_bench，warmup=20，rep=100，5 次平均）

| n | h | hc | h_blk | Kernel-only | E2E | PyTorch (CANN) | Kernel 加速比 | E2E 加速比 |
|---|---|---|-------|-------------|-----|----------------|----------------|-------------|
| 512 | 2560 | 4 | 2560 | 0.34 ms | 0.34 ms | 0.25 ms | 0.74x | 0.74x |
| 4096 | 2560 | 4 | 2560 | 0.38 ms | 0.38 ms | 2.25 ms | 5.98x | 5.98x |
| 4096 | 7168 | 4 | 3584 | 0.75 ms | 0.75 ms | 6.07 ms | 8.10x | 8.10x |

> 单一统一 kernel。h=2560/7168 均整除 h_blk（无 tail），E2E ≈ kernel-only。V10 删除 host pad 并开启 TL_ASCEND_TAIL_MASK；性能与 V9 持平（h=2560），h=7168 提升（0.82->0.75 ms，更简洁的循环结构）。
> n=512 比 CANN 慢是因为数据量小（24MB）未充分并行双 V 核。

## 5. h_blk Sweep（V7 1D kernel，kernel-only）

| h_blk | n=512, h=2560 | n=4096, h=2560 | n=4096, h=7168 |
|-------|---------------|-----------------|------------------|
| 512 | 0.84x | 2.26x | 2.40x |
| 1024 | 0.81x | 3.08x | 3.93x |
| 2048 | 0.84x | 3.49x | 5.16x |
| 2560 | 0.75x | 5.44x | 6.05x |
| 3072 | 0.79x | 5.13x | 5.59x |
| 3584 | 0.87x | 4.77x | **7.26x** |

> Sweep 用 V7 的 1D UB kernel 测量。V9 统一用单 kernel（2D res merged copy + 1D out）覆盖所有 h_blk 含 3584。
> h_blk=2560 时统一版 5.88x vs V7 的 5.44x（+8%）。

## 6. 流水线 Ablation（n=4096, h=7168, h_blk=3584, kernel-only）

| 调度方式 | 延迟 | vs CANN | vs serial |
|----------|---------|---------|-----------|
| T.serial | 0.876 ms | 6.93x | 基准 |
| T.Pipelined(stage=2) | 0.840 ms | 7.23x | +4.1% |

stage=2 比 serial 快 4.1%。stage=3 不可用（h_blk=3584 × 3 stages 超出 UB 容量）。

## 7. 性能分析（n=4096, h=7168, h_blk=3584）

### V10 泛化有效带宽（do_bench）

| shape | 数据量 | Kernel 延迟 | 有效带宽 | HBM 峰值占比 |
|-------|--------|------------|---------|--------------|
| n=4096, h=2560 | 189 MB | 0.38 ms | 497 GB/s | 41% |
| n=4096, h=7168 | 529 MB | 0.75 ms | 705 GB/s | 59% |

> 统一 2D-res merged copy 将 h=2560 带宽从 451 GB/s（V7）提升到 497 GB/s，通过减少 MTE2/MTE3 launch 开销。V10 的统一流水线（tail 并入主循环）进一步将 h=7168 从 647 GB/s（V9）提升到 705 GB/s。

### V6 msprof 参考（h_blk=2048, kernel 1.18 ms）

> V6 硬件级分解通过 msprof 测量（h_blk=2048, kernel 1.18 ms）。
> V10 使用 h_blk=3584（kernel 0.75 ms），数据搬运布局不同（2D-res merged copy + 2D bf16 merged store）；未采集 V10 的 msprof profile，因此 V6 结果作为历史参考保留，而非 V10 的定论瓶颈表征。h=7168 带宽（705 GB/s）来自消除 padded 数据搬运（同 V7）；merged 2D-res copy 将 h=2560 带宽提升到 497 GB/s。

| 指标 | 值 |
|--------|-------|
| Task Duration | 1,194 us |
| Block 数 | 6,144（每个 AI Core 3 硬件核：1 Cube + 2 Vector；逻辑块 = n/2 = 2048）|
| Vector compute | 21,701 us (1818%) |
| MTE2 (GM->UB load) | 8,348 us (699%) |
| MTE3 (UB->GM store) | 9,980 us (836%) |
| Scalar | 6,921 us (580%) |
| 并行度 | 37.6x |
| 有效带宽 | 449 GB/s |

> 百分比 = 每核累计时间 / Task Duration。值 >100% 表示多核并发。

### 瓶颈：Vector 计算受限（V6 历史）

V6 msprof（h_blk=2048）显示 Vector compute（1818%）> MTE 总计（1535%），即早期 AXPY 实现主要受 Vector 计算限制，T.Pipelined 重叠计算和内存（37.6x 并行度）。V10 保留相同算术结构，同时改进数据搬运布局（2D-res merged copy + 2D bf16 merged store）。V10 本身未重新 profile，因此 Vector 计算瓶颈作为历史证据保留，而非 V10 的定论表征。

## 8. 精度

| 指标 | 值 |
|--------|-------|
| 测试用例 | 22/22 通过 |
| 容差 | rtol=1e-2, atol=0.2 |
| 最大差异 | 0.0625 |
| 差异来源 | BF16 输出舍入 + 累加顺序 |

> 15 个 hc=4 case（各种 h，含 tail path）+ 7 个 hc=1/2/3/8 case（覆盖 hc<4、hc>4、非 8 对齐 comb_row_stride、tail path）。

## 9. 停止条件

| 条件 | 状态 |
|-----------|--------|
| Kernel > CANN | 是（0.74x - 8.10x；小 shape 慢，大 shape 6-8x）|
| E2E > CANN（大 shape）| 是（5.98x - 8.10x）|
| 计算受限证据 | V6 msprof（历史）：Vector 1818% > MTE 1535%；V10 未重新 profile |
| 流水线已优化 | 是（stage=2，比 serial +4.1%；stage=3 受 UB 限制）|
| h_blk 已优化 | 是（adaptive：h 的最大因数，单 kernel）|
| 2D res merged copy | 是（比 V7 +10% at h=2560）|
| 2D bf16 merged store | 是（1 次 merged MTE3 替代 hc 次）|
| kernel 内 tail | 是（pad_value + TL_ASCEND_TAIL_MASK，无 host pad）|
| 泛化 hc | 是（hc 1-8，JIT 参数）|
| 有效带宽高 | 是（497-705 GB/s，HBM 峰值 41%-59%）|

优化停止：单一统一 kernel（2D res merged load + 2D bf16 merged store）覆盖所有 h_blk 含 3584 和所有 hc 1-8。非整除 h 在 kernel 内通过 pad_value + TL_ASCEND_TAIL_MASK 处理（无 host padding）。进一步提升需要减少 Vector 计算量（AXPY 循环展开已是 T.unroll(hc)）。
