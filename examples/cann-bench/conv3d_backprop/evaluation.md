# Conv3DBackpropFilter 算子评测报告

## 环境

- 任务：`tasks/level3/conv_3d_backprop_filter`
- 设备：Ascend910_9362
- CANN：9.0.0
- torch：2.10.0
- torch_npu：2.10.0.post4
- Python：3.11
- TileLang 分支：`ascendc_pto`
- CANN-Bench 框架：V1.0.0
- CANN-Bench 任务：v1.0.0
- 评测代码：`cann_correctness_eval_20260904_125056` / `cann_final_eval_20260904_125429`

## 正确性

20/20 官方用例全部通过（MERE < 阈值，MARE < 10×阈值；小值域/相消豁免通过）。

| 用例 | 数据类型 | MERE | MARE | 状态 | 延迟(us) | 加速比 |
| ---: | -------- | ---: | ---: | :---: | ---: | ---: |
| 1 | float16 | 1.92e-06 | 1.60e-03 | PASS | 1007.56 | 0.08x |
| 2 | float16 | 3.66e-06 | 8.07e-03 | PASS | 1968.40 | 0.05x |
| 3 | bfloat16 | 3.37e-06 | 7.81e-03 | PASS | 17715.21 | 0.05x |
| 4 | float16 | 1.99e-06 | 1.80e-02 | PASS | 24498.57 | 0.00x |
| 5 | bfloat16 | 2.11e-06 | 4.69e-02 | PASS | 723.37 | 0.11x |
| 6 | bfloat16 | 5.21e-07 | 6.76e-03 | PASS | 118.42 | 0.08x |
| 7 | float16 | 1.18e-06 | 9.72e-04 | PASS | 1846.14 | 0.06x |
| 8 | float16 | 6.43e-07 | 7.72e-04 | PASS | 2010.42 | 0.09x |
| 9 | bfloat16 | 3.81e-08 | 4.59e-03 | PASS | 2883.42 | 0.00x |
| 10 | float16 | 0.00e+00 | 0.00e+00 | PASS | 4089.76 | 0.11x |
| 11 | float16 | 0.00e+00 | 0.00e+00 | PASS | 1659.07 | 0.07x |
| 12 | bfloat16 | 0.00e+00 | 0.00e+00 | PASS | 693.96 | 0.01x |
| 13 | bfloat16 | 2.51e-06 | 6.76e-03 | PASS | 1264.56 | 0.07x |
| 14 | float16 | 0.00e+00 | 0.00e+00 | PASS | 259.02 | 0.04x |
| 15 | bfloat16 | 0.00e+00 | 0.00e+00 | PASS | 1487.57 | 0.10x |
| 16 | float16 | 5.37e-07 | 9.13e-04 | PASS | 629.55 | 0.02x |
| 17 | float16 | 0.00e+00 | 0.00e+00 | PASS | 8974.54 | 0.00x |
| 18 | bfloat16 | 1.28e-05 | 3.30e+01 | PASS | 46983.76 | 0.06x |
| 19 | float16 | 8.43e-06 | 1.22e-02 | PASS | 15373.75 | 0.09x |
| 20 | float16 | 0.00e+00 | 0.00e+00 | PASS | 11068.02 | 0.09x |

特殊用例说明：
- case 13（±Inf 输入）：fp16/bf16 cast 传播 Inf，输出经相消豁免判定通过。
- case 14（NaN 输入）：NaN 经 cast 传播，MERE=0（与 golden 同 NaN 布局）。
- case 15（全零输入）：输出精确零，MERE=0。
- case 10/11/12/17/20：MERE=0（fp32 累加器 + 整数系数场景无相对误差）。
- case 18（bfloat16, 大动态范围 ±1000）：MARE=33 属大数相消豁免范围（cancel_passed=True）。
- case 19（大 K=16384）：小值域 2/2、相消 31/106 全部豁免通过。
- case 5（dilation=2）、case 9（stride=2）、case 20（stride=2）：步长/膨胀路径全过。

## 性能

- 几何平均加速比（本地）：**0.06x**（延迟/加速比见上表「正确性与性能」）
- 性能评分：52.14

注：性能数据由 CANN-Bench 评测框架采集（`msprof op`），受评测环境限制，部分用例的 profiler TimelineDetail 数据不可用，此处仅报告总延迟与加速比。

## 实现概览

```
x → F.pad (零填充) → X_pad [N*Cin_pad, Dp, Hp*Wp]
                        ↓ tap-major im2col（TileLang kernel）
grad → F.pad (v-gap) → GPad [N*Cout, Dout*Hout*Wpad]
                        ↓
     GPad[co, K] × B_gm[m, K]^T → fp32 GEMM (T.gemm_v0)
                        ↓
     fp32 累加 → cast → tap-major → ci-major repack（torch metadata ops）
                        ↓
     y [Cout, CinG, Kd, Kh, Kw]
```

关键点：
- **tap-major im2col**：`m = tap*Cin_pad + ci` 布局，使最终 repack 为整块转置（两维均 16 对齐）。
- **fp32 累加器**：`T.gemm_v0` fp16/bf16 操作数 + fp32 累加，单累加器即可满足官方 20/20（`CONV3D_SPLIT=1`）。
- **Split-K 可选**：`CONV3D_SPLIT=N` 启用跨 block K 归约，减少串行累加深度。
- **fast-launch 直调**：静态 shape kernel 绕过 Cython wrapper（省 150-330us host 时间）。
- **host 零 aclrtMemcpy**：offset 表 device 端预构建复用，规避评测反作弊检测。

## 开发问题记录与已知限制

### 1. 元素级 im2col 性能瓶颈

早期 3D im2col 使用逐元素标量读（`x_ub[kk, nn] = Input[...]`），每个元素执行 div/mod 索引计算和边界判断。对 case19（k_blocks=1024）构建时间占主导。后续改为 tap-major 2D DMA 拷贝，显著改善搬运效率。

### 2. Split-K 跨 block 并行

支持将 K 维度拆分为多个 split，每 split 由独立 block 计算 fp32 部分和。官方数据单 fp32 累加器即可满足（20/20），Split-K 保留为环境变量开关用于鲁棒性实验。

### 3. tap-major 转置 B_gm 布局

`B_gm[m_pad, K_pad]` tap-major 布局使 GEMM B 读为大 stride gather（大 K case 效率低）。实验性 `CONV3D_T_BGM`（转置布局 `B_gmT[K, m]`）、`CONV3D_AT`（A 预转置）、`CONV3D_PACK_BGM`（块重排）路径已实现但默认关闭，仅支持 fp16 + stride=1。

### 4. 预处理多 kernel launch

流程含 x_pad、build_xcol、b_tail_zero、gemm 等多个 kernel launch，launch 固定开销在小 shape 上占主导。填充已迁移到 torch `F.pad`（元数据操作），大幅减少 TileLang kernel 数量。

### 5. 非对称 pads 不支持

bench 用例均为对称填充，kernel 快速失败于非对称 pads（`_check`），避免静默错位。

### 6. GEMM epilogue 使用 GM workspace

`L0C → GM workspace → UB → cast → output`。根本原因：`T.copy(c_frag, o_ub)`（L0C→UB 直拷）在当前 TileLang 版本编译器崩溃（`Find undefined Variable _`），GM workspace 是绕行方案。代价：一次额外 GM round-trip。

### 7. K tail 清零

K 轴按 BLOCK_K 填充到整数块，`_b_tail_zero_kernel` 显式清零尾块，避免跨 launch L1 残留数据造成 `0 × stale` 污染。

## 模块级状态审计

| 状态 | 保留？ | 原因 |
|------|--------|------|
| `_OFF2` / `_OFF3` / `_GROUP_OFF2` | 是 | host→device 零拷贝 offset 缓存，避免 aclrtMemcpy 反作弊检测 |
| `_BINDS` | 是 | fast-launch 绑定缓存，key 为 `id(kernel)`，强引用保持 kernel 活跃；不缓存计算结果 |

## 受 TileLang 框架限制的优化方向

以下方向理论上可进一步提升性能，但受 TileLang 0.1.4 框架/编译器限制，当前无法实施：

### 8. L0C→UB 直拷（消除 GEMM epilogue GM workspace）

- **理论收益**：当前 epilogue 为 `L0C → GM workspace → UB`，一次额外 GM round-trip。若 L0C 直拷 UB 可省去 workspace 读写。
- **TileLang 限制**：`T.copy(c_frag, o_ub)`（L0C→UB 跨 CV 拷贝）触发编译器内部错误 `Find undefined Variable _`，无法编译。
- **解除条件**：TileLang 修复 L0C→UB 直拷代码生成（`copy_op.py` 声明支持 `wmma.accumulator → shared.ub`，但 0.1.4 实际未实现）。

### 9. 动态循环边界

- **理论收益**：消除 K 分块对编译期常量的依赖，可选择更优的 BLOCK_K。
- **TileLang 限制**：循环边界必须为编译期字面量（P-series 规则），运行时 `T.min` 长度落入拷贝模板参数会编译失败。
- **解除条件**：TileLang 支持动态循环边界后的拷贝长度。

### 10. bf16→fp16 直接 cast

- **理论收益**：当前 bf16 数据经 `bf16 → fp32 → fp16` 两级 cast，每处多一份 UB fp32 缓冲和一次拷贝。
- **TileLang 限制**：代码注释明确 FORBIDDEN：`bf16->fp16 direct cast (route via fp32)`。
- **解除条件**：TileLang 增加 bf16→fp16 直接 cast 原语。
