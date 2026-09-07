# Conv2D 算子评测报告

## 环境

- 任务：`tasks/level3/conv_2d`
- 设备：Ascend910_9362
- CANN：9.0.0
- torch：2.10.0
- torch_npu：2.10.0.post4
- Python：3.11
- TileLang 分支：`ascendc_pto`
- CANN-Bench 框架：V1.0.0
- CANN-Bench 任务：v1.0.0
- 评测代码：`cann_correctness_eval_20260903_200346`

## 正确性

20/20 官方用例全部通过（MERE < 阈值，MARE < 10×阈值）。

| 用例 | 数据类型 | MERE | MARE | 状态 | 延迟(us) | 加速比 |
| ---: | -------- | ---: | ---: | :---: | ---: | ---: |
| 1 | float16 | 2.36e-07 | 9.56e-04 | PASS | 68.60 | 0.15x |
| 2 | float32 | 1.04e-06 | 5.56e-05 | PASS | 553.59 | 0.15x |
| 3 | bfloat16 | 9.73e-07 | 7.81e-03 | PASS | 2350.09 | 0.18x |
| 4 | float16 | 7.51e-07 | 9.77e-04 | PASS | 2452.77 | 0.11x |
| 5 | float32 | 2.10e-06 | 3.53e-03 | PASS | 8088.22 | 0.13x |
| 6 | bfloat16 | 9.28e-07 | 6.59e-02 | PASS | 520.67 | 0.18x |
| 7 | float16 | 8.10e-08 | 9.48e-04 | PASS | 81.88 | 0.12x |
| 8 | float32 | 1.91e-07 | 1.25e-06 | PASS | 368.81 | 0.17x |
| 9 | bfloat16 | 2.04e-07 | 4.15e-03 | PASS | 85.76 | 0.12x |
| 10 | float16 | 0.00e+00 | 0.00e+00 | PASS | 239.03 | 0.17x |
| 11 | float16 | 4.12e-07 | 1.17e-03 | PASS | 72.22 | 0.14x |
| 12 | float32 | 4.34e-07 | 6.06e-05 | PASS | 48.72 | 0.21x |
| 13 | float32 | 0.00e+00 | 0.00e+00 | PASS | 87.74 | 0.18x |
| 14 | float16 | 0.00e+00 | 0.00e+00 | PASS | 25.74 | 0.39x |
| 15 | bfloat16 | 0.00e+00 | 0.00e+00 | PASS | 95.80 | 0.10x |
| 16 | float16 | 4.76e-07 | 5.74e-03 | PASS | 87.78 | 0.11x |
| 17 | float32 | 2.88e-07 | 1.68e-06 | PASS | 598.89 | 0.12x |
| 18 | bfloat16 | 7.34e-07 | 7.81e-03 | PASS | 1233.64 | 0.23x |
| 19 | float16 | 3.21e-07 | 6.86e-03 | PASS | 89.86 | 0.11x |
| 20 | float32 | 1.97e-07 | 1.38e-06 | PASS | 900.12 | 0.14x |

特殊用例说明：
- case 13（Inf）：hi/lo clamp 使 `lo=0`，输出有限值，MERE=0。
- case 14（NaN）：hi cast 传播 NaN，lo clamp 为 0，MERE=0。
- case 15（全零）：输出精确零，MERE=0。
- case 6（1×1，Cin=2048）。
- cases 4/9/17（stride=2）。
- cases 2/5/8/12/13/17/20（fp32 hi/lo 路径）。

## 性能

- 几何平均加速比（本地）：**0.16x**（延迟/加速比见上表「正确性与性能」）
- 性能评分：5.18

注：性能数据由 CANN-Bench 评测框架采集（`msprof op`），受评测环境限制，部分用例的 profiler TimelineDetail 数据不可用，此处仅报告总延迟与加速比。

## 开发问题记录与已知限制

### 1. im2col 数据搬运策略

早期方案生成完整 global im2col tensor 到 GM，再运行 GEMM——额外的 GM 写入+读取在小/中 case 占主导。后续逐元素 im2col 又引入 div/mod、边界判断和 scalar GM 访问。最终方案：沿 W 方向识别连续 run，使用 2D/连续 `T.copy` 直接加载到当前 L1 tile——无全局 im2col workspace。

### 2. DMA 效率优化

`w_out_k = 48` 时 `block_w` 退化为 16，MTE2 以 ~203 GB/s 运行（64-wide 时为 ~948 GB/s）。修复：48→64 lane widening（+33% 垃圾 lane，由 materialize dst-clamp 丢弃）。case11 GEMM：72.4us → 41.8us。

Stride-2 早期路径在密集行上做 GEMM 再下采样。修复：compact stride-2 rows（只计算有效输出行）；列保持密集 lane，由 materializer gather。case4 -19.3%，case17 -13.6%。

### 3. FP32 支持

Ascend Cube 原生操作数为 fp16/bf16；硬件 MMAD 支持 float×float（HF32，E8M11，20 位），但当前工具链（CANN 9.0.0）未开放该特性，且 HF32 的 ~11 位尾数精度不满足 fp32 卷积的 2^-13 精度要求。故采用 hi/lo 拆分模拟：

```
w_hi·x_hi + w_hi·x_lo + w_lo·x_hi
```

式中 `hi = fp16(x)`，`lo = fp16(clamp(x) - fp32(hi))`。`w_lo·x_lo` 省略（二阶项，对范围内值可忽略）。Inf/NaN：hi cast 传播，lo clamp 为 0。阈值：MERE < 2^-13（fp32）。

早期 input hi/lo 由两个 kernel 生成（X→X_hi, X→X_lo）。修复：合并为单次 launch，X 只读一次，UB 中派生 hi/lo，写入两个独立 GM tensor。

### 4. GM store 稳定性问题

`w_out < 16` 或尾部不足完整 vector 时，partial-width 2D store 可能错误解释行步长，损坏第二行起的输出。修复：使用完整对齐宽度写入，或逐行安全写入，在 materialize 阶段裁剪。

部分 preprocessing kernel 中一次 launch 执行多个同 shape 2D store 时出现数据损坏。修复：谨慎合并输出；仅对已验证的不同目标 tensor 使用双 store。

### 5. K tail 未清零导致 NaN

L1 跨 launch 保留旧数据。最后一个 K tile 中未覆盖的区域如果不清零，可能发生 `0 × stale Inf = NaN`。修复：最后一块显式清零，然后覆盖有效数据。

### 6. GM→L1 lowering 可能绕行 UB

`T.copy(GM, L1)` 是源级语义。编译器可能产生中间 UB 暂存，而非物理直连 GM→L1。如需确认物理路径，需检查生成代码。本 PR 描述为"TileLang 源级 L1 tile load"，不声称物理直连。

### 7. GEMM epilogue 使用 GM workspace

当前 GEMM epilogue：`L0C → GM workspace → UB → bias/cast → output`。根本原因是 `T.copy(c_frag, o_ub)`（L0C→UB 直拷）在当前 TileLang 版本中编译器崩溃（`Find undefined Variable _`），GM workspace 是绕行方案。代价：一次额外 GM round-trip。本 PR 不处理，留作后续优化。

### 8. 输出 NCHW materialization 为独立步骤

GEMM 输出不是最终 NCHW，需要独立 materialize kernel。将 store 融合进 GEMM 后部分 case device 时间退化（case1/case16/case19），因此保留独立 materialize。

## 模块级状态审计

| 状态 | 保留？ | 原因 |
|------|--------|------|
| `_GATHER_OFFSET_CACHE` | 是 | vgather 偏移表，key 为 `(w_out, cst, dtype)`；设备安全，有界。 |
| `_BINDS` | 是 | fast-launch 绑定缓存，key 为 `id(kernel)`，强引用保持 kernel 活跃；不缓存计算结果。 |

## 受 TileLang 框架限制的优化方向

以下方向理论上可进一步提升性能，但受 TileLang 0.1.4 框架/编译器限制，当前无法实施。每一项均经代码注释或实测验证：

### 9. L0C→UB 直拷（消除 GEMM epilogue GM workspace）

- **理论收益**：当前 GEMM epilogue 为 `L0C → GM workspace → UB → bias/cast → output`，一次额外 GM round-trip。若 L0C 可直接拷到 UB，可省去整个 workspace 读写。
- **TileLang 限制**：`T.copy(c_frag, o_ub)`（L0C→UB 跨 CV 拷贝）在当前版本触发编译器内部错误 `Find undefined Variable _`，无法编译。GM workspace 是绕行方案。
- **实测证据**：`_conv2d_pre.py` P11 记录；本地 probe 复现编译器 crash。
- **解除条件**：TileLang 修复 L0C→UB 直拷代码生成（`copy_op.py` 已声明支持 `wmma.accumulator → shared.ub`，但 0.1.4 实际未实现）。

### 10. fp32 原生 MMAD（消除 hi/lo 3-GEMM）

- **理论收益**：当前 fp32 用 3 次 fp16 GEMM（`w_hi·x_hi + w_hi·x_lo + w_lo·x_hi`）模拟，计算量为原生 3 倍。
- **TileLang/工具链限制**：硬件 MMAD 支持 float×float（HF32，E8M11，20 位），但当前工具链（CANN 9.0.0 + dav-c220）未开放 fp32 cube 特性：AscendC 后端编译通过但运行时 AICore 异常，PTO 后端直接编译失败（`function type does not support the given target feature`）。且 HF32 精度（~11 位尾数）不满足 fp32 卷积 2^-13 要求。

### 11. bf16→fp16 直接 cast（消除双级 cast）

- **理论收益**：当前 bf16 数据经 `bf16 → fp32 → fp16` 两级 cast，每处多一份 UB fp32 缓冲和一次拷贝。
- **TileLang 限制**：代码注释明确 FORBIDDEN：`bf16->fp16 direct cast (route via fp32)`（`_conv2d_pre.py` L24）。
- **解除条件**：TileLang 增加 bf16→fp16 直接 cast 原语。

### 12. 动态循环边界（消除 R 整除约束）

- **理论收益**：当前 K 分块 R 必须整除 Cin_pad（`while R > 16 and Cin_pad % R != 0: R -= 16`），有时被迫选非最优 R。
- **TileLang 限制**：循环边界必须为编译期字面量（P-series 规则）：`a runtime T.min length lands in the copy template parameter and fails compilation`（`_conv2d_pre.py` L436 等 4 处）。
- **解除条件**：TileLang 支持动态循环边界后的拷贝长度。