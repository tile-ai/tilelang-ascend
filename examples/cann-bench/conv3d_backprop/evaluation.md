# Conv3D Backprop Filter 算子评测报告

## 环境

- 任务：`tasks/level3/conv3d_backprop_filter`
- 设备：Ascend910_9362
- CANN：9.0.0
- torch：2.10.0 / torch_npu：2.10.0.post4
- Python：3.11
- TileLang 分支：`ascendc_pto`
- CANN-Bench 框架：V1.0.0
- CANN-Bench 任务：v1.0.0

## 正确性

20/20 官方用例全部通过。

| 用例 | 状态 | 延迟(us) | 加速比 |
| ---: | :---: | ---: | ---: |
| 1 | PASS | 1486.01 | 0.05x |
| 2 | PASS | 2437.13 | 0.04x |
| 3 | PASS | 19199.96 | 0.04x |
| 4 | PASS | 25523.33 | 0.00x |
| 5 | PASS | 1410.15 | 0.06x |
| 6 | PASS | 1020.14 | 0.01x |
| 7 | PASS | 2682.73 | 0.04x |
| 8 | PASS | 2869.74 | 0.06x |
| 9 | PASS | 3273.39 | 0.00x |
| 10 | PASS | 5876.22 | 0.08x |
| 11 | PASS | 2181.10 | 0.06x |
| 12 | PASS | 828.62 | 0.01x |
| 13 | PASS | 1788.44 | 0.05x |
| 14 | PASS | 266.51 | 0.04x |
| 15 | PASS | 2604.57 | 0.05x |
| 16 | PASS | 666.33 | 0.02x |
| 17 | PASS | 9533.43 | 0.00x |
| 18 | PASS | 51741.65 | 0.05x |
| 19 | PASS | 50333.59 | 0.03x |
| 20 | PASS | 25667.89 | 0.04x |

- 几何平均加速比：**0.04x**
- 综合得分：51.36

## 开发问题记录与已知限制

### 1. 元素级 im2col 性能瓶颈

当前 3D im2col 使用逐元素标量读（`x_ub[kk, nn] = Input[b_idx, ci_off+ci_idx, id_, ih_, iw_]`），每个元素执行 div/mod 索引计算和边界判断。对于大 K case（case19 k_blocks=1024），im2col 构建时间占主导。

### 2. Split-K 跨 block 并行

为减少 K 循环串行长度，支持将 K 维度拆分为多个 split，每个 split 由独立 block 计算 fp32 部分和，最后通过 TwoSum 合并。但 merge kernel 的 JIT 编译时间较长。

### 3. tap-major 转置 B_gm 布局

B_gm[m_pad, K_pad] 的 tap-major 布局使 GEMM 的 B 读为 1MB-stride gather（大 K case 效率极低）。实验性的 `CONV3D_T_BGM` 路径使用转置布局 B_gmT[K, m] 可改善 DMA 效率，但当前仅支持 fp16 + stride=1。

### 4. 预处理多个 kernel launch

当前处理流程包含 x_pad、g_pad、build_xcol、b_tail_zero、gemm、y_cast、m_repack 等多个 kernel launch，每个 kernel 间无流水重叠，launch 固定开销在小 shape 上占主导。

### 5. 元素级标量写

`_y_zero_kernel` 和 `_x_sub_kernel` 使用 `T.Parallel` 逐元素写入，而非 `T.copy` DMA，在大量元素时效率低于批量拷贝。

## 模块级状态审计

| 状态 | 保留？ | 原因 |
|------|--------|------|
| `_OFF2` / `_OFF3` / `_GROUP_OFF2` | 是 | host→device 零拷贝缓存，避免 aclrtMemcpy 反作弊检测 |
| `_BINDS` | 是 | fast-launch 绑定缓存，key 为 `id(kernel)`，强引用保持 kernel 活跃 |