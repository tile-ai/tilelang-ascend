**中文** | [English](benchmark.md)

# MHC Pre 性能基准与优化路径

## 1. 算子

```
mHC Pre 前向流水线：
  1. out = x @ fn.T, sqrsum = x^2.sum(-1)
  2. mixes = out * rsqrt(sqrsum / (hc * hidden) + rms_eps)
  3. pre/post/comb = split(mixes) + Sinkhorn 归一化
  4. layer_input = sum over hc of (residual * pre_mix)
```

- 输入：residual [n, hc, hidden] bf16, fn [hc_mult3, hc*hidden] fp32, hc_scale [3] fp32, hc_base [hc_mult3] fp32
- 输出：post_mix [n, hc, 1] fp32, comb_mix [n, hc, hc] fp32, layer_input [n, hidden] bf16
- 约束：1 <= hc <= 8（B3 AXPY，JIT 参数，已验证范围）

## 2. 硬件与软件

| 项目 | 值 |
|------|-----|
| NPU | Ascend 910B |
| CANN | 9.0.0 |
| 工具 | do_bench (Python), msprof op (硬件级) |
| 数据类型 | bf16 输入, fp32 累加 |

## 3. 架构（3-kernel 流水线，从原始 5 个融合）

| Kernel | 功能 | 核类型 | 关键参数 |
|--------|------|--------|---------|
| A1 | GEMM: out = x @ fn.T | Cube (T.gemm_v0) | token_block=128, h_blk=512, T.Pipelined |
| A2+B1（融合） | sqrsum + RMSNorm | Vector (双 V 核) | sqr_h_blk=4096, T.Pipelined, kernel 内 tail |
| B2+B3（融合） | split + Sinkhorn + apply pre_mix | Vector (双 V 核) | T.alloc_shared 用于 Sinkhorn, UB 用于 apply, T.unroll(hc) |

原始 5-kernel 流水线（A1 + A2 + B1 + B2 + B3）通过合并 A2+B1（sqrsum 结果留 UB）
和 B2+B3（pre_mix 留 shared/L1）融合为 3 个 kernel。

A1 使用 Cube GEMM。A2+B1 和 B2+B3 使用双 V 核划分（bid = cid * 2 + vid）。

## 4. 优化路径

| 步骤 | 改动 | 效果 |
|------|------|------|
| A1 token_block | 16 -> 128 | 提升 Cube 利用率 |
| A1 h_blk | 128 -> 512 | K-tile sweep 最优值 |
| A1 删 guard | 删 h_num=1 T.serial guard | Guard 在 910B CI 失败（post §3.10） |
| A2 h_blk | 128 -> 4096 | 减少循环次数 |
| A2 kernel 内 tail | pad_value + TAIL_MASK | 删除 host sqrsum pad |
| B3 2D merged load | 4 个独立 1D -> 2D res_ub[hc, h_blk] | 1 次 T.copy 替代 hc 次 |
| B3 hc 泛化 | hc=4 硬编码 -> hc 1-8 JIT | T.unroll(hc), assert 1<=hc<=8 |
| B3 kernel 内 tail | pad_value + TAIL_MASK | 删除 host _pad_3d |
| A2+B1 融合 | sqrsum + RMSNorm 合一 | 省 1 次 launch + sqrsum GM 往返 |
| B2+B3 融合 | Sinkhorn + apply 合一 | 省 1 次 launch + pre_mix GM 往返 |
| B2 T.unroll(hc) | T.serial(hc) -> T.unroll(hc) | 编译期展开（无性能变化，意图更清晰） |
| pass_configs | 加 TL_ASCEND_TAIL_MASK | 启用 pad_value 支持 kernel 内 tail |
| fn 预打包/缓存 | prepare_fn + fn_packed | 避免推理时重复 cast/transpose |
| kernel 编译缓存 | _kernel_cache dict | 避免重复 JIT 查找 |

## 5. 最终性能（E2E, do_bench, warmup=20, rep=100, 5 次平均, 预打包 fn）

| n | h | hc | TileLang | PyTorch (CANN) | 加速比 |
|---|---|---|----------|----------------|---------|
| 512 | 2560 | 4 | 1.59 ms | 1.23 ms | 0.77x |
| 4096 | 2560 | 4 | 1.65 ms | 2.31 ms | **1.40x** |
| 4096 | 7168 | 4 | 1.99 ms | 5.22 ms | **2.62x** |

小 shape（512x2560）比 CANN 慢是因为 3 次 kernel launch 固定开销。
大 shape 从融合流水线受益。4096x2560 在 kernel 融合后从 0.96x 提升到 1.40x。

## 6. Kernel 分解（n=4096, h=2560, 融合后）

| Kernel | 延迟 | 占比 |
|--------|------|------|
| A1 GEMM | 0.30 ms | 18.7% |
| A2+B1 sqrsum+RMSNorm | 0.30 ms | 18.5% |
| B2+B3 sinkhorn+apply | 0.86 ms | 52.7% |
| Host 开销（3 次 launch）| 0.16 ms | 10.0% |

B2+B3 融合 kernel 是主导组件（52.7%）。Host 开销从 0.32 ms（5 次 launch）
降到 0.16 ms（3 次 launch）。

## 7. B2 Sinkhorn 优化尝试

B2（Sinkhorn）曾是 #1 瓶颈（融合前 28-39%）。尝试了 6 种优化方案，全部被
codegen 限制挡住：

| 方案 | 结果 | 阻塞原因 |
|------|------|---------|
| T.alloc_shared -> T.alloc_ub | 507015 | 1D UB slice 作为 T.copy dst/src (§2.2b) |
| T.tile.cast 替代 1D->2D-row | 精度错误 | column reduce 不支持 narrow real_shape (dim=0) |
| T.Scope("V") + T.alloc_shared | 507015 | V scope 导致 shared 分配到 UB |
| T.serial(hc) -> T.unroll(hc) | 无变化 (+0.1%) | 编译器已自动展开短循环 |
| T.unroll(sinkhorn_iters) | 无变化 (-0.6%) | 噪声范围 |
| 消除 workspace GM 往返 | 507015 | 1D shared slice 作为 T.copy source |

msprof 分析显示 B2 **三路均衡**（Vec 18.8% / MTE 19.4% / Scalar 18.3% /
Wait 25.2%）——无单一主导组件。瓶颈来自 ~130 个小操作在 4-8 元素 buffer 上的
scalar dispatch 开销，不是计算或内存。

## 8. 精度

| 指标 | 值 |
|--------|-------|
| 测试用例 | 7/7 通过（含 distinct-eps 参数测试）|
| 容差 | rtol=1e-2, atol=1e-2 |
| 最大差异 | 0.0156 (layer_input, n=4096) |
| 差异来源 | BF16 量化和 AXPY apply kernel 的不同累加顺序 |

## 9. 停止条件

| 条件 | 状态 |
|-----------|--------|
| E2E > CANN（大 shape）| 是（1.40x - 2.62x）|
| Kernel 融合完成 | 是（5 -> 3 kernel，延迟 -28.3%）|
| B2 Sinkhorn 已优化 | 被 codegen 阻塞（尝试了 6 种方案）|
| 所有参数正确路由 | 是（distinct-eps 测试验证）|
| Host 开销最小化 | 是（3 次 launch，10%）|

优化停止：5-kernel 流水线融合为 3 个 kernel（A1 + A2B1 + B2B3）。B2+B3 融合
kernel 是主要瓶颈（52.7%），但 B2 Sinkhorn 被 codegen 限制阻塞（1D UB/shared
slice 507015，column reduce narrow real_shape）。进一步优化需要 codegen 支持
1D buffer slice T.copy 或 narrow column reduce。
