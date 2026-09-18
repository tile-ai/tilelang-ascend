**中文** | [English](benchmark.md)

# MHC BWD（Sinkhorn 隐式 CG）性能基准

## 1. 算子

```
前向：R = sinkhorn(M)                        （双重随机矩阵）
反向：dL/dM = (dR - x1 - x2^T) * R
      其中 x1、x2 通过 2*n_stream 次共轭梯度（CG）迭代求解
      （隐式微分，避免对 20 次前向 Sinkhorn 迭代反传）
```

- 输入：out (R) [seqlen, n_stream, n_stream] fp32，dout (dR) 同 shape
- 输出：res (dL/dM) [seqlen, n_stream, n_stream] fp32
- 精度：相对 torch autograd 的 max_abs_diff < 1e-3

## 2. 硬件与软件

| 项目 | 值 |
|------|-----|
| NPU | Ascend 910B3 |
| CANN | 9.2.0 |
| 工具 | do_bench (tilelang.profiler)，warmup=10 ms, rep=100 ms, 中位数 |
| 数据类型 | fp32 输入 / fp32 累加 |

## 3. 架构（单 kernel，纯 Vector）

| 组件 | 选择 | 原因 |
|------|------|------|
| 核使用 | 每个 AI Core block 用单个 V 核（vid == 0 guard） | CG 流水线是 scalar dispatch 瓶颈，不是计算瓶颈 |
| buffer | 批量 [tilesize, NS] UB buffer | 一条指令处理 tilesize 行，摊薄每轮 CG 约 15 个小算子 |
| CG 循环 | T.serial(2 * n_stream) + T.Parallel 逐元素更新 | 短循环编译期展开，无需收敛判断 |
| tail 处理 | host 侧 pad 到 tilesize 整数倍，算完裁剪 | pad 行退化（梯度恰好为 0），pad-算-裁精确 |

kernel 按 tilesize 行分 block。host 适配器 `sinkhorn_bwd` 对非整除 seqlen 做零
填充：pad 行的 b1 = b2 = 0，CG 解退化为 0 梯度，最后裁剪掉 pad 行。

## 4. 优化路径

| 步骤 | 改动 | 效果 |
|------|------|------|
| baseline | PR head：批量单 V 核 kernel | seqlen=256 时 1.12 ms（do_bench）|
| tail 修复 | `sinkhorn_bwd` 适配器 host pad | 非整除 seqlen 不再越界（原实现会静默越界读写） |
| shape 测试 | 1 -> 6 个用例（seqlen 100-512，n_stream 8/16/32）| 6/6 通过 |
| 尝试：双 V 核逐 tile | 改成 [NS, NS] 逐 tile buffer + vid 0/1 分工 | 否决：慢 12-23%，批量宽指令更优 |
| dispatch 削减 | matvec 的 "+x" 外提出逐 tile 循环（2 次全块更新替代 2*tilesize 次）；跳过初始 `A @ 0` matvec（直接 r0 = b） | seqlen 2048/4096 快 3.1%（交错 A/B 两轮平均）；1024 以下在噪声内 |
| 测试移入 pytest | 批量 shape 测试移至 test_example_mhc_bwd.py，注册进 operator_test_manifest；example 只留单 shape 简单用例（seqlen=100 非整除，走 pad 路径） | 纳入 CI；共 6 个 shape，默认集按编译键选择（3 个 case = --forked 下 3 次 kernel 编译），其余 3 个 shape 标 low_priority |

双 V 核逐 tile 方案还暴露了两个 codegen 限制（均已绕开但得不偿失）：[1] 元素
UB buffer 的 `T.copy` 触发 aicore exception（507015）；`T.Parallel` 更新携带
跨迭代标量依赖会产生 NaN（改成 `T.tile` ops 可解）。批量设计两者都不涉及。

## 5. 最终性能（do_bench, warmup=10 ms, rep=100 ms, 中位数）

| seqlen | kernel | torch autograd (NPU) | 加速比 |
|--------|--------|----------------------|--------|
| 256 | 1.12 ms | 8.85 ms | **7.88x** |
| 512 | 1.08 ms | 9.32 ms | **8.61x** |
| 1024 | 1.15 ms | 9.12 ms | **7.91x** |
| 2048 | 1.95 ms | 9.19 ms | **4.71x** |
| 4096 | 3.90 ms | 9.04 ms | **2.32x** |

说明：

- seqlen < 1024 时 kernel 在 910B3 上处于 dispatch 瓶颈（约 1.1 ms 固定
  开销：32-128 个小 block，每个约 500 条标量指令）。从 2048 起延迟随行数
  线性增长，与原 PR 在 910B 上的 do_bench 数字一致（2048 为 2.19 ms，
  4096 为 4.38 ms，本构建略快）。
- torch autograd 耗时主要来自对 20 次前向 Sinkhorn 迭代的反传（launch 开销
  固定，随 seqlen 几乎不变）——这正是隐式 CG 方案所规避的。

## 6. 精度

| 指标 | 值 |
|--------|-------|
| 测试用例 | 6/6 通过（seqlen 100/250/256/512，n_stream 8/16/32，含非整除 seqlen）|
| 容差 | 相对 torch autograd 的 max_abs_diff < 1e-3 |
| 最大差异 | 8.16e-07（seqlen=250, n_stream=8）|
| 相对 manual-CG 参考 | 1.79e-07 max |
| 差异来源 | CG 迭代中 fp32 reduce 累加顺序 |

## 7. 已知限制

每 block 单 V 核（vid 1 空闲）：CG 流水线是 scalar dispatch 瓶颈（每轮约 15 个
小算子作用在 8-16 元素的行上），因此批量单核设计比双核逐 tile 方案快 12-23%。
要用上第二个 V 核需要更宽的每指令行数（更大 tilesize），而批量设计已通过
y1/y2 的 [tilesize, NS] reduce 实现了这一点。