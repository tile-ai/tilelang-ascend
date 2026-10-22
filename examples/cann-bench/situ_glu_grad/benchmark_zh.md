**中文** | [English](benchmark.md)

# SiTU-GLU Grad（SituGluGrad）性能基准

## 1. 算子

```
t = tanh(gate / beta),  s = sigmoid(gate),  situ_a = beta * t * s
up' = linear_beta * tanh(up / linear_beta)   (linear_beta > 0)，否则 up' = up
grad_gate = (grad_y * up') * s * [(1 - t^2) + beta * t * (1 - s)]
grad_up   = (grad_y * situ_a) * (1 - tanh^2(up / linear_beta))   (linear_beta > 0)
grad_up   = (grad_y * situ_a)                                   (linear_beta <= 0)
```

- Kimi K3 Stable LatentMoE 门控激活（SiTU-GLU）的反向；`x` 按 `activate_left`
  沿 `dim` 轴切分为 gate / up 两半
- 输入：x [..., cut]（cut 为偶数）、grad_y [..., cut/2]；fp32 / fp16 / bf16
- 输出：grad_x，shape 与 dtype 与 x 一致
- 配置：dim=-1、beta=4.0、linear_beta=25.0、activate_left=True（Kimi K3 生产配置）
- 参考实现：CANN ops-nn `activation/situ_glu_grad`（aclnnSituGluGrad / `cann_ops_nn.situ_glu_grad`）

## 2. 硬件与软件

| 项目 | 值 |
|------|-----|
| NPU | Ascend 910B3（Atlas A2） |
| CANN | 9.2.0 |
| 框架 | torch 2.7.1 + torch_npu 2.7.1.post4 |
| 基线 | `cann_ops_nn.situ_glu_grad`（官方 AscendC kernel） |
| 测量方式 | NPUGraph 捕获回放，Kernel-only |

## 3. 实现架构（单 kernel，纯 Vector）

| 组件 | 选择 | 理由 |
|------|------|------|
| 核间划分 | 每个 block 双 V 核行拆分（`vid` 拆分行 tile） | 逐元素算子，带宽 bound |
| 分块 | 16x128 tile，沿列 strip 做 `T.Pipelined(num_stages=2)` | 重叠 MTE2 加载 / 向量计算 / MTE3 写回 |
| 计算精度 | fp32 中间计算，`CAST_RINT` 写回 | 与官方实现内部精度策略一致 |
| tanh | `2 * sigmoid(2x) - 1` | 无 tanh tile 原语；数值稳定 |
| 编译期特化 | (dtype, beta, linear_beta > 0, activate_left) | 恒等分支不产生 `dtu` 相关算子 |
| Host 适配 | 合轴为 [rows, cut*inner]，零填充到 block 整数倍后裁剪 | 对齐 shape（含 Kimi）零填充开销 |

## 4. 性能调优

- **strip 选择**：`tiles_per_block` 由形状感知的代价模型
  （`waves * (c * per_tile + fill)`）选择，权衡长 strip（降低每任务流水
  fill 开销）与波次数量。上机验证：M=512 -> 4（47.2 µs，24 时为 50.1 µs）、
  M=1024/2048 -> 8、M=4096 -> 12、M=8192 -> 24。朴素“让每个核都忙”
  （短 strip）规则在所有 shape 上都会回退。
- **带宽 bound**：读 1.5x cut 元素（x + grad_y），写 1x cut（grad_x）；
  优化方向是打满内存带宽而非算术。

## 5. 性能（Kernel-only，Kimi K3 形态）

形态：x [M, 6144] bf16、grad_y [M, 3072]、dim=-1、beta=4.0、linear_beta=25.0、
activate_left=True。协议：NPUGraph 捕获回放（抹除 host 开销），每配置
warmup 5 / rep 20，交错 A/B 取中位数。

| M | tilelang kernel | aclnn 基线 | Speedup |
|---|-----------------|------------|---------|
| 512 | 47.3 µs | 42.6 µs | 0.90x |
| 1024 | 83.9 µs | 80.3 µs | 0.96x |
| 2048 | 161.6 µs | 156.3 µs | 0.97x |
| 4096 | 305.3 µs | 305.9 µs | 1.00x |
| 8192 | 595.2 µs | 605.4 µs | 1.02x |

说明：
- M=512 属小 batch 形态：每任务固定的流水 fill/launch 开销占比高，
  kernel-only 为 0.90x；M>=1024 与官方差距在 4% 以内，M=4096/8192 达到
  持平或略优。

## 6. 精度（910B3，对比独立 fp64 golden）

混合容差：`abs_err <= atol + rtol * |golden|`，`matched_ratio >= 0.99`；
atol/rtol = 2^-10 / 2^-6（bf16）、2^-14 / 2^-9（fp16）、2^-16 / 2^-10（fp32）。

| 分层 | 覆盖内容 | 结果 |
|------|----------|------|
| L0 | bf16 / fp16 / fp32 x Kimi 配置 + 默认参数 | 4/4 |
| L1 | dim 变体（0/1/-2/-1，2D-4D）、activate_left=False、linear_beta=0/负值、beta=100、非对齐 cut/rows、非连续输入、M=1、小 shape | 17/17 |
| L2 | 奇数 cut / beta=0 / grad_y shape 不匹配 / 非法 dim（拒绝） | 4/4 |
| Boundary | 空张量 / 饱和值 / 极小值 | 2/2 |

关键数字：bf16 Kimi max_abs = 1.56e-2（matched_ratio = 1.0）；fp16 = 1.95e-3；
fp32 = 9.06e-6。与官方 aclnn 交叉校验：最大相对偏差 ~3e-3（bf16），符合
bf16 舍入预期。

## 7. 说明与限制

- 非整除 shape 在 host 侧零填充；填充区域产生零梯度并被裁剪，对齐 shape
  无额外开销。
- strip 代价模型常数基于 A2（910B3，40 AIV 核）标定。
- M=512 相对官方 kernel 低于 1.0x（小 batch 开销所致）；如实呈现，不以
  prefill 规模形态的性能为代价做交换。
