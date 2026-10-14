**中文** | [English](benchmark.md)

# SituGlu 性能基准

## 1. 算子

```
situ_a = beta * tanh(gate / beta) * sigmoid(gate)
up'    = linear_beta * tanh(up / linear_beta)   (linear_beta > 0)，否则 up' = up
y      = situ_a * up'
```

- SiTU 门控线性单元，Kimi K3 Stable LatentMoE 的前向门控激活；`x` 按
  `activate_left` 沿 `dim` 轴切分为 gate / up 两半
- 输入：x [..., cut]（cut 为偶数），1-8 维，fp32 / fp16 / bf16，ND
- 输出：y [..., cut/2]，dtype 与 x 一致
- 配置：dim=-1、beta=1.0、linear_beta=0.0、activate_left=True（默认值）
- 参考实现：CANN ops-nn `activation/situ_glu`（aclnnSituGlu / `cann_ops_nn.situ_glu`）

## 2. 硬件与软件

| 项目 | 值 |
|------|-----|
| NPU | Ascend 910_9362（Atlas A3），40 AIV 核 |
| CANN | 9.0.0 |
| 框架 | torch 2.7.1 + torch_npu 2.7.1.post4 |
| 基线 | 由 ops-nn 源码独立编译的官方 AscendC kernel（`op_kernel/situ_glu.hpp` 原码 + learning-hub 直调 harness + 复刻 op_host tiling）；aclnnSituGlu 随 CANN 9.2.0 发布，9.0.0 环境不可用 |
| 测量方式 | aclrtEvent 事件对包住 N 次在途 launch，kernel-only，5 轮取中位数 |

## 3. 实现架构（单 kernel，纯 Vector）

| 组件 | 选择 | 理由 |
|------|------|------|
| 核间划分 | 每个输出 tile 一个 grid block，双 V 核行拆分（`vid`） | 逐元素算子，带宽 bound |
| 分块 | (block_M, block_N) 输出 tile，按 `sub_M` 行块推进 | 行块是流水单元，见下 |
| 流水 | 显式 flag 双缓冲：第 mm+1 行块的 MTE2 搬入与第 mm 行块的向量计算、第 mm-1 行块的 MTE3 写回重叠（`set_flag`/`wait_flag`，`elementwise_add_pipeline.py` 模式） | 不开启任何自动同步 pass（自动屏障会串行化循环并与手动 flag 冲突） |
| shape 策略 | `M`/`N` 默认 symbolic（一次编译服务任意 shape）；传具体 int 得到静态特化（dispatch 更轻） | 兼顾 serving 通用性与固定 shape 部署 |
| 计算精度 | fp32 中间计算，bf16 回写 `CAST_RINT`（fp16 用 `CAST_NONE`） | 与官方实现内部精度策略一致；非 A2 平台 bf16 必须 CAST_RINT |
| tanh | `2 * sigmoid(2x) - 1` | 无 tanh tile 原语；数值稳定（大输入正确饱和，不像 exp 差商形式产生 `inf/inf = NaN`） |
| 尾块处理 | ceildiv 网格 + 标量起点全 tile `T.copy`；前端自动把拷贝范围 clamp 到张量边界并对读入补零 | 无 host 侧 padding / 裁剪 |
| 特化 | (dtype, beta, linear_beta > 0, activate_left) 编译期特化 | `linear_beta <= 0` 分支不含有界 tanh 计算；beta == 1 折叠标量乘 |

## 4. 性能调优

tile 标定（(4096, 18432) bf16，静态特化，kernel-only）：

| tile（block_M x block_N, sub_M） | 时延 | 带宽 |
|---|---|---|
| 128 x 128, 32 | 514.9 us | 440 GB/s |
| 256 x 128, 32 | 455.0 us | 498 GB/s |
| 256 x 128, 64 | 420.7 us | 538 GB/s |
| 512 x 128, 32 | 432.3 us | 524 GB/s |
| 256 x 256, 32 | 360.3 us | 629 GB/s |
| 512 x 64, 32 | 572.3 us | 396 GB/s |
| **512 x 256, 32** | **352.5 us** | **643 GB/s** |

更宽的 `block_N` 胜在拉长连续 DMA 段；更深的 block（更大的
`block_M / sub_M`）摊薄流水 prologue/epilogue。host 适配层按输入 shape
选择 tile（纯 metadata 启发式）：M >= 4096 -> (512, 256)；
M >= 1024 -> (256, 256)；否则 (128, 128)。

## 5. 性能（kernel-only）

shape 采用 LLM FFN 剖面（cut = 2 x hidden）。协议同第 2 节。

| shape | dtype | tilelang | 官方 kernel | 加速比 | tilelang 带宽 | 未融合 torch |
|---|---|---|---|---|---|---|
| (4096, 18432) | bf16 | 352.5 us | 727.6 us | 2.06x | 643 GB/s | 1421 us |
| (8192, 18432) | bf16 | 678.5 us | 1444.0 us | 2.13x | 668 GB/s | 2746 us |
| (4096, 18432) | fp16 | 348.1 us | 723.1 us | 2.08x | 651 GB/s | 1402 us |
| (4096, 18432) | fp32 | 520.1 us | 816.6 us | 1.57x | 871 GB/s | 2158 us |

"未融合 torch" = `chunk + tanh + sigmoid + mul` 的独立 torch 算子组合
（本算子所替代的融合前形态）。

host 侧说明：symbolic shape 每次调用有固定 ~0.57 ms 的 dispatch 开销
（JIT 适配层的逐次 shape 推断），静态特化约 ~0.18 ms。上表 kernel-only
数字不含该项；小 batch 场景下端到端时延由 dispatch 主导。

## 6. 精度

Ascend 910_9362，对比 fp64 PyTorch golden。分层套件：L0（9）/ L1（24）/
L2 负向（9）/ Boundary（8）——覆盖 dtype 全集、非对齐/质数/退化 shape、
1-8 维、非默认参数、稀疏 ±inf / NaN（位置敏感 mask 比较）、dtype 边界值、
空张量。全量 3 轮重复，无 flaky。

混合容差 `abs_err <= atol + rtol * |golden|`，`matched_ratio >= 0.99`：

| dtype | atol | rtol |
|---|---|---|
| float16 | 2^-14 | 2^-9 |
| bfloat16 | 2^-10 | 2^-6 |
| float32 | 2^-16 | 2^-10 |
