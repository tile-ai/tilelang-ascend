# RoPE 基准测试：TileLang vs aclnnRotaryPositionEmbedding

TileLang RoPE（`rope_half_interleaved.py`）对比 CANN 官方算子
`aclnnRotaryPositionEmbedding`（`ops-transformer/posembedding/rotary_position_embedding`），
涵盖**性能**与**精度**两部分。

## 一、性能测试

### 测量方法

- **工具**：`msprof` device 侧 Task Duration（取自 `op_summary_*.csv`，逐次 launch）
- **协议**：`--repeats 6`，丢首次 launch（冷启动 / DVFS），取其余平均
- **加速比**：`ac_us / tl_us`（>1.0 表示 TileLang 更快）
- 两侧使用相同 shape / dtype / layout / seed=0

### 性能结果


| 场景                   | Shape          | Layout      | Dtype    | TileLang (us) | AscendC (us) | 加速比 |
| ---------------------- | -------------- | ----------- | -------- | ------------- | ------------ | ------ |
| decode_bs1             | 1 32 128 128   | half        | float16  | 7.98          | 7.63         | 0.96x  |
| decode_bs64            | 64 64 128 128  | half        | float16  | 14.61         | 19.82        | 1.36x  |
| prefill_bs4_h64        | 4 64 128 128   | half        | float16  | 6.99          | 18.89        | 2.70x  |
| prefill_bs8_h64        | 8 64 128 128   | half        | float16  | 7.34          | 19.90        | 2.71x  |
| prefill_bs32_h64       | 32 64 128 128  | half        | float16  | 10.09         | 20.08        | 1.99x  |
| prefill_bs4_d256       | 4 64 256 256   | half        | float16  | 9.43          | 18.99        | 2.01x  |
| prefill_bs4_d512       | 4 64 512 512   | half        | float16  | 12.98         | 18.68        | 1.44x  |
| prefill_bs4_bf16       | 4 64 128 128   | half        | bfloat16 | 7.61          | 18.81        | 2.47x  |
| prefill_bs8_bf16       | 8 64 128 128   | half        | bfloat16 | 9.90          | 19.88        | 2.01x  |
| prefill_bs4_inter      | 4 64 128 128   | interleaved | float16  | 15.06         | 19.00        | 1.26x  |
| prefill_bs8_inter      | 8 64 128 128   | interleaved | float16  | 15.03         | 19.32        | 1.29x  |
| prefill_bs4_inter_bf16 | 4 64 128 128   | interleaved | bfloat16 | 15.44         | 18.40        | 1.19x  |
| bsnd_bs4_s4            | 4 4 64 128 128 | half        | float16  | 9.38          | 21.54        | 2.30x  |

### 关键观察

- **TileLang 在 13 个场景中胜出 12 个**，加速比 1.19x–2.71x。
- **half layout 收益最大**（1.36x–2.70x）：TileLang 的 copy-swap 路径（前后半交换）避免了 AscendC `RotateHalf` 实现中的 gather 开销。
- **interleaved layout 收益较小**（1.19x–1.29x）：两侧均采用 gather-based rotate，差距主要来自 TileLang 的 UB 调度与多核切分。
- **decode_bs1 是唯一 AscendC 略胜的场景**（0.96x）：该 shape 极小（M=32，单核），启动开销占主导，AscendC 固定功能路径开销更低。
- AscendC 延迟在不同 shape 下非常稳定（~18–21 us），说明其 tiling 策略对 shape 变化自适应较差；TileLang 随实际工作量缩放，中大 shape 收益更大。

## 二、精度测试

### 当前方案

精度 golden 目前采用 **PyTorch CPU fp32 参考实现**（`torch_rope_ref`），即用小算子拼接在 CPU 上模拟 RoPE：

```
x_part = x[..., dim_start:].float()
x_rotated = layout 相关的 rotate（stack / cat）
out = (x_part * cos + x_rotated * sin).to(x.dtype)
```

### 精度标准

混合容差双门限（`check_precision`）：


| dtype    | atol            | rtol           | max_abs_limit | 要求匹配率 |
| -------- | --------------- | -------------- | ------------- | ---------- |
| float16  | 2^-14 (6.10e-5) | 2^-9 (1.95e-3) | 1e-1          | 99%        |
| bfloat16 | 2^-10 (9.77e-4) | 2^-6 (1.56e-2) | 1e0           | 99%        |
| float32  | 2^-16           | 2^-10          | 1e-2          | 99%        |

通过条件：匹配率 ≥ 要求 **且** 最大绝对误差 ≤ max_abs_limit。

### 测试覆盖


| 级别     | 用例数 | 覆盖范围                                                       |
| -------- | ------ | -------------------------------------------------------------- |
| L0 门槛  | 6      | half/interleaved × fp16/bf16 × TND/BSND 核心组合             |
| L1 功能  | 12     | 部分/全旋转、变 head_num/rope_dim、tail 行、大 batch、最小用例 |
| L2 负向  | 4      | 奇数 rope_dim、1D/5D 输入、rope_dim > hidden_size（须抛异常）  |
| boundary | 6      | 大值、零值、全旋转大 batch、最小 rope_dim、单行、BSND tail     |

**当前状态**：L0 / L1 / boundary 全部通过（torch_rope_ref 作为 golden）。
