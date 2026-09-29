# Conv2D

TileLang Conv2D 算子，用于 CANN-Bench 评测集（`tasks/level3/conv_2d`）。

## API

```python
def conv_2d(
    x: torch.Tensor,      # [N, Cin, H, W]
    filter: torch.Tensor,  # [Cout, Cin, Kh, Kw]
    bias: torch.Tensor,    # [Cout]
    strides: list,         # [sh, sw]
    pads: list,            # [pt, pb, pl, pr]
    dilations: list = None, # [dh, dw]
) -> torch.Tensor:         # [N, Cout, Hout, Wout]
```

## 支持的数据类型和 shape

| 数据类型 | GEMM 策略 | 说明 |
|----------|-----------|------|
| float16 | direct im2col + `T.gemm_v0` | 默认路径。小输出（n_real < 8192）使用融合 GEMM+NCHW。 |
| bfloat16 | 同 float16 | bf16→fp32→fp16 cast 在预处理 kernel 中完成。 |
| float32 | fp16 hi/lo 拆分，3× `T.gemm_v0` | `w_hi·x_hi + w_hi·x_lo + w_lo·x_hi`，fp32 累加器。硬件 MMAD 支持 HF32（float×float）但工具链未开放，且精度不满足要求。 |

- 1×1 kernel：纯 GEMM 快速路径（无 im2col）。
- 矩形 kernel：任意 (kh, kw)。
- 步长：任意 (sh, sw)。stride=2 使用 compact row GEMM + band-gather materialization。
- 膨胀：任意 (dh, dw)。
- 填充：非对称 [top, bottom, left, right]。
- 通道尾部：非 16 对齐的 Cin 内部填充到 16 对齐。

## 实现概览

```
输入 (NCHW)
  ├─ 1×1? → Input transpose → 纯 GEMM → NCHW materialize
  └─ 其他 → 预处理 (pad + cast + weight transpose)
               ↓
            直接 L1 im2col + T.gemm_v0
               ↓
            Bias + cast → NCHW materialize
```

通用路径**不生成完整 global im2col workspace**。每个 tile 通过连续 `T.copy` DMA 直接将当前 patch 加载到 L1。

### 1×1 路径

标准 GEMM：`y[co, m] = Σ_ci w[co, ci] · x[ci, m] + b[co]`。无 im2col。

### 直接 local-im2col 路径

每个 GEMM tile 加载：
- Weight：`[BLOCK_M, BLOCK_K]` 从 tap-major 布局 → L1。
- Input：每 tap 16-wide DMA 从 padded plane → L1。
- Cube GEMM → L0C 累加器 → bias → cast → output。

### FP32 hi/lo 路径

Ascend910 `T.gemm_v0` 使用 fp16/bf16 操作数。FP32 近似为：

```
w_hi·x_hi + w_hi·x_lo + w_lo·x_hi
```

其中 `hi = fp16(x)`，`lo = fp16(clamp(x) - fp32(hi))`。`w_lo·x_lo` 项省略（二阶项，对范围内值保持 fp32 精度）。

## 运行最小示例

```bash
# 设置 TileLang 环境
source /path/to/tilelang-ascend/set_env.sh

# 运行自包含测试
python examples/cann-bench/conv2d/example_conv2d.py
```

预期输出：`Test Passed!`

## 运行 CANN-Bench 评测

```bash
# 创建临时评测包
mkdir /tmp/conv2d_pkg && cp examples/cann-bench/conv2d/example_conv2d.py /tmp/conv2d_pkg/
# （需创建 setup.py + __init__.py 包装为 cann_bench 包）

# 运行正确性评测
bash scripts/run_evaluation.sh \
    --bench-name cann \
    --task-dir tasks/level3/conv_2d \
    --operator Conv2D \
    --source-dir /tmp/conv2d_pkg \
    --device-id 0 \
    --no-perf

# 运行性能评测（需 profiler 环境正常）
bash scripts/run_evaluation.sh \
    --bench-name cann \
    --task-dir tasks/level3/conv_2d \
    --operator Conv2D \
    --source-dir /tmp/conv2d_pkg \
    --device-id 0
```

## 评测摘要

详见 `evaluation.md`。

- 正确性：20/20 全部通过。
- 几何平均加速比（本地）：0.16x。
- fp16 8 用例、bf16 5 用例、fp32 7 用例全部通过。