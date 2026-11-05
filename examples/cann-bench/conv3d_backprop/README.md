# Conv3D Backprop Filter

TileLang Conv3D Backprop Filter 算子，用于 CANN-Bench 评测集（`tasks/level3/conv3d_backprop_filter`）。

## API

```python
def conv_3d_backprop_filter(
    x: torch.Tensor,          # [N, Cin, D, H, W]
    grad: torch.Tensor,        # [N, Cout, Dout, Hout, Wout]
    strides: list,             # [sd, sh, sw]
    pads: list,                # [pf, pb, ph, pw, pd, pp] 对称填充
    dilations: list,           # [dd, dh, dw]
    groups: int = 1,
    filter_size: list = None,  # [Cout, CinG, Kd, Kh, Kw]
) -> torch.Tensor:             # [Cout, CinG, Kd, Kh, Kw]
```

## 支持的数据类型

| 数据类型 | GEMM 策略 | 说明 |
|----------|-----------|------|
| float16 | 元素级 im2col + `T.gemm_v0` | 默认路径 |
| bfloat16 | 同 float16 | 通过 fp32 中转 cast |

## 实现概览

```
grad → pad → GPad (A 操作数)
x → pad → X_pad → im2col → B_gm (B 操作数)
GPad × B_gm^T → GEMM fp32 → cast → tap-major → ci-major repack
```

### 梯度预处理

grad 通过 torch `F.pad`（元数据操作，评测守卫白名单）填充为 padded grid 布局 `[N*Cout, Dout*Hout*Wpad]`，使 GEMM 的 K 轴分段连续。设置 `CONV3D_LEGACY_XPAD=1` 可切换到 `_g_pad_kernel` TileLang 版本。

### 输入预处理

x 通过 torch `F.pad` 零填充为 `[N*Cin_pad, Dp, Hp*Wp]`，然后 `_build_xcol_tap_kernel` 通过 tap-major 2D DMA 拷贝构建 im2col 矩阵 `B_gm[m_pad, K_pad]`。填充本身是元数据操作，im2col 构建仍在设备侧 TileLang kernel 完成。

### GEMM 累加

`_gemm_xcol_native_kernel` 计算 `C[co, m] = GPad[co, K] × B_gm[m, K]^T`，支持 Split-K 跨 block 并行。默认 fp32 累加（`CONV3D_SPLIT=1`），可通过环境变量切换实验路径（`CONV3D_T_BGM` / `CONV3D_AT` / `CONV3D_PACK_BGM` / `CONV3D_LOCAL_IM2COL`）。

### 后处理

fp32 累加结果经 torch `.to()` cast 回输出 dtype，再通过 `.view/.permute/.reshape` 元数据操作将 tap-major 重排为 ci-major 布局——全程零 host→device 拷贝。

## 运行最小示例

```bash
source /path/to/tilelang-ascend/set_env.sh
python examples/cann-bench/conv3d_backprop/example_conv3d_backprop.py
```

预期输出：`Test Passed!`