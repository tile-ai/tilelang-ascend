# pto-isa Cube 量化矩阵乘（QMM）代码总结

## 1. 核心概念

在 Ascend NPU 上，**Cube** 是矩阵乘硬件单元（类似 NVIDIA Tensor Core）。量化矩阵乘通过 **MXFP（Microscaling FP）** 格式实现，对应 ISA 指令 **`TMATMUL_MX`**。仓库中不使用 "qmm" 这个术语，统一使用 `TMATMUL_MX` / `MX` 命名。

## 2. 指令体系（3 条核心指令）

| 指令 | 功能 | 文件位置 |
|------|------|----------|
| **`TMATMUL_MX`** | 量化矩阵乘（带缩放 Tile） | `include/pto/common/pto_instr.hpp:457-512` |
| **`TQUANT`** | 量化 FP→FP8/FP4（生成 exp/scale/max） | `include/pto/common/pto_instr.hpp:1894-1920` |
| **`TDEQUANT`** | 反量化 | 同上 |

每种指令有 3 个变体：基础、ACC（累加）、BIAS（带偏置）。

## 3. 支持的数据类型组合

**`TMATMUL_MX` 的 FP8 组合**（`include/pto/npu/a5/TMatmul.hpp:91-95`）:
- `float8_e4m3_t × float8_e4m3_t`
- `float8_e4m3_t × float8_e5m2_t`（及其对称）
- `float8_e5m2_t × float8_e4m3_t`
- `float8_e5m2_t × float8_e5m2_t`

**`TMATMUL_MX` 的 FP4 组合**（`include/pto/npu/a5/TMatmul.hpp:86-89`）:
- `float4_e1m2x2_t × float4_e1m2x2_t`
- `float4_e1m2x2_t × float4_e2m1x2_t`（及其交叉组合）
- `float4_e2m1x2_t × float4_e2m1x2_t`
- `float4_e2m1x2_t × float4_e1m2x2_t`

累加器类型固定为 `float`（FP32）。缩放 Tile 使用 `float8_e8m0_t`（E8M0 指数格式）。

## 4. 架构支持

| 架构 | TMATMUL_MX | 说明 |
|------|-----------|------|
| **A5**（Ascend 910C/新版） | ✅ 支持 | `include/pto/npu/a5/TMatmul.hpp` — 调用底层 `mad_mx()` |
| **A2A3**（Ascend 910B） | ❌ 不支持 | 仅有标准 `TMATMUL` |
| **Kirin9030** | ✅ 支持 | `include/pto/npu/kirin9030/TMatmul.hpp` |
| **CPU Sim** | ✅ 支持 | `include/pto/cpu/TMatmul.hpp` — golden reference |

## 5. 量化流程（TQUANT 实现）

实现位于 `include/pto/npu/a5/TQuant.hpp`，TQUANT 将高精度数据量化为 MXFP8，共 3 阶段：

```
源数据 (FP32/BF16/FP16)
  ↓ 1. AbsReduceMax: 按 32 元素 block 求绝对最大值
  ↓ 2. ExtractB8ExponentAndScaling: 计算共享指数和缩放因子 (E8M0)
  ↓ 3. CalcQuantizedFP8Values: 乘缩放 + 类型转换 (→FP8 e4m3)
量化输出 + exp Tile + scale Tile
```

### AbsReduceMax 实现细节

提供多种优化路径，根据数据规模自动选择：

- **FP32 路径**:
  - `AbsReduceMax_Naive`: 小规模（≤1024 元素），逐 VL 循环
  - `AbsReduceMax_f32_opt`: 中等规模，DINTLV_B32 + vcgmax
  - `AbsReduceMax_f32_opt_largesizes`: 大规模（2K 对齐），32 VL/outer loop

- **BF16/FP16 路径**:
  - `AbsReduceMax_b16_ND`: 通用 1D 路径，DINTLV_B16 窗口
  - `AbsReduceMax_b16_ND_largesizes`: 大规模 1D（2K 对齐）
  - `AbsReduceMax_b16_ND_2D`: 2D per-row 路径（validCols ≠ srcCols 时）
  - `AbsReduceMax_b16_DintlvWindow`: 单 DINTLV 窗口原子操作

### ExtractB8ExponentAndScaling 实现细节

从 group-max 计算共享指数（E8M0 格式）和缩放因子：

```
共享指数 = float_exp - b8_emax         (如 FP32: exp - 8)
缩放因子 = exp_max_val - 共享指数       (如 FP32: 0xFE - shared_exp)
```

处理 NaN/次正规数特殊情况：
- NaN 检测: 指数全 1 时设置 NaN 标志
- 次正规数: 缩放 < -127 时特殊处理

提供 1D/2D 两种路径：
- 1D 路径: 连续内存，逐 VL 处理
- 2D 路径: per-row 处理，支持 padded buffer（validCols ≠ srcCols）

### CalcQuantizedFP8Values 实现细节

将源数据乘以缩放因子后转换为 FP8 e4m3：

**FP32 → FP8**:
```cpp
vcvt(vb8_out, vb32_out, ROUND_R, RS_ENABLE, PART_P0);
```

**BF16/FP16 → FP8**（需中间 FP32 转换）:
```cpp
vcvt(vb32_cvt, vb16_out, PART_EVEN/ODD);  // b16→fp32
vcvt(vb8_p, vb32_cvt, ROUND_R, RS_ENABLE, PART_P0/P1/P2/P3);  // fp32→fp8
```

关键优化:
- Unroll2 版本: 每次处理 2 个 VL，使用 DINTLV_B32 分离奇偶
- Window 版本: 每次处理 256 元素 DINTLV_B16 窗口
- 2D 版本: per-row 处理，支持 padded layout

量化模式枚举（`npu/a5/TQuant.hpp:23-28`）：
- `QuantType::MXFP8` — MX 格式 FP8
- `QuantType::INT8_SYM` — INT8 对称量化
- `QuantType::INT8_ASYM` — INT8 非对称量化

## 6. 性能 Kernel 示例

### MXFP8 Kernel（`kernels/manual/a5/matmul_mxfp8_performance/`）
- 类型: `float8_e5m2_t` 输入 + `float8_e8m0_t` 缩放 → `bfloat16_t` 输出
- 规模: M=6144, K=6144, N=6144, 32 cores
- Tiling: baseM=128, baseK=128, baseN=256

### MXFP4 Kernel（`kernels/manual/a5/matmul_mxfp4_performance/`）
- 类型: `float4_e2m1x2_t` 输入 + `float8_e8m0_t` 缩放 → `bfloat16_t` 输出
- 规模: M=2040, K=8192, N=8100, 32 cores
- 支持动态尾部（tail handling），无需整除

### 两层流水线架构

```
GM ──TLOAD──→ L1 (Mat Tile, 双缓冲)
               ↓ TEXTRACT
              L0A/L0B (Left/Right Tile, 乒乓缓冲, 32KiB/slot)
               ↓ TMATMUL_MX (PIPE_M, Cube)
              L0C (Accum Tile)
               ↓ TSTORE
              GM (输出)
```

Scale Tile 伴随数据 Tile 一起搬运：
- GM Scale (E8M0) → L1 ScaleMat Tile → TEXTRACT → L0 Scale Tile
- 地址通过 `GetScaleAddr()` 自动关联 L0A/L0B tile 地址

同步信号: `PIPE_MTE2`（GM→L1）、`PIPE_MTE1`（L1→L0）、`PIPE_M`（Cube 计算），三者通过 `set_flag`/`wait_flag` 驱动三级流水线。

## 7. CPU 仿真器 Golden Reference

实现位于 `include/pto/cpu/TMatmul.hpp:44-67`：

```cpp
// 核心数学: C[i][j] = Σ(A[i][k] * B[k][j] * scaleA[i][k/32] * scaleB[k/32][j])
for (k = 0; k < K; k++) {
    double scaleFactor = scale0[i][k/32] * scale1[k/32][j];  // block=32
    mul_acc += src0[i][k] * src1[k][j] * scaleFactor;
}
```

每 32 个元素共享一个缩放因子（block size = 32），遵循 OCP MX 规范。MXFP 类型定义在 `include/pto/cpu/MXTypes.hpp`。

## 8. 关键约束

- K 维度必须是 64 的倍数（`BASEK = 64`）
- FP4 类型 K 维度额外要求偶数
- 分形布局: Left=ColMajor+RowFractal, Right=RowMajor+ColFractal, Acc=ColMajor+RowFractal
- 运行时 M/K/N 范围 `[1, 4095]`
- Bias 必须是 `float`，单行 RowMajor
- 量化时需 zero-padding 源 tile 的 pad columns 以保证 max 计算正确
- 内存对齐要求:
  - NORM 操作: 32B 对齐
  - E2B_B16 加载: 16B 对齐
  - DINTLV_B16: 源地址需满足 VL 对齐

## 9. 关键文件索引

### 公共 API
- `include/pto/common/pto_instr.hpp` — 所有 PTO 指令的公共声明
- `include/pto/common/event.hpp` — Pipe/event 枚举定义
- `include/pto/cpu/MXTypes.hpp` — MXFP 类型定义

### NPU 实现
- `include/pto/npu/a5/TMatmul.hpp` — A5 TMATMUL + TMATMUL_MX 实现
- `include/pto/npu/a5/TQuant.hpp` — A5 TQUANT 实现（量化）
- `include/pto/npu/a5/TDeQuant.hpp` — A5 TDEQUANT 实现
- `include/pto/npu/a2a3/TMatmul.hpp` — A2A3 TMATMUL 实现
- `include/pto/npu/kirin9030/TMatmul.hpp` — Kirin9030 TMATMUL 实现

### CPU 实现
- `include/pto/cpu/TMatmul.hpp` — CPU TMATMUL + TMatmulMX golden reference
- `include/pto/cpu/TQuant.hpp` — CPU TQUANT golden reference
- `include/pto/cpu/TDeQuant.hpp` — CPU TDEQUANT golden reference

### 文档
- `docs/isa/TMATMUL_MX_zh.md` — TMATMUL_MX 指令文档
- `docs/isa/TQUANT_zh.md` — TQUANT 指令文档
- `docs/isa/TDEQUANT_zh.md` — TDEQUANT 指令文档
- `docs/coding/tutorials/gemm_zh.md` — GEMM 教程

### 性能 Kernel
- `kernels/manual/a5/matmul_mxfp8_performance/` — MXFP8 性能示例
- `kernels/manual/a5/matmul_mxfp4_performance/` — MXFP4 性能示例

### 测试用例
- `tests/cpu/st/testcase/tmatmul_mx/` — CPU TMATMUL_MX 测试
- `tests/cpu/st/testcase/tquant/` — CPU TQUANT 测试
- `tests/npu/a5/src/st/testcase/tmatmul_mx/` — NPU A5 TMATMUL_MX 测试
- `tests/npu/a5/src/st/testcase/tquant/` — NPU A5 TQUANT 测试

## 10. 使用示例

### 基础 TMATMUL_MX 用法

```cpp
#include <pto/pto-inst.hpp>
using namespace pto;

void example() {
  using A = TileLeft<float8_e5m2_t, 16, 64>;
  using B = TileRight<float8_e5m2_t, 64, 32>;
  using ScaleA = TileLeftScale<float8_e8m0_t, 16, 2>;
  using ScaleB = TileRightScale<float8_e8m0_t, 2, 32>;
  using C = TileAcc<float, 16, 32>;
  
  A a;
  B b;
  ScaleA scaleA;
  ScaleB scaleB;
  C c;
  
  // 基础矩阵乘
  TMATMUL_MX(c, a, scaleA, b, scaleB);
  
  // 累加模式
  TMATMUL_MX(c, c, a, scaleA, b, scaleB);
  
  // 带 bias
  using Bias = Tile<TileType::Bias, float, 1, 32>;
  Bias bias;
  TMATMUL_MX(c, a, scaleA, b, scaleB, bias);
}
```

### TQUANT 量化用法

```cpp
#include <pto/pto-inst.hpp>
using namespace pto;

void quantize_example() {
  using SrcTile = Tile<TileType::Vec, float, 16, 128>;
  using DstTile = Tile<TileType::Vec, float8_e4m3_t, 16, 128>;
  using ExpTile = Tile<TileType::Vec, uint8_t, 1, 64>;
  using MaxTile = Tile<TileType::Vec, float, 1, 64>;
  using ScaleTile = Tile<TileType::Vec, float, 1, 64>;
  
  SrcTile src;
  DstTile dst;
  ExpTile exp;
  MaxTile max;
  ScaleTile scale;
  
  // 量化 FP32 → MXFP8
  TQUANT<QuantType::MXFP8>(dst, src, &exp, &max, &scale);
}
```

## 11. 调试与验证

### CPU 仿真器验证

```bash
# 运行所有 CPU 测试
python3 tests/run_cpu.py --clean --verbose

# 运行 TMATMUL_MX 测试
python3 tests/run_cpu.py --testcase tmatmul_mx

# 运行 TQUANT 测试
python3 tests/run_cpu.py --testcase tquant
```

### NPU 硬件验证

```bash
# 构建 NPU 测试
python3 tests/script/build_st.py -r npu -v a5 -t all

# 运行单个测试
python3 tests/script/run_st.py -r npu -v a5 -t tmatmul_mx
```

### 调试技巧

1. 使用 `TSTORE` 将中间 tile 写回 GM 检查
2. 对比 CPU 仿真器和 NPU 硬件结果
3. 检查量化后的 exp/scale 值是否符合预期
4. 验证 zero-padding 是否正确处理边界

## 12. 相关资源

- [PTO-ISA 主页](https://gitee.com/ascend/pto-isa)
- [OCP MX 规范](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf)
- [Ascend C 编程指南](https://www.hiascend.com/document/detail/zh/canncommercial/700/apiref/ascendcappdevguide/ascendcappdevguide_0001.html)
