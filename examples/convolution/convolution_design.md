# conv2d (im2col + GEMM) 算子设计文档

## 1. 概述

### 1.1 算子名称

`conv2d` (2D 卷积，im2col + GEMM 实现)

### 1.2 功能描述

将 2D 卷积操作转换为 im2col 图像展平 + 矩阵乘法(GEMM)，在华为 Ascend NPU 上实现高性能二维卷积。

### 1.3 数学公式

标准 2D 卷积公式：

$$
\text{Output}(n, oc, i, j) = \sum_{c=0}^{C-1} \sum_{m=0}^{KH-1} \sum_{k=0}^{KW-1} \text{Input}(n, c, i \cdot s + m - p, j \cdot s + k - p) \cdot \text{Kernel}(oc, c, m, k)
$$

其中：
- **Input**: `(B, C, H, W)` — 批大小、输入通道数、高、宽
- **Kernel**: `(OC, C, KH, KW)` — 输出通道数、输入通道数、卷积核高、宽
- **Output**: `(B, OC, HO, WO)` — 输出特征图
- **stride** (`s`): 步长
- **padding** (`p`): 填充

$$
HO = \left\lfloor \frac{H + 2p - KH}{s} \right\rfloor + 1, \quad WO = \left\lfloor \frac{W + 2p - KW}{s} \right\rfloor + 1
$$

### 1.4 算法描述

本算子采用 **im2col + GEMM** 两步分解策略：

**步骤 1 — im2col（图像展平）**：
将 4D 输入 `(B, C, H, W)` 按滑动窗口展开为 2D 矩阵 `(C·KH·KW, B·HO·WO)`。
每个输出位置 `(n, i, j)` 对应一个长度为 `C·KH·KW` 的列向量（感受野展平）。

**步骤 2 — GEMM（矩阵乘法）**：
将 Kernel 重塑为 `(OC, C·KH·KW)`，与 im2col 结果 `(C·KH·KW, B·HO·WO)` 做矩阵乘法，
得到 `(OC, B·HO·WO)`，再 reshape + permute 为 `(B, OC, HO, WO)`。

**步骤 3 — 动态 Padding（非整除处理）**：
当 GEMM 维度 `M=OC, N=B·HO·WO, K=C·KH·KW` 不是 block 大小(128)的整数倍时，
在主机侧对矩阵 zero-padding 到下一个 128 的倍数，GEMM 完成后再裁剪回原始尺寸。
零填充区域在矩阵乘法中产生零贡献，不影响正确结果。

### 1.5 数据流图

```
Input (B,C,H,W)                Kernel (OC,C,KH,KW)
      │                               │
      ▼ im2col (host torch)           ▼ view (OC, C*KH*KW)
Input_flat (C*KH*KW, B*HO*WO)    Kernel_flat (OC, C*KH*KW)
      │                               │
      │◄──── 动态 Padding ────►       │
      ▼                               ▼
Input_pad (K_pad, N_pad)        Kernel_pad (M_pad, K_pad)
      │                               │
      └───────────┬───────────────────┘
                  ▼
         GEMM (NPU TileLang)
         C = K_pad @ I_pad
         M = OC, N = B*HO*WO, K = C*KH*KW
                  │
                  ▼
         Output_pad (M_pad, N_pad)
                  │
                  ▼ 裁剪 [:M, :N]
         Output (OC, B*HO*WO)
                  │
                  ▼ view + permute
         Output (B, OC, HO, WO)
```

---

## 2. 编程模式选型

### 2.1 模式结论

**选定模式**: Developer（自动化模式）

### 2.2 选型理由

| 算子特征 | 分析 | 结论 |
|---------|------|------|
| 计算类型 | 核心为 GEMM（矩阵乘法），T.gemm_v0 原语 | 编译器可自动管理 |
| 是否含 matmul | 是，GEMM 是主要计算 | Developer 模式对 GEMM 支持完善 |
| 是否含归约 | 否（K 维的累加由 GEMM 内部完成） | 无需手动 reduce |
| 是否需要流水线 | 否 | 标准 GEMM 无需 Pipelined |
| 内存分配 | GEMM 的 shared/fragment 层级遵循固定模式 | T.alloc_shared / T.alloc_fragment 编译器自动映射 |

### 2.3 模式影响

| 维度 | 本算子的选择 |
|------|-------------|
| 内存分配 | `T.alloc_shared` 编译器自动判断 L1 或 UB；`T.alloc_fragment` 编译器自动判断 L0A/L0B/L0C |
| 计算方式 | `T.gemm_v0(A, B, C, init=...)` 块级矩阵乘 |
| 作用域 | 编译器自动分离 Cube / Vector 作用域 |
| 同步方式 | pass_configs 开启 `TL_ASCEND_AUTO_SYNC` 自动插入同步 |

---

## 3. API 映射设计

### 3.1 公式拆解

| 步骤 | 数学表达 | 说明 |
|------|----------|------|
| 1 | `Input_flat = im2col(Input)` | 将 (B,C,H,W)→(C·KH·KW, B·HO·WO) |
| 2 | `Kernel_flat = Kernel.view(OC, -1)` | 将 (OC,C,KH,KW)→(OC, C·KH·KW) |
| 3 | `Output = Kernel_flat @ Input_flat` | 矩阵乘法 C(M,N) = A(M,K) × B(K,N) |
| 4 | `Output = Output.view(OC,B,HO,WO).permute(1,0,2,3)` | 恢复 4D 卷积输出形状 |

### 3.2 TileLang API 映射（GEMM 核心）

| 步骤 | 数学表达 | TileLang API | 参数 | 模式 |
|------|----------|-------------|------|------|
| 数据搬入 A | A[block] → L1 | `T.copy(A[offset], A_L1)` | implicitly: GM→L1 | Developer |
| 数据搬入 B | B[block] → L1 | `T.copy(B[offset], B_L1)` | implicitly: GM→L1 | Developer |
| 矩阵乘 | C_L0 += A_L1 × B_L1 | `T.gemm_v0(A_L1, B_L1, C_L0, init=(k==0))` | init 标识首次迭代清零 | Developer |
| 数据搬出 | C_L0 → C[block] | `T.copy(C_L0, C[offset])` | implicitly: L0C→GM | Developer |

### 3.3 计算伪代码（GEMM Kernel）

```python
@T.prim_func
def main(A: T.Tensor((M, K), "float16"),
         B: T.Tensor((K, N), "float16"),
         C: T.Tensor((M, N), "float16")):
    with T.Kernel(m_num * n_num, is_npu=True) as (cid, _):
        bx = cid // n_num        # M 方向的 block 索引
        by = cid % n_num         # N 方向的 block 索引

        # 1. 分配片上缓冲区
        A_L1 = T.alloc_shared((block_M, block_K), "float16")      # L1/UB
        B_L1 = T.alloc_shared((block_K, block_N), "float16")      # L1/UB
        C_L0 = T.alloc_fragment((block_M, block_N), "float")       # L0C (accum)

        # 2. K 方向分块迭代累加
        loop_k = T.ceildiv(K, block_K)
        for k in T.serial(loop_k):
            # GM → L1 双缓冲搬入
            T.copy(A[bx * block_M, k * block_K], A_L1)
            T.copy(B[k * block_K, by * block_N], B_L1)

            # L1 × L1 → L0C（矩阵乘累加）
            T.gemm_v0(A_L1, B_L1, C_L0, init=(k == 0))

        # 3. L0C → GM 搬出结果
        T.copy(C_L0, C[bx * block_M, by * block_N])
```

### 3.4 API 可行性确认

| API | 来源 | 状态 |
|-----|------|------|
| `T.alloc_shared` | api-kernel-memory.md §2.1 | ✅ 已验证 |
| `T.alloc_fragment` | api-kernel-memory.md §2.2 | ✅ 已验证 |
| `T.copy` | api-kernel-memory.md §3 | ✅ 已验证 |
| `T.gemm_v0` | api-compute.md §1 | ✅ 已验证，examples/developer_mode/gemm_developer.py |
| `T.ceildiv` | TileLang DSL 内置 | ✅ 已验证 |
| `T.serial` | api-schedule-sync.md §1 | ✅ 已验证 |
| `T.Kernel(..., is_npu=True)` | api-kernel-memory.md §1.3 | ✅ 已验证 |
| `@tilelang.jit(out_idx=[-1], ...)` | api-kernel-memory.md §1.4 | ✅ 已验证 |

---

## 3.5 技术约束确认

### 3.5.1 本项目已知限制检查

| 约束 | 本算子是否涉及 | 处理方案 |
|------|---------------|----------|
| 不支持三维 Kernel | **No** | 使用一维 T.Kernel(m_num * n_num)，通过 bx/by 线性化 2D block grid |
| threads 参数限制（仅 1 或 2） | **No** | 不使用 threads 参数，使用默认值 |
| 动态循环边界不支持 | **No** | block_M/N/K 均为编译期常量(128)，loop_k 由 T.ceildiv 处理 |
| 流水线不支持动态边界 | **No** | 未使用 T.Pipelined |

### 3.5.2 参考实现差异说明

本算子无外部 GPU 参考实现，为标准 im2col+GEMM 算法在 Ascend 上的移植实现。

### 3.5.3 本项目同类实现参考

| 文件路径 | 相似度 | 关键参考点 |
|----------|--------|-----------|
| `examples/developer_mode/gemm_developer.py` | 高度相似 | GEMM Kernel 结构、T.gemm_v0 用法、T.serial K 循环 |
| `examples/convolution/example_convolution_autotune.py` | 同一算子 | autotune 变体，block 搜索空间 [64,128] |
| `examples/developer_mode/matmul_add_developer.py` | 相似 | T.copy + T.gemm_v0 组合模式 |

---

## 4. 数据规格与内存规划

### 4.1 输入张量（GEMM Kernel）

| 参数名 | Shape | dtype | 说明 |
|--------|-------|-------|------|
| A (kernel) | `(M, K)` = `(OC, C·KH·KW)` | float16 | 卷积核展平矩阵 |
| B (input) | `(K, N)` = `(C·KH·KW, B·HO·WO)` | float16 | im2col 展平特征图 |

### 4.2 输出张量

| 参数名 | Shape | dtype | 说明 |
|--------|-------|-------|------|
| C (output) | `(M, N)` = `(OC, B·HO·WO)` | float16 | GEMM 结果，后续 reshape 为 (B,OC,HO,WO) |

### 4.3 中间缓冲区（片上）

| Buffer 名 | Shape | dtype | 存储层级 | 大小 (Bytes) | 用途 |
|-----------|-------|-------|----------|-------------|------|
| A_L1 | `(128, 128)` | float16 | L1/UB (自动) | 32768 | 矩阵 A 的 K 方向分块缓冲 |
| B_L1 | `(128, 128)` | float16 | L1/UB (自动) | 32768 | 矩阵 B 的 K 方向分块缓冲 |
| C_L0 | `(128, 128)` | float32 | L0C (自动) | 65536 | 矩阵乘累加结果（accum 精度） |

### 4.4 内存搬运路径

```
GM[Kernel] ──T.copy──► L1/UB[A_L1] ──T.gemm_v0──► L0C[C_L0]
GM[Input]  ──T.copy──► L1/UB[B_L1] ────────┘         │
                                                      │ T.copy
                                                      ▼
                                                   GM[C_output]
```

**详细搬运流程**：
1. K 循环每一轮：从 GM 搬入 `block_M × block_K` 的 Kernel 分块 → L1/UB 的 A_L1
2. 同时：从 GM 搬入 `block_K × block_N` 的 Input 分块 → L1/UB 的 B_L1
3. `T.gemm_v0`：A_L1 (L1/UB) × B_L1 (L1/UB) → C_L0 (L0C)，累加
4. K 循环结束后：C_L0 搬回 GM 的对应输出分块

### 4.5 UB 内存预算

| Buffer | Shape | dtype | 大小 (Bytes) |
|--------|-------|-------|-------------|
| A_L1 | `(128, 128)` | float16 | 32,768 |
| B_L1 | `(128, 128)` | float16 | 32,768 |
| **A + B 总计** | | | **65,536 (64 KB)** |

> 注：A2/A3 设备 UB 容量 192KB，64KB 用量仅占 1/3，留有充足余量。
> C_L0 分配在 L0C（独立于 UB），不占用 UB 空间。

### 4.6 L0 内存预算

| Buffer | Shape | dtype | 大小 (Bytes) |
|--------|-------|-------|-------------|
| C_L0 | `(128, 128)` | float32 | 65,536 (64 KB) |

> 注：A2/A3 设备 L0C 容量为 64KB，`block_M=128, block_N=128` 恰好填满 L0C。
> 若 block 尺寸超过 128（如 256），L0C 将溢出，本算子搜索空间已限制 block ≤ 128。

### 4.7 动态轴定义

无。M, N, K 为编译期可变（通过 `@tilelang.jit` 参数传入），但 block_M/N/K 为常量 128。

### 4.8 JIT 配置

```python
@tilelang.jit(
    out_idx=[-1],      # 最后一个参数 C 为输出
    pass_configs={
        tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,    # 自动 Cube/Vector 分离
        tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,          # 自动同步插入
        tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,    # 自动内存规划
    },
)
```

---

## 5. Tiling 策略

### 5.1 计算类型

**类型**: 纯 Cube（GEMM 在 Cube 核上执行，数据搬运用 MTE）

**判定依据**: 算子核心为矩阵乘法 `T.gemm_v0`，该原语在 Cube 计算单元上执行。im2col 步骤在主机 CPU/NPU 上用 PyTorch 完成，不占用 TileLang 计算资源。

### 5.2 Block 划分

```python
block_M = 128   # M 方向分块，匹配 L0C 容量：128×128×4B = 64KB = L0C 容量
block_N = 128   # N 方向分块，匹配 L0C 容量
block_K = 128   # K 方向分块，匹配 L0A/L0B 容量：128×128×2B = 32KB（L0A/L0B 各 64KB）

m_num = M // block_M          # M 方向 block 数
n_num = N // block_N          # N 方向 block 数
block_num = m_num * n_num     # 总 block 数 = Cube 核数
loop_k = T.ceildiv(K, block_K)  # K 方向迭代次数
```

| 参数 | 取值 | 选择理由 |
|------|------|---------|
| block_M | 128 | 平衡计算效率与 L0C 容量（128×128×4B=64KB） |
| block_N | 128 | 同上 |
| block_K | 128 | 平衡 L1 带宽利用与 L0A/L0B 容量（128×128×2B=32KB） |

### 5.3 约束分析

- **对齐约束**: block 大小均为 16 的倍数，满足 Ascend 硬件对齐要求（Matrix Unit 基本块为 16×16） ✓
- **UB 容量**: A_L1 + B_L1 = 64KB < 192KB (A2/A3 UB) ✓
- **L0C 容量**: C_L0 = 64KB = L0C 容量，恰好利用完整 L0C ✓
- **L0A/L0B 容量**: 128×128×2B = 32KB < 64KB ✓
- **Block 数**: m_num × n_num ≥ 1（当 M,N 均 ≥ 128 时）✓

### 5.4 注意事项

- **非整除处理**: GEMM Kernel 内部使用 `M // block_M` 和 `N // block_N`，要求 M、N 必须是 block 大小的整数倍。本实现在调用 GEMM 前将 M、N、K 都 zero-padding 到 128 的倍数，GEMM 完成后再裁剪回原始尺寸。详见第 6.4 节。
- **Padding 实现**: padding 在主机侧用 PyTorch 的 `torch.zeros` 完成，无需修改 GEMM Kernel。
- **Autotune 变体**: `example_convolution_autotune.py` 中 block 搜索空间为 `block_M, block_N ∈ [64, 128]`、`K_L1 ∈ [64, 128]`，过滤了 `block > dimension` 的无效配置，并排除了 block=256（避免 L0C 溢出）。

---

## 6. 循环与调度结构

### 6.1 循环结构总结

| 维度 | 循环类型 | API | 理由 |
|------|----------|-----|------|
| M 方向 | Block 级并行 | `T.Kernel(m_num * n_num)` | 每个 block 处理一个 (M, N) 分块 |
| N 方向 | Block 级并行 | 同 M 方向（线性 block ID → 2D 坐标） | — |
| K 方向 | 串行迭代 | `T.serial(T.ceildiv(K, block_K))` | K 维分块迭代累加 |
| Block 内元素 | Cube 指令并行 | `T.gemm_v0` 内部 | Cube 计算单元内置并行 |

### 6.2 循环伪代码

```python
# Block 级并行（隐式，由 T.Kernel 管理）
with T.Kernel(m_num * n_num, is_npu=True) as (cid, _):
    bx = cid // n_num           # M 方向 block 索引: [0, m_num)
    by = cid % n_num            # N 方向 block 索引: [0, n_num)

    # K 方向串行迭代累积
    for k in T.serial(T.ceildiv(K, block_K)):
        # 第 k 轮：搬入 A[block_M×block_K]、B[block_K×block_N] 分块
        T.copy(A[bx * block_M, k * block_K], A_L1)
        T.copy(B[k * block_K, by * block_N], B_L1)

        # GEMM: C_L0 += A_L1 × B_L1（k==0 时清零 C_L0）
        T.gemm_v0(A_L1, B_L1, C_L0, init=(k == 0))
```

### 6.3 流水线优化

未使用 `T.Pipelined`。本算子采用标准 K 方向串行迭代，
T.copy 和 T.gemm_v0 的顺序执行天然形成软件流水线：
copy 下一块 → gemm 计算当前块（隐式重叠）。

### 6.4 尾块处理

**策略**: 主机侧 zero-padding，GEMM Kernel 不做尾块特殊处理。

```python
# 在 conv_im2col_gemm 中（主机侧）：
M_pad = ((M + block_M - 1) // block_M) * block_M    # 向上舍入到 128 倍数
N_pad = ((N + block_N - 1) // block_N) * block_N
K_pad = ((K + block_K - 1) // block_K) * block_K

# 对短边 zero-padding
if need_pad:
    kernel_padded[:M, :K] = kernel_flat       # 其余为 0
    input_padded[:K, :N] = input_flat         # 其余为 0
    func = matmul(M_pad, N_pad, K_pad, ...)

# GEMM 后裁剪
output = output[:M, :N]
```

**正确性保证**:
- 零填充的 K 维元素：`0 × kernel_elem = 0`，对 GEMM 结果无贡献
- 零填充的 M/N 维：GEMM 会计算，但输出被裁剪丢弃

---

## 7. 同步策略

### 7.1 同步模式

**模式**: 自动同步（Developer 模式）

### 7.2 同步点说明

| 位置 | 同步方式 | 理由 |
|------|----------|------|
| K 循环内 T.copy → T.gemm_v0 | pass_configs 自动插入 | 确保数据搬入完成后才启动 GEMM |
| T.gemm_v0 → 下一轮 T.copy | pass_configs 自动插入 | 确保 GEMM 完成后才覆盖 L1 buffer |
| K 循环结束后 T.copy(C_L0→GM) | pass_configs 自动插入 | 确保最终 GEMM 结果写入 GM |

### 7.3 pass_configs 配置

```python
pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,   # 自动 Cube/Vector 核分离
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,         # 自动同步（T.copy ↔ T.gemm_v0）
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,   # 自动内存复用规划
}
```

> 注：本算子仅为纯 GEMM（Cube 核计算），无 Vector 核后处理，但 `TL_ASCEND_AUTO_CV_COMBINE` 开启后编译器仍会尝试 CV 分离优化。

---

## 8. 融合算子设计

本算子 **不是融合算子**（纯 GEMM，无 element-wise 后处理）。

---

## 9. 验证方案

### 9.1 Golden 函数

```python
import torch.nn.functional as F

def golden_conv2d(input_tensor, kernel, stride=1, padding=0):
    """基于 PyTorch Conv2d 的参考实现"""
    return F.conv2d(input_tensor.cpu(), kernel.cpu(),
                     stride=stride, padding=padding).npu()
```

### 9.2 测试用例

| 用例名 | 级别 | B | C | H | W | OC | KH | KW | stride | padding | M | N | K | 覆盖点 |
|--------|------|---|---|---|---|----|----|----|--------|---------|---|---|---|--------|
| Case 1: 完美对齐 | L0 | 2 | 2 | 15 | 15 | 128 | 8 | 8 | 1 | 0 | 128 | 128 | 128 | M=N=K=128，零 padding |
| Case 2: M padding | L1 | 1 | 2 | 32 | 32 | 50 | 3 | 3 | 1 | 0 | 50→128 | 900→1024 | 18→128 | OC < 128，M 需 padding |
| Case 3: N padding | L1 | 1 | 4 | 17 | 17 | 128 | 3 | 3 | 1 | 0 | 128 | 225→256 | 36→128 | 非整除空间输出，N 需 padding |
| Case 4: K padding | L1 | 2 | 3 | 28 | 28 | 128 | 3 | 3 | 2 | 1 | 128 | 392→512 | 27→128 | 小通道+stride+padding，K 需 padding |
| Case 5: 全 padding | L2 | 1 | 3 | 17 | 17 | 64 | 3 | 3 | 1 | 0 | 64→128 | 225→256 | 27→128 | M/N/K 同时需要 padding |
| Case 6: 多 block | L3 | 4 | 8 | 28 | 28 | 256 | 5 | 5 | 1 | 0 | 256 | 2304 | 200→256 | 大尺寸，M/N 方向多 block |

### 9.3 精度标准

| dtype | atol | rtol |
|-------|------|------|
| float16 | 1e-2 | 1e-2 |

> 精度验证使用 `torch.testing.assert_close(result, ref, rtol=1e-2, atol=1e-2)`。

---

## 10. 风险点与注意事项

### 10.1 已知约束

| 约束 | 说明 |
|------|------|
| GEMM 只支持 float16 输入 | `dtype="float16"` 硬编码，accum 为 float32 |
| Block 大小固定 | block_M=128, block_N=128, block_K=128，不可运行时调整（autotune 版本可搜索） |
| M, N 须为 block 倍数 | Kernel 内部 `M // block_M` 整除依赖。通过主机侧 padding 规避 |
| 无 Bias 支持 | 当前实现不含 conv bias 项 |
| 无 Dilation 支持 | 仅支持 dilation=1 |
| 无 Stride 非对称支持 | 实现中 stride 统一应用于 H 和 W 方向 |

### 10.2 常见错误

| 错误 | 触发场景 | 影响 | 解决方案 |
|------|----------|------|----------|
| M // block_M = 0 | OC < 128 | 无 block 启动，输出全零 | 主机侧 M_pad padding 到 128 |
| N // block_N = 0 | B×HO×WO < 128 | 同上 | 主机侧 N_pad padding |
| Divide by zero (bx=cid//n_num) | n_num = 0 | 编译崩溃 | 主机侧确保 N ≥ block_N |
| L0C 溢出（Segfault） | block_M=256, block_N=256 | 运行时 crash | autotune 限制 block ≤ 128 |
| im2col 内存爆炸 | B,C,H,W,KH,KW 大 | 主机侧 OOM | im2col 结果 (C×KH×KW, B×HO×WO) 可能很大，需分段处理 |

### 10.3 特殊场景处理

| 场景 | 处理策略 |
|------|---------|
| OC 不能被 128 整除 | M_pad = ceil_div(OC, 128) × 128，GEMM 后裁剪 |
| B×HO×WO 不能被 128 整除 | N_pad = ceil_div(..., 128) × 128，GEMM 后裁剪 |
| C×KH×KW 不能被 128 整除 | K_pad = ceil_div(..., 128) × 128，GEMM 后裁剪 |
| 极小输入（M/N/K < 128） | padding 后将只有 1 个 block，正确执行 |
| 极大输入（M,N 数倍于 128） | 正常多 block 并行，Padding 开销占比可忽略 |
| padding=0, KH/KW > 1 时的边界 | im2col 中越界位置填零（zero-padding），符合标准卷积语义 |

---

## 11. 交付清单

### 11.1 目录结构

```
examples/convolution/
├── example_convolution.py            # 算子实现 + 6 组测试
├── example_convolution_autotune.py   # Autotune 变体
├── autotuner.log                     # Autotune 日志
└── convolution_design.md             # 本设计文档
```

### 11.2 文件清单

| 文件 | 状态 | 说明 |
|------|------|------|
| `convolution_design.md` | ✅ 已完成 | 设计文档 |
| `example_convolution.py` | ✅ 已实现 | 固定 block 的实现 + 6 组 Level 0~3 测试 |
| `example_convolution_autotune.py` | ✅ 已实现 | Autotune 变体，block 搜索空间 [64,128] |

### 11.3 命名规范

- 目录名: `convolution`
- 实现文件: `example_convolution.py`
- Autotune 文件: `example_convolution_autotune.py`
- 设计文档: `convolution_design.md`

### 11.4 实现顺序

1. ✅ 设计文档（convolution_design.md）
2. ✅ 算子实现（example_convolution.py）
3. ✅ 基础测试（Case 1: 完美对齐，Level 0）
4. ✅ 典型测试（Case 2-4: 单维 padding，Level 1）
5. ✅ 边界测试（Case 5: 全 padding，Level 2）
6. ✅ 性能测试（Case 6: 多 block 大尺寸，Level 3）
7. ✅ Autotune 变体（example_convolution_autotune.py）
