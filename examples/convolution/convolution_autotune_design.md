# Convolution Autotune 算子设计文档

## 1. 概述

### 1.1 算子名称

`convolution_autotune` — 基于 im2col + GEMM 的 2D 卷积，集成 block size 自动调优。

### 1.2 功能描述

将 2D 卷积计算分解为 CPU/PyTorch 端 im2col 变换 + NPU 端块级 GEMM 矩阵乘法，通过 `@tilelang.autotune` 自动探索最优 block 分块配置（block_M、block_N、K_L1），支持任意 stride 和 padding 的 2D 卷积。

### 1.3 数学公式

标准 2D 卷积：

$$
\text{Output}(n, oc, h_o, w_o) = \sum_{c=0}^{C-1} \sum_{kh=0}^{KH-1} \sum_{kw=0}^{KW-1} \text{Input}(n, c, h_o \cdot S + kh - P, w_o \cdot S + kw - P) \cdot \text{Kernel}(oc, c, kh, kw)
$$

其中：
- `(B, C, H, W)` = 输入 shape
- `(OC, C, KH, KW)` = 卷积核 shape
- `S` = stride，`P` = padding
- `HO = ⌊(H + 2P - KH) / S⌋ + 1`，`WO = ⌊(W + 2P - KW) / S⌋ + 1`

im2col 变换后转换为矩阵乘法：

$$
\text{Input\_flat} \in \mathbb{R}^{(C \cdot KH \cdot KW) \times (B \cdot HO \cdot WO)}
$$

$$
\text{Kernel\_flat} \in \mathbb{R}^{OC \times (C \cdot KH \cdot KW)}
$$

$$
\text{Output\_flat} = \text{Kernel\_flat} \cdot \text{Input\_flat}
$$

$$
\text{Output} \in \mathbb{R}^{B \times OC \times HO \times WO}
$$

### 1.4 算法描述

本算子为两步算法：

1. **im2col 变换**（PyTorch CPU/NPU 端）：
   - 将输入 `(B, C, H, W)` 按卷积窗口展开为 `(C·KH·KW, B·HO·WO)` 矩阵
   - 将卷积核 `(OC, C, KH, KW)` 展开为 `(OC, C·KH·KW)` 矩阵
   - 若 M/N/K 不能被 block size 整除，进行零填充对齐

2. **GEMM 矩阵乘法**（TileLang NPU 端）：
   - 在 Ascend NPU 上执行 `C = A @ B`，其中 `A = Kernel_flat`，`B = Input_flat`
   - 使用 block 级分块 + K 维顺序迭代累加
   - 结果 reshape + permute 回 `(B, OC, HO, WO)`

### 1.5 数据流图

```
Input(B,C,H,W)                    Kernel(OC,C,KH,KW)
     │                                  │
     │ [PyTorch im2col]                 │ [PyTorch view]
     ▼                                  ▼
Input_flat(C·KH·KW, B·HO·WO)      Kernel_flat(OC, C·KH·KW)
     │                                  │
     │ [Pytorch pad to               │ [Pytorch pad to
     │  align block size]             │  align block size]
     ▼                                  ▼
Input_flat_pad(K_pad, N_pad)       Kernel_flat_pad(M_pad, K_pad)
     │                                  │
     │                                  │
     └──────────┬───────────────────────┘
                │
                │ [T.copy: GM → L1(A_L1, B_L1)]
                ▼
           ┌──────────────────────┐
           │   T.gemm_v0            │  ◄── K 维 T.serial 迭代累加
           │   L1 @ L1 → L0C        │      每次加载 block_M×K_L1 和 K_L1×block_N
           └──────────┬───────────┘
                      │
                      │ [T.copy: L0C → GM]
                      ▼
              Output_flat_pad(M_pad, N_pad)
                      │
                      │ [PyTorch slice to original size + reshape]
                      ▼
              Output(B, OC, HO, WO)
```

---

## 2. 编程模式选型

### 2.1 模式结论

**选定模式**: Developer（自动化）

### 2.2 选型理由

| 考量维度 | 分析 | 结论 |
|---------|------|------|
| 计算类型 | 纯 GEMM 矩阵乘 | Developer 模式 `T.gemm_v0` 完全覆盖 |
| 是否含 matmul | 是，核心计算即为标准 GEMM | Developer 模式提供 `T.gemm_v0` 直接支持 |
| 是否含归约 | 是，GEMM 内含 K 维归约累加 | `T.gemm_v0` 已封装归约语义 |
| 是否需要手动流水线 | 否，仅需 K 维顺序迭代 | `T.serial` 足够 |
| 是否需要核间通信 | 否，各 block 独立计算不重叠的输出区域 | 无需手动同步 |
| 是否需要特殊硬件原语 | 否，标准 GEMM 即可 | 无需 `T.mma` 等底层原语 |
| GEMM 后是否需要 element-wise 后处理 | 否，直接输出到 GM | 无融合需求，非融合算子 |

### 2.3 模式影响

| 维度 | 本算子的选择 |
|------|-------------|
| 内存分配 | `T.alloc_shared` 分配 L1（自动映射），`T.alloc_fragment` 分配 L0C（自动映射） |
| 计算方式 | `T.gemm_v0` 标准块级矩阵乘 + `T.serial` K 维迭代 |
| 作用域 | 编译器自动处理（无显式 `T.Scope`），仅 Vector 核参与（纯 Cube 计算由编译器自动调度） |
| 同步方式 | 通过 `TL_ASCEND_AUTO_SYNC: True` 自动同步，无需手动 `T.barrier_all` |

---

## 3. API 映射设计

### 3.1 公式拆解

im2col 变换在 PyTorch 端完成，NPU 端仅涉及 GEMM。NPU GEMM 的计算拆解如下：

| 步骤 | 数学表达 | 说明 |
|------|----------|------|
| 1 | `block_num = (M // block_M) × (N // block_N)` | 将 M×N 输出空间划分为 m_num × n_num 个 block |
| 2 | `A_tile = A[bx·block_M : (bx+1)·block_M, k·K_L1 : (k+1)·K_L1]` | 从 GM 加载 A 的一个 tile 到 L1 |
| 3 | `B_tile = B[k·K_L1 : (k+1)·K_L1, by·block_N : (by+1)·block_N]` | 从 GM 加载 B 的一个 tile 到 L1 |
| 4 | `C_L0 += A_L1 · B_L1` | K 维分块累加，accum_dtype=float32 |
| 5 | `C[bx·block_M, by·block_N] = C_L0` | 将 L0C 结果搬回 GM 输出 |

### 3.2 TileLang API 映射

| 步骤 | 数学表达 | TileLang API | 参数 | 模式 |
|------|----------|-------------|------|------|
| 1 | block 划分 | `T.Kernel(m_num * n_num, is_npu=True) as (cid, _)` | block_num, is_npu | Developer |
| 2 | L1 buffer 分配 | `T.alloc_shared((block_M, K_L1), dtype)` | shape, dtype | Developer |
| 3 | L0C fragment 分配 | `T.alloc_fragment((block_M, block_N), accum_dtype)` | shape, accum_dtype | Developer |
| 4 | GM→L1 搬运 A tile | `T.copy(A[bx * block_M, k * K_L1], A_L1)` | src(DDR slice), dst(L1) | Developer |
| 5 | GM→L1 搬运 B tile | `T.copy(B[k * K_L1, by * block_N], B_L1)` | src(DDR slice), dst(L1) | Developer |
| 6 | 矩阵乘累加 | `T.gemm_v0(A_L1, B_L1, C_L0, init=(k == 0))` | A_L1, B_L1, C_L0, init | Developer |
| 7 | K 维迭代 | `T.serial(loop_k)` | range=T.ceildiv(K, K_L1) | Developer |
| 8 | L0C→GM 搬运 | `T.copy(C_L0, C[bx * block_M, by * block_N])` | src(L0C), dst(DDR slice) | Developer |

### 3.3 计算伪代码

```python
# NPU 端 GEMM kernel (Developer 模式)
@T.prim_func
def main(
    A: T.Tensor((M, K), "float16"),    # 卷积核展开矩阵
    B: T.Tensor((K, N), "float16"),    # im2col 展开矩阵
    C: T.Tensor((M, N), "float16"),    # 卷积输出展开矩阵
):
    with T.Kernel(m_num * n_num, is_npu=True) as (cid, _):
        bx = cid // n_num              # M 维 block 索引
        by = cid % n_num               # N 维 block 索引

        # 1. 分配 L1 buffer（A tile + B tile）
        A_L1 = T.alloc_shared((block_M, K_L1), "float16")
        B_L1 = T.alloc_shared((K_L1, block_N), "float16")

        # 2. 分配 L0C fragment（累加器，float32）
        C_L0 = T.alloc_fragment((block_M, block_N), "float")

        # 3. K 维分块迭代累加
        loop_k = T.ceildiv(K, K_L1)
        for k in T.serial(loop_k):
            # 搬运 A tile: GM(A) → L1(A_L1)
            T.copy(A[bx * block_M, k * K_L1], A_L1)
            # 搬运 B tile: GM(B) → L1(B_L1)
            T.copy(B[k * K_L1, by * block_N], B_L1)
            # 矩阵乘累加: L1 @ L1 → L0C
            # k==0 时 init=True 清零 C_L0，后续 init=False 累加
            T.gemm_v0(A_L1, B_L1, C_L0, init=(k == 0))

        # 4. 输出: L0C → GM(C)
        T.copy(C_L0, C[bx * block_M, by * block_N])
```

### 3.4 API 可行性确认

| API | 来源 | 验证状态 |
|-----|------|---------|
| `T.prim_func` | `api-kernel-memory.md` §1 | ✅ 已通过 examples/gemm/example_gemm.py 验证 |
| `T.Tensor((shape), dtype)` | `api-kernel-memory.md` §1 | ✅ 已通过所有 GEMM 示例验证 |
| `T.Kernel(block_num, is_npu=True) as (cid, _)` | `api-kernel-memory.md` §1 | ✅ 已通过所有 GEMM 示例验证 |
| `T.alloc_shared(shape, dtype)` | `api-kernel-memory.md` §2 (Developer) | ✅ 已通过 examples/gemm/ 验证 |
| `T.alloc_fragment(shape, dtype)` | `api-kernel-memory.md` §2 (Developer) | ✅ 已通过 examples/gemm/ 验证 |
| `T.copy(src, dst)` | `api-kernel-memory.md` §3 | ✅ 已通过所有搬运场景验证 |
| `T.gemm_v0(A, B, C, init)` | `api-compute.md` §1 | ✅ 已通过 examples/gemm/ 验证 |
| `T.serial(N)` | `api-schedule-sync.md` | ✅ 已通过所有循环场景验证 |
| `T.ceildiv(a, b)` | `examples/gemm/example_gemm.py:40` | ✅ 已通过 GEMM 示例验证 |
| `@tilelang.autotune(configs, ref_prog, supply_prog, atol, rtol)` | `examples/convolution/example_convolution_autotune.py:45-51` | ✅ 本算子已直接使用并验证 |

---

## 3.5 技术约束确认

### 3.5.1 本项目已知限制检查

| 约束 | 本算子是否涉及 | 处理方案 |
|------|---------------|----------|
| 不支持三维 Kernel | **No** | 使用一维 `T.Kernel(m_num * n_num, is_npu=True)` 足以表达 2D block 划分 |
| threads 参数限制（仅 1 或 2） | **No** | 未使用 `threads` 参数，默认值即可 |
| 动态循环边界不支持 | **No** | `loop_k = T.ceildiv(K, K_L1)` 是静态表达式（K 和 K_L1 均在 JIT 编译时已知） |
| 流水线不支持动态边界 | **No** | 未使用 `T.Pipelined`，仅使用 `T.serial` |

### 3.5.2 参考实现差异说明

本算子为 Ascend 原生实现，无 GPU 外部参考。但与标准 2D 卷积对比：

| 差异项 | 标准 Conv2d (cuDNN/Direct) | 本项目（Ascend im2col+GEMM） | 转换方案 |
|--------|---------------------------|-------------------------------|----------|
| 计算方式 | 直接卷积或 Winograd 等 | im2col 展开 + GEMM | 用 PyTorch 端 im2col + NPU 端 GEMM |
| 内存占用 | 无中间矩阵膨胀 | im2col 矩阵膨胀 C·KH·KW 倍 | 通过零填充对齐去除整除约束，在可接受范围内 |
| Kernel 维度 | 3D (batch, out_h, out_w) | 2D (M, N) | 通过 `cid // n_num` / `cid % n_num` 将 1D cid 映射到 2D |
| 并行粒度 | 线程级/warp级 | block 级（每 block 处理 block_M × block_N 输出） | block 粒度由 autotune 自动选择 |

### 3.5.3 本项目同类实现参考

| 文件路径 | 相似度 | 关键参考点 |
|----------|--------|-----------|
| `examples/convolution/example_convolution.py` | **几乎相同** | 非 autotune 版本，核函数结构完全一致 |
| `examples/gemm/example_gemm.py` | **高度相似** | T.gemm_v0 + alloc_shared + alloc_fragment 模式、K 维 tiling、T.serial 迭代结构 |
| `examples/gemm/example_gemm_autotune.py` | **高度相似** | autotune 装饰器用法、configs 生成逻辑 |

---

## 4. 数据规格与内存规划

### 4.1 输入张量

| 参数名 | Shape | dtype | 说明 |
|--------|-------|-------|------|
| `input_t` | `(B, C, H, W)` | `float16` | 输入特征图，位于 GM (NPU) |
| `kernel_t` | `(OC, C, KH, KW)` | `float16` | 卷积核权重，位于 GM (NPU) |

### 4.2 输出张量

| 参数名 | Shape | dtype | 说明 |
|--------|-------|-------|------|
| `output` | `(B, OC, HO, WO)` | `float16` | 卷积输出，位于 GM (NPU) |

### 4.3 中间缓冲区（PyTorch 端）

| Buffer 名 | Shape | dtype | 存储层级 | 用途 |
|-----------|-------|-------|----------|------|
| `input_flat` | `(C·KH·KW, B·HO·WO)` | `float16` | GM (NPU) | im2col 变换后的输入矩阵 |
| `kernel_flat` | `(OC, C·KH·KW)` | `float16` | GM (NPU) | 卷积核展平矩阵 |
| `input_padded` | `(K_pad, N_pad)` | `float16` | GM (NPU) | 零填充后的输入矩阵（非整除时） |
| `kernel_padded` | `(M_pad, K_pad)` | `float16` | GM (NPU) | 零填充后的卷积核矩阵（非整除时） |

### 4.4 中间缓冲区（NPU 端 / TileLang）

| Buffer 名 | Shape | dtype | 存储层级 | 用途 |
|-----------|-------|-------|----------|------|
| `A_L1` | `(block_M, K_L1)` | `float16` | L1 | 卷积核 tile 缓冲（Cube 左矩阵） |
| `B_L1` | `(K_L1, block_N)` | `float16` | L1 | 输入 tile 缓冲（Cube 右矩阵） |
| `C_L0` | `(block_M, block_N)` | `float32` | L0C | 输出累加器 fragment |

### 4.5 内存搬运路径

```
=== im2col + pad (PyTorch NPU 端) ===
Input(B,C,H,W) --im2col--> Input_flat(K, N) --pad--> Input_flat_pad(K_pad, N_pad)
Kernel(OC,C,KH,KW) --view--> Kernel_flat(M, K) --pad--> Kernel_flat_pad(M_pad, K_pad)

=== GEMM (TileLang NPU 端) ===
GM[Kernel_flat_pad] --T.copy--> L1[A_L1] --T.gemm_v0--> L0C[C_L0]
GM[Input_flat_pad]   --T.copy--> L1[B_L1] --T.gemm_v0--> L0C[C_L0]
L0C[C_L0] --T.copy--> GM[Output_flat_pad]

=== Reshape (PyTorch NPU 端) ===
Output_flat_pad --slice[:M,:N]--> Output_flat(M, N) --view--> Output(OC, B, HO, WO) --permute--> Output(B, OC, HO, WO)
```

### 4.6 UB 内存预算

本算子仅使用 L1 + L0C，不直接分配 UB（shared 层级由编译器自动映射到 L1）：

| Buffer | Shape | dtype | 大小 (Bytes) | 存储层级 |
|--------|-------|-------|-------------|---------|
| `A_L1` | `(128, 128)` | `float16` | `128 × 128 × 2 = 32,768` | L1 |
| `B_L1` | `(128, 128)` | `float16` | `128 × 128 × 2 = 32,768` | L1 |
| `C_L0` | `(128, 128)` | `float32` | `128 × 128 × 4 = 65,536` | L0C |
| **L1 总计** | (A_L1 + B_L1) | | **65,536 (64KB)** | L1 |
| **L0C 总计** | C_L0 | | **65,536 (64KB)** | L0C |


### 4.7 动态轴定义

无。所有维度在 autotune 阶段通过参数化 JIT 编译为静态 shape。

### 4.8 JIT 配置

```python
pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,   # 自动 CV 分离（为 GEMM 提供 Cube 调度）
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,          # 自动同步插入
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,    # 自动内存规划
}

@tilelang.jit(out_idx=[-1], pass_configs=pass_configs)
def matmul(M, N, K, block_M, block_N, K_L1, dtype="float16", accum_dtype="float"):
    ...
```

---

## 5. Tiling 策略

### 5.1 计算类型

**类型**: 纯 Cube + Vector 混合（由编译器自动调度）

**判定依据**: GEMM 核心运算位于 Cube 核，`T.copy` 搬运和地址计算由 Vector 核辅助完成。通过 `TL_ASCEND_AUTO_CV_COMBINE: True` 让编译器自动分离 Cube/Vector 指令。

### 5.2 Block 划分

```python
# autotune 搜索空间（含维度过滤）
block_M_pool = [64, 128]        # M 维候选池
block_N_pool = [64, 128]        # N 维候选池
K_L1_pool   = [32, 64, 128]     # K 维候选池

block_M = [bs for bs in block_M_pool if bs <= M]   # 过滤 ≤ M 的候选
block_N = [bs for bs in block_N_pool if bs <= N]   # 过滤 ≤ N 的候选
K_L1    = [bs for bs in K_L1_pool   if bs <= K]    # 过滤 ≤ K 的候选

# 退化兜底：当某维度所有候选均 > 实际维度时，使用 max(1, dim) 作为唯一候选
if not block_M: block_M = [max(1, M)]
if not block_N: block_N = [max(1, N)]
if not K_L1:    K_L1    = [max(1, K)]

# 笛卡尔积生成所有配置组合
configs = list(itertools.product(block_M, block_N, K_L1))

# Block 布局
m_num = M // block_M   # M 维 block 个数
n_num = N // block_N   # N 维 block 个数
block_num = m_num * n_num  # 总 block 数

# 每个 block 的 2D 位置映射
bx = cid // n_num   # block 在 M 维的索引
by = cid % n_num    # block 在 N 维的索引
```

**选择理由**：
- `block_M`、`block_N` 从 {64, 128} 中选取，`K_L1` 从 {32, 64, 128} 中选取，与 Ascend 910B 的 Cube 计算单元对齐（128 为最优 tile 尺寸）
- 每个维度至少 ≤ 原始维度，避免无效搜索
- 当实际维度小于所有候选时（如 M=16），兜底使用 `max(1, M)` 作为唯一候选，确保始终有至少一种配置
- `block_M × block_N × sizeof(float32)` = 128×128×4 = 64KB，在 L0C 容量范围内
- `(block_M × K_L1 + K_L1 × block_N) × sizeof(float16)` = 128×128×2×2 = 64KB（取 block_M=128, K_L1=128, block_N=128），在 L1 容量范围内

### 5.3 约束分析

- **对齐约束**: 使用零填充（pad）策略，将 M/N/K 向上对齐到 block size 的整数倍，消除非整除问题。填充区域计算结果被 `output[:M, :N]` 切片丢弃。
- **UB/L1 容量**: 最大 L1 占用 64KB（A_L1 + B_L1），L0C 占用 64KB，安全。
- **L0 容量**: L0C `(128, 128)` = 64KB，在 Ascend 910B L0C 容量内。

### 5.4 注意事项

- **零填充策略**: 当 M、N 或 K 不能被对应 block size 整除时，在 PyTorch 端对矩阵进行零填充，将 GEMM 的输入 shape 变为对齐后的整倍数。这是可选策略（仅 `need_pad` 为 True 时执行）。
- **无效数据过滤**: GEMM 计算出的填充区域结果通过 `output[:M, :N]` 切片丢弃，不影响最终正确性。
- **内存开销**: 零填充在最坏情况下最多额外分配 `(block_M-1) × (block_N-1)` 的 padding 空间，相对于原始 GEMM 矩阵尺寸可忽略。
- **性能影响**: 填充引入的额外计算量在 block 边界上，占总计算量比例 < 1/block_M + 1/block_N，在实际场景中几乎不影响性能。

---

## 6. 循环与调度结构

### 6.1 循环结构总结

| 维度 | 循环类型 | API | 理由 |
|------|----------|-----|------|
| M×N 方向（block 级并行） | 空间并行 | `T.Kernel(m_num * n_num, is_npu=True)` | 每个 block 独立处理 block_M × block_N 的输出区域，block 间无依赖 |
| K 方向（迭代） | 时序迭代 | `T.serial(T.ceildiv(K, K_L1))` | K 维分块累加，需按顺序执行（读写同一 C_L0 累加器） |
| GEMM 指令内部 | 向量化 | `T.gemm_v0` 内部自动 | Cube 单元自动处理 block_M × block_N 内的并行 |

### 6.2 循环伪代码

```python
# Block 级并行（隐式，由 T.Kernel 管理）
# 外层并行：T.Kernel 将 m_num × n_num 个 block 分配到各 AI Core
with T.Kernel(m_num * n_num, is_npu=True) as (cid, _):
    bx = cid // n_num    # M 维 block 索引
    by = cid % n_num     # N 维 block 索引

    # 初始化 buffer
    A_L1 = T.alloc_shared((block_M, K_L1), dtype)
    B_L1 = T.alloc_shared((K_L1, block_N), dtype)
    C_L0 = T.alloc_fragment((block_M, block_N), accum_dtype)

    # K 维时序迭代
    loop_k = T.ceildiv(K, K_L1)
    for k in T.serial(loop_k):
        T.copy(A[bx * block_M, k * K_L1], A_L1)
        T.copy(B[k * K_L1, by * block_N], B_L1)
        T.gemm_v0(A_L1, B_L1, C_L0, init=(k == 0))

    T.copy(C_L0, C[bx * block_M, by * block_N])
```

### 6.3 流水线优化

**未使用流水线**。理由：
- K 维迭代使用简单的 `T.serial` 顺序执行
- 每次迭代的 A tile 和 B tile 通过两条独立的 `T.copy` 搬运
- 通过 `TL_ASCEND_AUTO_SYNC: True` 让编译器自动插入必要的同步屏障
- 如需 pipelined 优化（搬运与计算重叠），可升级为 `T.Pipelined`，但需关注 K 维不是 block_K 整数倍的情况

### 6.4 尾块处理

**K 维尾块**：`loop_k = T.ceildiv(K, K_L1)` 确保最后一次迭代覆盖剩余数据。`T.copy` 自动处理不足 block 大小的搬运（编译器生成正确的搬运指令）。

**M/N 维尾块**：通过 PyTorch 端零填充消除尾块问题。M_pad、N_pad 分别为 block_M 和 block_N 的整数倍，所有 block 尺寸一致，无尾块特殊情况。填充区域的输出结果在 `output[:M, :N]` 切片中被丢弃。

---

## 7. 同步策略

### 7.1 同步模式

**模式**: 自动同步

### 7.2 同步点说明

本算子使用 `TL_ASCEND_AUTO_SYNC: True`，由编译器自动在以下位置插入同步屏障：

| 位置 | 编译器行为 | 理由 |
|------|----------|------|
| `T.copy(GM→L1)` 之后 | 插入 `pipe_barrier` | 等待 DMA 搬运完成后再进行 GEMM 计算 |
| `T.gemm_v0()` 之后 | 插入 `pipe_barrier` | 等待 Cube 计算完成后再进行下一次 K 维迭代的数据搬运 |
| K 维迭代回边 | 自动插入同步 | 确保上次迭代的 C_L0 未被覆盖前，下一次搬运不开始 |
| `T.copy(L0C→GM)` 之前 | 插入同步 | 确保 K 维迭代全部完成后才开始输出搬运 |

### 7.3 pass_configs 配置

```python
pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,   # 自动 CV 分离
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,          # 自动同步插入
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,    # 自动内存规划
}
```

---

## 8. 融合算子设计（如有）

### 8.1 融合算子判定

**判定结果**: **否**（非融合算子）

**判定依据**: 本算子的 NPU 端仅包含 GEMM 计算，GEMM 结果直接通过 `T.copy(C_L0, C[...])` 写入 GM 输出。GEMM 之后无 element-wise 后处理（如 bias add、ReLU 激活等），不构成融合算子场景。im2col 变换在 PyTorch 端执行，不属于 NPU kernel 的融合范畴。

> 如需添加 bias add / ReLU 后处理，应参考 Flash Attention 类融合算子的 workspace 设计方案。

---

## 9. 验证方案

### 9.1 Golden 函数

```python
# 使用 PyTorch 原生 conv2d 作为参考实现
import torch.nn.functional as F

def golden_conv2d(input_tensor, kernel, stride=1, padding=0):
    """PyTorch 原生 2D 卷积（Golden 实现）"""
    return F.conv2d(input_tensor, kernel, stride=stride, padding=padding)
```

### 9.2 测试用例

| 用例名 | 级别 | 输入 Shape | 说明 |
|--------|------|-----------|------|
| Case 1: Perfect alignment | Level 0 | `B=2, C=2, H=15, W=15, OC=128, KH=8, KW=8` | M=N=K=128，三个维度均为 block size 整数倍，无 padding |
| Case 2: M padding | Level 1 | `B=1, C=2, H=32, W=32, OC=50, KH=3, KW=3` | OC=50 → M=50，需 pad 到 128 |
| Case 3: N padding | Level 1 | `B=1, C=4, H=17, W=17, OC=128, KH=3, KW=3` | N=225，需 pad 到 256 |
| Case 4: K padding | Level 1 | `B=2, C=3, H=28, W=28, OC=128, KH=3, KW=3, stride=2, padding=1` | K=27，需 pad 到 128 |
| Case 5: All-dim padding | Level 2 | `B=1, C=3, H=17, W=17, OC=64, KH=3, KW=3` | M=64, N=225, K=27 同时需要 padding |
| Case 6: Multi-block large | Level 3 | `B=4, C=8, H=28, W=28, OC=256, KH=5, KW=5` | M=256, N=2304, K=200，多 block + 大尺寸性能测试 |

### 9.3 精度标准

| dtype | atol | rtol |
|-------|------|------|
| float16 | 1e-2 | 1e-2 |

> 注：本算子使用 float16 计算（accum_dtype=float32），与 PyTorch 参考实现的精度容差为 1e-2。

---

## 10. 风险点与注意事项

### 10.1 已知约束

- im2col 矩阵膨胀比 = `C·KH·KW / C = KH·KW`，当 kernel 尺寸较大时（如 7×7），中间矩阵显存占用显著增加
- 零填充在非整除时引入额外计算，但影响可忽略（< 1%）
- autotune 搜索空间为 `block_M × block_N × K_L1 ∈ {64,128} × {64,128} × {32,64,128}`，最多 2×2×3=12 种配置，搜索开销小
- 当前不支持 dilation 参数（dilated convolution）

### 10.2 常见错误

| 错误 | 触发场景 | 影响 | 解决方案 |
|------|----------|------|----------|
| M/N/K 为 0 | 输入尺寸 < kernel 尺寸且无 padding | Kernel 编译失败 | 确保 `HO > 0` 且 `WO > 0`，或添加足够的 padding |
| im2col 内存爆炸 | 大 batch + 大 HW + 大 kernel | NPU OOM | 使用分块 im2col 或切换到 direct convolution |
| autotune 已添加了错误处理逻辑：`block_M = [max(1, M)]` 确保始终有至少一种配置 |
| 输入维度非法 | key_args 非 tuple/list、维度 ≤0 | 抛出 `ValueError`，autotune 提前终止 | 调用方确保传参正确，padding 后维度 > 0 |
| stride > KH 或 stride > KW | stride 大于 kernel 尺寸 | im2col 窗口越界（已处理为 0 填充） | 正常，符合卷积语义 |

### 10.3 特殊场景处理

- **输入校验**: `get_configs` 在入口处校验 `key_args` 类型（必须为 tuple/list 且长度 ≥ 3）和维度的有效性（M, N, K > 0），非法输入直接抛出 `ValueError`
- **非整除分块**: 通过 PyTorch 端零填充对齐，GEMM 结果切片恢复原始尺寸
- **极小 shape**（如 1×1 kernel、1×1 输入）: im2col 后矩阵极小（K=1 或 N=1），autotune 退化兜底 `block_M = [max(1, M)]` 确保始终有至少一种候选配置
- **混合精度**: 输入/输出为 float16，累加器为 float32（K 维累加精度保护）
- **autotune 编译缓存**: `tilelang.cache.clear_cache()` 在脚本开头清除缓存，确保每次运行重新编译
- **im2col 在 NPU 上执行**: im2col 和 padding 均在 `torch.Tensor.npu()` 上执行，相对于 CPU 端 im2col 避免了 CPU↔NPU 数据传输

---

## 11. 交付清单

### 11.1 目录结构

```
examples/convolution/
├── example_convolution.py            # 非 autotune 版本（固定 block size）
├── example_convolution_autotune.py   # autotune 版本（自动探索最优 block size）
└── convolution_autotune_design.md    # 本设计文档
```

### 11.2 文件清单

| 文件 | 状态 | 说明 |
|------|------|------|
| `example_convolution.py` | ✅ 已完成 | 非 autotune 版本，固定 block_M=128, block_N=256, block_K=64 |
| `example_convolution_autotune.py` | ✅ 已完成 | autotune 版本，block_M/N ∈ {64,128}、K_L1 ∈ {32,64,128}，含维度校验和退化兜底 |
| `convolution_autotune_design.md` | ✅ 已完成 | 本设计文档 |

### 11.3 命名规范

- 目录名: `convolution`（遵循已有目录）
- 实现文件: `example_convolution_autotune.py`
- 设计文档: `convolution_autotune_design.md`

### 11.4 实现顺序

1. ✅ 设计文档（convolution_autotune_design.md）
2. ✅ 非 autotune 版本（example_convolution.py）— 验证基础 GEMM 路径
3. ✅ autotune 版本（example_convolution_autotune.py）— 添加自动调优
4. ✅ 功能测试 (Level 0-3，6 个用例) — 已在 `__main__` 中实现
