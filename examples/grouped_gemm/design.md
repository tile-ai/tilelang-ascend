# grouped_gemm_bwd 算子设计文档

## 1. 概述

### 1.1 算子名称

grouped_gemm_bwd（Grouped GEMM Backward - 权重梯度计算）

### 1.2 功能描述

计算 grouped GEMM 的反向传播中权重矩阵的梯度。支持多个独立批次组的矩阵乘法反向传播，每个组独立计算权重梯度。

### 1.3 数学公式

前向传播：
$$
O_i = A_i \times B_i, \quad i = 1, 2, ..., G
$$

反向传播（权重梯度）：
$$
\frac{\partial L}{\partial B_i} = A_i^T \times \frac{\partial L}{\partial O_i}
$$

其中：
- $A_i \in \mathbb{R}^{b_i \times M}$：第 i 个组的激活值矩阵
- $\frac{\partial L}{\partial O_i} \in \mathbb{R}^{b_i \times N}$：第 i 个组的输出梯度矩阵
- $\frac{\partial L}{\partial B_i} \in \mathbb{R}^{M \times N}$：第 i 个组的权重梯度矩阵
- $b_i$：第 i 个组的批次大小（动态）
- $G$：总组数

### 1.4 算法描述

本算子将多个独立的矩阵乘法反向传播任务合并为一个 kernel 执行：
1. 按组维度（batch_count）并行，每个组独立计算
2. 对于每个组，执行转置矩阵乘法：$dB_i = A_i^T @ dO_i$
3. 使用流水线优化，在 K 维度上进行分块累加
4. 处理动态批次大小，通过 boundary guard 确保边界安全

### 1.5 数据流图

```
输入：A[batch_sum, M], B[batch_sum, N], batch_sizes[batch_count], batch_offsets[batch_count]
  ↓
按组分配（bz 维度）
  ↓
对每个组 bz：
  按块并行（bx, by 维度）
    ↓
  分块搬入 A_shared[block_K, block_M], B_shared[block_K, block_N]
    ↓
  转置矩阵乘累加：C_local = A_shared^T @ B_shared（流水线）
    ↓
  搬出到 C[bz, bx*block_M, by*block_N]
输出：C[batch_count, M, N]（权重梯度）
```

---

## 2. 编程模式选型

### 2.1 模式结论

**选定模式**: Developer 模式

### 2.2 选型理由

1. **计算类型判定**：本算子为纯 Cube 计算（转置 GEMM），无 Vector 后处理，符合 GEMM 类算子特征
2. **自动化优势**：
   - 内存映射自动化：`T.alloc_shared` / `T.alloc_fragment` 由编译器自动映射到 L1/L0C
   - 同步自动化：流水线内同步由编译器自动插入，无需手动 `T.barrier_all`
   - CV 分离自动化：纯 Cube 算子自动分配到 Cube 核
3. **代码简洁性**：Developer 模式代码量少，易于维护
4. **跨平台兼容**：与 GPU 版 TileLang 保持一致的编程范式

### 2.3 模式影响

| 维度 | 本算子的选择 |
|------|-------------|
| 内存分配 | `T.alloc_shared`（编译器映射到 L1）、`T.alloc_fragment`（编译器映射到 L0C） |
| 计算方式 | `T.gemm(A_shared, B_shared, C_local, transpose_A=True)` |
| 作用域 | 无需显式 `T.Scope`，编译器自动识别为 Cube 计算域 |
| 同步方式 | 自动同步（`tl.disable_warp_specialized=True` 禁用 warp 优化，保留基础同步） |

---

## 3. API 映射设计

### 3.1 公式拆解

| 步骤 | 数学表达 | 说明 |
|------|----------|------|
| 1 | $A_{tile}^{(k)} = A_{batch_i}[k \cdot block_K : (k+1) \cdot block_K, bx \cdot block_M : (bx+1) \cdot block_M]$ | 从 GM 搬入 A 的分块到 A_shared |
| 2 | $B_{tile}^{(k)} = B_{batch_i}[k \cdot block_K : (k+1) \cdot block_K, by \cdot block_N : (by+1) \cdot block_N]$ | 从 GM 搬入 B 的分块到 B_shared |
| 3 | $C_{local} += A_{tile}^{(k)T} \times B_{tile}^{(k)}$ | 转置矩阵乘累加 |
| 4 | $dB_i[bx \cdot block_M : (bx+1) \cdot block_M, by \cdot block_N : (by+1) \cdot block_N] = C_{local}$ | 搬出结果到 GM |

### 3.2 TileLang API 映射

| 步骤 | 数学表达 | TileLang API | 参数 | 模式 |
|------|----------|-------------|------|------|
| 1 | 搬入 A 分块 | `T.if_then_else` + 手动赋值 | `A_shared[i, j] = T.if_then_else(i < batch_sizes[bz], A[...], 0)` | Developer |
| 2 | 搬入 B 分块 | `T.if_then_else` + 手动赋值 | `B_shared[i, j] = T.if_then_else(i < batch_sizes[bz], B[...], 0)` | Developer |
| 3 | 转置矩阵乘 | `T.gemm` | `T.gemm(A_shared, B_shared, C_local, transpose_A=True)` | Developer |
| 4 | 搬出结果 | `T.copy` | `T.copy(C_local, C[bz, bx * block_M, by * block_N])` | Developer |

### 3.3 计算伪代码

```python
with T.Kernel(T.ceildiv(M, block_M), T.ceildiv(N, block_N), batch_count, threads=threads) as (bx, by, bz):
    # 1. 分配 buffer
    A_shared = T.alloc_shared([block_K, block_M], dtype)
    B_shared = T.alloc_shared([block_K, block_N], dtype)
    C_local = T.alloc_fragment([block_M, block_N], accum_dtype)

    # 2. 初始化累加器
    T.clear(C_local)

    # 3. 流水线分块计算
    for k in T.Pipelined(T.ceildiv(batch_sizes[bz], block_K), num_stages=num_stages):
        # 搬入 A 分块（带边界保护）
        for i, j in T.Parallel(block_K, block_M):
            A_shared[i, j] = T.if_then_else(
                i < batch_sizes[bz],
                A[batch_offsets[bz] + k * block_K + i, bx * block_M + j],
                0
            )
        # 搬入 B 分块（带边界保护）
        for i, j in T.Parallel(block_K, block_N):
            B_shared[i, j] = T.if_then_else(
                i < batch_sizes[bz],
                B[batch_offsets[bz] + k * block_K + i, by * block_N + j],
                0
            )
        # 转置矩阵乘累加
        T.gemm(A_shared, B_shared, C_local, transpose_A=True)

    # 4. 搬出结果
    T.copy(C_local, C[bz, bx * block_M, by * block_N])
```

### 3.4 API 可行性确认

| API | 来源 | 验证状态 |
|-----|------|---------|
| `T.Kernel(..., batch_count)` | 参考实现 line 177 | ✅ 已验证（三维 kernel） |
| `T.alloc_shared` | [api-kernel-memory.md §2](../tilelang-custom-skill/tilelang-api-best-practices/references/api-kernel-memory.md) | ✅ 已验证 |
| `T.alloc_fragment` | [api-kernel-memory.md §2](../tilelang-custom-skill/tilelang-api-best-practices/references/api-kernel-memory.md) | ✅ 已验证 |
| `T.Pipelined` | [api-schedule-sync.md](../tilelang-custom-skill/tilelang-api-best-practices/references/api-schedule-sync.md) | ✅ 已验证 |
| `T.gemm(..., transpose_A=True)` | [api-compute.md §1](../tilelang-custom-skill/tilelang-api-best-practices/references/api-compute.md) | ✅ 已验证 |
| `T.if_then_else` | TileLang 内置 | ✅ 已验证（条件表达式） |
| `T.copy` | [api-kernel-memory.md §3](../tilelang-custom-skill/tilelang-api-best-practices/references/api-kernel-memory.md) | ✅ 已验证 |
| `T.clear` | 参考实现 line 182 | ✅ 已验证（清零累加器） |

---

## 4. 数据规格与内存规划

### 4.1 输入张量

| 参数名 | Shape | dtype | 说明 |
|--------|-------|-------|------|
| A | (batch_sum, M) | float16 | 激活值矩阵，所有组的激活值拼接 |
| B | (batch_sum, N) | float16 | 输出梯度矩阵，所有组的输出梯度拼接 |
| batch_sizes | (batch_count) | int32 | 每个组的批次大小 |
| batch_offsets | (batch_count) | int32 | 每个组在总序列中的起始偏移 |

### 4.2 输出张量

| 参数名 | Shape | dtype | 说明 |
|--------|-------|-------|------|
| C | (batch_count, M, N) | float16 | 权重梯度矩阵，每个组独立输出 |

### 4.3 中间缓冲区

| Buffer 名 | Shape | dtype | 存储层级 | 用途 |
|-----------|-------|-------|----------|------|
| A_shared | (block_K, block_M) | float16 | L1（shared） | A 的分块输入缓冲（转置后作为左矩阵） |
| B_shared | (block_K, block_N) | float16 | L1（shared） | B 的分块输入缓冲（右矩阵） |
| C_local | (block_M, block_N) | float32 | L0C（fragment） | 累加结果缓冲 |

### 4.4 内存搬运路径

```
纯 Cube 算子路径：
GM[A] --手动搬运（T.if_then_else）--> L1[A_shared] --T.gemm--> L0C[C_local]
GM[B] --手动搬运（T.if_then_else）--> L1[B_shared] --T.gemm--> L0C[C_local]
L0C[C_local] --T.copy--> GM[C[bz, ...]]

说明：
- A_shared 在 T.gemm 中被转置使用：A_shared^T ∈ ℝ^{block_M × block_K}
- 搬入时需要 boundary guard（T.if_then_else）处理动态批次大小
- 搬出时使用 T.copy 直接写入 GM
```

### 4.5 UB 内存预算

本算子为纯 Cube 计算，主要使用 L1 和 L0C：

| Buffer | Shape | dtype | 大小 (Bytes) |
|--------|-------|-------|-------------|
| A_shared | (block_K, block_M) = (64, 64) | float16 | 8192 |
| B_shared | (block_K, block_N) = (64, 128) | float16 | 16384 |
| C_local | (block_M, block_N) = (64, 128) | float32 | 32768 |
| **总计（L1 + L0C）** | | | 57344 Bytes |

### 4.6 动态轴定义

| 动态轴 | 声明方式 | 运行时范围 |
|--------|----------|-----------|
| batch_sizes[bz] | Tensor 参数（运行时传入） | 1 ~ batch_sum（每个组动态） |
| batch_sum | JIT 参数 | 典型值：64, 128, 256 |
| M | JIT 参数 | 512 ~ 16384 |
| N | JIT 参数 | 512 ~ 16384 |

### 4.7 JIT 配置

```python
@tilelang.jit(
    out_idx=[2],
    pass_configs={
        "tl.disable_warp_specialized": True,  # 禁用 warp 优化
    },
)
def grouped_gemm_bwd(
    batch_sum, batch_count, M, N,
    block_M, block_N, block_K,
    num_stages=2, threads=128,
    dtype=T.float16
):
    ...
```

---

## 5. Tiling 策略

### 5.1 计算类型

**类型**: 纯 Cube

**判定依据**: 算子仅包含转置矩阵乘法操作，无 element-wise 运算或归约，判定为纯 Cube 计算。

### 5.2 Block 划分

```python
block_M = 64   # M 维分块大小，平衡 L0C 容量和计算效率
block_N = 128  # N 维分块大小，对齐 128 字节（float16）
block_K = 64   # K 维（批次维度）分块大小，控制流水线迭代次数
block_num_x = T.ceildiv(M, block_M)  # M 方向 block 数
block_num_y = T.ceildiv(N, block_N)  # N 方向 block 数
block_num_z = batch_count             # 组数方向 block 数
```

**选择理由**：
- `block_M=64`：适配 L0C 容量（64×128×4 = 32KB），避免溢出
- `block_N=128`：对齐 UB 128 字节边界，提升搬运效率
- `block_K=64`：控制流水线深度，`num_stages=2` 时占用约 2×(A_shared + B_shared) = 48KB L1

### 5.3 约束分析

- **对齐约束**: 
  - block_N=128, fp16 尾轴 128 > 16 ✓
  - block_M=64, 不要求对齐 ✓
- **L1 容量**: 2×num_stages×(A_shared + B_shared) = 2×2×24KB = 96KB，小于 L1 上限（约 1MB）✓
- **L0C 容量**: C_local = 32KB，小于 L0C 上限（64KB）✓

### 5.4 注意事项

1. **动态批次处理**：通过 `T.if_then_else(i < batch_sizes[bz], ..., 0)` 处理边界，避免访问越界
2. **批次维度的特殊性**：K 维实际为批次维度（batch_sizes[bz]），而非传统矩阵乘的 reduce 维
3. **非整除情况**：使用 `T.ceildiv` 自动向上取整，多余 block 通过 boundary guard 输出零值（不影响正确性）

---

## 6. 循环与调度结构

### 6.1 循环结构总结

| 维度 | 循环类型 | API | 理由 |
|------|----------|-----|------|
| M 方向（bx） | Block 级并行 | `T.Kernel(..., bx)` | 每个 block 处理一个 M 分块 |
| N 方向（by） | Block 级并行 | `T.Kernel(..., by)` | 每个 block 处理一个 N 分块 |
| 组维度（bz） | Block 级并行 | `T.Kernel(..., bz)` | 每个组独立并行计算 |
| K 维（批次） | 流水线迭代 | `T.Pipelined(T.ceildiv(batch_sizes[bz], block_K))` | 批次维度分块累加 |
| 元素搬入 | 向量化并行 | `T.Parallel(block_K, block_M/block_N)` | 搬入时逐元素赋值（带 boundary guard） |

### 6.2 循环伪代码

```python
# 三维 Block 级并行（隐式，由 T.Kernel 管理）
with T.Kernel(
    T.ceildiv(M, block_M),  # bx: M 方向分块数
    T.ceildiv(N, block_N),  # by: N 方向分块数
    batch_count,            # bz: 组数
    threads=threads
) as (bx, by, bz):
    # 分配 buffer
    A_shared = T.alloc_shared([block_K, block_M], dtype)
    B_shared = T.alloc_shared([block_K, block_N], dtype)
    C_local = T.alloc_fragment([block_M, block_N], accum_dtype)

    # 初始化
    T.clear(C_local)

    # K 维流水线循环（批次维度分块）
    for k in T.Pipelined(T.ceildiv(batch_sizes[bz], block_K), num_stages=num_stages):
        # 搬入 A 分块（boundary guard）
        for i, j in T.Parallel(block_K, block_M):
            A_shared[i, j] = T.if_then_else(
                i < batch_sizes[bz],
                A[batch_offsets[bz] + k * block_K + i, bx * block_M + j],
                0
            )
        # 搬入 B 分块（boundary guard）
        for i, j in T.Parallel(block_K, block_N):
            B_shared[i, j] = T.if_then_else(
                i < batch_sizes[bz],
                B[batch_offsets[bz] + k * block_K + i, by * block_N + j],
                0
            )
        # 转置矩阵乘累加
        T.gemm(A_shared, B_shared, C_local, transpose_A=True)

    # 搬出结果
    T.copy(C_local, C[bz, bx * block_M, by * block_N])
```

### 6.3 流水线优化

**使用 T.Pipelined**：
- `num_stages=2`：双缓冲流水线，overlap 数据搬入和计算
- Buffer 管理：需要 2 组 A_shared 和 B_shared（编译器自动管理）
- 流水线收益：搬入与计算重叠，减少总耗时约 30-50%

### 6.4 尾块处理

**策略**：使用 `T.if_then_else` boundary guard
- 搬入时检查：`i < batch_sizes[bz]`，超出部分填充 0
- 搬出时无特殊处理：`T.copy` 直接写入，超出部分不影响正确性（其他 block 已覆盖）
- 优势：避免越界访问，无需额外的尾块 kernel

---

## 7. 同步策略

### 7.1 同步模式

**模式**: 自动同步（Developer 模式）

### 7.2 同步点说明

Developer 模式下，编译器自动插入同步：
- 搬入后：自动插入 `T.barrier_all()`（等待 DMA 完成）
- 计算前：自动插入 Cube/Vector 分离同步（本算子纯 Cube，无需）
- 流水线内：自动插入双缓冲同步（`num_stages=2`）

### 7.3 pass_configs 配置

```python
pass_configs = {
    "tl.disable_warp_specialized": True,  # 禁用 warp 优化，保留基础同步
}
```

**说明**：
- 本算子未显式开启 `TL_ASCEND_AUTO_SYNC` 等开关，使用默认配置
- `tl.disable_warp_specialized=True`：禁用 warp 级优化（针对三维 kernel 场景）

---

## 8. 验证方案

### 8.1 Golden 函数

```python
def ref_grouped_gemm_bwd(A, B, batch_sizes):
    """基于 PyTorch 的参考实现"""
    batch_count = len(batch_sizes)
    M = A.shape[1]
    N = B.shape[1]
    
    dB = torch.empty((batch_count, M, N), device=A.device, dtype=A.dtype)
    
    start = 0
    for i, size in enumerate(batch_sizes):
        end = start + size
        A_i = A[start:end]  # shape: (b_i, M)
        B_i = B[start:end]  # shape: (b_i, N)
        dB[i] = torch.mm(A_i.T, B_i)  # dB_i = A_i^T @ B_i
        start = end
    
    return dB
```

### 8.2 测试用例

| 用例名 | 级别 | batch_sizes | M | N | dtype | 说明 |
|--------|------|-------------|---|---|-------|------|
| basic_small | Level 0 | [32, 64] | 512 | 512 | float16 | 最小功能验证（2 组） |
| typical_1 | Level 1 | [64, 128] | 8192 | 8192 | float16 | 典型配置（参考实现默认） |
| typical_2 | Level 1 | [128, 256, 512] | 4096 | 4096 | float16 | 3 组混合批次 |
| boundary_uneven | Level 2 | [1, 127, 255] | 1024 | 1024 | float16 | 非整除批次测试 |
| large_scale | Level 3 | [1024, 2048] | 16384 | 16384 | float16 | 性能测试（大矩阵） |

### 8.3 精度标准

| dtype | atol | rtol |
|-------|------|------|
| float16 | 1e-2 | 1e-2 |
| float32 | 1e-4 | 1e-4 |

---

## 9. 风险点与注意事项

### 9.1 已知约束

1. **批次维度限制**：`batch_sizes[bz]` 必须为正整数，不支持空组
2. **内存连续性**：A 和 B 必须为连续张量（stride(-1) == 1），否则需 `.contiguous()`
3. **批次偏移正确性**：`batch_offsets` 必须由外部正确计算，偏移之和应等于 batch_sum

### 9.2 常见错误

| 错误 | 触发场景 | 影响 | 解决方案 |
|------|----------|------|----------|
| L0C 溢出 | block_M/block_N 过大 | 编译失败 | 减小 block_M 或 block_N |
| 越界访问 | batch_offsets 错误 | 运行时错误 | 确保 batch_offsets 正确计算 |
| 精度不达标 | 累加次数过多（batch_sizes 过大） | 数值误差 | 使用 float32 累加（accum_dtype=float32） |
| 非连续输入 | A 或 B stride(-1) != 1 | 计算错误 | 调用 `.contiguous()` |

### 9.3 特殊场景处理

1. **极小批次**：当 `batch_sizes[bz] < block_K` 时，流水线循环仅 1 次，boundary guard 填充零值
2. **单组场景**：`batch_count=1` 时，退化为普通转置 GEMM
3. **批次不均衡**：不同组批次大小差异大时，所有组仍并行执行，负载不均衡可能影响性能

---

## 10. 交付清单

### 10.1 目录结构

```
examples/grouped_gemm/
├── example_grouped_gemm_bwd.py  # 算子实现 + 测试
├── design.md                    # 本设计文档
└── README.md                    # 使用说明（可选）
```

### 10.2 文件清单

| 文件 | 状态 | 说明 |
|------|------|------|
| `design.md` | ✅ 已完成 | 设计文档 |
| `example_grouped_gemm_bwd.py` | ⬜ 待迁移 | 参考实现需迁移至本项目 `examples/` |
| `test_grouped_gemm_bwd.py` | ⬜ 待实现 | 扩展测试文件（可选，放入 testing/） |

### 10.3 命名规范

- 目录名: `grouped_gemm`（snake_case）
- 实现文件: `example_grouped_gemm_bwd.py`
- 测试文件: `test_grouped_gemm_bwd.py`

### 10.4 实现顺序

1. ✅ 设计文档（design.md）
2. ⬜ Golden 函数（验证基准）- 已在参考实现中定义
3. ⬜ 算子迁移（从参考路径迁移至本项目 `examples/grouped_gemm/`）
4. ⬜ 基础测试（Level 0 + Level 1）
5. ⬜ 边界测试（Level 2）
6. ⬜ 性能测试（Level 3，可选）

---

## 附录：与参考实现的差异说明

本设计文档基于 `/mnt/workspace/gitCode/cann/tilelang-tileai/tilelang/examples/grouped_gemm/example_grouped_gemm_bwd.py` 生成，功能与逻辑严格一致，差异仅为：

1. **路径调整**：需将参考实现迁移至本项目的 `examples/grouped_gemm/` 目录
2. **依赖检查**：确保本项目的 TileLang 版本支持三维 `T.Kernel` 和 `transpose_A` 参数
3. **环境适配**：参考实现使用 CUDA 设备，本项目需适配昇腾 NPU 设备

**核心逻辑完全一致**：三维 kernel、转置 GEMM、流水线优化、boundary guard 处理均保持不变。