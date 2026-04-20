# Grouped GEMM (Forward) 算子设计文档

## 1. 概述

### 1.1 算子名称

Grouped GEMM (Forward)

### 1.2 功能描述

对多组独立矩阵执行分组矩阵乘法。每组 g 的输入为 A_g [M_g, K] 和 B_g [K, N]，输出为 C_g = A_g @ B_g [M_g, N]。各组 M 维度大小可不同，K 和 N 全局共享。采用尾块 padding 策略，将每组 M 维度补齐到 block_M 的整数倍，避免核内越界访问。

### 1.3 数学公式

$$
C_g = A_g \times B_g, \quad \forall g \in \{0, 1, \ldots, G-1\}
$$

其中：
- $A_g \in \mathbb{R}^{M_g \times K}$
- $B_g \in \mathbb{R}^{K \times N}$
- $C_g \in \mathbb{R}^{M_g \times N}$

### 1.4 算法描述

1. **尾块 Padding**：将每组 M 维度补齐到 block_M 的整数倍，各组在连续内存中按 padded offset 排列
2. **元数据表构建**：为每个 M-tile 预计算 [batch_idx, m_start_padded, valid_rows]
3. **分块 GEMM**：每个 kernel block 处理一个 M-tile × N-tile 的矩阵乘，沿 K 维度迭代累加
4. **结果裁剪**：验证时只取每组前 M_g 行有效数据，忽略 padding 区域

### 1.5 数据流图

```
A_g [M_g, K] --pad--> A_padded [Σpadded_M, K]
B_g [K, N]   --stack--> B [G, K, N]
                              ↓
                    [Kernel: block_M × block_N tile GEMM]
                    for k in K: T.copy → L1 → T.gemm_v0 → L0C
                              ↓
C_padded [Σpadded_M, N] --slice valid rows--> C_g [M_g, N]
```

---

## 2. 编程模式选型

### 2.1 模式结论

**选定模式**: Expert 模式

### 2.2 选型理由

- **纯 Cube 计算**：算子核心为矩阵乘法（GEMM），需要手动管理 Ascend NPU 的 Cube 核内存层级（GM → L1 → Cube 核 → L0C）
- **显式内存控制**：需显式分配 `T.alloc_L1` 和 `T.alloc_L0C`，编译器无法自动推断 Cube 核的 buffer 布局
- **手动同步**：Cube 核的数据搬运和计算之间需要 `T.barrier_all()` 显式同步
- **动态分组索引**：通过元数据表查表获取 cur_batch_idx 和 m_start，属于 Expert 模式下的手动控制流

### 2.3 模式影响

| 维度 | 本算子的选择 |
|------|-------------|
| 内存分配 | `T.alloc_L1` / `T.alloc_L0C` 显式指定 |
| 计算方式 | `T.gemm_v0(A_L1, B_L1, C_L0, init=...)` |
| 作用域 | 显式 `with T.Scope("C")` 包裹 Cube 计算 |
| 同步方式 | 手动 `T.barrier_all()` 在搬入后和计算后各一次 |
| pass_configs | 全部关闭（Expert 模式默认） |

---

## 3. API 映射设计

### 3.1 公式拆解

| 步骤 | 数学表达 | 说明 |
|------|----------|------|
| 1 | $C_g[:, :] = \sum_{k} A_g[:, k:k+block_K] \times B_g[k:k+block_K, :]$ | 沿 K 维度分块矩阵乘累加 |

### 3.2 TileLang API 映射

| 步骤 | 数学表达 | TileLang API | 参数 | 模式 |
|------|----------|-------------|------|------|
| 1 | 分配 L1 缓冲 | `T.alloc_L1` | `(block_M, block_K), dtype` | Expert |
| 2 | 分配 L1 缓冲 | `T.alloc_L1` | `(block_K, block_N), dtype` | Expert |
| 3 | 分配 L0C 累加器 | `T.alloc_L0C` | `(block_M, block_N), "float32"` | Expert |
| 4 | 数据搬入 A | `T.copy` | `A[m_start:m_start+block_M, k*block_K:(k+1)*block_K] → A_L1` | Expert |
| 5 | 数据搬入 B | `T.copy` | `B[cur_batch_idx, k*block_K:(k+1)*block_K, by*block_N:(by+1)*block_N] → B_L1` | Expert |
| 6 | 同步等待 | `T.barrier_all()` | 等待 DMA 搬运完成 | Expert |
| 7 | 矩阵乘累加 | `T.gemm_v0` | `A_L1, B_L1, C_L0, init=(k==0)` | Expert |
| 8 | 同步等待 | `T.barrier_all()` | 等待 GEMM 计算完成 | Expert |
| 9 | 结果搬出 | `T.copy` | `C_L0 → C[m_start:m_start+block_M, by*block_N:(by+1)*block_N]` | Expert |

### 3.3 计算伪代码

```python
@tilelang.jit(out_idx=[2])
def grouped_gemm_fwd(batch_sizes_list, K, N, block_M, block_N, block_K, dtype="float16"):
    @T.prim_func
    def kernel(A, B, C, block_metadata):
        with T.Kernel(total_m_blocks * n_num, is_npu=True) as (cid, _):
            bx = cid // n_num
            by = cid % n_num

            cur_batch_idx = block_metadata[bx, 0]
            m_start = block_metadata[bx, 1]

            A_L1 = T.alloc_L1((block_M, block_K), dtype)
            B_L1 = T.alloc_L1((block_K, block_N), dtype)
            C_L0 = T.alloc_L0C((block_M, block_N), "float32")

            with T.Scope("C"):
                loop_k = T.ceildiv(K, block_K)
                for k in T.serial(loop_k):
                    T.copy(A[m_start:m_start + block_M, k * block_K:(k + 1) * block_K], A_L1)
                    T.copy(B[cur_batch_idx, k * block_K:(k + 1) * block_K, by * block_N:(by + 1) * block_N], B_L1)
                    T.barrier_all()
                    T.gemm_v0(A_L1, B_L1, C_L0, init=(k == 0))
                    T.barrier_all()

                T.copy(C_L0, C[m_start:m_start + block_M, by * block_N:(by + 1) * block_N])

    return kernel
```

### 3.4 API 可行性确认

| API | 来源确认 | 验证状态 |
|-----|----------|----------|
| `T.alloc_L1` | `tilelang/language/allocate.py` + `examples/gemm/example_gemm.py` | ✅ 已验证 |
| `T.alloc_L0C` | `tilelang/language/allocate.py` + `examples/gemm/example_gemm.py` | ✅ 已验证 |
| `T.copy` (带切片) | `tilelang/language/copy.py` + `examples/gemm/example_gemm_fwd.py` | ✅ 已验证 |
| `T.gemm_v0` | `tilelang/language/customize.py` (npu_gemm alias) | ✅ 已验证 |
| `T.barrier_all` | `tilelang/language/ascend.py` | ✅ 已验证 |
| `T.Scope("C")` | `tilelang/language/warpgroup.py` | ✅ 已验证 |
| `T.Kernel(is_npu=True)` | `tilelang/language/kernel.py` | ✅ 已验证 |

---

## 4. 数据规格与内存规划

### 4.1 输入张量

| 参数名 | Shape | dtype | 说明 |
|--------|-------|-------|------|
| A | `[Σ padded_M_g, K]` | float16 | 各组 A 拼接后的 padded 输入矩阵 |
| B | `[G, K, N]` | float16 | 各组独立的权重矩阵（stacked） |
| block_metadata | `[total_m_blocks, 3]` | int32 | 元数据表：[batch_idx, m_start_padded, valid_rows] |

### 4.2 输出张量

| 参数名 | Shape | dtype | 说明 |
|--------|-------|-------|------|
| C | `[Σ padded_M_g, N]` | float16 | 各组 C 拼接后的 padded 输出矩阵 |

### 4.3 中间缓冲区

| Buffer 名 | Shape | dtype | 存储层级 | 用途 |
|-----------|-------|-------|----------|------|
| A_L1 | `(block_M, block_K)` | float16 | L1 (Cube 缓存) | A 矩阵 tile 缓冲 |
| B_L1 | `(block_K, block_N)` | float16 | L1 (Cube 缓存) | B 矩阵 tile 缓冲 |
| C_L0 | `(block_M, block_N)` | float32 | L0C (Cube 累加器) | GEMM 累加结果 |

### 4.4 内存搬运路径

```
GM[A] --T.copy--> L1[A_L1] ──────────────────────┐
GM[B] --T.copy--> L1[B_L1]                       │
                                                  │ (硬件自动从 L1 读取到 Cube 核)
                                                  ↓
                                            T.gemm_v0
                                                  ↓
                                             L0C[C_L0]
                                                  │
GM[C] <--T.copy-- L0C[C_L0] ◄────────────────────┘
```

### 4.5 L1 内存预算

以默认配置 `block_M=64, block_N=128, block_K=64` 为例：

| Buffer | Shape | dtype | 存储层级 | 大小 (Bytes) |
|--------|-------|-------|----------|-------------|
| A_L1 | (64, 64) | float16 | L1 | 8,192 |
| B_L1 | (64, 128) | float16 | L1 | 16,384 |
| **L1 总计** | | | | **24,576** / ~192KB (容量充足) |
| C_L0 | (64, 128) | float32 | L0C | 32,768 |

### 4.6 动态轴定义

| 动态轴 | 声明方式 | 运行时范围 |
|--------|----------|-----------|
| `batch_sizes_list` | JIT 编译期参数（tuple） | 任意正整数列表 |
| `K` | 编译期参数 | 需能被 block_K 整除 |
| `N` | 编译期参数 | 需能被 block_N 整除 |
| `Σ padded_M_g` | 由 batch_sizes_list + block_M 推导 | 自动计算 |

### 4.7 JIT 配置

```python
@tilelang.jit(out_idx=[2])
```

- `out_idx=[2]`：返回第三个参数 C（索引从 0 开始：A=0, B=1, C=2）
- Expert 模式不设置 pass_configs，全部使用默认关闭状态

---

## 5. Tiling 策略

### 5.1 计算类型

**类型**: 纯 Cube

**判定依据**: 算子仅包含矩阵乘法（GEMM），无 element-wise 后处理或归约操作，完全由 Cube 核执行。

### 5.2 Block 划分

```python
block_M = 64   # M 方向 tile 大小，与 L0C 累加器行维度匹配
block_N = 128  # N 方向 tile 大小，与 L0C 累加器列维度匹配
block_K = 64   # K 方向分块大小，控制 L1 缓冲占用
total_m_blocks = sum(ceil(M_g / block_M) for g in groups)
n_num = ceil(N / block_N)
block_num = total_m_blocks * n_num
```

**选择理由**:
- `block_M=64, block_N=128` 是 Ascend Cube 核的典型高效 tile shape，与 `T.gemm_v0` 的硬件矩阵乘尺寸对齐
- `block_K=64` 平衡 L1 占用和 K 方向迭代次数

### 5.3 约束分析

- **对齐约束**：
  - 每组 M_g 经 padding 后为 block_M 的整数倍，kernel 内无需处理尾块
  - N 需能被 block_N 整除
  - K 需能被 block_K 整除
- **L1 容量**：24KB < ~192KB，容量充足 ✓
- **L0C 容量**：32KB，经实际编译验证通过（Ascend A2/A3 NPU 的 L0C 寄存器可容纳该尺寸）

### 5.4 注意事项

- **尾块处理**：本算子采用 **padding 策略**而非 mask 策略。每组 M 维度在 host 侧补齐到 block_M 的整数倍，kernel 内始终读写完整 block_M 行，简化核内逻辑
- **元数据表**：`block_metadata[bx, 0]` 给出组索引，`block_metadata[bx, 1]` 给出 padded 后的全局偏移，核内直接查表，无运行时控制流开销
- **N 方向尾块**：当前实现要求 N 能被 block_N 整除，不支持 N 方向尾块

---

## 6. 循环与调度结构

### 6.1 循环结构总结

| 维度 | 循环类型 | API | 理由 |
|------|----------|-----|------|
| Block 级 | 隐式并行 | `T.Kernel(total_m_blocks * n_num, is_npu=True)` | 每个 block 处理一个 M-tile × N-tile |
| K 方向 | 串行迭代 | `T.serial(loop_k)` | 沿 K 维度分块累加，必须串行 |
| 元素级 | 无 | `T.gemm_v0` 硬件加速 | 矩阵乘由 Cube 硬件并行执行 |

### 6.2 循环伪代码

```python
with T.Kernel(total_m_blocks * n_num, is_npu=True) as (cid, _):
    bx = cid // n_num  # M-tile 索引
    by = cid % n_num   # N-tile 索引

    # 查表获取组信息和偏移
    cur_batch_idx = block_metadata[bx, 0]
    m_start = block_metadata[bx, 1]

    # 分配 buffer
    A_L1 = T.alloc_L1((block_M, block_K), dtype)
    B_L1 = T.alloc_L1((block_K, block_N), dtype)
    C_L0 = T.alloc_L0C((block_M, block_N), accum_dtype)

    with T.Scope("C"):
        loop_k = T.ceildiv(K, block_K)
        for k in T.serial(loop_k):
            # 数据搬入
            T.copy(A[m_start:m_start + block_M, k * block_K:(k + 1) * block_K], A_L1)
            T.copy(B[cur_batch_idx, k * block_K:(k + 1) * block_K, by * block_N:(by + 1) * block_N], B_L1)
            T.barrier_all()

            # 矩阵乘累加
            T.gemm_v0(A_L1, B_L1, C_L0, init=(k == 0))
            T.barrier_all()

        # 结果搬出
        T.copy(C_L0, C[m_start:m_start + block_M, by * block_N:(by + 1) * block_N])
```

### 6.3 流水线优化

当前实现未使用 `T.Pipelined`。原因：
- 当前 `T.serial` + `T.barrier_all` 已保证正确性，且该算子已在多组不规则尺寸下验证通过
- `T.Pipelined` 在元数据表驱动的动态分组场景下尚未验证正确性，后续优化可尝试引入以提升吞吐

### 6.4 尾块处理

- **M 方向**：Host 侧 padding，每组 M_g → `ceil(M_g / block_M) * block_M`，kernel 内无尾块
- **N 方向**：要求 N % block_N == 0，不支持尾块
- **K 方向**：要求 K % block_K == 0，不支持尾块

---

## 7. 同步策略

### 7.1 同步模式

**模式**: 手动同步

### 7.2 同步点说明

| 位置 | 同步 API | 理由 |
|------|----------|------|
| A/B 搬入后 | `T.barrier_all()` | 等待 DMA 将 A、B 从 GM 搬运到 L1 完成 |
| GEMM 计算后 | `T.barrier_all()` | 等待 Cube 核完成矩阵乘，确保 C_L0 数据就绪 |

每个 K 迭代包含 2 个同步点，共 `2 * ceil(K / block_K)` 次同步。

### 7.3 pass_configs 配置

```python
# Expert 模式：不设置 pass_configs，全部使用默认关闭状态
# 即：
#   TL_ASCEND_AUTO_SYNC: False (手动同步)
#   TL_ASCEND_MEMORY_PLANNING: False (手动 buffer 管理)
#   TL_ASCEND_AUTO_CV_COMBINE: False
#   TL_ASCEND_AUTO_CV_SYNC: False
```

---

## 8. 验证方案

### 8.1 Golden 函数

```python
def torch_grouped_gemm(a_list, b_list):
    """PyTorch 参考实现：逐组 matmul"""
    assert len(a_list) == len(b_list), "A/B group count mismatch"
    return [torch.matmul(a, b) for a, b in zip(a_list, b_list)]
```

### 8.2 测试用例

| 用例 | 级别 | batch_sizes | K | N | block_M | block_N | block_K | 说明 |
|------|------|-------------|---|---|---------|---------|---------|------|
| small_multi_a | Level 0 | `[16, 33, 96]` | 128 | 96 | 32 | 32 | 32 | 多组含尾块（33→64） |
| small_multi_b | Level 0 | `[16, 64, 128]` | 128 | 96 | 32 | 32 | 32 | 中等规模多组 |
| irregular | Level 2 | `[29, 57, 101]` | 128 | 96 | 32 | 32 | 32 | 不规则尺寸尾块验证 |
| large_multi | Level 2 | `[100, 200, 300]` | 128 | 96 | 32 | 32 | 32 | 大规模多组 |

> 注：以下用例可通过 CLI 入口运行：
> - `[16]` (单组最小验证) — `--batch_sizes 16 --K 128 --N 96`
> - `[64, 128, 256]` (生产规模) — `--batch_sizes 64,128,256 --K 4096 --N 4096 --profile`

### 8.3 精度标准

| dtype | atol | rtol |
|-------|------|------|
| float16 | 1e-3 | 1e-3 |

---

## 9. 风险点与注意事项

### 9.1 已知约束

| 约束 | 说明 |
|------|------|
| K 对齐 | K 必须能被 block_K 整除 |
| N 对齐 | N 必须能被 block_N 整除 |
| M padding | 每组 M_g 在 host 侧自动 padding 到 block_M 的倍数 |
| 动态 shape | `batch_sizes_list` 在 JIT 编译期确定，运行时不可变 |

### 9.2 常见错误

| 错误 | 触发场景 | 影响 | 解决方案 |
|------|----------|------|----------|
| 元数据表越界 | `batch_sizes_list` 为空 | 编译期错误 | 确保至少一组 |
| L1 溢出 | block_M/block_K/block_N 过大 | 编译失败或运行时崩溃 | 减小 tile size |
| 精度不匹配 | 累加溢出（极大 K） | 验证失败 | 使用 float32 累加（已采用） |
| N 方向尾块 | N % block_N != 0 | 结果错误或崩溃 | 当前不支持，需 host 侧 padding |

### 9.3 特殊场景处理

- **极小 M 尺寸**：如 M_g < block_M，padding 后仍按完整 block_M 处理，kernel 读取 padding 零值，结果正确
- **单组场景**：退化为标准 GEMM，元数据表仅一行 `[0, 0, M]`
- **大组数**：元数据表线性增长，但查表开销可忽略（核内仅一次整数查找）

---

## 10. 交付清单

### 10.1 目录结构

```
examples/grouped_gemm/
├── example_grouped_gemm_fwd_ptr.py  # 算子实现 + 测试 + benchmark
├── example_grouped_gemm_fwd.py      # 已有变体（metadata 表方式）
├── design.md                        # 本设计文档
└── README.md                        # 使用说明
```

### 10.2 文件清单

| 文件 | 状态 | 说明 |
|------|------|------|
| `design.md` | ✅ 已完成 | 设计文档 |
| `example_grouped_gemm_fwd_ptr.py` | ✅ 已完成 | 算子实现 + 4 组测试 + benchmark |
| `README.md` | ✅ 已有 | 使用说明 |

### 10.3 命名规范

- 目录名: `grouped_gemm`（snake_case）
- 实现文件: `example_grouped_gemm_fwd_ptr.py`
- 测试函数: `test_grouped_gemm_fwd()`

### 10.4 实现顺序

1. ✅ 设计文档（design.md）
2. ✅ Golden 函数（`torch_grouped_gemm`）
3. ✅ 算子实现（`example_grouped_gemm_fwd_ptr.py`）
4. ✅ 基础测试（Level 0: 2 组用例全部通过）
5. ✅ 边界测试（Level 2: `[29, 57, 101]` 和 `[100, 200, 300]` 通过）
6. ✅ 性能测试（Level 3: CLI `--profile` 模式可用）
