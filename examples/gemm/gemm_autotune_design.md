# GEMM Autotune 算子设计文档

## 1. 概述

### 1.1 算子名称

GEMM Autotune（支持自动调优、流水线和Swizzle优化的矩阵乘法）

### 1.2 功能描述

计算矩阵乘法 C = A @ B.T，其中 A 和 B 分别为 shape (M, K) 和 (N, K) 的矩阵，输出 C 为 shape (M, N) 的矩阵。支持自动调优框架、流水线优化和 Swizzle 优化，可在多种分块配置中选择最优性能配置。

### 1.3 数学公式

$$
C_{ij} = \sum_{k=0}^{K-1} A_{ik} \times B_{jk}
$$

即 $C = A \times B^T$，其中 $A \in \mathbb{R}^{M \times K}$，$B \in \mathbb{R}^{N \times K}$，$C \in \mathbb{R}^{M \times N}$。

### 1.4 算法描述

采用分块策略进行矩阵乘法计算：

1. **Block划分**：将输出矩阵 C 划分为 (block_M, block_N) 大小的分块
2. **K轴迭代**：对 K 维度进行分块迭代，每次处理 block_K 大小的数据
3. **累加计算**：每个 block 对 K 方向迭代累加，得到最终结果
4. **流水线优化**：支持多阶段流水线（num_stages），提升计算与搬运并行度
5. **Swizzle优化**：支持数据块重排，优化 Bank Conflict

### 1.5 数据流图

```
GM[A] ──T.copy──> L1[A_L1] ──┐
                              ├──T.gemm_v0──> L0C[C_L0] ──T.copy──> GM[C]
GM[B] ──T.copy──> L1[B_L1] ──┘

可选优化：
- Pipeline: 多阶段流水线，计算与搬运并行
- Swizzle: Block重排，优化Bank Conflict
```

---

## 2. 编程模式选型

### 2.1 模式结论

**选定模式**: Developer 模式

### 2.2 选型理由

1. **与GPU版本风格一致**：原始GPU版本使用 `T.alloc_shared` 和 `T.alloc_fragment`，属于Developer模式风格
2. **纯Cube计算**：算子仅包含矩阵乘法，无element-wise后处理，编译器可自动分离Cube域
3. **自动调优需求**：Developer模式下内存规划自动化，便于快速切换不同配置进行性能测试
4. **简化开发**：无需手动管理L1/L0层级和同步，降低实现复杂度

### 2.3 模式影响

| 维度 | 本算子的选择 |
|------|-------------|
| 内存分配 | T.alloc_shared（输入缓冲）、T.alloc_fragment（累加缓冲） |
| 计算方式 | T.gemm_v0（矩阵乘法） |
| 作用域 | 编译器自动分离 Cube 域 |
| 同步方式 | pass_configs 开启自动同步 |

---

## 3. API 映射设计

### 3.1 公式拆解

| 步骤 | 数学表达 | 说明 |
|------|----------|------|
| 1 | $A_{tile} = A[by \cdot block_M : (by+1) \cdot block_M, k \cdot block_K : (k+1) \cdot block_K]$ | 分块读取A |
| 2 | $B_{tile} = B[bx \cdot block_N : (bx+1) \cdot block_N, k \cdot block_K : (k+1) \cdot block_K]$ | 分块读取B（transpose_B=True处理转置） |
| 3 | $C_{tile} += A_{tile} \times B_{tile}^T$ | 分块矩阵乘累加 |
| 4 | $C[by \cdot block_M, bx \cdot block_N] = C_{tile}$ | 写回结果 |

### 3.2 TileLang API 映射

| 步骤 | 数学表达 | TileLang API | 参数 | 模式 |
|------|----------|-------------|------|------|
| 1 | 分配缓冲区 | `T.alloc_shared((block_M, block_K), dtype)` | 输入缓冲A_L1 | Developer |
| 2 | 分配缓冲区 | `T.alloc_shared((block_N, block_K), dtype)` | 输入缓冲B_L1（注意shape为(N,K)用于transpose_B） | Developer |
| 3 | 分配缓冲区 | `T.alloc_fragment((block_M, block_N), accum_dtype)` | 累加缓冲C_L0 | Developer |
| 4 | 搬入A分块 | `T.copy(A[by*block_M, k*block_K], A_L1)` | 从GM读取A分块 | Developer |
| 5 | 搬入B分块 | `T.copy(B[bx*block_N, k*block_K], B_L1)` | 从GM读取B分块 | Developer |
| 6 | 矩阵乘累加 | `T.gemm_v0(A_L1, B_L1, C_L0, transpose_B=True, init=(k==0))` | GEMM计算，B自动转置 | Developer |
| 7 | 搬出结果 | `T.copy(C_L0, C[by*block_M, bx*block_N])` | 写回GM | Developer |

### 3.3 计算伪代码

```python
@tilelang.autotune(
    configs=get_configs(),
    ref_prog=ref_program,
    supply_prog=supply_prog,
    atol=1e-2,
    rtol=1e-2,
)
@tilelang.jit(out_idx=[-1], pass_configs=pass_configs)
def matmul(M, N, K, block_M, block_N, block_K, num_stages=0, dtype="float16", accum_dtype="float"):
    m_num = T.ceildiv(M, block_M)
    n_num = T.ceildiv(N, block_N)

    @T.prim_func
    def main(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((N, K), dtype),  # 注意：B shape为(N, K)，transpose_B=True处理
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, _):
            # Swizzle 优化（可选）
            if USE_SWIZZLE:
                cid = T.use_swizzle(cid, M, N, K, block_M, block_N, off=1)

            bx = cid // n_num
            by = cid % n_num

            # 1. 分配 buffer（Developer模式）
            A_L1 = T.alloc_shared((block_M, block_K), dtype)
            B_L1 = T.alloc_shared((block_N, block_K), dtype)
            C_L0 = T.alloc_fragment((block_M, block_N), accum_dtype)

            # 2. K轴迭代计算（支持流水线）
            loop_k = T.ceildiv(K, block_K)
            if USE_PIPELINE and num_stages > 0:
                for k in T.Pipelined(loop_k, num_stages=num_stages):
                    T.copy(A[bx * block_M, k * block_K], A_L1)
                    T.copy(B[by * block_N, k * block_K], B_L1)
                    T.gemm_v0(A_L1, B_L1, C_L0, transpose_B=True, init=(k == 0))
            else:
                for k in T.serial(loop_k):
                    T.copy(A[bx * block_M, k * block_K], A_L1)
                    T.copy(B[by * block_N, k * block_K], B_L1)
                    T.gemm_v0(A_L1, B_L1, C_L0, transpose_B=True, init=(k == 0))

            # 3. 写回结果
            T.copy(C_L0, C[bx * block_M, by * block_N])

    return main
```

### 3.4 API 可行性确认

| API | 来源 | 验证状态 |
|-----|------|----------|
| `T.alloc_shared` | api-kernel-memory.md | ✅ 已验证（examples/developer_mode/gemm_developer.py:76） |
| `T.alloc_fragment` | api-kernel-memory.md | ✅ 已验证（examples/developer_mode/gemm_developer.py:79） |
| `T.copy` | api-kernel-memory.md | ✅ 已验证（多个示例） |
| `T.gemm_v0(..., transpose_B=True)` | api-compute.md:17 | ✅ 已验证（api文档明确支持） |
| `T.ceildiv` | Programming Guide | ✅ 已验证 |
| `T.serial` | api-schedule-sync.md | ✅ 已验证 |
| `T.Pipelined` | api-schedule-sync.md | ✅ 已验证（流水线循环） |
| `T.use_swizzle` | api-schedule-sync.md | ✅ 已验证（Swizzle优化） |
| `@tilelang.autotune` | examples/autotune/example_gemm_autotune.py:54 | ✅ 已验证 |

---

## 4. 数据规格与内存规划

### 4.1 输入张量

| 参数名 | Shape | dtype | 说明 |
|--------|-------|-------|------|
| A | (M, K) | float16/bfloat16 | 输入矩阵A |
| B | (N, K) | float16/bfloat16 | 输入矩阵B（transpose_B=True处理转置） |

### 4.2 输出张量

| 参数名 | Shape | dtype | 说明 |
|--------|-------|-------|------|
| C | (M, N) | float16/bfloat16 | 输出矩阵C |

### 4.3 中间缓冲区

| Buffer 名 | Shape | dtype | 存储层级 | 用途 |
|-----------|-------|-------|----------|------|
| A_L1 | (block_M, block_K) | float16 | L1（shared） | A分块输入缓冲 |
| B_L1 | (block_N, block_K) | float16 | L1（shared） | B分块输入缓冲（用于transpose_B） |
| C_L0 | (block_M, block_N) | float32 | L0C（fragment） | 矩阵乘累加输出缓冲 |

### 4.4 内存搬运路径

```
GM[A] ──T.copy──> L1[A_L1] ──┐
                              ├──T.gemm_v0(transpose_B=True)──> L0C[C_L0] ──T.copy──> GM[C]
GM[B] ──T.copy──> L1[B_L1] ──┘
```

### 4.5 UB/L1 内存预算（示例配置 block_M=128, block_N=256, block_K=64）

| Buffer | Shape | dtype | 大小 (Bytes) |
|--------|-------|-------|-------------|
| A_L1 | (128, 64) | float16 | 16384 |
| B_L1 | (256, 64) | float16 | 32768 |
| C_L0 | (128, 256) | float32 | 131072 |
| **L1总计** | | | 49152 |
| **L0C总计** | | | 131072 |

### 4.6 动态轴定义

无动态轴，M、N、K 均为编译时常量。

### 4.7 JIT 配置

```python
pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}

@tilelang.jit(out_idx=[-1], pass_configs=pass_configs)
```

---

## 5. Tiling 策略

### 5.1 计算类型

**类型**: 纯 Cube

**判定依据**: 算子仅包含矩阵乘法操作，无 element-wise 运算或归约操作。

### 5.2 Block 划分

```python
# 自动调优搜索空间配置
block_M_options = [64, 128, 256]     # M方向分块大小
block_N_options = [64, 128, 256]     # N方向分块大小
block_K_options = [32, 64, 128]      # K方向分块大小
num_stages_options = [0, 2, 3]       # 流水线阶段数

# Block数量
m_num = T.ceildiv(M, block_M)
n_num = T.ceildiv(N, block_N)
block_num = m_num * n_num

# 约束：block_M * block_N <= 256 * 256
```

### 5.3 推荐配置

| 场景 | block_M | block_N | block_K | num_stages | 说明 |
|------|---------|---------|---------|------------|------|
| 默认 | 128 | 256 | 64 | 0 | 平衡性能和内存 |
| 大规模 | 128 | 256 | 64 | 3 | 适合大矩阵，启用流水线 |
| 中等规模 | 128 | 128 | 64 | 2 | 适合中等矩阵，启用流水线 |
| 小规模 | 64 | 64 | 32 | 0 | 适合小矩阵调试 |

### 5.4 注意事项

当前实现假设 M、N、K 可被 block_M、block_N、block_K 整除。非整除情况需要额外处理（参考 examples/gemm/example_gemm_tail_block_developer.py）。

---

## 6. 循环与调度结构

### 6.1 循环结构总结

| 维度 | 循环类型 | API | 理由 |
|------|----------|-----|------|
| Block级 | 并行 | `T.Kernel(m_num * n_num)` | 每个block处理一个输出分块 |
| K方向 | 串行迭代 | `T.serial(loop_k)` 或 `T.Pipelined(loop_k, num_stages)` | K维分块迭代累加，可选流水线优化 |

### 6.2 循环伪代码

```python
# Block 级并行
with T.Kernel(m_num * n_num, is_npu=True) as (cid, _):
    # Swizzle 优化（可选）
    if USE_SWIZZLE:
        cid = T.use_swizzle(cid, M, N, K, block_M, block_N, off=1)
    
    bx = cid // n_num  # M方向的block索引
    by = cid % n_num   # N方向的block索引

    # K轴迭代（串行或流水线）
    if USE_PIPELINE and num_stages > 0:
        for k in T.Pipelined(loop_k, num_stages=num_stages):
            T.copy(A[bx * block_M, k * block_K], A_L1)
            T.copy(B[by * block_N, k * block_K], B_L1)
            T.gemm_v0(A_L1, B_L1, C_L0, transpose_B=True, init=(k == 0))
    else:
        for k in T.serial(loop_k):
            T.copy(A[bx * block_M, k * block_K], A_L1)
            T.copy(B[by * block_N, k * block_K], B_L1)
            T.gemm_v0(A_L1, B_L1, C_L0, transpose_B=True, init=(k == 0))
```

### 6.3 流水线优化

当 `num_stages > 0` 且启用流水线时，使用 `T.Pipelined` 循环：

```python
# 流水线版本
for k in T.Pipelined(loop_k, num_stages=num_stages):
    T.copy(A[bx * block_M, k * block_K], A_L1)
    T.copy(B[by * block_N, k * block_K], B_L1)
    T.gemm_v0(A_L1, B_L1, C_L0, transpose_B=True, init=(k == 0))
```

**num_stages 选择**：
- `num_stages=0`: 不使用流水线（默认）
- `num_stages=2`: 2阶段流水线，适合中等规模
- `num_stages=3`: 3阶段流水线，适合大规模矩阵

### 6.4 Swizzle 优化

Swizzle 优化通过重排 Block 计算顺序，减少 Bank Conflict：

```python
if USE_SWIZZLE:
    cid = T.use_swizzle(cid, M, N, K, block_M, block_N, off=1)
```

**参数说明**：
- `cid`: 原始 Block ID
- `M, N, K`: 矩阵维度
- `block_M, block_N`: 分块大小
- `off`: 偏移量参数

### 6.5 启发式配置

根据问题规模自动选择最优配置：

```python
def get_heuristic_config():
    if M >= 4096 and N >= 4096 and K >= 4096:
        return {"block_M": 128, "block_N": 256, "block_K": 64, "num_stages": 3}
    elif M >= 2048 and N >= 2048:
        return {"block_M": 128, "block_N": 128, "block_K": 64, "num_stages": 2}
    else:
        return {"block_M": 64, "block_N": 64, "block_K": 32, "num_stages": 0}
```

### 6.6 尾块处理

当前版本假设整除。非整除场景可参考：
- examples/gemm/example_gemm_tail_block_developer.py（Developer模式尾块处理）
- 使用条件判断处理边界情况

---

## 7. 同步策略

### 7.1 同步模式

**模式**: 自动同步

### 7.2 同步点说明

Developer模式下，编译器自动插入必要同步：
- T.copy 搬运后自动同步
- T.gemm_v0 前后自动同步
- 无需手动调用 T.barrier_all()

### 7.3 pass_configs 配置

```python
pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,        # 自动同步插入
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,  # 自动内存规划
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,  # 自动CV分离（纯Cube无需）
}
```

---

## 8. 验证方案

### 8.1 Golden 函数

```python
def ref_program(A, B):
    """
    Compute the matrix product of A and the transpose of B.
    A: (M, K), B: (N, K) -> C: (M, N) = A @ B.T
    """
    return A @ B.T
```

### 8.2 测试用例

| 用例名 | 级别 | Shape | dtype | 配置 | 说明 |
|--------|------|-------|-------|------|------|
| basic_small | Level 0 | (128, 128, 128) | float16 | num_stages=0 | 最小功能验证 |
| typical_1024 | Level 1 | (1024, 1024, 1024) | float16 | num_stages=2 | 典型配置正确性 |
| typical_4096 | Level 1 | (4096, 4096, 4096) | float16 | num_stages=3 | 典型大配置正确性 |
| pipeline_2 | Level 1 | (2048, 2048, 2048) | float16 | num_stages=2, pipeline=True | 流水线验证 |
| pipeline_3 | Level 1 | (4096, 4096, 4096) | float16 | num_stages=3, pipeline=True | 3阶段流水线验证 |
| swizzle | Level 1 | (4096, 4096, 4096) | float16 | swizzle=True | Swizzle优化验证 |
| boundary | Level 2 | (64, 64, 32) | float16 | num_stages=0 | 最小分块边界 |
| large_scale | Level 3 | (8192, 8192, 8192) | float16 | num_stages=3 | 性能测试 |

### 8.3 精度标准

| dtype | atol | rtol |
|-------|------|------|
| float16 | 1e-2 | 1e-2 |
| bfloat16 | 1e-2 | 1e-2 |
| float32 | 1e-4 | 1e-4 |

---

## 9. 风险点与注意事项

### 9.1 已知约束

1. **整除假设**：当前实现假设 M、N、K 可被 block size 整除
2. **transpose_B用法**：B_L1 的 shape 必须为 (block_N, block_K)，用于 transpose_B=True
3. **无roller支持**：Ascend版本暂不支持GPU版本的roller自动搜索空间生成

### 9.2 常见错误

| 错误 | 触发场景 | 影响 | 解决方案 |
|------|----------|------|----------|
| L0C容量溢出 | block_M*block_N过大 | 编译失败 | 减小block_M或block_N |
| shape不匹配 | B_L1 shape错误 | 运行错误 | B_L1=(block_N, block_K)用于transpose_B |
| 非整除 | M/block_M有余数 | 结果错误 | 添加尾块处理或确保整除 |

### 9.3 GPU与Ascend差异

| 特性 | GPU版本 | Ascend版本 |
|------|----------|------------|
| GEMM API | `T.gemm(..., transpose_B=True)` | `T.gemm_v0(..., transpose_B=True)` |
| 搜索空间 | 支持roller自动生成 | 手动配置搜索空间 |
| Swizzle | `T.use_swizzle(panel_size=10)` | `T.use_swizzle(cid, M, N, K, block_M, block_N, off=1)` |
| Pipeline | `T.Pipelined(num_stages)` | 支持，用法相同 |

---

## 10. 交付清单

### 10.1 目录结构

```
examples/gemm/
├── example_gemm_autotune.py  # 算子实现 + 自动调优 + 流水线 + Swizzle
├── gemm_autotune_design.md   # 本设计文档
```

### 10.2 文件清单

| 文件 | 状态 | 说明 |
|------|------|------|
| `gemm_autotune_design.md` | ✅ 已完成 | 设计文档 |
| `example_gemm_autotune.py` | ✅ 已完成 | 算子实现（支持流水线和Swizzle） |

### 10.3 命名规范

- 实现文件: `example_gemm_autotune.py`

### 10.4 实现顺序

1. ✅ 设计文档（design.md）
2. ✅ Golden 函数（验证基准）
3. ✅ 算子实现（example_gemm_autotune_advanced.py）
4. ✅ 基础测试（Level 0 + Level 1）
5. ✅ 边界测试（Level 2）
6. ✅ 性能测试（Level 3）

### 10.5 运行方式

```bash
# 使用启发式配置（不启用自动调优）
python example_gemm_autotune.py --m 4096 --n 4096 --k 4096

# 启用自动调优
python example_gemm_autotune.py --m 4096 --n 4096 --k 4096 --use_autotune

# 启用流水线优化
python example_gemm_autotune.py --m 4096 --n 4096 --k 4096 --use_pipeline

# 启用 Swizzle 优化
python example_gemm_autotune.py --m 4096 --n 4096 --k 4096 --use_swizzle
```