# dequantize_gemv (FP16×INT4) 算子设计文档

## 1. 概述

### 1.1 算子名称

dequantize_gemv_fp16xint4

### 1.2 功能描述

INT4反量化向量乘法（GEMV）：将 INT4 packed 格式的权重矩阵反量化为 FP16，然后与 FP16 激活向量执行向量矩阵乘法（GEMV）。

### 1.3 数学公式

$$
C[j] = \sum_{k=0}^{K-1} A[k] \times \text{Dequant}(B[j,k])
$$

其中：
- $A[k]$：FP16 激活向量（M=1）
- $\text{Dequant}(B[j,k])$：INT4 → FP16 反量化
- INT4 packed 格式：每字节存储 2 个 INT4 值

INT4 反量化公式：
$$
\text{Dequant}(B[j,k]) = \text{cast\_to\_fp16}((B[j, k//2] >> (4 \times (k \% 2))) \& 0xF)
$$

### 1.4 算法描述

计算步骤分解：
1. **INT4 解包**：从 packed INT8 数据中提取 4-bit 值
2. **INT4 → FP16 反量化**：将 4-bit 整数转换为 FP16 浮点数
3. **Dot Product**：计算激活向量与反量化权重的点积
4. **结果输出**：将累加结果写入输出向量

### 1.5 数据流图

```
GM[A (FP16)] → UB[A_UB]
GM[B (INT8 packed)] → UB[B_packed_UB] → 解包 → UB[B_dequant_UB]
                                        ↓
                        A_UB × B_dequant_UB → UB[accum_UB] → 归约 → GM[C]
```

---

## 2. 编程模式选型

### 2.1 模式结论

**选定模式**: Developer 模式

### 2.2 选型理由

基于算子特征分析：
- **计算类型**: 纯 Vector（无 matmul/Cube 计算）
  - INT4 解包：位操作，Vector 核执行
  - INT4 → FP16 反量化：类型转换，Vector 核执行
  - Dot Product：FP16 × FP16 累加，Vector 核执行
- **复杂度**: 多步运算（解包 → 反量化 → dot product → 归约）
- **并行度**: N 方向 block 级并行，K 方向分块迭代累加

选择 Developer 模式的原因：
- 算子无 matmul，不需要 Cube 核和 L0 内存层级
- 纯 Vector 计算可使用 `T.alloc_shared`（自动映射为 UB）
- `T.Parallel` + 符号运算简化代码
- 自动同步管理（AUTO_SYNC）

### 2.3 模式影响

| 维度 | 本算子的选择 |
|------|-------------|
| 内存分配 | `T.alloc_shared`（编译器自动映射为 UB） |
| 计算方式 | `T.Parallel` + 符号运算（位操作、乘法、累加） |
| 作用域 | 无需显式 Scope，编译器自动管理 |
| 同步方式 | 自动同步（开启 AUTO_SYNC） |

---

## 3. API 映射设计

### 3.1 公式拆解

| 步骤 | 数学表达 | 说明 |
|------|----------|------|
| 1 | $i4 = (B[j, k//2] >> (4 \times (k \% 2))) \& 0xF$ | INT4 解包 |
| 2 | $B_{fp16}[j,k] = \text{cast}(i4, \text{fp16})$ | INT4 → FP16 反量化 |
| 3 | $accum[k] = A[k] \times B_{fp16}[j,k]$ | Dot Product 计算 |
| 4 | $C[j] = \sum_{k} accum[k]$ | K 方向归约 |

### 3.2 TileLang API 映射

| 步骤 | 数学表达 | TileLang API | 参数 | 模式 |
|------|----------|-------------|------|------|
| 1 | INT4 解包 | `T.Parallel` + 位运算 | `(val >> shift) & mask` | Developer |
| 2 | INT4 → FP16 | `T.cast(i4, "float16")` | 或直接乘法 | Developer |
| 3 | Dot Product | `T.Parallel` + 乘法累加 | `accum += a * b` | Developer |
| 4 | 归约 | `T.reduce_sum` 或手动累加 | 按实现方式选择 | Developer |

### 3.3 计算伪代码

```python
# Developer 模式实现
with T.Kernel(n_num, is_npu=True) as (cid, vid):
    bn = cid  # 处理第 bn 个 block_N
    
    # 分配 buffer（编译器自动映射为 UB）
    A_UB = T.alloc_shared((block_K,), "float16")       # 激活向量 tile
    B_packed_UB = T.alloc_shared((block_N, block_K // 2), "int8")  # packed INT4
    B_dequant_UB = T.alloc_shared((block_K,), "float16")  # 反量化结果
    accum_UB = T.alloc_shared((block_K,), "float32")   # FP32 累加
    C_UB = T.alloc_shared((block_N,), "float16")       # 输出
    
    for bk in T.serial(k_num):
        # 搬入数据
        T.copy(A[bk * block_K], A_UB)
        T.copy(B[bn * block_N, bk * block_K // 2], B_packed_UB)
        
        # 对每个 N 元素处理
        for n_idx in T.serial(block_N):
            # INT4 解包 + 反量化
            for ki in T.Parallel(block_K):
                packed_val = B_packed_UB[n_idx, ki // 2]
                shift = (ki % 2) * 4
                i4_val = (packed_val >> shift) & 0xF
                B_dequant_UB[ki] = T.cast(i4_val, "float16")
            
            # Dot Product
            T.clear(accum_UB)
            for ki in T.Parallel(block_K):
                accum_UB[ki] = A_UB[ki] * B_dequant_UB[ki]
            
            # 归约
            C_UB[n_idx] = T.reduce_sum(accum_UB)
    
    # 输出
    T.copy(C_UB, C[bn * block_N])
```

### 3.4 API 可行性确认

| API | 来源 | 验证状态 |
|-----|------|---------|
| `T.alloc_shared` | api-kernel-memory.md | ✅ 已验证 |
| `T.copy` | api-kernel-memory.md | ✅ 已验证 |
| `T.Parallel` | api-compute.md | ✅ 已验证 |
| `T.cast` | api-compute.md | ✅ 已验证 |
| `T.reduce_sum` | api-compute.md | ✅ 已验证 |
| 位运算 `>>`, `&` | api-compute.md | ✅ 已验证 |

---

## 4. 数据规格与内存规划

### 4.1 输入张量

| 参数名 | Shape | dtype | 说明 |
|--------|-------|-------|------|
| A | (M, K) 或 (K,) | float16 | 激活向量（M=1） |
| B | (N, K // 2) | int8 | 权重矩阵（packed INT4，每字节 2 个 INT4） |

### 4.2 输出张量

| 参数名 | Shape | dtype | 说明 |
|--------|-------|-------|------|
| C | (M, N) 或 (N,) | float16 | 输出向量 |

### 4.3 中间缓冲区

| Buffer 名 | Shape | dtype | 存储层级 | 用途 |
|-----------|-------|-------|----------|------|
| A_UB | (block_K,) | float16 | UB | 激活向量 tile |
| B_packed_UB | (block_N, block_K // 2) | int8 | UB | packed INT4 数据 |
| B_dequant_UB | (block_K,) | float16 | UB | 反量化结果 |
| accum_UB | (block_K,) | float32 | UB | Dot Product 累加器 |
| C_UB | (block_N,) | float16 | UB | 输出缓冲 |

### 4.4 内存搬运路径

```
GM[A] --T.copy--> UB[A_UB]
                   ↓
GM[B] --T.copy--> UB[B_packed_UB] --解包+反量化--> UB[B_dequant_UB]
                   ↓                              ↓
                   A_UB × B_dequant_UB ---------> UB[accum_UB]
                                                   ↓
                                              T.reduce_sum
                                                   ↓
                   UB[C_UB] --T.copy--> GM[C]
```

### 4.5 UB 内存预算

假设 block_N=128, block_K=128:

| Buffer | Shape | dtype | 大小 (Bytes) |
|--------|-------|-------|-------------|
| A_UB | (128,) | float16 | 256 |
| B_packed_UB | (128, 64) | int8 | 8192 |
| B_dequant_UB | (128,) | float16 | 256 |
| accum_UB | (128,) | float32 | 512 |
| C_UB | (128,) | float16 | 256 |
| **总计** | | | **9.5KB < 128KB ✓** |

### 4.6 动态轴定义

无动态轴（所有维度在编译时确定）

### 4.7 JIT 配置

```python
@tilelang.jit(
    out_idx=[-1],
    pass_configs={
        tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
        tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
    },
)
```

---

## 5. Tiling 策略

### 5.1 计算类型

**类型**: 纯 Vector

**判定依据**: 
- 算子仅包含 element-wise 运算（解包、反量化、乘法）
- 无 matmul，不需要 Cube 核
- K 方向归约在 Vector 核完成

### 5.2 Block 划分

```python
block_N = 128  # N 方向分块，每个 block 处理 128 个输出元素
block_K = 128  # K 方向分块，每次迭代处理 128 个 K 元素

n_num = N // block_N
k_num = K // block_K
```

### 5.3 约束分析

- **对齐约束**: 
  - K % 2 == 0 ✓（INT4 packed format，每字节 2 个 INT4）
  - N % block_N == 0 ✓（输出对齐）
  - K % block_K == 0 ✓（分块对齐）

- **UB 容量**: 
  - 总 buffer = 9.5KB < 128KB ✓

### 5.4 注意事项

- **非整除**: 当 N 或 K 不能被 block size 整除时，需要特殊处理尾块
- **INT4 解包顺序**: 低 4-bit 为第 0 个元素，高 4-bit 为第 1 个元素

---

## 6. 循环与调度结构

### 6.1 循环结构总结

| 维度 | 循环类型 | API | 理由 |
|------|----------|-----|------|
| Block 级 | 并行 | T.Kernel(n_num) | 每个 block 处理 block_N 个输出 |
| K 方向 | 迭代 | T.serial(k_num) | K 维分块迭代累加 |
| N 方向 | 顺序 | T.serial(block_N) | 每个 block 内处理 block_N 个元素 |
| 解包 | 并行 | T.Parallel(block_K) | block_K 个元素并行解包 |
| Dot Product | 并行 | T.Parallel(block_K) | block_K 个乘法并行 |

### 6.2 循环伪代码

```python
# Block 级并行
with T.Kernel(n_num, is_npu=True) as (cid, _):
    bn = cid
    
    # 初始化输出
    for n_idx in T.serial(block_N):
        C_UB[n_idx] = 0
    
    # K 方向迭代累加
    for bk in T.serial(k_num):
        T.copy(A[bk * block_K], A_UB)
        T.copy(B[bn * block_N, bk * block_K // 2], B_packed_UB)
        
        # 对每个 N 元素计算
        for n_idx in T.serial(block_N):
            # 解包 + 反量化 + dot product
            for ki in T.Parallel(block_K):
                packed_val = B_packed_UB[n_idx, ki // 2]
                shift = (ki % 2) * 4
                i4_val = (packed_val >> shift) & 0xF
                fp16_val = T.cast(i4_val, "float16")
                C_UB[n_idx] += A_UB[ki] * fp16_val
    
    T.copy(C_UB, C[bn * block_N])
```

### 6.3 流水线优化

**暂不使用 T.Pipelined**：
- 当前实现为简单循环结构
- 后续优化可考虑：
  - K 方向双缓冲（ping-pong）减少搬入延迟
  - N 方向并行化（使用多 Vector 单元）

### 6.4 尾块处理

暂不实现尾块处理，假设输入 shape 满足：
- N % block_N == 0
- K % block_K == 0
- K % 2 == 0

---

## 7. 同步策略

### 7.1 同步模式

**模式**: 自动同步（Developer pass_configs）

### 7.2 同步点说明

由编译器自动插入同步（开启 AUTO_SYNC）：
- T.copy 后自动插入 barrier（等待搬运完成）
- 计算完成后自动同步

无需手动管理同步点。

### 7.3 pass_configs 配置

```python
pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}
```

---

## 8. 验证方案

### 8.1 Golden 函数

```python
def int4_to_fp16(packed_val: int, pos: int) -> float:
    """
    INT4 → FP16 反量化
    
    参数:
        packed_val: int8 值，包含 2 个 packed INT4
        pos: 位置索引（0 或 1）
    
    返回:
        float16 值
    """
    i4 = (packed_val >> (pos * 4)) & 0xF
    return float(i4)


def torch_unpack_int4(B_packed: torch.Tensor) -> torch.Tensor:
    """
    PyTorch 实现：INT4 → FP16 反量化
    
    参数:
        B_packed: (N, K // 2) int8
    
    返回:
        (N, K) float16
    """
    N, K_half = B_packed.shape
    K = K_half * 2
    
    result = torch.zeros(N, K, dtype=torch.float16)
    B_packed_np = B_packed.numpy()
    
    for j in range(N):
        for k in range(K):
            packed_val = B_packed_np[j, k // 2]
            pos = k % 2
            result[j, k] = int4_to_fp16(packed_val, pos)
    
    return result


def golden_dequant_gemv_fp16xint4(
    A: torch.Tensor,
    B_packed: torch.Tensor,
):
    """
    PyTorch 参考实现
    
    参数:
        A: (K,) 或 (1, K) float16
        B_packed: (N, K // 2) int8
    
    返回:
        (N,) 或 (1, N) float16
    """
    if A.dim() == 2:
        A = A.squeeze(0)  # (K,)
    
    # INT4 → FP16
    B_dequant = torch_unpack_int4(B_packed)  # (N, K)
    
    # GEMV: C = A @ B.T
    C = torch.matmul(A.to(torch.float32), B_dequant.to(torch.float32).T)
    
    return C.to(torch.float16)
```

### 8.2 测试用例

| 用例名 | 级别 | Shape (N, K) | dtype | 说明 |
|--------|------|-------------|-------|------|
| basic_small | Level 0 | (128, 128) | float16 | 最小功能验证 |
| typical_256 | Level 1 | (256, 256) | float16 | 典型配置 |
| typical_512 | Level 1 | (512, 512) | float16 | 中等规模 |
| typical_1k | Level 1 | (1024, 1024) | float16 | 常用规模 |
| large_2k | Level 3 | (2048, 2048) | float16 | 大规模性能测试 |

### 8.3 精度标准

| dtype | atol | rtol |
|-------|------|------|
| float16 | 1e-2 | 1e-2 |
| float32 | 1e-4 | 1e-4 |

---

## 9. 风险点与注意事项

### 9.1 已知约束

1. **INT4 范围有限**：
   - INT4 值范围 0-15（无符号）
   - 反量化后 FP16 精度可能受限

2. **K 必须是偶数**：
   - packed INT4 格式要求 K % 2 == 0

3. **Ascend NPU 不支持 LOP3/dp4a**：
   - NVIDIA 的 LOP3 intrinsic 不可用
   - dp4a（INT8 dot product）不可用
   - 需使用纯 FP16 × FP16 计算

### 9.2 常见错误

| 错误 | 触发场景 | 影响 | 解决方案 |
|------|----------|------|----------|
| K 不是偶数 | K % 2 != 0 | 解包错误 | 确保 K % 2 == 0 |
| 累加溢出 | FP16 累加精度不足 | 精度下降 | 使用 FP32 累加 |
| UB 溢出 | block size 过大 | 编译失败 | 减小 block_N/block_K |

### 9.3 特殊场景处理

- **非整除分块**: 暂不处理，要求输入 shape 满足整除条件
- **极小 shape**: 可能导致 block 数量过少，性能不佳
- **有符号 INT4**: 当前实现为无符号 INT4（0-15），有符号需调整解包逻辑

---

## 10. 交付清单

### 10.1 目录结构

```
examples/dequantize_gemm/
├── design_gemv_fp16xint4.md    # 本设计文档
├── example_dequant_gemv_fp16xint4.py  # 算子实现 + 测试
└── README.md                   # 使用说明（可选）
```

### 10.2 文件清单

| 文件 | 状态 | 说明 |
|------|------|------|
| `design_gemv_fp16xint4.md` | ✅ 已完成 | 设计文档 |
| `example_dequant_gemv_fp16xint4.py` | ⬜ 待实现 | 算子实现 + 测试 |

### 10.3 命名规范

- 目录名: `dequantize_gemm`（与已有目录共用，不同变体）
- 实现文件: `example_dequant_gemv_fp16xint4.py`
- 设计文档: `design_gemv_fp16xint4.md`

### 10.4 实现顺序

1. ✅ 设计文档（design_gemv_fp16xint4.md）
2. ⬜ Golden 函数（验证基准）
3. ⬜ 算子实现（example_dequant_gemv_fp16xint4.py）
4. ⬜ 基础测试（Level 0 + Level 1）
5. ⬜ 边界测试（Level 2）
6. ⬜ 性能测试（Level 3，可选）