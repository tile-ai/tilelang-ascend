# dequantize_gemm (bf16+mxfp4) 算子设计文档

## 1. 概述

### 1.1 算子名称

dequantize_gemm (bf16+mxfp4)

### 1.2 功能描述

MXFP4反量化矩阵乘法：将 MXFP4 格式的权重矩阵反量化为 BF16，然后与 BF16 激活矩阵执行矩阵乘法，支持 per-block scale 缩放因子和可选 bias 加法。

### 1.3 数学公式

$$
C[i,j] = \sum_k A[i,k] \times \left( \text{Dequant}(B^T[j,k]) \times 2^{\text{Scale}[j, k_{idx}]} \right) + \text{Bias}[i,j]
$$

其中：
- $\text{Dequant}(B^T[j,k])$：MXFP4 → BF16 反量化
- $\text{Scale}[j, k_{idx}]$：per-block scale 因子，$k_{idx} = k // 32$
- $2^{\text{Scale}[j, k_{idx}]}$：scale 应用（通过位移实现）

### 1.4 算法描述

计算步骤分解：
1. **MXFP4 反量化**：将 packed MXFP4 (UINT8) 解包为 BF16 基础值
2. **Scale 应用**：将 per-block scale 因子应用于反量化结果
3. **GEMM 计算**：执行 A @ B.T 矩阵乘法（BF16 × BF16，FP32 累加）
4. **Bias 加法**（可选）：在累加结果上加 bias

### 1.5 数据流图

```
GM[A (BF16)] → L1[A_L1] → L0A → GEMM → L0C[C_L0C] → UB[C_UB] → GM[C]
GM[B (UINT8)] → L1[B_L1] → UB[B_UB] → 反量化 → UB[B_dequant] → Scale应用 → UB[B_scaled] → L1 → L0B
GM[Scale (UINT8)] → UB[Scale_UB] → Scale应用
GM[Bias (BF16)] → UB[Bias_UB] → Bias加法（可选）
```

---

## 2. 编程模式选型

### 2.1 模式结论

**选定模式**: 混合模式（Developer pass_configs + 少量手动控制）

### 2.2 选型理由

基于算子特征分析：
- **计算类型**: Cube + Vector 混合（GEMM 在 Cube 核，反量化/Scale/Bias 在 Vector 核）
- **复杂度**: 多步运算（反量化 → Scale → GEMM → Bias）
- **流水线**: 需要核间流水线（CV 分离）
- **同步需求**: Cube 和 Vector 核之间需要同步

选择混合模式的原因：
- Developer 模式的 pass_configs 可以自动管理 CV 分离和核间同步
- 反量化、Scale 应用等 Vector 核操作可以使用 T.Parallel 简化代码
- GEMM 部分使用 T.gemm_v0，编译器自动管理内存层级
- 相比 Expert 模式，开发复杂度更低，性能损失可控

### 2.3 模式影响

| 维度 | 本算子的选择 |
|------|-------------|
| 内存分配 | `T.alloc_shared`（编译器自动判断 L1/UB） + `T.alloc_fragment`（L0C） |
| 计算方式 | Vector: `T.Parallel` + 符号运算；Cube: `T.gemm_v0` |
| 作用域 | 编译器自动分离 Cube/Vector Scope（开启 AUTO_CV_COMBINE） |
| 同步方式 | 自动核间同步（开启 AUTO_CV_SYNC） |

---

## 3. API 映射设计

### 3.1 公式拆解

| 步骤 | 数学表达 | 说明 |
|------|----------|------|
| 1 | $B_{bf16} = \text{Dequant}(B_{mxfp4})$ | MXFP4 → BF16 基础反量化 |
| 2 | $B_{scaled} = B_{bf16} \times 2^{\text{Scale}}$ | 应用 per-block scale 因子 |
| 3 | $C_{fp32} = \sum_k A[i,k] \times B_{scaled}[j,k]$ | GEMM 矩阵乘法（FP32 累加） |
| 4 | $C_{final} = C_{fp32} + \text{Bias}[i,j]$ | Bias 加法（可选） |

### 3.2 TileLang API 映射

| 步骤 | 数学表达 | TileLang API | 参数 | 模式 |
|------|----------|-------------|------|------|
| 1 | MXFP4 → BF16 | `T.Parallel` + 自定义反量化函数 | 自定义 `_tir_u8_to_f4_to_bf16` | Developer |
| 2 | Scale 应用 | `T.Parallel` + 位移 | `T.shift_left(1, scale_value)` 或乘法 | Developer |
| 3 | GEMM | `T.gemm_v0(A_L1, B_L1, C_L0C, init=True)` | transpose_B=True, init=(k==0) | Developer |
| 4 | Bias 加法 | `T.Parallel` + 加法 | 或 `T.copy(Bias, C)` 初始化 | Developer |

### 3.3 计算伪代码

```python
with T.Kernel(m_num * n_num, is_npu=True) as (cid, _):
    bx = cid // n_num
    by = cid % n_num
    
    # 1. 分配 buffer
    A_L1 = T.alloc_shared((block_M, block_K), "bfloat16")
    B_L1 = T.alloc_shared((block_K, block_N), "bfloat16")
    C_L0C = T.alloc_fragment((block_M, block_N), "float")  # FP32 累加
    
    B_packed_UB = T.alloc_shared((block_K, block_N // 2), "uint8")
    Scale_UB = T.alloc_shared((block_K, block_N // scale_size), "uint8")
    B_dequant_UB = T.alloc_shared((block_K, block_N), "bfloat16")
    
    # 2. 初始化累加器（考虑 Bias）
    if with_bias:
        Bias_UB = T.alloc_shared((block_M, block_N), "bfloat16")
        T.copy(Bias[bx * block_M, by * block_N], Bias_UB)
        T.copy(Bias_UB, C_L0C)  # 将 Bias 作为初始值
    else:
        T.clear(C_L0C)
    
    # 3. K 方向迭代
    for k in T.serial(K // block_K):
        # 搬入数据
        T.copy(A[bx * block_M, k * block_K], A_L1)
        T.copy(B[k * block_K, by * block_N // 2], B_packed_UB)
        T.copy(Scale[k * block_K, by * block_N // scale_size], Scale_UB)
        
        # 反量化 + Scale 应用（Vector 核）
        for i, j in T.Parallel(block_K, block_N):
            # MXFP4 → BF16
            val = B_packed_UB[i, j // 2]
            pos = j % 2
            scale_idx = j // scale_size
            scale_val = Scale_UB[i, scale_idx]
            
            # 反量化
            bf16_bits = _tir_u8_to_f4_to_bf16(val, pos)
            
            # Scale 应用
            scale_factor = T.shift_left(1, scale_val)
            B_dequant_UB[i, j] = bf16_bits * scale_factor
        
        # 拷贝反量化结果到 L1
        T.copy(B_dequant_UB, B_L1)
        
        # GEMM（Cube 核）
        T.gemm_v0(A_L1, B_L1, C_L0C, init=False)
    
    # 4. FP32 → BF16 输出
    T.copy(C_L0C, C[bx * block_M, by * block_N])
```

### 3.4 API 可行性确认

| API | 来源 | 验证状态 |
|-----|------|---------|
| `T.alloc_shared` | tilelang-api-best-practices | ✅ 已验证 |
| `T.alloc_fragment` | tilelang-api-best-practices | ✅ 已验证 |
| `T.copy` | tilelang-api-best-practices | ✅ 已验证 |
| `T.gemm_v0` | api-compute.md, examples/gemm | ✅ 已验证 |
| `T.Parallel` | api-compute.md | ✅ 已验证 |
| `T.shift_left` | 需确认（符号运算支持） | ⚠️ 待验证 |
| `_tir_u8_to_f4_to_bf16` | 需自定义 intrinsic | ⚠️ 待实现 |

**注意**：
- `_tir_u8_to_f4_to_bf16` 需要在 `tilelang/language/` 中实现自定义 intrinsic
- Ascend NPU 不支持 BF16 矩阵乘法，需要改为 FP16 输入或软件模拟

---

## 4. 数据规格与内存规划

### 4.1 输入张量

| 参数名 | Shape | dtype | 说明 |
|--------|-------|-------|------|
| A | (M, K) | bfloat16 | 激活矩阵 |
| B | (N, K // 2) | uint8 | 权重矩阵（packed MXFP4，每字节 2 个 FP4） |
| Scale | (N, K // 32) | uint8 | Per-block scale 因子（每 32 元素共享） |
| Bias | (M, N) | bfloat16 | Bias（可选） |

### 4.2 输出张量

| 参数名 | Shape | dtype | 说明 |
|--------|-------|-------|------|
| C | (M, N) | bfloat16 | 输出矩阵 |

### 4.3 中间缓冲区

| Buffer 名 | Shape | dtype | 存储层级 | 用途 |
|-----------|-------|-------|----------|------|
| A_L1 | (block_M, block_K) | bfloat16 | L1 (shared) | 激活 tile 缓冲 |
| B_L1 | (block_K, block_N) | bfloat16 | L1 (shared) | 权重 tile 缓冲（反量化后） |
| C_L0C | (block_M, block_N) | float | L0C (fragment) | GEMM 累加器（FP32） |
| B_packed_UB | (block_K, block_N // 2) | uint8 | UB (shared) | MXFP4 packed 数据缓冲 |
| Scale_UB | (block_K, block_N // 32) | uint8 | UB (shared) | Scale 因子缓冲 |
| B_dequant_UB | (block_K, block_N) | bfloat16 | UB (shared) | 反量化结果缓冲 |
| Bias_UB | (block_M, block_N) | bfloat16 | UB (shared) | Bias 缓冲（可选） |

### 4.4 内存搬运路径

```
GM[A] --T.copy--> L1[A_L1]
                        ↓
GM[B] --T.copy--> UB[B_packed_UB] --反量化--> UB[B_dequant_UB] --T.copy--> L1[B_L1]
                                                                        ↓
GM[Scale] --T.copy--> UB[Scale_UB] --Scale应用--> UB[B_dequant_UB]      ↓
                                                                        ↓
GM[Bias] --T.copy--> UB[Bias_UB] --初始化--> L0C[C_L0C]                ↓
                                                                        ↓
                    L1[A_L1] + L1[B_L1] --T.gemm_v0--> L0C[C_L0C]       ↓
                                                                        ↓
                    L0C[C_L0C] --T.copy--> GM[C]
```

### 4.5 UB 内存预算

假设 block_M=128, block_N=128, block_K=128, scale_size=32:

| Buffer | Shape | dtype | 大小 (Bytes) |
|--------|-------|-------|-------------|
| A_L1 | (128, 128) | bfloat16 | 32KB |
| B_L1 | (128, 128) | bfloat16 | 32KB |
| B_packed_UB | (128, 64) | uint8 | 8KB |
| Scale_UB | (128, 4) | uint8 | 512B |
| B_dequant_UB | (128, 128) | bfloat16 | 32KB |
| Bias_UB | (128, 128) | bfloat16 | 32KB |
| **总计（不含 Bias）** | | | **104.5KB < 128KB ✓** |
| **总计（含 Bias）** | | | **136.5KB > 128KB ❌** |

**注意**：当 with_bias=True 时，UB 内存可能超限，需要优化：
- 方案 1：减小 block size（如 block_M=64, block_N=128）
- 方案 2：Bias 直接初始化到 L0C，不占用 UB
- 方案 3：B_dequant_UB 和 B_L1 共用同一块内存（反量化后直接拷贝）

### 4.6 动态轴定义

无动态轴（所有维度在编译时确定）

### 4.7 JIT 配置

```python
@tilelang.jit(
    out_idx=[-1],
    pass_configs={
        tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
        tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
        tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
        tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
    },
)
```

---

## 5. Tiling 策略

### 5.1 计算类型

**类型**: Cube + Vector 混合

**判定依据**: 
- 算子包含 GEMM（matmul），需要 Cube 核执行
- 反量化、Scale 应用、Bias 加法需要 Vector 核执行
- 需要核间流水线（CV 分离）

### 5.2 Block 划分

```python
block_M = 128  # M 方向分块，平衡并行度和内存占用
block_N = 128  # N 方向分块，需满足 scale_size=32 的对齐约束
block_K = 128  # K 方向分块，减少迭代次数

m_num = M // block_M
n_num = N // block_N
block_num = m_num * n_num
```

### 5.3 约束分析

- **对齐约束**: 
  - K % 2 == 0 ✓（MXFP4 packed format）
  - K % 32 == 0 ✓（scale_size=32）
  - block_N % scale_size == 0 → block_N % 32 == 0 ✓（block_N=128）
  - block_K % scale_size == 0 → block_K % 32 == 0 ✓（block_K=128）

- **UB 容量**: 
  - 不含 Bias: 104.5KB < 128KB ✓
  - 含 Bias: 136.5KB > 128KB ❌（需优化）

- **L0 容量**: 
  - C_L0C (128, 128) FP32 = 64KB < 128KB ✓

### 5.4 注意事项

- **非整除**: 当 M/N/K 不能被 block size 整除时，需要特殊处理尾块（暂不实现）
- **Scale 对齐**: block_N 和 block_K 必须是 scale_size（32）的倍数，否则 scale_idx 计算错误
- **Bias 内存优化**: 推荐方案 2（Bias 直接初始化到 L0C），避免 UB 溢出

---

## 6. 循环与调度结构

### 6.1 循环结构总结

| 维度 | 循环类型 | API | 理由 |
|------|----------|-----|------|
| Block 级 | 并行 | T.Kernel(m_num * n_num) | 每个 block 处理一个 (M, N) tile |
| K 方向 | 迭代 | T.serial(K // block_K) | K 维分块迭代累加 |
| 反量化 | 向量化 | T.Parallel(block_K, block_N) | block 内逐元素反量化 |
| Scale 应用 | 向量化 | T.Parallel(block_K, block_N) | block 内逐元素 scale 应用 |

### 6.2 循环伪代码

```python
# Block 级并行
with T.Kernel(m_num * n_num, is_npu=True) as (cid, _):
    bx = cid // n_num
    by = cid % n_num
    
    # 初始化
    if with_bias:
        T.copy(Bias[bx * block_M, by * block_N], C_L0C)
    else:
        T.clear(C_L0C)
    
    # K 方向迭代
    for k in T.serial(K // block_K):
        # 搬入 A
        T.copy(A[bx * block_M, k * block_K], A_L1)
        
        # 搬入 B 和 Scale
        T.copy(B[k * block_K, by * block_N // 2], B_packed_UB)
        T.copy(Scale[k * block_K, by * block_N // scale_size], Scale_UB)
        
        # 反量化 + Scale（Vector 核）
        for i, j in T.Parallel(block_K, block_N):
            # ... 反量化逻辑
        
        # GEMM（Cube 核）
        T.gemm_v0(A_L1, B_dequant_UB, C_L0C)
    
    # 搬出
    T.copy(C_L0C, C[bx * block_M, by * block_N])
```

### 6.3 流水线优化

**暂不使用 T.Pipelined**：
- 反量化 + Scale + GEMM 的流水线需要精细控制 buffer 生命周期
- 当前实现依赖 AUTO_CV_SYNC 自动管理核间同步
- 后续优化可考虑：
  - 使用 T.Pipelined(num_stages=2) 实现 K 方向流水线
  - 使用双缓冲（ping-pong buffer）减少搬入搬出延迟

### 6.4 尾块处理

暂不实现尾块处理，假设输入 shape 满足：
- M % block_M == 0
- N % block_N == 0
- K % block_K == 0

后续可扩展：
- 使用条件判断处理尾块
- 使用 T.if 语句动态选择 block size

---

## 7. 同步策略

### 7.1 同步模式

**模式**: 自动同步 + 自动核间同步（Developer pass_configs）

### 7.2 同步点说明

由编译器自动插入同步（开启 AUTO_SYNC 和 AUTO_CV_SYNC）：
- T.copy 后自动插入 barrier（等待 DMA 完成）
- T.gemm_v0 后自动插入 barrier（等待 Cube 计算完成）
- Vector 核和 Cube 核之间自动插入 cross_flag（核间同步）

无需手动管理同步点。

### 7.3 pass_configs 配置

```python
pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,        # 自动核内同步
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,   # 自动内存规划
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,   # 自动 CV 分离
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,      # 自动核间同步
}
```

---

## 8. 验证方案

### 8.1 Golden 函数

```python
def mxfp4_to_bf16_bits(val: int, pos: int) -> int:
    """
    MXFP4 → BF16 位转换
    
    MXFP4 格式: s1e2m1 (与标准 FP4 相同)
    BF16 格式: s1e8m7
    
    参数:
        val: uint8 值，包含 2 个 packed MXFP4
        pos: 位置索引（0 或 1）
    
    返回:
        uint16 值，表示 BF16 的位模式
    """
    f4 = (val >> (pos * 4)) & 0xF
    
    s = f4 >> 3                  # 符号位 (bit 3)
    e_f4 = (f4 >> 1) & 0x3       # 指数位 (bits 1-2)
    m_f4 = f4 & 1                # 尾数位 (bit 0)
    
    # BF16 exponent: e_f4 的 bias 是 1, BF16 的 bias 是 127
    # e_f4 - 1 = e_bf16 - 127, 所以 e_bf16 = e_f4 + 126
    e_bf16 = e_f4 + 126
    
    # BF16 mantissa: 将 FP4 的1位尾数扩展到7位
    m_bf16 = m_f4 << 6
    
    bf16_bits = (s << 15) | (e_bf16 << 7) | m_bf16
    return bf16_bits


def torch_unpack_mxfp4_to_bf16(B_packed: torch.Tensor, Scale: torch.Tensor) -> torch.Tensor:
    """
    PyTorch 实现：MXFP4 → BF16 反量化 + Scale 应用
    
    参数:
        B_packed: (N, K // 2) uint8
        Scale: (N, K // 32) uint8
    
    返回:
        (N, K) bfloat16
    """
    N, K_packed = B_packed.shape
    K = K_packed * 2
    
    result_bits = np.empty((N, K), dtype=np.uint16)
    B_packed_np = B_packed.numpy()
    Scale_np = Scale.numpy()
    
    for i in range(N):
        for j in range(K):
            val = B_packed_np[i, j // 2]
            pos = j % 2
            bf16_bits = mxfp4_to_bf16_bits(val, pos)
            
            # Scale 应用
            scale_idx = j // 32
            scale_val = Scale_np[i, scale_idx]
            scale_factor = 2 ** scale_val
            
            # 转换为 float 再乘 scale
            bf16_float = np.frombuffer(bf16_bits.astype(np.uint16).tobytes(), dtype=np.float16)[0]
            result_bits[i, j] = np.frombuffer((bf16_float * scale_factor).astype(np.float16).tobytes(), dtype=np.uint16)[0]
    
    result = torch.from_numpy(result_bits.view(np.float16)).to(torch.bfloat16)
    return result


def golden_dequant_gemm_mxfp4(
    A: torch.Tensor,
    B_packed: torch.Tensor,
    Scale: torch.Tensor,
    Bias: torch.Tensor = None,
    output_dtype: str = "bfloat16"
):
    """
    PyTorch 参考实现
    
    参数:
        A: (M, K) bfloat16
        B_packed: (N, K // 2) uint8
        Scale: (N, K // 32) uint8
        Bias: (M, N) bfloat16 (可选)
        output_dtype: bfloat16 / float16
    
    返回:
        (M, N) output_dtype
    """
    # MXFP4 → BF16
    B_dequant = torch_unpack_mxfp4_to_bf16(B_packed, Scale)  # (N, K)
    
    # 矩阵乘法（FP32 计算精度）
    A_fp32 = A.to(torch.float32)
    B_fp32 = B_dequant.to(torch.float32)
    C_fp32 = torch.matmul(A_fp32, B_fp32.T)  # (M, N)
    
    # Bias 加法（可选）
    if Bias is not None:
        C_fp32 = C_fp32 + Bias.to(torch.float32)
    
    # 输出类型转换
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16}
    C_out = C_fp32.to(dtype_map[output_dtype])
    
    return C_out
```

### 8.2 测试用例

| 用例名 | 级别 | Shape (M, N, K) | dtype | scale_size | with_bias | 说明 |
|--------|------|----------------|-------|------------|-----------|------|
| basic_small | Level 0 | (128, 128, 128) | bfloat16 | 32 | False | 最小功能验证 |
| basic_with_bias | Level 0 | (128, 128, 128) | bfloat16 | 32 | True | Bias 功能验证 |
| typical_256 | Level 1 | (256, 256, 256) | bfloat16 | 32 | False | 典型配置 |
| typical_512 | Level 1 | (512, 512, 512) | bfloat16 | 32 | True | 中等规模 + Bias |
| large_1k | Level 3 | (1024, 1024, 1024) | bfloat16 | 32 | False | 大规模性能测试 |

### 8.3 精度标准

| dtype | atol | rtol |
|-------|------|------|
| bfloat16 | 1e-2 | 1e-2 |
| float16 | 1e-2 | 1e-2 |
| float32 | 1e-4 | 1e-4 |

---

## 9. 险点与注意事项

### 9.1 已知约束

1. **Ascend NPU 不支持 BF16 矩阵乘法**：
   - Cube 核不支持 BF16 输入
   - 替代方案：改为 FP16 输入，或使用 FP32 软件模拟（性能下降）

2. **MXFP4 反量化 intrinsic 未实现**：
   - `_tir_u8_to_f4_to_bf16` 需要在 `tilelang/language/` 中新增
   - 或使用 Python 端反量化（性能下降）

3. **Scale 应用需要 Vector 核**：
   - Scale 应用需要在 Vector 核执行（位移或乘法）
   - 与 Cube 核 GEMM 需要核间同步

4. **UB 内存限制**：
   - with_bias=True 时 UB 可能超限（136.5KB > 128KB）
   - 需要优化 Bias 初始化方式

### 9.2 常见错误

| 错误 | 触发场景 | 影响 | 解决方案 |
|------|----------|------|----------|
| UB 溢出 | with_bias=True, block size 过大 | 编译失败 | Bias 直接初始化到 L0C |
| Scale 累引错误 | block_N % scale_size != 0 | 精度错误 | 确保 block_N % 32 == 0 |
| BF16 GEMM 失败 | Cube 核不支持 BF16 | 编译失败 | 改为 FP16 输入 |
| 同步缺失 | 关闭 AUTO_CV_SYNC | 运行时错误 | 开启 AUTO_CV_SYNC |

### 9.3 特殊场景处理

- **非整除分块**: 暂不处理，要求输入 shape 满足整除条件
- **极小 shape**: 可能导致 block 数量过少，性能不佳
- **混合精度**: 输入 BF16，输出 BF16，中间 FP32 累加

---

## 10. 交付清单

### 10.1 目录结构

```
examples/dequantize_gemm_mxfp4/
├── design.md                          # 本设计文档
├── example_dequant_gemm_bf16_mxfp4.py # 算子实现 + 测试
└── README.md                          # 使用说明（可选）
```

### 10.2 文件清单

| 文件 | 状态 | 说明 |
|------|------|------|
| `design.md` | ✅ 已完成 | 设计文档 |
| `example_dequant_gemm_bf16_mxfp4.py` | ⬜ 待实现 | 算子实现 + 测试 |
| `test_dequant_gemm_mxfp4.py` | ⬜ 待实现 | 测试文件（可选） |

### 10.3 命名规范

- 目录名: `dequantize_gemm_mxfp4`（snake_case）
- 实现文件: `example_dequant_gemm_bf16_mxfp4.py`
- 测试文件: `test_dequant_gemm_mxfp4.py`（可选）

### 10.4 实现顺序

1. ✅ 设计文档（design.md）
2. ⬜ Golden 函数（验证基准）
3. ⬜ MXFP4 反量化 intrinsic（如需 NPU 端实现）
4. ⬜ 算子实现（example_dequant_gemm_bf16_mxfp4.py）
5. ⬜ 基础测试（Level 0 + Level 1）
6. ⬜ 边界测试（Level 2）
7. ⬜ 性能测试（Level 3，可选）