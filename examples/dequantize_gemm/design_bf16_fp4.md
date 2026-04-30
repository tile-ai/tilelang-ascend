# dequantize_gemm 算子设计文档

## 1. 概述

### 1.1 算子名称

dequantize_gemm

### 1.2 功能描述

FP4 反量化矩阵乘法算子，将 packed FP4 量化权重反量化为 FP16 后执行矩阵乘法运算，支持 BF16 输入和 BF16 输出。

### 1.3 数学公式

$$
C[i,j] = \sum_k A[i,k] \times \text{Dequant}(B^T[j,k])
$$

其中，$\text{Dequant}$ 函数将 UINT8 中打包的 FP4 数据解包为 FP16：

FP4 格式：**s1e2m1**（1位符号 + 2位指数 + 1位尾数）
FP16 格式：**s1e5m10**（1位符号 + 5位指数 + 10位尾数）

$$
\begin{aligned}
f4 &= (val >> (pos \times 4)) \& 0xF \\
s &= f4 >> 3 \\
e_{f4} &= (f4 \& 6) >> 1 \\
e_{fp16} &= e_{f4} + 14 \\
m_{f4} &= f4 \& 1 \\
fp16 &= (s << 15) | (e_{fp16} << 10) | (m_{f4} << 9)
\end{aligned}
$$

### 1.4 算法描述

1. **FP4 解包（Python 端）**：将 UINT8 中打包的 FP4 数据（每字节存储 2 个 FP4 值）解包为 FP16
2. **BF16 → FP16 转换（Python 端）**：将输入 A 从 BF16 转换为 FP16（Ascend Cube 核不支持 BF16 矩阵乘）
3. **FP16×FP16 矩阵乘法（NPU 端）**：执行 $C = A @ B.T$，累加类型为 FP32
4. **类型转换（Python 端）**：将 FP32 累加结果转换为 BF16 输出

### 1.5 数据流图

```
Python 端：
输入A (BF16, M×K) → FP16 转换 → A_fp16 (FP16, M×K)
输入B (UINT8, N×K/2) → FP4 解包 → B_fp16 (FP16, N×K)

NPU 端：
A_fp16 (FP16, M×K) → L1 → L0A → Cube核计算
B_fp16 (FP16, N×K) → L1 → L0B → Cube核计算
L0C (FP32累加, M×N) → GM

Python 端：
输出C (FP32, M×N) → FP32→BF16 转换 → 输出C (BF16, M×N)
```

---

## 2. 编程模式选型

### 2.1 模式结论

**选定模式**: Developer 模式（简化方案）

### 2.2 选型理由

- **实际实现**：
  - FP4 解包在 Python 端完成（FP4→FP16 位操作复杂）
  - BF16→FP16 转换在 Python 端完成
  - NPU 端只执行 FP16×FP16 矩阵乘法
  - FP32→BF16 转换在 Python 端完成
  
- **简化方案的优势**：
  - Python 端处理精度转换逻辑清晰
  - NPU 端只做 GEMM，避免复杂的核间协作
  - FP4→FP16 位操作在 Python 中易于调试和维护

### 2.3 模式影响

| 维度 | 本算子的选择 |
|------|-------------|
| 内存分配 | Developer: `T.alloc_shared/fragment` |
| 计算 | Developer: `T.gemm_v0` 矩阵乘法 |
| 作用域 | Developer: 编译器自动管理 |
| FP4 解包 | Python 端处理 |
| 精度转换 | Python 端处理 |

---

## 3. API 映射设计

### 3.1 公式拆解

| 步骤 | 数学表达 | 说明 | 执行位置 |
|------|----------|------|---------|
| 1 | $B_{int8} = \text{unpack}(B_{packed})$ | INT4 → INT8 解包 | Python 端 |
| 2 | $C_{int32} = B_{int8} @ A.T$ | INT8×INT8 矩阵乘 | NPU 端 |
| 3 | $C_{bf16} = \text{cast}(C_{int32})$ | INT32 → BF16 转换 | Python 端 |

### 3.2 TileLang API 映射

| 步骤 | 数学表达 | TileLang API | 参数 | 模式 |
|------|----------|-------------|------|------|
| 1 | INT4解包 | Python 函数 `torch_unpack_int4` | `tensor, num_bits=4` | Python |
| 2 | 数据搬入 | `T.copy(A[...], A_L1)` | - | Expert |
| 2 | 数据搬入 | `T.copy(B[...], B_L1)` | - | Expert |
| 2 | 矩阵乘 | `T.mma(A_L0, B_L0, C_L0, init)` | `init=(k==0)` | Expert |
| 2 | 数据搬出 | `T.copy(C_L0, C[...])` | - | Expert |
| 3 | 类型转换 | Python `.to(torch.bfloat16)` | - | Python |

### 3.3 计算伪代码

```python
# Python 端：INT4 解包
def torch_unpack_int4(B_packed: torch.Tensor) -> torch.Tensor:
    """将 UINT8 打包的 INT4 解包为 INT8"""
    for i in range(N):
        for j in range(K):
            val = B_packed[i, j // 2].item()
            nibble = (val >> (j % 2 * 4)) & 0xF
            if nibble >= 8:
                nibble -= 16
            B[i, j] = nibble
    return B

# NPU 端：INT8×INT8 GEMM
@T.prim_func
def main(A: T.Tensor((M, K), "int8"), B: T.Tensor((N, K), "int8"), C: T.Tensor((N, M), "int32")):
    with T.Kernel(n_num * m_num, is_npu=True) as (cid, _):
        bx = cid // m_num
        by = cid % m_num
        
        A_L1 = T.alloc_L1((block_M, block_K), "int8")
        B_L1 = T.alloc_L1((block_N, block_K), "int8")
        A_L0 = T.alloc_L0A((block_N, block_K), "int8")
        B_L0 = T.alloc_L0B((block_K, block_M), "int8")
        C_L0 = T.alloc_L0C((block_N, block_M), "int32")
        
        with T.Scope("C"):
            for k in T.serial(k_num):
                T.copy(A[by * block_M, k * block_K], A_L1)
                T.copy(B[bx * block_N, k * block_K], B_L1)
                T.barrier_all()
                
                T.copy(B_L1, A_L0)
                T.copy(A_L1, B_L0, transpose=True)
                T.barrier_all()
                
                T.mma(A_L0, B_L0, C_L0, init=(k == 0))
                T.barrier_all()
            
            T.copy(C_L0, C[bx * block_N, by * block_M])

# Python 端：类型转换
C_int = kernel(A.npu(), B.npu())
C_out = C_int.cpu().to(torch.bfloat16)
```

### 3.4 API 可行性确认

| API | 来源 | 状态 |
|-----|------|------|
| `T.copy` | api-kernel-memory.md | ✅ 已验证 |
| `T.Parallel` + 符号运算 | api-compute.md | ✅ 已验证 |
| `T.mma` | api-compute.md | ✅ 已验证 |
| `T.alloc_L1/L0A/L0B/L0C/ub` | api-kernel-memory.md | ✅ 已验证 |
| `T.tile.cast` | api-compute.md | ✅ 已验证 |
| `T.barrier_all` | api-schedule-sync.md | ✅ 已验证 |
| `T.Scope("C"/"V")` | examples/pipeline/gemm_v0_pipeline.py | ✅ 已验证 |
| `_tir_u8_to_i4_to_i8` | examples/old_gemm_test/example_dequant_gemm_correct.py | ✅ 已验证 |

---

## 4. 数据规格与内存规划

### 4.1 输入张量

| 参数名 | Shape | dtype | 说明 |
|--------|-------|-------|------|
| A | (M, K) | bfloat16 / float16 | 激活矩阵 |
| B | (N, K//2) | uint8 | 权重矩阵，packed FP4，每字节存 2 个 FP4 值 |

### 4.2 输出张量

| 参数名 | Shape | dtype | 说明 |
|--------|-------|-------|------|
| C | (M, N) | bfloat16 | 输出矩阵（注意：[M, N] 不是 [N, M]） |

### 4.3 中间缓冲区（NPU 端）

| Buffer 名 | Shape | dtype | 存储层级 | 用途 |
|-----------|-------|-------|----------|------|
| A_L1 | (block_M, block_K) | float16 | L1 | A 矩阵 tile 缓存（转换为 FP16） |
| B_L1 | (block_K, block_N) | float16 | L1 | B 矩阵 tile 缓存（FP4→FP16 后转置） |
| C_L0 | (block_M, block_N) | float | L0C | Cube 输出累加（FP32） |

### 4.4 内存搬运路径

```
Python 端：
GM[A_bf16] --.to(float16)--> A_fp16
GM[B_packed] --fp4_to_fp16--> B_fp16 --.T--> B_fp16_T

NPU 端：
GM[A_fp16] --T.copy--> L1[A_L1] --T.copy--> L0A
GM[B_fp16_T] --T.copy--> L1[B_L1] --T.copy--> L0B
L0A + L0B --T.gemm_v0--> L0C[C_L0] --T.copy--> GM[C_fp32]

Python 端：
GM[C_fp32] --.to(bfloat16)--> GM[C_bf16]
```

### 4.5 UB 内存预算

不使用 UB（INT4 解包和类型转换在 Python 端完成）

### 4.6 L1 内存预算（block_N=128, block_M=128, block_K=128）

| Buffer | Shape | dtype | 大小 (Bytes) |
|--------|-------|-------|-------------|
| A_L1 | (128, 128) | int8 | 16,384 |
| B_L1 | (128, 128) | int8 | 16,384 |
| **总计** | | | **32,768** / 1,048,576 (1MB L1) ✅ |

### 4.7 L0 内存预算

| Buffer | Shape | dtype | 大小 (Bytes) |
|--------|-------|-------|-------------|
| A_L0 | (128, 128) | int8 | 16,384 |
| B_L0 | (128, 128) | int8 | 16,384 |
| C_L0 | (128, 128) | int32 | 65,536 |
| **总计** | | | **98,304** / ~512KB (L0A+L0B+L0C) ✅ |

### 4.8 动态轴定义

无（静态 shape）

### 4.9 JIT 配置

```python
@tilelang.jit(out_idx=[2])
```

---

## 5. Tiling 策略

### 5.1 计算类型

**类型**: 纯 Cube（INT8×INT8 矩阵乘）

**判定依据**: INT4 解包和类型转换在 Python 端完成，NPU 端只执行矩阵乘法

### 5.2 Block 划分

```python
block_M = 128  # M 方向分块，适配 L0C 容量
block_N = 128  # N 方向分块，适配 L0C 容量
block_K = 128  # K 方向分块，适配 L1/L0 容量

m_num = M // block_M
n_num = N // block_N
k_num = K // block_K

block_num = n_num * m_num
```

### 5.3 约束分析

- **对齐约束**: K 必须为偶数（INT4 packed），block_K 为 128 ✅
- **UB 容量**: 总 buffer = 122KB < 256KB ✅
- **L1 容量**: 总 buffer = 41KB < 1MB ✅
- **L0 容量**: 总 buffer = 98KB < ~512KB ✅

### 5.4 注意事项

- INT4 packed 格式要求 K 为偶数
- 非整除情况需要边界块处理（当前实现暂不支持）
- 可通过调整 block_M/block_N/block_K 优化性能

---

## 6. 循环与调度结构

### 6.1 循环结构总结

| 维度 | 循环类型 | API | 理由 |
|------|----------|-----|------|
| Block 级 | 并行 | `T.Kernel` | 每个 block 处理一个输出 tile |
| K 方向 | 迭代 | `T.serial(k_num)` | K 维分块迭代累加 |
| 解包 | 向量化 | `T.Parallel(block_N, block_K)` | INT4→INT8 解包并行 |

### 6.2 循环伪代码

```python
# Block 级并行
with T.Kernel(n_num * m_num, is_npu=True) as (cid, _):
    bx = cid // m_num  # N 方向 block ID
    by = cid % m_num  # M 方向 block ID
    
    # Cube 核计算循环
    with T.Scope("C"):
        for k in T.serial(k_num):  # K 方向迭代
            # 数据搬运 + 解包 + 矩阵乘
            ...
    
    # Vector 核后处理
    with T.Scope("V"):
        # 类型转换 + 输出
        ...
```

### 6.3 流水线优化

暂不使用 `T.Pipelined`，采用简单串行结构。可考虑：
- K 方向流水线：搬运下一块数据时计算当前块
- 需要额外的 buffer 用于流水线 stage

### 6.4 尾块处理

当前实现暂不支持非整除情况。后续可扩展：
- 检测边界块，使用不同的 block size
- 或使用 mask 操作处理不规则 tile

---

## 7. 同步策略

### 7.1 同步模式

**模式**: 手动同步（Expert 模式）

### 7.2 同步点说明

| 位置 | 同步 API | 理由 |
|------|----------|------|
| 数据搬入后 | `T.barrier_all()` | 等待 DMA 搬运完成 |
| L1→L0 搬运后 | `T.barrier_all()` | 等待数据进入 L0 |
| T.mma 后 | `T.barrier_all()` | 等待 Cube 核计算完成 |

### 7.3 pass_configs 配置

```python
pass_configs = {}  # 不使用自动同步，手动控制
```

---

## 8. 验证方案

### 8.1 Golden 函数

```python
def torch_unpack_int4(tensor: torch.Tensor, num_bits: int = 4) -> torch.Tensor:
    """
    PyTorch 参考实现：将 UINT8 打包的 INT4 数据解包为 INT8
    """
    assert tensor.dtype == torch.uint8
    num_elems_per_byte = 8 // num_bits
    N, K_packed = tensor.shape
    K = K_packed * num_elems_per_byte

    def _unpack(val, pos):
        mask = (1 << num_bits) - 1
        i4_shifted = (val >> (pos * num_bits)) & mask
        i4 = (i4_shifted << 4) >> 4
        return i4.view(torch.int8)

    result = torch.empty(N, K, dtype=torch.int8, device=tensor.device)
    for i in range(N):
        for j in range(K):
            result[i, j] = _unpack(tensor[i, j // num_elems_per_byte], j % num_elems_per_byte)

    return result


def golden_dequantize_gemm(A: torch.Tensor, B_packed: torch.Tensor):
    """
    PyTorch Golden 实现
    A: (M, K), int8
    B_packed: (N, K//2), uint8
    输出: (N, M), bfloat16
    """
    B = torch_unpack_int4(B_packed, num_bits=4)
    C_int = torch.matmul(B.cpu().to(torch.int32), A.cpu().T.to(torch.int32))
    C_out = C_int.to(torch.bfloat16)
    return C_out
```

### 8.2 测试用例

| 用例名 | 级别 | Shape | dtype | 说明 |
|--------|------|-------|-------|------|
| basic_small | Level 0 | M=128, N=128, K=128 | int8→bfloat16 | 最小功能验证 |
| typical_1 | Level 1 | M=256, N=256, K=256 | int8→bfloat16 | 典型配置 |
| typical_2 | Level 1 | M=1024, N=512, K=512 | int8→bfloat16 | 大规模验证 |
| bf16_output | Level 1 | M=512, N=512, K=512 | int8→bfloat16 | BF16 输出验证 |

### 8.3 精度标准

| dtype | atol | rtol |
|-------|------|------|
| bfloat16 | 1e-2 | 1e-2 |

---

## 9. 风险点与注意事项

### 9.1 已知约束

- K 必须为偶数（INT4 packed 格式）
- M、N、K 必须能被对应的 block size 整除
- 当前不支持动态 shape
- INT4 解包在 Python 端完成，可能增加主机端开销

### 9.2 常见错误

| 错误 | 触发场景 | 影响 | 解决方案 |
|------|----------|------|----------|
| L0 溢出 | block size 过大 | 编译失败 | 减小 block_M/N/K |
| K 非偶数 | INT4 packed 格式要求 | 运行错误 | 确保 K 为偶数 |
| 同步缺失 | Cube 核计算 | 结果错误 | 添加 `T.barrier_all()` |
| INT4 解包错误 | 符号位扩展不正确 | 精度错误 | nibble >= 8 时减 16 |

### 9.3 特殊场景处理

- 非整除分块：暂不支持，后续可扩展 mask 操作
- 极小 shape：block size 需调整为更小值
- Python 端处理：INT4 解包和类型转换在 Python 端完成，适合中等规模数据

---

## 10. 交付清单

### 10.1 目录结构

```
examples/dequantize_gemm/
├── example_dequant_gemm.py     # 算子实现 + 简单测试
├── design.md                   # 本设计文档
└── README.md                   # 使用说明（可选）
```

### 10.2 文件清单

| 文件 | 状态 | 说明 |
|------|------|------|
| `design.md` | ✅ 已完成 | 设计文档 |
| `example_dequant_gemm.py` | ✅ 已完成 | 算子实现（Python 解包 + NPU GEMM） |
| `test_dequant_gemm.py` | ⬜ 待实现 | 测试文件（可选，放入 testing/） |

### 10.3 命名规范

- 目录名: `dequantize_gemm`（snake_case）
- 实现文件: `example_dequant_gemm.py`
- 测试文件: `test_dequant_gemm.py`

### 10.4 实现顺序

1. ✅ 设计文档（design.md）
2. ✅ Golden 函数（验证基准）
3. ✅ 算子实现（example_dequant_gemm.py）
4. ✅ 基础测试（Level 0 + Level 1）
5. ⬜ 边界测试（Level 2，可选）
6. ⬜ 性能测试（Level 3，可选）