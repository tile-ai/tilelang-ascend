# dequantize_gemm (FP16-FP4) 算子设计文档

## 1. 概述

### 1.1 算子名称

dequantize_gemm_fp16_fp4

### 1.2 功能描述

FP4反量化矩阵乘法算子，将UINT8打包的FP4权重反量化为FP16后，与FP16激活矩阵执行矩阵乘法，输出FP16精度的转置结果。

### 1.3 数学公式

$$
Ct[i,j] = \sum_{k=0}^{K-1} Dequant(B[i,k]) \times A^T[j,k]
$$

其中:
- $A$: (M, K) FP16 激活矩阵
- $B$: (N, K/2) UINT8 packed FP4 权重矩阵
- $Ct$: (N, M) FP16 输出矩阵（转置形式）

### 1.4 FP4 → FP16 反量化公式

FP4 格式: s1e2m1 (1位符号 + 2位指数 + 1位尾数)
FP16 格式: s1e5m10 (1位符号 + 5位指数 + 10位尾数)

$$
\begin{aligned}
f4 &= (val >> (pos \times 4)) \& 0xF \\
s &= f4 >> 3 \\
e_{f4} &= (f4 \& 6) >> 1 \\
e_{f16} &= e_{f4} + 14 \\
m_{f4} &= f4 \& 1 \\
fp16 &= (s << 15) | (e_{f16} << 10) | (m_{f4} << 9)
\end{aligned}
$$

### 1.5 数据流图

```
输入A (M, K) FP16 → [矩阵分块搬运] → L1 → L0A
输入B (N, K/2) UINT8 → [FP4反量化] → FP16 → L1 → L0B
L0A + L0B → [GEMM累加FP32] → L0C → UB → 输出Ct (N, M) FP16
```

---

## 2. 编程模式选型

### 2.1 模式结论

**选定模式**: Developer 模式（混合模式）

### 2.2 选型理由

1. **计算类型**: 混合算子（FP4反量化 Vector核 + GEMM Cube核）
2. **内存层级**: 需要GM→L1→L0数据搬运，但编译器可自动分离
3. **同步需求**: 核间流水线需要同步，但AUTO_CV_SYNC可自动处理
4. **现有参考**: `examples/dequantize_gemm/example_dequant_gemm_bf16_fp4.py` 使用Developer模式

### 2.3 模式影响

| 维度 | 本算子的选择 |
|------|-------------|
| 内存分配 | `T.alloc_shared` + `T.alloc_fragment` (编译器自动映射) |
| 计算方式 | FP4反量化Python端执行 + `T.gemm_v0` Cube核计算 |
| 作用域 | 编译器自动分离 Cube/Vector |
| 同步方式 | `pass_configs` 自动同步 |

---

## 3. API 映射设计

### 3.1 公式拆解

| 步骤 | 数学表达 | 说明 |
|------|----------|------|
| 1 | FP4 → FP16 反量化 | Python端执行，将UINT8解包为FP16 |
| 2 | B^T 转置 | Python端执行，将(N, K)转为(K, N) |
| 3 | C = A @ B^T | NPU端GEMM计算，FP32累加 |
| 4 | Ct = C^T | 输出为(N, M)转置形式 |

### 3.2 TileLang API 映射

| 步骤 | 数学表达 | TileLang API | 参数 | 模式 |
|------|----------|-------------|------|------|
| 1 | FP4→FP16 | Python函数 `torch_unpack_fp4_to_fp16` | val, pos | Host |
| 2 | 数据搬运 | `T.copy(A[...], A_L1)` | GM→L1 | Developer |
| 3 | 数据搬运 | `T.copy(B[...], B_L1)` | GM→L1 | Developer |
| 4 | GEMM计算 | `T.gemm_v0(A_L1, B_L1, C_L0, init=...)` | L1→L0 | Developer |
| 5 | 结果搬出 | `T.copy(C_L0, C[...])` | L0→GM | Developer |

### 3.3 计算伪代码

```python
@tilelang.jit(out_idx=[-1], pass_configs={
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
})
def dequant_gemm_fp16(M, N, K, block_M, block_N, block_K):
    dtype = "float16"
    accum_dtype = "float"
    
    @T.prim_func
    def main(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        m_num = M // block_M
        n_num = N // block_N
        
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, _):
            bx = cid // n_num
            by = cid % n_num
            
            A_L1 = T.alloc_shared((block_M, block_K), dtype)
            B_L1 = T.alloc_shared((block_K, block_N), dtype)
            C_L0 = T.alloc_fragment((block_M, block_N), accum_dtype)
            
            for k in T.serial(K // block_K):
                T.copy(A[bx * block_M, k * block_K], A_L1)
                T.copy(B[k * block_K, by * block_N], B_L1)
                T.gemm_v0(A_L1, B_L1, C_L0, init=(k == 0))
            
            T.copy(C_L0, C[bx * block_M, by * block_N])
    
    return main
```

### 3.4 API 可行性确认

| API | 来源 | 验证状态 |
|-----|------|---------|
| `T.gemm_v0` | [api-compute.md](.agents/skills/tilelang-custom-skill/tilelang-api-best-practices/references/api-compute.md) | ✅ 已验证 |
| `T.alloc_shared` | [api-kernel-memory.md](.agents/skills/tilelang-custom-skill/tilelang-api-best-practices/references/api-kernel-memory.md) | ✅ 已验证 |
| `T.alloc_fragment` | [api-kernel-memory.md](.agents/skills/tilelang-custom-skill/tilelang-api-best-practices/references/api-kernel-memory.md) | ✅ 已验证 |
| `T.copy` | [api-kernel-memory.md](.agents/skills/tilelang-custom-skill/tilelang-api-best-practices/references/api-kernel-memory.md) | ✅ 已验证 |
| `T.serial` | [api-schedule-sync.md](.agents/skills/tilelang-custom-skill/tilelang-api-best-practices/references/api-schedule-sync.md) | ✅ 已验证 |

---

## 4. 数据规格与内存规划

### 4.1 输入张量

| 参数名 | Shape | dtype | 说明 |
|--------|-------|-------|------|
| A | (M, K) | float16 | 激活矩阵，FP16精度 |
| B_packed | (N, K/2) | uint8 | 权重矩阵，packed FP4 |
| B | (K, N) | float16 | 反量化后的权重矩阵（Python端生成） |

### 4.2 输出张量

| 参数名 | Shape | dtype | 说明 |
|--------|-------|-------|------|
| Ct | (N, M) | float16 | 输出矩阵，转置形式 |

### 4.3 中间缓冲区

| Buffer 名 | Shape | dtype | 存储层级 | 用途 |
|-----------|-------|-------|----------|------|
| A_L1 | (block_M, block_K) | float16 | L1 | 激活矩阵分块缓冲 |
| B_L1 | (block_K, block_N) | float16 | L1 | 权重矩阵分块缓冲 |
| C_L0 | (block_M, block_N) | float | L0C | GEMM累加器(FP32) |

### 4.4 内存搬运路径

```
GM[A] --T.copy--> L1[A_L1] --T.copy--> L0A (gemm_v0内部)
GM[B] --T.copy--> L1[B_L1] --T.copy--> L0B (gemm_v0内部)
L0A + L0B --T.gemm_v0--> L0C[C_L0] --T.copy--> GM[C]
```

### 4.5 UB 内存预算

| Buffer | Shape | dtype | 大小 (Bytes) |
|--------|-------|-------|-------------|
| A_L1 | (128, 128) | float16 | 32768 |
| B_L1 | (128, 128) | float16 | 32768 |
| C_L0 | (128, 128) | float32 | 65536 |
| **总计** | | | 131072 / 131072 (128KB) |

### 4.6 动态轴定义

| 动态轴 | 声明方式 | 运行时范围 |
|--------|----------|-----------|
| M | 参数传递 | 128 ~ 8192 |
| N | 参数传递 | 128 ~ 8192 |
| K | 参数传递 | 128 ~ 8192 (需偶数) |

### 4.7 JIT 配置

```python
@tilelang.jit(
    out_idx=[-1],
    pass_configs={
        tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
        tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
        tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
        tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
    },
)
```

---

## 5. Tiling 策略

### 5.1 计算类型

**类型**: 混合（Vector反量化 + Cube GEMM）

**判定依据**: 包含FP4→FP16反量化（Vector核可实现）和矩阵乘法（Cube核），判定为混合算子。

### 5.2 Block 划分

```python
block_M = 128  # 典型值，适配L0C容量
block_N = 128  # 典型值，适配L0C容量
block_K = 128  # 典型值，平衡搬运次数和计算量
block_num = (M // block_M) * (N // block_N)
```

### 5.3 约束分析

- **对齐约束**: K必须为偶数（FP4 packed format，每个UINT8包含2个FP4）
- **UB容量**: 总buffer ≈ 128KB，满足限制 ✓
- **L0容量**: L0C (128, 128) float32 = 64KB，满足限制 ✓
- **整除约束**: M % block_M == 0, N % block_N == 0, K % block_K == 0

### 5.4 注意事项

- 非整除情况暂不支持，需外部pad或使用尾块处理版本
- Split-K优化可提升大K场景的并行度（可选扩展）

---

## 6. 循环与调度结构

### 6.1 循环结构总结

| 维度 | 循环类型 | API | 理由 |
|------|----------|-----|------|
| Block级 | 并行 | `T.Kernel(m_num * n_num)` | 每个block处理一个输出tile |
| K方向 | 串行迭代 | `T.serial(K // block_K)` | K维分块迭代累加 |
| 元素级 | 向量化 | `gemm_v0内部` | Cube核批量计算 |

### 6.2 循环伪代码

```python
with T.Kernel(m_num * n_num, is_npu=True) as (cid, _):
    # Block内循环结构
    for k in T.serial(K // block_K):
        # 搬入A和B的分块
        T.copy(A[bx * block_M, k * block_K], A_L1)
        T.copy(B[k * block_K, by * block_N], B_L1)
        # GEMM累加
        T.gemm_v0(A_L1, B_L1, C_L0, init=(k == 0))
    # 搬出结果
    T.copy(C_L0, C[bx * block_M, by * block_N])
```

### 6.3 流水线优化

暂不使用 `T.Pipelined`。后续可扩展支持：
- `num_stages=2`: 双缓冲A_L1和B_L1
- 流水线搬运与计算重叠

### 6.4 尾块处理

当前版本假设M、N、K可被block size整除。非整除场景可参考:
- `examples/gemm/example_gemm_tail_block.py`

---

## 7. 同步策略

### 7.1 同步模式

**模式**: 自动同步

### 7.2 同步点说明

通过 `pass_configs` 自动插入同步：
- `TL_ASCEND_AUTO_SYNC`: 核内同步（copy后、gemm前后）
- `TL_ASCEND_AUTO_CV_SYNC`: 核间同步（Cube/Vector协作）

### 7.3 pass_configs 配置

```python
pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
}
```

---

## 8. 验证方案

### 8.1 Golden 函数

```python
def fp4_to_fp16_bits(val: int, pos: int) -> int:
    """FP4 → FP16 位转换"""
    f4 = (val >> (pos * 4)) & 0xF
    s = f4 >> 3
    e_f4 = (f4 >> 1) & 0x3
    m_f4 = f4 & 1
    e_fp16 = e_f4 + 14
    m_fp16 = m_f4 << 9
    fp16_bits = (s << 15) | (e_fp16 << 10) | m_fp16
    return fp16_bits

def torch_unpack_fp4_to_fp16(tensor: torch.Tensor) -> torch.Tensor:
    """PyTorch实现：UINT8 packed FP4 → FP16"""
    assert tensor.dtype == torch.uint8
    N, K_packed = tensor.shape
    K = K_packed * 2
    result_bits = np.empty((N, K), dtype=np.uint16)
    tensor_np = tensor.numpy()
    for i in range(N):
        for j in range(K):
            val = tensor_np[i, j // 2]
            pos = j % 2
            result_bits[i, j] = fp4_to_fp16_bits(val, pos)
    result = torch.from_numpy(result_bits.view(np.float16)).to(torch.float16)
    return result

def ref_program(A: torch.Tensor, B_packed: torch.Tensor, output_dtype: str):
    """PyTorch参考实现"""
    B_fp16 = torch_unpack_fp4_to_fp16(B_packed)
    A_fp32 = A.to(torch.float32)
    B_fp32 = B_fp16.to(torch.float32)
    C_fp32 = torch.matmul(A_fp32, B_fp32.T)
    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16}
    C_out = C_fp32.to(dtype_map[output_dtype])
    return C_out
```

### 8.2 测试用例

| 用例名 | 级别 | Shape | dtype | 说明 |
|--------|------|-------|-------|------|
| basic_small | Level 0 | (128, 128, 128) | float16 | 最小功能验证 |
| typical_256 | Level 1 | (256, 256, 256) | float16 | 典型配置 |
| typical_512 | Level 1 | (512, 512, 512) | float16 | 中等规模 |
| large_1024 | Level 3 | (1024, 1024, 1024) | float16 | 性能测试 |

### 8.3 精度标准

| dtype | atol | rtol |
|-------|------|------|
| float16 | 1e-2 | 1e-2 |
| float32 | 1e-4 | 1e-4 |
| bfloat16 | 1e-2 | 1e-2 |

---

## 9. 风险点与注意事项

### 9.1 已知约束

1. K必须为偶数（FP4 packed format）
2. M、N、K必须能被block_M、block_N、block_K整除
3. FP4→FP16反量化当前在Python端执行，非NPU端

### 9.2 常见错误

| 错误 | 触发场景 | 影响 | 解决方案 |
|------|----------|------|----------|
| UB溢出 | block过大 | 编译失败 | 减小block_M/N/K |
| K非偶数 | FP4格式要求 | 运行时错误 | 外部pad或截断 |
| 精度误差 | FP4→FP16转换 | 数值偏差 | 使用FP32累加 |

### 9.3 特殊场景处理

- **非整除分块**: 使用尾块处理版本
- **极小shape**: 当M/N/K < block时，需特殊处理
- **大K场景**: 可启用Split-K优化（待扩展）

---

## 10. 交付清单

### 10.1 目录结构

```
examples/dequantize_gemm/
├── design_fp16_fp4.md     # 本设计文档
├── example_dequant_gemm_fp16_fp4.py  # 算子实现 + 测试
└── README.md              # 使用说明（可选）
```

### 10.2 文件清单

| 文件 | 状态 | 说明 |
|------|------|------|
| `design_fp16_fp4.md` | ✅ 已完成 | 设计文档 |
| `example_dequant_gemm_fp16_fp4.py` | ⬜ 待实现 | 算子实现 |
| 测试用例 | ⬜ 待实现 | 集成在主文件中 |

### 10.3 命名规范

- 目录名: `dequantize_gemm`
- 实现文件: `example_dequant_gemm_fp16_fp4.py`
- 设计文档: `design_fp16_fp4.md`

### 10.4 实现顺序

1. ✅ 设计文档（design_fp16_fp4.md）
2. ⬜ Golden 函数（验证基准）
3. ⬜ 算子实现（example_dequant_gemm_fp16_fp4.py）
4. ⬜ 基础测试（Level 0 + Level 1）
5. ⬜ 边界测试（Level 2）
6. ⬜ 性能测试（Level 3）