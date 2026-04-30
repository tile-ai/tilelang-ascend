# dequantize_gemm (W4A8) 算子设计文档

## 1. 概述

### 1.1 算子名称

dequantize_gemm (W4A8)

### 1.2 功能描述

W4A8量化矩阵乘法：将INT4打包权重反量化为INT8后，执行INT8×INT8矩阵乘法，输出INT32累加结果。

### 1.3 数学公式

$$
Ct[i,j] = \sum_k B_{dequant}[i,k] \times A^T[j,k]
$$

其中 $B_{dequant}$ 由 packed INT4 反量化得到：

$$
i4 = (val >> (pos \times 4)) \& 0xF
$$

$$
i8 = (i4 << 4) >> 4  \quad \text{(符号扩展)}
$$

### 1.4 算法描述

算法分为两步：

1. **INT4反量化（Python端）**：将UINT8打包的INT4数据解包为INT8（符号扩展）
2. **INT8 GEMM（NPU端）**：执行INT8×INT8矩阵乘法，累加为INT32

### 1.5 数据流图

```
输入A (M, K) INT8 ─────────────────────┐
                                        │
输入B (N, K/2) UINT8 ──► [Python解包] ──► B_dequant (N, K) INT8
                                        │
                                        ▼
                              [NPU GEMM: B @ A.T]
                                        │
                                        ▼
                               输出Ct (N, M) INT32
```

---

## 2. 编程模式选型

### 2.1 模式结论

**选定模式**: Developer 模式（Python端反量化 + NPU端INT8 GEMM）

### 2.2 选型理由

| 特征 | 分析 |
|------|------|
| 计算类型 | 混合（反量化 + GEMM） |
| 反量化复杂度 | INT4符号扩展逻辑在TIR层面实现复杂，Python端实现更可靠 |
| GEMM部分 | 纯INT8矩阵乘法，Cube核原生支持 |
| 参考实现 | `example_dequant_gemm_mxfp4.py` 采用Python端反量化，验证可靠 |
| NPU能力 | Ascend Cube核原生支持INT8×INT8→INT32 |

**选择Developer模式的原因**：
1. Python端解包INT4逻辑清晰、易于调试
2. NPU端仅执行INT8 GEMM，编译器自动管理内存层级
3. 避免TIR层面`tir.reinterpret`在Ascend上的已知问题
4. 与MXFP4实现保持一致，便于维护

### 2.3 模式影响

| 维度 | 本算子的选择 |
|------|-------------|
| 内存分配 | `T.alloc_shared` / `T.alloc_fragment`（编译器自动映射） |
| 计算方式 | `T.gemm_v0(A_L1, B_L1, C_L0, transpose_B=True)` |
| 作用域 | 编译器自动分离Cube核计算 |
| 同步方式 | `TL_ASCEND_AUTO_SYNC: True`（自动同步） |

---

## 3. API 映射设计

### 3.1 公式拆解

| 步骤 | 数学表达 | 说明 |
|------|----------|------|
| 1 | $i4 = (val >> (pos \times 4)) \& 0xF$ | 提取4-bit nibble |
| 2 | $i8 = (i4 << 4) >> 4$ | 符号扩展到INT8 |
| 3 | $Ct = B_{dequant} @ A^T$ | INT8矩阵乘法累加为INT32 |

### 3.2 TileLang API 映射

| 步骤 | 数学表达 | TileLang API | 参数 | 模式 |
|------|----------|-------------|------|------|
| 1-2 | INT4→INT8解包 | Python函数 `torch_unpack_int4` | `B_packed, num_bits=4` | Python |
| 3 | $Ct = B @ A^T$ | `T.gemm_v0(A_L1, B_L1, C_L0, transpose_B=True, init=True)` | `transpose_B=True` | Developer |

### 3.3 计算伪代码

```python
@tilelang.jit(out_idx=[2])
def dequant_gemm(M, N, K, block_M, block_N, block_K):
    m_num = M // block_M
    n_num = N // block_N
    k_num = K // block_K

    @T.prim_func
    def main(
        A: T.Tensor((M, K), "int8"),
        B: T.Tensor((N, K), "int8"),  # Python端已解包
        C: T.Tensor((N, M), "int32"),
    ):
        with T.Kernel(n_num * m_num, is_npu=True) as (cid, _):
            bx = cid // m_num
            by = cid % m_num

            A_L1 = T.alloc_shared((block_M, block_K), "int8")
            B_L1 = T.alloc_shared((block_N, block_K), "int8")
            C_L0 = T.alloc_fragment((block_N, block_M), "int32")

            for k in T.serial(k_num):
                T.copy(A[by * block_M, k * block_K], A_L1)
                T.copy(B[bx * block_N, k * block_K], B_L1)
                T.gemm_v0(B_L1, A_L1, C_L0, transpose_B=True, init=(k == 0))

            T.copy(C_L0, C[bx * block_N, by * block_M])

    return main
```

### 3.4 API 可用性确认

| API | 来源 | 验证状态 |
|-----|------|---------|
| `T.gemm_v0` | [api-compute.md](../.agents/skills/tilelang-custom-skill/tilelang-api-best-practices/references/api-compute.md) | ✅ 验证 |
| `T.alloc_shared` | [api-kernel-memory.md](../.agents/skills/tilelang-custom-skill/tilelang-api-best-practices/references/api-kernel-memory.md) | ✅ 验证 |
| `T.alloc_fragment` | [api-kernel-memory.md](../.agents/skills/tilelang-custom-skill/tilelang-api-best-practices/references/api-kernel-memory.md) | ✅ 验证 |
| `T.copy` | [api-kernel-memory.md](../.agents/skills/tilelang-custom-skill/tilelang-api-best-practices/references/api-kernel-memory.md) | ✅ 验证 |
| INT8 GEMM | `examples/old_gemm_test/test_int8_transpose_gemm.py` | ✅ 参考实现存在 |

---

## 4. 数据规格与内存规划

### 4.1 输入张量

| 参数名 | Shape | dtype | 说明 |
|--------|-------|-------|------|
| A | (M, K) | int8 | 激活矩阵 |
| B_packed | (N, K/2) | uint8 | 权重矩阵（packed INT4，Python端输入） |
| B | (N, K) | int8 | 解包后的权重矩阵（NPU端输入） |

### 4.2 输出张量

| 参数名 | Shape | dtype | 说明 |
|--------|-------|-------|------|
| C | (N, M) | int32 | 矩阵乘法累加结果 |

### 4.3 中间缓冲区

| Buffer 名 | Shape | dtype | 存储层级 | 用途 |
|-----------|-------|-------|----------|------|
| A_L1 | (block_M, block_K) | int8 | L1 (shared) | A矩阵tile缓冲 |
| B_L1 | (block_N, block_K) | int8 | L1 (shared) | B矩阵tile缓冲 |
| C_L0 | (block_N, block_M) | int32 | L0C (fragment) | GEMM累加输出 |

### 4.4 内存搬运路径

```
GM[A] --T.copy--> L1[A_L1]
GM[B] --T.copy--> L1[B_L1]
L1[B_L1] + L1[A_L1] --T.gemm_v0(transpose_B=True)--> L0C[C_L0]
L0C[C_L0] --T.copy--> GM[C]
```

### 4.5 UB 内存预算

Developer模式下UB由编译器自动管理，无需手动计算。

### 4.6 动态轴定义

无动态轴，所有维度在编译时确定。

### 4.7 JIT 配置

```python
@tilelang.jit(
    out_idx=[2],
    pass_configs={
        tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
        tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
    },
)
```

---

## 5. Tiling 策略

### 5.1 计算类型

**类型**: 纯 Cube（NPU端仅执行GEMM）

**判定依据**: Python端完成反量化，NPU端仅执行INT8×INT8矩阵乘法

### 5.2 Block 划分

```python
block_M = 128  # M维度分块，适配L0C容量
block_N = 128  # N维度分块，适配L0C容量
block_K = 128  # K维度分块，适配L1容量
block_num = (M // block_M) * (N // block_N)
```

**选择理由**：
- `block_M × block_N = 128 × 128 = 16384` int32元素，约64KB，适配L0C
- `block_M × block_K = 128 × 128 = 16384` int8元素，约16KB，适配L1
- 与参考实现 `example_dequant_gemm.py` 保持一致

### 5.3 约束分析

- **对齐约束**: M % block_M == 0, N % block_N == 0, K % block_K == 0
- **K约束**: K % 2 == 0（INT4打包要求）
- **L0容量**: block_M × block_N × 4B ≤ 64KB ✓
- **L1容量**: (block_M + block_N) × block_K × 1B ≤ 128KB ✓

### 5.4 注意事项

- 非整除情况需使用 `T.ceildiv` 或边界处理
- INT4打包比例固定为 2:1（K/2个UINT8存储K个INT4）

---

## 6. 循环与调度结构

### 6.1 循环结构总结

| 维度 | 循环类型 | API | 理由 |
|------|----------|-----|------|
| Block级 | 并行 | `T.Kernel(n_num * m_num)` | 每个block处理一个输出tile |
| K方向 | 串行迭代 | `T.serial(k_num)` | K维度分块迭代累加 |

### 6.2 循环伪代码

```python
with T.Kernel(n_num * m_num, is_npu=True) as (cid, _):
    bx = cid // m_num  # N方向block索引
    by = cid % m_num  # M方向block索引

    for k in T.serial(k_num):
        T.copy(A[by * block_M, k * block_K], A_L1)
        T.copy(B[bx * block_N, k * block_K], B_L1)
        T.gemm_v0(B_L1, A_L1, C_L0, transpose_B=True, init=(k == 0))

    T.copy(C_L0, C[bx * block_N, by * block_M])
```

### 6.3 流水线优化

当前版本不使用 `T.Pipelined`，采用简单串行K迭代。后续可优化：
- 使用 `T.Pipelined(num_stages=2)` 实现K方向流水线
- 预取下一个k tile的数据

### 6.4 尾块处理

假设输入shape能被block size整除。非整除情况需：
- 使用 `T.ceildiv(K, block_K)` 计算实际迭代次数
- 边界块使用mask或特殊处理

---

## 7. 同步策略

### 7.1 同步模式

**模式**: 自动同步

### 7.2 同步点说明

Developer模式下启用 `TL_ASCEND_AUTO_SYNC: True`，编译器自动插入同步：
- T.copy后：等待数据搬运完成
- T.gemm_v0后：等待矩阵乘法完成

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
def torch_unpack_int4(tensor: torch.Tensor, num_bits: int = 4) -> torch.Tensor:
    """PyTorch参考实现：INT4解包为INT8"""
    assert tensor.dtype == torch.uint8
    num_elems_per_byte = 8 // num_bits
    N, K_packed = tensor.shape
    K = K_packed * num_elems_per_byte

    result = torch.empty(N, K, dtype=torch.int8)
    for i in range(N):
        for j in range(K):
            val = tensor[i, j // num_elems_per_byte].item()
            pos = j % num_elems_per_byte
            nibble = (val >> (pos * num_bits)) & 0xF
            if nibble >= 8:
                nibble -= 16
            result[i, j] = nibble
    return result

def ref_program(A: torch.Tensor, B_packed: torch.Tensor, out_dtype: str = "float16"):
    """PyTorch参考实现：完整W4A8 GEMM"""
    B = torch_unpack_int4(B_packed, num_bits=4)
    C_int = torch.matmul(B.to(torch.int32), A.T.to(torch.int32))
    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16}
    return C_int.to(dtype_map[out_dtype])
```

### 8.2 测试用例

| 用例名 | 级别 | M, N, K | dtype | 说明 |
|--------|------|---------|-------|------|
| basic_small | Level 0 | 128, 128, 128 | int8→int32 | 最小功能验证 |
| typical_256 | Level 1 | 256, 256, 256 | int8→int32 | 典型配置 |
| typical_512 | Level 1 | 512, 512, 512 | int8→int32 | 中等规模 |
| large_1024 | Level 3 | 1024, 1024, 1024 | int8→int32 | 大规模性能测试 |

### 8.3 精度标准

| dtype | atol | rtol |
|-------|------|------|
| int32 (转为float16) | 1e-2 | 1e-2 |
| int32 (转为bfloat16) | 1e-2 | 1e-2 |

---

## 9. 风险点与注意事项

### 9.1 已知约束

1. **INT4打包约束**: K必须是2的倍数（每个UINT8存储2个INT4）
2. **整除约束**: M/N/K必须能被对应block size整除
3. **数据类型**: 输入必须为INT8，输出为INT32
4. **转置计算**: 输出shape为(N, M)，即C = B @ A.T

### 9.2 常见错误

| 错误 | 触发场景 | 影响 | 解决方案 |
|------|----------|------|----------|
| INT4符号错误 | nibble>=8时未做符号扩展 | 结果错误 | 使用 `(nibble << 4) >> 4` 或 `nibble - 16` |
| 转置错误 | 忘记设置 `transpose_B=True` | shape不匹配 | 在 `T.gemm_v0` 中设置 transpose_B=True |
| 累加未初始化 | `init=False` 时首次累加 | 结果偏大 | 使用 `init=(k == 0)` |

### 9.3 特殊场景处理

- **非整除分块**: 使用 `T.ceildiv` 或添加边界mask处理
- **极小shape**: 当M/N/K小于block size时，调整block size
- **混合精度输出**: INT32累加结果可转换为float16/bfloat16

---

## 10. 交付清单

### 10.1 目录结构

```
examples/dequantize_gemm/
├── design.md                    # 本设计文档
├── example_dequant_gemm_w4a8.py # 算子实现 + 测试
└── README.md                    # 使用说明（可选）
```

### 10.2 文件清单

| 文件 | 状态 | 说明 |
|------|------|------|
| `design.md` | ✅ 已完成 | 设计文档 |
| `example_dequant_gemm_w4a8.py` | ⬜ 待实现 | 算子实现 |
| `test_dequant_gemm_w4a8.py` | ⬜ 待实现 | 测试文件（可选） |

### 10.3 命名规范

- 目录名: `dequantize_gemm`（snake_case）
- 实现文件: `example_dequant_gemm_w4a8.py`
- 测试文件: `test_dequant_gemm_w4a8.py`

### 10.4 实现顺序

1. ✅ 设计文档（design.md）
2. ⬜ Python端INT4解包函数（torch_unpack_int4）
3. ⬜ NPU端INT8 GEMM kernel
4. ⬜ 基础测试（Level 0 + Level 1）
5. ⬜ 边界测试（Level 2，非整除情况）
6. ⬜ 性能测试（Level 3）