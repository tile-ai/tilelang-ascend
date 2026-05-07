# dequantize_gemm (Unsigned INT4) 算子设计文档

## 1. 概述

### 1.1 算子名称

dequantize_gemm (Unsigned INT4)

### 1.2 功能描述

细粒度反量化矩阵乘法，将 UINT8 打包的 Unsigned INT4 权重反量化为 FP16/INT8，然后执行矩阵乘法。支持两种计算模式：

- **FP16 模式**：A:FP16, B:INT4→FP16, C:FP16（FP32 累加）
- **INT8 模式**：A:INT8, B:INT4→INT8, C:INT32（INT32 累加）

### 1.3 数学公式

#### 主公式

$$
C[i,j] = \sum_k A[i,k] \times B_{dequant}^T[j,k]
$$

#### Unsigned INT4 反量化公式

```
// 提取 4-bit 值（无符号）
u4 = (val >> (pos * 4)) & 0xF

// FP16 模式
fp16 = float16(u4)    // 范围 0-15

// INT8 模式
i8 = int8(u4)         // 范围 0-15
```

---

## 2. 数据流详解

### 2.1 输入输出规格

| 阶段 | 数据 | 形状 | 类型 | 说明 |
|------|------|------|------|------|
| **输入 A** | A | (M, K) | FP16/INT8 | 激活矩阵 |
| **输入 B** | B_packed | (N, K/2) | UINT8 | 权重（packed INT4） |
| **反量化后** | B_dequant | (K, N) | FP16/INT8 | 反量化并转置 |
| **输出** | C | (M, N) | FP16/INT32 | 输出矩阵 |

### 2.2 两种模式对比

| 特性 | FP16 模式 | INT8 模式 |
|------|----------|----------|
| **输入 A** | FP16 | INT8 |
| **反量化输出** | FP16 | INT8 |
| **累加类型** | FP32 | INT32 |
| **输出类型** | FP16 | INT32 |
| **计算单元** | Cube (FP16) | Cube (INT8) |
| **精度** | 较高 | 中等 |
| **范围** | FP16 范围 | INT8 范围 (0-15) |

---

## 3. 编程模式选型

### 3.1 模式选择

| 模式 | FP16 GEMM | INT8 GEMM |
|------|-----------|-----------|
| **编程模式** | Developer | Expert |
| **理由** | 纯 Cube 计算，编译器自动优化 | 需手动同步控制 |

### 3.2 pass_configs 配置

#### FP16 模式（Developer）

```python
pass_configs={
    tl.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tl.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}
```

#### INT8 模式（Expert）

```python
pass_configs={
    tl.PassConfigKey.TL_ASCEND_AUTO_SYNC: False,
    tl.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: False,
}
```

---

## 4. API 映射

### 4.1 内存分配

| 操作 | FP16 模式 | INT8 模式 |
|------|----------|----------|
| A 缓冲区 | `T.alloc_shared((block_M, block_K), "float16")` | `T.alloc_L1((block_M, block_K), "int8")` |
| B 缓冲区 | `T.alloc_shared((block_K, block_N), "float16")` | `T.alloc_L1((block_K, block_N), "int8")` |
| C 缓冲区 | `T.alloc_fragment((block_M, block_N), "float")` | `T.alloc_L0C((block_M, block_N), "int32")` |

### 4.2 数据搬运

| 操作 | API |
|------|-----|
| GM → L1/Shared | `T.copy(src, dst)` |
| L0C → GM | `T.copy(C_L0, C[...])` |

### 4.3 计算

| 操作 | FP16 模式 | INT8 模式 |
|------|----------|----------|
| GEMM | `T.gemm_v0(A_L1, B_L1, C_L0, init=(k==0))` | `T.gemm_v0(A_L1, B_L1, C_L0, init=(k==0))` |

### 4.4 同步（INT8 模式）

| 操作 | API |
|------|-----|
| 全核同步 | `T.barrier_all()` |

---

## 5. Tiling 策略

### 5.1 分块参数

| 参数 | FP16 模式 | INT8 模式 | 说明 |
|------|----------|----------|------|
| block_M | 128 | 128 | M 维度分块 |
| block_N | 256 | 128 | N 维度分块 |
| block_K | 64 | 64 | K 维度分块 |

### 5.2 内存预算分析

#### FP16 模式

```
A_L1: 128 * 64 * 2 = 16 KB
B_L1: 64 * 256 * 2 = 32 KB
C_L0: 128 * 256 * 4 = 128 KB (FP32 累加)
总计: 176 KB < L1 容量 (约 1MB)
```

#### INT8 模式

```
A_L1: 128 * 64 * 1 = 8 KB
B_L1: 64 * 128 * 1 = 8 KB
C_L0: 128 * 128 * 4 = 64 KB (INT32 累加)
总计: 80 KB < L1 容量
```

### 5.3 对齐约束

- M % block_M == 0
- N % block_N == 0
- K % block_K == 0
- K % 2 == 0（INT4 打包格式要求）

---

## 6. 实现策略

### 6.1 Python 端反量化

由于 `tir.reinterpret` 在 Ascend 平台存在兼容性问题，采用 Python 端反量化：

```python
def torch_unpack_uint4_to_fp16(tensor: torch.Tensor) -> torch.Tensor:
    """
    将 UINT8 打包的 Unsigned INT4 解包为 FP16
    tensor: shape (N, K//2), dtype uint8
    return: shape (N, K), dtype float16
    """
    N, K_packed = tensor.shape
    K = K_packed * 2
    result = torch.empty(N, K, dtype=torch.float16)
    for i in range(N):
        for j in range(K):
            val = tensor[i, j // 2].item()
            pos = j % 2
            u4 = (val >> (pos * 4)) & 0xF
            result[i, j] = float(u4)
    return result
```

### 6.2 NPU 端 GEMM

#### FP16 GEMM (Developer 模式)

```python
@T.prim_func
def main(A: T.Tensor((M, K), "float16"),
         B: T.Tensor((K, N), "float16"),
         C: T.Tensor((M, N), "float16")):
    with T.Kernel(m_num * n_num, is_npu=True) as (cid, _):
        A_L1 = T.alloc_shared((block_M, block_K), "float16")
        B_L1 = T.alloc_shared((block_K, block_N), "float16")
        C_L0 = T.alloc_fragment((block_M, block_N), "float")
        
        for k in T.serial(K // block_K):
            T.copy(A[bx * block_M, k * block_K], A_L1)
            T.copy(B[k * block_K, by * block_N], B_L1)
            T.gemm_v0(A_L1, B_L1, C_L0, init=(k == 0))
        
        T.copy(C_L0, C[bx * block_M, by * block_N])
```

#### INT8 GEMM (Expert 模式)

```python
@T.prim_func
def main(A: T.Tensor((M, K), "int8"),
         B: T.Tensor((K, N), "int8"),
         C: T.Tensor((M, N), "int32")):
    with T.Kernel(m_num * n_num, is_npu=True) as (cid, _):
        A_L1 = T.alloc_L1((block_M, block_K), "int8")
        B_L1 = T.alloc_L1((block_K, block_N), "int8")
        C_L0 = T.alloc_L0C((block_M, block_N), "int32")
        
        with T.Scope("C"):
            for k in T.serial(K // block_K):
                T.copy(A[bx * block_M, k * block_K], A_L1)
                T.copy(B[k * block_K, by * block_N], B_L1)
                T.barrier_all()
                T.gemm_v0(A_L1, B_L1, C_L0, init=(k == 0))
                T.barrier_all()
            
            T.copy(C_L0, C[bx * block_M, by * block_N])
```

---

## 7. 验证方案

### 7.1 测试用例

| Level | M | N | K | 模式 | 说明 |
|-------|---|---|---|------|------|
| 0 | 256 | 256 | 256 | FP16 | 基础功能验证 |
| 1 | 512 | 512 | 512 | FP16 | 典型规模验证 |
| 2 | 用户指定 | 用户指定 | 用户指定 | FP16 | 自定义维度 |
| 0 | 256 | 256 | 256 | INT8 | 基础功能验证 |
| 1 | 512 | 512 | 512 | INT8 | 典型规模验证 |
| 2 | 用户指定 | 用户指定 | 用户指定 | INT8 | 自定义维度 |

### 7.2 精度验证

| 模式 | RTOL | ATOL | 说明 |
|------|------|------|------|
| FP16 | 1e-2 | 1e-2 | FP16 精度容差 |
| INT8 | 0 | 0 | INT32 精确匹配 |

### 7.3 参考实现

```python
def ref_program_fp16(A: torch.Tensor, B_packed: torch.Tensor):
    B_fp16 = torch_unpack_uint4_to_fp16(B_packed)
    A_fp32 = A.to(torch.float32)
    B_fp32 = B_fp16.to(torch.float32)
    C_fp32 = torch.matmul(A_fp32, B_fp32.T)
    return C_fp32.to(torch.float16)

def ref_program_int8(A: torch.Tensor, B_packed: torch.Tensor):
    B_int8 = torch_unpack_uint4_to_int8(B_packed)
    A_int32 = A.to(torch.int32)
    B_int32 = B_int8.to(torch.int32)
    C_int32 = torch.matmul(A_int32, B_int32.T)
    return C_int32
```

---

## 8. 文件清单

| 文件 | 说明 |
|------|------|
| `design_uint4.md` | 本设计文档 |
| `example_dequant_gemm_uint4.py` | 算子实现（FP16 + INT8 双模式） |
| `test_dequant_gemm_uint4.py` | 单元测试 |

---

## 9. 运行命令

```bash
# 设置环境
source set_env.sh

# 运行主程序
python examples/dequantize_gemm/example_dequant_gemm_uint4.py

# 运行测试
python examples/dequantize_gemm/test_dequant_gemm_uint4.py

# 自定义维度
python examples/dequantize_gemm/example_dequant_gemm_uint4.py --m 1024 --n 1024 --k 1024
```