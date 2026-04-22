# INT8×INT4 Dequantize GEMM 算子设计文档

## 1. 概述

### 1.1 算子名称
dequant_gemm_int8xint4

### 1.2 功能描述
矩阵乘法算子，支持INT4量化权重。输入矩阵A为INT8，权重矩阵B为INT4量化（存储在INT8），计算 C = A × B^T。

### 1.3 数学公式
$$
C = A \times B^T
$$

INT4 unpack公式（带符号扩展）：
$$
B_{dequant}[j] = \text{sign\_extend}((B_{packed}[j // 2] >> (4 \times (j \% 2))) \& 0xF)
$$

## 2. Ascend硬件限制分析

### 2.1 关键限制

| 操作 | Ascend支持情况 | 说明 |
|-----|--------------|------|
| `_tir_packed_int_to_int_convert` | 不支持 | GPU专用TIR intrinsic |
| `_tir_u8_to_i4_to_i8` | 不支持 | GPU专用转换函数 |
| `T.gemm_v0(int8×int8→int32)` | 支持 | Ascend原生INT8 matmul |

## 3. 设计方案

### 3.1 方案选择：Host端预处理 + NPU INT8 matmul

**选定理由**：
1. 完全避开Ascend不支持的INT4 unpack操作
2. NPU端使用已验证的标准INT8×INT8→INT32 matmul（参考quant_matmul）
3. INT8 matmul是Ascend原生支持的高效操作

### 3.2 数据流

```
Host (CPU):
A (M, K, int8) ─────────────────────┐
                                    │
B_packed (N, K//2, int8) → unpack → B_int8 (N, K, int8)
                                    │
                              Send to NPU

NPU:
A (M, K, int8) × B_int8^T (K, N, int8) → C (M, N, int32)
```

### 3.3 核心代码结构

```python
# Host端unpack (PyTorch)
def unpack_int4_to_int8(B_packed):
    N, K_compressed = B_packed.shape
    K = K_compressed * 2
    B = torch.zeros(N, K, dtype=torch.int8)
    for j in range(K):
        shift = 4 * (j % 2)
        i4 = (B_packed[:, j // 2].to(torch.int32) >> shift) & 0xF
        i4_signed = ((i4 << 28) >> 28)  # 符号扩展
        B[:, j] = i4_signed.to(torch.int8)
    return B

# NPU端GEMM (TileLang)
@tl.jit(out_idx=[-1])
def gemm_int8(M, N, K, block_M, block_N, block_K):
    @T.prim_func
    def main(A, B, C):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, _):
            with T.Scope("C"):
                for k in T.serial(k_num):
                    T.copy(A[bx * block_M, k * block_K], A_L1)
                    T.copy(B[k * block_K, by * block_N], B_L1)
                    T.barrier_all()
                    T.gemm_v0(A_L1, B_L1, C_L0, init=(k == 0))
                    T.barrier_all()
                T.copy(C_L0, C[bx * block_M, by * block_N])
    return main
```

## 4. 文件结构

```
examples/dequant_gemv/
├── example_dequant_gemv_int8xint4.py  # 算子实现
├── design_dequant_gemv_int8xint4.md   # 本设计文档
├── example_dequant_gemv_fp16xint4.py  # FP16版本
└── README.md                          # 使用说明
```

## 5. 性能考虑

### 5.1 Block参数

| 参数 | 推荐值 | 说明 |
|-----|-------|------|
| block_M | 128 | M方向分块 |
| block_N | 256 | N方向分块 |
| block_K | 64 | K方向分块 |

### 5.2 维度要求
- M, N, K 应能被对应block整除，避免tail处理

## 6. 验证标准

| dtype | atol | rtol |
|-------|------|------|
| int32 | 0 | 0 (精确匹配) |

## 7. 参考

- [tilelang/examples/dequantize_gemm/example_dequant_gemm_w4a8.py](../../tilelang/examples/dequantize_gemm/) - GPU W4A8版本
- [examples/quant_batch_matmul/example_quant_matmul.py](../quant_batch_matmul/example_quant_matmul.py) - Ascend INT8 matmul模式
- [examples/gemm/example_gemm.py](../gemm/example_gemm.py) - Ascend GEMM模式