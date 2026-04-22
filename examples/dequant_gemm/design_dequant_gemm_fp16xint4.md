# FP16×INT4 Dequantize GEMV 算子设计文档

## 1. 概述

### 1.1 算子名称
dequant_gemv_fp16xint4

### 1.2 功能描述
矩阵-向量乘法算子，支持INT4量化权重。输入向量A为FP16，权重矩阵B为INT4量化（存储在INT8），计算 C = A × B^T。

### 1.3 数学公式
$$
C = A \times B^T
$$

INT4 unpack公式（每个INT8存储两个INT4）：
$$
B_{dequant}[j] = (B_{packed}[j // 2] >> (4 \times (j \% 2))) \& 0xF
$$

## 2. Ascend硬件限制分析

### 2.1 关键限制

| 操作 | Ascend支持情况 | 说明 |
|-----|--------------|------|
| `_tir_packed_int_to_int_convert` | 不支持 | GPU专用TIR intrinsic，无Ascend codegen |
| `T.tile.cast(int8→int16)` | 不支持 | 只支持int8→half, half→int16等 |
| `T.Parallel + bitwise + cast` | 导致错误 | 生成v_thread变量，Ascend无法处理 |

### 2.2 支持的操作

| 操作 | Ascend支持 | 示例 |
|-----|-----------|-----|
| `T.gemm_v0(fp16×fp16→fp32)` | 支持 | 标准FP16 matmul |
| Host端PyTorch bitwise操作 | 支持 | INT4 unpack在CPU执行 |

## 3. 设计方案

### 3.1 方案选择：Host端预处理 + NPU标准GEMV

**选定理由**：
1. 完全避开Ascend不支持的INT4 unpack操作
2. NPU端使用已验证的标准FP16 GEMV（参考gemv_c）
3. 简单可靠，易于调试

### 3.2 数据流

```
Host (CPU):
B_packed (N, K//2, int8) → unpack → B_fp16 (N, K, fp16)
                                    ↓
                              Send to NPU

NPU:
A (1, K, fp16) + B_fp16 (N, K, fp16) → GEMV → C (1, N, fp16)
```

### 3.3 核心代码结构

```python
# Host端unpack (PyTorch)
def unpack_int4_to_fp16(B_packed):
    N, K_compressed = B_packed.shape
    K = K_compressed * 2
    B = torch.zeros(N, K, dtype=torch.float16)
    for j in range(K):
        shift = 4 * (j % 2)
        B[:, j] = ((B_packed[:, j // 2].int() >> shift) & 0xF).half()
    return B

# NPU端GEMV (TileLang)
@tl.jit(out_idx=[-1], pass_configs={...})
def gemv_fp16(N, K, block_N, block_K):
    @T.prim_func
    def main(A, B, C):
        with T.Kernel(n_num, is_npu=True) as (bn_idx, _):
            for bk in T.serial(k_num):
                T.copy(A[0, bk * block_K], A_L1)
                T.copy(B[bn_idx * block_N, bk * block_K], B_L1)
                T.gemm_v0(A_L1, B_L1, C_L0, transpose_B=True, init=(bk == 0))
            T.copy(C_L0, C[0, bn_idx * block_N])
    return main
```

## 4. 文件结构

```
examples/dequant_gemv/
├── example_dequant_gemv_fp16xint4.py  # 算子实现
├── design_dequant_gemv_fp16xint4.md   # 本设计文档
├── example_dequant_gemv_int8xint4.py  # INT8版本
└── README.md                          # 使用说明
```

## 5. 性能考虑

### 5.1 Host端开销
- INT4 unpack在CPU执行，有额外开销
- 数据传输：B_packed → unpack → B_fp16 → NPU
- 可优化：预处理权重，避免每次推理都unpack

### 5.2 NPU端性能
- 使用标准FP16 GEMV，性能可预测
- 可进一步优化block大小

## 6. 验证标准

| dtype | atol | rtol |
|-------|------|------|
| float16 | 1e-3 | 1e-3 |

## 7. 参考

- [tilelang/examples/dequantize_gemm/example_dequant_gemv_fp16xint4.py](../../tilelang/examples/dequantize_gemm/) - GPU版本
- [examples/gemv/example_gemv_c.py](../gemv/example_gemv_c.py) - Ascend GEMV模式