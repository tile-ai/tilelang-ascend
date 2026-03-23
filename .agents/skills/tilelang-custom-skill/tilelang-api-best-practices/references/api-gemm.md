# 矩阵计算（GEMM）

## 概述

TileLang 提供两种 GEMM 接口用于块级矩阵乘法运算，计算 C += op(A) × op(B)。

## API

### T.gemm_v0(A, B, C, transpose_A=False, transpose_B=False, init=False)

标准 GEMM 操作。左矩阵和右矩阵位于 shared 存储层级，输出位于 fragment 存储层级。

**参数**：
- `A`：左输入矩阵（shared 层级，支持 Buffer 或 BufferRegion）
- `B`：右输入矩阵（shared 层级，支持 Buffer 或 BufferRegion）
- `C`：结果累加输出矩阵（fragment 层级，必须为 2D）
- `transpose_A`：是否转置 A（默认 False）
- `transpose_B`：是否转置 B（默认 False）
- `init`：是否在计算前将 C 清零（默认 False）

**功能**：
```
C_fragment += A_shared * B_shared  (init=False)
C_fragment = A_shared * B_shared   (init=True)
```

**示例**：
```python
A_L1 = T.alloc_shared((block_M, block_K), dtype)
B_L1 = T.alloc_shared((block_K, block_N), dtype)
C_L0 = T.alloc_fragment((block_M, block_N), accum_dtype)

for k in T.serial(loop_k):
    T.copy(A[bx * block_M, k * block_K], A_L1)
    T.copy(B[k * block_K, by * block_N], B_L1)
    T.gemm_v0(A_L1, B_L1, C_L0, init=(k == 0))
```

### T.gemm_v1(A, B, C, transpose_A=False, transpose_B=False, init=False)

分层 GEMM 操作，适用于分层内存管理场景。从输入 buffer 的 shape 中提取 L1 和计算块大小。

**参数**：与 `T.gemm_v0` 相同。

**区别**：`gemm_v1` 会根据 A/B/C 的 shape 自动推导 L1_BLOCK_M/N/K 和 BLOCK_M/N 等分层参数。

## 最佳实践

1. **init 参数**：循环第一次迭代使用 `init=True`（或 `init=(k == 0)`），后续迭代使用 `init=False` 累加
2. **配合流水线**：在 `T.Pipelined` 中使用 GEMM，实现搬运与计算重叠
   ```python
   for ko in T.Pipelined(T.ceildiv(K, BK), num_stages=3):
       T.copy(A[by * BM, ko * BK], A_s)
       T.copy(B[ko * BK, bx * BN], B_s)
       T.gemm_v0(A_s, B_s, C_f)
   ```
3. **数据类型**：A 和 B 通常为 float16/bfloat16，C（accumulator）通常为 float32
4. **转置支持**：通过 `transpose_A`/`transpose_B` 参数控制矩阵转置，无需手动转置数据
