# 模板 B：GEMM 矩阵乘法（Developer 模式）

**适用于：** 标准矩阵乘法 C = A @ B

## pass_configs

含 Cube 操作，4 个开关全开：

```python
pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
}
```

## 完整模板

```python
import tilelang
import tilelang.language as T
import torch

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
}

@tilelang.jit(out_idx=[-1], pass_configs=pass_configs)
def matmul(M, N, K, block_M, block_N, K_L1, dtype="float16", accum_dtype="float"):
    m_num = M // block_M
    n_num = N // block_N

    @T.prim_func
    def main(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, _):
            bx = cid // n_num
            by = cid % n_num

            A_L1 = T.alloc_shared((block_M, K_L1), dtype)
            B_L1 = T.alloc_shared((K_L1, block_N), dtype)
            C_L0 = T.alloc_fragment((block_M, block_N), accum_dtype)

            loop_k = T.ceildiv(K, K_L1)
            for k in T.serial(loop_k):
                T.copy(A[bx * block_M, k * K_L1], A_L1)
                T.copy(B[k * K_L1, by * block_N], B_L1)
                T.gemm_v0(A_L1, B_L1, C_L0, init=(k == 0))

            T.copy(C_L0, C[bx * block_M, by * block_N])

    return main


# 实例化
func = matmul(1024, 1024, 1024, 128, 256, 64)

# 测试
torch.manual_seed(0)
a = torch.randn(1024, 1024).half().npu()
b = torch.randn(1024, 1024).half().npu()
c = func(a, b)

ref_c = a @ b
torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)
print("Kernel Output Match!")
```

## 变体：流水线 GEMM

使用 `T.Pipelined` 实现搬运-计算重叠：

```python
@tilelang.jit(out_idx=[-1], pass_configs=pass_configs)
def matmul_pipelined(M, N, K, block_M, block_N, K_L1, dtype="float16", accum_dtype="float"):
    m_num = M // block_M
    n_num = N // block_N

    @T.prim_func
    def main(A: T.Tensor((M, K), dtype), B: T.Tensor((K, N), dtype), C: T.Tensor((M, N), dtype)):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, _):
            bx = cid // n_num
            by = cid % n_num

            A_L1 = T.alloc_shared((block_M, K_L1), dtype)
            B_L1 = T.alloc_shared((K_L1, block_N), dtype)
            C_L0 = T.alloc_fragment((block_M, block_N), accum_dtype)

            for k in T.Pipelined(T.ceildiv(K, K_L1), num_stages=2):
                T.copy(A[bx * block_M, k * K_L1], A_L1)
                T.copy(B[k * K_L1, by * block_N], B_L1)
                T.gemm_v0(A_L1, B_L1, C_L0, init=(k == 0))

            T.copy(C_L0, C[bx * block_M, by * block_N])

    return main
```

## 变体：转置 GEMM

```python
# C = A @ B^T
T.gemm_v0(A_L1, B_L1, C_L0, transpose_B=True, init=(k == 0))

# C = A^T @ B
T.gemm_v0(A_L1, B_L1, C_L0, transpose_A=True, init=(k == 0))
```

## 关键参数说明

- `block_M`：M 方向分块大小，建议 128
- `block_N`：N 方向分块大小，建议 256
- `K_L1`：K 方向每次搬入 L1 的大小，建议 64
- `accum_dtype`：累加器类型，必须用 `"float"`（float32）
- `init=(k == 0)`：首次迭代初始化 L0C，后续累加
- Kernel 数量 = `m_num * n_num`，每个核处理一个 tile
