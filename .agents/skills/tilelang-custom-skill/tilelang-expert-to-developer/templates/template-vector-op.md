# 模板 A：纯 Vector 算子（Developer 模式）

**适用于：** ElementWise Add/Mul/Sub/Div, ReLU, GELU, Sigmoid, 激活函数等

## pass_configs

纯 Vector 算子无 GEMM，只需 2 个开关：

```python
pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}
```

## 完整模板

```python
import tilelang
import tilelang.language as T
import torch

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}

@tilelang.jit(out_idx=[-1], pass_configs=pass_configs)
def vector_op(M, N, block_M, block_N, dtype="float16"):
    m_num = M // block_M
    n_num = N // block_N
    VEC_NUM = 2  # 每核 2 个 Vector 线程

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),
        B: T.Tensor((M, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num

            a_ub = T.alloc_shared((block_M // VEC_NUM, block_N), dtype)
            b_ub = T.alloc_shared((block_M // VEC_NUM, block_N), dtype)
            c_ub = T.alloc_shared((block_M // VEC_NUM, block_N), dtype)

            # 每个 vid 处理 block_M 的一半
            T.copy(A[bx * block_M + vid * block_M // VEC_NUM, by * block_N], a_ub)
            T.copy(B[bx * block_M + vid * block_M // VEC_NUM, by * block_N], b_ub)

            # ==========================================
            # 核心计算 — 替换为具体运算
            # ==========================================
            for i, j in T.Parallel(block_M // VEC_NUM, block_N):
                c_ub[i, j] = a_ub[i, j] + b_ub[i, j]
            # ==========================================

            T.copy(c_ub, C[bx * block_M + vid * block_M // VEC_NUM, by * block_N])

    return main


# 实例化
func = vector_op(1024, 1024, 128, 256)

# 测试
torch.manual_seed(0)
a = torch.randn(1024, 1024).half().npu()
b = torch.randn(1024, 1024).half().npu()
c = func(a, b)

ref_c = a + b
torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)
print("Kernel Output Match!")
```

## 核心计算替换示例

### ElementWise Mul
```python
for i, j in T.Parallel(block_M // VEC_NUM, block_N):
    c_ub[i, j] = a_ub[i, j] * b_ub[i, j]
```

### ReLU（单输入）
```python
# 只需 1 个输入 buffer
for i, j in T.Parallel(block_M // VEC_NUM, block_N):
    c_ub[i, j] = T.max(a_ub[i, j], 0.0)
```

### GELU（近似）
```python
for i, j in T.Parallel(block_M // VEC_NUM, block_N):
    x = a_ub[i, j]
    c_ub[i, j] = 0.5 * x * (1.0 + T.tanh(0.7978845608 * (x + 0.044715 * x * x * x)))
```

### Sigmoid
```python
for i, j in T.Parallel(block_M // VEC_NUM, block_N):
    c_ub[i, j] = 1.0 / (1.0 + T.exp(-a_ub[i, j]))
```

## 关键参数说明

- `VEC_NUM = 2`：每核 2 个 Vector 线程，每个处理 `block_M // 2` 行
- `vid`：当前 Vector 线程编号（0 或 1），用于计算偏移
- `block_M`：建议 64-256，必须为 16 的倍数
- `block_N`：建议 64-256，必须为 16 的倍数
