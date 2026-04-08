# 模板 C：Cube + Vector 融合算子（Developer 模式）

**适用于：** MatMul + Add, MatMul + Activation, MatMul + Bias + ReLU 等

## 核心特点

- Cube 部分做矩阵乘法（GEMM）
- Vector 部分做后处理（加法、激活等）
- 通过 **workspace tensor** 实现 Cube→Vector 数据中转
- 编译器自动处理核间同步

## pass_configs

```python
pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
}
```

## 完整模板：MatMul + Add

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

@tilelang.jit(out_idx=[2], workspace_idx=4, pass_configs=pass_configs)
def matmul_add(M, N, K, block_M, block_N, block_K, dtype="float16", accum_dtype="float"):
    m_num = M // block_M
    n_num = N // block_N
    VEC_NUM = 2

    @T.prim_func
    def main(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), dtype),       # 输出（out_idx=[2]）
        D: T.Tensor((M, N), dtype),       # 加数
        workspace: T.Tensor((M, N), dtype),  # workspace（workspace_idx=4）
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num

            A_L1 = T.alloc_shared((block_M, block_K), dtype)
            B_L1 = T.alloc_shared((block_K, block_N), dtype)
            C_L0 = T.alloc_fragment((block_M, block_N), accum_dtype)
            d_ub = T.alloc_shared((block_M // VEC_NUM, block_N), dtype)
            c_ub = T.alloc_shared((block_M // VEC_NUM, block_N), dtype)

            # ==========================================
            # Cube 部分：GEMM
            # ==========================================
            loop_k = T.ceildiv(K, block_K)
            for k in T.serial(loop_k):
                T.copy(A[bx * block_M, k * block_K], A_L1)
                T.copy(B[k * block_K, by * block_N], B_L1)
                T.gemm_v0(A_L1, B_L1, C_L0, init=(k == 0))

            # 结果写入 workspace（Cube→Vector 中转）
            T.copy(C_L0, workspace[bx * block_M, by * block_N])

            # ==========================================
            # Vector 部分：后处理
            # ==========================================
            T.copy(workspace[bx * block_M + vid * block_M // VEC_NUM, by * block_N], c_ub)
            T.copy(D[bx * block_M + vid * block_M // VEC_NUM, by * block_N], d_ub)

            for i, j in T.Parallel(block_M // VEC_NUM, block_N):
                c_ub[i, j] = c_ub[i, j] + d_ub[i, j]  # <-- 替换为具体后处理

            T.copy(c_ub, C[bx * block_M + vid * block_M // VEC_NUM, by * block_N])

    return main


# 实例化
func = matmul_add(1024, 1024, 1024, 128, 256, 64)

# 测试
torch.manual_seed(0)
a = torch.randn(1024, 1024).half().npu()
b = torch.randn(1024, 1024).half().npu()
d = torch.randn(1024, 1024).half().npu()
c = func(a, b, d)

ref_c = a @ b + d
torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)
print("Kernel Output Match!")
```

## Vector 后处理替换示例

### MatMul + ReLU
```python
for i, j in T.Parallel(block_M // VEC_NUM, block_N):
    c_ub[i, j] = T.max(c_ub[i, j], 0.0)
```

### MatMul + Bias（行广播）
```python
# bias: T.Tensor((N,), dtype)
bias_ub = T.alloc_shared((block_N,), dtype)
T.copy(bias[by * block_N], bias_ub)
for i, j in T.Parallel(block_M // VEC_NUM, block_N):
    c_ub[i, j] = c_ub[i, j] + bias_ub[j]
```

### MatMul + Scale + Sigmoid
```python
for i, j in T.Parallel(block_M // VEC_NUM, block_N):
    c_ub[i, j] = c_ub[i, j] * scale
    c_ub[i, j] = 1.0 / (1.0 + T.exp(-c_ub[i, j]))
```

## 也可用混合模式后处理

```python
# 使用 T.tile.add 代替 T.Parallel（效果相同）
T.tile.add(c_ub, c_ub, d_ub)
```

## 关键要点

1. **workspace 必须声明**：`workspace_idx=4` 指向参数列表中 workspace 的位置
2. **out_idx 注意**：`out_idx=[2]` 表示第 3 个参数 C 是输出
3. **VEC_NUM = 2**：Vector 部分每个 vid 处理 `block_M // 2` 行
4. **数据流**：`GEMM → workspace → Vector 后处理 → 输出`
