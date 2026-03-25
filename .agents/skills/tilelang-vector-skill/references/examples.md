# Vector算子示例代码

## 示例1: Elementwise Add（逐元素加法）

```python
import tilelang
import tilelang.language as T
import torch

tilelang.cache.clear_cache()

@tilelang.jit(out_idx=[-1])
def vec_add(M, N, block_M, block_N, dtype="float"):
    m_num = M // block_M
    n_num = N // block_N
    VEC_NUM = 2

    @T.prim_func
    def main(
            A: T.Tensor((M, N), dtype),
            B: T.Tensor((M, N), dtype),
            C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num

            a_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
            b_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
            c_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
            
            with T.Scope("V"):
                T.copy(A[bx * block_M + vid * block_M // VEC_NUM, by * block_N], a_ub)
                T.copy(B[bx * block_M + vid * block_M // VEC_NUM, by * block_N], b_ub)

                T.barrier_all()
                T.tile.add(c_ub, a_ub, b_ub)
                T.barrier_all()

                T.copy(c_ub, C[bx * block_M + vid * block_M // VEC_NUM, by * block_N])

    return main
```

## 示例2: Sigmoid（激活函数）

```python
import tilelang
import tilelang.language as T
import torch

tilelang.cache.clear_cache()

@tilelang.jit(out_idx=[1])
def sigmoid(M, N, block_M, block_N, dtype="float"):
    m_num = T.ceildiv(M, block_M)
    n_num = T.ceildiv(N, block_N)
    VEC_NUM = 2

    @T.prim_func
    def main(
            A: T.Tensor((M, N), dtype),
            B: T.Tensor((M, N), dtype)
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num
            
            a_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
            b_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
            neg_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
            
            with T.Scope("V"):
                T.copy(A[bx * block_M + vid * block_M // VEC_NUM, by * block_N], a_ub)
                
                T.barrier_all()
                T.tile.neg(neg_ub, a_ub)           # -x
                T.tile.exp(a_ub, neg_ub)           # exp(-x)
                T.tile.add(a_ub, a_ub, 1.0)        # exp(-x) + 1
                T.tile.reciprocal(b_ub, a_ub)      # 1/(exp(-x)+1)
                T.barrier_all()
                
                T.copy(b_ub, B[bx * block_M + vid * block_M // VEC_NUM, by * block_N])

    return main
```

## 示例4: RMS Norm（归一化）

```python
import tilelang
from tilelang import DataType, language as T
import torch

tilelang.cache.clear_cache()

@tilelang.jit(out_idx=[1])
def rms_norm(M, N, block_M, block_N, eps=1e-5, dtype="float"):
    m_num = M // block_M
    n_num = N // block_N
    VEC_NUM = 2

    @T.prim_func
    def main(
            A: T.Tensor((M, N), dtype),
            B: T.Tensor((M, N), dtype)
    ):
        with T.Kernel(m_num, is_npu=True) as (cid, vid):
            bx = cid
            a_ub = T.alloc_ub([block_M // VEC_NUM, block_N], dtype)
            sum_square_i = T.alloc_ub([block_M // VEC_NUM, block_N], dtype)
            sum_square_ub = T.alloc_ub([block_M // VEC_NUM], dtype)
            rms_ub = T.alloc_ub([block_M // VEC_NUM], dtype)
            tmp_ub = T.alloc_ub([3 * DataType(dtype).bits // 8 * block_M // VEC_NUM * block_N], "uint8")

            with T.Scope("V"):
                T.tile.fill(sum_square_i, 0.0)
                T.tile.fill(sum_square_ub, 0.0)

                # 累加平方和
                for by in T.serial(n_num):
                    T.copy(A[bx*block_M+vid*block_M//VEC_NUM:bx*block_M+(vid+1)*block_M//VEC_NUM,
                             by*block_N:(by+1)*block_N], a_ub)
                    T.tile.mul(a_ub, a_ub, a_ub)  # x^2
                    T.tile.add(sum_square_i, sum_square_i, a_ub)
                
                # 归约
                T.reduce_sum(sum_square_i, sum_square_ub, tmp_ub, dim=-1)
                
                # 计算RMS
                T.tile.div(rms_ub, sum_square_ub, N)  # mean(x^2)
                T.tile.add(rms_ub, rms_ub, eps)
                T.tile.sqrt(rms_ub, rms_ub)  # sqrt(mean(x^2) + eps)

                # 归一化
                for by in T.serial(n_num):
                    T.copy(A[bx*block_M+vid*block_M//VEC_NUM:bx*block_M+(vid+1)*block_M//VEC_NUM,
                             by*block_N:(by+1)*block_N], a_ub)
                    for i in range(block_M // VEC_NUM):
                        T.tile.div(a_ub[i, :], a_ub[i, :], rms_ub[i])
                    T.copy(a_ub, B[bx*block_M+vid*block_M//VEC_NUM:bx*block_M+(vid+1)*block_M//VEC_NUM,
                                   by*block_N:(by+1)*block_N])

    return main
```

## 示例5: ReLU（激活函数）

```python
import tilelang
import tilelang.language as T
import torch

tilelang.cache.clear_cache()

@tilelang.jit(out_idx=[1])
def relu(M, N, block_M, block_N, dtype="float"):
    m_num = T.ceildiv(M, block_M)
    n_num = T.ceildiv(N, block_N)
    VEC_NUM = 2

    @T.prim_func
    def main(
            A: T.Tensor((M, N), dtype),
            B: T.Tensor((M, N), dtype)
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num
            
            a_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
            
            with T.Scope("V"):
                T.copy(A[bx * block_M + vid * block_M // VEC_NUM, by * block_N], a_ub)
                
                T.barrier_all()
                T.tile.relu(a_ub, a_ub)  # max(0, x)
                T.barrier_all()
                
                T.copy(a_ub, B[bx * block_M + vid * block_M // VEC_NUM, by * block_N])

    return main
```

## 测试框架模板

```python
torch.manual_seed(0)
test_configs = [
    (256, 256, 64, 64, "float"),       # 小规模
    (1024, 1024, 64, 128, "float"),    # 中等规模
    (512, 4096, 32, 128, "float"),     # 大规模
]

for M, N, block_M, block_N, dtype in test_configs:
    print(f"Testing with M={M}, N={N}, block_M={block_M}, block_N={block_N}")
    func = my_vector_op(M, N, block_M, block_N, dtype=dtype)
    print("Init successful!")
    
    a = torch.randn(M, N, dtype=getattr(torch, dtype)).npu()
    b = func(a)
    
    # 使用PyTorch的参考实现验证
    ref_b = torch.my_operation(a)
    torch.testing.assert_close(b.cpu(), ref_b.cpu(), rtol=1e-2, atol=1e-2)
    print("Test passed!")

print("All Kernel Output Match!")
```