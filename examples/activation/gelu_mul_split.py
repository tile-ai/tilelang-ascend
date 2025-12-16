import tilelang
import tilelang.language as T
import torch
import torch.nn as nn

tilelang.cache.clear_cache()

@tilelang.jit(out_idx=[1])
def gelu_mul(M, N, block_M, block_N, dtype="float"):
    m_num = T.ceildiv(M, block_M)
    # 算子将输入Tensor按照最后一个维度分为左右两个Tensor：x1和x2，对x1进行GELU计算，结果与x2相乘，因此分核只相对于x1的维度
    n_num = T.ceildiv(N // 2, block_N)
    
    VEC_NUM = 2


    @T.prim_func
    def main(
            A: T.Tensor((M, N), dtype),
            B: T.Tensor((M, N // 2), dtype)
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num
            a1_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
            a2_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
            b_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
            temp_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
            with T.Scope("V"):
                # 左半部分用a1_ub缓存
                T.copy(A[bx * block_M + vid * block_M // VEC_NUM, by * block_N], a1_ub)
                T.barrier_all()
                # 右半部分用a2_ub缓存
                T.copy(A[bx * block_M + vid * block_M // VEC_NUM, by * block_N + N // 2], a2_ub)
                T.barrier_all()
                # 计算公式：x^2
                T.tile.mul(temp_ub, a1_ub, a1_ub)
                T.barrier_all()
                # 计算公式：x^3
                T.tile.mul(temp_ub, a1_ub, temp_ub)
                T.barrier_all()
                # 计算公式：0.044715 * x^3
                T.tile.mul(temp_ub, temp_ub, 0.044715)
                T.barrier_all()
                # 计算公式：x + 0.044715 * x^3
                T.tile.add(temp_ub, a1_ub, temp_ub)
                T.barrier_all()
                # 计算公式：-sqrt(8/pi)(x + 0.044715 * x^3)
                T.tile.mul(temp_ub, temp_ub, -1.5957691)
                T.barrier_all()
                # 计算公式：exp(-sqrt(8/pi)(x + 0.044715 * x^3))
                T.tile.exp(temp_ub, temp_ub)
                T.barrier_all()
                # 计算公式：1 + exp(-sqrt(8/pi)(x + 0.044715 * x^3))
                T.tile.add(temp_ub, temp_ub, 1.0)
                T.barrier_all()
                # 计算公式：x / (1 + exp(-sqrt(8/pi)(x + 0.044715 * x^3)))
                T.tile.div(temp_ub, a1_ub, temp_ub)
                T.barrier_all()
                # 左半部分计算结果与右半部分相乘
                T.tile.mul(b_ub, temp_ub, a2_ub)
                T.barrier_all()
                T.copy(b_ub, B[bx * block_M + vid * block_M // VEC_NUM, by * block_N])

    return main


torch.manual_seed(0)
# 测试数据
test_configs = [
    (256, 256, 64, 64),
    (1024, 1024, 128, 128),
]

for M, N, block_M, block_N in test_configs:
    print(f"Testing gelu_mul with M={M}, N={N}, block_M={block_M}, block_N={block_N}")
    func = gelu_mul(M, N, block_M, block_N)
    print("Init successful!")
    a = torch.randn(M, N, dtype=torch.float).npu()
    b = func(a)
    print(func.get_kernel_source())
    gelu = nn.GELU(approximate='tanh')
    a1, a2 = torch.split(a, N // 2, dim=1)
    ref_b = gelu(a1) * a2
    torch.testing.assert_close(b.cpu(), ref_b.cpu(), rtol=1e-2, atol=1e-2)
    print("Test passed!")

print("Kernel Output Match!")