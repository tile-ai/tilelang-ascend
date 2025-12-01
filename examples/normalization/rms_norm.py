import tilelang
from tilelang import DataType, language as T
import torch

tilelang.cache.clear_cache()

@tilelang.jit(out_idx=[1])
def rms_norm(M, N, block_M, block_N, eps=1e-5, dtype="float"):
    """
    RMS Norm
    """

    m_num = M // block_M
    n_num = N // block_N
    VEC_NUM = 2

    @T.prim_func
    def main(
            A: T.Tensor((M, N), dtype), # type: ignore
            B: T.Tensor((M, N), dtype)  # type: ignore
    ):
        with T.Kernel(m_num, is_npu=True) as (cid, vid):
            bx = cid
            a_ub = T.alloc_ub([block_M // VEC_NUM, block_N], dtype)
            sum_square_i = T.alloc_ub([block_M // VEC_NUM, block_N], dtype)
            sum_square_ub = T.alloc_ub([block_M // VEC_NUM], dtype)
            mean_square_ub = T.alloc_ub([block_M // VEC_NUM], dtype)
            tmp_ub = T.alloc_ub([3 * DataType(dtype).bits // 8 * block_M // VEC_NUM * block_N], "uint8")

            with T.Scope("V"):
                # Initialize
                T.tile.fill(sum_square_i, 0.0)
                T.tile.fill(sum_square_ub, 0.0)
                T.tile.fill(mean_square_ub, N)
                T.barrier_all()

                # Accumulation
                for by in T.serial(n_num):
                    T.copy(A[bx*block_M+vid*block_M//VEC_NUM:bx*block_M+(vid+1)*block_M//VEC_NUM,
                             by*block_N:(by+1)*block_N], a_ub)
                    T.barrier_all()
                    T.tile.mul(a_ub, a_ub, a_ub)
                    T.barrier_all()
                    T.tile.add(sum_square_i, sum_square_i, a_ub)
                    T.barrier_all()
                
                # Reduce
                T.tile.reduce_sum(sum_square_ub, sum_square_i, tmp_ub, dim=-1)
                T.barrier_all()

                # Compute mean and variance
                T.tile.div(mean_square_ub, sum_square_ub, mean_square_ub)
                T.barrier_all()
                T.tile.fill(sum_square_ub, eps)
                T.barrier_all()
                T.tile.add(mean_square_ub, mean_square_ub, sum_square_ub)
                T.barrier_all()
                T.tile.sqrt(mean_square_ub, mean_square_ub)
                T.barrier_all()

                # Normalize
                for by in T.serial(n_num):
                    T.copy(A[bx*block_M+vid*block_M//VEC_NUM:bx*block_M+(vid+1)*block_M//VEC_NUM,
                             by*block_N:(by+1)*block_N], a_ub)
                    T.barrier_all()
                    for i in range(block_M // VEC_NUM):
                        T.tile.div(a_ub[i, :], a_ub[i, :], mean_square_ub[i])
                        T.barrier_all()

                    T.copy(a_ub, B[bx*block_M+vid*block_M//VEC_NUM:bx*block_M+(vid+1)*block_M//VEC_NUM,
                                   by*block_N:(by+1)*block_N])
                    T.barrier_all()

    return main


torch.manual_seed(0)
# Tests
test_configs = [
    (256, 256, 64, 64, "float"),
    (1024, 1024, 128, 128, "float"),
    (1024, 51200, 128, 128, "float"),
]

for M, N, block_M, block_N, dtype in test_configs:
    print(f"Testing rms_norm with M={M}, N={N}, block_M={block_M}, block_N={block_N}, dtype={dtype}")
    func = rms_norm(M, N, block_M, block_N, dtype=dtype)
    print("Init successful!")
    a = torch.randn(M, N).npu()
    b = func(a)
    ref_b = torch.rms_norm(a, normalized_shape=[N])
    torch.testing.assert_close(b.cpu(), ref_b.cpu(), rtol=1e-2, atol=1e-2)
    print("Test passed!")

print("Kernel Output Match!")