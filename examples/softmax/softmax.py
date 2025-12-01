import tilelang
from tilelang import DataType, language as T
import torch

tilelang.cache.clear_cache()

@tilelang.jit(out_idx=[1])
def softmax_online(M, N, block_M, block_N, dtype="float"):
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
            exp_ub = T.alloc_ub([block_M // VEC_NUM, block_N], dtype)

            max_i = T.alloc_ub([block_M // VEC_NUM], dtype)
            max_prev = T.alloc_ub([block_M // VEC_NUM], dtype)
            sum_i = T.alloc_ub([block_M // VEC_NUM], dtype)
            local_max = T.alloc_ub([block_M // VEC_NUM], dtype)
            local_sum = T.alloc_ub([block_M // VEC_NUM], dtype)
            correction = T.alloc_ub([block_M // VEC_NUM], dtype)

            tmp_ub = T.alloc_ub([3 * DataType(dtype).bits // 8 * block_M // VEC_NUM * block_N], "uint8")

            with T.Scope("V"):
                # Initialize
                T.tile.fill(max_i, -1e38)
                T.tile.fill(sum_i, 0.0)
                T.barrier_all()

                # Pass 1: Compute exp values with online max, store unnormalized
                for by in T.serial(n_num):
                    # Load input
                    T.copy(A[bx * block_M + vid * block_M // VEC_NUM: bx * block_M + (vid + 1) * block_M // VEC_NUM,
                             by * block_N: (by + 1) * block_N], a_ub)
                    T.barrier_all()

                    # Find local max
                    T.tile.reduce_max(local_max, a_ub, tmp_ub, dim=-1)
                    T.barrier_all()

                    # Save prevois max
                    T.copy(max_i, max_prev)
                    T.barrier_all()

                    # Update global max
                    T.tile.max(max_i, max_i, local_max)
                    T.barrier_all()

                    # Compute correction factor
                    T.tile.sub(correction, max_prev, max_i)
                    T.barrier_all()
                    T.tile.exp(correction, correction)
                    T.barrier_all()

                    # Correct running sum
                    T.tile.mul(sum_i, sum_i, correction)
                    T.barrier_all()

                    # Compute exp(x - max_i) and store
                    for i in range(block_M // VEC_NUM):
                        T.barrier_all()
                        T.tile.sub(exp_ub[i, :], a_ub[i, :], max_i[i])
                        T.barrier_all()
                    T.tile.exp(exp_ub, exp_ub)
                    T.barrier_all()

                    # Sum current block
                    T.tile.reduce_sum(local_sum, exp_ub, tmp_ub, dim=-1)
                    T.barrier_all()

                    # Update running sum
                    T.tile.add(sum_i, sum_i, local_sum)
                    T.barrier_all()

                    # Store unnormalized exp values
                    T.copy(exp_ub,
                           B[bx * block_M + vid * block_M // VEC_NUM: bx * block_M + (vid + 1) * block_M // VEC_NUM,
                             by * block_N: (by + 1) * block_N])
                    T.barrier_all()

                    # Correct previously stored blocks
                    if by > 0:
                        for prev_by in T.serial(by):
                            # Load previous block
                            T.copy(B[bx * block_M + vid * block_M // VEC_NUM: bx * block_M + (vid + 1) * block_M // VEC_NUM,
                                     prev_by * block_N: (prev_by + 1) * block_N], a_ub)
                            T.barrier_all()

                            # Multiply by correction factor
                            for i in range(block_M // VEC_NUM):
                                T.barrier_all()
                                T.tile.mul(a_ub[i, :], a_ub[i, :], correction[i])
                                T.barrier_all()

                            # Store back
                            T.copy(a_ub,
                                   B[bx * block_M + vid * block_M // VEC_NUM: bx * block_M + (vid + 1) * block_M // VEC_NUM,
                                     prev_by * block_N: (prev_by + 1) * block_N])
                            T.barrier_all()

                # Pass 2: Normalize
                for by in T.serial(n_num):
                    # Load unnormalized exp values
                    T.copy(B[bx * block_M + vid * block_M // VEC_NUM: bx * block_M + (vid + 1) * block_M // VEC_NUM,
                             by * block_N: (by + 1) * block_N], exp_ub)
                    T.barrier_all()

                    # Normalize
                    for i in range(block_M // VEC_NUM):
                        T.barrier_all()
                        T.tile.div(exp_ub[i, :], exp_ub[i, :], sum_i[i])
                        T.barrier_all()
                    
                    # Store result
                    T.copy(exp_ub,
                           B[bx * block_M + vid * block_M // VEC_NUM: bx * block_M + (vid + 1) * block_M // VEC_NUM,
                             by * block_N: (by + 1) * block_N])
                    T.barrier_all()

    return main

torch.manual_seed(0)
# Tests
test_configs = [
    (256, 256, 64, 64, "float"),
    (512, 512, 64, 64, "float"),
    (1024, 1024, 128, 128, "float"),
    (1024, 51200, 128, 128, "float"),
]

for M, N, block_M, block_N, dtype in test_configs:
    print(f"Testing softmax_online with M={M}, N={N}, block_M={block_M}, block_N={block_N}, dtype={dtype}")
    func = softmax_online(M, N, block_M, block_N, dtype=dtype)
    print("Init successful!")
    a = torch.randn(M, N).npu()
    b = func(a)
    ref_b = torch.softmax(a, dim=-1)
    torch.testing.assert_close(b.cpu(), ref_b.cpu(), rtol=1e-2, atol=1e-2)
    print("Test passed!")

print("Kernel Output Match!")