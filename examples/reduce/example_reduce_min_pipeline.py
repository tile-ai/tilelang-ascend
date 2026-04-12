import argparse
import tilelang
import tilelang.language as T
import torch

tilelang.cache.clear_cache()

parser = argparse.ArgumentParser(description="NPU Kernel Compilation")
parser.add_argument("--m", type=int, default=512, help="Matrix M dimension")
parser.add_argument("--n", type=int, default=32, help="Matrix N dimension")
args = parser.parse_args()

M = args.m
N = args.n

@tilelang.jit(out_idx=[1], target="ascendc")
def reduce_min_pipeline(M, N, block_M, block_N, sub_M, dtype="float"):
    m_num = M // block_M
    n_num = N // block_N

    VEC_NUM = 2
    stages = 2

    @T.prim_func
    def main(
            A: T.Tensor((M, N), dtype),
            B: T.Tensor((M), dtype),
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num

            vec_proc = block_M // sub_M

            a_ub = T.alloc_ub((stages, sub_M // VEC_NUM, block_N), dtype)
            b_ub = T.alloc_ub((stages, sub_M // VEC_NUM), dtype)

            with T.Scope("V"):
                T.barrier_all()

                T.copy(A[bx * block_M + vid * sub_M // VEC_NUM + 0 * sub_M, by * block_N], a_ub[0, :, :])
                T.barrier_all()

                for mm in T.serial(vec_proc):
                    cur = mm % stages
                    nxt = (mm + 1) % stages

                    if mm < vec_proc - 1:
                        T.barrier_all()
                        T.copy(A[bx * block_M + vid * sub_M // VEC_NUM + (mm + 1) * sub_M, by * block_N],
                               a_ub[nxt, :, :])
                        T.barrier_all()

                    T.barrier_all()

                    T.reduce_min(a_ub[cur, :, :], b_ub[cur, :], dim=-1)

                    T.barrier_all()

                    T.copy(b_ub[cur, :], B[bx * block_M + vid * sub_M // VEC_NUM + mm * sub_M])
                    T.barrier_all()

    return main


if __name__ == "__main__":
    M = args.m
    N = args.n
    block_M = 32
    block_N = 32
    sub_M = 16

    func = reduce_min_pipeline(M, N, block_M, block_N, sub_M)

    torch.manual_seed(0)

    a = torch.randn(M, N).npu()

    torch.npu.synchronize()
    print("init successful!")

    c = func(a)

    torch.npu.synchronize()
    torch.set_printoptions(threshold=float('inf'), sci_mode=False)

    ref_c = torch.min(a, dim=-1).values

    torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)
    print("Kernel Output Match!")