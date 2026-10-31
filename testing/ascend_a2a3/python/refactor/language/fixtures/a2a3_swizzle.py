"""Complete factory from TA 0deb8b93; imports select the private dialect."""

import tilelang
import tilelang.ascend.a2a3.language as T


@tilelang.jit(out_idx=[-1], target="ascendc")
def matmul(M, N, K, block_M, block_N, block_K, K_L1, S1, S2, dtype="float16", accum_dtype="float"):
    m_num = M // block_M
    n_num = N // block_N

    core_num = 20

    @T.macro
    def init_flag():
        T.set_flag("mte1", "mte2", 0)
        T.set_flag("mte1", "mte2", 1)
        T.set_flag("m", "mte1", 0)
        T.set_flag("m", "mte1", 1)
        T.set_flag("fix", "m", 0)

    @T.macro
    def clear_flag():
        T.wait_flag("mte1", "mte2", 0)
        T.wait_flag("mte1", "mte2", 1)
        T.wait_flag("m", "mte1", 0)
        T.wait_flag("m", "mte1", 1)
        T.wait_flag("fix", "m", 0)

    @T.prim_func
    def main(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(core_num, is_npu=True) as (cid, _):
            A_L1 = T.alloc_L1((S1, block_M, K_L1), dtype)
            B_L1 = T.alloc_L1((S1, K_L1, block_N), dtype)

            A_L0 = T.alloc_L0A((S2, block_M, block_K), dtype)
            B_L0 = T.alloc_L0B((S2, block_K, block_N), dtype)
            C_L0 = T.alloc_L0C((block_M, block_N), accum_dtype)

            with T.Scope("C"):
                init_flag()

                for i in T.serial(T.ceildiv(m_num * n_num, core_num)):
                    cid = T.use_swizzle(i * core_num + cid, M, N, K, block_M, block_N, off=3)
                    if cid < m_num * n_num:
                        bx = cid // n_num
                        by = cid % n_num

                        loop_k = T.ceildiv(K, K_L1)

                        T.wait_flag("mte1", "mte2", 0)
                        T.copy(A[bx * block_M, 0], A_L1[0, :, :])
                        T.copy(B[0, by * block_N], B_L1[0, :, :])
                        T.set_flag("mte2", "mte1", 0)
                        T.wait_flag("fix", "m", 0)
                        for k in T.serial(loop_k):
                            if k < loop_k - 1:
                                T.wait_flag("mte1", "mte2", (k + 1) % S1)
                                T.copy(A[bx * block_M, (k + 1) * K_L1], A_L1[(k + 1) % S1, :, :])
                                T.copy(B[(k + 1) * K_L1, by * block_N], B_L1[(k + 1) % S1, :, :])
                                T.set_flag("mte2", "mte1", (k + 1) % S1)

                            loop_kk = T.ceildiv(K_L1, block_K)

                            for kk in T.serial(loop_kk):
                                if kk == 0:
                                    T.wait_flag("mte2", "mte1", k % S1)
                                T.wait_flag("m", "mte1", kk % S2)
                                T.copy(A_L1[k % S1, 0, kk * block_K], A_L0[kk % S2, :, :])
                                T.copy(B_L1[k % S1, kk * block_K, 0], B_L0[kk % S2, :, :])
                                if kk == 3:
                                    T.set_flag("mte1", "mte2", k % S1)
                                T.set_flag("mte1", "m", kk % S2)
                                T.wait_flag("mte1", "m", kk % S2)

                                T.mma(A_L0[kk % S2, :, :], B_L0[kk % S2, :, :], C_L0, init=T.And(k == 0, kk == 0))

                                T.set_flag("m", "mte1", kk % S2)

                        T.set_flag("m", "fix", 0)
                        T.wait_flag("m", "fix", 0)
                        T.copy(C_L0, C[bx * block_M, by * block_N])
                        T.set_flag("fix", "m", 0)

                clear_flag()

    return main
