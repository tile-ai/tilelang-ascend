"""
Test T.sync_all("Mix") with proven GEMM pattern.
Uses alloc_shared (Developer mode) for correct L1 layout, then
sync_all("Mix") at top level after C scope to verify no hang.
"""
import tilelang, tilelang.language as T, torch

tilelang.cache.clear_cache()
M, K, N = 256, 128, 128


@tilelang.jit(out_idx=[-1], target="pto", pass_configs={
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
})
def matmul_with_syncall(M, N, K, block_M=128, block_N=128, block_K=64,
                        dtype="float16", accum_dtype="float"):
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

            A_L1 = T.alloc_shared((block_M, block_K), dtype)
            B_L1 = T.alloc_shared((block_K, block_N), dtype)
            C_L0 = T.alloc_fragment((block_M, block_N), accum_dtype)

            with T.Scope("C"):
                loop_k = T.ceildiv(K, block_K)
                for k in T.serial(loop_k):
                    T.copy(A[bx * block_M, k * block_K], A_L1)
                    T.copy(B[k * block_K, by * block_N], B_L1)
                    T.gemm_v0(A_L1, B_L1, C_L0, init=(k == 0))

                T.copy(C_L0, C[bx * block_M, by * block_N])
                T.barrier_all()
                T.set_cross_flag("FIX", 0)

            T.sync_all("Mix")

            with T.Scope("V"):
                T.wait_cross_flag(0)

    return main


func = matmul_with_syncall(M, N, K)
a = torch.randn(M, K, dtype=torch.float16).npu()
b = torch.randn(K, N, dtype=torch.float16).npu()
torch.npu.synchronize()
c = func(a, b)
torch.npu.synchronize()

ref = a @ b
torch.testing.assert_close(c.cpu(), ref.cpu(), rtol=0.05, atol=0.1)
print("Mix: PASS — GEMM correct, sync_all(Mix) no hang")