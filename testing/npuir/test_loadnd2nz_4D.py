# Copyright (c) Huawei Technologies Co., Ltd. 2025.
import torch
import tilelang
import tilelang.language as T

torch.npu.set_device(12)
@tilelang.jit(target="npuir")
def kernel_mha_qk_matmul(b, n, s, d, block_d, dtype="float16", accum_dtype="float32"):

    @T.prim_func
    def mhaQKMatmul(
            Q: T.Tensor((b, n, s, d), dtype),
            K: T.Tensor((b, n, s, d), dtype),
            A: T.Tensor((b * n * s, s), accum_dtype)
    ):
        with T.Kernel(b * n, is_npu=True) as (cid, _):
            with T.Scope("Cube"):
                b_id = cid // n
                n_id = cid % n

                Q_BUF = T.alloc_L1([s, block_d], dtype)
                K_BUF = T.alloc_L1([s, block_d], dtype)
                A_BUF = T.alloc_L0C([s, s], accum_dtype)

                offset = ((b_id * n) + n_id) * s
                for i in T.serial(T.ceildiv(d, block_d)):
                    T.npuir_load_nd2nz(Q[b_id, n_id, 0, i * block_d], Q_BUF, [s, block_d])
                    T.npuir_load_nd2nz(K[b_id, n_id, 0, i * block_d], K_BUF, [s, block_d])

                    T.npuir_dot(Q_BUF, K_BUF, A_BUF, initC=(i == 0), b_transpose=True, size=[s, block_d, s])

                T.npuir_store_fixpipe(A_BUF, A[offset, 0], size=[s, s], enable_nz2nd=True)

    return mhaQKMatmul


def test_loadnd2nz_4D():
    b = 4
    n = 16
    s = 64
    d = 2048
    block_d = 64
    kernel = kernel_mha_qk_matmul(b, n, s, d, block_d)
    q = torch.randn((b, n, s, d), dtype=torch.float16).npu()
    k = torch.randn((b, n, s, d), dtype=torch.float16).npu()
    a = torch.randn((b, n, s, s), dtype=torch.float32).npu()

    kernel(q, k, a)
    print("actual:\n", a)

    ref_a = (q @ k.transpose(-1, -2)).to(dtype=torch.float32)
    print("expect:\n", ref_a)

    torch.testing.assert_close(a, ref_a, rtol=5e-3, atol=1e-2)
    print("\033[92mAll check passed!\033[0m")


if __name__ == "__main__":
    test_loadnd2nz_4D()
