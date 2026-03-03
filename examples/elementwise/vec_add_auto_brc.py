# Copyright (c) Huawei Technologies Co., Ltd. 2025.
import os
import argparse
import torch
import tilelang
import tilelang.language as T


def ref_program(x, y):
    return x + y[0, :]


@tilelang.jit(out_idx=[-1], target="npuir")
def elementwise_add(M, N, block_M, block_N, in_dtype="float32", out_dtype="float32"):
    @T.prim_func
    def elemAdd(
            A: T.Tensor((M, N), in_dtype),
            B: T.Tensor((M, N), in_dtype),
            C: T.Tensor((M, N), out_dtype)
    ):
        with T.Kernel(T.ceildiv(N, block_N) * T.ceildiv(M, block_M), is_npu=True) as (cid, _):
            by = cid // T.ceildiv(N, block_N)
            bx = cid % T.ceildiv(N, block_N)

            A_shared = T.alloc_shared((block_M, block_N), in_dtype)
            B_shared = T.alloc_shared((block_M, block_N), in_dtype)
            C_local = T.alloc_fragment((block_M, block_N), out_dtype)
            C_shared = T.alloc_shared((block_M, block_N), out_dtype)

            T.copy(A[by * block_M, bx * block_N], A_shared)
            T.copy(B[0, bx * block_N], B_shared)
            T.vadd(A_shared[:, :], B_shared[0, :], C_local[:, :])
            T.copy(C_local, C_shared)
            T.copy(C_shared, C[by * block_M, bx * block_N])

    return elemAdd


def main(M=256, N=256, use_autotune=False):
    os.environ["TILELANG_ASCEND_MODE"] = "Developer"
    a = torch.randn(M, N, dtype=torch.float32, device="npu")
    b = torch.randn(M, N, dtype=torch.float32, device="npu")

    if use_autotune:
        kernel = elementwise_add(M, N, in_dtype="float32", out_dtype="float32")
    else:
        # Default config
        config = {"block_M": 64, "block_N": 64}
        kernel = elementwise_add(M, N, **config, in_dtype="float32", out_dtype="float32")

    c = kernel(a, b)
    torch.testing.assert_close(c, ref_program(a, b), rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=256)
    parser.add_argument("--n", type=int, default=256)
    args, _ = parser.parse_known_args()
    main(args.m, args.n)
    print("\033[92mAll check passed!\033[0m")