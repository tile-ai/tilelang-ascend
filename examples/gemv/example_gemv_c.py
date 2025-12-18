import argparse

import tilelang as tl
import tilelang.language as T

import torch

@tl.jit(
    out_idx=[-1],
    pass_configs={
        tl.PassConfigKey.TIR_MERGE_STATIC_SMEM: True,
        tl.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
        tl.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    }
)
def simple_gemv(
    N:int, K:int, block_N:int, block_K:int, 
    dtype:str = "float16", accum_dtype:str = "float32"
):
    """ Cube core GEMV implementation"""
    FRACTAL_SIZE = 16  
    # one fractal is 16x16, even (1, 16) will actually take (16, 16) spaces in L1

    n_num = T.ceildiv(N, block_N)
    k_num = T.ceildiv(K, block_K)

    @T.prim_func
    def main(
        x: T.Tensor((K,), dtype),   # type: ignore
        A: T.Tensor((N, K), dtype), # type: ignore
        y: T.Tensor((N,), dtype),   # type: ignore
    ):
        with T.Kernel(n_num, is_npu=True) as (cid, _):
            bn = cid  % n_num

            A_L1 = T.alloc_L1((block_N, block_K), dtype)
            x_L1 = T.alloc_L1((FRACTAL_SIZE, block_K), dtype)
            C_L0 = T.alloc_L0C((FRACTAL_SIZE, block_N), accum_dtype)

            # block_N * K  per cube core
            for bk in T.serial(k_num):
                T.copy(x[bk * block_K], x_L1)
                T.copy(A[bn * block_N, bk * block_K], A_L1)
                T.gemm_v0(x_L1, A_L1, C_L0, transpose_B=True, init=(bk == 0))
            
            T.copy(C_L0, y[bn * block_N])

    return main


def ref_program(x, A):
    return x @ A.T


def check_case(N:int, K:int, block_N: int = 64, block_K: int = 128, dtype="float16"):
    torch_dtype_map = {"float16": torch.half, "float32": torch.float32, "float": torch.float32}
    x = torch.randn(K).to(torch_dtype_map[dtype]).npu()
    A = torch.randn(N, K).to(torch_dtype_map[dtype]).npu()

    kernel = simple_gemv(N, K, block_N, block_K, dtype=dtype)

    y = kernel(x, A)
    ref_y = ref_program(x, A)

    torch.testing.assert_close(y, ref_y, rtol=1e-2, atol=1e-2)


def main(custom_args=None):
    parser = argparse.ArgumentParser(description="GEMV Example")
    parser.add_argument("--n", type=int, default=1024, help="Matrix dimension N")
    parser.add_argument("--k", type=int, default=1024, help="Matrix dimension K")
    args, remains = parser.parse_known_args(custom_args)
    if remains:
        print(f"[{parser.description}]", "Unknown args:", remains)
    N, K = args.n, args.k

    torch.manual_seed(0)

    check_case(N, K, 128, 128)
    check_case(N, K, 128, 128, dtype="float32")
    check_case(64, 64, 16, 16)

    print("GEMV example passed!")
    print("Kernel Output Match!")

if __name__ == "__main__":
    main()