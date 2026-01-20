import torch
import torch_npu
import tilelang
import tilelang.language as T
import os

torch.npu.set_device(0)
tilelang.cache.clear_cache()

dtype = "float16"

def select_kernel(M, N, block_M, dtype="float16"):
    grid_M = (M + block_M - 1) // block_M

    @T.prim_func
    def main(
        A: T.Tensor((N,), dtype),
        B: T.Tensor((N,), dtype),
        Out: T.Tensor((N,), dtype),
    ):
        with T.Kernel(grid_M, is_npu=True) as (bx, _):
            # UB buffers
            cond_ub = T.alloc_shared((N,), "bool")     
            acc_A = T.alloc_shared((N,), dtype)
            acc_B = T.alloc_shared((N,), dtype)
            out_ub = T.alloc_shared((N,), dtype)

            # GM -> UB
            T.copy(A, acc_A)
            T.copy(B, acc_B)

            # cond_ub = (A >= B)
            T.npuir_cmp(acc_A, acc_B, cond_ub, "ge")

            # npuir_select
            for i in T.serial(block_M):
                T.npuir_select(
                    cond_ub,       
                    acc_A,         
                    acc_B,        
                    out_ub   
                )

                T.copy(out_ub, Out)

    return main

def select_partial_kernel(M, N, block_M, dtype="float16"):
    grid_M = (M + block_M - 1) // block_M

    @T.prim_func
    def main(
        A: T.Tensor((N,), dtype),
        B: T.Tensor((N,), dtype),
        Out: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(grid_M, is_npu=True) as (bx, _):
            # UB buffers
            cond_ub = T.alloc_shared((N,), "bool")     
            acc_A = T.alloc_shared((N,), dtype)
            acc_B = T.alloc_shared((N,), dtype)
            out_ub = T.alloc_shared((M, N), dtype)

            # GM -> UB
            T.copy(A, acc_A)
            T.copy(B, acc_B)

            # cond_ub = (A >= B)
            T.npuir_cmp(acc_A, acc_B, cond_ub, "ge")

            # npuir_select
            for i in T.serial(block_M):
                T.npuir_select(
                    cond_ub,       
                    acc_A,         
                    acc_B,        
                    out_ub[i, :]   
                )

                T.copy(out_ub, Out)

    return main


def main():
    M, N = 8, 32
    block_M = 8

    func1 = select_kernel(M, N, block_M, dtype)
    func2 = select_partial_kernel(M, N, block_M, dtype)

    compiled1 = tilelang.compile(func1, target="npuir")
    compiled2 = tilelang.compile(func2, target="npuir")

    A = torch.randn(N, dtype=torch.float16).npu()
    B = torch.randn(N, dtype=torch.float16).npu()

    print("A (true values):")
    print(A.cpu())
    print("\nB (false values):")
    print(B.cpu())

    print("\n================ func1: full shape =================\n")

    Out1 = torch.zeros((N,), dtype=torch.float16).npu()
    compiled1(A, B, Out1)

    print("NPU Out (func1):")
    print(Out1.cpu())

    ref1 = torch.where(
        A.cpu() >= B.cpu(),
        A.cpu(),
        B.cpu()
    )
    print("\nReference (func1):")
    print(ref1)

    if torch.allclose(Out1.cpu(), ref1, rtol=1e-3, atol=1e-3, equal_nan=True):
        print("\033[92mfunc1 PASSED\033[0m")
    else:
        print("\033[91mfunc1 FAILED\033[0m")

    print("\n============== func2: partial / fast-path ==============\n")

    Out2 = torch.zeros((M, N), dtype=torch.float16).npu()
    compiled2(A, B, Out2)

    print("NPU Out (func2):")
    print(Out2.cpu())

    ref2 = torch.where(
        (A.cpu() >= B.cpu())[None, :],
        A.cpu()[None, :],
        B.cpu()[None, :]
    )

    print("\nReference (func2):")
    print(ref2)

    if torch.allclose(Out2.cpu(), ref2, rtol=1e-3, atol=1e-3, equal_nan=True):
        print("\033[92mfunc2 PASSED\033[0m")
    else:
        print("\033[91mfunc2 FAILED\033[0m")



if __name__ == "__main__":
    os.environ["TILELANG_ASCEND_MODE"] = "Developer"
    print("Running in developer mode")
    main()
