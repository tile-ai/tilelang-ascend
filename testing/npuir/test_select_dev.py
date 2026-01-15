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

    # 编译 kernel
    func = select_kernel(M, N, block_M, dtype)
    compiled = tilelang.compile(func, target="npuir")

    # -------------------------
    # 构造输入
    # -------------------------
    A = torch.randn(N, dtype=torch.float16).npu()
    B = torch.randn(N, dtype=torch.float16).npu()
    Out = torch.zeros((M, N), dtype=torch.float16).npu()

    print("A (true values):")
    print(A.cpu())
    print("\nB (false values):")
    print(B.cpu())

    compiled(A, B, Out)

    print("\nNPU Out:")
    print(Out.cpu())

    # -------------------------
    # reference: use A >= B as condition
    # -------------------------
    ref = torch.where(
        (A.cpu() >= B.cpu())[None, :],  # shape (1, N), broadcast 到 (M, N)
        A.cpu()[None, :],
        B.cpu()[None, :]
    )
    print("\nReference:")
    print(ref)

    # 校验结果
    if torch.allclose(Out.cpu(), ref, rtol=1e-3, atol=1e-3, equal_nan=True):
        print("\n\033[92mResults match!\033[0m")
    else:
        print("\n\033[91mResults do NOT match!\033[0m")


if __name__ == "__main__":
    os.environ["TILELANG_ASCEND_MODE"] = "Developer"
    main()
