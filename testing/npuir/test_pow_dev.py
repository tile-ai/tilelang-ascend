import torch
import torch_npu
import tilelang
import tilelang.language as T
import os

os.environ["TILELANG_ASCEND_MODE"] = "Developer"
torch.npu.set_device(0)
tilelang.cache.clear_cache()

def pow_int_kernel(M, N, block_M):
    grid_M = (M + block_M - 1) // block_M

    @T.prim_func
    def main(
        A: T.Tensor((N,), "int32"),   # base: int32
        B: T.Tensor((N,), "int32"),   # exponent: int32
        Out: T.Tensor((M, N), "int32"),
    ):
        with T.Kernel(grid_M, is_npu=True) as (bx, _):
            # UB buffers
            acc_A  = T.alloc_shared((N,), "int32")
            acc_B  = T.alloc_shared((N,), "int32")
            out_ub = T.alloc_shared((M, N), "int32")

            # GM -> UB
            T.copy(A, acc_A)
            T.copy(B, acc_B)

            # Each row: elementwise pow
            for i in T.serial(block_M):
                T.npuir_pow(acc_A, acc_B, out_ub[i, :])

            # UB -> GM
            T.copy(out_ub, Out)

    return main


def reference(A, B, M):
    ref = torch.pow(A, B)[None, :].expand(M, -1)
    return ref


def main():
    M, N = 4, 32
    block_M = 4

    A = torch.randint(
        low=0, high=5, size=(N,), dtype=torch.int32
    ).npu()

    B = torch.randint(
        low=0, high=4, size=(N,), dtype=torch.int32
    ).npu()

    print("A (base):")
    print(A.cpu())
    print("\nB (exp):")
    print(B.cpu())

    Out = torch.zeros((M, N), dtype=torch.int32).npu()

    func = pow_int_kernel(M, N, block_M)
    compiled = tilelang.compile(func, target="npuir")

    compiled(A, B, Out)

    ref = reference(A.cpu(), B.cpu(), M)

    ok = torch.equal(Out.cpu(), ref)

    print("\n================================")
    if ok:
        print("pow(int32, int32): PASS")
    else:
        print("pow(int32, int32): FAIL")
        print("Out:")
        print(Out.cpu())
        print("Ref:")
        print(ref)
    print("================================")


if __name__ == "__main__":
    print("Running in developer mode")
    main()
