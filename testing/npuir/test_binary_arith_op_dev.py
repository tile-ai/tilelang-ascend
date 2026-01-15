import torch
import torch_npu
import tilelang
import tilelang.language as T
import os

# ------------------------------------------------------------
# Environment setup
# ------------------------------------------------------------
os.environ["TILELANG_ASCEND_MODE"] = "Developer"
torch.npu.set_device(7)
tilelang.cache.clear_cache()

dtype = "float16"

# ------------------------------------------------------------
# Generic binary kernel
# ------------------------------------------------------------
def binary_kernel(M, N, block_M, op, dtype="float16"):
    grid_M = (M + block_M - 1) // block_M

    @T.prim_func
    def main(
        A: T.Tensor((N,), dtype),
        B: T.Tensor((N,), dtype),
        Out: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(grid_M, is_npu=True) as (bx, _):
            # UB buffers
            acc_A = T.alloc_shared((N,), dtype)
            acc_B = T.alloc_shared((N,), dtype)
            out_ub = T.alloc_shared((M, N), dtype)

            # GM -> UB
            T.copy(A, acc_A)
            T.copy(B, acc_B)

            # Each row: elementwise binary op
            for i in T.serial(block_M):
                if op == "add":
                    T.npuir_add(acc_A, acc_B, out_ub[i, :])
                elif op == "sub":
                    T.npuir_sub(acc_A, acc_B, out_ub[i, :])
                elif op == "mul":
                    T.npuir_mul(acc_A, acc_B, out_ub[i, :])
                elif op == "div":
                    T.npuir_div(acc_A, acc_B, out_ub[i, :])
                else:
                    T.assert_(False, "Unsupported op")

            # UB -> GM
            T.copy(out_ub, Out)

    return main

# ------------------------------------------------------------
# Reference implementation
# ------------------------------------------------------------
def reference(A, B, M, op):
    if op == "add":
        ref = (A + B)[None, :].expand(M, -1)
    elif op == "sub":
        ref = (A - B)[None, :].expand(M, -1)
    elif op == "mul":
        ref = (A * B)[None, :].expand(M, -1)
    elif op == "div":
        ref = (A / B)[None, :].expand(M, -1)
    else:
        raise ValueError(op)
    return ref

# ------------------------------------------------------------
# Main test
# ------------------------------------------------------------
def main():
    M, N = 4, 64
    block_M = 4

    A = torch.randn(N, dtype=torch.float16).npu()
    B = torch.randn(N, dtype=torch.float16).npu()

    print("A:")
    print(A.cpu())
    print("\nB:")
    print(B.cpu())

    all_pass = True   

    for op in ["add", "sub", "mul", "div"]:
        print("\n================================")
        print(f"Testing op: {op}")
        print("================================")

        Out = torch.zeros((M, N), dtype=torch.float16).npu()

        func = binary_kernel(M, N, block_M, op, dtype)
        compiled = tilelang.compile(func, target="npuir")

        compiled(A, B, Out)

        ref = reference(A.cpu(), B.cpu(), M, op)

        ok = torch.allclose(
            Out.cpu(), ref, rtol=1e-3, atol=1e-3, equal_nan=True
        )

        if ok:
            print(f"{op}: PASS")
        else:
            print(f"{op}: FAIL")
            all_pass = False

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------
    print("\n================================")
    if all_pass:
        print("\033[92mALL OPS PASS ✔\033[0m")
    else:
        print("\033[91mSOME OPS FAILED ✘\033[0m")
    print("================================")

# ------------------------------------------------------------
# Entry
# ------------------------------------------------------------
if __name__ == "__main__":
    main()
