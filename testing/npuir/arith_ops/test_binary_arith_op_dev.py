import torch
import torch_npu
import tilelang
import tilelang.language as T
import os

os.environ["TILELANG_ASCEND_MODE"] = "Developer"
torch.npu.set_device(0)
tilelang.cache.clear_cache()

dtype = "float16"

def binary_kernel(M, N, block_M, op, dtype="float16"):
    grid_M = (M + block_M - 1) // block_M

    @T.prim_func
    def binaryArithFullDev(
        A: T.Tensor((N,), dtype),
        B: T.Tensor((N,), dtype),
        Out: T.Tensor((N,), dtype),
    ):
        with T.Kernel(grid_M, is_npu=True) as (bx, _):
            # UB buffers
            acc_A = T.alloc_shared((N,), dtype)
            acc_B = T.alloc_shared((N,), dtype)
            out_ub = T.alloc_shared((N,), dtype)

            # GM -> UB
            T.copy(A, acc_A)
            T.copy(B, acc_B)

            # Each row: elementwise binary op
            for i in T.serial(block_M):
                if op == "add":
                    T.npuir_add(acc_A, acc_B, out_ub)
                elif op == "sub":
                    T.npuir_sub(acc_A, acc_B, out_ub)
                elif op == "mul":
                    T.npuir_mul(acc_A, acc_B, out_ub)
                elif op == "div":
                    T.npuir_div(acc_A, acc_B, out_ub)
                else:
                    T.assert_(False, "Unsupported op")

            # UB -> GM
            T.copy(out_ub, Out)

    return binaryArithFullDev

def binary_partial_kernel(M, N, block_M, op, dtype="float16"):
    grid_M = (M + block_M - 1) // block_M

    @T.prim_func
    def binaryArithPartialDev(
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

    return binaryArithPartialDev


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

def main():
    M, N = 4, 64
    block_M = 4

    A = torch.randn(N, dtype=torch.float16).npu()
    B = torch.randn(N, dtype=torch.float16).npu()

    print("================================")
    print("Input A:")
    print(A.cpu())
    print("\nInput B:")
    print(B.cpu())
    print("================================")

    all_pass = True

    for op in ["add", "sub", "mul", "div"]:
        print("\n################################")
        print(f"Testing op: {op}")
        print("################################")

        print("\n--- Full Kernel ---")

        Out_full = torch.zeros((N,), dtype=torch.float16).npu()

        func_full = binary_kernel(M, N, block_M, op, dtype)
        compiled_full = tilelang.compile(func_full, target="npuir")

        compiled_full(A, B, Out_full)

        ref_full = getattr(torch, op)(A.cpu(), B.cpu())

        print("NPU Out (full):")
        print(Out_full.cpu())
        print("\nReference (full):")
        print(ref_full)

        ok_full = torch.allclose(
            Out_full.cpu(), ref_full, rtol=1e-3, atol=1e-3, equal_nan=True
        )

        print(f"Full kernel result: {'PASS' if ok_full else 'FAIL'}")

        print("\n--- Partial Kernel ---")

        Out_partial = torch.zeros((M, N), dtype=torch.float16).npu()

        func_partial = binary_partial_kernel(M, N, block_M, op, dtype)
        compiled_partial = tilelang.compile(func_partial, target="npuir")

        compiled_partial(A, B, Out_partial)

        ref_partial = reference(A.cpu(), B.cpu(), M, op)

        print("NPU Out (partial):")
        print(Out_partial.cpu())
        print("\nReference (partial):")
        print(ref_partial)

        ok_partial = torch.allclose(
            Out_partial.cpu(), ref_partial, rtol=1e-3, atol=1e-3, equal_nan=True
        )

        print(f"Partial kernel result: {'PASS' if ok_partial else 'FAIL'}")

        if not (ok_full and ok_partial):
            all_pass = False

    print("\n================================")
    if all_pass:
        print("\033[92mALL OPS (FULL + PARTIAL) PASS ✔\033[0m")
    else:
        print("\033[91mSOME OPS FAILED ✘\033[0m")
    print("================================")

if __name__ == "__main__":
    print("Running in developer mode")
    main()
