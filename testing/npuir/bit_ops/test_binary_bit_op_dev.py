import torch
import torch_npu
import tilelang
import tilelang.language as T
import os

# ------------------------------------------------------------
# Environment setup
# ------------------------------------------------------------
os.environ["TILELANG_ASCEND_MODE"] = "Developer"
torch.npu.set_device(0)
tilelang.cache.clear_cache()

# ------------------------------------------------------------
# Generic binary kernel
# ------------------------------------------------------------
def binary_kernel(M, N, block_M, op_name):
    grid_M = (M + block_M - 1) // block_M

    @T.prim_func
    def main(
        A: T.Tensor((N,), "int32"),
        B: T.Tensor((N,), "int32"),
        Out: T.Tensor((N,), "int32"),
    ):
        with T.Kernel(grid_M, is_npu=True) as (bx, _):
            # UB buffers
            acc_A  = T.alloc_shared((N,), "int32")
            acc_B  = T.alloc_shared((N,), "int32")
            out_ub = T.alloc_shared((N,), "int32")

            # GM -> UB
            T.copy(A, acc_A)
            T.copy(B, acc_B)

            # Elementwise binary op per row
            for i in T.serial(block_M):
                if op_name == "max":
                    T.npuir_max(acc_A, acc_B, out_ub)
                elif op_name == "min":
                    T.npuir_min(acc_A, acc_B, out_ub)
                elif op_name == "and":
                    T.npuir_and(acc_A, acc_B, out_ub)
                elif op_name == "or":
                    T.npuir_or(acc_A, acc_B, out_ub)
                elif op_name == "xor":
                    T.npuir_xor(acc_A, acc_B, out_ub)
                elif op_name == "shl":
                    T.npuir_shl(acc_A, acc_B, out_ub)
                elif op_name == "shr":
                    T.npuir_shr(acc_A, acc_B, out_ub)
                else:
                    raise ValueError(f"Unsupported op: {op_name}")

            # UB -> GM
            T.copy(out_ub, Out)

    return main

def binary_partial_kernel(M, N, block_M, op_name):
    grid_M = (M + block_M - 1) // block_M

    @T.prim_func
    def main(
        A: T.Tensor((N,), "int32"),
        B: T.Tensor((N,), "int32"),
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

            # Elementwise binary op per row
            for i in T.serial(block_M):
                if op_name == "max":
                    T.npuir_max(acc_A, acc_B, out_ub[i, :])
                elif op_name == "min":
                    T.npuir_min(acc_A, acc_B, out_ub[i, :])
                elif op_name == "and":
                    T.npuir_and(acc_A, acc_B, out_ub[i, :])
                elif op_name == "or":
                    T.npuir_or(acc_A, acc_B, out_ub[i, :])
                elif op_name == "xor":
                    T.npuir_xor(acc_A, acc_B, out_ub[i, :])
                elif op_name == "shl":
                    T.npuir_shl(acc_A, acc_B, out_ub[i, :])
                elif op_name == "shr":
                    T.npuir_shr(acc_A, acc_B, out_ub[i, :])
                else:
                    raise ValueError(f"Unsupported op: {op_name}")

            # UB -> GM
            T.copy(out_ub, Out)

    return main

# ------------------------------------------------------------
# CPU reference
# ------------------------------------------------------------
def reference(A, B, M, op_name):
    if op_name == "max":
        ref = torch.max(A, B)[None, :].expand(M, -1)
    elif op_name == "min":
        ref = torch.min(A, B)[None, :].expand(M, -1)
    elif op_name == "and":
        ref = (A & B)[None, :].expand(M, -1)
    elif op_name == "or":
        ref = (A | B)[None, :].expand(M, -1)
    elif op_name == "xor":
        ref = (A ^ B)[None, :].expand(M, -1)
    elif op_name == "shl":
        ref = (A << B)[None, :].expand(M, -1)
    elif op_name == "shr":
        ref = (A >> B)[None, :].expand(M, -1)
    else:
        raise ValueError(f"Unsupported op: {op_name}")
    return ref

def main():
    M, N = 4, 32
    block_M = 4
    all_pass = True

    A = torch.randint(0, 10, (N,), dtype=torch.int32).npu()
    B = torch.randint(0, 10, (N,), dtype=torch.int32).npu()
    

    ops = ["max", "min", "and", "or", "xor", "shl", "shr"]

    for op in ops:
        print("\n################################")
        print(f"Testing op: {op}")
        print("################################")

        print("\n--- Full Kernel ---")

        Out_full = torch.zeros((N,), dtype=torch.int32).npu()
        func_full = binary_kernel(M, N, block_M, op)
        compiled_full = tilelang.compile(func_full, target="npuir")

        compiled_full(A, B, Out_full)

        ref_full = reference(A.cpu(), B.cpu(), M, op)

        print("NPU Out (full):")
        print(Out_full.cpu())
        print("\nReference (full):")
        print(ref_full)

        ok_full = torch.allclose(
            Out_full.cpu(), ref_full, rtol=1e-3, atol=1e-3, equal_nan=True
        )

        print(f"Full kernel result: {'PASS' if ok_full else 'FAIL'}")
            

        print("\n--- Partial Kernel ---")

        Out_partial = torch.zeros((M, N), dtype=torch.int32).npu()

        func_partial = binary_partial_kernel(M, N, block_M, op)
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
