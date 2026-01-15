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

# ------------------------------------------------------------
# Generic binary kernel
# ------------------------------------------------------------
def binary_kernel(M, N, block_M, op_name):
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

# ------------------------------------------------------------
# Main test
# ------------------------------------------------------------
def main():
    M, N = 4, 32
    block_M = 4
    all_pass = True

    ops = ["max", "min", "and", "or", "xor", "shl", "shr"]

    for op in ops:
        print("\n==============================")
        print(f"Testing op: {op}")

        # 随机生成输入
        A = torch.randint(0, 10, (N,), dtype=torch.int32).npu()
        B = torch.randint(0, 10, (N,), dtype=torch.int32).npu()
        Out = torch.zeros((M, N), dtype=torch.int32).npu()

        print("A:")
        print(A.cpu())
        print("B:")
        print(B.cpu())

        # 编译执行
        func = binary_kernel(M, N, block_M, op)
        compiled = tilelang.compile(func, target="npuir")
        compiled(A, B, Out)

        # CPU reference
        ref = reference(A.cpu(), B.cpu(), M, op)

        # 比对
        if torch.equal(Out.cpu(), ref):
            print(f"{op}: PASS")
        else:
            print(f"{op}: FAIL")
            all_pass = False
            print("Out:")
            print(Out.cpu())
            print("Ref:")
            print(ref)

    # --------------------------------------------------------
    # 总结
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
