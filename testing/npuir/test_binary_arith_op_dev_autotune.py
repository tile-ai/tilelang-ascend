import torch
import torch_npu
import tilelang
import tilelang.language as T
import os

os.environ["TILELANG_ASCEND_MODE"] = "Developer"

os.environ["TILELANG_AUTO_TUNING_CPU_UTILITIES"] = "0.8"
os.environ["TILELANG_AUTO_TUNING_CPU_COUNTS"] = "4"
os.environ["TILELANG_AUTO_TUNING_MAX_CPU_COUNT"] = "8"

torch.npu.set_device(6)
tilelang.cache.clear_cache()

dtype = "float16"

# --------------------------------------------------
# problem size
# --------------------------------------------------
N = 1024

# --------------------------------------------------
# 1. autotune configs
# --------------------------------------------------
def get_config():
    return [
        {"block_M": 4},
    ]

# --------------------------------------------------
# 2. reference program (CPU / PyTorch semantics)
# --------------------------------------------------
def ref_prog(A, B):
    return A + B

# --------------------------------------------------
# 3. input supplier (for autotune & correctness check)
# --------------------------------------------------
def supply_prog(N):
    torch.manual_seed(0)
    return [
        torch.randn(N).half().npu(),
        torch.randn(N).half().npu(),
    ]

# --------------------------------------------------
# 4. autotuned + jitted kernel
# --------------------------------------------------
@tilelang.autotune(
    configs=get_config(),
    ref_prog=ref_prog,
    supply_prog=supply_prog,
    atol=1e-3,
    rtol=1e-3,
)
@tilelang.jit(out_idx=[-1])
def binary_add(N, block_M, dtype="float16"):
    # ⚠️ 不要 return main，这样 JITKernel 才会生成
    @T.prim_func
    def main(
        A: T.Tensor((N,), dtype),
        B: T.Tensor((N,), dtype),
        Out: T.Tensor((N,), dtype),
    ):
        with T.Kernel(block_M, is_npu=True):
            A_ub = T.alloc_shared((N,), dtype)
            B_ub = T.alloc_shared((N,), dtype)
            Out_ub = T.alloc_shared((N,), dtype)

            # GM -> UB
            T.copy(A, A_ub)
            T.copy(B, B_ub)

            # binary add
            T.npuir_add(A_ub, B_ub, Out_ub)

            # UB -> GM
            T.copy(Out_ub, Out)
    return main

# --------------------------------------------------
# 5. trigger autotune
# --------------------------------------------------
# ⚠️ 不要提供 block_M，否则会跳过 autotune
func = binary_add(N)

print("Best Config:", func.get_tuner_result())

# --------------------------------------------------
# 6. correctness check
# --------------------------------------------------
A = torch.randn(N).half().npu()
B = torch.randn(N).half().npu()
Out = torch.zeros(N).half().npu()

func(A, B, Out)  # JITKernel 调用

ref = A + B
print("Correct:", torch.allclose(Out.cpu(), ref.cpu(), atol=1e-3, rtol=1e-3))
