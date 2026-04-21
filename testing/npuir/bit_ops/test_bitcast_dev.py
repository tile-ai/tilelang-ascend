import os
import pytest
import torch
import torch_npu # noqa: F401 # registers NPU backend for torch
import tilelang
import tilelang.language as T

os.environ["TILELANG_ASCEND_MODE"] = "Developer"
torch.npu.set_device(0)
tilelang.cache.clear_cache()

# ---------------------------------------------------------------------------
# Kernel definition
# ---------------------------------------------------------------------------
def bitcast_kernel(M, N, src_dtype, dst_dtype):
    """Build a kernel that reinterprets the bits of A (src_dtype) as dst_dtype.

    The bitcast is in-place on the UB buffer (the op's source region is `rw`).
    We then copy into a dst_dtype-typed UB buffer so the final UB->GM DMA
    preserves the reinterpreted element type.
    """

    @T.prim_func
    def main(
        A: T.Tensor((M, N), src_dtype),
        Out: T.Tensor((M, N), dst_dtype),
    ):
        with T.Kernel(1, is_npu=True) as (cid, _):
            a_ub = T.alloc_shared((M, N), src_dtype)
            out_ub = T.alloc_shared((M, N), dst_dtype)

            # GM -> UB
            T.copy(A, a_ub)

            # Bitwise reinterpretation: src_dtype -> dst_dtype (same bit width)
            T.npuir_bitcast(a_ub, dst_dtype)

            # Propagate reinterpreted bits into a dst_dtype-typed buffer
            T.copy(a_ub, out_ub)

            # UB -> GM
            T.copy(out_ub, Out)

    return main

# ---------------------------------------------------------------------------
# Input generation
# ---------------------------------------------------------------------------
def generate_tensor(shape, dtype, clear=False):
    """Generate a CPU tensor of the requested dtype."""
    if clear:
        return torch.zeros(shape, dtype=getattr(torch, dtype))
    if dtype in ("float32", "float16", "bfloat16"):
        return torch.randn(size=shape, dtype=getattr(torch, dtype))
    if dtype in ("int32", "int64", "int16"):
        return torch.randint(low=-2000, high=2000, size=shape,
                            dtype=getattr(torch, dtype))
    if dtype == "int8":
        return torch.randint(low=-128, high=127, size=shape,
                            dtype=getattr(torch, dtype))
    raise ValueError(f'Invalid parameter "dtype" is found : {dtype}')

# ---------------------------------------------------------------------------
# Positive tests: compile on npuir, run on NPU, numerically check against torch
# ---------------------------------------------------------------------------
SAME_WIDTH_PAIRS = [
("float16", "int16"), # 16-bit
("int16", "float16"),
("float32", "int32"), # 32-bit
("int32", "float32"),
]

@pytest.mark.parametrize("src_dtype,dst_dtype", SAME_WIDTH_PAIRS)
def test_bitcast_dev(src_dtype, dst_dtype):
    M, N = 4, 64

    # Input / output tensors on the NPU
    A = generate_tensor((M, N), src_dtype).npu()
    Out = torch.zeros((M, N), dtype=getattr(torch, dst_dtype)).npu()

    # Build + compile in developer mode
    func = bitcast_kernel(M, N, src_dtype, dst_dtype)
    compiled = tilelang.compile(func, target="npuir")
    print("A:\n", A.cpu())
    # Execute
    compiled(A, Out)

    print("New A:\n", A.cpu())

    # torch.Tensor.view(dtype) performs a bitwise reinterpretation, which
    # is the exact semantic of npuir_bitcast.
    ref = A.cpu().view(getattr(torch, dst_dtype))



    # bitcast must be bit-exact, so use torch.equal, not allclose.
    assert torch.equal(Out.cpu(), ref), (
    f"bitcast {src_dtype}->{dst_dtype} mismatch:\n"
    f"got:\n{Out.cpu()}\nref:\n{ref}"
    )



if __name__ == "__main__":
    # Run a quick positive + negative sanity check when invoked directly.
    for src, dst in SAME_WIDTH_PAIRS:
        print(f"Running bitcast {src} -> {dst}")
        test_bitcast_dev(src, dst)
        print(f" PASS: {src} -> {dst}")

