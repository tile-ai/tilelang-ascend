import pytest
import tilelang
import tilelang.language as T
import torch
import argparse


def bit_pack_mask_cpu(mask_bool: torch.Tensor) -> torch.Tensor:
    """Pack Bool mask into uint8 on CPU (8 bits per byte)"""
    M, N = mask_bool.shape
    mask_reshaped = mask_bool.view(M, N // 8, 8)
    mask_packed = torch.zeros((M, N // 8), dtype=torch.uint8)
    for i in range(8):
        bit_val = mask_reshaped[..., i].to(torch.uint8)
        mask_packed |= bit_val << i
    return mask_packed


def select_kernel_mod1(M, N, block_M, block_N, dtype="float16"):
    m_num, n_num = M // block_M, N // block_N
    mask_width = N // 8
    block_mask_width = block_N // 8
    VEC_NUM = 2
    sub_block_m = block_M // VEC_NUM

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),  # type: ignore
        Mask: T.Tensor((M, mask_width), "uint8"),  # type: ignore
        C: T.Tensor((M, N), dtype),  # type: ignore
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx, by = cid // n_num, cid % n_num

            # Allocate Unified Buffer (UB)
            a_ub = T.alloc_ub((sub_block_m, block_N), dtype)
            c_ub = T.alloc_ub((sub_block_m, block_N), dtype)
            mask_ub = T.alloc_ub((sub_block_m, block_mask_width), "uint8")

            # Calculate coordinate offsets
            offset_m = bx * block_M + vid * sub_block_m
            offset_n = by * block_N
            offset_mask_n = offset_n // 8

            with T.Scope("V"):
                # 1. Copy data from GM to UB
                T.copy(A[offset_m : offset_m + sub_block_m, offset_n : offset_n + block_N], a_ub)
                T.copy(Mask[offset_m : offset_m + sub_block_m, offset_mask_n : offset_mask_n + block_mask_width], mask_ub)

                T.barrier_all()

                # 2. Execute Select instruction: select src0(A) or src1(B) based on bits
                T.tile.select(c_ub, mask_ub, a_ub, 1.0, "VSEL_TENSOR_SCALAR_MODE")

                T.barrier_all()

                # 3. Copy results back to GM
                T.copy(c_ub, C[offset_m : offset_m + sub_block_m, offset_n : offset_n + block_N])

    return main


def select_kernel_mod2(M, N, block_M, block_N, dtype="float16"):
    m_num, n_num = M // block_M, N // block_N
    mask_width = N // 8
    block_mask_width = block_N // 8
    VEC_NUM = 2
    sub_block_m = block_M // VEC_NUM

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),  # type: ignore
        B: T.Tensor((M, N), dtype),  # type: ignore
        Mask: T.Tensor((M, mask_width), "uint8"),  # type: ignore
        C: T.Tensor((M, N), dtype),  # type: ignore
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx, by = cid // n_num, cid % n_num

            # Allocate Unified Buffer (UB)
            a_ub = T.alloc_ub((sub_block_m, block_N), dtype)
            b_ub = T.alloc_ub((sub_block_m, block_N), dtype)
            c_ub = T.alloc_ub((sub_block_m, block_N), dtype)
            mask_ub = T.alloc_ub((sub_block_m, block_mask_width), "uint8")

            # Calculate coordinate offsets
            offset_m = bx * block_M + vid * sub_block_m
            offset_n = by * block_N
            offset_mask_n = offset_n // 8

            with T.Scope("V"):
                # 1. Copy data from GM to UB
                T.copy(A[offset_m : offset_m + sub_block_m, offset_n : offset_n + block_N], a_ub)
                T.copy(B[offset_m : offset_m + sub_block_m, offset_n : offset_n + block_N], b_ub)
                T.copy(Mask[offset_m : offset_m + sub_block_m, offset_mask_n : offset_mask_n + block_mask_width], mask_ub)

                T.barrier_all()

                # 2. Execute Select instruction: select src0(A) or src1(B) based on bits
                T.tile.select(c_ub, mask_ub, a_ub, b_ub, "VSEL_TENSOR_TENSOR_MODE")

                T.barrier_all()

                # 3. Copy results back to GM
                T.copy(c_ub, C[offset_m : offset_m + sub_block_m, offset_n : offset_n + block_N])

    return main


def run_test_mod1(M, N, block_M, block_N, target):
    device = "npu"
    torch.manual_seed(0)
    tilelang.cache.clear_cache()

    # 1. Compile the operator
    func_def = select_kernel_mod1(M, N, block_M, block_N, dtype="float16")
    func = tilelang.compile(func_def, out_idx=[-1], target=target)

    # 2. Prepare data
    a = torch.randn(M, N).to(device).half()
    b = torch.ones(M, N).to(device).half()

    # Generate and pack Mask on CPU to avoid NPU bitwise operation limitations
    raw_mask_bool_cpu = torch.randint(0, 2, (M, N)).bool()
    mask_packed = bit_pack_mask_cpu(raw_mask_bool_cpu).to(device)

    # 3. Run the operator
    torch.npu.synchronize()
    c = func(a, mask_packed)

    # 4. Verify accuracy
    ref_c = torch.where(raw_mask_bool_cpu.to(device), a, b)
    torch.testing.assert_close(c, ref_c, rtol=1e-3, atol=1e-3)
    print("Test Passed")


def run_test_mod2(M, N, block_M, block_N, target):
    device = "npu"
    torch.manual_seed(0)
    tilelang.cache.clear_cache()

    # 1. Compile the operator
    func_def = select_kernel_mod2(M, N, block_M, block_N, dtype="float16")
    func = tilelang.compile(func_def, out_idx=[-1], target=target)

    # 2. Prepare data
    a = torch.randn(M, N).to(device).half()
    b = torch.randn(M, N).to(device).half()

    # Generate and pack Mask on CPU to avoid NPU bitwise operation limitations
    raw_mask_bool_cpu = torch.randint(0, 2, (M, N)).bool()
    mask_packed = bit_pack_mask_cpu(raw_mask_bool_cpu).to(device)

    # 3. Run the operator
    torch.npu.synchronize()
    c = func(a, b, mask_packed)

    # 4. Verify accuracy
    ref_c = torch.where(raw_mask_bool_cpu.to(device), a, b)
    torch.testing.assert_close(c, ref_c, rtol=1e-3, atol=1e-3)
    print("Test Passed")


# -----------------------------------------------------------------------------
# Pytest entry point
# -----------------------------------------------------------------------------
@pytest.mark.parametrize("target", ["ascendc", "pto"])
@pytest.mark.parametrize("shape", [(1024, 1024), (512, 256)])
def test_select_tensor_op(target, shape):
    M, N = shape
    # Alignment check for N dimension
    if N % 8 != 0:
        pytest.skip("N must be multiple of 8")
    run_test_mod2(M, N, 128, 256, target=target)


@pytest.mark.parametrize("target", ["ascendc", "pto"])
@pytest.mark.parametrize("shape", [(1024, 1024), (512, 256)])
def test_select_scalar_op(target, shape):
    M, N = shape
    # Alignment check for N dimension
    if N % 8 != 0:
        pytest.skip("N must be multiple of 8")
    run_test_mod1(M, N, 128, 256, target=target)


# -----------------------------------------------------------------------------
# Standalone command-line entry point
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=1024)
    parser.add_argument("--n", type=int, default=1024)
    parser.add_argument("--target", type=str, choices=["ascendc", "pto"], default="ascendc")
    args = parser.parse_args()

    # Align N to a multiple of 8
    final_n = args.n if args.n % 8 == 0 else (args.n // 8 + 1) * 8
    run_test_mod1(args.m, final_n, 128, 256, target=args.target)
    run_test_mod2(args.m, final_n, 128, 256, target=args.target)
