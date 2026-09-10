import pytest
import tilelang
import tilelang.language as T
import torch
import argparse

TORCH_DTYPE = {"float16": torch.float16, "float32": torch.float32}
RTOL = {"float16": 1e-3, "float32": 1e-4}
ATOL = {"float16": 1e-3, "float32": 1e-4}
CMPMASK_SPR_MAX = {"float16": 128, "float32": 64}

PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}


def bit_pack_mask_cpu(mask_bool: torch.Tensor) -> torch.Tensor:
    """Pack Bool mask into uint8 on CPU (8 bits per byte, bit i controls element i)"""
    last = mask_bool.shape[-1]
    leading = mask_bool.shape[:-1]
    reshaped = mask_bool.reshape(-1, last // 8, 8)
    packed = torch.zeros(reshaped.shape[:2], dtype=torch.uint8)
    for i in range(8):
        packed |= reshaped[:, :, i].to(torch.uint8) << i
    return packed.reshape(*leading, last // 8)


# -----------------------------------------------------------------------------
# 2D Kernels
# -----------------------------------------------------------------------------
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

            # 1. Copy data from GM to UB
            T.copy(A[offset_m : offset_m + sub_block_m, offset_n : offset_n + block_N], a_ub)
            T.copy(Mask[offset_m : offset_m + sub_block_m, offset_mask_n : offset_mask_n + block_mask_width], mask_ub)

            # 2. Execute Select instruction: select src0(A) or scalar src1 based on bits
            T.tile.select(c_ub, mask_ub, a_ub, 1.0, "VSEL_TENSOR_SCALAR_MODE")

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

            # 1. Copy data from GM to UB
            T.copy(A[offset_m : offset_m + sub_block_m, offset_n : offset_n + block_N], a_ub)
            T.copy(B[offset_m : offset_m + sub_block_m, offset_n : offset_n + block_N], b_ub)
            T.copy(Mask[offset_m : offset_m + sub_block_m, offset_mask_n : offset_mask_n + block_mask_width], mask_ub)

            # 2. Execute Select instruction: select src0(A) or src1(B) based on bits
            T.tile.select(c_ub, mask_ub, a_ub, b_ub, "VSEL_TENSOR_TENSOR_MODE")

            # 3. Copy results back to GM
            T.copy(c_ub, C[offset_m : offset_m + sub_block_m, offset_n : offset_n + block_N])

    return main


# -----------------------------------------------------------------------------
# 1D Kernels
# -----------------------------------------------------------------------------
def select_kernel_1d_scalar(N, dtype="float16"):
    mask_width = N // 8

    @T.prim_func
    def main(
        A: T.Tensor((N,), dtype),  # type: ignore
        Mask: T.Tensor((mask_width,), "uint8"),  # type: ignore
        C: T.Tensor((N,), dtype),  # type: ignore
    ):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            a_ub = T.alloc_ub((N,), dtype)
            c_ub = T.alloc_ub((N,), dtype)
            mask_ub = T.alloc_ub((mask_width,), "uint8")

            T.copy(A, a_ub)
            T.copy(Mask, mask_ub)

            T.tile.select(c_ub, mask_ub, a_ub, 1.0, "VSEL_TENSOR_SCALAR_MODE")

            T.copy(c_ub, C)

    return main


def select_kernel_1d_tensor(N, dtype="float16"):
    mask_width = N // 8

    @T.prim_func
    def main(
        A: T.Tensor((N,), dtype),  # type: ignore
        B: T.Tensor((N,), dtype),  # type: ignore
        Mask: T.Tensor((mask_width,), "uint8"),  # type: ignore
        C: T.Tensor((N,), dtype),  # type: ignore
    ):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            a_ub = T.alloc_ub((N,), dtype)
            b_ub = T.alloc_ub((N,), dtype)
            c_ub = T.alloc_ub((N,), dtype)
            mask_ub = T.alloc_ub((mask_width,), "uint8")

            T.copy(A, a_ub)
            T.copy(B, b_ub)
            T.copy(Mask, mask_ub)

            T.tile.select(c_ub, mask_ub, a_ub, b_ub, "VSEL_TENSOR_TENSOR_MODE")

            T.copy(c_ub, C)

    return main


# -----------------------------------------------------------------------------
# CMPMASK Kernel (VSEL_CMPMASK_SPR + T.tile.compare)
# -----------------------------------------------------------------------------
def select_kernel_mod3(N, dtype="float16"):
    mask_width = N // 8

    @T.prim_func
    def main(
        A: T.Tensor((N,), dtype),  # type: ignore
        B: T.Tensor((N,), dtype),  # type: ignore
        C: T.Tensor((N,), dtype),  # type: ignore
    ):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            a_ub = T.alloc_ub((N,), dtype)
            b_ub = T.alloc_ub((N,), dtype)
            c_ub = T.alloc_ub((N,), dtype)
            cmp_ub = T.alloc_ub((mask_width,), "uint8")

            T.copy(A, a_ub)
            T.copy(B, b_ub)

            T.tile.compare(cmp_ub, a_ub, b_ub, "GT")
            T.tile.select(c_ub, cmp_ub, a_ub, b_ub, "VSEL_CMPMASK_SPR")

            T.copy(c_ub, C)

    return main


# -----------------------------------------------------------------------------
# Run functions
# -----------------------------------------------------------------------------
def run_test_mod1(M, N, block_M, block_N, dtype, target):
    device = "npu"
    torch.manual_seed(0)
    tilelang.cache.clear_cache()

    # 1. Compile the operator
    func_def = select_kernel_mod1(M, N, block_M, block_N, dtype=dtype)
    func = tilelang.compile(func_def, out_idx=[-1], pass_configs=PASS_CONFIGS, target=target)

    # 2. Prepare data
    tdtype = TORCH_DTYPE[dtype]
    a = torch.randn(M, N).to(device).to(tdtype)
    b = torch.ones(M, N).to(device).to(tdtype)

    # Generate and pack Mask on CPU to avoid NPU bitwise operation limitations
    raw_mask_bool_cpu = torch.randint(0, 2, (M, N)).bool()
    mask_packed = bit_pack_mask_cpu(raw_mask_bool_cpu).to(device)

    # 3. Run the operator
    torch.npu.synchronize()
    c = func(a, mask_packed)

    # 4. Verify accuracy
    ref_c = torch.where(raw_mask_bool_cpu.to(device), a, b)
    torch.testing.assert_close(c, ref_c, rtol=RTOL[dtype], atol=ATOL[dtype])
    print("Test Passed")


def run_test_mod2(M, N, block_M, block_N, dtype, target):
    device = "npu"
    torch.manual_seed(0)
    tilelang.cache.clear_cache()

    # 1. Compile the operator
    func_def = select_kernel_mod2(M, N, block_M, block_N, dtype=dtype)
    func = tilelang.compile(func_def, out_idx=[-1], pass_configs=PASS_CONFIGS, target=target)

    # 2. Prepare data
    tdtype = TORCH_DTYPE[dtype]
    a = torch.randn(M, N).to(device).to(tdtype)
    b = torch.randn(M, N).to(device).to(tdtype)

    # Generate and pack Mask on CPU to avoid NPU bitwise operation limitations
    raw_mask_bool_cpu = torch.randint(0, 2, (M, N)).bool()
    mask_packed = bit_pack_mask_cpu(raw_mask_bool_cpu).to(device)

    # 3. Run the operator
    torch.npu.synchronize()
    c = func(a, b, mask_packed)

    # 4. Verify accuracy
    ref_c = torch.where(raw_mask_bool_cpu.to(device), a, b)
    torch.testing.assert_close(c, ref_c, rtol=RTOL[dtype], atol=ATOL[dtype])
    print("Test Passed")


def run_test_1d_scalar(N, dtype, target):
    device = "npu"
    torch.manual_seed(0)
    tilelang.cache.clear_cache()

    # 1. Compile the operator
    func_def = select_kernel_1d_scalar(N, dtype=dtype)
    func = tilelang.compile(func_def, out_idx=[-1], pass_configs=PASS_CONFIGS, target=target)

    # 2. Prepare data
    tdtype = TORCH_DTYPE[dtype]
    a = torch.randn(N).to(device).to(tdtype)
    b = torch.ones(N, dtype=tdtype, device=device)

    # Generate and pack Mask on CPU to avoid NPU bitwise operation limitations
    raw_mask_bool_cpu = torch.randint(0, 2, (N,)).bool()
    mask_packed = bit_pack_mask_cpu(raw_mask_bool_cpu).to(device)

    # 3. Run the operator
    torch.npu.synchronize()
    c = func(a, mask_packed)

    # 4. Verify accuracy
    ref_c = torch.where(raw_mask_bool_cpu.to(device), a, b)
    torch.testing.assert_close(c, ref_c, rtol=RTOL[dtype], atol=ATOL[dtype])
    print("Test Passed")


def run_test_1d_tensor(N, dtype, target):
    device = "npu"
    torch.manual_seed(0)
    tilelang.cache.clear_cache()

    # 1. Compile the operator
    func_def = select_kernel_1d_tensor(N, dtype=dtype)
    func = tilelang.compile(func_def, out_idx=[-1], pass_configs=PASS_CONFIGS, target=target)

    # 2. Prepare data
    tdtype = TORCH_DTYPE[dtype]
    a = torch.randn(N).to(device).to(tdtype)
    b = torch.randn(N).to(device).to(tdtype)

    # Generate and pack Mask on CPU to avoid NPU bitwise operation limitations
    raw_mask_bool_cpu = torch.randint(0, 2, (N,)).bool()
    mask_packed = bit_pack_mask_cpu(raw_mask_bool_cpu).to(device)

    # 3. Run the operator
    torch.npu.synchronize()
    c = func(a, b, mask_packed)

    # 4. Verify accuracy
    ref_c = torch.where(raw_mask_bool_cpu.to(device), a, b)
    torch.testing.assert_close(c, ref_c, rtol=RTOL[dtype], atol=ATOL[dtype])
    print("Test Passed")


def run_test_mod3(dtype, target):
    device = "npu"
    torch.manual_seed(0)
    tilelang.cache.clear_cache()

    # 1. Compile the operator
    N = CMPMASK_SPR_MAX[dtype]
    func_def = select_kernel_mod3(N, dtype=dtype)
    func = tilelang.compile(func_def, out_idx=[-1], pass_configs=PASS_CONFIGS, target=target)

    # 2. Prepare data
    tdtype = TORCH_DTYPE[dtype]
    a = torch.randn(N).to(device).to(tdtype)
    b = torch.randn(N).to(device).to(tdtype)

    # 3. Run the operator
    torch.npu.synchronize()
    c = func(a, b)

    # 4. Verify accuracy (select larger element via compare GT + VSEL_CMPMASK_SPR)
    ref_c = torch.where(a > b, a, b)
    torch.testing.assert_close(c, ref_c, rtol=RTOL[dtype], atol=ATOL[dtype])
    print("Test Passed")


# -----------------------------------------------------------------------------
# Pytest entry point
# -----------------------------------------------------------------------------
BLOCK_M = {"float16": 128, "float32": 32}
BLOCK_N = {"float16": 256, "float32": 256}


@pytest.mark.parametrize("dtype", ["float16", "float32"])
@pytest.mark.parametrize(
    "target",
    ["ascendc", pytest.param("pto", marks=pytest.mark.low_priority)],
)
@pytest.mark.parametrize("shape", [(1024, 1024), (512, 256), (256,)])
def test_select_tensor_op(dtype, target, shape):
    if len(shape) == 1:
        N = shape[0]
        if N % 8 != 0:
            pytest.skip("N must be multiple of 8")
        run_test_1d_tensor(N, dtype, target)
    else:
        M, N = shape
        if N % 8 != 0:
            pytest.skip("N must be multiple of 8")
        run_test_mod2(M, N, BLOCK_M[dtype], BLOCK_N[dtype], dtype, target)


@pytest.mark.parametrize("dtype", ["float16", "float32"])
@pytest.mark.parametrize(
    "target",
    ["ascendc", pytest.param("pto", marks=pytest.mark.low_priority)],
)
@pytest.mark.parametrize("shape", [(1024, 1024), (512, 256), (256,)])
def test_select_scalar_op(dtype, target, shape):
    if len(shape) == 1:
        N = shape[0]
        if N % 8 != 0:
            pytest.skip("N must be multiple of 8")
        run_test_1d_scalar(N, dtype, target)
    else:
        M, N = shape
        if N % 8 != 0:
            pytest.skip("N must be multiple of 8")
        run_test_mod1(M, N, BLOCK_M[dtype], BLOCK_N[dtype], dtype, target)


@pytest.mark.low_priority
@pytest.mark.parametrize("dtype", ["float16", "float32"])
@pytest.mark.parametrize(
    "target",
    [
        "ascendc",
        pytest.param("pto", marks=pytest.mark.ci_skip),
    ],
)
def test_select_cmpmask_op(dtype, target):
    run_test_mod3(dtype, target)


# -----------------------------------------------------------------------------
# Standalone command-line entry point
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=1024)
    parser.add_argument("--n", type=int, default=1024)
    parser.add_argument("--dtype", type=str, choices=["float16", "float32"], default="float16")
    parser.add_argument("--target", type=str, choices=["ascendc", "pto"], default="ascendc")
    args = parser.parse_args()

    # Align N to a multiple of 8
    final_n = args.n if args.n % 8 == 0 else (args.n // 8 + 1) * 8
    run_test_mod1(args.m, final_n, BLOCK_M[args.dtype], BLOCK_N[args.dtype], dtype=args.dtype, target=args.target)
    run_test_mod2(args.m, final_n, BLOCK_M[args.dtype], BLOCK_N[args.dtype], dtype=args.dtype, target=args.target)
    run_test_mod3(args.dtype, args.target)
