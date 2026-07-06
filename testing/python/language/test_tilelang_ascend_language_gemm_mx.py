import pytest
import tilelang
import tilelang.language as T
import torch
import numpy as np
import argparse
from tilelang.utils.target import determine_platform


# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------
def e8m0_to_float(e8m0_val):
    """Convert E8M0 value to float"""
    exp = int(e8m0_val) - 127
    return 2.0 ** exp


def compute_golden(A, B, scale_a, scale_b):
    """Compute golden reference using CPU - matching pypto's approach"""
    M, K = A.shape
    _, N = B.shape
    kScale = scale_a.shape[1]
    K_MX = kScale * 2
    
    # Convert scale from E8M0 to float
    sa_flat = scale_a.reshape(M, K_MX).astype(np.float64)
    sb_flat = scale_b.transpose(0, 2, 1).reshape(K_MX, N).astype(np.float64)
    
    for i in range(M):
        for j in range(K_MX):
            sa_flat[i, j] = 2.0 ** (int(sa_flat[i, j]) - 127)
    for i in range(K_MX):
        for j in range(N):
            sb_flat[i, j] = 2.0 ** (int(sb_flat[i, j]) - 127)
    
    # Expand scale to match K dimension (like pypto does with repeat_interleave)
    scale_a_expanded = np.repeat(sa_flat, 32, axis=1)  # [M, K_MX*32]
    scale_b_expanded = np.repeat(sb_flat, 32, axis=0)  # [K_MX*32, N]
    
    # Apply scale BEFORE matmul (like pypto does)
    A_scaled = A * scale_a_expanded
    B_scaled = B * scale_b_expanded
    
    # Do matmul with scaled data
    C = A_scaled @ B_scaled
    return C.astype(np.float32)


def make_kernel(M, N, K, block_M, block_N, K_L1, format="e5m2"):
    """Create MXFP8 GEMM kernel with L1 scale"""
    m_num = M // block_M
    n_num = N // block_N
    # Compute actual data size (padded if K < K_L1)
    K_data = K_L1 if K < K_L1 else K
    kScale_data = (K_data + 63) // 64  # Scale groups for actual data size
    kScale_per_iter = K_L1 // 64
    
    fp8_dtype = f"{format}_float8"

    pass_configs = {
        tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
        tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
    }

    @tilelang.jit(out_idx=[-1], target="pto", pass_configs=pass_configs)
    def mxfp8_matmul(M, N, K, block_M, block_N, K_L1):
        @T.prim_func
        def main(
                A: T.Tensor((M, K_data), fp8_dtype),
                B: T.Tensor((K_data, N), fp8_dtype),
                A_scale: T.Tensor((M, kScale_data, 2), "uint8"),
                B_scale: T.Tensor((kScale_data, N, 2), "uint8"),
                C: T.Tensor((M, N), "float32"),
        ):
            with T.Kernel(m_num * n_num, is_npu=True) as (cid, _):
                bx = cid // n_num
                by = cid % n_num
                A_L1 = T.alloc_L1((block_M, K_L1), fp8_dtype)
                B_L1 = T.alloc_L1((K_L1, block_N), fp8_dtype)
                SA_L1 = T.alloc_L1_scale_a((block_M, K_L1 // 32), "uint8")
                SB_L1 = T.alloc_L1_scale_b((K_L1 // 32, block_N), "uint8")
                C_L0 = T.alloc_L0C((block_M, block_N), "float32")
                with T.Scope("C"):
                    loop_k = T.ceildiv(K, K_L1)
                    for k in T.serial(loop_k):
                        T.copy(A[bx * block_M, k * K_L1], A_L1)
                        T.copy(B[k * K_L1, by * block_N], B_L1)
                        # Copy scale using 3D slice (view) to 2D tile
                        T.copy(A_scale[bx * block_M : (bx + 1) * block_M, k * kScale_per_iter : (k + 1) * kScale_per_iter, :], SA_L1, src_layout="MX_A_ND")
                        T.copy(B_scale[k * kScale_per_iter : (k + 1) * kScale_per_iter, by * block_N : (by + 1) * block_N, :], SB_L1, src_layout="MX_B_ND")
                        T.gemm_mx(A_L1, B_L1, C_L0, SA_L1, SB_L1, init=(k == 0), format=format)
                    T.copy(C_L0, C[bx * block_M, by * block_N])
        return main

    return mxfp8_matmul(M, N, K, block_M, block_N, K_L1)


def run_test(M, N, K, block_M, block_N, K_L1, scale_mode, format="e5m2", rel_tol=0.05):
    """Run a single test case"""
    assert M % block_M == 0
    assert N % block_N == 0
    assert K % K_L1 == 0 or K <= K_L1  # K can be <= K_L1 (will be padded)
    assert K_L1 % 64 == 0  # K_L1 must be multiple of 64

    device = "npu"
    torch.manual_seed(42)
    tilelang.cache.clear_cache()

    # Prepare FP8 data
    a_fp16 = torch.randn(M, K, dtype=torch.float16).to(device)
    b_fp16 = torch.randn(K, N, dtype=torch.float16).to(device)
    
    # Convert to appropriate FP8 format
    if format == "e5m2":
        a_fp8 = a_fp16.to(torch.float8_e5m2)
        b_fp8 = b_fp16.to(torch.float8_e5m2)
    elif format == "e4m3":
        a_fp8 = a_fp16.to(torch.float8_e4m3fn)
        b_fp8 = b_fp16.to(torch.float8_e4m3fn)
    else:
        raise ValueError(f"Unknown format: {format}")

    # Pad A and B to K_L1 if K < K_L1
    if K < K_L1:
        a_fp8_padded = torch.zeros((M, K_L1), dtype=a_fp8.dtype).to(device)
        b_fp8_padded = torch.zeros((K_L1, N), dtype=b_fp8.dtype).to(device)
        a_fp8_padded[:, :K] = a_fp8
        b_fp8_padded[:K, :] = b_fp8
        a_fp8 = a_fp8_padded
        b_fp8 = b_fp8_padded

    # Prepare scale data
    kScale = (K + 63) // 64  # Use ceil(K/64) for scale size (total scale groups)
    kScale_L1 = K_L1 // 64  # Use K_L1 for padded scale size
    np.random.seed(42)

    if scale_mode == "uniform_identity":
        sa = np.full((M, kScale, 2), 0x7F, dtype=np.uint8)
        sb = np.full((kScale, N, 2), 0x7F, dtype=np.uint8)
    elif scale_mode == "uniform_2x":
        sa = np.full((M, kScale, 2), 0x80, dtype=np.uint8)
        sb = np.full((kScale, N, 2), 0x7F, dtype=np.uint8)
    elif scale_mode == "random_narrow":
        sa = np.random.choice([0x7E, 0x7F, 0x80], size=(M, kScale, 2)).astype(np.uint8)
        sb = np.random.choice([0x7E, 0x7F, 0x80], size=(kScale, N, 2)).astype(np.uint8)
    elif scale_mode == "random_wide":
        sa = np.random.randint(124, 130, size=(M, kScale, 2)).astype(np.uint8)
        sb = np.random.randint(124, 130, size=(kScale, N, 2)).astype(np.uint8)
    elif scale_mode == "random":
        sa = np.random.choice([0x7D, 0x7E, 0x7F, 0x80, 0x81], size=(M, kScale, 2)).astype(np.uint8)
        sb = np.random.choice([0x7D, 0x7E, 0x7F, 0x80, 0x81], size=(kScale, N, 2)).astype(np.uint8)
    else:
        raise ValueError(f"Unknown scale_mode: {scale_mode}")

    # Pad scale data to K_L1 if K < K_L1
    if K < K_L1:
        # Pad with identity scale (0x7F = 1.0)
        sa_padded = np.full((M, kScale_L1, 2), 0x7F, dtype=np.uint8)
        sb_padded = np.full((kScale_L1, N, 2), 0x7F, dtype=np.uint8)
        sa_padded[:, :kScale, :] = sa
        sb_padded[:kScale, :, :] = sb
        sa = sa_padded
        sb = sb_padded

    a_scale_t = torch.from_numpy(sa).to(device)
    b_scale_t = torch.from_numpy(sb).to(device)

    # Compile and run
    func = make_kernel(M, N, K, block_M, block_N, K_L1, format=format)
    c_npu = func(a_fp8, b_fp8, a_scale_t, b_scale_t)

    # Compute golden reference
    a_np = a_fp8.float().cpu().numpy()
    b_np = b_fp8.float().cpu().numpy()
    c_ref = compute_golden(a_np, b_np, sa, sb)
    c_ref_t = torch.from_numpy(c_ref).to(device)
    
    # Debug output
    print(f"\nDebug: K={K}, K_L1={K_L1}, kScale={kScale}, kScale_L1={kScale_L1}")
    print(f"  sa.shape={sa.shape}, sb.shape={sb.shape}")
    print(f"  a_np.shape={a_np.shape}, b_np.shape={b_np.shape}")
    print(f"  c_ref.shape={c_ref.shape}, c_npu.shape={c_npu.shape}")
    print(f"  c_ref[0,:5]={c_ref[0,:5]}")
    print(f"  c_npu[0,:5]={c_npu[0,:5].cpu().numpy()}")

    # Verify accuracy
    max_diff = (c_npu - c_ref_t).abs().max().item()
    max_val = c_ref_t.abs().max().item()
    rel_err = max_diff / (max_val + 1e-10)

    assert rel_err < rel_tol, f"Relative error {rel_err:.6f} exceeds tolerance {rel_tol}"
    assert not np.any(np.isnan(c_npu.cpu().numpy())), "Output contains NaN values"

    print(f"Test Passed! max_diff={max_diff:.4f}, max_val={max_val:.4f}, rel_err={rel_err:.6f}")


# -----------------------------------------------------------------------------
# Pytest entry points
# -----------------------------------------------------------------------------
@pytest.mark.skipif(determine_platform() != "A5", reason="Requires A5 platform")
@pytest.mark.parametrize("M,N,K,block_M,block_N,K_L1,scale_mode,format", [
    # Basic tests (E5M2)
    (128, 128, 64, 128, 128, 64, "uniform_identity", "e5m2"),
    (128, 128, 64, 128, 128, 64, "uniform_2x", "e5m2"),
    # Non-square matrices (E5M2)
    (256, 128, 128, 128, 128, 64, "random_narrow", "e5m2"),
    (128, 256, 128, 128, 128, 64, "random_narrow", "e5m2"),
    # K-split tests (E5M2)
    (128, 128, 192, 128, 128, 64, "random", "e5m2"),
    (128, 128, 256, 128, 128, 64, "random_wide", "e5m2"),
    (128, 128, 512, 128, 128, 64, "random", "e5m2"),
    (128, 128, 1024, 128, 128, 64, "random_narrow", "e5m2"),
    # K_L1=128 tests (E5M2)
    (128, 128, 256, 128, 128, 128, "random", "e5m2"),
    # Multi-block tests (E5M2)
    (256, 256, 128, 128, 128, 64, "random", "e5m2"),
    (256, 256, 256, 128, 128, 64, "random_wide", "e5m2"),
    # block_N=64 tests (E5M2)
    (128, 64, 128, 128, 64, 64, "random", "e5m2"),
    (256, 64, 128, 128, 64, 64, "random", "e5m2"),
    # Extreme scale values (E5M2)
    (128, 128, 128, 128, 128, 64, "random_wide", "e5m2"),
    # E4M3 format tests
    (128, 128, 64, 128, 128, 64, "uniform_identity", "e4m3"),
    (128, 128, 128, 128, 128, 64, "random", "e4m3"),
    (256, 128, 192, 128, 128, 64, "random_narrow", "e4m3"),
    (128, 256, 256, 128, 128, 64, "random_wide", "e4m3"),
    (128, 128, 512, 128, 128, 64, "random", "e4m3"),
    (256, 256, 128, 128, 128, 64, "random", "e4m3"),
    # Non-aligned K tests (K not multiple of 64, will be padded to K_L1)
    (128, 128, 65, 128, 128, 128, "random", "e5m2"),      # K=65 -> K_L1=128
    (128, 128, 127, 128, 128, 128, "random", "e5m2"),     # K=127 -> K_L1=128
    (128, 128, 129, 128, 128, 192, "random", "e5m2"),     # K=129 -> K_L1=192
    (128, 128, 193, 128, 128, 256, "random", "e5m2"),     # K=193 -> K_L1=256
    (128, 128, 257, 128, 128, 320, "random", "e5m2"),     # K=257 -> K_L1=320
    (128, 128, 383, 128, 128, 448, "random", "e5m2"),     # K=383 -> K_L1=448
])
def test_gemm_mx(M, N, K, block_M, block_N, K_L1, scale_mode, format):
    """Test MXFP8 GEMM with L1 scale"""
    run_test(M, N, K, block_M, block_N, K_L1, scale_mode, format=format)


# -----------------------------------------------------------------------------
# Standalone command-line entry point
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MXFP8 GEMM Test Suite")
    parser.add_argument("--m", type=int, default=128, help="Matrix M dimension")
    parser.add_argument("--n", type=int, default=128, help="Matrix N dimension")
    parser.add_argument("--k", type=int, default=128, help="Matrix K dimension")
    parser.add_argument("--block-m", type=int, default=128, help="Block M size")
    parser.add_argument("--block-n", type=int, default=128, help="Block N size")
    parser.add_argument("--k-l1", type=int, default=64, help="K_L1 size")
    parser.add_argument("--scale-mode", type=str, default="random",
                        choices=["uniform_identity", "uniform_2x", "random_narrow", 
                                 "random_wide", "random"],
                        help="Scale generation mode")
    parser.add_argument("--format", type=str, default="e5m2",
                        choices=["e5m2", "e4m3"],
                        help="FP8 data format")
    args = parser.parse_args()

    if determine_platform() != "A5":
        print(f"[SKIP] Requires A5 platform; detected: {determine_platform()}")
        exit(0)

    run_test(args.m, args.n, args.k, args.block_m, args.block_n, args.k_l1, 
             args.scale_mode, format=args.format)
