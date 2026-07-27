import torch
import torch.nn.functional as F
import itertools

import tilelang
import tilelang.language as T

tilelang.cache.clear_cache()


seed = 42

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}


def get_configs(key_args, _key_kwargs=None):
    if not isinstance(key_args, (tuple, list)) or len(key_args) < 3:
        raise ValueError(f"get_configs: key_args must be (M, N, K), got {key_args}")
    M, N, K = int(key_args[0]), int(key_args[1]), int(key_args[2])
    if M <= 0 or N <= 0 or K <= 0:
        raise ValueError(f"get_configs: dimensions must be > 0, got M={M}, N={N}, K={K}")

    block_M = [bs for bs in [64, 128] if bs <= M]
    block_N = [bs for bs in [64, 128] if bs <= N]
    K_L1 = [bs for bs in [32, 64, 128] if bs <= K]

    if not block_M:
        block_M = [max(1, M)]
    if not block_N:
        block_N = [max(1, N)]
    if not K_L1:
        K_L1 = [max(1, K)]

    _configs = list(itertools.product(block_M, block_N, K_L1))
    if not _configs:
        return []

    configs = [{"block_M": c[0], "block_N": c[1], "K_L1": c[2]} for c in _configs]
    return configs


def supply_prog(params):
    M_val, K_val = int(params[0].shape[0]), int(params[0].shape[1])
    _, N_val = int(params[1].shape[0]), int(params[1].shape[1])
    torch.manual_seed(42)
    return [
        torch.randn(M_val, K_val).half().npu(),
        torch.randn(K_val, N_val).half().npu(),
    ]


def ref_prog(A, B):
    return A @ B


@tilelang.autotune(
    configs=get_configs,
    ref_prog=ref_prog,
    supply_prog=supply_prog,
    atol=1e-2,
    rtol=1e-2,
)
@tilelang.jit(out_idx=[-1], pass_configs=pass_configs)
def matmul(M, N, K, block_M, block_N, K_L1, dtype="float16", accum_dtype="float"):
    m_num = M // block_M
    n_num = N // block_N

    @T.prim_func
    def main(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, _):
            bx = cid // n_num
            by = cid % n_num

            A_L1 = T.alloc_shared((block_M, K_L1), dtype)
            B_L1 = T.alloc_shared((K_L1, block_N), dtype)

            C_L0 = T.alloc_fragment((block_M, block_N), accum_dtype)

            loop_k = T.ceildiv(K, K_L1)
            for k in T.serial(loop_k):
                T.copy(A[bx * block_M, k * K_L1], A_L1)
                T.copy(B[k * K_L1, by * block_N], B_L1)

                T.gemm_v0(A_L1, B_L1, C_L0, init=(k == 0))

            T.copy(C_L0, C[bx * block_M, by * block_N])

    return main


def im2col(input_tensor: torch.Tensor, KH: int, KW: int, stride: int, padding: int) -> torch.Tensor:
    B, C, H, W = input_tensor.shape
    HO = (H + 2 * padding - KH) // stride + 1
    WO = (W + 2 * padding - KW) // stride + 1

    input_flat = torch.zeros((C * KH * KW, B * HO * WO), dtype=input_tensor.dtype, device=input_tensor.device)

    for n in range(B):
        for i in range(HO):
            for j in range(WO):
                h_start = i * stride - padding
                w_start = j * stride - padding

                col_idx = n * HO * WO + i * WO + j
                row_idx = 0

                for c in range(C):
                    for m in range(KH):
                        for k in range(KW):
                            h = h_start + m
                            w = w_start + k

                            if 0 <= h < H and 0 <= w < W:
                                input_flat[row_idx, col_idx] = input_tensor[n, c, h, w]
                            else:
                                input_flat[row_idx, col_idx] = 0

                            row_idx += 1

    return input_flat


def conv_im2col_gemm(
    input_tensor: torch.Tensor, kernel: torch.Tensor, stride: int = 1, padding: int = 0, use_autotune: bool = False
) -> torch.Tensor:
    B, C, H, W = input_tensor.shape
    OC, C_k, KH, KW = kernel.shape
    assert C_k == C, "input channels mismatch: %d vs %d" % (C, C_k)
    HO = (H + 2 * padding - KH) // stride + 1
    WO = (W + 2 * padding - KW) // stride + 1

    # im2col
    input_flat = im2col(input_tensor, KH, KW, stride, padding)
    input_flat = input_flat.contiguous()

    kernel_flat = kernel.view(OC, -1)
    kernel_flat = kernel_flat.contiguous()

    M, K = kernel_flat.shape
    N = input_flat.shape[1]

    block_M, block_N, block_K = 128, 128, 128

    M_pad = ((M + block_M - 1) // block_M) * block_M
    N_pad = ((N + block_N - 1) // block_N) * block_N
    K_pad = ((K + block_K - 1) // block_K) * block_K

    need_pad = (M_pad > M) or (N_pad > N) or (K_pad > K)

    if need_pad:
        if M_pad > M or K_pad > K:
            kernel_padded = torch.zeros(M_pad, K_pad, dtype=kernel_flat.dtype, device=kernel_flat.device)
            kernel_padded[:M, :K] = kernel_flat
            kernel_flat = kernel_padded.contiguous()
        if K_pad > K or N_pad > N:
            input_padded = torch.zeros(K_pad, N_pad, dtype=input_flat.dtype, device=input_flat.device)
            input_padded[:K, :N] = input_flat
            input_flat = input_padded.contiguous()

    if use_autotune:
        func = matmul(M_pad, N_pad, K_pad)
        print("    GEMM(M=%d->%d, N=%d->%d, K=%d->%d) [autotune]" % (M, M_pad, N, N_pad, K, K_pad))
        print("    Best config:", func.get_tuner_result())
    else:
        func = matmul(M_pad, N_pad, K_pad, block_M=128, block_N=128, K_L1=128)
        print("    GEMM(M=%d->%d, N=%d->%d, K=%d->%d)" % (M, M_pad, N, N_pad, K, K_pad))

    print("    init successful!")
    output = func(kernel_flat, input_flat)

    output = output[:M, :N]
    output = output.view(OC, B, HO, WO).permute(1, 0, 2, 3)

    return output


def run_test(name, B_val, C_val, H_val, W_val, OC_val, KH_val, KW_val, stride_val=1, padding_val=0, use_autotune=False):
    torch.manual_seed(seed)

    input_t = torch.randn(B_val, C_val, H_val, W_val).half().npu()
    kernel_t = torch.randn(OC_val, C_val, KH_val, KW_val).half().npu()

    result = conv_im2col_gemm(input_t, kernel_t, stride_val, padding_val, use_autotune=use_autotune)
    ref = F.conv2d(input_t.cpu(), kernel_t.cpu(), stride=stride_val, padding=padding_val).npu()

    torch.testing.assert_close(result, ref, rtol=1e-2, atol=1e-2)
    print("    PASS: %s\n" % name)


if __name__ == "__main__":
    use_autotune = True
    tests = [
        dict(
            name="Case 1: Perfect alignment (M=N=K=128)",
            B=2,
            C=2,
            H=15,
            W=15,
            OC=128,
            KH=8,
            KW=8,
            stride=1,
            padding=0,
        ),
        dict(
            name="Case 2: M padding (OC=50 < 128)",
            B=1,
            C=2,
            H=32,
            W=32,
            OC=50,
            KH=3,
            KW=3,
            stride=1,
            padding=0,
        ),
        dict(
            name="Case 3: N padding (N=225 from B=1 C=4 H=17 W=17 KH=3 KW=3)",
            B=1,
            C=4,
            H=17,
            W=17,
            OC=128,
            KH=3,
            KW=3,
            stride=1,
            padding=0,
        ),
        dict(
            name="Case 4: K padding (K=27 from C=3 KH=3 KW=3, stride=2 pad=1)",
            B=2,
            C=3,
            H=28,
            W=28,
            OC=128,
            KH=3,
            KW=3,
            stride=2,
            padding=1,
        ),
        dict(
            name="Case 5: All-dim padding (M=64 N=225 K=27)",
            B=1,
            C=3,
            H=17,
            W=17,
            OC=64,
            KH=3,
            KW=3,
            stride=1,
            padding=0,
        ),
        dict(
            name="Case 6: Multi-block (M=256 N=2304 K=200)",
            B=4,
            C=8,
            H=28,
            W=28,
            OC=256,
            KH=5,
            KW=5,
            stride=1,
            padding=0,
        ),
    ]

    mode = "Autotune" if use_autotune else "Fixed Config"
    print("=" * 60)
    print("TileLang Ascend 2D Convolution Test Suite (%s, 6 scenarios)" % mode)
    print("=" * 60)
    for i, tc in enumerate(tests, 1):
        print("[%d/6] %s" % (i, tc["name"]))
        run_test(
            tc["name"],
            tc["B"],
            tc["C"],
            tc["H"],
            tc["W"],
            tc["OC"],
            tc["KH"],
            tc["KW"],
            tc.get("stride", 1),
            tc.get("padding", 0),
            use_autotune=use_autotune,
        )
    print("=" * 60)
    print("TEST PASSED!")
    print("=" * 60)
