import torch
import torch.nn.functional as F

import tilelang
import tilelang.language as T
import torch

tilelang.cache.clear_cache()


B, C, H, W, OC, KH, KW, stride, padding, seed  = 2, 2, 15, 15, 128, 8, 8, 1, 0, 42
HO = (H + 2 * padding - KH) // stride + 1
WO = (W + 2 * padding - KW) // stride + 1


pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}


@tilelang.jit(out_idx=[-1], pass_configs=pass_configs)
def matmul(M, N, K, block_M=128, block_N=256, block_K=64, dtype="float16", accum_dtype="float"):
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

            A_L1 = T.alloc_shared((block_M, block_K), dtype)
            B_L1 = T.alloc_shared((block_K, block_N), dtype)

            C_L0 = T.alloc_fragment((block_M, block_N), accum_dtype)

            loop_k = T.ceildiv(K, block_K)
            for k in T.serial(loop_k):
                T.copy(A[bx * block_M, k * block_K], A_L1)
                T.copy(B[k * block_K, by * block_N], B_L1)

                T.gemm_v0(A_L1, B_L1, C_L0, init=(k == 0))

            T.copy(C_L0, C[bx * block_M, by * block_N])

    return main


def im2col(input_tensor: torch.Tensor, KH: int, KW: int, 
           stride: int, padding: int) -> torch.Tensor:
    
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


def conv_im2col_gemm(input_tensor: torch.Tensor, kernel: torch.Tensor, 
                     stride: int = 1, padding: int = 0) -> torch.Tensor:
    
    # im2col
    input_flat = im2col(input_tensor, KH, KW, stride, padding)
    input_flat = input_flat.contiguous()
    
    kernel_flat = kernel.view(OC, -1)
    kernel_flat = kernel_flat.contiguous()

    func = matmul(kernel_flat.shape[0], input_flat.shape[1], kernel_flat.shape[1], 128, 128, 128)
    print("init successful!")
    ouput = func(kernel_flat, input_flat)

    output = ouput.view(OC, B, HO, WO).permute(1, 0, 2, 3)
    
    return output


torch.manual_seed(seed)

input_torch = torch.randn(B, C, H, W).half().npu()
kernel_torch = torch.randn(OC, C, KH, KW).half().npu()

result_np = conv_im2col_gemm(input_torch, kernel_torch, stride, padding)
# Use CPU for reference to avoid NPU environment issues
result_torch = F.conv2d(input_torch.cpu(), kernel_torch.cpu(), stride=stride, padding=padding).npu()

torch.testing.assert_close(result_np, result_torch, rtol=1e-2, atol=1e-2)
print("Kernel Output Match!")