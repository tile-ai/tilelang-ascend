import argparse
from typing import Literal

import tilelang as tl
import tilelang.language as T
import torch

@tl.jit(
    out_idx=[3],
    workspace_idx=[4],
    pass_configs={
        tl.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
        tl.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
        tl.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    }
)
def simple_quant_batch_matmul(
    Batch:int, M:int, N:int, K:int, scale_size:Literal["1", "N"],
    block_M:int, block_N:int, block_K:int, 
    in_dtype: Literal["int8"] = "int8", out_dtype: Literal["float16", "bfloat16"] = "float16", 
    accum_dtype: Literal["int32"] = "int32", scale_dtype: Literal["float32"] = "float32"
):
    """Simple QuantMatmul implementation with per-tensor / per-channel quantization scale"""
    
    VEC_NUM = 2
    CAST_MODE = "CAST_RINT"

    N_scale = N if scale_size == "N" else 1
    
    m_num = T.ceildiv(M, block_M)
    n_num = T.ceildiv(N, block_N)
    k_num = T.ceildiv(K, block_K)

    block_M_2 = T.ceildiv(block_M, VEC_NUM)

    @T.prim_func
    def main(
            A: T.Tensor([Batch, M, K], in_dtype),              # type: ignore
            B: T.Tensor([Batch, K, N], in_dtype),              # type: ignore
            scale: T.Tensor([N_scale], scale_dtype),       # type: ignore
            C: T.Tensor([Batch, M, N], out_dtype),             # type: ignore
            workspace_1: T.Tensor([Batch, M, N], accum_dtype), # type: ignore
    ):
        with T.Kernel(Batch * m_num * n_num, is_npu=True) as (cid, vid):
            bb = cid // (m_num * n_num)
            bm = (cid % (m_num * n_num)) // n_num
            bn = (cid % (m_num * n_num)) % n_num

            A_L1 = T.alloc_L1([block_M, block_K], in_dtype)
            B_L1 = T.alloc_L1([block_K, block_N], in_dtype)

            C_L0 = T.alloc_L0C([block_M, block_N], accum_dtype)

            c_ub = T.alloc_ub([block_M_2, block_N], accum_dtype)
            c_scale = T.alloc_ub([block_M_2, block_N], scale_dtype)
            c_out = T.alloc_ub([block_M_2, block_N], out_dtype)
            
            scale_ub = T.alloc_ub([block_N], scale_dtype)

            for bk in T.serial(k_num):
                T.copy(A[bb, bm * block_M, bk * block_K], A_L1)
                T.copy(B[bb, bk * block_K, bn * block_N], B_L1)

                T.gemm_v0(A_L1, B_L1, C_L0, init=(bk == 0))

            T.copy(C_L0, workspace_1[bb, bm * block_M, bn * block_N])
            
            T.copy(workspace_1[bb, bm * block_M + vid * block_M_2, bn * block_N], c_ub)

            if scale_size == "N":
                T.copy(scale[bn * block_N], scale_ub)
            else:
                T.copy(scale[0], scale_ub)
                T.tile.fill(scale_ub, scale_ub[0])  # scale_ub (1,) => (block_N,)
            
            if accum_dtype != scale_dtype:
                T.tile.cast_tl(c_scale, c_ub, mode=CAST_MODE, count=block_M_2 * block_N)
            else:
                T.copy(c_ub, c_scale)

            for bm_v, bn_v in T.Parallel(block_M_2, block_N):
                c_scale[bm_v, bn_v] *= scale_ub[bn_v]

            if out_dtype != scale_dtype:
                T.tile.cast_tl(c_out, c_scale, mode=CAST_MODE, count=block_M_2 * block_N)
            else:
                T.copy(c_scale, c_out)

            T.copy(c_out, C[bb, bm * block_M + vid * block_M_2, bn * block_N])

    return main

def ref_program(A, B, scale, out_dtype, accum_dtype):
    C = A.to(accum_dtype) @ B.to(accum_dtype)
    return (C.to(scale.dtype) * scale).to(out_dtype)

def check_case(
    Batch:int, M:int, N:int, K:int, scale_size:Literal["1", "N"],
    block_M:int, block_N:int, block_K:int, 
    in_dtype: Literal["int8"] = "int8", out_dtype: Literal["float16", "bfloat16"] = "float16", 
    accum_dtype: Literal["int32"] = "int32", scale_dtype: Literal["float32"] = "float32"
):
    torch_dtype_map = {
        "float16": torch.half, "float32": torch.float32, "bfloat16": torch.bfloat16,
        "int8": torch.int8, "int32": torch.int32, "int64": torch.int64, "uint64": torch.uint64
    }

    A = torch.randint(-128, 127, [Batch, M, K], dtype=torch_dtype_map[in_dtype])
    B = torch.randint(-128, 127, [Batch, K, N], dtype=torch_dtype_map[in_dtype])
    scale = torch.randn([N if scale_size == "N" else 1], dtype=torch_dtype_map[scale_dtype])

    kernel = simple_quant_batch_matmul(Batch, M, N, K, scale_size, block_M, block_N, block_K, in_dtype, out_dtype, accum_dtype, scale_dtype)
    C = kernel(A.npu(), B.npu(), scale.npu())
    ref_C = ref_program(A, B, scale, torch_dtype_map[out_dtype], torch_dtype_map[accum_dtype])

    torch.testing.assert_close(C.cpu(), ref_C.cpu(), rtol=1e-2, atol=1e-2)

def main(custom_args=None):
    parser = argparse.ArgumentParser(description="QuantBatchMatmul Example")
    parser.add_argument("--b", type=int, default=8, help="Batch size")
    parser.add_argument("--m", type=int, default=1024, help="Matrix M dimension")
    parser.add_argument("--n", type=int, default=1024, help="Matrix N dimension")
    parser.add_argument("--k", type=int, default=1024, help="Matrix K dimension")
    args, remains = parser.parse_known_args(custom_args)
    if remains:
        print(f"[{parser.description}]", "Unknown args:", remains)
    Batch, M, N, K = args.b, args.m, args.n, args.k

    tl.cache.clear_cache()
    torch.manual_seed(0)

    check_case(Batch, M, N, K, scale_size="1", block_M=128, block_N=256, block_K=64)
    check_case(Batch, M, N, K, scale_size="N", block_M=128, block_N=256, block_K=64, out_dtype="bfloat16")

    print("QuantBatchMatmul example passed!")
    print("Kernel Output Match!")

if __name__ == "__main__":
    main()
