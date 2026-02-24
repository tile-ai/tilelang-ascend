import argparse
import itertools

import tilelang
import tilelang.language as T
import torch
import os
from tilelang import carver
from tilelang.carver.arch.ascend import Ascend
from tilelang.carver.triton.tile_generator import KernelMeta, TileGenerator

tilelang.cache.clear_cache()

os.environ['TILELANG_ASCEND_MODE'] = 'Developer'
parser = argparse.ArgumentParser(description="NPU Kernel Compilation")
parser.add_argument("--m", type=int, default=522240)
args = parser.parse_args()

M = args.m

def ref_prog(x, y):
    return x + y

def get_config() -> list[dict]:
    kernel_meta = KernelMeta(
        axis_sizes={"x": M},
        split_params={"x":"XBLOCK"},
        tiling_params={"x":"XBLCOK_SUB"},
        low_dims=["x"],
        dtype=torch.float32,
        persistent_reduction=False,
        dual_reduction=False,
        num_buffers=1,
        is_simt_mode=False,
    )
    tile_gen=TileGenerator(kernel_meta=kernel_meta)
    
    configs = []
    tile_gen.descend_split_tiling()
    hints = tile_gen.configs
    for hint in hints:
        if hint.kwargs["XBLCOK_SUB"] < 17000:
            print(hint.kwargs)
            config = {
                "block_M": hint.kwargs["XBLOCK"],
                "block_M_sub": hint.kwargs["XBLCOK_SUB"],
            }
            configs.append(config)
    
    return configs

def supply_prog(params):
    torch.manual_seed(0)
    return [
        torch.randn(M, ).npu(),
        torch.randn(M, ).npu(),
        torch.randn(M, ).npu(),
    ]

@tilelang.autotune(
    configs=get_config(),
    ref_prog=ref_prog,
    supply_prog=supply_prog,
    atol=1e-2,
    rtol=1e-2,
)
@tilelang.jit(out_idx=[-1], target="npuir")
def elementwise_add(M, block_M, block_M_sub, in_dtype="float32", out_dtype="float32"):
    @T.prim_func
    def elemAdd(
            A: T.Tensor((M, ), in_dtype),
            B: T.Tensor((M, ), in_dtype),
            C: T.Tensor((M, ), out_dtype)
    ):
        with T.Kernel(T.ceildiv(M, block_M), is_npu=True) as (cid, _):
            offset = cid * block_M

            A_shared = T.alloc_shared((block_M_sub, ), in_dtype)
            B_shared = T.alloc_shared((block_M_sub, ), in_dtype)
            C_local = T.alloc_fragment((block_M_sub, ), out_dtype)

            for k in T.Pipelined(T.ceildiv(block_M, block_M_sub), num_stages=2):
                offset_sub = k * block_M_sub
                dynamic_shape = T.min(block_M_sub, block_M - offset_sub)
                offset_final = offset + offset_sub

                T.copy(A[offset_final], A_shared, size=[dynamic_shape, ])
                T.copy(B[offset_final], B_shared, size=[dynamic_shape, ])
                
                T.vadd(A_shared, B_shared, C_local)

                T.copy(C_local, C[offset_final], size=[dynamic_shape, ])

    return elemAdd

func = elementwise_add(M)

print("Best Config:", func.get_tuner_result())
print("Test passed!")

