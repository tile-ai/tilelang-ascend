import os
import sys
import argparse
import traceback
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path

import torch
import tilelang
import tilelang.language as T
from tilelang import carver
from tilelang.carver.arch.ascend import Ascend


os.environ["TILELANG_ASCEND_MODE"] = "Developer"

# 你给的所有 shape
SHAPES = [
    (8,256),
    (8,768),
    (8,1280),
    (8,1792),
    (32,256),
    (32,768),
    (32,1280),
    (32,1792),
    (256,512),
    (256,1536),
    (256,2560),
    (256,3584),
    (64,8192),
    (64,12288),
    (64,16384),
    (64,20480),
    (1024,10240),
    (1024,14336),
    (1024,18432),
    (1024,22528),
    (1024,1048576),
    (1024,2097152),
    (1024,3145728),
]


def run_single_shape(shape, log_dir: Path):
    tilelang.cache.clear_cache()

    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "log.log"

    with open(log_file, "w") as f:
        with redirect_stdout(f), redirect_stderr(f):
            print("=" * 80)
            print("Running shape:", shape)
            print("=" * 80)

            try:
                M, N = shape if len(shape) == 2 else (shape[0], 1)

                def ref_prog(x):
                    return torch.nn.functional.softmax(x, dim=1)

                def get_config():
                    arch = Ascend()
                    carver_template = carver.ElementwiseFixTemplate(
                        shape=[M, N],
                        dtype="float16",
                    ).with_arch(arch)

                    hints = carver_template.recommend_hints(topk=20)
                    configs = []

                    for hint in hints:
                        print("Hint:", hint)
                        configs.append({
                            "block_M": hint.block[0],
                            "block_N": hint.block[1],
                        })

                    return configs

                def supply_prog(params):
                    torch.manual_seed(0)
                    return [
                        torch.randn(M, N, dtype=torch.float16).npu(),
                    ]

                @tilelang.autotune(
                    configs=get_config(),
                    ref_prog=ref_prog,
                    supply_prog=supply_prog,
                    atol=1e-2,
                    rtol=1e-2,
                )
                @tilelang.jit(out_idx=[-1], target="npuir")
                def softmax(M, N, block_M, block_N):
                    
                    @T.prim_func
                    def softmax_kernel(
                        A: T.Tensor((M, N), "float16"),
                        C: T.Tensor((M, N), "float16"),
                    ):
                        with T.Kernel(
                            T.ceildiv(N, block_N) * T.ceildiv(M, block_M),
                            is_npu=True,
                        ) as (cid, _):

                            by = cid // T.ceildiv(N, block_N)
                            bx = cid % T.ceildiv(N, block_N)

                            A_shared = T.alloc_shared((block_M, block_N), "float16")

                            # fragment buffers
                            local_scores = T.alloc_fragment((block_M, block_N), "float16")
                            row_max = T.alloc_fragment((block_M,1), "float16")
                            row_sum = T.alloc_fragment((block_M,1), "float16")

                            # load
                            T.copy(A[by * block_M, bx * block_N], A_shared)
                            T.copy(A_shared, local_scores)

                            # --------------------------
                            # step1: reduce max per row
                            # --------------------------
                            T.reduce(local_scores, row_max, dims=1, reduce_mode="max")

                            # --------------------------
                            # step2: subtract max
                            # --------------------------
                            T.vsub(local_scores, row_max, local_scores)

                            # --------------------------
                            # step3: exp
                            # --------------------------
                            T.vexp(local_scores, local_scores)

                            # --------------------------
                            # step4: sum
                            # --------------------------
                            T.reduce(local_scores, row_sum, dims=1, reduce_mode="sum")

                            # --------------------------
                            # step5: normalize
                            # --------------------------
                            T.vdiv(local_scores, row_sum, local_scores)

                            # store
                            T.copy(local_scores, C[by * block_M, bx * block_N])

                    return softmax_kernel

                    return siluAndMul

                func = softmax(M, N)

                print("\nBest Config:")
                print(func.get_tuner_result())
                print("\nTest passed!")

            except Exception:
                print("\nERROR OCCURRED\n")
                traceback.print_exc()

    print(f"Finished shape {shape}, log saved to {log_file}")


def main():
    root_log_dir = Path("./shape_logs_2d_f16")
    root_log_dir.mkdir(exist_ok=True)

    for shape in SHAPES:
        # 为每个shape创建目录名
        shape_str = "x".join(map(str, shape))
        log_dir = root_log_dir / shape_str

        run_single_shape(shape, log_dir)


if __name__ == "__main__":
    main()
