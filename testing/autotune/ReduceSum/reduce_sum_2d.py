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
SHAPES = [
    (8, 256),
    (8, 768),
    (8, 1280),
    (8, 1792),
    (32, 256),
    (32, 768),
    (32, 1280),
    (32, 1792),
    (256, 512),
    (256, 1536),
    (256, 2560),
    (256, 3584),
    (64, 8192),
    (64, 12288),
    (64, 16384),
    (64, 20480),
    (1024, 10240),
    (1024, 14336),
    (1024, 18432),
    (1024, 22528),
    (1024, 1048576),
    (1024, 2097152),
    (1024, 3145728),
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
                    return torch.sum(x, dim=1, keepdim=True)

                def get_config():
                    arch = Ascend()
                    carver_template = carver.GeneralReductionTemplate(
                        structure="SR",
                        shape=[M, N],
                        dtype="float16",
                    ).with_arch(arch)

                    hints = carver_template.recommend_hints(topk=20)
                    configs = []

                    for hint in hints:
                        print("Hint:", hint)
                        configs.append({
                            "block_M": hint.block[0],
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
                def compute_reduce_sum(M, N, block_M):
                    @T.prim_func
                    def reduce_sum_2D(
                        A: T.Tensor((M, N), "float16"),
                        B: T.Tensor((M, 1), "float16"),
                    ):
                        with T.Kernel(T.ceildiv(M, block_M),
                            is_npu=True,
                        ) as (cid, _):

                            A_shared = T.alloc_shared((block_M, N), "float16")
                            B_local = T.alloc_fragment((block_M, 1), "float16")
                            offset = cid * block_M

                            T.copy(A[offset, 0], A_shared, size=[block_M, N])
                            T.reduce(A_shared, B_local, dims=1, reduce_mode="sum", clear = True)
                            T.copy(B_local, B[offset, 0], size=[block_M, 1])

                    return reduce_sum_2D

                func = compute_reduce_sum(M, N)

                print("\nBest Config:")
                print(func.get_tuner_result())
                print("\nTest passed!")

            except Exception:
                print("\nERROR OCCURRED\n")
                traceback.print_exc()

    print(f"Finished shape {shape}, log saved to {log_file}")

def main():
    root_log_dir = Path("./shape_logs_2d")
    root_log_dir.mkdir(exist_ok=True)

    for shape in SHAPES:
        shape_str = "x".join(map(str, shape))
        log_dir = root_log_dir / shape_str
        run_single_shape(shape, log_dir)

if __name__ == "__main__":
    main()
