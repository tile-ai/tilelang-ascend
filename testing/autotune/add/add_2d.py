import os
import traceback
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path

import torch
import tilelang
import tilelang.language as T
from tilelang import carver
from tilelang.carver.arch.ascend import Ascend

os.environ["TILELANG_ASCEND_MODE"] = "Developer"

torch.npu.set_device(15)

SHAPES = [
    (8, 64),
    (8, 128),
    (8, 2048),
    (8, 127),
    (16, 255),
    (32, 1025),
    (1024, 10240),
    (1024, 14336),
    (1024, 18432),
    (1024, 22528),
    (1024, 1048576),
]


def run_single_shape(shape, log_dir: Path):
    tilelang.cache.clear_cache()

    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "log.log"

    with open(log_file, "w") as f, redirect_stdout(f), redirect_stderr(f):
        print("=" * 80)
        print("Running shape:", shape)
        print("=" * 80)

        try:
            M, N = shape if len(shape) == 2 else (shape[0], 1)

            def ref_prog(x, y):
                return x + y

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
                    configs.append(
                        {
                            "block_M": hint.block[0],
                            "block_N": hint.block[1],
                        }
                    )

                return configs

            def supply_prog(params):
                torch.manual_seed(0)
                return [
                    torch.randn(M, N, dtype=torch.float16).npu(),
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
            def elementwise_add(M, N, block_M, block_N):
                @T.prim_func
                def elemAdd(
                    A: T.Tensor((M, N), "float16"),
                    B: T.Tensor((M, N), "float16"),
                    C: T.Tensor((M, N), "float16"),
                ):
                    with T.Kernel(
                        T.ceildiv(N, block_N) * T.ceildiv(M, block_M),
                        is_npu=True,
                    ) as (cid, _):
                        by = cid // T.ceildiv(N, block_N)
                        bx = cid % T.ceildiv(N, block_N)

                        A_shared = T.alloc_shared((block_M, block_N), "float16")
                        B_shared = T.alloc_shared((block_M, block_N), "float16")
                        C_local = T.alloc_fragment((block_M, block_N), "float16")

                        T.copy(A[by * block_M, bx * block_N], A_shared)
                        T.copy(B[by * block_M, bx * block_N], B_shared)

                        T.vadd(A_shared, B_shared, C_local)

                        T.copy(C_local, C[by * block_M, bx * block_N])

                return elemAdd

            func = elementwise_add(M, N)

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
