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

SHAPES = [
    (64,),
    (128,),
    (2048,),
    (127,),
    (255,),
    (1025,),
]


def run_single_shape(shape, log_dir: Path):

    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "log.log"

    with open(log_file, "w") as f, redirect_stdout(f), redirect_stderr(f):
        print("=" * 80)
        print("Running shape:", shape)
        print("=" * 80)

        try:
            M = shape[0]

            def ref_prog(x):
                return torch.sigmoid(x)

            def get_config():
                arch = Ascend()

                # 1D template
                carver_template = carver.ElementwiseTemplate(
                    shape=[M],
                    dtype="float16",
                ).with_arch(arch)

                hints = carver_template.recommend_hints(topk=20)

                configs = []
                for hint in hints:
                    print("Hint:", hint)
                    configs.append(
                        {
                            "block_M": hint.block[0],
                        }
                    )

                return configs

            def supply_prog(params):
                torch.manual_seed(0)
                return [
                    torch.randn(M, dtype=torch.float16).npu(),
                ]

            @tilelang.autotune(
                configs=get_config(),
                ref_prog=ref_prog,
                supply_prog=supply_prog,
                atol=1e-2,
                rtol=1e-2,
            )
            @tilelang.jit(out_idx=[-1], target="npuir")
            def compute_sigmoid(M, block_M):

                @T.prim_func
                def sigmoid_1D(
                    A: T.Tensor((M,), "float16"),
                    B: T.Tensor((M,), "float16"),
                ):
                    with T.Kernel(
                        T.ceildiv(M, block_M),
                        is_npu=True,
                    ) as (bid, _):
                        offset = bid * block_M
                        A_shared = T.alloc_shared((block_M,), "float16")
                        B_local = T.alloc_fragment((block_M,), "float16")

                        T.copy(A[offset], A_shared)
                        T.npuir_sigmoid(A_shared, B_local)
                        T.copy(B_local, B[offset])

                return sigmoid_1D

            func = compute_sigmoid(M)

            print("\nBest Config:")
            print(func.get_tuner_result())
            print("\nTest passed!")

        except Exception:
            print("\nERROR OCCURRED\n")
            traceback.print_exc()

    print(f"Finished shape {shape}, log saved to {log_file}")


def main():
    root_log_dir = Path("./shape_logs_1d_f16")
    root_log_dir.mkdir(exist_ok=True)

    for shape in SHAPES:
        shape_str = "x".join(map(str, shape))
        log_dir = root_log_dir / shape_str
        run_single_shape(shape, log_dir)


if __name__ == "__main__":
    main()
