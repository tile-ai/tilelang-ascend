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
    (64,),
    (128,),
    (2048,),
    (127,),
    (255,),
    (1025,),
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
            M = shape[0]

            # ------------------------
            # reference
            # ------------------------
            def ref_prog(x):
                return torch.abs(x)

            # ------------------------
            # config search
            # ------------------------
            def get_config():
                arch = Ascend()

                carver_template = carver.ElementwiseFixTemplate(
                    shape=[M],
                    dtype="float32",
                ).with_arch(arch)

                hints = carver_template.recommend_hints(topk=4)

                configs = []
                for hint in hints:
                    print("Hint:", hint)
                    configs.append(
                        {
                            "block_M": hint.block[0],
                        }
                    )

                return configs

            # ------------------------
            # input supplier
            # ------------------------
            def supply_prog(params):
                torch.manual_seed(0)
                return [
                    torch.randn(M, dtype=torch.float32).npu(),
                ]

            # ------------------------
            # kernel
            # ------------------------
            @tilelang.autotune(
                configs=get_config(),
                ref_prog=ref_prog,
                supply_prog=supply_prog,
                atol=1e-2,
                rtol=1e-2,
            )
            @tilelang.jit(out_idx=[-1], target="npuir")
            def elementwise_abs(M, block_M):

                @T.prim_func
                def elemAbs(
                    A: T.Tensor((M,), "float32"),
                    C: T.Tensor((M,), "float32"),
                ):
                    with T.Kernel(
                        T.ceildiv(M, block_M),
                        is_npu=True,
                    ) as (bid, _):
                        offset = bid * block_M

                        A_shared = T.alloc_shared((block_M,), "float32")
                        C_local = T.alloc_fragment((block_M,), "float32")

                        T.copy(A[offset], A_shared)

                        T.vabs(A_shared, C_local)

                        T.copy(C_local, C[offset])

                return elemAbs

            func = elementwise_abs(M)

            print("\nBest Config:")
            print(func.get_tuner_result())
            print("\nTest passed!")

        except Exception:
            print("\nERROR OCCURRED\n")
            traceback.print_exc()

    print(f"Finished shape {shape}, log saved to {log_file}")


def main():
    root_log_dir = Path("./shape_logs_1d")
    root_log_dir.mkdir(exist_ok=True)

    for shape in SHAPES:
        shape_str = "x".join(map(str, shape))
        log_dir = root_log_dir / shape_str
        run_single_shape(shape, log_dir)


if __name__ == "__main__":
    main()
