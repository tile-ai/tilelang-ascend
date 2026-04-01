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

torch.npu.set_device(10)

SHAPES = [
    (1024, 10240),
    (1024, 14336),
    (1024, 18432),
    (1024, 22528),
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

            # ------------------------
            # input supplier
            # ------------------------
            def supply_prog(params):
                torch.manual_seed(0)
                return [
                    torch.randn(M, N, dtype=torch.float16).npu(),
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
            def elementwise_abs(M, N, block_M, block_N):
                num_physical_kernels = 40
                num_logical_kernels = (N // block_N) * (M // block_M)

                @T.prim_func
                def elemAbs(
                    A: T.Tensor((M, N), "float16"),
                    C: T.Tensor((M, N), "float16"),
                ):
                    with T.Kernel(num_physical_kernels, is_npu=True) as (kernel_id, _):
                        num_local_tasks = T.ceildiv(
                            num_logical_kernels - kernel_id, num_physical_kernels
                        )

                        for task_id in T.serial(num_local_tasks):
                            cid = task_id * num_physical_kernels + kernel_id
                            by = cid // T.ceildiv(N, block_N)
                            bx = cid % T.ceildiv(N, block_N)

                            A_shared = T.alloc_shared((block_M, block_N), "float16")

                            C_local = T.alloc_fragment((block_M, block_N), "float16")

                            T.copy(A[by * block_M, bx * block_N], A_shared)

                            T.vabs(A_shared, C_local)

                            T.copy(C_local, C[by * block_M, bx * block_N])

                return elemAbs

            func = elementwise_abs(M, N)

            print("\nBest Config:")
            print(func.get_tuner_result())
            print("\nTest passed!")

        except Exception:
            print("\nERROR OCCURRED\n")
            traceback.print_exc()

    print(f"Finished shape {shape}, log saved to {log_file}")


def main():
    root_log_dir = Path("./shape_logs_2d_f16_new")
    root_log_dir.mkdir(exist_ok=True)

    for shape in SHAPES:
        shape_str = "x".join(map(str, shape))
        log_dir = root_log_dir / shape_str
        run_single_shape(shape, log_dir)


if __name__ == "__main__":
    main()
