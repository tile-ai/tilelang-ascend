import os
import traceback
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path

import torch
import tilelang
import tilelang.language as T
from tilelang import carver
from tilelang.carver.arch.ascend import Ascend

torch.npu.set_device(15)
os.environ["TILELANG_ASCEND_MODE"] = "Developer"

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

            def ref_prog(x):
                return x * 0.5 * (1.0 + torch.erf(x / torch.sqrt(torch.tensor(2.0))))

            def get_config():
                arch = Ascend()
                carver_template = carver.ElementwiseFixTemplate(
                    shape=[M, N],
                    dtype="float32",
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
                    torch.empty(M, N).uniform_(-1.0, 1.0).npu(),
                    # torch.zeros(M, N).half().npu(),
                ]

            @tilelang.autotune(
                configs=get_config(),
                ref_prog=ref_prog,
                supply_prog=supply_prog,
                atol=1e-2,
                rtol=1e-2,
            )
            @tilelang.jit(out_idx=[-1], target="npuir")
            def compute_gelu(M, N, block_M, block_N):
                @T.prim_func
                def gelu_2D(
                    A: T.Tensor((M, N), "float32"),
                    B: T.Tensor((M, N), "float32"),
                ):
                    with T.Kernel(
                        T.ceildiv(N, block_N) * T.ceildiv(M, block_M),
                        is_npu=True,
                    ) as (cid, _):
                        by = cid // T.ceildiv(N, block_N)
                        bx = cid % T.ceildiv(N, block_N)
                        scale1 = 1 / (2.0**0.5)
                        scale2 = 1.0
                        scale3 = 0.5
                        A_shared = T.alloc_shared((block_M, block_N), "float32")
                        B_local = T.alloc_fragment((block_M, block_N), "float32")
                        C_local = T.alloc_fragment((block_M, block_N), "float32")
                        D_local = T.alloc_fragment((block_M, block_N), "float32")

                        T.copy(A[by * block_M, bx * block_N], A_shared)

                        T.vmul(A_shared, scale1, B_local)
                        T.npuir_verf(B_local, C_local)
                        T.vadd(C_local, scale2, C_local)
                        T.vmul(C_local, scale3, C_local)
                        T.vmul(A_shared, C_local, D_local)

                        T.copy(D_local, B[by * block_M, bx * block_N])

                return gelu_2D

            func = compute_gelu(M, N)

            print("\nBest Config:")
            print(func.get_tuner_result())
            print("\nTest passed!")

        except Exception:
            print("\nERROR OCCURRED\n")
            traceback.print_exc()

    print(f"Finished shape {shape}, log saved to {log_file}")


def main():
    root_log_dir = Path("./shape_logs_2d_f32")
    root_log_dir.mkdir(exist_ok=True)

    for shape in SHAPES:
        shape_str = "x".join(map(str, shape))
        log_dir = root_log_dir / shape_str

        run_single_shape(shape, log_dir)


if __name__ == "__main__":
    main()
