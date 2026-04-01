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
    (1, 32, 64),
    (1, 32, 128),
    (1, 32, 2048),
    (1, 22, 127),
    (1, 44, 255),
    (1, 88, 1025),
    (32, 1632, 1025),
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
            B, M, N = shape

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
                    shape=[B, M, N],
                    dtype="float32",
                ).with_arch(arch)

                hints = carver_template.recommend_hints(topk=20)

                configs = []
                for hint in hints:
                    print("Hint:", hint)

                    blocks = hint.block
                    ndim = len(blocks)

                    shape_dims = [B, M, N]
                    result_blocks = []

                    j = 0

                    for dim in shape_dims:
                        if dim == 1:
                            result_blocks.append(1)
                        else:
                            if j < ndim:
                                result_blocks.append(blocks[j])
                                j += 1
                            else:
                                result_blocks.append(1)

                    configs.append(
                        {
                            "block_B": result_blocks[0],
                            "block_M": result_blocks[1],
                            "block_N": result_blocks[2],
                        }
                    )

                return configs

            # ------------------------
            # input supplier
            # ------------------------
            def supply_prog(params):
                torch.manual_seed(0)
                return [
                    torch.randn(B, M, N, dtype=torch.float32).npu(),
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
            def elementwise_abs(B, M, N, block_B, block_M, block_N):

                @T.prim_func
                def elemAbs(
                    A: T.Tensor((B, M, N), "float32"),
                    C: T.Tensor((B, M, N), "float32"),
                ):
                    with T.Kernel(
                        T.ceildiv(B, block_B)
                        * T.ceildiv(M, block_M)
                        * T.ceildiv(N, block_N),
                        is_npu=True,
                    ) as (cid, _):
                        tmp = cid

                        bz = tmp // (T.ceildiv(M, block_M) * T.ceildiv(N, block_N))
                        tmp = tmp % (T.ceildiv(M, block_M) * T.ceildiv(N, block_N))

                        by = tmp // T.ceildiv(N, block_N)
                        bx = tmp % T.ceildiv(N, block_N)

                        A_shared = T.alloc_shared((block_M, block_N), "float32")
                        C_local = T.alloc_fragment((block_M, block_N), "float32")

                        T.copy(
                            A[bz * block_B, by * block_M, bx * block_N],
                            A_shared,
                        )

                        T.vabs(A_shared, C_local)

                        T.copy(
                            C_local,
                            C[bz * block_B, by * block_M, bx * block_N],
                        )

                return elemAbs

            func = elementwise_abs(B, M, N)

            print("\nBest Config:")
            print(func.get_tuner_result())
            print("\nTest passed!")

        except Exception:
            print("\nERROR OCCURRED\n")
            traceback.print_exc()

    print(f"Finished shape {shape}, log saved to {log_file}")


def main():
    root_log_dir = Path("./shape_logs_3d")
    root_log_dir.mkdir(exist_ok=True)

    for shape in SHAPES:
        shape_str = "x".join(map(str, shape))
        log_dir = root_log_dir / shape_str
        run_single_shape(shape, log_dir)


if __name__ == "__main__":
    main()
