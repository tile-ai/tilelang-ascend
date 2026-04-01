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
    (8, 4, 8, 64),
    (8, 4, 8, 128),
    (8, 4, 2048, 8),
    (8, 4, 8, 127),
    (16, 8, 16, 255),
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
            N, C, H, W = shape

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
                    shape=[N, C, H, W],
                    dtype="float16",
                ).with_arch(arch)

                hints = carver_template.recommend_hints(topk=20)

                configs = []
                for hint in hints:
                    print("Hint:", hint)

                    blocks = hint.block
                    ndim = len(blocks)

                    shape_dims = [N, C, H, W]
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
                            "block_N": result_blocks[0],
                            "block_C": result_blocks[1],
                            "block_H": result_blocks[2],
                            "block_W": result_blocks[3],
                        }
                    )

                return configs

            # ------------------------
            # input supplier
            # ------------------------
            def supply_prog(params):
                torch.manual_seed(0)
                return [
                    torch.randn(N, C, H, W, dtype=torch.float16).npu(),
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
            def elementwise_abs(N, C, H, W, block_N, block_C, block_H, block_W):

                @T.prim_func
                def elemAbs(
                    A: T.Tensor((N, C, H, W), "float16"),
                    C_out: T.Tensor((N, C, H, W), "float16"),
                ):
                    with T.Kernel(
                        T.ceildiv(N, block_N)
                        * T.ceildiv(C, block_C)
                        * T.ceildiv(H, block_H)
                        * T.ceildiv(W, block_W),
                        is_npu=True,
                    ) as (cid, _):
                        tmp = cid

                        bz = tmp // (
                            T.ceildiv(C, block_C)
                            * T.ceildiv(H, block_H)
                            * T.ceildiv(W, block_W)
                        )
                        tmp %= (
                            T.ceildiv(C, block_C)
                            * T.ceildiv(H, block_H)
                            * T.ceildiv(W, block_W)
                        )

                        bc = tmp // (T.ceildiv(H, block_H) * T.ceildiv(W, block_W))
                        tmp %= T.ceildiv(H, block_H) * T.ceildiv(W, block_W)

                        by = tmp // T.ceildiv(W, block_W)
                        bx = tmp % T.ceildiv(W, block_W)

                        A_shared = T.alloc_shared(
                            (block_N, block_C, block_H, block_W),
                            "float16",
                        )
                        C_local = T.alloc_fragment(
                            (block_N, block_C, block_H, block_W),
                            "float16",
                        )

                        T.copy(
                            A[
                                bz * block_N,
                                bc * block_C,
                                by * block_H,
                                bx * block_W,
                            ],
                            A_shared,
                        )

                        T.vabs(A_shared, C_local)

                        T.copy(
                            C_local,
                            C_out[
                                bz * block_N,
                                bc * block_C,
                                by * block_H,
                                bx * block_W,
                            ],
                        )

                return elemAbs

            func = elementwise_abs(N, C, H, W)

            print("\nBest Config:")
            print(func.get_tuner_result())
            print("\nTest passed!")

        except Exception:
            print("\nERROR OCCURRED\n")
            traceback.print_exc()

    print(f"Finished shape {shape}, log saved to {log_file}")


def main():
    root_log_dir = Path("./shape_logs_4d_f16")
    root_log_dir.mkdir(exist_ok=True)

    for shape in SHAPES:
        shape_str = "x".join(map(str, shape))
        log_dir = root_log_dir / shape_str
        run_single_shape(shape, log_dir)


if __name__ == "__main__":
    main()
