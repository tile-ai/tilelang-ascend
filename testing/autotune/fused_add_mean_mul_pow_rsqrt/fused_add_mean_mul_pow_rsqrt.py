import os
import traceback
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path

import torch
import tilelang
import tilelang.language as T
from tilelang import carver
from tilelang.carver.arch.ascend import Ascend

torch.npu.set_device(10)
os.environ["TILELANG_ASCEND_MODE"] = "Developer"

# 你给的所有 shape
SHAPES = [
    (32, 64),
    (32, 128),
    (32, 2048),
    (22, 127),
    (16, 255),
    (44, 255),
    (32, 1025),
    (800, 4090),
    (16384, 4096),
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
            M, N = shape
            eps = 1e-5

            # -----------------------------
            # reference
            # -----------------------------
            def ref_prog(x, w):
                return torch.nn.functional.rms_norm(
                    x,
                    normalized_shape=(N,),
                    weight=w,
                    eps=eps,
                )

            # -----------------------------
            # autotune configs
            # -----------------------------
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

            # -----------------------------
            # input generator
            # -----------------------------
            def supply_prog(params):

                torch.manual_seed(0)

                x = torch.randn(M, N, dtype=torch.float16).npu()
                w = torch.randn(N, dtype=torch.float16).npu()

                return [x, w]

            # -----------------------------
            # kernel
            # -----------------------------
            @tilelang.autotune(
                configs=get_config(),
                ref_prog=ref_prog,
                supply_prog=supply_prog,
                atol=1e-2,
                rtol=1e-2,
            )
            @tilelang.jit(out_idx=[-1], target="npuir")
            def rms_norm(M, N, block_M, block_N):
                @T.prim_func
                def rms_norm_kernel(
                    X: T.Tensor((M, N), "float16"),
                    W: T.Tensor((N,), "float16"),
                    Out: T.Tensor((M, N), "float16"),
                ):
                    with T.Kernel(T.ceildiv(M, block_M), is_npu=True) as (bx, _):
                        eps = 1e-5
                        X_shared = T.alloc_shared((block_M, block_N), "float16")
                        W_shared = T.alloc_shared((block_N,), "float16")
                        W_reshape = T.alloc_shared((1, block_N), "float16")
                        local_reduce = T.alloc_shared((block_M, 1), "float16")
                        row_rms = T.alloc_shared((block_M, 1), "float16")
                        row_rstd = T.alloc_shared((block_M, 1), "float16")

                        T.clear(row_rms)

                        for ko in T.serial(T.ceildiv(N, block_N)):
                            col_base = ko * block_N
                            T.copy(X[bx * block_M, col_base], X_shared)
                            T.vmul(X_shared, X_shared, X_shared)
                            T.reduce(X_shared, local_reduce, dims=1, reduce_mode="sum")
                            for i in T.serial(block_M):
                                row_rms[i, 0] = row_rms[i, 0] + local_reduce[i, 0]

                        for i in T.serial(block_M):
                            row = bx * block_M + i
                            if row < M:
                                row_rms[i, 0] = row_rms[i, 0] - row_rms[i, 0] * (
                                    1.0 - 1.0 / N
                                )
                                row_rms[i, 0] = row_rms[i, 0] + eps
                                T.vrsqrt(row_rms[i, 0], row_rstd[i, 0])

                        for ko in T.serial(T.ceildiv(N, block_N)):
                            col_base = ko * block_N
                            T.copy(X[bx * block_M, col_base], X_shared)
                            T.copy(W[col_base], W_shared)
                            T.reshape(W_shared, W_reshape)

                            T.vmul(X_shared, row_rstd, X_shared)
                            T.vmul(X_shared, W_reshape, X_shared)

                            T.copy(X_shared, Out[bx * block_M, col_base])

                return rms_norm_kernel

            func = rms_norm(M, N)

            torch.npu.synchronize()

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
