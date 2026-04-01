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
            def ref_prog(x, w, b, m, r):

                return torch.nn.functional.layer_norm(
                    x,
                    normalized_shape=(N,),
                    weight=w,
                    bias=b,
                    eps=eps,
                )

            # -----------------------------
            # autotune configs
            # -----------------------------
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

            # -----------------------------
            # input generator
            # -----------------------------
            def supply_prog(params):

                torch.manual_seed(0)

                x = torch.randn(M, N, dtype=torch.float32).npu()
                w = torch.randn(N, dtype=torch.float32).npu()
                b = torch.randn(N, dtype=torch.float32).npu()
                m = torch.randn(M, dtype=torch.float32).npu()
                r = torch.randn(M, dtype=torch.float32).npu()

                return [x, w, b, m, r]

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
            @tilelang.jit(out_idx=[1], target="npuir")
            def layer_norm_fwd(M, N, block_M, block_N):
                @T.prim_func
                def layer_norm_fwd_kernel(
                    X: T.Tensor((M, N), "float32"),
                    Y: T.Tensor((M, N), "float32"),
                    W: T.Tensor((N,), "float32"),
                    B: T.Tensor((N,), "float32"),
                    Mean: T.Tensor((M,), "float32"),
                    Rstd: T.Tensor((M,), "float32"),
                ):
                    eps = 1e-5
                    with T.Kernel(T.ceildiv(M, block_M), is_npu=True) as (bx, _):
                        X_shared = T.alloc_shared((block_M, block_N), "float32")
                        W_shared = T.alloc_shared((block_N,), "float32")
                        W_reshape = T.alloc_shared((1, block_N), "float32")

                        B_shared = T.alloc_shared((block_N,), "float32")
                        B_reshape = T.alloc_shared((1, block_N), "float32")

                        local_reduce = T.alloc_shared((block_M, 1), "float32")
                        row_mean = T.alloc_shared((block_M, 1), "float32")
                        row_var = T.alloc_shared((block_M, 1), "float32")
                        row_rstd = T.alloc_shared((block_M, 1), "float32")

                        T.clear(row_mean)
                        T.clear(row_var)

                        for ko in T.serial(T.ceildiv(N, block_N)):
                            col_base = ko * block_N

                            T.copy(X[bx * block_M, col_base], X_shared)

                            T.reduce(X_shared, local_reduce, dims=1, reduce_mode="sum")

                            for i in T.serial(block_M):
                                row = bx * block_M + i
                                if row < M:
                                    row_mean[i, 0] = row_mean[i, 0] + local_reduce[i, 0]

                        for i in T.serial(block_M):
                            row = bx * block_M + i
                            if row < M:
                                row_mean[i, 0] = row_mean[i, 0] / N

                        for ko in T.serial(T.ceildiv(N, block_N)):
                            col_base = ko * block_N

                            T.copy(X[bx * block_M, col_base], X_shared)

                            T.vsub(X_shared, row_mean, X_shared)
                            T.vmul(X_shared, X_shared, X_shared)

                            T.reduce(X_shared, local_reduce, dims=1, reduce_mode="sum")

                            for i in T.serial(block_M):
                                row_var[i, 0] = row_var[i, 0] - (-local_reduce[i, 0])

                        for i in T.serial(block_M):
                            row = bx * block_M + i
                            if row < M:
                                row_var[i, 0] = row_var[i, 0] - row_var[i, 0] * (
                                    1.0 - 1.0 / N
                                )
                                row_var[i, 0] = row_var[i, 0] + eps
                                T.vrsqrt(row_var[i, 0], row_rstd[i, 0])

                        for i in T.serial(block_M):
                            row = bx * block_M + i
                            if row < M:
                                Mean[row] = row_mean[i, 0]
                                Rstd[row] = row_rstd[i, 0]

                        for ko in T.serial(T.ceildiv(N, block_N)):
                            col_base = ko * block_N

                            T.copy(X[bx * block_M, col_base], X_shared)
                            T.copy(W[col_base], W_shared)
                            T.copy(B[col_base], B_shared)
                            T.reshape(W_shared, W_reshape)
                            T.reshape(B_shared, B_reshape)

                            T.vsub(X_shared, row_mean, X_shared)
                            T.vmul(X_shared, row_rstd, X_shared)
                            T.vmul(X_shared, W_reshape, X_shared)
                            T.vadd(X_shared, B_reshape, X_shared)

                            T.copy(X_shared, Y[bx * block_M, col_base])

                return layer_norm_fwd_kernel

            # -----------------------------
            # compile
            # -----------------------------
            func = layer_norm_fwd(M, N)

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
