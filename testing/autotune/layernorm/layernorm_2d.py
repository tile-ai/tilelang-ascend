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

# 你给的所有 shape
SHAPES = [
    (8,256),
    (8,768),
    (8,1280),
    (8,1792),
    (32,256),
    (32,768),
    (32,1280),
    (32,1792),
    (256,512),
    (256,1536),
    (256,2560),
    (256,3584),
    (64,8192),
    (64,12288),
    (64,16384),
    (64,20480),
    (1024,10240),
    (1024,14336),
    (1024,18432),
    (1024,22528),
    (1024,1048576),
    (1024,2097152),
    (1024,3145728),
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

                M, N = shape
                eps = 1e-5

                # -----------------------------
                # reference
                # -----------------------------
                def ref_prog(x, gamma, beta):

                    return torch.nn.functional.layer_norm(
                        x,
                        normalized_shape=(N,),
                        weight=gamma,
                        bias=beta,
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

                        configs.append({
                            "block_M": hint.block[0],
                            "block_N": hint.block[1],
                        })

                    return configs

                # -----------------------------
                # input generator
                # -----------------------------
                def supply_prog(params):

                    torch.manual_seed(0)

                    x = torch.randn(M, N, dtype=torch.float16).npu()
                    gamma = torch.randn(N, dtype=torch.float16).npu()
                    beta = torch.randn(N, dtype=torch.float16).npu()

                    return [x, gamma, beta]

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
                def layernorm(M, N, block_M, block_N):

                    @T.prim_func
                    def layernorm_kernel(
                        A: T.Tensor((M, N), "float16"),
                        Gamma: T.Tensor((N,), "float16"),
                        Beta: T.Tensor((N,), "float16"),
                        C: T.Tensor((M, N), "float16"),
                    ):

                        with T.Kernel(
                            T.ceildiv(N, block_N) * T.ceildiv(M, block_M),
                            is_npu=True,
                        ) as (cid, _):

                            by = cid // T.ceildiv(N, block_N)
                            bx = cid % T.ceildiv(N, block_N)

                            A_shared = T.alloc_shared((block_M, block_N), "float16")

                            local_x = T.alloc_fragment((block_M, block_N), "float16")

                            row_mean = T.alloc_fragment((block_M, 1), "float16")
                            row_var = T.alloc_fragment((block_M, 1), "float16")

                            tmp = T.alloc_fragment((block_M, block_N), "float16")

                            gamma = T.alloc_fragment((1, block_N), "float16")
                            beta = T.alloc_fragment((1, block_N), "float16")

                            eps_buf = T.alloc_fragment((block_M, 1), "float16")

                            # load
                            T.copy(A[by * block_M, bx * block_N], A_shared)
                            T.copy(A_shared, local_x)

                            T.copy(Gamma[bx * block_N], gamma)
                            T.copy(Beta[bx * block_N], beta)

                            # ------------------
                            # mean
                            # ------------------
                            T.reduce(local_x, row_mean, dims=1, reduce_mode="sum")

                            inv_n = 1.0 / N
                            T.vmul(row_mean, inv_n, row_mean)

                            # ------------------
                            # x - mean
                            # ------------------
                            T.vsub(local_x, row_mean, tmp)

                            # ------------------
                            # variance
                            # ------------------
                            T.vmul(tmp, tmp, tmp)

                            T.reduce(tmp, row_var, dims=1, reduce_mode="sum")

                            T.vmul(row_var, inv_n, row_var)

                            # ------------------
                            # std
                            # ------------------


                            T.vadd(row_var, eps, row_var)

                            T.vsqrt(row_var, row_var)

                            # ------------------
                            # normalize
                            # ------------------
                            T.vsub(local_x, row_mean, local_x)

                            T.vdiv(local_x, row_var, local_x)

                            # ------------------
                            # gamma * x + beta
                            # ------------------
                            T.vmul(local_x, gamma, local_x)

                            T.vadd(local_x, beta, local_x)

                            # store
                            T.copy(local_x, C[by * block_M, bx * block_N])

                    return layernorm_kernel

                # -----------------------------
                # compile
                # -----------------------------
                func = layernorm(M, N)

                # trigger autotune
                x = torch.randn(M, N, dtype=torch.float16).npu()
                gamma = torch.randn(N, dtype=torch.float16).npu()
                beta = torch.randn(N, dtype=torch.float16).npu()

                y = func(x, gamma, beta)

                torch.npu.synchronize()

                print("\nBest Config:")
                print(func.get_tuner_result())

                print("\nOutput shape:", y.shape)

                print("\nTest passed!")

            except Exception:
                print("\nERROR OCCURRED\n")
                traceback.print_exc()

    print(f"Finished shape {shape}, log saved to {log_file}")


def main():
    root_log_dir = Path("./shape_logs_2d_f16")
    root_log_dir.mkdir(exist_ok=True)

    for shape in SHAPES:
        # 为每个shape创建目录名
        shape_str = "x".join(map(str, shape))
        log_dir = root_log_dir / shape_str

        run_single_shape(shape, log_dir)


if __name__ == "__main__":
    main()
