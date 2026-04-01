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
dtype = "float32"
accum_dtype = "float32"


def run_single_shape(shape, log_dir: Path):

    tilelang.cache.clear_cache()

    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "log.log"

    with open(log_file, "w") as f, redirect_stdout(f), redirect_stderr(f):
        print("=" * 80)
        print("Running shape:", shape)
        print("=" * 80)

        try:
            num_tokens, num_heads = shape

            # -----------------------------
            # reference
            # -----------------------------
            def ref_prog(state, cos, sin):
                half = 16

                x = state[..., :half]
                y = state[..., half:]

                cos_expanded = cos.expand(-1, state.shape[1], -1)
                sin_expanded = sin.expand(-1, state.shape[1], -1)

                out_x = x * cos_expanded - y * sin_expanded
                out_y = x * sin_expanded + y * cos_expanded

                return torch.cat([out_x, out_y], dim=-1)

            # -----------------------------
            # autotune configs
            # -----------------------------
            def get_config():

                arch = Ascend()

                carver_template = carver.ElementwiseFixTemplate(
                    shape=[num_tokens, num_heads],
                    dtype="float32",
                ).with_arch(arch)

                hints = carver_template.recommend_hints(topk=20)

                configs = []

                for hint in hints:
                    print("Hint:", hint)

                    configs.append({"BLOCK_N": hint.block[0], "BLOCK_H": hint.block[1]})

                return configs

            # -----------------------------
            # input generator
            # -----------------------------
            def supply_prog(params):

                torch.manual_seed(0)

                State = torch.randn(
                    num_tokens, num_heads, 32, dtype=torch.float32
                ).npu()
                Cos = torch.randn(num_tokens, 1, 16, dtype=torch.float32).npu()
                Sin = torch.randn(num_tokens, 1, 16, dtype=torch.float32).npu()

                return [State, Cos, Sin]

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
            def rotary_embedding(num_tokens, num_heads, head_dim, BLOCK_N, BLOCK_H):

                @T.prim_func
                def rotary_embedding_kernel(
                    State: T.Tensor((num_tokens, num_heads, head_dim), dtype),
                    Cos: T.Tensor((num_tokens, 1, head_dim // 2), dtype),
                    Sin: T.Tensor((num_tokens, 1, head_dim // 2), dtype),
                    Out: T.Tensor((num_tokens, num_heads, head_dim), dtype),
                ):
                    with T.Kernel(
                        T.ceildiv(num_tokens, BLOCK_N) * T.ceildiv(num_heads, BLOCK_H),
                        is_npu=True,
                    ) as (cid, _):
                        by = cid // T.ceildiv(num_tokens, BLOCK_N)
                        bx = cid % T.ceildiv(num_tokens, BLOCK_N)
                        half = head_dim // 2

                        X_shared = T.alloc_shared(
                            (BLOCK_N, BLOCK_H, head_dim // 2), dtype
                        )
                        Y_shared = T.alloc_shared(
                            (BLOCK_N, BLOCK_H, head_dim // 2), dtype
                        )
                        Cos_shared = T.alloc_shared((BLOCK_N, head_dim // 2), dtype)
                        Sin_shared = T.alloc_shared((BLOCK_N, head_dim // 2), dtype)

                        for i in T.serial(BLOCK_N):
                            token_idx = bx * BLOCK_N + i
                            if token_idx < num_tokens:
                                T.copy(
                                    Cos[token_idx, 0, 0],
                                    Cos_shared[i, 0],
                                    size=[1, half],
                                )
                                T.copy(
                                    Sin[token_idx, 0, 0],
                                    Sin_shared[i, 0],
                                    size=[1, half],
                                )

                        for i, h in T.Parallel(BLOCK_N, BLOCK_H):
                            token_idx = bx * BLOCK_N + i
                            head_idx = by * BLOCK_H + h
                            if token_idx < num_tokens and head_idx < num_heads:
                                T.copy(
                                    State[token_idx, head_idx, 0],
                                    X_shared[i, h, 0],
                                    size=[1, 1, half],
                                )
                                T.copy(
                                    State[token_idx, head_idx, half],
                                    Y_shared[i, h, 0],
                                    size=[1, 1, half],
                                )

                        for i, h, d in T.Parallel(BLOCK_N, BLOCK_H, half):
                            x = X_shared[i, h, d]
                            y = Y_shared[i, h, d]
                            c = Cos_shared[i, d]
                            s = Sin_shared[i, d]

                            X_shared[i, h, d] = x * c - y * s
                            Y_shared[i, h, d] = x * s + y * c

                        for i, h in T.Parallel(BLOCK_N, BLOCK_H):
                            token_idx = bx * BLOCK_N + i
                            head_idx = by * BLOCK_H + h
                            if token_idx < num_tokens and head_idx < num_heads:
                                T.copy(
                                    X_shared[i, h, 0],
                                    Out[token_idx, head_idx, 0],
                                    size=[1, 1, half],
                                )
                                T.copy(
                                    Y_shared[i, h, 0],
                                    Out[token_idx, head_idx, half],
                                    size=[1, 1, half],
                                )

                return rotary_embedding_kernel

            # -----------------------------
            # compile
            # -----------------------------
            func = rotary_embedding(num_tokens, num_heads, 32)

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
