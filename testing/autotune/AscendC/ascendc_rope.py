import torch

torch.npu.set_device(10)

from triton.backends.ascend.testing import do_bench_npu


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


def fn_torch(state, cos, sin):

    _, _, dim = state.shape

    state_x = state[:, :, : dim // 2]
    state_y = state[:, :, dim // 2 :]

    out_x = state_x * cos - state_y * sin
    out_y = state_x * sin + state_y * cos

    return torch.cat((out_x, out_y), dim=-1)


def run_test():

    dtype = torch.float16
    perf_list = []

    head_dim = 32

    for shape in SHAPES:
        print(f"torch_rope|dtype=float16|shape={shape}")

        tokens, hidden_dim = shape

        num_heads = hidden_dim // head_dim

        state = torch.randn((tokens, num_heads, head_dim), dtype=dtype, device="npu")
        cos = torch.randn((tokens, 1, head_dim // 2), dtype=dtype, device="npu")
        sin = torch.randn((tokens, 1, head_dim // 2), dtype=dtype, device="npu")

        # benchmark
        time_torch = do_bench_npu(
            lambda state=state, cos=cos, sin=sin: fn_torch(state, cos, sin)
        )

        print("<<<<< time_torch in us", time_torch * 1000)

        msg = f"torch_rope|float16|{shape}|{time_torch * 1000}"
        perf_list.append(msg)

    print("\n==== Performance Result ====")

    for m in perf_list:
        print(m)


if __name__ == "__main__":
    run_test()
