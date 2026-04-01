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


def fn_torch(x, y):
    return torch.mul(torch.nn.functional.silu(x), y)


def run_test():
    dtype = torch.float16
    perf_list = []

    for shape in SHAPES:
        print(f"torch_silu_mul|dtype=float16|shape={shape}")

        # create input tensor
        x = torch.randn(shape, dtype=dtype).npu()
        y = torch.randn(shape, dtype=dtype).npu()

        # benchmark
        time_torch = do_bench_npu(lambda x=x, y=y: fn_torch(x, y))
        print("<<<<< time_torch in us", time_torch * 1000)

        msg = f"torch_silu_mul|float16|{shape}|{time_torch * 1000}"  # in us
        perf_list.append(msg)

    print("\n==== Performance Result ====")
    for m in perf_list:
        print(m)


if __name__ == "__main__":
    run_test()
