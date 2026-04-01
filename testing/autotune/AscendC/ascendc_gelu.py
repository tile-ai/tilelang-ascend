import torch

torch.npu.set_device(10)

from triton.backends.ascend.testing import do_bench_npu

SHAPES = [
    (64,),
    (8, 64),
    (1, 32, 64),
    (8, 4, 8, 64),
    (128,),
    (8, 128),
    (1, 32, 128),
    (8, 4, 8, 128),
    (2048,),
    (8, 2048),
    (1, 32, 2048),
    (8, 4, 2048, 8),
    (127,),
    (8, 127),
    (1, 22, 127),
    (8, 4, 8, 127),
    (255,),
    (16, 255),
    (1, 44, 255),
    (16, 8, 16, 255),
    (1025,),
    (32, 1025),
    (1, 88, 1025),
    (32, 1632, 1025),
    (1024, 10240),
    (1024, 14336),
    (1024, 18432),
    (1024, 22528),
    # (1024,1048576),
]


def fn_torch(x0):
    res = x0 * 0.5 * (1.0 + torch.erf(x0 / torch.sqrt(torch.tensor(2.0))))
    return res


def run_test():
    dtype = torch.float32
    perf_list = []

    for shape in SHAPES:
        print(f"torch_gelu|dtype=float32|shape={shape}")

        # create input tensor
        x = torch.randn(shape, dtype=dtype).npu()

        # benchmark
        time_torch = do_bench_npu(lambda x=x: fn_torch(x))
        print("<<<<< time_torch in us", time_torch * 1000)

        msg = f"torch_gelu|float32|{shape}|{time_torch * 1000}"  # in us
        perf_list.append(msg)

    print("\n==== Performance Result ====")
    for m in perf_list:
        print(m)


if __name__ == "__main__":
    run_test()
