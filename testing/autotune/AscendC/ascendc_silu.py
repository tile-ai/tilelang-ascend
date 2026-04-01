import torch

torch.npu.set_device(10)

from triton.backends.ascend.testing import do_bench_npu

SHAPES = [
    (32, 64),
    (32, 128),
    (32, 2048),
    (16384, 4096),
    (22, 127),
    (16, 255),
    (44, 255),
    (32, 1025),
    (800, 4090),
]


def fn_torch(in_tensor, scale_tensor, r1_numel):
    squared = in_tensor**2
    sum_squared = torch.sum(squared, dim=1, keepdim=True)
    mean_squared = sum_squared / r1_numel
    normalized = mean_squared + 1e-05
    rsqrt_norm = torch.rsqrt(normalized)
    normalized_input = in_tensor * rsqrt_norm
    output = normalized_input * scale_tensor
    return output


def run_test():
    dtype = torch.float32
    perf_list = []

    for shape in SHAPES:
        print(f"torch_silu|dtype=float32|shape={shape}")

        # create input tensor
        x = torch.randn(shape, dtype=dtype).npu()

        # benchmark
        time_torch = do_bench_npu(lambda x=x: fn_torch(x))
        print("<<<<< time_torch in us", time_torch * 1000)

        msg = f"torch_silu|float32|{shape}|{time_torch * 1000}"  # in us
        perf_list.append(msg)

    print("\n==== Performance Result ====")
    for m in perf_list:
        print(m)


if __name__ == "__main__":
    run_test()
