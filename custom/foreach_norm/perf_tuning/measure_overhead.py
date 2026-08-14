"""Measure TileLang kernel launch overhead vs CANN native op."""
import os, sys, time
import torch
import tilelang
from tilelang import language as T

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}


@tilelang.jit(out_idx=[1], pass_configs=pass_configs)
def trivial_kernel(N=1):
    @T.prim_func
    def main(X: T.Tensor((N,), "float32"), Y: T.Tensor((N,), "float32")):  # type: ignore
        with T.Kernel(1, is_npu=True) as (cid, vid):
            x = T.alloc_shared((N,), "float32")
            T.copy(X, x)
            T.copy(x, Y)
    return main


@tilelang.jit(out_idx=[1], pass_configs=pass_configs)
def trivial_multicore_kernel(launch_cores=24, N=1):
    @T.prim_func
    def main(X: T.Tensor((N,), "float32"), Y: T.Tensor((launch_cores,), "float32")):  # type: ignore
        with T.Kernel(launch_cores, is_npu=True) as (cid, vid):
            acc = T.alloc_shared((1,), "float32")
            T.tile.fill(acc, 0.0)
            T.copy(acc, Y[cid])
    return main


def main():
    k1 = trivial_kernel()
    k2 = trivial_multicore_kernel()

    x = torch.randn(1, dtype=torch.float32, device="npu")

    # warmup
    for _ in range(10):
        y = k1(x)
        y2 = k2(x)
    torch.npu.synchronize()

    # Measure trivial 1-block TileLang kernel
    times = []
    for _ in range(50):
        torch.npu.synchronize()
        t0 = time.perf_counter()
        y = k1(x)
        torch.npu.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1e6)
    times.sort()
    print(f"TileLang 1-block trivial: {times[len(times)//2]:.1f} us (median of 50)")

    # Measure trivial 24-core TileLang kernel
    times = []
    for _ in range(50):
        torch.npu.synchronize()
        t0 = time.perf_counter()
        y2 = k2(x)
        torch.npu.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1e6)
    times.sort()
    print(f"TileLang 24-core trivial: {times[len(times)//2]:.1f} us (median of 50)")

    # CANN native: torch.sum on 1 element
    for _ in range(10):
        _ = x.sum()
    torch.npu.synchronize()
    times = []
    for _ in range(50):
        torch.npu.synchronize()
        t0 = time.perf_counter()
        _ = x.sum()
        torch.npu.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1e6)
    times.sort()
    print(f"CANN torch.sum(1 elem): {times[len(times)//2]:.1f} us (median of 50)")

    # CANN native: torch.norm on 1M fp32
    x_big = torch.randn(1000003, dtype=torch.float32, device="npu")
    for _ in range(10):
        _ = torch.norm(x_big, p=float("inf"))
    torch.npu.synchronize()
    times = []
    for _ in range(50):
        torch.npu.synchronize()
        t0 = time.perf_counter()
        _ = torch.norm(x_big, p=float("inf"))
        torch.npu.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1e6)
    times.sort()
    print(f"CANN torch.norm(1M fp32, inf): {times[len(times)//2]:.1f} us (median of 50)")

    # CANN native: torch.norm on 1M bf16 (upcast to fp32)
    x_bf16 = torch.randn(1000003, dtype=torch.bfloat16, device="npu")
    x_bf16_f32 = x_bf16.to(torch.float32)
    for _ in range(10):
        _ = torch.norm(x_bf16_f32, p=float("inf"))
    torch.npu.synchronize()
    times = []
    for _ in range(50):
        torch.npu.synchronize()
        t0 = time.perf_counter()
        _ = torch.norm(x_bf16_f32, p=float("inf"))
        torch.npu.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1e6)
    times.sort()
    print(f"CANN torch.norm(1M bf16->fp32, inf): {times[len(times)//2]:.1f} us (median of 50)")


if __name__ == "__main__":
    main()
