# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""The profiler and convert to torch utils"""

import torch
from typing import Callable, List, Literal, Optional, Union


def do_bench(
    fn: Callable,
    warmup: float = 25,
    rep: float = 100,
    _n_warmup: int = 0,
    _n_repeat: int = 0,
    grad_to_none: Optional[List[torch.Tensor]] = None,
    quantiles: Optional[List[float]] = None,
    fast_flush: bool = True,
    return_mode: Literal["min", "max", "mean", "median"] = "mean",
) -> Union[float, List[float]]:
    """Benchmarks the runtime of a PyTorch function.

    This function handles:
    - L2 cache flushing between runs for consistent timing
    - Automatic warmup and repeat count calculation
    - Optional gradient clearing for backward passes
    - Multiple measurement modes (mean, median, min, max)

    Args:
        fn: Function to benchmark
        warmup: Target warmup time in milliseconds
        rep: Target number of repetitions
        _n_warmup: Override for number of warmup iterations
        _n_repeat: Override for number of timing iterations
        grad_to_none: Tensors whose gradients should be cleared between runs
        quantiles: Optional performance percentiles to compute
        fast_flush: Whether to use faster L2 cache flushing
        return_mode: How to aggregate timing results ("mean", "median", "min", "max")

    Returns:
        float: Aggregated runtime in milliseconds
    """
    assert return_mode in ["min", "max", "mean", "median"]
    fn()
    torch.npu.synchronize()

    # We maintain a buffer of 256 MB that we clear
    # before each kernel call to make sure that the L2
    # doesn't contain any input data before the run
    if fast_flush:
        cache = torch.empty(int(256e6 // 4), dtype=torch.int, device="npu")
    else:
        cache = torch.empty(int(256e6), dtype=torch.int8, device="npu")

    # Estimate the runtime of the function
    start_event = torch.npu.Event(enable_timing=True)
    end_event = torch.npu.Event(enable_timing=True)
    start_event.record()
    for _ in range(5):
        cache.zero_()
        fn()
    end_event.record()
    torch.npu.synchronize()
    estimate_ms = start_event.elapsed_time(end_event) / 5

    # compute number of warmup and repeat
    n_warmup = max(1, int(warmup / estimate_ms))
    n_repeat = max(1, int(rep / estimate_ms))
    if _n_warmup > 0:
        n_warmup = _n_warmup
    if _n_repeat > 0:
        n_repeat = _n_repeat
    start_event = [torch.npu.Event(enable_timing=True) for i in range(n_repeat)]
    end_event = [torch.npu.Event(enable_timing=True) for i in range(n_repeat)]
    # Warm-up
    for _ in range(n_warmup):
        fn()
    # Benchmark
    for i in range(n_repeat):
        # we don't want `fn` to accumulate gradient values
        # if it contains a backward pass. So we clear the
        # provided gradients
        if grad_to_none is not None:
            for x in grad_to_none:
                x.grad = None
        # we clear the L2 cache before each run
        cache.zero_()
        # record time of `fn`
        start_event[i].record()
        fn()
        end_event[i].record()
    # Record clocks
    torch.npu.synchronize()
    times = torch.tensor(
        [s.elapsed_time(e) for s, e in zip(start_event, end_event, strict=True)],
        dtype=torch.float,
    )
    if quantiles is not None:
        ret = torch.quantile(times, torch.tensor(quantiles, dtype=torch.float)).tolist()
        if len(ret) == 1:
            ret = ret[0]
        return ret
    return getattr(torch, return_mode)(times).item()


import builtins
import os


def do_bench_npu(funcs, warmup: float = 5, rep=30, prof_dir=None, keep_res=False):
    import torch_npu

    for fn in funcs:
        fn()
        torch.npu.synchronize()

    experimental_config = torch_npu.profiler._ExperimentalConfig(
        aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
        profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
        data_simplification=False,
    )
    if prof_dir is not None:
        torch_path = prof_dir
    else:
        torch_path = os.path.join(os.environ.get("TILELANG_CACHE_DIR"), "autotuner_tmp")

    total = warmup + rep
    with torch_npu.profiler.profile(
        activities=[torch_npu.profiler.ProfilerActivity.NPU],
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(torch_path),
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
        with_flops=False,
        with_modules=False,
        experimental_config=experimental_config,
    ):
        for fn in funcs:
            for _ in builtins.range(total):
                fn()
            torch.npu.synchronize()

    time_cost = _collect_prof_result(torch_path, funcs, warmup, rep)
    _rm_dic(keep_res, torch_path)
    return time_cost


def _rm_dic(keep_res, torch_path):
    if keep_res:
        return
    import shutil

    if os.path.exists(torch_path):
        shutil.rmtree(torch_path)


def _collect_prof_result(
    base_dir: str, funcs, num_warmup: int, num_active: int, key: str = None
):
    """
    Collect kernel performance from kernel_details.csv, returned in millisecond.
    The first `num_warmup` rows of each function are warmup data and will be ignored, the next `num_active` rows will be averaged.

    :param base_dir: the profiler path
    :type base_dir: str
    :param funcs: a list of Callable being profiled
    :type funcs: List[Callable]
    :param num_warmup: warmup count in kernel_details.csv of each fn
    :type num_warmup: int
    :param num_active: active count in kernel_details.csv of each fn
    :type num_active: int
    :param key: filter key for kernel name
    :type key: str
    """

    import numpy as np
    import pandas as pd

    kernel_details_file = None
    for root, _, files in os.walk(base_dir):
        for file in files:
            if file == "kernel_details.csv":
                kernel_details_file = os.path.join(root, file)
                break
    num_funcs = len(funcs)
    if kernel_details_file is None:
        if num_funcs == 1:
            return float("inf")
        else:
            return [float("inf")] * num_funcs

    df = pd.read_csv(kernel_details_file)
    # filter out l2 cache clearing operation
    filter_cond = ~df["Type"].str.contains(r"^ReduceSum$", case=False, na=False)
    filter_df = df[filter_cond]
    if key is not None:
        key_rows = filter_df[filter_df["Name"].str.contains(key, na=False)]
    else:
        key_rows = filter_df
    time_cost = [0] * num_funcs
    for func_idx in np.arange(0, num_funcs):
        for active_index in np.arange(0, num_active):
            row_index = func_idx * (num_warmup + num_active) + num_warmup + active_index
            time_cost[func_idx] += key_rows.iloc[row_index]["Duration(us)"]
    time_cost = [x / num_active / 1e3 for x in time_cost]

    if num_funcs == 1:
        return time_cost[0]
    else:
        return time_cost
