"""Frozen original protocol definitions; see protocol_manifest.json."""

import math
import statistics
import time
import torch


def summaries(samples):
    ordered = sorted(samples)
    return {
        "unit": "microseconds",
        "count": len(samples),
        "samples": samples,
        "minimum": min(samples),
        "median": statistics.median(samples),
        "mean": statistics.mean(samples),
        "p90": ordered[math.ceil(len(ordered) * 0.9) - 1],
    }


def eager_timing(function, warmup, repeats):
    for _ in range(warmup):
        output = function()
    torch.npu.synchronize()
    event_times, wall_times = [], []
    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    for _ in range(repeats):
        start.record()
        before = time.perf_counter()
        output = function()
        end.record()
        end.synchronize()
        wall_times.append((time.perf_counter() - before) * 1e6)
        event_times.append(start.elapsed_time(end) * 1000)
    return {"event_interval": summaries(event_times), "wall_with_end_event_sync": summaries(wall_times)}, output


def graph_timing(function, batch, repeats):
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        for _ in range(batch):
            output = function()
    for _ in range(5):
        graph.replay()
    torch.npu.synchronize()
    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    times = []
    for _ in range(repeats):
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end) * 1000 / batch)
    return {"calls_per_graph": batch, "per_call": summaries(times)}, output
