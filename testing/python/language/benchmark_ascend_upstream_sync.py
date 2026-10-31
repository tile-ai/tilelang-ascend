# Copyright (c) Tile-AI Organization.
# Licensed under the MIT License.
"""Bounded A3 AscendC comparison; run directly with --output report.json.

Uses the same device factories as the correctness regressions. Compilation,
correctness and NPU event latency are recorded separately. No speedup assertion
is made: these small kernels are a migration checkpoint, not a workload survey.
"""

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import time

import torch
import torch_npu  # noqa: F401

import tilelang
from tilelang import tvm

from test_ascend_pipeline_ring_provenance import _pipelined_abs
from test_ascend_upstream_sync_composite_loop import _ordinary_k_gemm
from test_ascend_upstream_sync_cube_pipeline import _pipelined_k_gemm
from test_ascend_upstream_sync_device import _PASS_CONFIGS
from test_tilelang_ascend_language_elementwise import vec_abs


def _cases():
    generator = torch.Generator().manual_seed(20261030)
    values = torch.randn((64, 128), generator=generator)
    yield "vector_abs", vec_abs(64, 128, 64, 128, "float"), [values], values.abs()
    values = torch.randn((8, 64), generator=generator)
    yield (
        "static_pipeline_b3",
        _pipelined_abs(8, 3),
        [values],
        values.abs().unsqueeze(0).expand(2, -1, -1),
    )
    first = torch.randn((16, 192), generator=generator, dtype=torch.float16)
    second = torch.randn((192, 16), generator=generator, dtype=torch.float16)
    yield (
        "gemm_kloop",
        _ordinary_k_gemm(3),
        [first, second],
        (first.float() @ second.float()).half(),
    )
    first = torch.randn((16, 320), generator=generator, dtype=torch.float16)
    second = torch.randn((320, 16), generator=generator, dtype=torch.float16)
    yield (
        "cube_pipeline_b2",
        _pipelined_k_gemm(5),
        [first, second],
        (first.float() @ second.float()).half(),
    )


def _ir_statistics(kernel):
    operations = Counter()
    event_ids = {}
    memory_maps = []
    for _name, function in kernel.artifact.device_mod.functions_items():
        if not isinstance(function, tvm.tir.PrimFunc):
            continue

        def visit(node):
            if isinstance(node, tvm.tir.Call):
                name = getattr(node.op, "name", "")
                if name in {
                    "tl.ascend_auto_set_flag",
                    "tl.ascend_auto_wait_flag",
                    "tl.ascend_auto_barrier",
                    "tl.ascend_pipe_barrier",
                }:
                    operations[name] += 1
                    if name in {"tl.ascend_auto_set_flag", "tl.ascend_auto_wait_flag"}:
                        event_ids.setdefault(str(node.args[0]), set()).add(str(node.args[1]))

        tvm.tir.stmt_functor.post_order_visit(function.body, visit)
        sizes = function.attrs.get("size_map") if function.attrs else None
        if sizes is not None:
            memory_maps.append({str(var): str(size) for var, size in sizes.items()})
    return {
        "sync_call_sites": dict(operations),
        "event_id_expressions": {name: sorted(values) for name, values in event_ids.items()},
        "physical_size_maps_bytes": memory_maps,
    }


def _measure(program, inputs, expected, backend, *, auto_cv):
    tilelang.disable_cache()
    started = time.perf_counter()
    kernel = tilelang.compile(
        program,
        out_idx=[],
        target="ascendc",
        platform="A3",
        pass_configs={
            **_PASS_CONFIGS,
            tilelang.PassConfigKey.TL_ASCEND_SYNC_BACKEND: backend,
            tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: auto_cv,
        },
        compile_flags=["--cce-auto-sync=off"],
    )
    compile_seconds = time.perf_counter() - started
    output = torch.empty_like(expected, device="npu")
    arguments = [value.npu() for value in inputs] + [output]
    kernel(*arguments)
    torch.npu.synchronize()
    tolerance = 1e-2 if expected.dtype == torch.float16 else 1e-6
    torch.testing.assert_close(output.cpu(), expected, rtol=tolerance, atol=tolerance)
    for _warmup in range(20):
        kernel(*arguments)
    torch.npu.synchronize()
    samples = []
    for _sample in range(30):
        begin, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
        begin.record()
        for _repeat in range(100):
            kernel(*arguments)
        end.record()
        end.synchronize()
        samples.append(begin.elapsed_time(end) * 1000 / 100)
    return {
        "correctness": "passed",
        "compile_seconds": compile_seconds,
        "latency_us_median": statistics.median(samples),
        "latency_us_min": min(samples),
        "latency_us_max": max(samples),
        "latency_us_samples": samples,
        **_ir_statistics(kernel),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args()
    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "device": torch.npu.get_device_name(),
        "torch_version": torch.__version__,
        "target": "ascendc",
        "platform": "A3",
        "cce_auto_sync": False,
        "cache": False,
        "warmup": 20,
        "samples": 30,
        "invocations_per_sample": 100,
        "latency_scope": "NPU event interval around a host-submitted batch; includes launch gaps",
        "limitations": "Four small kernels; this does not isolate device execution time or imply speedup",
        "sync_engine_abi": tvm.get_global_func("tl.ascend_sync.EngineCacheABI")(),
        "cases": {},
    }
    # Alternate order to avoid consistently favoring one engine's warm device.
    for index, (name, program, inputs, expected) in enumerate(_cases()):
        report["cases"][name] = {}
        for backend in ("legacy", "upstream") if index % 2 == 0 else ("upstream", "legacy"):
            print(f"{name}: {backend}", flush=True)
            report["cases"][name][backend] = _measure(program, inputs, expected, backend, auto_cv=name == "vector_abs")
            options.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
