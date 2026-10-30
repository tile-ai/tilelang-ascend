"""Correctness-gated timing and single-entry profiling of the fixed decode domain."""

import argparse
from functools import partial
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

import torch
import torch_npu
import tilelang

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from gated_mla_golden import assert_fp16_close, validate_inputs
from example_gated_mla_decode import (
    gated_mla_core,
    latent_readout_kernel,
    q_absorb_kernel,
    score_kernel,
    softmax_kernel,
    value_gate_kernel,
)
from gated_mla_golden import expanded_reference
from gated_mla_golden import make_decode_inputs, stage_reference

STAGES = ("q_absorb", "scores", "softmax", "latent_readout", "value_gate")
SCALE = 1 / math.sqrt(192)


def validate_domain(inputs, metadata):
    spec = validate_inputs(**inputs, **metadata, require_npu=True, allowed_dtypes=(torch.float16,))
    if (spec.B, spec.Tq, spec.Tk, spec.H, spec.D, spec.E, spec.R, spec.V) != (1, 1, 64, 1, 128, 64, 512, 128):
        raise ValueError("This benchmark supports only the frozen Tk=64 decode domain")
    return spec


def torch_fp32_core(inputs, metadata):
    spec = validate_domain(inputs, metadata)
    if spec.q_lens[0] == 0:
        return torch.zeros_like(inputs["gate_logits"])
    visible = min(spec.kv_lens[0], spec.query_start[0] + 1)
    ql = inputs["q_content"][0, 0, 0].float() @ inputs["w_k"][0].float()
    cache = inputs["latent_kv"][0, :visible].float()
    aux = inputs["key_aux"][0, :visible].float()
    scores = (cache @ ql + aux @ inputs["q_aux"][0, 0, 0].float()) * spec.scale
    latent = torch.softmax(scores, dim=0) @ cache
    value = inputs["w_v"][0].float() @ latent
    return (value * torch.sigmoid(inputs["gate_logits"][0, 0, 0].float())).half().reshape(1, 1, 1, 128)


def fia_core(inputs, metadata, block_table):
    spec = validate_domain(inputs, metadata)
    if spec.q_lens[0] == 0:
        return torch.zeros_like(inputs["gate_logits"])
    visible = min(spec.kv_lens[0], spec.query_start[0] + 1)
    ql = (inputs["q_content"][0, 0, 0].float() @ inputs["w_k"][0].float()).half()
    cache = inputs["latent_kv"].view(4, 1, 16, 512)
    aux_cache = inputs["key_aux"].view(4, 1, 16, 64)
    latent, _ = torch_npu.npu_fused_infer_attention_score_v2(
        ql.view(1, 1, 1, 512),
        cache,
        cache,
        query_rope=inputs["q_aux"].view(1, 1, 1, 64),
        key_rope=aux_cache,
        block_table=block_table,
        block_size=16,
        actual_seq_kvlen=[visible],
        num_query_heads=1,
        num_key_value_heads=1,
        input_layout="BNSD",
        softmax_scale=spec.scale,
        sparse_mode=0,
    )
    value = inputs["w_v"][0].float() @ latent.reshape(512).float()
    return (value * torch.sigmoid(inputs["gate_logits"][0, 0, 0].float())).half().reshape(1, 1, 1, 128)


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


def make_stages(cpu_inputs, inputs, visible):
    metadata = dict(q_lens=(1,), kv_lens=(64,), query_start=(visible - 1,))
    reference = stage_reference(cpu_inputs, metadata)
    ql = reference["q_latent"].float().npu()
    scores = reference["scores"].float().npu()
    probabilities = reference["probs"].float().npu()
    latent = reference["latent_out"].float().npu()
    kernels = {
        "q_absorb": q_absorb_kernel(),
        "scores": score_kernel(visible, SCALE),
        "softmax": softmax_kernel(),
        "latent_readout": latent_readout_kernel(visible),
        "value_gate": value_gate_kernel(),
    }
    arguments = {
        "q_absorb": (inputs["q_content"], inputs["w_k"]),
        "scores": (ql, inputs["q_aux"], inputs["latent_kv"], inputs["key_aux"]),
        "softmax": (scores,),
        "latent_readout": (probabilities, inputs["latent_kv"]),
        "value_gate": (latent, inputs["w_v"], inputs["gate_logits"]),
    }
    return kernels, arguments


def command_metadata(command):
    try:
        result = subprocess.run(command, capture_output=True, text=True)
    except OSError as error:
        return {"status": "unavailable", "returncode": None, "stdout": "", "stderr": str(error)}
    return {
        "status": "ok" if result.returncode == 0 else "failed",
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def git_revision(root):
    result = command_metadata(["git", "-C", str(root), "rev-parse", "HEAD"])
    return result["stdout"].strip() if result["returncode"] == 0 else "unavailable"


def environment(source):
    native = sorted(
        {
            line.split()[-1]
            for line in Path("/proc/self/maps").read_text().splitlines()
            if ("libtilelang" in line or "libtvm" in line) and ".so" in line
        }
    )
    root = Path(tilelang.__file__).resolve().parents[1]
    return {
        "date": datetime.now(timezone(timedelta(hours=9))).isoformat(),
        "python": sys.version,
        "python_executable": sys.executable,
        "torch": torch.__version__,
        "torch_npu": torch_npu.__version__,
        "tilelang": tilelang.__version__,
        "tilelang_file": tilelang.__file__,
        "tilelang_commit": git_revision(root),
        "native_libraries": native,
        "device": torch.npu.get_device_name(0),
        "device_properties": str(torch.npu.get_device_properties(0)),
        "npu_smi": command_metadata(["npu-smi", "info"]),
        "cann_version": next(
            (
                p.read_text()
                for p in (
                    Path(os.environ.get("ASCEND_HOME_PATH", "/usr/local/Ascend/ascend-toolkit/latest")) / "opp/version.info",
                    Path(os.environ.get("ASCEND_OPP_PATH", "/usr/local/Ascend/ascend-toolkit/latest/opp")) / "version.info",
                )
                if p.is_file()
            ),
            "unavailable",
        ),
        "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source.glob("*.py")},
        "benchmark_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }


def profile_entry(stage, output, visible):
    cpu = make_decode_inputs(0)
    inputs = {name: value.npu() for name, value in cpu.items()}
    metadata = dict(q_lens=(1,), kv_lens=(64,), query_start=(visible - 1,))
    kernels, arguments = make_stages(cpu, inputs, visible)
    for name, kernel in kernels.items():
        (output / (name + ".cpp")).write_text(kernel.get_kernel_source(), encoding="utf-8")
    torch.npu.synchronize()
    if stage == "chain":
        result = gated_mla_core(**inputs, **metadata)
        torch.npu.synchronize()
        assert_fp16_close(result.cpu(), expanded_reference(**cpu, **metadata))
    else:
        result = kernels[stage](*arguments[stage])
        torch.npu.synchronize()
        reference = stage_reference(cpu, metadata)
        if stage == "value_gate":
            torch.testing.assert_close(result[0].cpu().double(), reference["ungated"], rtol=1e-3, atol=5e-5)
            assert_fp16_close(result[1].cpu(), expanded_reference(**cpu, **metadata))
        else:
            key = {"q_absorb": "q_latent", "scores": "scores", "softmax": "probs", "latent_readout": "latent_out"}[stage]
            torch.testing.assert_close(result.cpu().double(), reference[key], rtol=1e-3, atol=5e-5)
    expected = list(STAGES) if stage == "chain" else [stage]
    (output / "profile_entry.json").write_text(
        json.dumps(
            {
                "stage": stage,
                "visible": visible,
                "expected_launches": len(expected),
                "launch_order": expected,
                "runtime_kernel_name": "main_kernel",
                "warmup_in_entry": 0,
            },
            indent=2,
        )
        + "\n"
    )
    print("GMLA_PROFILE_ENTRY_PASS " + json.dumps({"stage": stage, "expected_launches": len(expected)}), flush=True)


def benchmark(output, warmup, repeats, graph_batch):
    result = {
        "shape": {"B": 1, "H": 1, "Tq": 1, "Tk": 64, "D": 128, "E": 64, "R": 512, "V": 128},
        "dtype": "float16",
        "scale": SCALE,
        "warmup_calls": warmup,
        "cache_policy": "reused inputs and weights; no L2 flush",
        "eager_event_scope": "complete call interval including device idle gaps from host dispatch",
        "graph_scope": "batched captured operator calls; Python metadata checks and allocation dispatch occur during capture",
        "fia_layout": "identity block table prepared once; zero-copy cache views created inside each call",
        "candidate_validation": [],
        "cases": [],
    }
    block_table = torch.arange(4, dtype=torch.int32).reshape(1, 4).npu()
    fia_supported = True
    for seed in (0, 1, 2, 3, 17):
        cpu = make_decode_inputs(seed)
        inputs = {name: value.npu() for name, value in cpu.items()}
        for visible in (1, 17, 64):
            meta = dict(q_lens=(1,), kv_lens=(64,), query_start=(visible - 1,))
            expected = expanded_reference(**cpu, **meta)
            validation = {"seed": seed, "visible": visible}
            validation["torch_fp32"] = assert_fp16_close(torch_fp32_core(inputs, meta).cpu(), expected)
            if fia_supported:
                try:
                    actual = fia_core(inputs, meta, block_table)
                    torch.npu.synchronize()
                    validation["cann_fia"] = assert_fp16_close(actual.cpu(), expected)
                except RuntimeError as error:
                    result["fia_unavailable"] = {"seed": seed, "visible": visible, "error": str(error)}
                    (output / "candidate_status.json").write_text(json.dumps(result["fia_unavailable"], indent=2) + "\n")
                    if "507015" in str(error) or "aicore execution is abnormal" in str(error):
                        raise
                    fia_supported = False
            result["candidate_validation"].append(validation)
    for visible in (64, 17):
        cpu = make_decode_inputs(0)
        inputs = {name: value.npu() for name, value in cpu.items()}
        meta = dict(q_lens=(1,), kv_lens=(64,), query_start=(visible - 1,))
        expected = expanded_reference(**cpu, **meta)
        calls = {
            "tilelang_five_stage": partial(gated_mla_core, **inputs, **meta),
            "torch_npu_fp32_composition": partial(torch_fp32_core, inputs, meta),
        }
        if fia_supported:
            calls["cann_fia_composition"] = partial(fia_core, inputs, meta, block_table)
        case = {"visible": visible, "methods": {}, "tilelang_stages": {}}
        for name, function in calls.items():
            eager, actual = eager_timing(function, warmup, repeats)
            assert_fp16_close(actual.cpu(), expected)
            entry = {"eager": eager}
            try:
                graph, actual = graph_timing(function, graph_batch, repeats)
                assert_fp16_close(actual.cpu(), expected)
                entry["graph"] = graph
            except RuntimeError as error:
                entry["graph_error"] = str(error)
            case["methods"][name] = entry
            print(
                "GMLA_TIMING "
                + json.dumps(
                    {
                        "visible": visible,
                        "method": name,
                        "eager_event_median_us": eager["event_interval"]["median"],
                        "graph_median_us": entry.get("graph", {}).get("per_call", {}).get("median"),
                    }
                ),
                flush=True,
            )
        kernels, arguments = make_stages(cpu, inputs, visible)
        for name, kernel in kernels.items():
            function = partial(kernel, *arguments[name])
            for _ in range(warmup):
                function()
            torch.npu.synchronize()
            graph, _ = graph_timing(function, graph_batch, repeats)
            case["tilelang_stages"][name] = graph
            (output / f"visible_{visible}_{name}.cpp").write_text(kernel.get_kernel_source(), encoding="utf-8")
        result["cases"].append(case)
        (output / "benchmark.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("benchmark", "profile"), default="benchmark")
    parser.add_argument("--stage", choices=("chain", *STAGES), default="chain")
    parser.add_argument("--visible", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--graph-batch", type=int, default=50)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.warmup, args.repeats, args.graph_batch) < 1:
        parser.error("measurement counts must be positive")
    if not 1 <= args.visible <= 64:
        parser.error("visible prefix must be in [1,64]")
    if args.output.exists() and (not args.output.is_dir() or any(args.output.iterdir())):
        parser.error("--output must be a new or empty directory; previous evidence is preserved")
    args.output.mkdir(parents=True, exist_ok=True)
    torch.npu.set_device(0)
    source = Path(__import__("example_gated_mla_decode").__file__).parent
    (args.output / "environment.json").write_text(json.dumps(environment(source), indent=2) + "\n")
    if args.mode == "profile":
        profile_entry(args.stage, args.output, args.visible)
    else:
        benchmark(args.output, args.warmup, args.repeats, args.graph_batch)
    print("GMLA_MEASUREMENT_PASS", flush=True)


if __name__ == "__main__":
    main()
