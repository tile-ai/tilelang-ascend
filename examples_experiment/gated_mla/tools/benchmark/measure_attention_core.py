"""Compare the absorbed MLA attention core with CANN FIA on identical cache data.

Query absorption, value projection and gating are outside this measurement.
Event/graph timings are supplementary diagnostics; use the profile entry with
msprof op for hardware profiling, or msprof op simulator for simulation.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import torch
import torch_npu

SCALE = 1.0 / math.sqrt(192)
QUERY_POLICIES = ("native_fp32_query", "matched_fp16_query")


def core_reference(q_latent, q_aux, latent_kv, key_aux, visible):
    """FP64 prefix-slicing reference, using the declared query input values."""
    c = latent_kv[0, :visible].double()
    e = key_aux[0, :visible].double()
    scores = (c @ q_latent.double() + e @ q_aux[0, 0, 0].double()) * SCALE
    exponentials = torch.exp(scores - scores.max())
    probabilities = exponentials / exponentials.sum()
    return (probabilities[:, None] * c).sum(dim=0)


def prepare(cpu_inputs, policy, device):
    ql = (cpu_inputs["q_content"][0, 0, 0].double() @ cpu_inputs["w_k"][0].double()).float()
    ql_half = ql.half()
    if policy == "matched_fp16_query":
        ql = ql_half.float()
    elif policy != "native_fp32_query":
        raise ValueError(f"Unsupported query policy: {policy}")
    data = {
        "q_latent_fp32": ql.to(device),
        "q_latent_fp16": ql_half.to(device),
        "q_aux": cpu_inputs["q_aux"].to(device),
        "latent_kv": cpu_inputs["latent_kv"].to(device),
        "key_aux": cpu_inputs["key_aux"].to(device),
        "block_table": torch.arange(4, dtype=torch.int32).reshape(1, 4).to(device),
    }
    # These zero-copy views and all dtype conversions are outside measurements.
    data["fia_query"] = data["q_latent_fp16"].view(1, 1, 1, 512)
    data["fia_cache"] = data["latent_kv"].view(4, 1, 16, 512)
    data["fia_key_aux"] = data["key_aux"].view(4, 1, 16, 64)
    validate_core_inputs(data)
    return data, ql


def validate_core_inputs(data):
    expected = {
        "q_latent_fp32": ((512,), torch.float32),
        "q_latent_fp16": ((512,), torch.float16),
        "q_aux": ((1, 1, 1, 64), torch.float16),
        "latent_kv": ((1, 64, 512), torch.float16),
        "key_aux": ((1, 64, 64), torch.float16),
        "block_table": ((1, 4), torch.int32),
    }
    device = data["q_latent_fp32"].device
    if device.type not in ("npu", "privateuseone"):
        raise ValueError("The attention core benchmark requires an NPU")
    for name, (shape, dtype) in expected.items():
        value = data[name]
        if tuple(value.shape) != shape or value.dtype != dtype:
            raise ValueError(f"Unsupported {name} shape/dtype: {value.shape}, {value.dtype}")
        if value.device != device or not value.is_contiguous():
            raise ValueError(f"{name} must be contiguous and on {device}")


def make_calls(data, visible, operator):
    if type(visible) is not int or not 1 <= visible <= 64:
        raise ValueError("visible must be an integer in [1, 64]")
    kernels = {
        "scores": operator.score_kernel(visible, SCALE),
        "softmax": operator.softmax_kernel(),
        "latent_readout": operator.latent_readout_kernel(visible),
    }

    def tilelang_attention():
        scores = kernels["scores"](data["q_latent_fp32"], data["q_aux"], data["latent_kv"], data["key_aux"])
        probabilities = kernels["softmax"](scores)
        return kernels["latent_readout"](probabilities, data["latent_kv"])

    def cann_fia_attention():
        latent, _ = torch_npu.npu_fused_infer_attention_score_v2(
            data["fia_query"],
            data["fia_cache"],
            data["fia_cache"],
            query_rope=data["q_aux"],
            key_rope=data["fia_key_aux"],
            block_table=data["block_table"],
            block_size=16,
            actual_seq_kvlen=[visible],
            num_query_heads=1,
            num_key_value_heads=1,
            input_layout="BNSD",
            softmax_scale=SCALE,
            sparse_mode=0,
        )
        return latent.view(512)

    return {"tilelang_attention": tilelang_attention, "cann_fia_attention": cann_fia_attention}, kernels


def check_result(actual, expected, name, golden):
    expected_dtype = torch.float32 if name == "tilelang_attention" else torch.float16
    if tuple(actual.shape) != (512,) or actual.dtype != expected_dtype:
        raise AssertionError(f"Unexpected {name} output: {actual.shape}, {actual.dtype}")
    metrics = golden.assert_fp16_close(actual, expected)
    if name == "tilelang_attention":
        # Preserve the existing stricter FP32 stage acceptance in addition to the
        # common comparison budget used for the FP16 FIA result.
        torch.testing.assert_close(actual.cpu().double(), expected, rtol=1e-3, atol=5e-5)
    return metrics


def backend_metadata(kernels):
    """Record the backend from the instantiated kernels, not the CLI default.

    A framework-auto target is intentionally not relabeled as PTO. If it is
    retained by a runtime, resolving it needs separate generated-code evidence.
    """
    targets = {name: str(kernel.target) for name, kernel in kernels.items()}
    unique = set(targets.values())
    backend = next(iter(unique)) if len(unique) == 1 and unique <= {"pto", "ascendc"} else "unresolved"
    return {"candidate_kernel_targets": targets, "candidate_backend": backend, "backend_evidence": "instantiated JITKernel.target"}


def result_header():
    return {
        "scope": "absorbed MLA attention only: scores + softmax + latent readout",
        "excluded": ["query absorption", "value projection", "sigmoid gate", "output projection"],
        "shape": {"B": 1, "H": 1, "Tq": 1, "Tk": 64, "R": 512, "E": 64},
        "scale": SCALE,
        "mask": "single-query causal/length prefix; exactly visible keys on both paths",
        "query_policies": {
            "native_fp32_query": "Candidate gets FP32 absorbed-query values; FIA rounds those values to FP16. Both are checked against the same FP64 reference built from the FP32 values.",
            "matched_fp16_query": "Only the input query values are matched: both receive exactly the same FP16-representable values, stored in FP32 by the candidate and FP16 by FIA. Candidate output is FP32 and FIA output FP16; equal internal or output precision is not claimed.",
        },
        "output_dtypes": {"tilelang_attention": "float32", "cann_fia_attention": "float16"},
        "common_accuracy_budget": {"rtol": 1e-2, "atol": 2e-3, "max_normalized": 5e-3},
        "candidate_fp32_stage_budget": {"rtol": 1e-3, "atol": 5e-5},
        "preparation": "CPU query absorption, dtype conversion, H2D copies, identity block table and input/cache views are prepared before measurement for both methods. The FIA call includes a zero-copy output view to shape (512,); it does not cast the output. Each method retains its own output allocation and call dispatch.",
        "reference": "Independent FP64 prefix slicing, stable exponential normalization, and weighted latent sum; no candidate output is reused as a golden.",
        "evidence_boundary": "Event/graph timings are supplementary. msprof op hardware measurements and msprof op simulator results are separate evidence.",
        "cache_policy": "reused inputs and weights; no L2 flush",
    }


def benchmark(args, operator, golden, shared):
    report = result_header()
    report.update(warmup_calls=args.warmup, repeats=args.repeats, calls_per_graph=args.graph_batch)
    report["validation"] = []
    report["cases"] = []
    for policy in QUERY_POLICIES:
        for seed in (0, 1, 2, 3, 17):
            cpu = golden.make_decode_inputs(seed)
            data, ql = prepare(cpu, policy, args.device)
            for visible in (1, 17, 64):
                calls, kernels = make_calls(data, visible, operator)
                expected = core_reference(ql, cpu["q_aux"], cpu["latent_kv"], cpu["key_aux"], visible)
                row = {
                    "policy": policy,
                    "seed": seed,
                    "visible": visible,
                    "fia_query_rounding_max_abs": float((ql - ql.half().float()).abs().max()),
                    "methods": {},
                }
                row.update(backend_metadata(kernels))
                for name, function in calls.items():
                    actual = function()
                    torch.npu.synchronize()
                    row["methods"][name] = check_result(actual, expected, name, golden)
                report["validation"].append(row)
    for policy in QUERY_POLICIES:
        cpu = golden.make_decode_inputs(0)
        data, ql = prepare(cpu, policy, args.device)
        for visible in (64, 17):
            calls, kernels = make_calls(data, visible, operator)
            expected = core_reference(ql, cpu["q_aux"], cpu["latent_kv"], cpu["key_aux"], visible)
            case = {"policy": policy, "visible": visible, "methods": {}}
            case.update(backend_metadata(kernels))
            for name, function in calls.items():
                eager, actual = shared.eager_timing(function, args.warmup, args.repeats)
                check_result(actual, expected, name, golden)
                graph, actual = shared.graph_timing(function, args.graph_batch, args.repeats)
                check_result(actual, expected, name, golden)
                case["methods"][name] = {"eager": eager, "graph": graph}
                print(
                    "GMLA_CORE_TIMING "
                    + json.dumps(
                        {
                            "policy": policy,
                            "visible": visible,
                            "method": name,
                            "eager_event_median_us": eager["event_interval"]["median"],
                            "graph_median_us": graph["per_call"]["median"],
                        }
                    ),
                    flush=True,
                )
            for name, kernel in kernels.items():
                (args.output / f"visible_{visible}_{name}.cpp").write_text(kernel.get_kernel_source(), encoding="utf-8")
            report["cases"].append(case)
            (args.output / "attention_core_benchmark.json").write_text(json.dumps(report, indent=2) + "\n")


def profile(args, operator, golden):
    cpu = golden.make_decode_inputs(0)
    data, ql = prepare(cpu, args.query_policy, args.device)
    calls, kernels = make_calls(data, args.visible, operator)
    if args.method == "tilelang_attention":
        for name, kernel in kernels.items():
            (args.output / f"{name}.cpp").write_text(kernel.get_kernel_source(), encoding="utf-8")
    expected = core_reference(ql, cpu["q_aux"], cpu["latent_kv"], cpu["key_aux"], args.visible)
    torch.npu.synchronize()
    actual = calls[args.method]()
    torch.npu.synchronize()
    metrics = check_result(actual, expected, args.method, golden)
    report = result_header()
    report.update(backend_metadata(kernels))
    report.update(
        method=args.method,
        query_policy=args.query_policy,
        visible=args.visible,
        expected_python_operator_calls=3 if args.method == "tilelang_attention" else 1,
        expected_candidate_kernel_launches=3 if args.method == "tilelang_attention" else None,
        fia_launch_note="FIA is one public API call; its physical kernel launch count must be read from profiling.",
        accuracy=metrics,
        warmup_in_entry=0,
    )
    report["executed_backend"] = report["candidate_backend"] if args.method == "tilelang_attention" else "cann_fia_v2"
    report["executed_candidate_kernel_targets"] = report["candidate_kernel_targets"] if args.method == "tilelang_attention" else {}
    (args.output / "profile_entry.json").write_text(json.dumps(report, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("benchmark", "profile"), default="benchmark")
    parser.add_argument("--operator-dir", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--method", choices=("tilelang_attention", "cann_fia_attention"), default="tilelang_attention")
    parser.add_argument("--query-policy", choices=QUERY_POLICIES, default="matched_fp16_query")
    parser.add_argument("--visible", type=int, default=64)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--graph-batch", type=int, default=50)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.warmup, args.repeats, args.graph_batch) < 1 or not 1 <= args.visible <= 64:
        parser.error("counts must be positive and visible must be in [1,64]")
    if args.output.exists() and (not args.output.is_dir() or any(args.output.iterdir())):
        parser.error("--output must be new or empty; previous evidence is preserved")
    if not torch.npu.is_available():
        raise RuntimeError("An Ascend NPU is required; this benchmark cannot skip FIA or device execution")
    torch.npu.set_device(args.device)
    args.device = torch.device(f"npu:{args.device}")
    args.operator_dir = args.operator_dir.resolve()
    for name in ("example_gated_mla_decode.py", "gated_mla_golden.py", "tools/benchmark/measure_decode.py"):
        if not (args.operator_dir / name).is_file():
            parser.error(f"Missing operator source: {args.operator_dir / name}")
    sys.path.insert(0, str(args.operator_dir))
    sys.path.insert(0, str(args.operator_dir / "tools/benchmark"))
    import example_gated_mla_decode as operator
    import gated_mla_golden as golden
    import measure_decode as shared

    args.output.mkdir(parents=True, exist_ok=True)
    environment = shared.environment(args.operator_dir)
    environment["attention_core_benchmark_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (args.output / "environment.json").write_text(json.dumps(environment, indent=2) + "\n")
    if args.mode == "profile":
        profile(args, operator, golden)
    else:
        benchmark(args, operator, golden, shared)
    print("GMLA_ATTENTION_CORE_PASS", flush=True)


if __name__ == "__main__":
    main()
