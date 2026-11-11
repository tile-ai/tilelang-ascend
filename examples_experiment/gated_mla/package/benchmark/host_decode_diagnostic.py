"""Separate uninstrumented public-call timing from warm cProfile attribution.

Fixed case: FP16 B1/H96/Tq1/Tk4096, visible17. This is a candidate-only host
diagnostic, not a source-baseline comparison or a validator-cost subtraction.
"""

import argparse
import cProfile
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import pstats
import shlex
import statistics
import sys
import time
import traceback


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def summary(samples):
    ordered = sorted(samples)
    return {
        "unit": "microseconds",
        "count": len(samples),
        "samples": samples,
        "minimum": min(samples),
        "median": statistics.median(samples),
        "mean": statistics.mean(samples),
        "p90": ordered[math.ceil(0.9 * len(ordered)) - 1],
    }


def profile_rows(profiler, public_calls):
    """Inclusive times overlap; self times exclude profiled child calls."""
    if not profiler.getstats():
        return []
    rows = []
    for (filename, line, function), (primitive, total, self_time, cumulative, _) in pstats.Stats(profiler).stats.items():
        rows.append(
            {
                "file": filename,
                "line": line,
                "function": function,
                "primitive_calls": primitive,
                "total_calls": total,
                "calls_per_public_forward": total / public_calls,
                "self_us_per_public_forward": self_time * 1e6 / public_calls,
                "inclusive_us_per_public_forward": cumulative * 1e6 / public_calls,
            }
        )
    return sorted(rows, key=lambda row: row["inclusive_us_per_public_forward"], reverse=True)


def run_host_profile(function, synchronize, *, calls, output_dir):
    """Reusable driver hook: profile the unchanged callable; sync outside cProfile.

    No data access, validation bypass, or NPU dependency is implemented here.
    On any exception no extra synchronization/call is attempted by this helper.
    """
    if calls < 1:
        raise ValueError("profile calls must be positive")
    output_dir = Path(output_dir)
    profiler = cProfile.Profile()
    host_times, drain_times = [], []
    completed, attempted = 0, 0
    try:
        for _ in range(calls):
            synchronize()
            started = time.perf_counter()
            attempted += 1
            profiler.enable()
            try:
                output = function()
            finally:
                profiler.disable()
            host_times.append((time.perf_counter() - started) * 1e6)
            started = time.perf_counter()
            synchronize()
            drain_times.append((time.perf_counter() - started) * 1e6)
            completed += 1
    finally:
        profiler.disable()
        profiler.dump_stats(str(output_dir / "host_profile.pstats"))
        rows = profile_rows(profiler, max(attempted, 1))
        (output_dir / "host_profile_functions.json").write_text(
            json.dumps({"completed_calls": completed, "profiled_attempts": attempted, "requested_calls": calls, "rows": rows}, indent=2)
            + "\n",
            encoding="utf-8",
        )
    selected_names = {"validate_forward_inputs", "require_dense_base", "guarded_view", "_view", "lambda_forward", "_forward_validated"}
    return {
        "instrumented": True,
        "completed_calls": completed,
        "profiled_call_including_profiler_overhead": summary(host_times),
        "unprofiled_post_call_drain": summary(drain_times),
        "top_inclusive": rows[:20],
        "top_self": sorted(rows, key=lambda row: row["self_us_per_public_forward"], reverse=True)[:20],
        "selected_functions": [row for row in rows if row["function"] in selected_names],
        "interpretation": (
            "Elapsed host attribution includes launch/driver waits and profiler perturbation. Synchronization is outside cProfile. "
            "Inclusive rows overlap and must not be added. Missing Cython internals are not proof of zero cost. "
            "This is not the causal cost of removing validation and not an Event-minus-graph attribution."
        ),
    }, output


def load_protocol(name, root):
    spec = importlib.util.spec_from_file_location("gmla_host_diagnostic_" + name, root / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def verify_protocol(root):
    manifest_path = root / "protocol_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for name, record in manifest["files"].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root.resolve()) or sha(path) != record["extracted_sha256"]:
            raise RuntimeError(f"Frozen protocol source differs: {name}")
    return {"manifest_sha256": sha(manifest_path), "files": manifest["files"]}


def verify_install(gmla, installed_root, inventory_file):
    root = installed_root.resolve()
    if Path(gmla.__file__).resolve() != root / "gmla/__init__.py":
        raise RuntimeError("gmla must be normally imported from --installed-root; no sys.path/provider injection is allowed")
    distributions = list(root.glob("gated_mla_ascend_candidate-*.dist-info"))
    if len(distributions) != 1 or not (distributions[0] / "RECORD").is_file():
        raise RuntimeError("Normal installed distribution metadata is required")
    direct = distributions[0] / "direct_url.json"
    if direct.exists() and json.loads(direct.read_text(encoding="utf-8")).get("dir_info", {}).get("editable"):
        raise RuntimeError("Editable installs are excluded")
    inventory = gmla.package_sources()
    if inventory != json.loads(inventory_file.read_text(encoding="utf-8"))["sources"]:
        raise RuntimeError("Installed source bytes differ from --inventory")
    return inventory


def tensor_record(value):
    """Called on the original CPU fixture only, outside timing."""
    import torch

    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "stride": list(value.stride()),
        "storage_offset": value.storage_offset(),
        "sha256": hashlib.sha256(value.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest(),
    }


def write_table(report, path):
    lines = ["# Decode host diagnostic", "", f"Status: `{report['status']}`", ""]
    formal = report.get("uninstrumented_timing", {})
    if formal:
        lines += ["The following measurements contain no cProfile instrumentation.", "", "| Measurement | Median (us) |", "|---|---:|"]
        for key in ("event_interval", "wall_with_end_event_sync"):
            lines.append(f"| {key} | {formal[key]['median']:.3f} |")
    diagnostic = report.get("host_diagnostic")
    if diagnostic:
        lines += [
            "",
            "Warm cProfile attribution only; not an operator benchmark or removable-validator-cost estimate.",
            "",
            "| Function | Calls / forward | Self us / forward | Inclusive us / forward |",
            "|---|---:|---:|---:|",
        ]
        for row in diagnostic["top_inclusive"][:15]:
            name = f"{Path(row['file']).name}:{row['line']} {row['function']}".replace("|", "\\|")
            lines.append(
                f"| {name} | {row['calls_per_public_forward']:.2f} | "
                f"{row['self_us_per_public_forward']:.3f} | {row['inclusive_us_per_public_forward']:.3f} |"
            )
        lines += ["", diagnostic["interpretation"]]
    lines += ["", "Numerical/read-only checks are outside both timing sections. CPU-reference mode is not NPU validation.", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--installed-root", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--protocol-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("cpu-reference", "diagnose"), default="diagnose")
    parser.add_argument("--target", choices=("pto", "ascendc"), default="pto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--profile-calls", type=int, default=20)
    parser.add_argument("--cpu-threads", type=int, default=4)
    args = parser.parse_args()
    if min(args.warmup, args.repeats, args.profile_calls, args.cpu_threads) < 1:
        parser.error("All counts must be positive")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Output directory must be new or empty")
    args.output.mkdir(parents=True, exist_ok=True)
    report = {
        "status": "running",
        "phase": "setup",
        "mode": args.mode,
        "date": datetime.now(timezone.utc).isoformat(),
        "argv": [sys.executable, *sys.argv],
        "command": shlex.join([sys.executable, *sys.argv]),
        "tool_sha256": sha(__file__),
        "shape": {"B": 1, "Tq": 1, "H": 96, "Tk": 4096, "D": 128, "E": 64, "R": 512, "V": 128},
        "dtype": "float16",
        "visible": 17,
        "target": args.target,
        "seed": args.seed,
        "metadata": [[1, 4096, 16, 0, 0, 0, 0, 0]],
        "gate_case": "head_specific",
        "budget": {"rtol": 0.01, "atol": 0.002, "max_normalized": 0.005},
        "entry": "normally installed gmla.build_runtime(target=...).forward",
        "scope": "Seven-input candidate only; no external projections/norm/cache append/W_O or baseline ratio.",
        "warmup": args.warmup,
        "repeats": args.repeats,
        "profile_calls": args.profile_calls,
        "cache_policy": "Reused finite fixture inputs and weights; no L2 flush; no validation bypass.",
        "checks": [],
        "NPU_execution": False,
    }

    def save():
        (args.output / "host_decode_diagnostic.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        write_table(report, args.output / "SUMMARY.md")

    try:
        save()
        report["inventory_sha256"] = sha(args.inventory)
        import torch
        import gmla
        from gmla import reference as golden

        torch.set_grad_enabled(False)
        torch.set_num_threads(args.cpu_threads)
        report["environment"] = {
            "python": sys.version,
            "executable": sys.executable,
            "platform": platform.platform(),
            "torch": str(torch.__version__),
            "torch_file": torch.__file__,
            "cpu_threads": torch.get_num_threads(),
            "gmla_file": gmla.__file__,
            "matmul_precision": torch.get_float32_matmul_precision(),
            "env": {
                key: os.environ.get(key) for key in ("TL_PLATFORM", "TL_RUN_MODE", "SOC_VERSION", "ASCEND_HOME_PATH", "TILELANG_CACHE_DIR")
            },
        }
        report["package_sources"] = verify_install(gmla, args.installed_root, args.inventory)
        report["protocol"] = verify_protocol(args.protocol_root)
        if (golden.FP16_RTOL, golden.FP16_ATOL, golden.FP16_MAX_NORMALIZED_ERROR) != (0.01, 0.002, 0.005):
            raise RuntimeError("Frozen numerical budget changed")
        expected_scale = 1 / math.sqrt(192)
        if expected_scale != gmla.DEFAULT_SCALE:
            raise RuntimeError("Frozen default scale changed")
        report["scale"] = gmla.DEFAULT_SCALE
        fixture = load_protocol("decode_fixture", args.protocol_root)
        protocol = load_protocol("decode_protocol", args.protocol_root)
        timing = load_protocol("timing_protocol", args.protocol_root)
        shape = fixture.CASES["model_4k"]
        if (shape["B"], shape["H"], shape["Tk"]) != (1, 96, 4096):
            raise RuntimeError("Frozen case domain changed")
        cpu = fixture.make_inputs(shape, args.seed, torch.float16, "head_specific")
        cpu_meta = gmla.prepare_metadata([[1, 4096, 16]], batch=1, query_capacity=1, cache_capacity=4096)
        gmla.validate_runtime_metadata_cpu(cpu_meta, B=1, Tq=1, Tk=4096)
        report["input_tensors"] = {name: tensor_record(value) for name, value in cpu.items()}
        report["phase"] = "cpu_reference"
        save()
        refs = fixture.goldens(cpu, cpu_meta, golden)
        report["cpu_references"] = {name: tensor_record(value) for name, value in refs.items()}
        report["cpu_reference_agreement"] = {"expanded_vs_absorbed": "passed", "rtol": 2e-12, "atol": 2e-12}
        if args.mode == "cpu-reference":
            if "tilelang" in sys.modules or "torch_npu" in sys.modules or getattr(getattr(torch, "npu", None), "_initialized", False):
                raise RuntimeError("CPU-only mode loaded compiler/device runtime")
            report.update(status="cpu_reference_passed", phase="complete", device_initialized=False)
            return 0
        report["phase"] = "runtime_initialization"
        save()
        for key, expected in (("TL_PLATFORM", "A2"), ("TL_RUN_MODE", "npu")):
            if os.environ.get(key) != expected:
                raise RuntimeError(f"Owner must explicitly configure {key}={expected}")
        if getattr(getattr(torch, "npu", None), "_initialized", False):
            raise RuntimeError("Start clean; load pinned TileLang before NPU initialization")
        runtime = gmla.build_runtime(target=args.target)
        _ = runtime.kernel_bundle  # Public preload access, same ordering as source_decode_benchmark.
        import torch_npu
        import tilelang

        if not torch.npu.is_available():
            raise RuntimeError("A real NPU is required; no skip or CPU fallback")
        device_index = torch.npu.current_device()
        report["environment"].update(
            torch_npu=str(torch_npu.__version__),
            torch_npu_file=torch_npu.__file__,
            tilelang=str(tilelang.__version__),
            tilelang_file=tilelang.__file__,
            device_index=device_index,
            device_name=torch.npu.get_device_name(device_index),
            device_properties=str(torch.npu.get_device_properties(device_index)),
        )
        device = {name: value.to(f"npu:{device_index}") for name, value in cpu.items()}
        metadata = cpu_meta.to(f"npu:{device_index}")

        def candidate():
            return runtime.forward(**device, metadata=metadata)

        def check_output(actual, label):
            entry = {"phase": label, "numerical": {}, "readonly": "pending"}
            report["checks"].append(entry)
            save()
            for name in ("expanded", "absorbed"):
                entry["numerical"][name] = protocol.qualify(actual, refs[name], torch.float16, golden)
                save()
                if not entry["numerical"][name]["qualified"]:
                    raise AssertionError(f"{label}: {name} FP64 budget failed")
            for name, value in device.items():
                torch.testing.assert_close(value.cpu(), cpu[name], rtol=0, atol=0)
            torch.testing.assert_close(metadata.cpu(), cpu_meta, rtol=0, atol=0)
            entry["readonly"] = "passed"
            save()

        report.update(phase="entry_qualification", NPU_execution=True)
        actual, stages, audit = runtime.forward(**device, metadata=metadata, return_intermediates=True)
        torch.npu.synchronize()
        report["kernel_report"] = gmla.kernel_report(audit)
        generated = args.output / "generated_sources"
        generated.mkdir()
        for name, kernel in audit["kernels"].items():
            (generated / (name + ".cpp")).write_text(kernel.get_kernel_source(), encoding="utf-8")
        report["stage_accuracy"] = {
            "query": golden.assert_fp16_close(stages["q_latent_hi"], refs["q_latent"]),
            "latent": golden.assert_fp16_close(stages["latent_out"], refs["latent_out"]),
        }
        check_output(actual, "entry")
        report["phase"] = "uninstrumented_timing"
        save()
        report["uninstrumented_timing"], actual = timing.eager_timing(candidate, args.warmup, args.repeats)
        save()  # Keep valid formal measurements even if later diagnostic work fails.
        check_output(actual, "post_uninstrumented")
        for _ in range(args.warmup):
            actual = candidate()
        torch.npu.synchronize()
        # No D2H/read-only scan between this rewarm and the profiled calls.
        report["phase"] = "cprofile"
        save()
        report["host_diagnostic"], actual = run_host_profile(
            candidate, torch.npu.synchronize, calls=args.profile_calls, output_dir=args.output
        )
        save()
        check_output(actual, "post_profile")
        report.update(status="passed", phase="complete")
        return 0
    except Exception as error:
        # Abort on every error, including numerical failures. No recovery NPU calls.
        report.update(status="failed", error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
        return 1
    finally:
        save()


if __name__ == "__main__":
    raise SystemExit(main())
