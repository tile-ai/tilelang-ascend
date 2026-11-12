"""Private fixed-shape prefill experiment; production installation is read-only.

Core experiments share rounded LP operands and do not represent BF16 production
HiLo. Full-chain experiments preserve the production Q/value providers and HiLo.
Use core/validate under msprof to select the variant's first TileLang launch.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shlex
import sys
import traceback


BASE = Path("/mnt/workspace/Gated_MLA_Workspace/evidence/20261112_benchmark_alignment")
NAMES = ("q_content", "q_aux", "latent_kv", "key_aux", "w_k", "w_v", "gate_logits")
HELPER_SHA = {
    "aligned_execution.py": "feb1efc0755226a967c8abf647adff4feb6cd004a698db3f6cee528f2e351f57",
    "paired_timing.py": "13f6b117b6df7d63063111b551cd0d70fc7c95211eff6212acab7c0a39ad4e1d",
}
BUDGET = {"rtol": 0.01, "atol": 0.002, "max_normalized": 0.005}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_file(name, path):
    path = Path(path).resolve()
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    if Path(module.__file__).resolve() != path:
        raise RuntimeError("module loaded from an unexpected file")
    return module


def exact_helper(root, filename):
    path = root / filename
    if sha(path) != HELPER_SHA[filename]:
        raise RuntimeError("frozen alignment helper differs: " + filename)
    return load_file("_gmla_prefill_experiment_" + filename[:-3] + "_" + sha(path)[:16], path)


def identity(torch, tensor):
    raw = tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    return {"shape": list(tensor.shape), "dtype": str(tensor.dtype), "sha256": hashlib.sha256(raw).hexdigest()}


def dump_kernels(kernels, prefix, output):
    result = {}
    for name, kernel in kernels.items():
        source = kernel.get_kernel_source()
        filename = prefix + "_" + name + ".cpp"
        (output / filename).write_text(source, encoding="utf-8")
        result[name] = {
            "target": str(kernel.target),
            "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
            "file": filename,
        }
    return result


def build_core_methods(torch, runtime, variant, device, metadata, cpu, cpu_metadata, api, execution):
    shared = execution.prepare_shared_core(torch, runtime, device, metadata, cpu, cpu_metadata, api=api)
    operands = shared.prepared_inputs

    def candidate_debug():
        output, kernels = variant.cube_prefill(
            operands["ql"],
            operands["qa"],
            operands["cache"],
            operands["auxiliary_cache"],
            metadata,
            block_m=32,
            block_n=128,
            target=runtime.target,
            max_blocks=20,
            scale=1 / math.sqrt(192),
            ql_residual=None,
            return_kernels=True,
        )
        if output.dtype != torch.float32 or tuple(output.shape) != (1, 128, 96, 512):
            raise AssertionError("variant shared-core output violates shape/dtype")
        return output.view(1, 128, 96, 512), kernels, None

    def previous_debug():
        # The first variant launch has already happened when this is called.
        output, kernels = shared._candidate_with_kernels()
        return output, kernels, None

    methods = {
        "candidate": lambda: candidate_debug()[0],
        "previous": shared.methods["candidate"],
        "official_fia_core": shared.methods["official_fia_core"],
    }
    return {
        "methods": methods,
        "debug": {"candidate": candidate_debug, "previous": previous_debug},
        "references": {"core": shared.golden},
        "readonly": shared.readonly,
        "description": shared.describe(),
        "prepared_inputs": shared.prepared_inputs,
    }


def build_full_methods(torch, runtime, variant, device, metadata, cpu, cpu_metadata, api, execution, fixture, protocol, golden):
    previous = execution.prepare_candidate(runtime, device, metadata)
    candidate = replace(previous, bundle=replace(previous.bundle, prefill_module=variant))
    table_cpu = torch.zeros((1, 1), dtype=torch.int32)
    mask_cpu = torch.triu(torch.ones((2048, 2048), dtype=torch.bool), diagonal=1)
    table, mask = table_cpu.to(device["q_content"].device), mask_cpu.to(device["q_content"].device)
    source = protocol.OfficialFullChains(device, mask, table, api.npu_fused_infer_attention_score_v2, 1 / math.sqrt(192))

    def debug(binding):
        output, stages, audit = binding.bundle._forward_validated(*binding.tensors, binding.contract, return_intermediates=True)
        return output, audit["kernels"], stages

    def readonly():
        for name in NAMES:
            execution._assert_bytes(torch, device[name], cpu[name], name)
        execution._assert_bytes(torch, metadata, cpu_metadata, "metadata")
        execution._assert_bytes(torch, table, table_cpu, "historical_block_table")
        execution._assert_bytes(torch, mask, mask_cpu, "historical_causal_mask")

    return {
        "methods": {"candidate": candidate, "previous": previous, "historical_fp32": source.mla_paged},
        "debug": {"candidate": lambda: debug(candidate), "previous": lambda: debug(previous)},
        "references": fixture.references(cpu, cpu_metadata, golden, 1 / math.sqrt(192)),
        "readonly": readonly,
        "description": {
            "candidate": candidate.describe(),
            "previous": previous.describe(),
            "historical_fp32": "unchanged absorbed FIA-v2 composition with dynamic FP32 Torch projections/gate; not source-prefill",
            "production_BF16_HiLo_preserved": True,
            "first_launch": "production Q projection; use core/validate for first-launch variant profiling",
        },
        "prepared_inputs": {"historical_block_table": table, "historical_causal_mask": mask},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant-file", type=Path, required=True)
    parser.add_argument("--scope", choices=("core", "full-chain"), required=True)
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    parser.add_argument("--target", choices=("pto", "ascendc"), default="pto")
    parser.add_argument("--mode", choices=("validate", "benchmark"), default="validate")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--installed-root", type=Path, default=BASE / "installed/site")
    parser.add_argument(
        "--inventory",
        type=Path,
        default=Path("/mnt/workspace/Gated_MLA_Workspace/evidence/20261112_perf_followup/tools_v2/pinned_inventory.json"),
    )
    parser.add_argument("--alignment-tools", type=Path, default=BASE / "tools_v5")
    parser.add_argument(
        "--protocol-root",
        type=Path,
        default=Path("/mnt/workspace/gated-mla-l3-pr/examples_experiment/gated_mla/package/benchmark/protocol"),
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 17])
    parser.add_argument("--rounds", type=int, default=6)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--graph-repeats", type=int, default=5)
    parser.add_argument("--graph-batch", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--order-seed", type=int, default=1700)
    parser.add_argument("--skip-unqualified-timing", action="store_true")
    args = parser.parse_args()
    if len(args.seeds) < 3 or len(set(args.seeds)) != len(args.seeds):
        parser.error("at least three distinct qualification seeds are required")
    if args.rounds < 6 or args.rounds % 6 or min(args.repeats, args.graph_repeats, args.graph_batch, args.warmup) < 1:
        parser.error("rounds must be a positive multiple of6; other counts must be positive")
    if not args.variant_file.is_file() or args.variant_file.suffix != ".py":
        parser.error("variant-file must be an existing Python source")
    if args.output.exists():
        parser.error("output must be a new directory")
    args.output.mkdir(parents=True)
    destination = args.output / "prefill_experiment.json"
    report = {
        "schema": "gmla-private-prefill-experiment-v1",
        "status": "preparing",
        "date_UTC": datetime.now(timezone.utc).isoformat(),
        "scope": args.scope,
        "dtype": args.dtype,
        "target": args.target,
        "mode": args.mode,
        "shape": {"B": 1, "Tq": 128, "Tk": 128, "H": 96, "D": 128, "E": 64, "R": 512, "V": 128},
        "qualification_seeds": args.seeds,
        "timed_seed": args.seeds[0],
        "budget": BUDGET,
        "driver_sha256": sha(__file__),
        "variant_sha256": sha(args.variant_file),
        "first_TileLang_launch": "variant attention" if args.scope == "core" else "production query projection",
        "accuracy": [],
        "measurement": None,
        "command": shlex.join([sys.executable, *sys.argv]),
        "core_BF16_is_production_HiLo": False,
    }

    def save():
        temporary = destination.with_suffix(".tmp")
        temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        temporary.replace(destination)

    snapshot = None
    initial_snapshot = None
    try:
        for key, value in (("TL_PLATFORM", "A2"), ("TL_RUN_MODE", "npu")):
            os.environ.setdefault(key, value)
            if os.environ[key] != value:
                raise RuntimeError("this910B3 experiment requires " + key + "=" + value)
        import torch
        import gmla
        from gmla import reference as golden

        torch.set_grad_enabled(False)
        torch.set_num_threads(4)
        installed = args.installed_root.resolve()
        if Path(gmla.__file__).resolve() != installed / "gmla/__init__.py":
            raise RuntimeError("candidate must be imported normally from the requested installation")
        distributions = list(installed.glob("gated_mla_ascend_candidate-*.dist-info"))
        if len(distributions) != 1 or not (distributions[0] / "RECORD").is_file():
            raise RuntimeError("normal wheel/site metadata is required")
        direct = distributions[0] / "direct_url.json"
        if direct.exists() and json.loads(direct.read_text()).get("dir_info", {}).get("editable"):
            raise RuntimeError("editable candidate installation is excluded")
        if args.variant_file.resolve().is_relative_to(installed / "gmla"):
            raise RuntimeError("private variant must remain outside the production installation")
        inventory = gmla.package_sources()
        if inventory != json.loads(args.inventory.read_text())["sources"]:
            raise RuntimeError("normally installed baseline source inventory differs")
        if (golden.FP16_RTOL, golden.FP16_ATOL, golden.FP16_MAX_NORMALIZED_ERROR) != (0.01, 0.002, 0.005):
            raise AssertionError("original numerical budget changed")
        manifest = json.loads((args.protocol_root / "protocol_manifest.json").read_text())
        for name, record in manifest["files"].items():
            if sha(args.protocol_root / name) != record["extracted_sha256"]:
                raise RuntimeError("frozen protocol changed: " + name)
        execution = exact_helper(args.alignment_tools.resolve(), "aligned_execution.py")
        timing = exact_helper(args.alignment_tools.resolve(), "paired_timing.py")
        fixture = load_file("_gmla_prefill_experiment_fixture", args.protocol_root / "prefill_fixture.py")
        protocol = load_file("_gmla_prefill_experiment_protocol", args.protocol_root / "prefill_protocol.py")

        def snapshot():
            return {
                "baseline_sources": gmla.package_sources(),
                "driver_sha256": sha(__file__),
                "variant_sha256": sha(args.variant_file),
                "alignment_execution_sha256": sha(execution.__file__),
                "actual_timing_engine_sha256": sha(timing.__file__),
                "protocol_manifest_sha256": sha(args.protocol_root / "protocol_manifest.json"),
                "protocol_sources": {name: sha(args.protocol_root / name) for name in manifest["files"]},
            }

        initial_snapshot = snapshot()
        report["source_identity"] = initial_snapshot
        report["timing_engine_identity"] = {
            "actual_source_sha256": sha(timing.__file__),
            "expected_source_sha256": HELPER_SHA["paired_timing.py"],
            "same_exact_file": Path(timing.__file__).resolve() == (args.alignment_tools / "paired_timing.py").resolve(),
        }
        if not report["timing_engine_identity"]["same_exact_file"]:
            raise AssertionError("actual timing helper path differs")
        runtime = gmla.build_runtime(target=args.target)
        if getattr(getattr(torch, "npu", None), "_initialized", False):
            raise RuntimeError("use a fresh process and preload TileLang before NPU allocation")
        _ = runtime.kernel_bundle  # Import compiler and fixed production providers first.
        variant = load_file("gmla.kernels._prefill_variant_" + sha(args.variant_file)[:16], args.variant_file)
        if not callable(getattr(variant, "cube_prefill", None)) or not callable(getattr(variant, "build_cube_attention", None)):
            raise TypeError("variant must preserve the prefill module interface")
        import torch_npu
        import tilelang

        if not torch.npu.is_available():
            raise RuntimeError("real NPU required; there is no CPU timing fallback")
        torch.npu.set_device(0)
        torch.set_float32_matmul_precision("highest")
        torch.npu.matmul.allow_hf32 = False
        report["environment"] = {
            "python": sys.version,
            "torch": str(torch.__version__),
            "torch_npu": str(torch_npu.__version__),
            "tilelang": str(tilelang.__version__),
            "device": torch.npu.get_device_name(0),
            "properties": str(torch.npu.get_device_properties(0)),
            "matmul_precision": torch.get_float32_matmul_precision(),
            "allow_hf32": torch.npu.matmul.allow_hf32,
        }
        dtype = getattr(torch, args.dtype)
        first = None
        qualified = None

        def qualify(name, output, references):
            width = 512 if args.scope == "core" else 128
            expected_dtype = torch.float32 if args.scope == "core" and name != "official_fia_core" else dtype
            if not isinstance(output, torch.Tensor) or tuple(output.shape) != (1, 128, 96, width) or output.dtype != expected_dtype:
                raise AssertionError(name + " output shape/dtype contract failed")
            checks = {key: protocol.metrics(output, value) for key, value in references.items() if key in ("core", "expanded", "absorbed")}
            return {"qualified": bool(all(row["passed"] for row in checks.values())), "checks": checks, "output_dtype": str(output.dtype)}

        for seed in args.seeds:
            cpu = fixture.make_inputs(report["shape"], dtype, seed, "head_specific")
            cpu_metadata = gmla.prepare_metadata([[128, 128, 0]], batch=1, query_capacity=128, cache_capacity=128)
            device = {name: value.to("npu") for name, value in cpu.items()}
            metadata = cpu_metadata.to("npu")
            if args.scope == "core":
                context = build_core_methods(torch, runtime, variant, device, metadata, cpu, cpu_metadata, torch_npu, execution)
            else:
                context = build_full_methods(
                    torch, runtime, variant, device, metadata, cpu, cpu_metadata, torch_npu, execution, fixture, protocol, golden
                )
            methods, references, readonly = context["methods"], context["references"], context["readonly"]
            if qualified is None:
                qualified = dict.fromkeys(methods, True)
            record = {"seed": seed, "methods": {}}
            report["accuracy"].append(record)
            # Dict insertion order deliberately makes candidate the first launch.
            for name, function in methods.items():
                if name in context["debug"]:
                    output, kernels, stages = context["debug"][name]()
                    if first is None:
                        report[name + "_kernels"] = dump_kernels(kernels, name, args.output)
                    if stages is not None:
                        logical_ql = stages["q_latent_hi"].cpu().double()
                        if stages["q_latent_lo"] is not None:
                            logical_ql += stages["q_latent_lo"].cpu().double()
                        record[name + "_stages"] = {
                            "query": golden.assert_fp16_close(logical_ql, references["q_latent"]),
                            "latent": golden.assert_fp16_close(stages["latent_out"], references["latent_out"]),
                        }
                else:
                    output = function()
                torch.npu.synchronize()
                check = qualify(name, output, references)
                record["methods"][name] = check
                qualified[name] &= check["qualified"]
                if name in ("candidate", "previous") and not check["qualified"]:
                    raise AssertionError(name + " failed original FP64 budget at seed " + str(seed))
                readonly()
                save()
            if first is None:
                first = context
                report["method_description"] = context["description"]
                report["timed_inputs"] = {name: identity(torch, value) for name, value in cpu.items()}
                report["timed_metadata"] = identity(torch, cpu_metadata)
                report["timed_prepared_inputs"] = {name: identity(torch, value) for name, value in context["prepared_inputs"].items()}
        report["initial_qualification"] = qualified
        if args.mode == "validate":
            report["status"] = "NPU_validation_passed_with_explicit_baseline_qualification"
        else:
            methods, references, readonly = first["methods"], first["references"], first["readonly"]

            def postcheck(name, output, boundary):
                result = qualify(name, output, references)
                readonly()
                return result

            def progress(partial):
                report["measurement"] = partial
                save()

            report["measurement"] = timing.run_measurements(
                methods,
                qualified=qualified,
                check=postcheck,
                rounds=args.rounds,
                repeats=args.repeats,
                warmup=args.warmup,
                graph_batch=args.graph_batch,
                graph_repeats=args.graph_repeats,
                order_seed=args.order_seed,
                time_unqualified=not args.skip_unqualified_timing,
                graph_error_policy=protocol.graph_error_policy,
                on_progress=progress,
            )
            report["status"] = "balanced_benchmark_completed"
    except Exception as error:
        if hasattr(error, "partial_report"):
            report["measurement"] = error.partial_report
        report.update(status="failed", error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        if snapshot is not None and initial_snapshot is not None:
            try:
                if snapshot() != initial_snapshot:
                    report.update(status="failed", source_integrity_error="experiment sources changed during execution")
            except Exception as error:
                report.update(status="failed", source_integrity_error=repr(error))
        save()
        print("GMLA_PREFILL_EXPERIMENT " + report["status"], flush=True)
        if report.get("source_integrity_error"):
            raise RuntimeError(report["source_integrity_error"])


if __name__ == "__main__":
    main()
