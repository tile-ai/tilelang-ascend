"""Same-process installed candidate, historical FP32 prefill composition and source prefill."""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shlex
import sys
import traceback


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_protocol(name, root):
    spec = importlib.util.spec_from_file_location("gmla_source_benchmark_" + name, root / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--installed-root", type=Path, required=True)
    parser.add_argument("--inventory", "--installed-inventory", dest="inventory", type=Path, required=True)
    parser.add_argument("--protocol-root", type=Path, required=True)
    parser.add_argument("--legacy-route", choices=("expanded_standard", "mla_paged"), default="expanded_standard")
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    parser.add_argument("--target", choices=("pto", "ascendc"), default="pto")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 17])
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--graph-batch", type=int, default=5)
    parser.add_argument("--time-unqualified", action="store_true", help="Diagnostic timing only; never permits a qualified ratio")
    parser.add_argument("--order", choices=("candidate-first", "baseline-first"), default="candidate-first")
    parser.add_argument("--mode", choices=("cpu-reference", "validate", "benchmark", "profile"), default="benchmark")
    parser.add_argument("--gate-module", help="Isolated module, not the complete vllm/vllm_ascend package")
    parser.add_argument("--gate-callable", default="flash_attention_gate")
    parser.add_argument("--gate-label", help="Explicit provenance label for the injected gate")
    parser.add_argument("--profile-method", choices=("candidate", "historical_fp32", "source_prefill"), default="source_prefill")
    parser.add_argument("--validated-report", type=Path, help="Required by profile: prior report with matching source/runtime identity")
    args = parser.parse_args()
    if len(set(args.seeds)) < 3:
        parser.error("at least three distinct seeds are required before comparison")
    if min(args.warmup, args.repeats, args.graph_batch) < 1:
        parser.error("timing counts must be positive")
    if args.mode == "profile" and args.validated_report is None:
        parser.error("profile requires --validated-report")
    if args.gate_module and (not args.gate_label or args.gate_module.split(".")[0] in ("vllm", "vllm_ascend")):
        parser.error("inject an isolated gate module with an explicit label; do not import a complete vLLM stack here")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("output must be new or empty")
    args.output.mkdir(parents=True, exist_ok=True)
    for name, value in (("TL_PLATFORM", "A2"), ("TL_RUN_MODE", "npu")):
        os.environ.setdefault(name, value)
        if os.environ[name] != value:
            parser.error(f"This910B3 protocol requires {name}={value}")

    import torch

    torch.set_grad_enabled(False)  # This dedicated benchmark process is inference-only.
    import gmla
    from gmla import reference as golden
    from adapter_prefill import VllmSourcePrefill

    installed = args.installed_root.resolve()
    if Path(gmla.__file__).resolve() != installed / "gmla/__init__.py":
        raise RuntimeError("gmla must be normally imported from requested installed root; this CLI never injects candidate paths")
    distributions = list(installed.glob("gated_mla_ascend_candidate-*.dist-info"))
    if len(distributions) != 1 or not (distributions[0] / "RECORD").is_file():
        raise RuntimeError("normal installed distribution metadata required")
    direct = distributions[0] / "direct_url.json"
    if direct.exists() and json.loads(direct.read_text()).get("dir_info", {}).get("editable"):
        raise RuntimeError("editable candidate installs are excluded")
    inventory = gmla.package_sources()
    if inventory != json.loads(args.inventory.read_text())["sources"]:
        raise RuntimeError("installed candidate differs from requested inventory")
    if (golden.FP16_RTOL, golden.FP16_ATOL, golden.FP16_MAX_NORMALIZED_ERROR) != (0.01, 0.002, 0.005):
        raise AssertionError("frozen accuracy budget changed")
    protocol_manifest = json.loads((args.protocol_root / "protocol_manifest.json").read_text())
    for name, record in protocol_manifest["files"].items():
        if sha(args.protocol_root / name) != record["extracted_sha256"]:
            raise RuntimeError(f"frozen protocol changed: {name}")
    fixture = load_protocol("prefill_fixture", args.protocol_root)
    protocol = load_protocol("prefill_protocol", args.protocol_root)
    error_policy = protocol
    timing = load_protocol("timing_protocol", args.protocol_root)
    expected_scale = 1 / math.sqrt(192)
    if expected_scale != gmla.DEFAULT_SCALE:
        raise AssertionError("installed default scale differs from the frozen seven-input contract")
    dtype = getattr(torch, args.dtype)
    torch.set_num_threads(4)
    report = {
        "status": "running",
        "date": datetime.now(timezone.utc).isoformat(),
        "case": "prefill128",
        "legacy_route": args.legacy_route,
        "dtype": args.dtype,
        "target": args.target,
        "seeds": args.seeds,
        "gate_case": "head_specific",
        "scale": gmla.DEFAULT_SCALE,
        "shape": {"B": 1, "Tq": 128, "H": 96, "Tk": 128, "D": 128, "E": 64, "R": 512, "V": 128},
        "scope": "same original seven inputs; excludes external input projections/norm/cache append/W_O",
        "budget": {"rtol": 0.01, "atol": 0.002, "max_normalized": 0.005},
        "package_sources": inventory,
        "protocol_manifest_sha256": sha(args.protocol_root / "protocol_manifest.json"),
        "inventory_sha256": sha(args.inventory),
        "harness_sha256": sha(__file__),
        "adapter_sha256": sha(Path(__file__).with_name("adapter_prefill.py")),
        "linear_source_sha256": sha(Path(__file__).with_name("source_unquantized_gemm.py")),
        "gate_helper_sha256": sha(Path(__file__).with_name("adapter_decode.py")),
        "timing": {
            "warmup": args.warmup,
            "repeats": args.repeats,
            "graph_batch": args.graph_batch,
            "graph_warmup_replays": 5,
            "order": args.order,
            "cache": "reused inputs/weights, no L2 flush",
            "unit": "microseconds",
        },
        "candidate_entry": "normally installed gmla.build_runtime(target=...).forward",
        "accuracy": [],
        "methods": {},
        "ratios": {},
        "command": shlex.join([sys.executable, *sys.argv]),
    }
    destination = args.output / "source_prefill_benchmark.json"

    def save():
        destination.write_text(json.dumps(report, indent=2) + "\n")

    def read_only(device, cpu, metadata, cpu_meta, new_base):
        for name, value in device.items():
            torch.testing.assert_close(value.cpu(), cpu[name], rtol=0, atol=0)
        torch.testing.assert_close(metadata.cpu(), cpu_meta, rtol=0, atol=0)
        expected_weight = torch.cat((cpu["w_k"], cpu["w_v"]), dim=1).view(96 * 256, 512)
        torch.testing.assert_close(new_base.constants["packed_w_kv"].cpu(), expected_weight, rtol=0, atol=0)
        expected_mask = torch.triu(torch.ones(2048, 2048), diagonal=1).to(torch.int8)
        torch.testing.assert_close(new_base.constants["causal_mask"].cpu(), expected_mask, rtol=0, atol=0)

    def qualify(actual, refs):
        if not isinstance(actual, torch.Tensor) or tuple(actual.shape) != (1, 128, 96, 128) or actual.dtype != dtype:
            raise AssertionError("prefill output shape/dtype contract changed")
        expanded = protocol.metrics(actual, refs["expanded"])
        absorbed = protocol.metrics(actual, refs["absorbed"])
        return {"qualified": expanded["passed"] and absorbed["passed"], "expanded": expanded, "absorbed": absorbed}

    try:
        save()
        prepared = []
        for seed in args.seeds:
            shape = dict(B=1, H=96, Tq=128, Tk=128)
            cpu = fixture.make_inputs(shape, dtype, seed, "head_specific")
            cpu_meta = gmla.prepare_metadata([[128, 128, 0]], batch=1, query_capacity=128, cache_capacity=128)
            refs = fixture.references(cpu, cpu_meta, golden, gmla.DEFAULT_SCALE)
            host = gmla.metadata_for_reference(cpu_meta)
            prepared.append((seed, cpu, cpu_meta, host, refs))
        if args.mode == "cpu-reference":
            if "tilelang" in sys.modules or getattr(getattr(torch, "npu", None), "_initialized", False):
                raise AssertionError("CPU mode touched compiler/device initialization")
            report.update(status="cpu_reference_passed", NPU_execution=False)
            return
        runtime = gmla.build_runtime(target=args.target)
        if getattr(getattr(torch, "npu", None), "_initialized", False):
            raise RuntimeError("start a clean process and preload pinned TileLang before any NPU operation")
        _ = runtime.kernel_bundle
        import torch_npu
        import tilelang

        if not torch.npu.is_available():
            raise RuntimeError("real NPU required; no silent skip")
        torch.npu.set_device(0)
        gate_impl, gate_info = None, {"mode": "torch_rounding", "name": "torch_lp_rounding_expression_not_fused"}
        if args.gate_module:
            module = importlib.import_module(args.gate_module)
            gate_impl = getattr(module, args.gate_callable)
            if not callable(gate_impl):
                raise TypeError("gate callable is unavailable")
            gate_info = {
                "mode": "injected_callable",
                "name": args.gate_label,
                "module": args.gate_module,
                "callable": args.gate_callable,
                "file": module.__file__,
                "source_sha256": sha(module.__file__),
            }
        report["gate"] = gate_info
        report["environment"] = {
            "python": sys.version,
            "torch": str(torch.__version__),
            "torch_npu": str(torch_npu.__version__),
            "torch_npu_file": torch_npu.__file__,
            "tilelang": tilelang.__version__,
            "tilelang_file": tilelang.__file__,
            "device": torch.npu.get_device_name(0),
            "properties": str(torch.npu.get_device_properties(0)),
            "matmul_precision": torch.get_float32_matmul_precision(),
            "matmul_allow_hf32": getattr(getattr(torch.npu, "matmul", None), "allow_hf32", "unavailable"),
        }
        identity = {
            key: report[key]
            for key in (
                "case",
                "legacy_route",
                "linear_source_sha256",
                "gate_helper_sha256",
                "dtype",
                "target",
                "seeds",
                "package_sources",
                "adapter_sha256",
                "harness_sha256",
                "protocol_manifest_sha256",
                "gate",
                "environment",
            )
        }
        report["identity"] = identity
        page_table = torch.zeros((1, 1), dtype=torch.int32).npu()
        legacy_mask = torch.triu(torch.ones((2048, 2048), dtype=torch.bool), diagonal=1).npu()

        def make_methods(record):
            seed, cpu, cpu_meta, host, refs = record
            device = {name: value.npu() for name, value in cpu.items()}
            device_meta = cpu_meta.npu()
            new_base = VllmSourcePrefill(device, host, api=torch_npu, gate_impl=gate_impl, gate_name=args.gate_label)
            old_composition = protocol.OfficialFullChains(
                device, legacy_mask, page_table, torch_npu.npu_fused_infer_attention_score_v2, gmla.DEFAULT_SCALE
            )
            old_base = getattr(old_composition, args.legacy_route)

            def candidate():
                return runtime.forward(**device, metadata=device_meta)

            return {"candidate": candidate, "historical_fp32": old_base, "source_prefill": new_base}, device, device_meta, new_base

        if args.mode == "profile":
            previous = json.loads(args.validated_report.read_text())
            if previous.get("identity") != identity:
                raise ValueError("profile identity differs from validated report; do not reuse another runtime/source/gate qualification")
            qualified = previous.get("initial_qualification", {}).get(args.profile_method, False)
            if not qualified and (args.profile_method == "candidate" or not args.time_unqualified):
                raise ValueError("profile method lacks matching FP64 qualification; unqualified baseline requires explicit diagnostic flag")
            methods, device, device_meta, new_base = make_methods(prepared[0])
            actual = methods[args.profile_method]()
            torch.npu.synchronize()
            check = qualify(actual, prepared[0][4])
            if qualified and not check["qualified"]:
                raise AssertionError("profile call no longer passes original FP64")
            read_only(device, prepared[0][1], device_meta, prepared[0][2], new_base)
            report.update(
                status="profile_single_call_completed",
                profile_method=args.profile_method,
                profile_accuracy=check,
                profile_qualified=qualified and check["qualified"],
                source_adapter=new_base.describe(),
                performance="one selected full callable only; obtain real kernel Task metrics from msprof, not this Python entry",
            )
            return

        qualification = dict(candidate=True, historical_fp32=True, source_prefill=True)
        first_data = None
        for record in prepared:
            seed, cpu, cpu_meta, host, refs = record
            methods, device, device_meta, new_base = make_methods(record)
            row = {"seed": seed, "methods": {}}
            report["accuracy"].append(row)
            report["source_adapter"] = new_base.describe()
            for name, function in methods.items():
                try:
                    actual = function()
                    torch.npu.synchronize()
                    check = qualify(actual, refs)
                    row["methods"][name] = check
                    qualification[name] &= check["qualified"]
                    if name == "candidate":
                        if not check["qualified"]:
                            raise AssertionError("candidate failed original FP64; timing refused")
                        debug, stages, audit = runtime.forward(**device, metadata=device_meta, return_intermediates=True)
                        torch.npu.synchronize()
                        torch.testing.assert_close(debug.cpu(), actual.cpu(), rtol=0, atol=0)
                        effective_q = stages["q_latent_hi"].cpu().double()
                        if stages["q_latent_lo"] is not None:
                            effective_q = effective_q + stages["q_latent_lo"].cpu().double()
                        row["candidate_stages"] = {
                            "query": golden.assert_fp16_close(effective_q, refs["q_latent"]),
                            "latent": golden.assert_fp16_close(stages["latent_out"], refs["latent_out"]),
                        }
                        if "candidate_kernels" not in report:
                            report["candidate_kernels"] = {}
                            for label, kernel in audit["kernels"].items():
                                source = kernel.get_kernel_source()
                                (args.output / ("candidate_" + label + ".cpp")).write_text(source)
                                report["candidate_kernels"][label] = {
                                    "target": str(kernel.target),
                                    "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
                                }
                    read_only(device, cpu, device_meta, cpu_meta, new_base)
                except Exception as error:
                    row["methods"].setdefault(name, {}).update(execution_error=repr(error))
                    save()
                    raise
                save()
            if first_data is None:
                first_data = record, methods, device, device_meta, new_base
        report["initial_qualification"] = qualification
        save()
        if args.mode == "validate":
            report["status"] = "validated_with_baseline_qualification_recorded"
            return

        record, methods, device, device_meta, new_base = first_data
        _, cpu, cpu_meta, _, refs = record
        names = list(methods) if args.order == "candidate-first" else ["historical_fp32", "source_prefill", "candidate"]
        for name in names:
            eligible = qualification[name]
            item = {"qualified_for_comparison": eligible, "diagnostic_only": not eligible, "timing_status": "pending"}
            report["methods"][name] = item
            if not eligible and not args.time_unqualified:
                item["timing_status"] = "not_timed_accuracy_failed"
                save()
                continue
            item["timing_status"] = "running_eager"
            save()
            item["eager"], actual = timing.eager_timing(methods[name], args.warmup, args.repeats)
            item["post_eager_accuracy"] = qualify(actual, refs)
            save()  # Eager result survives a later graph failure.
            if eligible and not item["post_eager_accuracy"]["qualified"]:
                raise AssertionError(f"post-eager original FP64 failure: {name}")
            try:
                item["graph"], actual = timing.graph_timing(methods[name], args.graph_batch, args.repeats)
            except Exception as error:
                policy = error_policy.graph_error_policy(error)
                item.update(
                    timing_status="graph_unavailable", qualified_for_comparison=False, graph_error=repr(error), graph_error_policy=policy
                )
                save()
                if not policy["continue"]:
                    raise
                continue
            item["post_graph_accuracy"] = qualify(actual, refs)
            item["qualified_for_comparison"] = (
                eligible and item["post_eager_accuracy"]["qualified"] and item["post_graph_accuracy"]["qualified"]
            )
            item["timing_status"] = "measured"
            if eligible and not item["post_graph_accuracy"]["qualified"]:
                raise AssertionError(f"post-graph original FP64 failure: {name}")
            read_only(device, cpu, device_meta, cpu_meta, new_base)
            save()
        candidate_item = report["methods"]["candidate"]
        for name in ("historical_fp32", "source_prefill"):
            item = report["methods"][name]
            pair = all(x.get("qualified_for_comparison") and x.get("timing_status") == "measured" for x in (candidate_item, item))
            report["ratios"][name + "_over_candidate_graph"] = (
                item["graph"]["per_call"]["median"] / candidate_item["graph"]["per_call"]["median"] if pair else None
            )
        report["status"] = "benchmark_completed_with_explicit_qualification"
    except Exception as error:
        report.update(status="failed", error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        changed = gmla.package_sources() != inventory
        if changed:
            report.update(status="failed", error="installed candidate sources changed during run")
        save()
        print("GMLA_SOURCE_PREFILL_BENCHMARK " + report["status"], flush=True)
        if changed:
            raise RuntimeError("installed candidate sources changed during run")


if __name__ == "__main__":
    main()
