"""One fresh-process trial of balanced Gated MLA/reference measurements.

The production package and the historical timing protocol are never modified.
"""

from __future__ import annotations

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


def tensor_identity(torch, tensor):
    raw = tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    return {"shape": list(tensor.shape), "dtype": str(tensor.dtype), "sha256": hashlib.sha256(raw).hexdigest()}


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def baseline_identity(root, prefill, original_gate):
    record_path = root.parent / "FILELIST.sha256"
    pinned = {line[66:]: line[:64] for line in record_path.read_text().splitlines() if line}
    required = ["adapter_decode.py"]
    if prefill:
        required += ["adapter_prefill.py", "source_unquantized_gemm.py"]
    if original_gate:
        required += [
            "source_gate/isolated_gate.py",
            "source_gate/source_manifest.json",
            "source_gate/original/flash_attention_output.py.txt",
            "source_gate/original/triton_utils.py.txt",
        ]
    hashes = {}
    for name in required:
        actual = sha(root / name)
        if pinned.get("benchmark/" + name) != actual:
            raise RuntimeError("baseline source changed or unregistered: " + name)
        hashes[name] = actual
    return {"filelist_sha256": sha(record_path), "sources": hashes}


def make_candidate(runtime, device, metadata):
    return lambda: runtime.forward(**device, metadata=metadata)


def assert_same_bytes(torch, actual, expected, name):
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        raise AssertionError(f"readonly shape/dtype changed: {name}")
    a = actual.detach().cpu().contiguous().view(torch.uint8)
    b = expected.detach().cpu().contiguous().view(torch.uint8)
    if not torch.equal(a, b):
        raise AssertionError(f"readonly payload changed: {name}")


def make_readonly(torch, device, cpu, metadata, cpu_metadata, source, prefill, historical_table, historical_mask=None):
    expected_table = torch.zeros((1, 1), dtype=torch.int32) if prefill else torch.arange(32, dtype=torch.int32).view(1, 32)
    expected_mask = torch.triu(torch.ones(2048, 2048, dtype=torch.bool), diagonal=1) if prefill else None
    expected_weight = torch.cat((cpu["w_k"], cpu["w_v"]), dim=1).view(96 * 256, 512) if prefill else cpu["w_v"].transpose(1, 2).contiguous()

    def verify():
        for name, value in device.items():
            assert_same_bytes(torch, value, cpu[name], name)
        assert_same_bytes(torch, metadata, cpu_metadata, "metadata")
        assert_same_bytes(torch, historical_table, expected_table, "historical_block_table")
        if prefill:
            assert_same_bytes(torch, source.constants["packed_w_kv"], expected_weight, "source_packed_w_kv")
            assert_same_bytes(torch, source.constants["causal_mask"], expected_mask.to(torch.int8), "source_causal_mask")
            assert_same_bytes(torch, historical_mask, expected_mask, "historical_causal_mask")
        else:
            assert_same_bytes(torch, source.constants["w_uv"], expected_weight, "source_w_uv")
            assert_same_bytes(torch, source.constants["block_table"], expected_table, "source_block_table")

    return verify


def finalize(report, save, package_sources, inventory):
    source_error = None
    try:
        if package_sources() != inventory:
            source_error = "installed candidate sources changed during trial"
    except Exception as error:
        source_error = "source inventory could not be rechecked: " + repr(error)
    if source_error:
        report.update(status="failed", source_integrity_error=source_error)
    save()
    print("GMLA_PAIRED_BENCHMARK " + report["status"], flush=True)
    if source_error:
        raise RuntimeError(source_error)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("decode4096", "decode17", "prefill128"), required=True)
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    parser.add_argument("--target", choices=("pto", "ascendc"), default="pto")
    parser.add_argument("--benchmark-root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--protocol-root", type=Path)
    parser.add_argument("--installed-root", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("cpu-reference", "validate", "benchmark"), default="benchmark")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 17])
    parser.add_argument("--trial-id", type=int, default=0)
    parser.add_argument("--rounds", type=int, default=6)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--graph-batch", type=int)
    parser.add_argument("--graph-repeats", type=int, default=5)
    parser.add_argument("--order-seed", type=int, default=1700)
    parser.add_argument("--time-unqualified", action="store_true")
    parser.add_argument("--gate-module")
    parser.add_argument("--gate-callable", default="flash_attention_gate")
    parser.add_argument("--gate-label")
    args = parser.parse_args()
    if len(set(args.seeds)) < 3 or len(set(args.seeds)) != len(args.seeds):
        parser.error("supply at least three distinct qualification seeds")
    if args.rounds < 6 or args.rounds % 6 or min(args.repeats, args.warmup, args.graph_repeats) < 1:
        parser.error("rounds must be a positive multiple of six; timing counts must be positive")
    if args.gate_module and (not args.gate_label or args.gate_module.split(".")[0] in ("vllm", "vllm_ascend")):
        parser.error("use an isolated gate module and an explicit provenance label")
    if args.output.exists():
        parser.error("output must be a new directory")
    args.output.mkdir(parents=True)
    prefill = args.case == "prefill128"
    args.graph_batch = args.graph_batch if args.graph_batch is not None else (5 if prefill else 50)
    if args.graph_batch < 1:
        parser.error("graph batch must be positive")
    root = args.benchmark_root.resolve()
    protocol_root = (args.protocol_root or root / "protocol").resolve()
    for key, value in (("TL_PLATFORM", "A2"), ("TL_RUN_MODE", "npu")):
        os.environ.setdefault(key, value)
        if os.environ[key] != value:
            parser.error(f"this 910B3 protocol requires {key}={value}")

    import torch
    import gmla
    from gmla import reference as golden

    torch.set_grad_enabled(False)
    torch.set_num_threads(4)
    installed = args.installed_root.resolve()
    if Path(gmla.__file__).resolve() != installed / "gmla/__init__.py":
        raise RuntimeError("normally installed candidate import does not match --installed-root")
    distributions = list(installed.glob("gated_mla_ascend_candidate-*.dist-info"))
    if len(distributions) != 1 or not (distributions[0] / "RECORD").is_file():
        raise RuntimeError("normal wheel/site installation metadata is required")
    direct = distributions[0] / "direct_url.json"
    if direct.exists() and json.loads(direct.read_text()).get("dir_info", {}).get("editable"):
        raise RuntimeError("editable candidate installs are excluded")
    inventory = gmla.package_sources()
    if inventory != json.loads(args.inventory.read_text())["sources"]:
        raise RuntimeError("installed source inventory mismatch")
    if (golden.FP16_RTOL, golden.FP16_ATOL, golden.FP16_MAX_NORMALIZED_ERROR) != (0.01, 0.002, 0.005):
        raise AssertionError("the precision budget must not be changed")
    manifest = json.loads((protocol_root / "protocol_manifest.json").read_text())
    for name, item in manifest["files"].items():
        if sha(protocol_root / name) != item["extracted_sha256"]:
            raise RuntimeError(f"historical protocol changed: {name}")
    original_gate = args.gate_module == "source_gate.isolated_gate"
    baseline_sources = baseline_identity(root, prefill, original_gate)
    timing_source = Path(__file__).with_name("paired_timing.py")

    def source_snapshot():
        return {
            "candidate": gmla.package_sources(),
            "baseline": baseline_identity(root, prefill, original_gate),
            "driver": sha(__file__),
            "paired_timing": sha(timing_source),
        }

    initial_snapshot = source_snapshot()
    fixture = load_module("gmla_paired_fixture", protocol_root / ("prefill_fixture.py" if prefill else "decode_fixture.py"))
    protocol = load_module("gmla_paired_protocol", protocol_root / ("prefill_protocol.py" if prefill else "decode_protocol.py"))
    error_policy = load_module("gmla_paired_error_policy", protocol_root / "prefill_protocol.py")
    if 1 / math.sqrt(192) != gmla.DEFAULT_SCALE:
        raise AssertionError("the default scale differs from the established contract")
    dtype = getattr(torch, args.dtype)
    source_name = "source_prefill" if prefill else "source_decode"
    names = ["candidate", "historical_fp32", source_name]
    shape = dict(B=1, H=96, Tq=128 if prefill else 1, Tk=128 if prefill else 4096, D=128, E=64, R=512, V=128)
    visible = 128 if prefill else (17 if args.case == "decode17" else 4096)
    report = {
        "schema": "gmla-balanced-comparison-v1",
        "status": "preparing",
        "date_UTC": datetime.now(timezone.utc).isoformat(),
        "case": args.case,
        "shape": shape,
        "visible": visible,
        "dtype": args.dtype,
        "target": args.target,
        "trial_id": args.trial_id,
        "qualification_seeds": args.seeds,
        "timed_seed": args.seeds[0],
        "budget": {"rtol": 0.01, "atol": 0.002, "max_normalized": 0.005},
        "scope": "seven-input forward before W_O; external projections/norm/cache append/communication excluded",
        "package_sources": inventory,
        "inventory_sha256": sha(args.inventory),
        "protocol_manifest_sha256": sha(protocol_root / "protocol_manifest.json"),
        "harness_sha256": sha(__file__),
        "paired_timing_sha256": sha(timing_source),
        "baseline_sources": baseline_sources,
        "rounds": args.rounds,
        "repeats_per_round": args.repeats,
        "graph_batch": args.graph_batch,
        "graph_repeats_per_round": args.graph_repeats,
        "order_seed": args.order_seed,
        "cache_policy": "reuse inputs/weights; no L2 flush",
        "accuracy": [],
        "initial_qualification": {},
        "measurement": None,
        "command": shlex.join([sys.executable, *sys.argv]),
    }
    destination = args.output / "paired_benchmark.json"

    def save():
        temporary = destination.with_suffix(".tmp")
        temporary.write_text(json.dumps(report, indent=2) + "\n")
        temporary.replace(destination)

    def qualify(actual, refs):
        if (
            not isinstance(actual, torch.Tensor)
            or tuple(actual.shape) != (shape["B"], shape["Tq"], shape["H"], 128)
            or actual.dtype != dtype
        ):
            raise AssertionError("output shape or dtype contract failed")
        if prefill:
            a, b = protocol.metrics(actual, refs["expanded"]), protocol.metrics(actual, refs["absorbed"])
            good = a["passed"] and b["passed"]
        else:
            a = protocol.qualify(actual, refs["expanded"], dtype, golden)
            b = protocol.qualify(actual, refs["absorbed"], dtype, golden)
            good = a["qualified"] and b["qualified"]
        return {"qualified": bool(good), "expanded": a, "absorbed": b}

    try:
        save()
        prepared = []
        for seed in args.seeds:
            if prefill:
                cpu = fixture.make_inputs(shape, dtype, seed, "head_specific")
                cpu_meta = gmla.prepare_metadata([[128, 128, 0]], batch=1, query_capacity=128, cache_capacity=128)
                refs = fixture.references(cpu, cpu_meta, golden, gmla.DEFAULT_SCALE)
            else:
                cpu = fixture.make_inputs(fixture.CASES["model_4k"], seed, dtype, "head_specific")
                cpu_meta = gmla.prepare_metadata([[1, 4096, visible - 1]], batch=1, query_capacity=1, cache_capacity=4096)
                refs = fixture.goldens(cpu, cpu_meta, golden)
            prepared.append((seed, cpu, cpu_meta, gmla.metadata_for_reference(cpu_meta), refs))
            if seed == args.seeds[0]:
                report["timed_inputs"] = {name: tensor_identity(torch, value) for name, value in cpu.items()}
                report["timed_metadata"] = tensor_identity(torch, cpu_meta)
            report.setdefault("CPU_references", []).append(
                {"seed": seed, "expanded_absorbed_max_abs": float((refs["expanded"] - refs["absorbed"]).abs().max())}
            )
            save()
        if args.mode == "cpu-reference":
            if "tilelang" in sys.modules or getattr(getattr(torch, "npu", None), "_initialized", False):
                raise AssertionError("CPU reference validation initialized compiler/device")
            report.update(
                status="cpu_reference_passed", NPU_execution=False, environment={"python": sys.version, "torch": str(torch.__version__)}
            )
            return

        runtime = gmla.build_runtime(target=args.target)
        if getattr(getattr(torch, "npu", None), "_initialized", False):
            raise RuntimeError("use a fresh process and preload TileLang before NPU allocation")
        _ = runtime.kernel_bundle
        import torch_npu
        import tilelang

        if not torch.npu.is_available():
            raise RuntimeError("real NPU required; device tests are not silently skipped")
        torch.npu.set_device(0)
        # Only baseline/protocol modules are added, never a candidate package directory.
        sys.path.insert(0, str(root))
        adapter = importlib.import_module("adapter_prefill" if prefill else "adapter_decode")
        baseline_class = adapter.VllmSourcePrefill if prefill else adapter.VllmSourceDecode
        gate_impl = None
        report["gate"] = {"mode": "torch_rounding", "description": "unfused low-precision gate expression"}
        if args.gate_module:
            module = importlib.import_module(args.gate_module)
            gate_impl = getattr(module, args.gate_callable)
            report["gate"] = {
                "mode": "verified_original_kernel" if original_gate else "injected_callable",
                "label": args.gate_label,
                "module": args.gate_module,
                "source_sha256": sha(module.__file__),
            }
        report["environment"] = {
            "python": sys.version,
            "torch": str(torch.__version__),
            "torch_npu": str(torch_npu.__version__),
            "torch_npu_file": torch_npu.__file__,
            "tilelang": str(tilelang.__version__),
            "tilelang_file": tilelang.__file__,
            "device": torch.npu.get_device_name(0),
            "properties": str(torch.npu.get_device_properties(0)),
            "matmul_precision": torch.get_float32_matmul_precision(),
            "matmul_allow_hf32": getattr(getattr(torch.npu, "matmul", None), "allow_hf32", "unavailable"),
        }
        report["adapter_sha256"] = sha(root / ("adapter_prefill.py" if prefill else "adapter_decode.py"))
        first = None
        qualified = {name: True for name in names}
        for seed, cpu, cpu_meta, host, refs in prepared:
            device = {name: value.npu() for name, value in cpu.items()}
            meta = cpu_meta.npu()
            source = baseline_class(device, host, api=torch_npu, gate_impl=gate_impl, gate_name=args.gate_label)
            if prefill:
                table = torch.zeros((1, 1), dtype=torch.int32).npu()
                mask = torch.triu(torch.ones((2048, 2048), dtype=torch.bool), diagonal=1).npu()
                historical = protocol.OfficialFullChains(
                    device, mask, table, torch_npu.npu_fused_infer_attention_score_v2, gmla.DEFAULT_SCALE
                ).mla_paged
            else:
                table = torch.arange(32, dtype=torch.int32).view(1, 32).npu()
                historical = protocol.FiaComposition(torch_npu.npu_fused_infer_attention_score_v2, golden, device, host, table)

            methods = {"candidate": make_candidate(runtime, device, meta), "historical_fp32": historical, source_name: source}
            readonly = make_readonly(torch, device, cpu, meta, cpu_meta, source, prefill, table, mask if prefill else None)
            row = {"seed": seed, "methods": {}}
            report["accuracy"].append(row)
            for name, function in methods.items():
                out = function()
                torch.npu.synchronize()
                check = qualify(out, refs)
                row["methods"][name] = check
                qualified[name] &= check["qualified"]
                if name == "candidate":
                    if not check["qualified"]:
                        raise AssertionError("candidate failed original FP64 budget")
                    debug, stages, audit = runtime.forward(**device, metadata=meta, return_intermediates=True)
                    torch.npu.synchronize()
                    torch.testing.assert_close(debug.cpu(), out.cpu(), rtol=0, atol=0)
                    ql = stages["q_latent_hi"].cpu().double()
                    if stages["q_latent_lo"] is not None:
                        ql += stages["q_latent_lo"].cpu().double()
                    row["candidate_stages"] = {
                        "query": golden.assert_fp16_close(ql, refs["q_latent"]),
                        "latent": golden.assert_fp16_close(stages["latent_out"], refs["latent_out"]),
                    }
                    if "candidate_kernels" not in report:
                        report["candidate_kernels"] = {}
                        for label, kernel in audit["kernels"].items():
                            text = kernel.get_kernel_source()
                            (args.output / ("candidate_" + label + ".cpp")).write_text(text)
                            report["candidate_kernels"][label] = {
                                "target": str(kernel.target),
                                "source_sha256": hashlib.sha256(text.encode()).hexdigest(),
                            }
                readonly()
                save()
            if first is None:
                first = (methods, refs, readonly)
                report["source_adapter"] = source.describe()
        report["initial_qualification"] = qualified
        if args.mode == "validate":
            report.update(status="NPU_validation_passed_with_explicit_baseline_qualification", NPU_execution=True)
            return
        methods, refs, readonly = first

        def check_after(name, output, boundary):
            result = qualify(output, refs)
            readonly()
            report.setdefault("post_measurement_checks", []).append({"method": name, "boundary": boundary, **result})
            return result

        def progress(partial):
            report["measurement"] = partial
            save()

        from paired_timing import run_measurements

        report["measurement"] = run_measurements(
            methods,
            qualified=qualified,
            check=check_after,
            warmup=args.warmup,
            rounds=args.rounds,
            repeats=args.repeats,
            graph_batch=args.graph_batch,
            graph_repeats=args.graph_repeats,
            order_seed=args.order_seed,
            time_unqualified=args.time_unqualified,
            graph_error_policy=error_policy.graph_error_policy,
            on_progress=progress,
        )
        report.update(status="balanced_benchmark_completed", NPU_execution=True)
    except Exception as error:
        if hasattr(error, "partial_report"):
            report["measurement"] = error.partial_report
        report.update(status="failed", error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        finalize(report, save, source_snapshot, initial_snapshot)


if __name__ == "__main__":
    main()
