"""Private numerical/cache-lifetime regression matrix; no timing or CPU fallback."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import traceback


BASE = Path("/mnt/workspace/Gated_MLA_Workspace/evidence/20261112_benchmark_alignment")
BUDGET = {"rtol": 0.01, "atol": 0.002, "max_normalized": 0.005}
SHAPE = {"D": 128, "E": 64, "R": 512, "V": 128}


def cases():
    result = [
        ("representative", 1, 96, 128, 128, 32, 128, [[128, 128, 0]], True),
        ("batch_tags_inactive", 3, 96, 3, 127, 32, 128, [[3, 127, 124], [0, 0, 0], [2, 65, 63]], True),
        ("continuation_multikv", 1, 96, 17, 129, 32, 128, [[17, 129, 112]], True),
        ("small_m_multikv", 1, 96, 4, 128, 16, 64, [[4, 128, 124]], True),
        ("inactive_query_cache_tail", 1, 128, 2, 65, 32, 128, [[1, 65, 64]], True),
        ("generic_head3_m32", 1, 3, 5, 65, 32, 128, [[4, 65, 61]], True),
        ("generic_head3_m16", 1, 3, 5, 65, 16, 64, [[4, 65, 61]], True),
        ("generic_head3_prefetched_tail", 1, 3, 127, 127, 16, 128, [[127, 127, 0]], True),
        ("generic_head97_batches", 2, 97, 2, 128, 16, 128, [[2, 128, 126], [1, 17, 16]], True),
        ("tiny1", 1, 32, 1, 1, 16, 64, [[1, 1, 0]], True),
        ("tiny17", 1, 32, 1, 17, 16, 128, [[1, 17, 16]], True),
        ("tiny63", 1, 32, 1, 63, 32, 64, [[1, 63, 62]], True),
        ("tiny64_no_query_residual", 1, 32, 1, 64, 32, 128, [[1, 64, 63]], False),
    ]
    return [
        dict(name=n, B=b, H=h, Tq=t, Tk=k, block_m=m, block_n=bn, metadata=meta, bf16_residual=res)
        for n, b, h, t, k, m, bn, meta, res in result
    ]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_file(name, path):
    path = Path(path).resolve()
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    if Path(module.__file__).resolve() != path:
        raise RuntimeError("unexpected module source")
    return module


def same_bytes(torch, actual, expected, name):
    if actual.dtype != expected.dtype or tuple(actual.shape) != tuple(expected.shape):
        raise AssertionError("readonly shape/dtype changed: " + name)
    if not torch.equal(actual.detach().cpu().contiguous().view(torch.uint8), expected.detach().cpu().contiguous().view(torch.uint8)):
        raise AssertionError("readonly bytes changed: " + name)


def golden(torch, operands, cpu_metadata):
    """Independent per-query CPU64 readout for the exact supplied Hi(+Lo)."""
    hi, lo, qa, cache, aux = (operands[k] for k in ("hi", "lo", "qa", "cache", "aux"))
    q = hi.detach().cpu().double()
    if lo is not None:
        q = q + lo.detach().cpu().double()
    qa, cache, aux = (x.detach().cpu().double() for x in (qa, cache, aux))
    output = torch.zeros_like(q)
    for b, row in enumerate(cpu_metadata.tolist()):
        qlen, kvlen, start = row[:3]
        for t in range(qlen):
            visible = min(kvlen, start + t + 1)
            if visible < 1:
                raise AssertionError("active query has no visible key")
            scores = (q[b, t] @ cache[b, :visible].T + qa[b, t] @ aux[b, :visible].T) / math.sqrt(192)
            output[b, t] = torch.softmax(scores, dim=-1) @ cache[b, :visible]
    if not bool(torch.isfinite(output).all()):
        raise AssertionError("CPU64 reference must be finite")
    return output


def qualify(torch, metrics, output, reference, cpu_metadata=None):
    if output.dtype != torch.float32 or tuple(output.shape) != tuple(reference.shape):
        raise AssertionError("core output must have the declared shape and FP32 dtype")
    result = metrics(output, reference)
    if not result["passed"]:
        raise AssertionError("original FP64 budget failed: " + repr(result))
    if cpu_metadata is not None:
        actual = output.detach().cpu()
        for b, row in enumerate(cpu_metadata.tolist()):
            if int(torch.count_nonzero(actual[b, row[0] :])) != 0:
                raise AssertionError("inactive query output must be exactly zero")
        result["inactive_queries_exact_zero"] = True
    return result


def payload_probes(torch, methods, operands, cpu_metadata, readonly, metrics, graph=False):
    """Two independent probes; uniform E shift alone does not prove E reload."""
    cache, aux = operands["cache"], operands["aux"]
    saved_c, saved_e = cache.clone(), aux.clone()
    pointers = (cache.data_ptr(), aux.data_ptr())
    results, replay_outputs = {}, {}
    base_reference = golden(torch, operands, cpu_metadata)
    base_outputs = {}
    if graph:
        graphs = {}
        for name, function in methods.items():
            function()
            torch.npu.synchronize()
            handle = torch.npu.NPUGraph()
            with torch.npu.graph(handle):
                replay_outputs[name] = function()
            handle.replay()
            torch.npu.synchronize()
            qualify(torch, metrics, replay_outputs[name], base_reference, cpu_metadata)
            graphs[name] = handle
            base_outputs[name] = replay_outputs[name].clone()

        def call(name):
            graphs[name].replay()
            return replay_outputs[name]
    else:
        for name, function in methods.items():
            base_outputs[name] = function().clone()
        torch.npu.synchronize()

        def call(name):
            return methods[name]()

    for label in ("joint_C_plus_0125_E_minus_025", "E_only_token0_channel0_plus_075"):
        try:
            if label.startswith("joint"):
                cache.add_(0.125)
                aux.sub_(0.25)
            else:
                aux[:, 0, 0].add_(0.75)
            torch.npu.synchronize()
            if pointers != (cache.data_ptr(), aux.data_ptr()):
                raise AssertionError("payload update changed storage pointers")
            # Read back the actual updated LP operands, including their rounding.
            actual_c, actual_e = cache.cpu().clone(), aux.cpu().clone()
            reference = golden(torch, operands, cpu_metadata)
            row = {"same_storage": True, "reference_max_change": float((reference - base_reference).abs().max()), "methods": {}}
            if row["reference_max_change"] == 0:
                raise AssertionError("payload probe did not change the mathematical output")
            for name in methods:
                output = call(name)
                torch.npu.synchronize()
                row["methods"][name] = qualify(torch, metrics, output, reference, cpu_metadata)
                changed = not torch.equal(output.cpu(), base_outputs[name].cpu())
                if not changed:
                    raise AssertionError(name + " ignored live " + label + " payload")
                row["methods"][name]["observed_output_changed"] = True
                readonly(actual_c, actual_e)
            results[label] = row
        finally:
            cache.copy_(saved_c)
            aux.copy_(saved_e)
            torch.npu.synchronize()
            readonly()
        restored = {}
        for name in methods:
            output = call(name)
            torch.npu.synchronize()
            restored[name] = qualify(torch, metrics, output, base_reference, cpu_metadata)
            readonly()
        results[label]["restored_input_bytes_and_output"] = restored
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target", "--targets", dest="targets", nargs="+", choices=("pto", "ascendc"), default=["pto", "ascendc"])
    parser.add_argument("--dtype", "--dtypes", dest="dtypes", nargs="+", choices=("float16", "bfloat16"), default=["float16", "bfloat16"])
    parser.add_argument("--installed-root", type=Path, default=BASE / "installed/site")
    parser.add_argument(
        "--inventory",
        type=Path,
        default=Path("/mnt/workspace/Gated_MLA_Workspace/evidence/20261112_perf_followup/tools_v2/pinned_inventory.json"),
    )
    parser.add_argument(
        "--protocol-root",
        type=Path,
        default=Path("/mnt/workspace/gated-mla-l3-pr/examples_experiment/gated_mla/package/benchmark/protocol"),
    )
    args = parser.parse_args()
    if not args.variant_file.is_file() or args.variant_file.suffix != ".py":
        parser.error("variant-file must be an existing Python source")
    if args.output.exists() or len(set(args.targets)) != len(args.targets) or len(set(args.dtypes)) != len(args.dtypes):
        parser.error("output must be new and selection lists must not contain duplicates")
    args.output.mkdir(parents=True)
    report = {
        "schema": "gmla-prefill-private-regression-v1",
        "status": "preparing",
        "date_UTC": datetime.now(timezone.utc).isoformat(),
        "variant_sha256": sha(args.variant_file),
        "driver_sha256": sha(__file__),
        "targets": args.targets,
        "dtypes": args.dtypes,
        "seeds": [0, 17],
        "budget": BUDGET,
        "timing": False,
        "cases": [],
        "graph_probes": [],
    }
    destination = args.output / "prefill_regression.json"

    def save():
        tmp = destination.with_suffix(".tmp")
        tmp.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        tmp.replace(destination)

    snapshot, initial_snapshot = None, None
    try:
        for key, value in (("TL_PLATFORM", "A2"), ("TL_RUN_MODE", "npu")):
            os.environ.setdefault(key, value)
            if os.environ[key] != value:
                raise RuntimeError("this910B3 regression requires " + key + "=" + value)
        import torch
        import gmla
        from gmla import reference as reference_module

        torch.set_grad_enabled(False)
        torch.set_num_threads(4)
        installed = args.installed_root.resolve()
        if Path(gmla.__file__).resolve() != installed / "gmla/__init__.py":
            raise RuntimeError("normally installed baseline import differs")
        info = list(installed.glob("gated_mla_ascend_candidate-*.dist-info"))
        if len(info) != 1 or not (info[0] / "RECORD").is_file():
            raise RuntimeError("normal installation metadata required")
        direct = info[0] / "direct_url.json"
        if direct.exists() and json.loads(direct.read_text()).get("dir_info", {}).get("editable"):
            raise RuntimeError("editable installation excluded")
        if args.variant_file.resolve().is_relative_to(installed / "gmla"):
            raise RuntimeError("private variant must not replace the installed production module")
        inventory = gmla.package_sources()
        if inventory != json.loads(args.inventory.read_text())["sources"]:
            raise RuntimeError("baseline25 source inventory mismatch")
        if (reference_module.FP16_RTOL, reference_module.FP16_ATOL, reference_module.FP16_MAX_NORMALIZED_ERROR) != (0.01, 0.002, 0.005):
            raise AssertionError("original numerical budget changed")
        manifest = json.loads((args.protocol_root / "protocol_manifest.json").read_text())
        for name, record in manifest["files"].items():
            if sha(args.protocol_root / name) != record["extracted_sha256"]:
                raise RuntimeError("frozen source changed: " + name)
        fixture = load_file("_gmla_regression_fixture", args.protocol_root / "prefill_fixture.py")
        policy = load_file("_gmla_regression_policy", args.protocol_root / "prefill_protocol.py")

        def snapshot():
            return {
                "baseline_sources": gmla.package_sources(),
                "variant_sha256": sha(args.variant_file),
                "driver_sha256": sha(__file__),
                "protocol_sources": {name: sha(args.protocol_root / name) for name in manifest["files"]},
            }

        initial_snapshot = snapshot()
        report["source_identity"] = initial_snapshot
        if getattr(getattr(torch, "npu", None), "_initialized", False):
            raise RuntimeError("fresh process required; preload compiler before device allocation")
        runtime = gmla.build_runtime()
        previous = runtime.kernel_bundle.prefill_module
        variant = load_file("gmla.kernels._regression_" + sha(args.variant_file)[:16], args.variant_file)
        import torch_npu
        import tilelang

        if not torch.npu.is_available():
            raise RuntimeError("real NPU required; no silent skip")
        torch.npu.set_device(0)
        torch.set_float32_matmul_precision("highest")
        torch.npu.matmul.allow_hf32 = False
        report["environment"] = {
            "python": sys.version,
            "torch": str(torch.__version__),
            "torch_npu": str(torch_npu.__version__),
            "tilelang": str(tilelang.__version__),
            "device": torch.npu.get_device_name(0),
        }
        for target in args.targets:
            for dtype_name in args.dtypes:
                dtype = getattr(torch, dtype_name)
                for case in cases():
                    for seed in (0, 17):
                        shape = {key: case[key] for key in ("B", "H", "Tq", "Tk")}
                        cpu = fixture.make_inputs(shape, dtype, seed, "head_specific")
                        cpu_meta = gmla.prepare_metadata(
                            case["metadata"], batch=case["B"], query_capacity=case["Tq"], cache_capacity=case["Tk"]
                        )
                        q64 = torch.einsum("bthd,hdr->bthr", cpu["q_content"].double(), cpu["w_k"].double())
                        hi = q64.to(dtype).contiguous()
                        use_lo = dtype_name == "bfloat16" and case["bf16_residual"]
                        lo = (q64 - hi.double()).to(dtype).contiguous() if use_lo else None
                        device = {name: value.to("npu") for name, value in cpu.items()}
                        operands = {
                            "hi": hi.to("npu"),
                            "lo": None if lo is None else lo.to("npu"),
                            "qa": device["q_aux"],
                            "cache": device["latent_kv"],
                            "aux": device["key_aux"],
                        }
                        metadata = cpu_meta.to("npu")
                        for name, value in {**operands, "metadata": metadata}.items():
                            if value is not None and torch_npu.get_npu_format(value) not in (torch_npu.Format.ND, torch_npu.Format.NCHW):
                                raise AssertionError("input is not a dense base format: " + name)

                        def readonly(
                            updated_c=None,
                            updated_e=None,
                            *,
                            device=device,
                            cpu=cpu,
                            operands=operands,
                            hi=hi,
                            lo=lo,
                            metadata=metadata,
                            cpu_meta=cpu_meta,
                        ):
                            for name, value in device.items():
                                expected = (
                                    updated_c
                                    if name == "latent_kv" and updated_c is not None
                                    else (updated_e if name == "key_aux" and updated_e is not None else cpu[name])
                                )
                                same_bytes(torch, value, expected, name)
                            same_bytes(torch, operands["hi"], hi, "QL_hi")
                            if lo is not None:
                                same_bytes(torch, operands["lo"], lo, "QL_lo")
                            same_bytes(torch, metadata, cpu_meta, "metadata")

                        def bind(module, *, operands=operands, metadata=metadata, case=case, target=target):
                            return lambda: module.cube_prefill(
                                operands["hi"],
                                operands["qa"],
                                operands["cache"],
                                operands["aux"],
                                metadata,
                                block_m=case["block_m"],
                                block_n=case["block_n"],
                                max_blocks=20,
                                target=target,
                                scale=1 / math.sqrt(192),
                                ql_residual=operands["lo"],
                            )

                        methods = {"candidate": bind(variant), "previous": bind(previous)}
                        ref = golden(torch, operands, cpu_meta)
                        row = {"case": case, "target": target, "dtype": dtype_name, "query_residual": use_lo, "seed": seed, "methods": {}}
                        report["cases"].append(row)
                        for name, function in methods.items():
                            output = function()
                            torch.npu.synchronize()
                            row["methods"][name] = qualify(torch, policy.metrics, output, ref, cpu_meta)
                            readonly()
                        if case["name"] in ("representative", "batch_tags_inactive"):
                            row["live_payload_probes"] = payload_probes(torch, methods, operands, cpu_meta, readonly, policy.metrics)
                        if case["name"] == "representative" and seed == 0:
                            result = payload_probes(torch, methods, operands, cpu_meta, readonly, policy.metrics, graph=True)
                            report["graph_probes"].append(
                                {"target": target, "dtype": dtype_name, "query_residual": use_lo, "results": result}
                            )
                        save()
                        print(
                            "GMLA_PREFILL_REGRESSION_CASE "
                            + json.dumps({"case": case["name"], "target": target, "dtype": dtype_name, "seed": seed, "passed": True}),
                            flush=True,
                        )
        expected = len(args.targets) * len(args.dtypes) * len(cases()) * 2
        if len(report["cases"]) != expected or len(report["graph_probes"]) != len(args.targets) * len(args.dtypes):
            raise AssertionError("regression matrix incomplete")
        report.update(status="all_selected_regressions_passed", completed_cases=expected, skipped_cases=0)
    except Exception as error:
        report.update(status="failed", error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        if snapshot is not None and initial_snapshot is not None:
            try:
                if snapshot() != initial_snapshot:
                    report.update(status="failed", source_integrity_error="sources changed during execution")
            except Exception as error:
                report.update(status="failed", source_integrity_error=repr(error))
        save()
        print("GMLA_PREFILL_REGRESSION " + report["status"], flush=True)
        if report.get("source_integrity_error"):
            raise RuntimeError(report["source_integrity_error"])


if __name__ == "__main__":
    main()
