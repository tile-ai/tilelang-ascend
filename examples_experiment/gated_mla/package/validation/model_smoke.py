"""Actual frozen Kimi layer vs packed NPU adapter, before and after W_O.

Uses normally imported installed gmla providers at real H96/qrank1536/hidden7168.
No checkpoint or whole model is loaded; no prototype provider is injected.
"""

import argparse
import hashlib
import json
from pathlib import Path
import traceback
import time

import torch


def expected_package_sources(manifest_path):
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {}
    for entry in manifest["selected_sources"]:
        if entry["destination"].startswith("gmla/"):
            expected[entry["destination"][5:]] = entry["destination_sha256"]
    for entry in manifest["new_package_files"]:
        if entry["path"].startswith("gmla/"):
            expected[entry["path"][5:]] = entry["sha256"]
    return expected


class CostRecorder:
    """Diagnostic event intervals only; no math or provider replacement."""

    def __init__(self, layer, owner):
        self.active = False
        self.pending = {}
        self.rows = []
        self.handles = []
        for name in ("q_a_proj", "q_a_layernorm", "q_b_proj", "kv_a_proj_with_mqa", "kv_a_layernorm", "g_proj", "o_proj"):
            module = getattr(layer, name)
            self.handles.append(module.register_forward_pre_hook(self.before(name)))
            self.handles.append(module.register_forward_hook(self.after(name)))
        original_append = owner.append

        def append(*args, **kwargs):
            if not self.active:
                return original_append(*args, **kwargs)
            start, end = self.events()
            begin = time.perf_counter()
            start.record()
            result = original_append(*args, **kwargs)
            end.record()
            self.rows.append(("cache_append", start, end, (time.perf_counter() - begin) * 1000))
            return result

        # Observe the normal bound owner method; do not inject a provider/kernel.
        owner.append = append

    @staticmethod
    def events():
        return torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)

    def before(self, name):
        def hook(_module, _args):
            if self.active:
                start, end = self.events()
                start.record()
                self.pending[name] = (start, end, time.perf_counter())

        return hook

    def after(self, name):
        def hook(_module, _args, _result):
            if self.active:
                start, end, begin = self.pending.pop(name)
                end.record()
                self.rows.append((name, start, end, (time.perf_counter() - begin) * 1000))

        return hook

    def begin(self):
        self.rows = []
        self.pending = {}
        self.active = True

    def finish(self, whole_wall_ms):
        # The unchanged packaged wrapper synchronized core and W_O on success.
        # No extra synchronize/event query is performed on a failed forward.
        rows = [
            {"operation": name, "stream_interval_ms": start.elapsed_time(end), "host_entry_ms": wall}
            for name, start, end, wall in self.rows
        ]
        return {
            "whole_candidate_wall_ms": whole_wall_ms,
            "operations": rows,
            "external_projection_norm_stream_ms": sum(
                row["stream_interval_ms"] for row in rows if row["operation"] not in ("cache_append", "o_proj")
            ),
            "scope": "single instrumented call; includes cold initialization/JIT/host gaps; not steady-state kernel time or a performance gate",
            "input_H2D_included": False,
            "metadata_H2D_in_cache_append": True,
        }


def error_check(actual, expected):
    a, r = actual.detach().cpu().double(), expected.detach().cpu().double()
    assert a.shape == r.shape and torch.isfinite(a).all() and torch.isfinite(r).all()
    difference = (a - r).abs()
    max_abs = float(difference.max())
    normalized = max_abs / max(float(r.abs().max()), 1e-9)
    over = int(torch.count_nonzero(difference > (0.002 + 0.01 * r.abs())))
    result = {"max_abs": max_abs, "max_normalized": normalized, "elementwise_out_of_budget": over}
    result["passed"] = over == 0 and normalized <= 0.005
    return result


def external_input_comparison(actual, expected):
    a, e = actual.detach().cpu(), expected.detach().cpu()
    if a.shape != e.shape or a.dtype != e.dtype:
        return {"equal": False, "actual_shape": list(a.shape), "expected_shape": list(e.shape), "dtype_match": a.dtype == e.dtype}
    return {"equal": torch.equal(a.view(torch.int16), e.view(torch.int16)), "max_abs": float((a.double() - e.double()).abs().max())}


def diagnostic_continuation_allowed(enabled, comparisons, external_checks):
    oracle_keys = ("before_wo_vs_fp64_seven_input", "after_wo_vs_fp64_core_plus_wo")
    official_keys = ("before_wo_vs_actual_official_layer", "after_wo_vs_actual_official_layer")
    required_inputs = ("q_projection", "normalized_c", "raw_kv", "gate_logits")
    return (
        enabled
        and all(comparisons[name]["passed"] for name in oracle_keys)
        and any(not comparisons[name]["passed"] for name in official_keys)
        and all(external_checks.get(name, {}).get("equal", False) for name in required_inputs)
    )


def save_failure_diagnostics(output, label, comparisons, official_checks, tensors, external_inputs, context):
    """CPU artifacts only; preserve the failed gates and all offending indices."""
    saved = {name: tensor.detach().cpu() for name, tensor in tensors.items()}
    saved["external_inputs"] = {name: tensor.detach().cpu() for name, tensor in external_inputs.items()}
    saved["context"] = context
    index_tensors = {}
    details = {}
    pairs = {
        "before_wo_vs_actual_official_layer": ("candidate_before_wo", "official_before_wo"),
        "after_wo_vs_actual_official_layer": ("candidate_after_wo", "official_after_wo"),
        "before_wo_vs_fp64_seven_input": ("candidate_before_wo", "fp64_before_wo"),
        "after_wo_vs_fp64_core_plus_wo": ("candidate_after_wo", "fp64_after_wo"),
        "official_before_wo_vs_same_fp64": ("official_before_wo", "fp64_before_wo"),
        "official_after_wo_vs_same_fp64": ("official_after_wo", "fp64_after_wo"),
    }
    for name, (actual_key, reference_key) in pairs.items():
        actual, reference = saved[actual_key].double(), saved[reference_key].double()
        difference = (actual - reference).abs()
        tolerance = 0.002 + 0.01 * reference.abs()
        bad = torch.nonzero(difference > tolerance, as_tuple=False)
        index_tensors[name] = bad
        samples = []
        for row in bad[:64]:
            index = tuple(int(x) for x in row.tolist())
            samples.append(
                {
                    "index": index,
                    "actual": float(actual[index]),
                    "reference": float(reference[index]),
                    "absolute_error": float(difference[index]),
                    "allowed_error": float(tolerance[index]),
                }
            )
        flat_index = int(difference.argmax())
        max_index = tuple(int(x) for x in torch.unravel_index(torch.tensor(flat_index), difference.shape))
        details[name] = {"elementwise_count": len(bad), "samples_first64": samples, "max_absolute_error_index": max_index}
    saved["offending_indices"] = index_tensors
    tensor_path = output / f"{label}_failure_tensors.pt"
    torch.save(saved, tensor_path)
    summary = {
        "candidate_admission_gates_unchanged": comparisons,
        "official_vs_same_fp64_diagnostic_only": official_checks,
        "offending_indices": details,
        "context": context,
        "tensor_artifact": tensor_path.name,
        "tensor_artifact_sha256": hashlib.sha256(tensor_path.read_bytes()).hexdigest(),
        "notes": "All arrays and bad indices saved on CPU. No gate was relaxed; original failed reports remain separate.",
    }
    json_path = output / f"{label}_failure_diagnostics.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return {"tensors": tensor_path.name, "diagnostics": json_path.name}


def execute(args, report):
    import torch_npu

    import gmla
    from gmla import reference as golden
    from gmla.integration.kimi import FrozenKimiLayerAdapter
    from gmla.integration.official_layer_fixture import MinimalExpandedKVCache, actual_dense_layer_forward, load_frozen_layer, small_config
    import gmla._packed as packed_implementation
    import gmla.integration.kimi as kimi_implementation
    import gmla.integration.official_layer_fixture as official_fixture_implementation
    from gmla.kernels import query, value, decode, prefill

    package_root = Path(gmla.__file__).resolve().parent
    expected = expected_package_sources(args.package_manifest)
    actual = gmla.package_sources()
    if actual != expected:
        raise AssertionError("Installed gmla production sources differ from the supplied final manifest")
    runtime = gmla.build_runtime(target=args.target)
    if torch.npu.is_initialized():
        raise RuntimeError("Run this smoke in a fresh process; preload must precede NPU operations")
    _ = runtime.kernel_bundle
    if torch.npu.is_initialized():
        raise AssertionError("Package preload unexpectedly initialized NPU")
    if not torch.npu.is_available():
        raise RuntimeError("NPU required; actual layer fixture cannot be skipped")
    torch.npu.set_device(0)
    modules = {
        "golden": golden,
        "packed_adapter": packed_implementation,
        "kimi_adapter": kimi_implementation,
        "official_fixture": official_fixture_implementation,
        "query": query,
        "value": value,
        "decode": decode,
        "prefill": prefill,
    }
    for module in modules.values():
        assert Path(module.__file__).resolve().is_relative_to(package_root)
    report.update(
        package_module=str(Path(gmla.__file__).resolve()),
        package_sources=actual,
        supplied_manifest_sha256=hashlib.sha256(args.package_manifest.read_bytes()).hexdigest(),
        candidate_provider_injection=False,
        preload_before_NPU_operations=True,
    )
    definitions = load_frozen_layer(str(args.official_source))
    dtype = getattr(torch, args.dtype)
    torch.manual_seed(20261111)
    config = small_config(heads=args.heads, hidden_size=args.hidden_size, q_rank=args.q_rank)
    layer = definitions.attention_class(config, layer_idx=0).to(device="npu:0", dtype=dtype).eval()
    with torch.no_grad():
        owner = gmla.create_cache(2, 65, dtype=dtype, device="npu:0", target=args.target)
        core = gmla.build_packed_adapter(owner, heads=args.heads)
    adapter = FrozenKimiLayerAdapter(layer, core)
    for name, path in core.parts.packed_projection.source_files.items():
        assert Path(path).resolve().is_relative_to(package_root), (name, path)
    recorder = CostRecorder(layer, owner)
    report.update(
        official_class=layer.__class__.__name__,
        official_revision="c5d1dd4c428bd1ce8b88c5044f3b6ccde9e3b721",
        official_source_sha256=definitions.source_sha256,
        official_ast_sha256=definitions.ast_sha256,
        source_loading_method="exact four definition ASTs, unchanged bodies; no whole transformers module import",
        config_object_type="types.SimpleNamespace matching attributes read by the original class",
        actual_original_forward_executed=False,
        original_forward_calls=0,
        config=vars(config),
        device=torch.npu.get_device_name(0),
        torch_npu=torch_npu.__version__,
        cache_capacity=65,
        binding_metadata_h2d_bytes=core.binding_control_transfer_bytes,
        module_sources={
            name: {"path": module.__file__, "sha256": hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()}
            for name, module in modules.items()
        },
    )
    dense_cache = MinimalExpandedKVCache()
    generator = torch.Generator().manual_seed(29)
    report["steps"] = []
    report["failed_original_gates"] = []
    weights_before = {name: value.detach().cpu().clone() for name, value in layer.named_parameters()}
    expected_c = torch.zeros(2, 65, 512, dtype=dtype)
    expected_e = torch.zeros(2, 65, 64, dtype=dtype)
    with torch.no_grad():
        for label, tokens in (("prefill", 17), ("continuation", 3), ("decode", 1)):
            hidden_cpu = torch.randn(2, tokens, args.hidden_size, generator=generator).to(dtype)
            hidden = hidden_cpu.npu()
            previous = owner.lengths
            recorder.begin()
            start_wall = time.perf_counter()
            try:
                output, stages, audit = adapter.forward_new_tokens(hidden, (tokens, tokens), return_intermediates=True)
            finally:
                recorder.active = False
            candidate_wall_ms = (time.perf_counter() - start_wall) * 1000
            phase_costs = recorder.finish(candidate_wall_ms)
            official_external = {}
            dense_output, dense_before = actual_dense_layer_forward(
                layer, hidden, past_key_values=dense_cache, capture_external=official_external
            )
            torch.npu.synchronize()
            report["original_forward_calls"] += 1
            report["actual_original_forward_executed"] = True
            q, weight, normalized_c, raw_kv, gate = stages["prepared"]
            assert owner.lengths == (dense_cache.get_seq_length(layer.layer_idx),) * 2
            external_checks = {
                "q_projection": external_input_comparison(official_external["q_projection"], q),
                "normalized_c": external_input_comparison(official_external["normalized_c"], normalized_c),
                "raw_kv": external_input_comparison(official_external["raw_kv"], raw_kv),
                "gate_logits": external_input_comparison(official_external["gate_logits"], gate),
            }
            for b in range(2):
                expected_c[b, previous[b] : owner.lengths[b]] = normalized_c[b].cpu()
                expected_e[b, previous[b] : owner.lengths[b]] = raw_kv[b, :, 512:].cpu()
            assert torch.equal(owner.cache_c.cpu().view(torch.int16), expected_c.view(torch.int16))
            assert torch.equal(owner.cache_e.cpu().view(torch.int16), expected_e.view(torch.int16))
            assert torch.equal(hidden.cpu().view(torch.int16), hidden_cpu.view(torch.int16))
            for name, parameter in layer.named_parameters():
                assert torch.equal(parameter.detach().cpu().view(torch.int16), weights_before[name].view(torch.int16)), name
            q_cpu = q.detach().cpu().view(2, tokens, args.heads, 192)
            w_cpu = weight.detach().cpu().view(args.heads, 256, 512)
            reference = modules["golden"].expanded_reference(
                q_cpu[..., :128].contiguous(),
                q_cpu[..., 128:].contiguous(),
                owner.cache_c.cpu(),
                owner.cache_e.cpu(),
                w_cpu[:, :128].contiguous(),
                w_cpu[:, 128:].contiguous(),
                gate.detach().cpu().view(2, tokens, args.heads, 128),
                q_lens=(tokens, tokens),
                kv_lens=owner.lengths,
                query_start=previous,
                scale=layer.scaling,
            )
            absorbed = modules["golden"].absorbed_reference(
                q_cpu[..., :128].contiguous(),
                q_cpu[..., 128:].contiguous(),
                owner.cache_c.cpu(),
                owner.cache_e.cpu(),
                w_cpu[:, :128].contiguous(),
                w_cpu[:, 128:].contiguous(),
                gate.detach().cpu().view(2, tokens, args.heads, 128),
                q_lens=(tokens, tokens),
                kv_lens=owner.lengths,
                query_start=previous,
                scale=layer.scaling,
            )
            torch.testing.assert_close(reference, absorbed, rtol=2e-12, atol=2e-12)
            reference_before = reference.view(2, tokens, args.heads * 128)
            reference_after = torch.nn.functional.linear(reference_before, layer.o_proj.weight.detach().cpu().double())
            comparisons = {
                "before_wo_vs_actual_official_layer": error_check(stages["before_wo"], dense_before),
                "after_wo_vs_actual_official_layer": error_check(output, dense_output),
                "before_wo_vs_fp64_seven_input": error_check(stages["before_wo"], reference_before),
                "after_wo_vs_fp64_core_plus_wo": error_check(output, reference_after),
            }
            official_checks = {
                "official_before_wo_vs_same_fp64": error_check(dense_before, reference_before),
                "official_after_wo_vs_same_fp64": error_check(dense_output, reference_after),
            }
            kernel_hashes = {}
            for name, kernel in audit["kernels"].items():
                source = kernel.get_kernel_source()
                (args.output / f"{label}_{name}.cpp").write_text(source, encoding="utf-8")
                kernel_hashes[name] = hashlib.sha256(source.encode()).hexdigest()
            if label == "prefill":
                query_code = audit["kernels"]["packed_query"].get_kernel_source()
                value_code = audit["kernels"]["packed_value_gate"].get_kernel_source()
                assert "gemm_v0" in query_code, "Prefill must exercise selected Cube Query"
                assert "LowValueWorkspace" in value_code, "Prefill must exercise Value R3, not old Vector/Value R2"
            record = {
                "cold_inclusive_cost_diagnostics": phase_costs,
                "independent_expanded_absorbed_FP64_equal": True,
                "selected_Cube_Query_Value_R3_confirmed": label == "prefill",
                "label": label,
                "tokens": tokens,
                "old_lengths": previous,
                "new_lengths": owner.lengths,
                "route": audit["route"],
                "query_residual_enabled": audit["query_residual_enabled"],
                "metadata_h2d_bytes": audit["metadata_h2d_bytes"],
                "comparisons": comparisons,
                "official_vs_same_fp64_diagnostic_only": official_checks,
                "actual_source_external_inputs": external_checks,
                "official_cache_update": dense_cache.update_calls[-1],
                "kernel_source_sha256": kernel_hashes,
                "model_weights_and_hidden_read_only": True,
                "cache_prefix_append_and_future_exact": True,
            }
            report["steps"].append(record)
            (args.output / "package_model_result.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
            failed_gates = [name for name, metric in comparisons.items() if not metric["passed"]]
            same_external = all(x["equal"] for x in external_checks.values())
            record["original_gates_passed"] = not failed_gates and same_external
            if failed_gates or not same_external:
                report["failed_original_gates"].append(
                    {"phase": label, "failed_comparisons": failed_gates, "same_external_inputs": same_external}
                )
                record["failure_artifacts"] = save_failure_diagnostics(
                    args.output,
                    label,
                    comparisons,
                    official_checks,
                    {
                        "candidate_before_wo": stages["before_wo"],
                        "official_before_wo": dense_before,
                        "fp64_before_wo": reference_before,
                        "candidate_after_wo": output,
                        "official_after_wo": dense_output,
                        "fp64_after_wo": reference_after,
                        "official_expanded_kv_current_step": official_external["expanded_kv"],
                        "packed_kv_weight": weight,
                    },
                    {
                        "candidate_q_projection": q,
                        "official_q_projection": official_external["q_projection"],
                        "candidate_normalized_c": normalized_c,
                        "official_normalized_c": official_external["normalized_c"],
                        "candidate_raw_kv": raw_kv,
                        "official_raw_kv": official_external["raw_kv"],
                        "candidate_gate_logits": gate,
                        "official_gate_logits": official_external["gate_logits"],
                    },
                    {
                        "old_lengths": previous,
                        "new_lengths": owner.lengths,
                        "config": vars(config),
                        "external_checks": external_checks,
                        "admission_budget": {"rtol": 0.01, "atol": 0.002, "max_normalized": 0.005},
                    },
                )
                if diagnostic_continuation_allowed(args.collect_diagnostics, comparisons, external_checks):
                    record["diagnostic_continuation"] = True
                    print("GMLA_OFFICIAL_LAYER_STEP_DIAGNOSTIC_FAILED_GATE " + json.dumps(record), flush=True)
                    continue
                raise AssertionError("single-layer candidate exceeds an unchanged numerical gate; no fallback")
            print("GMLA_OFFICIAL_LAYER_STEP_PASS " + json.dumps(record), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-source", type=Path, required=True)
    parser.add_argument("--package-manifest", type=Path, required=True)
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    parser.add_argument("--target", choices=("pto", "ascendc"), default="pto")
    parser.add_argument("--heads", type=int, choices=(96,), default=96)
    parser.add_argument("--hidden-size", type=int, choices=(7168,), default=7168)
    parser.add_argument("--q-rank", type=int, choices=(1536,), default=1536)
    parser.add_argument(
        "--collect-diagnostics",
        action="store_true",
        help="continue cross-official-only failures while candidate FP64 gates and external-input equality pass; never declare acceptance",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("output must be new or empty")
    args.output.mkdir(parents=True, exist_ok=True)
    report = {
        "status": "running",
        "dtype": args.dtype,
        "target": args.target,
        "scope": "one exact frozen-source layer at H96/hidden7168/qrank1536 with random weights; original dense forward baseline",
        "external_costs": ["Q projections/norm", "raw KV projection", "latent RMSNorm", "gate logits projection", "W_O"],
        "timing": "instrumented single-call wall and NPU stream intervals, cold-inclusive; not a kernel performance gate",
        "hf_dynamic_cache_integration": False,
        "dense_baseline_cache": "MinimalExpandedKVCache fixture; actual frozen forward calls past_key_values.update",
        "dense_baseline_recomputes_old_hidden_states": False,
        "checkpoint_loaded": False,
        "automatic_fallback": False,
        "collect_diagnostics": args.collect_diagnostics,
        "probe_source": {"path": str(Path(__file__).resolve()), "sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
    }
    try:
        execute(args, report)
        if args.collect_diagnostics:
            report["status"] = "diagnostic_only_failed" if report["failed_original_gates"] else "diagnostic_only_completed"
        else:
            report["status"] = "passed"
    except Exception as error:
        report.update(status="failed", error=repr(error), traceback=traceback.format_exc())
    steps = report.get("steps", [])
    oracle_names = ("before_wo_vs_fp64_seven_input", "after_wo_vs_fp64_core_plus_wo")
    report["candidate_FP64_all_three_phases_passed"] = len(steps) == 3 and all(
        step["comparisons"][name]["passed"] for step in steps for name in oracle_names
    )
    report["actual_official_external_inputs_equal_all_phases"] = len(steps) == 3 and all(
        item["equal"] for step in steps for item in step["actual_source_external_inputs"].values()
    )
    (args.output / "package_model_result.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if report["status"] == "diagnostic_only_completed":
        print("GMLA_OFFICIAL_LAYER_DIAGNOSTIC_ONLY_COMPLETED", flush=True)
        return
    if report["status"] != "passed":
        if "traceback" in report:
            print(report["traceback"], flush=True)
        else:
            print("GMLA_OFFICIAL_LAYER_DIAGNOSTIC_ONLY_FAILED " + json.dumps(report["failed_original_gates"]), flush=True)
        raise SystemExit(1)
    print("GMLA_OFFICIAL_LAYER_PASS", flush=True)


if __name__ == "__main__":
    main()
