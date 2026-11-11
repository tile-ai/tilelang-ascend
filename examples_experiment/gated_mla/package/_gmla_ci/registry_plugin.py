"""Collect exact selected test parameters into an auditable factory case registry."""

import hashlib
import json
import math
from pathlib import Path

import pytest


def pytest_addoption(parser):
    parser.addoption("--gmla-registry-out", required=False, help="Write selected real test cases; use --collect-only")


def pytest_configure(config):
    existing = {line.split(":", 1)[0].strip() for line in config.getini("markers")}
    if "npu" not in existing:
        config.addinivalue_line("markers", "npu: real NPU required when executed")


def legacy_cases_for_item(item):
    params = getattr(getattr(item, "callspec", None), "params", {})
    name = getattr(item, "originalname", None) or item.name.split("[", 1)[0]
    target, visible, mode = params.get("target", "pto"), 64, "full"
    if name in ("test_decode_full_chain", "test_decode_gate_zero_and_saturation"):
        pass
    elif name == "test_decode_valid_length_and_causal":
        visible = min(params["kv_len"], params["position"] + 1)
    elif name in ("test_decode_zero_query_reads_mean", "test_decode_future_and_padding_invariance"):
        visible = 17
    elif name == "test_decode_inactive_query":
        mode = "zero"
    elif name == "test_latent_readout_prefix_mask":
        mode, visible = "latent", params["visible"]
    elif name == "test_decode_explicit_backend":
        mode, visible = ("full", params["position"] + 1) if params["active"] else ("zero", 0)
    elif name == "test_decode_extreme_scale_single_key":
        mode, visible = "full", 1
    elif name == "test_decode_contract_rejects_before_kernel":
        mode = "reject"
    else:
        raise pytest.UsageError(f"Unregistered selected legacy NPU test: {item.nodeid}")
    return [
        dict(
            id=item.nodeid,
            legacy=True,
            legacy_mode=mode,
            visible=visible,
            target=target,
            B=1,
            H=1,
            Tq=1,
            Tk=64,
            dtype="float16",
            stages=[],
            scale=1e27 if name == "test_decode_extreme_scale_single_key" else None,
        )
    ]


def selected_device_items(items):
    return [item for item in items if item.get_closest_marker("npu") is not None or "npu_device" in item.fixturenames]


def cases_for_item(item):
    if Path(item.path).name == "test_gated_mla_decode.py":
        return legacy_cases_for_item(item)
    params = getattr(getattr(item, "callspec", None), "params", {})
    target = params.get("target") or "pto"
    dtype = str(params.get("dtype", "float16")).removeprefix("torch.")
    name = getattr(item, "originalname", None) or item.name.split("[", 1)[0]
    cases = []

    def case(suffix, B, H, Q, K, stages, *, layout="separate", fixed_dtype=None, fixed_target=None):
        cases.append(
            dict(
                id=item.nodeid + "/" + suffix,
                B=B,
                H=H,
                Tq=Q,
                Tk=K,
                dtype=fixed_dtype or dtype,
                target=fixed_target or target,
                layout=layout,
                stages=stages,
            )
        )

    all_stages = ["query", "attention", "value"]
    if name == "test_empty_query_and_zero_cache_outputs":
        case("empty_query_no_launch", 1, 96, 0, 17, [])
        case("empty_cache_zero", 1, 96, 1, 0, ["zero"])
    elif name == "test_actual_nz_rejected_in_separate_process":
        case("format_rejection_no_factory", 1, 96, 1, 1, [], fixed_dtype="float16", fixed_target="pto")
    elif name == "test_packed_h96_tail_prefill_with_selected_providers":
        case("append_prefill_read", 1, 96, 17, 17, all_stages + ["cache_init", "cache_append"], layout="packed")
        from gmla._packed import checked_scale

        # The packed entry retains its own exact host default (one ULP from
        # the separate forward default); scale is part of the JIT identity.
        cases[-1]["scale"] = checked_scale(None)
    elif name == "test_owner_two_appends_inactive_and_no_overwrite":
        case("two_runtime_lengths_same_factories", 2, 1, 17, 65, ["cache_init", "cache_append"])
    elif name == "test_h96_t17_tail_reset_full_chain":
        case("tail_reset", 1, 96, 17, 17, all_stages)
    elif name == "test_value_r3_range_mixed_heads":
        case(
            "R3_range_same_Value_factory", 1, 96, 17, 17, ["value"], layout="packed" if params["packed"] else "separate", fixed_target="pto"
        )
    elif name == "test_prefill128_public_profile_core":
        case("public_N128_exact_profile", 1, 96, 128, 128, all_stages, fixed_target="pto")
    elif name == "test_original_bf16_visible17_failure_and_metadata_reuse":
        case("full_and_visible17_same_factories", 1, 96, 1, 4096, all_stages, fixed_dtype="bfloat16")
    elif name == "test_empty_splits_inactive_batch_and_tails":
        case("empty_splits_and_partials_reuse", 2, 97, 1, 129, all_stages)
    elif name == "test_continuation_equals_actual_token_decode":
        case("continuation", 1, 4, 17, 81, all_stages, fixed_dtype="float16", fixed_target="pto")
        case("17_actual_tokens_same_decode_factories", 1, 4, 1, 81, all_stages, fixed_dtype="float16", fixed_target="pto")
    elif name == "test_prefill_127_128_129":
        case("extended_preserved_K129", 1, 96, params["tq"], 129, all_stages)
    elif name == "test_h96_full_chain_16k":
        case("extended_16k", 1, 96, 1, 16384, all_stages)
    elif name == "test_decode_custom_scale_routes_prefill":
        case("custom_scale", 1, 4, 1, 65, all_stages, fixed_dtype="bfloat16")
        cases[-1]["scale"] = (1 / math.sqrt(192)) * 0.75
    elif name == "test_direct_owner_rejects_actual_nz_before_commit_in_child":
        case("owner_initialization_only", 1, 1, 17, 65, ["cache_init"], fixed_dtype="float16", fixed_target="pto")
    else:
        raise pytest.UsageError(f"Unregistered selected NPU test; do not silently omit compilation: {item.nodeid}")
    return cases


def registry_from_items(all_items):
    items = selected_device_items(all_items)
    cases, coverage = [], []
    for item in items:
        planned = cases_for_item(item)
        cases.extend(planned)
        coverage.append(
            {
                "nodeid": item.nodeid,
                "source": str(item.path),
                "source_sha256": hashlib.sha256(Path(item.path).read_bytes()).hexdigest(),
                "case_ids": [case["id"] for case in planned],
            }
        )
    return {
        "schema_version": 1,
        "scope": "actual selected core pytest nodes, no NPU execution",
        "selected_npu_tests": len(items),
        "selected_total_tests": len(all_items),
        "cases": cases,
        "coverage": coverage,
        "legacy_roots": sorted({str(Path(item.path).parent) for item in items if Path(item.path).name == "test_gated_mla_decode.py"}),
        "registry_plugin_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }


def pytest_collection_finish(session):
    output = session.config.getoption("--gmla-registry-out")
    if not output:
        return
    report = registry_from_items(session.items)
    destination = Path(output)
    if destination.exists():
        raise pytest.UsageError("registry output must be new")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2))
