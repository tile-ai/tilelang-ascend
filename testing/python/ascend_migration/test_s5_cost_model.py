"""Pure-host artifact checks with explicitly synthetic measurement fixtures."""

from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
_spec = importlib.util.spec_from_file_location("s5_cost_model_host", ROOT / "tilelang/transform/ascend_pipeline_cost.py")
cost = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cost)


def synthetic_profile():
    """No device evidence: these numbers exercise only the artifact protocol."""
    profile = {
        "schema_version": 1,
        "model": cost.MODEL,
        "unit": "ns",
        "quantum_ns": 32,
        "identity": {
            "platform": "A3",
            "soc": "synthetic-unit-test",
            "cann_version": "synthetic",
            "bisheng_sha256": "a" * 64,
            "flags": ["-O2", "-mllvm", "-cce-auto-sync=false"],
        },
        "records": [
            {
                "kind": "add",
                "dtype": "float32",
                "elements": 64,
                "bytes": 256,
                "rows": 1,
                "latency_effective_ns": 95,
                "ii_effective_ns": 63,
                "latency": 3,
                "ii": 2,
            }
        ],
        "evidence": {
            "status": "PASS",
            "method": "synthetic-unit-fixture",
            "raw_samples": [{"n": 8, "elapsed_ns": 1234}],
        },
    }
    return seal(profile)


def seal(profile):
    profile["profile_sha256"] = cost.cost_model_digest(profile)
    return profile


def test_canonical_profile_is_independent_of_mapping_order_and_copied():
    profile = synthetic_profile()
    snapshot = deepcopy(profile)
    checked = cost.validate_cost_model(dict(reversed(list(profile.items()))))
    assert checked == profile
    checked["records"][0]["latency"] = 99
    assert profile == snapshot
    assert cost.cost_model_digest(profile) == profile["profile_sha256"]


def test_same_path_updates_load_contents_instead_of_using_a_path_identity(tmp_path):
    path = tmp_path / "cost.json"
    first = synthetic_profile()
    path.write_text(json.dumps(first))
    loaded = cost.load_cost_model(path)
    second = deepcopy(first)
    second["records"][0]["latency_effective_ns"] = 128
    second["records"][0]["latency"] = 4
    seal(second)
    path.write_text(json.dumps(second))
    reloaded = cost.load_cost_model(path)
    assert loaded["profile_sha256"] != reloaded["profile_sha256"]
    assert loaded["records"] != reloaded["records"]


@pytest.mark.parametrize("tamper", ["sample", "flags", "cost"])
def test_artifact_digest_covers_actual_samples_flags_and_costs(tamper):
    profile = synthetic_profile()
    if tamper == "sample":
        profile["evidence"]["raw_samples"][0]["elapsed_ns"] += 1
    elif tamper == "flags":
        profile["identity"]["flags"].reverse()
    else:
        profile["records"][0]["ii_effective_ns"] -= 1
    with pytest.raises(cost.CostModelError, match="profile_sha256"):
        cost.validate_cost_model(profile)


@pytest.mark.parametrize("field,value", [("soc", "other-device"), ("flags", ["-O3"])])
def test_expected_calibration_identity_is_checked(field, value):
    with pytest.raises(cost.CostModelError, match="identity mismatch"):
        cost.validate_cost_model(synthetic_profile(), expected_identity={field: value})


@pytest.mark.parametrize(
    "field,value",
    [
        ("elements", True),
        ("elements", 0),
        ("bytes", 255),
        ("rows", 2),
        ("kind", "unmodeled"),
        ("kind", []),
        ("dtype", "int32"),
        ("latency", 2),
        ("ii", 0),
    ],
)
def test_invalid_record_is_rejected_even_with_a_matching_digest(field, value):
    profile = synthetic_profile()
    profile["records"][0][field] = value
    seal(profile)
    with pytest.raises(cost.CostModelError):
        cost.validate_cost_model(profile)


def test_duplicate_geometry_cannot_ambiguously_select_a_cost():
    profile = synthetic_profile()
    profile["records"].append(deepcopy(profile["records"][0]))
    seal(profile)
    with pytest.raises(cost.CostModelError, match="duplicate"):
        cost.validate_cost_model(profile)


@pytest.mark.parametrize("value", [None, 1.0, float("nan"), 1 << 63])
def test_non_lossless_evidence_is_rejected(value):
    profile = synthetic_profile()
    profile["evidence"]["extra"] = value
    with pytest.raises(cost.CostModelError):
        cost.cost_model_digest(profile)


@pytest.mark.parametrize("field,value", [("status", "BLOCKED"), ("raw_samples", [])])
def test_failed_or_missing_measurement_evidence_cannot_enable_pipeline(field, value):
    profile = synthetic_profile()
    profile["evidence"][field] = value
    seal(profile)
    with pytest.raises(cost.CostModelError, match="evidence"):
        cost.validate_cost_model(profile)


def test_filename_is_not_a_cost_model_configuration():
    with pytest.raises(cost.CostModelError, match="mapping"):
        cost.validate_cost_model("cost.json")
