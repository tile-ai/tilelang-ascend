"""Host capability contracts; no TileLang, TVM, device or solver import is needed."""

from dataclasses import FrozenInstanceError, replace
import importlib.util
import json
from pathlib import Path
import sys
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[3]
MODEL = ROOT / "tilelang/transform/ascend_schedule_capabilities.py"
SPEC = importlib.util.spec_from_file_location("s2_capabilities_host", MODEL)
caps = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = caps
SPEC.loader.exec_module(caps)

EXPECTED_BUDGETS = {
    "shared.ub": ("V", 196352),
    "shared.l1": ("C", 524032),
    "wmma.matrix_a": ("C", 65536),
    "wmma.matrix_b": ("C", 65536),
    "wmma.accumulator": ("C", 131072),
}


def test_declared_profile_distinguishes_compiler_and_sdk_evidence():
    profile = caps.get_capabilities("a3")
    profile.validate()
    assert {pool.scope for pool in profile.pools} == set(EXPECTED_BUDGETS)
    for pool in profile.pools:
        assert (pool.owner, pool.budget_bytes) == EXPECTED_BUDGETS[pool.scope]
        assert pool.allocation_alignment_bytes == 32
    report = profile.to_dict()
    assert report["hardware_validated"] is False
    assert "not hardware measurement" in report["validation_basis"]
    assert {e["kind"] for e in report["evidence"]} >= {"compiler_contract", "sdk_declaration"}
    assert report["timing_model"] is None
    with pytest.raises(FrozenInstanceError):
        profile.platform = "other"


@pytest.mark.parametrize("scope", EXPECTED_BUDGETS)
def test_budget_boundary_rounding_and_all_versions(scope):
    profile = caps.get_capabilities()
    budget = EXPECTED_BUDGETS[scope][1]
    assert profile.validate_allocation(scope, budget - 32) == budget - 32
    assert profile.validate_allocation(scope, budget) == budget
    assert profile.validate_allocation(scope, 1) == 32
    assert profile.validate_allocation(scope, budget - 32, offset_bytes=32) == budget - 32
    assert profile.validate_allocation(scope, budget // 2, versions=2) == budget
    for size, offset, versions in ((budget + 1, 0, 1), (budget, 32, 1), (budget // 2 + 1, 0, 2)):
        with pytest.raises(caps.CapabilityError, match="exceeding compiler budget"):
            profile.validate_allocation(scope, size, offset, versions)


@pytest.mark.parametrize(
    "kwargs,message",
    [
        ({"size_bytes": 0}, "size_bytes"),
        ({"size_bytes": -32}, "size_bytes"),
        ({"size_bytes": True}, "size_bytes"),
        ({"size_bytes": 1.0}, "size_bytes"),
        ({"size_bytes": 32, "offset_bytes": -32}, "offset_bytes"),
        ({"size_bytes": 32, "offset_bytes": 16}, "aligned"),
        ({"size_bytes": 32, "versions": 0}, "versions"),
    ],
)
def test_invalid_physical_allocation_has_explicit_diagnostics(kwargs, message):
    with pytest.raises(caps.CapabilityError, match=message):
        caps.get_capabilities().validate_allocation("shared.ub", **kwargs)


def test_unknown_pool_and_target_do_not_gain_a_profile():
    profile = caps.get_capabilities()
    with pytest.raises(caps.CapabilityError, match="Unknown local memory pool"):
        profile.validate_allocation("global", 32)
    with pytest.raises(caps.CapabilityError, match="expected A3"):
        caps.get_capabilities("A5")


@pytest.mark.parametrize(
    "owner,src,dst,pipe",
    [
        ("V", "global", "shared.ub", "PIPE_MTE2"),
        ("V", "shared.ub", "shared.ub", "v"),
        ("V", "shared.ub", "global", "MTE3"),
    ],
)
def test_only_audited_vector_routes_are_accepted(owner, src, dst, pipe):
    caps.get_capabilities().validate_route(owner, src, dst, pipe)


@pytest.mark.parametrize(
    "owner,src,dst,pipe",
    [
        ("C", "global", "shared.ub", "MTE2"),
        ("V", "global", "shared.ub", "MTE3"),
        ("C", "global", "wmma.matrix_a", "MTE2"),
        ("V", "shared.ub", "shared.l1", "MTE3"),
        ("C", "wmma.accumulator", "shared.ub", "FIX"),
        ("V", "shared.ub", "global", "S"),
    ],
)
def test_unaudited_routes_are_rejected(owner, src, dst, pipe):
    with pytest.raises(caps.CapabilityError, match="unsupported"):
        caps.get_capabilities().validate_route(owner, src, dst, pipe)


def test_directed_event_domains_and_reserved_ids_are_independent():
    profile = caps.get_capabilities()
    assert len(profile.local_event_domains) == 6
    for domain in profile.local_event_domains:
        assert domain.declared_ids == tuple(range(8))
        assert profile.available_event_ids(domain.owner, domain.src_pipe, domain.dst_pipe) == tuple(range(8))
    assert profile.available_event_ids("V", "MTE2", "V", [0, 7, 0]) == tuple(range(1, 7))
    assert profile.available_event_ids("V", "V", "MTE2") == tuple(range(8))
    assert profile.available_event_ids("V", "MTE2", "V", range(8)) == ()
    profile.validate_event_id("V", "MTE2", "V", 7)
    with pytest.raises(caps.CapabilityError, match="reserved"):
        profile.validate_event_id("V", "MTE2", "V", 7, [7])


@pytest.mark.parametrize("reserved", [[-1], [8], [True], [1.5], None])
def test_reserved_event_ids_are_validated(reserved):
    with pytest.raises(caps.CapabilityError, match="reserved|Reserved"):
        caps.get_capabilities().available_event_ids("V", "MTE2", "V", reserved)


@pytest.mark.parametrize("owner,src,dst", [("C", "MTE2", "V"), ("V", "V", "V"), ("V", "S", "V")])
def test_unaudited_local_domains_are_not_silently_enabled(owner, src, dst):
    with pytest.raises(caps.CapabilityError, match="Unaudited"):
        caps.get_capabilities().available_event_ids(owner, src, dst)


def test_cross_core_encoding_is_not_an_allocation_budget():
    profile = caps.get_capabilities()
    with pytest.raises(caps.CapabilityError, match="Cross-core.*unaudited"):
        profile.validate_cross_core_event()
    assert profile.to_dict()["cross_core_events"]["supported"] is False
    assert profile.to_dict()["cross_core_events"]["allocatable_ids"] is None


@pytest.mark.parametrize("budget,alignment", [(-32, 32), (64, 3), (33, 32)])
def test_inconsistent_pool_description_is_rejected(budget, alignment):
    profile = caps.get_capabilities()
    invalid = replace(profile.pools[0], budget_bytes=budget, allocation_alignment_bytes=alignment)
    with pytest.raises(caps.CapabilityError):
        replace(profile, pools=(invalid, *profile.pools[1:])).validate()


def test_wrong_route_pipe_and_out_of_range_declared_ids_are_rejected():
    profile = caps.get_capabilities()
    with pytest.raises(caps.CapabilityError, match="owner/pipe"):
        replace(profile, routes=(caps.Route("V", "global", "shared.ub", "V"),)).validate()
    invalid = replace(profile.local_event_domains[0], declared_ids=(0, 8))
    with pytest.raises(caps.CapabilityError, match="IDs 0..7"):
        replace(profile, local_event_domains=(invalid,)).validate()


def test_missing_pool_and_wrong_cube_owner_are_rejected():
    profile = caps.get_capabilities()
    with pytest.raises(caps.CapabilityError, match="five known"):
        replace(profile, pools=profile.pools[:-1]).validate()
    invalid = replace(profile.pools[-1], owner="V")
    with pytest.raises(caps.CapabilityError, match="Invalid owner"):
        replace(profile, pools=(*profile.pools[:-1], invalid)).validate()


@pytest.mark.parametrize("timeout", ["nan", "inf", "-inf"])
def test_entry_rejects_nonfinite_timeouts_with_json(tmp_path, timeout):
    output = tmp_path / "invalid-timeout.json"
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/ascend_migration/s2_check_capabilities.py"),
            f"--timeout={timeout}",
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 2, result.stderr
    report = json.loads(output.read_text())
    assert report["step"] == "S2"
    assert report["status"] == "BLOCKED"
    assert report["counts"]["total"] == 0


def test_entry_preserves_json_when_check_process_cannot_start(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts/ascend_migration"))
    script = ROOT / "scripts/ascend_migration/s2_check_capabilities.py"
    spec = importlib.util.spec_from_file_location("s2_capabilities_oserror", script)
    entry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(entry)

    def cannot_start(*args, **kwargs):
        raise OSError("injected process-start failure")

    output = tmp_path / "process-start.json"
    monkeypatch.setattr(entry, "run_pytest", cannot_start)
    monkeypatch.setattr(sys, "argv", [str(script), "--output", str(output)])
    assert entry.main() == 2
    report = json.loads(output.read_text())
    assert report["status"] == "BLOCKED"
    assert "injected process-start failure" in report["error"]
