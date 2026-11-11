"""CPU-only tests of measurement boundaries, failure stops and provenance guards."""

import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from ._perf_tools import load_perf_tool


_perf_tool = load_perf_tool("host_decode_diagnostic")
run_host_profile = _perf_tool.run_host_profile
verify_install = _perf_tool.verify_install
verify_protocol = _perf_tool.verify_protocol


def test_profile_sync_is_outside_and_forward_is_not_replaced(tmp_path):
    calls = {"forward": 0, "sync": 0}
    result = object()

    def synchronize():
        assert sys.getprofile() is None
        calls["sync"] += 1

    def forward():
        calls["forward"] += 1
        return result

    report, output = run_host_profile(forward, synchronize, calls=3, output_dir=tmp_path)
    assert output is result
    assert calls == {"forward": 3, "sync": 6}
    assert report["completed_calls"] == 3
    row = next(row for row in report["top_inclusive"] if row["function"] == "forward")
    assert row["total_calls"] == 3
    assert row["calls_per_public_forward"] == 1
    all_rows = json.loads((tmp_path / "host_profile_functions.json").read_text())["rows"]
    assert all(row["function"] != "synchronize" for row in all_rows)
    assert (tmp_path / "host_profile.pstats").is_file()
    assert "Event-minus-graph" in report["interpretation"]


def test_failure_disables_profiler_and_does_not_drain_or_retry(tmp_path):
    operations = []

    def synchronize():
        operations.append("sync")

    def forward():
        operations.append("forward")
        raise RuntimeError("aicore test sentinel")

    with pytest.raises(RuntimeError, match="aicore test sentinel"):
        run_host_profile(forward, synchronize, calls=3, output_dir=tmp_path)
    assert operations == ["sync", "forward"]
    assert sys.getprofile() is None
    partial = json.loads((tmp_path / "host_profile_functions.json").read_text())
    assert partial["completed_calls"] == 0
    assert partial["profiled_attempts"] == 1


def test_profile_invalid_count_never_calls_payload(tmp_path):
    def forbidden():
        pytest.fail("invalid count reached payload")

    with pytest.raises(ValueError, match="positive"):
        run_host_profile(forbidden, forbidden, calls=0, output_dir=tmp_path)


def test_initial_sync_failure_preserves_original_error(tmp_path):
    def synchronize():
        raise RuntimeError("device unavailable sentinel")

    def forbidden():
        pytest.fail("failed sync reached payload")

    with pytest.raises(RuntimeError, match="device unavailable sentinel"):
        run_host_profile(forbidden, synchronize, calls=3, output_dir=tmp_path)
    partial = json.loads((tmp_path / "host_profile_functions.json").read_text())
    assert partial["profiled_attempts"] == 0
    assert partial["rows"] == []


def test_protocol_tamper_is_rejected_before_loading(tmp_path):
    (tmp_path / "fixture.py").write_text("raise RuntimeError('must not import')\n")
    (tmp_path / "protocol_manifest.json").write_text(json.dumps({"files": {"fixture.py": {"extracted_sha256": "0" * 64}}}))
    with pytest.raises(RuntimeError, match="differs"):
        verify_protocol(tmp_path)


def test_wrong_install_root_rejected(tmp_path):
    module = SimpleNamespace(__file__=str(tmp_path / "other" / "gmla" / "__init__.py"))
    with pytest.raises(RuntimeError, match="normally imported"):
        verify_install(module, tmp_path / "site", tmp_path / "unused.json")


def test_editable_install_rejected(tmp_path):
    dist = tmp_path / "gated_mla_ascend_candidate-0.1.0a0.dist-info"
    dist.mkdir()
    (dist / "RECORD").write_text("")
    (dist / "direct_url.json").write_text(json.dumps({"dir_info": {"editable": True}}))
    module = SimpleNamespace(__file__=str(tmp_path / "gmla" / "__init__.py"))
    with pytest.raises(RuntimeError, match="Editable"):
        verify_install(module, tmp_path, Path("unused.json"))
