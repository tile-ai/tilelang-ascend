"""Host parsing contracts; these tests never provoke an NPU exception."""

import importlib
from pathlib import Path
import textwrap

import numpy as np
import pytest

parser = importlib.import_module("tilelang.tools.ascend.a2a3.ascend_exception_dump_bin")


def _dump(root, timestamp="200", kernel="main_kernel"):
    path = root / "extra-info" / "data-dump" / "0" / f"{kernel}.custom.{timestamp}"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("raw dump fixture")
    return path


def _tool(root, body):
    path = root / "tools" / "msaicerr" / "msaicerr.py"
    path.parent.mkdir(parents=True)
    path.write_text(textwrap.dedent(body))
    return path


def test_read_tensor_header_shape_and_owned_result(tmp_path):
    path = tmp_path / "tensor.bin"
    expected = np.array([[-7, 11], [23, 37]], dtype=np.int32)
    path.write_bytes(b"HEAD" + expected.tobytes())
    result = parser.read_msaicerr_bin(path, dtype=np.int32, shape=(2, 2), header_size=4)
    np.testing.assert_array_equal(result, expected)
    assert result.flags.writeable and result.flags.owndata


def test_parse_latest_dump_runs_sdk_tool_and_removes_stale_bins(tmp_path, monkeypatch):
    dump_root = tmp_path / "dump"
    _dump(dump_root, "100")
    latest = _dump(dump_root, "200")
    sdk = tmp_path / "sdk"
    _tool(
        sdk,
        """
        from pathlib import Path
        import struct
        import sys
        source = Path(sys.argv[sys.argv.index("-d") + 1])
        output = Path(sys.argv[sys.argv.index("-out") + 1])
        (output / "selected.txt").write_text(str(source))
        (output / (source.name + ".input.0.int32.bin")).write_bytes(struct.pack("4i", -7, 11, 23, 37))
        (output / (source.name + ".output.1.float32.bin")).write_bytes(struct.pack("2f", -7.5, 11.25))
        (output / (source.name + ".workspace.2.unknown.bin")).write_bytes(struct.pack("f", 23.5))
    """,
    )
    monkeypatch.setenv("ASCEND_HOME_PATH", str(sdk))
    output = dump_root / "parsed"
    output.mkdir()
    stale = output / "stale.bin"
    stale.write_bytes(b"stale")
    result = parser.parse_exception_dump(str(dump_root), wait_seconds=0)
    assert not stale.exists()
    assert (output / "selected.txt").read_text() == str(latest)
    assert [(entry["type"], entry["index"], entry["dtype"]) for entry in result] == [
        ("input", 0, "int32"),
        ("output", 1, "float32"),
        ("workspace", 2, "unknown"),
    ]
    np.testing.assert_array_equal(result[0]["data"], [-7, 11, 23, 37])
    np.testing.assert_array_equal(result[1]["data"], [-7.5, 11.25])
    # Preserve the original unknown-dtype float32 fallback.
    np.testing.assert_array_equal(result[2]["data"], [23.5])
    assert all(Path(entry["file"]).parent == output for entry in result)


def test_parse_missing_kernel_dump_preserves_error(tmp_path):
    _dump(tmp_path, kernel="other")
    with pytest.raises(FileNotFoundError, match="No exception dump file.*main_kernel"):
        parser.parse_exception_dump(str(tmp_path), wait_seconds=0)


def test_parse_missing_sdk_preserves_actionable_error(tmp_path, monkeypatch):
    _dump(tmp_path)
    monkeypatch.setenv("ASCEND_HOME_PATH", str(tmp_path / "missing-sdk"))
    with pytest.raises(RuntimeError, match="msaicerr.py not found.*ASCEND_HOME_PATH"):
        parser.parse_exception_dump(str(tmp_path), wait_seconds=0)


def test_parse_sdk_failure_preserves_exit_and_stderr(tmp_path, monkeypatch):
    dump_root = tmp_path / "dump"
    _dump(dump_root)
    sdk = tmp_path / "sdk"
    _tool(sdk, 'import sys\nprint("invalid fixture", file=sys.stderr)\nraise SystemExit(7)\n')
    monkeypatch.setenv("ASCEND_HOME_PATH", str(sdk))
    with pytest.raises(RuntimeError, match=r"msaicerr.py failed \(exit 7\):\ninvalid fixture"):
        parser.parse_exception_dump(str(dump_root), output_dir=str(tmp_path / "output"), wait_seconds=0)
