import pathlib

import pytest
import torch

from testcommon import assert_close
from tilelang.utils import prec_assert_close


pytestmark = [
    pytest.mark.op("precision_debug"),
    pytest.mark.mode("Developer"),
]


def _single_run_dir(root: pathlib.Path) -> pathlib.Path:
    runs = [path for path in root.iterdir() if path.is_dir()]
    assert len(runs) == 1
    return runs[0]


def test_prec_assert_close_exported_from_tilelang_utils():
    from tilelang.utils.precision_debug import prec_assert_close as direct_import

    assert prec_assert_close is direct_import


def test_assert_close_without_debug_does_not_write_reports(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.chdir(tmp_path)
    actual = torch.tensor([0.0, 1.0], dtype=torch.float32)
    expected = torch.tensor([0.0, 0.0], dtype=torch.float32)

    with pytest.raises(AssertionError):
        assert_close(actual, expected, dtype="float32")

    assert not (tmp_path / ".precision_debug").exists()


@pytest.mark.precision_debug
def test_precision_debug_marker_enables_report_output(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.chdir(tmp_path)
    actual = torch.tensor([0.0, 1.0], dtype=torch.float32)
    expected = torch.tensor([0.0, 0.0], dtype=torch.float32)

    with pytest.raises(AssertionError):
        assert_close(actual, expected, dtype="float32")

    run_dir = _single_run_dir(tmp_path / ".precision_debug")
    assert (run_dir / "report.txt").is_file()
    assert (run_dir / "diff_map.txt").is_file()
    assert (run_dir / "actual.pt").is_file()
    assert (run_dir / "expected.pt").is_file()


def test_prec_assert_close_respects_equal_nan_in_report(
    tmp_path: pathlib.Path,
):
    output_dir = tmp_path / "precision_debug_manual"
    actual = torch.tensor([float("nan")], dtype=torch.float32)
    expected = torch.tensor([float("nan")], dtype=torch.float32)

    with pytest.raises(AssertionError):
        prec_assert_close(
            actual,
            expected,
            output_dir=str(output_dir),
            save_tensors=False,
            print_map=False,
            equal_nan=False,
        )

    report_text = (_single_run_dir(output_dir) / "report.txt").read_text(
        encoding="utf-8"
    )
    assert "Mismatched: 1 (100.0000%)" in report_text
