"""Executable per-suite pytest entry; package-local configuration, no import-only pass."""

from pathlib import Path
import sys


def main(test_file):
    import pytest

    package = Path(__file__).resolve().parent.parent
    test = package / "tests" / test_file
    if not test.is_file():
        raise FileNotFoundError(test)
    return pytest.main(["-c", str(package / "pyproject.toml"), str(test), *sys.argv[1:]])
