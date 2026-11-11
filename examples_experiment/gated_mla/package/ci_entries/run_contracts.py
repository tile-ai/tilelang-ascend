"""Execute test_contracts.py through pytest, preserving arguments and exit status."""

from _run import main


if __name__ == "__main__":
    raise SystemExit(main("test_contracts.py"))
