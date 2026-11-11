"""Verify a normal, non-editable GMLA installation against its supplied source."""

import argparse
import hashlib
import json
from pathlib import Path
import sys


def inventory(path):
    return {str(p.relative_to(path)).replace("\\", "/"): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(path.rglob("*.py"))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output must be new")
    source = (args.source_root / "gmla").resolve()
    # Intentionally no sys.path insertion; run from a neutral working directory.
    import gmla

    actual = Path(gmla.__file__).resolve().parent
    if actual == source:
        raise AssertionError("Imported source checkout instead of a non-editable installation")
    expected_hashes, actual_hashes = inventory(source), inventory(actual)
    if expected_hashes != actual_hashes:
        raise AssertionError("Installed package Python source bytes differ from selected source")
    report = dict(
        status="pass",
        installed_root=str(actual.parent),
        installed_module=str(actual),
        source_root=str(source.parent),
        sources=actual_hashes,
        executable=sys.executable,
        python=sys.version,
        scope="normal installed import/source identity; no TileLang compile or NPU initialization",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print("GMLA_INSTALLED_SOURCE_IDENTITY_PASS " + str(actual.parent))


if __name__ == "__main__":
    main()
