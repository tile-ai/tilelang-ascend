#!/usr/bin/env python3
"""Validate and query the operator-to-Pytest migration manifest."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path, PurePosixPath
from typing import Iterable, Sequence


DEFAULT_MANIFEST = Path("ci/operator_test_manifest.yaml")
VERSION_PATTERN = re.compile(r"^version:\s*(\d+)\s*$")


class ManifestError(ValueError):
    """Raised when the operator test manifest is invalid."""


def _parse_manifest(manifest_path: Path) -> list[tuple[str, str]]:
    version: int | None = None
    in_mappings = False
    mappings: list[tuple[str, str]] = []

    for line_number, raw_line in enumerate(
        manifest_path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        version_match = VERSION_PATTERN.fullmatch(stripped)
        if version_match:
            if version is not None:
                raise ManifestError(f"{manifest_path}:{line_number}: duplicate version")
            version = int(version_match.group(1))
            continue

        if stripped == "mappings:":
            if in_mappings:
                raise ManifestError(f"{manifest_path}:{line_number}: duplicate mappings section")
            in_mappings = True
            continue

        if not in_mappings or not raw_line.startswith("  ") or ":" not in stripped:
            raise ManifestError(
                f"{manifest_path}:{line_number}: expected an indented source: test mapping"
            )

        source, test = (part.strip() for part in stripped.split(":", maxsplit=1))
        if not source or not test:
            raise ManifestError(f"{manifest_path}:{line_number}: source and test are required")
        mappings.append((source, test))

    if version != 1:
        raise ManifestError(f"{manifest_path}: unsupported or missing version: {version}")
    if not in_mappings:
        raise ManifestError(f"{manifest_path}: missing mappings section")
    if not mappings:
        raise ManifestError(f"{manifest_path}: mappings must not be empty")
    return mappings


def _validate_relative_path(value: str, field: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or str(path) != value:
        raise ManifestError(f"{field} must be a normalized repo-relative POSIX path: {value}")
    return path


def load_mappings(repo_root: Path, manifest_path: Path) -> list[tuple[str, str]]:
    if not manifest_path.is_absolute():
        manifest_path = repo_root / manifest_path
    if not manifest_path.is_file():
        raise ManifestError(f"manifest does not exist: {manifest_path}")

    mappings = _parse_manifest(manifest_path)
    seen_sources: set[str] = set()
    seen_tests: set[str] = set()

    for source, test in mappings:
        source_path = _validate_relative_path(source, "source")
        test_path = _validate_relative_path(test, "test")

        if source in seen_sources:
            raise ManifestError(f"duplicate source: {source}")
        if test in seen_tests:
            raise ManifestError(f"duplicate test: {test}")
        if source == test:
            raise ManifestError(f"source and test must be different: {source}")
        if source_path.suffix != ".py":
            raise ManifestError(f"source must be a Python file: {source}")
        if test_path.suffix != ".py" or not test_path.name.startswith("test_"):
            raise ManifestError(f"test must be named test_*.py: {test}")
        if not (repo_root / Path(*source_path.parts)).is_file():
            raise ManifestError(f"source does not exist: {source}")
        if not (repo_root / Path(*test_path.parts)).is_file():
            raise ManifestError(f"test does not exist: {test}")

        seen_sources.add(source)
        seen_tests.add(test)

    return mappings


def _print_lines(values: Iterable[str]) -> None:
    for value in values:
        print(value)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="repository root (defaults to the resolver's repository)",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help="manifest path, relative to the repository root by default",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("validate", help="validate the manifest and referenced files")

    list_parser = subparsers.add_parser("list", help="list source-to-test mappings")
    list_parser.add_argument("--format", choices=("tsv",), default="tsv")

    subparsers.add_parser("list-tests", help="list all migrated Pytest files")
    subparsers.add_parser("list-sources", help="list all sources excluded from the legacy runner")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        mappings = load_mappings(args.repo_root.resolve(), args.manifest)
    except (ManifestError, OSError) as error:
        print(f"operator test manifest error: {error}", file=sys.stderr)
        return 1

    if args.command == "validate":
        print(f"Validated {len(mappings)} operator test mapping(s)")
    elif args.command == "list":
        _print_lines(f"{source}\t{test}" for source, test in mappings)
    elif args.command == "list-tests":
        _print_lines(test for _, test in mappings)
    elif args.command == "list-sources":
        _print_lines(source for source, _ in mappings)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
