#!/usr/bin/env python3
"""Validate and query the operator-to-Pytest migration manifest."""

from __future__ import annotations

import argparse
import re
import sys
from collections.abc import Iterable, Sequence
from pathlib import Path, PurePosixPath


DEFAULT_MANIFEST = Path("ci/operator_test_manifest.yaml")
VERSION_PATTERN = re.compile(r"^version:\s*(\d+)\s*$")
# Roots scanned for operator tests that no runner would execute.
EXAMPLE_ROOTS = ("examples", "examples_experiment")
# A shell script line that starts a Python process, and the .py files named on it.
RUNNER_PATTERN = re.compile(r"\b(?:python[0-9.]*|pytest)\b")
PY_FILE_PATTERN = re.compile(r"[\w./-]+\.py\b")


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
            raise ManifestError(f"{manifest_path}:{line_number}: expected an indented source: test mapping")

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


def _shell_invoked_tests(repo_root: Path) -> set[str]:
    """Collect example tests that a shell script under the example roots runs.

    The legacy runner auto-discovers example entry points, but a test_*.py that
    a sibling .sh script calls explicitly still executes and is therefore not
    orphaned. Paths are resolved against the script directory because these
    scripts cd into their own directory before invoking Python.
    """
    invoked: set[str] = set()
    for root in EXAMPLE_ROOTS:
        root_path = repo_root / root
        if not root_path.is_dir():
            continue
        for script in root_path.rglob("*.sh"):
            try:
                text = script.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            for line in text.splitlines():
                if line.lstrip().startswith("#") or not RUNNER_PATTERN.search(line):
                    continue
                for token in PY_FILE_PATTERN.findall(line):
                    target = (script.parent / token).resolve()
                    if target.is_file():
                        invoked.add(target.as_posix())
    return invoked


def find_orphan_tests(repo_root: Path, mappings: list[tuple[str, str]]) -> list[str]:
    """List test_*.py files under the example roots that no runner would execute.

    The legacy runner excludes test_*.py by name and Pytest only collects the
    manifest targets, so an unregistered file that no shell script invokes never
    runs at all while CI still reports success.
    """
    registered = {test for _, test in mappings}
    invoked = _shell_invoked_tests(repo_root)
    orphans: list[str] = []
    for root in EXAMPLE_ROOTS:
        root_path = repo_root / root
        if not root_path.is_dir():
            continue
        for path in sorted(root_path.rglob("test_*.py")):
            relative = path.relative_to(repo_root).as_posix()
            if relative in registered or path.resolve().as_posix() in invoked:
                continue
            orphans.append(relative)
    return orphans


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
    subparsers.add_parser(
        "check-orphans",
        help="fail if an example test_*.py is run by neither the manifest nor a shell script",
    )
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
    elif args.command == "check-orphans":
        orphans = find_orphan_tests(args.repo_root.resolve(), mappings)
        if orphans:
            print("unreachable operator tests found:", file=sys.stderr)
            for orphan in orphans:
                print(f"  {orphan}", file=sys.stderr)
            print(
                "each one is skipped by the legacy runner and not collected by "
                "Pytest; add it to ci/operator_test_manifest.yaml, invoke it from "
                "the example shell script, or rename it so test_*.py no longer matches",
                file=sys.stderr,
            )
            return 1
        print("No unreachable operator tests found")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
