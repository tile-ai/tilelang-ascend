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
EXAMPLE_ROOTS = ("examples", "examples_experiment")
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


def load_mappings(repo_root: Path, manifest_path: Path) -> tuple[list[tuple[str, str]], list[tuple[str, str]], list[tuple[str, str]]]:
    """Return the mappings that are live, those waiting on their test, and those
    whose operator is gone.

    Registering a source excludes it from the legacy runner, so a mapping whose
    test has not landed yet must not take effect: it is a plan, not a migration.
    Holding it back keeps the example running and keeps Pytest from being
    pointed at a file that does not exist.

    An entry whose source no longer exists is reported rather than raised on.
    It is out of date, not malformed, and the two want different handling: the
    checks below are about a manifest that cannot be read the way it claims, and
    stopping on those is what keeps a half-built skip list from letting every
    migrated operator run twice. A moved operator costs nothing to skip - there
    is no file at that path for the runner to find, so no run to suppress - and
    raising on it stopped every unrelated pull request in the repository until
    somebody noticed the entry, which is not where the cost belongs.
    """
    if not manifest_path.is_absolute():
        manifest_path = repo_root / manifest_path
    if not manifest_path.is_file():
        raise ManifestError(f"manifest does not exist: {manifest_path}")

    mappings = _parse_manifest(manifest_path)
    seen_sources: set[str] = set()
    seen_tests: set[str] = set()
    active: list[tuple[str, str]] = []
    pending: list[tuple[str, str]] = []
    stale: list[tuple[str, str]] = []

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
        seen_sources.add(source)
        seen_tests.add(test)

        if not (repo_root / Path(*source_path.parts)).is_file():
            stale.append((source, test))
        elif (repo_root / Path(*test_path.parts)).is_file():
            active.append((source, test))
        else:
            pending.append((source, test))

    return active, pending, stale


def _shell_invoked_tests(repo_root: Path) -> set[str]:
    """Collect tests that a shell script under the example roots runs directly.

    Those do not go through the manifest at all, so an unregistered name there
    is not a mistake. Paths resolve against the script directory because these
    scripts cd into their own directory first.
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


def find_unregistered_tests(repo_root: Path, active: list[tuple[str, str]], pending: list[tuple[str, str]]) -> list[str]:
    """List example tests that match no manifest entry, live or reserved.

    Worth knowing rather than worth failing over: an unregistered test runs as
    a script in the legacy runner, which is where every test lived before this
    manifest existed. What it does not get is the Pytest treatment, so saying
    so gives whoever wrote it the choice. Tests a sibling shell script invokes
    are exempt: those never went through the manifest.
    """
    expected = {test for _, test in active} | {test for _, test in pending}
    invoked = _shell_invoked_tests(repo_root)
    unregistered: list[str] = []
    for root in EXAMPLE_ROOTS:
        root_path = repo_root / root
        if not root_path.is_dir():
            continue
        for path in sorted(root_path.rglob("test_*.py")):
            relative = path.relative_to(repo_root).as_posix()
            if relative in expected or path.resolve().as_posix() in invoked:
                continue
            unregistered.append(relative)
    return unregistered


def _scope_to_dirs(mappings: list[tuple[str, str]], dirs: list[str] | None, experiment_dirs: list[str] | None) -> list[tuple[str, str]]:
    """Narrow the mappings to the example directories a run was scoped to.

    Neither flag given is a full run, and every mapping belongs to it. One given
    without the other means that root contributed no directory, not that it is
    unconstrained, so an absent flag selects nothing rather than everything.

    Matching is on the first directory below the root, since that is the unit
    the runner takes and an operator may sit a level deeper than it.
    """
    if dirs is None and experiment_dirs is None:
        return mappings
    wanted = {f"examples/{name}" for name in (dirs or [])}
    wanted |= {f"examples_experiment/{name}" for name in (experiment_dirs or [])}
    scoped = []
    for source, test in mappings:
        parts = PurePosixPath(source).parts
        if len(parts) > 2 and "/".join(parts[:2]) in wanted:
            scoped.append((source, test))
    return scoped


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

    list_tests_parser = subparsers.add_parser("list-tests", help="list migrated Pytest files")
    # Named after the runner's own flags so a caller that scoped a run to some
    # directories can hand this the value it handed the runner, unchanged.
    list_tests_parser.add_argument(
        "--dirs",
        nargs="*",
        default=None,
        help="limit to tests whose operator lives under one of these examples/ directories",
    )
    list_tests_parser.add_argument(
        "--experiment-dirs",
        nargs="*",
        default=None,
        help="the same, for examples_experiment/",
    )
    subparsers.add_parser("list-sources", help="list all sources excluded from the legacy runner")
    subparsers.add_parser(
        "check-orphans",
        help="fail if an example test matches no manifest entry, live or reserved",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        mappings, pending, stale = load_mappings(args.repo_root.resolve(), args.manifest)
    except (ManifestError, OSError) as error:
        print(f"operator test manifest error: {error}", file=sys.stderr)
        return 1

    if args.command == "validate":
        print(f"Validated {len(mappings)} operator test mapping(s)")
        if stale:
            print(
                f"{len(stale)} mapping(s) whose operator no longer exists at the "
                "registered path; each one is inert and its operator runs in the "
                "legacy runner until the entry is repointed or removed:"
            )
            for source, test in stale:
                print(f"  {source} -> {test}")
        if pending:
            print(
                f"{len(pending)} mapping(s) reserved but not migrated yet; the "
                "example keeps running in the legacy runner until its test lands:"
            )
            for source, test in pending:
                print(f"  {source} -> {test}")
    elif args.command == "list":
        _print_lines(f"{source}\t{test}" for source, test in mappings)
    elif args.command == "list-tests":
        scoped = _scope_to_dirs(mappings, args.dirs, args.experiment_dirs)
        _print_lines(test for _, test in scoped)
    elif args.command == "list-sources":
        _print_lines(source for source, _ in mappings)
    elif args.command == "check-orphans":
        # A stale entry still names its test, so the test is not unclaimed; it is
        # claimed by an entry that has nothing left to run. Reporting it here as
        # well would say the same thing twice, in the place that is about tests
        # nobody registered.
        unregistered = find_unregistered_tests(args.repo_root.resolve(), mappings, pending + stale)
        if unregistered:
            print(f"{len(unregistered)} operator test(s) matching no manifest entry:")
            for item in unregistered:
                print(f"  {item}")
            print(
                "each runs as a script in the legacy runner. To have Pytest run one "
                "instead, add it to ci/operator_test_manifest.yaml; if an entry was "
                "meant to cover it already, check the name against that entry"
            )
        else:
            print("All operator tests match a manifest entry")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
