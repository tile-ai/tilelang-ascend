"""TA CI paths before and after the controlled A2/A3 ecosystem adoption."""

from __future__ import annotations

import argparse
from pathlib import Path, PurePosixPath

NAMESPACE = "ascend_a2a3"
ECOSYSTEM_ROOTS = ("examples", "examples_experiment", "testing")


class LayoutError(ValueError):
    """A main compiler checkout has not received its TA ecosystem paths yet."""


def adopted_layout(repo_root: Path) -> bool:
    """Never mistake a main CUDA file at an old TA name for the migrated file."""
    if (repo_root / "examples" / NAMESPACE).is_dir():
        return True
    if (repo_root / "tilelang/ascend/a2a3/backend.py").is_file():
        raise LayoutError(
            "A2/A3 compiler detected without examples/ascend_a2a3; complete the "
            "controlled ecosystem adoption before running TA CI (no legacy-path fallback)."
        )
    return False


def canonical_path(value: str) -> str:
    parts = PurePosixPath(value).parts
    if len(parts) > 1 and parts[0] in ECOSYSTEM_ROOTS and parts[1] == NAMESPACE:
        return PurePosixPath(parts[0], *parts[2:]).as_posix()
    return value


def resolve_path(repo_root: Path, value: str) -> str:
    """Resolve a registered TA path in one layout, without per-file fallback."""
    adopted = adopted_layout(repo_root)
    canonical = canonical_path(value)
    parts = PurePosixPath(canonical).parts
    if not adopted or not parts:
        return canonical
    if parts[0] == "examples":
        return PurePosixPath("examples", NAMESPACE, *parts[1:]).as_posix()
    if parts[0] == "examples_experiment":
        # The user-selected final layout retains the original experiment root.
        return canonical
    if parts[0] == "testing":
        if len(parts) > 1 and parts[1] == "ascend":
            if is_component_asset(canonical):
                return PurePosixPath("testing", NAMESPACE, "python", "refactor", *parts[2:]).as_posix()
            raise LayoutError("Main testing/ascend is outside the adopted TA CI scope; select its owned A2A3 root instead")
        return PurePosixPath("testing", NAMESPACE, *parts[1:]).as_posix()
    return canonical


def example_roots(repo_root: Path) -> tuple[str, str]:
    return tuple(resolve_path(repo_root, name) for name in ECOSYSTEM_ROOTS[:2])


def pytest_targets(repo_root: Path) -> list[str]:
    """Collect TA Python tests and relocated components through one owned root."""
    if not adopted_layout(repo_root):
        return ["testing/python"] if (repo_root / "testing/python").is_dir() else []
    target = "testing/ascend_a2a3/python"
    if not (repo_root / target).is_dir() or not (repo_root / target / "refactor").is_dir():
        raise LayoutError(
            "Incomplete A2/A3 test layout: require testing/ascend_a2a3/python "
            "including its refactor component subtree; main testing/ascend is not a fallback."
        )
    return [target]


def is_component_asset(value: str) -> bool:
    """A2/A3 component tests and their named helpers; never main A5 assets."""
    path = PurePosixPath(value)
    return value.startswith("testing/ascend/") and (
        path.name.startswith(("test_a2a3", "_a2a3", "a2a3_")) or value == "testing/ascend/target/test_tilelang_a2a3_target.py"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--path")
    parser.add_argument("--pytest-targets", action="store_true")
    args = parser.parse_args()
    root = args.repo_root.resolve()
    if args.pytest_targets:
        targets = pytest_targets(root)
        if not targets:
            raise LayoutError("No owned TA Pytest targets found")
        print("\n".join(targets))
    elif args.path:
        path = resolve_path(root, args.path)
        if not (root / path).exists():
            raise LayoutError(f"Required TA CI path is missing: {path}; no legacy-path fallback")
        print(path)
    else:
        parser.error("provide --path or --pytest-targets")


if __name__ == "__main__":
    main()
