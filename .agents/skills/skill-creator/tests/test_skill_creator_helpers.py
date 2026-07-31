"""Regression checks for the skill-creator helper scripts.

Covers the failure modes reported in issue #1376, which have to be verified
independently so that fixing one does not mask the others:

1. Every helper must parse as Python 3.9, the repository's declared target.
2. PEP 604 unions (``X | Y``) in evaluated positions break Python 3.9 even
   though they parse. Checked structurally rather than by running a 3.9
   interpreter, so the guard holds on any host version.
3. The optional ``anthropic`` SDK must not be imported at module load, or
   ``--help`` fails in environments without it.
4. A missing SDK must surface as a prerequisite message, not an import-time
   traceback, and must never trigger an install.

Run from the skill-creator directory::

    python -m pytest tests/
"""

import ast
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

# Resolved relative to this file so the suite travels with the skill and keeps
# working if the skill is relocated or synced elsewhere.
SKILL_DIR = Path(__file__).resolve().parents[1]

HELPERS = sorted(SKILL_DIR.glob("scripts/*.py")) + sorted(SKILL_DIR.glob("eval-viewer/*.py"))

# Helpers that expose an argparse CLI and may reach the anthropic SDK.
SDK_MODULES = ["scripts.improve_description", "scripts.run_loop"]

# Executed in a subprocess to make `anthropic` unimportable even when the SDK
# is installed on the host, so the test measures import timing rather than the
# environment it happens to run in.
_BLOCK_ANTHROPIC = """
import sys


class _BlockAnthropic:

    def find_spec(self, fullname, path=None, target=None):
        if fullname == "anthropic" or fullname.startswith("anthropic."):
            raise ImportError("anthropic is blocked for this test")
        return None


sys.meta_path.insert(0, _BlockAnthropic())
"""


def _run_without_anthropic(body: str) -> subprocess.CompletedProcess:
    """Run `body` in a subprocess where importing anthropic always fails."""
    return subprocess.run(
        [sys.executable, "-c", _BLOCK_ANTHROPIC + textwrap.dedent(body)],
        cwd=SKILL_DIR,
        capture_output=True,
        text=True,
        timeout=60,
    )


def test_helpers_are_discovered():
    """Guard against the glob silently matching nothing after a move."""
    assert len(HELPERS) == 9, [p.name for p in HELPERS]


@pytest.mark.parametrize("path", HELPERS, ids=lambda p: p.name)
def test_helpers_parse_under_python39_grammar(path: Path):
    """Helpers must parse under the repository's declared Python target.

    The CI format check runs an unpinned `ruff format`, so a future ruff
    release could emit syntax that is valid only on newer interpreters.
    `feature_version` pins the grammar without needing a 3.9 interpreter on
    the runner.

    Best-effort by nature: `feature_version` only rejects constructs CPython
    version-gates in its parser (`match` statements, for instance), and it
    cannot reject syntax newer than the interpreter running the tests. It is
    a cheap backstop, not a substitute for a real 3.9 build.

    Syntax-level only; runtime-evaluated constructs such as PEP 604 unions
    parse fine here and are covered by the test below.
    """
    try:
        ast.parse(path.read_text(), filename=str(path), feature_version=(3, 9))
    except SyntaxError as exc:
        pytest.fail(f"{path.relative_to(SKILL_DIR)} does not parse as Python 3.9: {exc}")


@pytest.mark.parametrize("path", HELPERS, ids=lambda p: p.name)
def test_pep604_unions_require_postponed_annotations(path: Path):
    """`X | Y` annotations need `from __future__ import annotations` on py3.9.

    Signature annotations are evaluated at definition time, so on Python 3.9
    a bare PEP 604 union raises TypeError at import rather than only failing
    lint. The future import defers the evaluation and makes it safe.
    """
    tree = ast.parse(path.read_text(), filename=str(path))

    annotations = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.AnnAssign, ast.arg)) and node.annotation is not None:
            annotations.append(node.annotation)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.returns is not None:
            annotations.append(node.returns)

    uses_pep604 = any(
        isinstance(child, ast.BinOp) and isinstance(child.op, ast.BitOr) for annotation in annotations for child in ast.walk(annotation)
    )

    if not uses_pep604:
        return

    has_future = any(
        isinstance(node, ast.ImportFrom) and node.module == "__future__" and any(alias.name == "annotations" for alias in node.names)
        for node in tree.body
    )
    assert has_future, (
        f"{path.relative_to(SKILL_DIR)} uses PEP 604 unions in evaluated "
        f"annotations but lacks `from __future__ import annotations`, so it "
        f"fails to import on Python 3.9"
    )


@pytest.mark.parametrize("path", HELPERS, ids=lambda p: p.name)
def test_no_unconditional_anthropic_import(path: Path):
    """The optional SDK may only be imported lazily or under TYPE_CHECKING."""
    tree = ast.parse(path.read_text(), filename=str(path))

    offenders = []
    for node in tree.body:  # module level only; nested imports are lazy by definition
        if isinstance(node, ast.Import):
            offenders += [a.name for a in node.names if a.name.split(".")[0] == "anthropic"]
        elif isinstance(node, ast.ImportFrom) and (node.module or "").split(".")[0] == "anthropic":
            offenders.append(node.module)

    assert not offenders, (
        f"{path.relative_to(SKILL_DIR)} imports {offenders} at module load; this breaks --help when the optional SDK is absent"
    )


@pytest.mark.parametrize("module", SDK_MODULES)
def test_help_succeeds_without_anthropic(module: str):
    """`--help` is pure argparse and must not require the optional SDK."""
    result = _run_without_anthropic(f"""
        import runpy
        import sys

        sys.argv = ["{module}", "--help"]
        try:
            runpy.run_module("{module}", run_name="__main__")
        except SystemExit as exc:
            sys.exit(exc.code or 0)
        """)
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout


def test_missing_sdk_reports_prerequisite_not_traceback():
    """Reaching an SDK path without the SDK yields an actionable message."""
    result = _run_without_anthropic("""
        import sys

        sys.path.insert(0, ".")
        from scripts.utils import load_anthropic

        try:
            load_anthropic()
        except SystemExit as exc:
            print(exc)
            sys.exit(0)
        sys.exit("expected SystemExit")
        """)
    assert result.returncode == 0, result.stderr
    assert "pip install anthropic" in result.stdout
    assert "ModuleNotFoundError" not in result.stderr


@pytest.mark.parametrize(
    ("module", "requires"),
    [
        ("scripts.utils", None),
        ("scripts.aggregate_benchmark", None),
        ("scripts.generate_report", None),
        ("scripts.improve_description", None),
        ("scripts.package_skill", "yaml"),
        ("scripts.quick_validate", "yaml"),
        ("scripts.run_eval", None),
        ("scripts.run_loop", None),
    ],
    ids=lambda v: v if isinstance(v, str) else "",
)
def test_modules_import_without_anthropic(module: str, requires):
    """Import must stay SDK-free; only actual API calls may need the SDK.

    `requires` names a declared repository dependency (see requirements.txt)
    that the module legitimately needs. Missing it is an environment gap, not
    the optional-SDK bug under test, so we skip rather than report a failure
    that would obscure the real signal.
    """
    if requires is not None:
        pytest.importorskip(requires, reason=f"{module} needs the declared dependency {requires}")

    result = _run_without_anthropic(f"""
        import importlib
        import sys

        sys.path.insert(0, ".")
        importlib.import_module("{module}")
        """)
    assert result.returncode == 0, result.stderr


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
