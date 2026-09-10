"""Create per-file pytest checks for uncovered example precision functions."""

from __future__ import annotations

import ast
import pathlib


EXCLUDED = {"__init__.py", "utils.py", "golden.py", "setup.py"}
MARKERS = ("matched_ratio", "max_abs", "PRECISION_FAIL", "check_precision", "_check_precision")


TEMPLATE = """\
import ast
import pathlib
import warnings
import torch


warnings.filterwarnings(
    "ignore",
    message=r"torch\\.jit\\.script_method is deprecated.*",
    category=DeprecationWarning,
    module=r"torch\\.jit\\._script",
)


def load_checker(source):
    tree = ast.parse(source)
    functions = {node.name: node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
    candidates = [node for node in functions.values() if "check_precision" in node.name]
    if not candidates:
        return None
    nodes = []
    pending = [candidates[0].name]
    included = set()
    while pending:
        name = pending.pop()
        if name in included or name not in functions:
            continue
        included.add(name)
        node = functions[name]
        nodes.append(node)
        for ref in ast.walk(node):
            if isinstance(ref, ast.Name) and isinstance(ref.ctx, ast.Load) and ref.id in functions:
                pending.append(ref.id)
    namespace = {"torch": torch}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "<checker>", "exec"), namespace)
    return namespace[candidates[0].name]


def test_precision_checker():
    source_path = pathlib.Path(__file__).resolve().parents[{depth}] / {source!r}
    checker = load_checker(source_path.read_text(encoding="utf-8"))
    if checker is None:
        import pytest
        pytest.skip("no check_precision function")
    actual = torch.zeros(100, dtype=torch.float16)
    golden = torch.zeros_like(actual)
    try:
        result = checker(actual, golden, "float16")
    except TypeError:
        result = checker(actual, golden)
    if isinstance(result, tuple):
        passed = bool(result[0])
    else:
        passed = True
    if not passed:
        raise AssertionError("zero-error case rejected")
    print("PASS: zero-error precision case")


if __name__ == "__main__":
    test_precision_checker()
"""


def main() -> None:
    root = pathlib.Path(__file__).resolve().parents[1]
    examples_roots = [root / "examples", root / "examples_experiment"]
    for examples in examples_roots:
        if not examples.exists():
            continue
        for source in examples.rglob("*.py"):
            if source.name in EXCLUDED:
                continue
            text = source.read_text(encoding="utf-8", errors="replace")
            if not any(marker in text for marker in MARKERS):
                continue
            if "__main__" in text or "def test_" in text or "pytest" in text:
                continue
            try:
                relative = source.relative_to(root)
                depth = len(relative.parts) - 1
                ast.parse(text, filename=str(source))
            except (ValueError, SyntaxError):
                continue
            harness = source.with_name(f"test_precision_{source.stem}.py")
            content = TEMPLATE.replace("[{depth}]", f"[{depth}]").replace("{source!r}", repr(relative.as_posix()))
            harness.write_text(content, encoding="utf-8")
            print(harness.relative_to(root))


if __name__ == "__main__":
    main()
