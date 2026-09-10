import ast
import pathlib
import warnings
import torch


warnings.filterwarnings(
    "ignore",
    message=r"torch\.jit\.script_method is deprecated.*",
    category=DeprecationWarning,
    module=r"torch\.jit\._script",
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
    source_path = pathlib.Path(__file__).resolve().parents[2] / "examples/pipeline/sparse_flash_attn_gqa_pipeline_pto.py"
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
