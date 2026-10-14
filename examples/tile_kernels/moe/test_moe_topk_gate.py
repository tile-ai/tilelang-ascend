import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest


def _load_example() -> ModuleType:
    source = Path(__file__).with_name("moe_topk_gate.py")
    spec = importlib.util.spec_from_file_location("_moe_topk_gate_example_for_test", source)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load example module: {source}")

    module = importlib.util.module_from_spec(spec)
    original_argv = sys.argv
    try:
        sys.argv = [str(source)]
        spec.loader.exec_module(module)
    finally:
        sys.argv = original_argv

    return module


# The example is already written as Pytest cases and asserts the selected
# indices against a stable topk, but nothing collects it: the legacy runner
# skips test_*.py by name and this file is not one. Its own parameter set is
# twenty expert configurations against two token counts, which is more than a
# per-commit run needs, so these take the smallest, the largest and one in
# between of the expert counts.
_EXPERTS = [16, 72, 144]


@pytest.mark.parametrize("num_tokens", [4001, 8001])
@pytest.mark.parametrize("num_experts", _EXPERTS)
def test_topk_gate(num_tokens, num_experts) -> None:
    example = _load_example()
    example.test_topk_gate({"num_tokens": num_tokens, "num_experts": num_experts, "num_topk": 6})


@pytest.mark.parametrize("num_tokens", [4001, 8001])
@pytest.mark.parametrize("num_experts", _EXPERTS)
def test_topk_gate_backward(num_tokens, num_experts) -> None:
    example = _load_example()
    example.test_topk_gate_backward({"num_tokens": num_tokens, "num_experts": num_experts, "num_topk": 6})
