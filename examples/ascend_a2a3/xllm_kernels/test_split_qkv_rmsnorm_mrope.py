import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest


HEAD_CONFIGS = [
    (32, 2),
    (24, 4),
    (16, 4),
    (16, 2),
    (16, 1),
    (12, 2),
    (8, 2),
    (8, 1),
    (6, 1),
    (4, 1),
    (3, 1),
    (2, 1),
    (1, 1),
]

SPLIT_QKV_RMSNORM_MROPE_CASES = [
    pytest.param(16, num_q_heads, num_kv_heads, True, id=f"tokens16_q{num_q_heads}_kv{num_kv_heads}_interleaved")
    for num_q_heads, num_kv_heads in HEAD_CONFIGS
]
SPLIT_QKV_RMSNORM_MROPE_CASES += [
    pytest.param(num_tokens, num_q_heads, num_kv_heads, True, id=f"tokens{num_tokens}_q{num_q_heads}_kv{num_kv_heads}_interleaved")
    for num_tokens in [1, 17, 4097]
    for num_q_heads, num_kv_heads in [(32, 2), (24, 4), (16, 2)]
]
SPLIT_QKV_RMSNORM_MROPE_CASES += [
    pytest.param(
        num_tokens, 16, 4, is_interleaved, id=f"tokens{num_tokens}_q16_kv4_{'interleaved' if is_interleaved else 'noninterleaved'}"
    )
    for num_tokens in [4097, 8192]
    for is_interleaved in [True, False]
]


def _load_split_qkv_rmsnorm_mrope_example() -> ModuleType:
    source = Path(__file__).with_name("split_qkv_rmsnorm_mrope.py")
    spec = importlib.util.spec_from_file_location("_split_qkv_rmsnorm_mrope_example_for_test", source)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load example module: {source}")

    module = importlib.util.module_from_spec(spec)
    original_argv = sys.argv
    original_sys_path = list(sys.path)
    try:
        sys.argv = [str(source)]
        sys.path.insert(0, str(source.parent))
        sys.path.insert(0, str(source.parents[2]))
        spec.loader.exec_module(module)
    finally:
        sys.argv = original_argv
        sys.path[:] = original_sys_path
    return module


@pytest.mark.parametrize(
    ("num_tokens", "num_q_heads", "num_kv_heads", "is_interleaved"),
    SPLIT_QKV_RMSNORM_MROPE_CASES,
)
def test_split_qkv_rmsnorm_mrope_accuracy(
    num_tokens: int,
    num_q_heads: int,
    num_kv_heads: int,
    is_interleaved: bool,
) -> None:
    example = _load_split_qkv_rmsnorm_mrope_example()
    head_size, rope_dim = example.SUPPORTED_HEAD_SPECS[0]

    example._run_ref_check(
        num_tokens=num_tokens,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        rope_dim=rope_dim,
        eps=example.REF_CHECK_EPS,
        mrope_section=tuple(example.DEFAULT_MROPE_SECTION),
        is_interleaved=is_interleaved,
    )
