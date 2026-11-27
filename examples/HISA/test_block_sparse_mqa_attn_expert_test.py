import importlib.util
import sys
from pathlib import Path
from types import ModuleType


def _load_example() -> ModuleType:
    source = Path(__file__).with_name("block_sparse_mqa_attn_expert_test.py")
    spec = importlib.util.spec_from_file_location("_block_sparse_mqa_example_for_test", source)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load example module: {source}")

    module = importlib.util.module_from_spec(spec)
    original_argv = sys.argv
    try:
        # The example parses arguments at import time.
        sys.argv = [str(source)]
        spec.loader.exec_module(module)
    finally:
        sys.argv = original_argv

    return module


def test_block_sparse_mqa_attn_accuracy() -> None:
    import torch

    example = _load_example()

    # As with the paged variant next to it, the reference receives the tensors
    # that were never moved, so the default device has to be the accelerator.
    previous = torch.get_default_device()
    torch.set_default_device("npu")
    try:
        torch.manual_seed(42)
        example.test_block_sparse_mqa_attn(
            seq_len=32,
            seq_len_kv=128 * 1024,
            heads=32,
            index_dim=128,
            kv_block_size=128,
            topk=64,
            dtype="float16",
            grid_size=example.get_npu_core_num(),
        )
    finally:
        torch.set_default_device(previous)
