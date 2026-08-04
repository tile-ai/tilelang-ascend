import importlib.util
import sys
from pathlib import Path
from types import ModuleType


def _load_example() -> ModuleType:
    source = Path(__file__).with_name("paged_block_sparse_mqa_attn_expert.py")
    spec = importlib.util.spec_from_file_location("_paged_block_sparse_mqa_example_for_test", source)
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


def test_paged_block_sparse_mqa_attn_accuracy() -> None:
    import torch

    example = _load_example()

    # The example builds its tensors without naming a device and hands the
    # unmoved ones to the reference, so the default has to be the device or the
    # comparison is across two of them. Restoring it keeps the choice from
    # reaching whatever else shares the process.
    previous = torch.get_default_device()
    torch.set_default_device("npu")
    try:
        torch.manual_seed(42)
        example.test_paged_block_sparse_mqa_attn(
            batch=1,
            seq_len=1,
            num_phys_blocks=1024,
            heads=32,
            index_dim=128,
            kv_block_size=128,
            topk=64,
            max_blocks=256,
            dtype="float16",
        )
    finally:
        torch.set_default_device(previous)
