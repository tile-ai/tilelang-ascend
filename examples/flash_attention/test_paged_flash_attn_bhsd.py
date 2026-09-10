import importlib.util
import sys
from pathlib import Path
from types import ModuleType


def _load_paged_flash_attn_example() -> ModuleType:
    source = Path(__file__).with_name("paged_flash_attn_bhsd.py")
    spec = importlib.util.spec_from_file_location("_paged_flash_attn_bhsd_example_for_test", source)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load example module: {source}")

    module = importlib.util.module_from_spec(spec)
    original_argv = sys.argv
    try:
        # paged_flash_attn_bhsd.py parses arguments in its __main__ block. Hide
        # Pytest arguments while loading it without changing the original example.
        sys.argv = [str(source)]
        spec.loader.exec_module(module)
    finally:
        sys.argv = original_argv

    return module


def test_paged_flash_attn_bhsd_accuracy() -> None:
    import torch

    example = _load_paged_flash_attn_example()

    # Values mirror the argparse defaults in the example's main().
    batch = 1
    heads = 16
    seq_len = 128
    dim = 128
    block_size = 128

    # gen_cache and the kernel place their tensors on the device of their inputs,
    # which is how the example reaches the NPU from its own main().
    torch.set_default_device("npu")
    torch.manual_seed(0)

    query = torch.randn((batch, heads, seq_len, dim), dtype=torch.float16)
    key = torch.randn((batch, heads, seq_len, dim), dtype=torch.float16)
    value = torch.randn((batch, heads, seq_len, dim), dtype=torch.float16)

    table_blocks = example.ceildiv(seq_len, block_size)
    block_table = torch.zeros((batch, table_blocks), dtype=torch.int32)
    key_cache = example.gen_cache(key, block_table, block_size)
    value_cache = example.gen_cache(value, block_table, block_size)

    kernel = example.paged_flash_attention_fwd(
        batch,
        heads,
        seq_len,
        dim,
        key_cache.shape[0],
        table_blocks,
        block_size=block_size,
    )

    actual = kernel(query, key_cache, value_cache, block_table)
    expected = example.ref_program(query, key_cache, value_cache, block_table)
    torch.npu.synchronize()

    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)
