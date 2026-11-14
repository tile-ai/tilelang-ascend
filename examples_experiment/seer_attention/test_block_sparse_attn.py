import importlib.util
import sys
from pathlib import Path
from types import ModuleType


def _load_example() -> ModuleType:
    source = Path(__file__).with_name("block_sparse_attn.py")
    spec = importlib.util.spec_from_file_location("_block_sparse_attn_example_for_test", source)
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


# The example already asserts against a reference it builds inline: the block
# mask is expanded with a Kronecker product, intersected with the causal mask
# and, for the shorter query, offset by the length difference. Restating that
# here would duplicate the part most likely to be transcribed wrong, so these
# call it instead. Loading the example under a private module name keeps Pytest
# from collecting its functions twice.
def test_topk_sparse_attention() -> None:
    example = _load_example()
    example.test_topk_sparse_attention()


def test_topk_sparse_attention_qlen_lt_klen() -> None:
    example = _load_example()
    example.test_topk_sparse_attention_qlen_lt_klen()
