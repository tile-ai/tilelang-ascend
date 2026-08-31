import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

# The cases a pull request runs; everything else is low_priority and waits for
# the scheduled run. One case per distinct combination of the three parameters
# that select a kernel path -- the input and output scale-factor encodings and
# the output block -- which is twelve of them, alternating hidden between the
# smallest and largest in the grid so both ends of the block_k derivation in
# _derive_cast_layout are covered. num_tokens is a T.symbolic dimension and so
# not part of the compile key, which is why one value of it stands in for both.
_SMOKE_CASES = frozenset(
    {
        (4001, 576, False, False, (1, 128)),
        (4001, 7168, False, False, (32, 32)),
        (4001, 576, False, False, (128, 128)),
        (4001, 7168, False, True, (1, 128)),
        (4001, 576, False, True, (32, 32)),
        (4001, 7168, False, True, (128, 128)),
        (4001, 576, True, False, (1, 128)),
        (4001, 7168, True, False, (32, 32)),
        (4001, 576, True, False, (128, 128)),
        (4001, 7168, True, True, (1, 128)),
        (4001, 576, True, True, (32, 32)),
        (4001, 7168, True, True, (128, 128)),
    }
)


def _test_params() -> list:
    params = []
    for num_tokens in (4001, 8001):
        for hidden in (576, 2048, 2560, 3072, 4096, 6144, 7168):
            for in_tma, in_round, in_packed in ((False, True, False), (True, True, True)):
                for out_tma, out_round, out_packed in ((False, True, False), (True, True, True)):
                    for out_sf_block in ((1, 128), (32, 32), (128, 128)):
                        for in_sf_block in ((1, 32),):
                            if out_sf_block[0] % in_sf_block[0] or out_sf_block[1] % in_sf_block[1]:
                                continue
                            case = {
                                "num_tokens": num_tokens,
                                "hidden": hidden,
                                "in_use_tma_aligned_col_major_sf": in_tma,
                                "in_round_sf": in_round,
                                "in_use_packed_ue8m0": in_packed,
                                "out_use_tma_aligned_col_major_sf": out_tma,
                                "out_round_sf": out_round,
                                "out_use_packed_ue8m0": out_packed,
                                "out_sf_block": out_sf_block,
                                "in_sf_block": in_sf_block,
                            }
                            params.append(
                                pytest.param(
                                    case,
                                    id=(
                                        f"num_tokens={num_tokens}_hidden={hidden}"
                                        f"_in_packed={in_packed}_out_packed={out_packed}"
                                        f"_out_sf_block={out_sf_block[0]}x{out_sf_block[1]}"
                                    ),
                                    marks=(
                                        ()
                                        if (num_tokens, hidden, in_packed, out_packed, out_sf_block) in _SMOKE_CASES
                                        else pytest.mark.low_priority
                                    ),
                                )
                            )
    return params


@pytest.fixture
def isolated_kernel_cache(tmp_path):
    """Give the case its own kernel cache directory.

    The Example clears the kernel cache when it is imported, and here that
    import happens inside every fork. Against the shared directory that would
    delete kernels the other operator tests in the same Pytest invocation are
    compiling into; against this one it deletes nothing anyone else holds. Each
    case runs in its own fork, so nothing it compiles was reusable regardless.
    """
    import tilelang.cache

    previous = tilelang.cache.get_cache_dir()
    tilelang.cache.set_cache_dir(str(tmp_path))
    try:
        yield
    finally:
        tilelang.cache.set_cache_dir(str(previous))


def _load_per_block_cast_example() -> ModuleType:
    source = Path(__file__).with_name("per_block_cast_lossless_kernel.py")
    spec = importlib.util.spec_from_file_location("_per_block_cast_example_for_test", source)
    module = importlib.util.module_from_spec(spec)
    original_argv = sys.argv
    original_path = list(sys.path)
    # The Example resolves its helpers as a top-level `utils`, a name
    # examples/xllm_kernels also claims. Both can be collected by one Pytest
    # invocation, so put this directory first and take the entry back after.
    original_utils = sys.modules.pop("utils", None)
    try:
        # per_block_cast_lossless_kernel.py reads arguments at import time. Hide
        # Pytest's own arguments while loading it, without changing the Example.
        sys.argv = [str(source)]
        sys.path.insert(0, str(source.parent))
        spec.loader.exec_module(module)
    finally:
        sys.argv = original_argv
        sys.path[:] = original_path
        if original_utils is not None:
            sys.modules["utils"] = original_utils
    return module


@pytest.mark.parametrize("params", _test_params())
def test_per_block_cast_lossless(params: dict, isolated_kernel_cache) -> None:
    import torch

    example = _load_per_block_cast_example()

    _, x_bf16, cast_func = example.generate_test_data(params)
    out, _ = cast_func()
    if hasattr(torch, "npu"):
        torch.npu.synchronize()

    out_ref = example.compute_reference_out(x_bf16, params)
    example.assert_equal(out, out_ref, check_stride=False)
