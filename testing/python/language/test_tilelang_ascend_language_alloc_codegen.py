import pytest
from unittest.mock import patch

import tilelang
import tilelang.language as T


DEV_CONFIGS = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}


@pytest.fixture(scope="session", autouse=True)
def clear_cache():
    tilelang.cache.clear_cache()


def dev_L1_explicit_splice(dim, block_N=128, block_size=128, live_max=64):
    dtype = "bfloat16"
    batch = T.symbolic("batch")

    @T.prim_func
    def main(
        K_live: T.Tensor([batch, live_max, dim], dtype),
        K_cache: T.Tensor([1, block_size, dim], dtype),
        Output: T.Tensor([batch, block_N, dim], dtype),
        prefix_lens: T.Tensor([batch], "int32"),
        live_lens: T.Tensor([batch], "int32"),
        block_table: T.Tensor([batch, 1], "int32"),
    ):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            k_l1 = T.alloc_L1([block_N, dim], dtype)

            for b_i in T.serial(batch):
                prefix_len_b = prefix_lens[b_i]
                live_len_b = live_lens[b_i]

                tile_cache_len = T.if_then_else(
                    prefix_len_b > 0,
                    T.min(prefix_len_b, block_N),
                    0,
                )
                tile_live_len = T.if_then_else(
                    tile_cache_len < block_N,
                    T.min(live_len_b, block_N - tile_cache_len),
                    0,
                )

                if tile_cache_len > 0 and tile_live_len > 0:
                    physical_block = block_table[b_i, 0]
                    T.copy(
                        K_cache[physical_block, 0:tile_cache_len, :],
                        k_l1[0:tile_cache_len, :],
                    )
                    T.copy(
                        K_live[b_i, 0:tile_live_len, :],
                        k_l1[tile_cache_len : tile_cache_len + tile_live_len, :],
                    )

                elif tile_cache_len > 0:
                    physical_block = block_table[b_i, 0]
                    T.copy(
                        K_cache[physical_block, 0:tile_cache_len, :],
                        k_l1[0:tile_cache_len, :],
                    )

                elif tile_live_len > 0:
                    T.copy(
                        K_live[b_i, 0:tile_live_len, :],
                        k_l1[0:tile_live_len, :],
                    )

    return main


def dev_ub_explicit_splice(dim, block_N=128, block_size=128, live_max=64):
    dtype = "bfloat16"
    batch = T.symbolic("batch")

    @T.prim_func
    def main(
        K_live: T.Tensor([batch, live_max, dim], dtype),
        K_cache: T.Tensor([1, block_size, dim], dtype),
        Output: T.Tensor([batch, block_N, dim], dtype),
        prefix_lens: T.Tensor([batch], "int32"),
        live_lens: T.Tensor([batch], "int32"),
        block_table: T.Tensor([batch, 1], "int32"),
    ):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            k_ub = T.alloc_ub([block_N, dim], dtype)

            for b_i in T.serial(batch):
                prefix_len_b = prefix_lens[b_i]
                live_len_b = live_lens[b_i]

                tile_cache_len = T.if_then_else(
                    prefix_len_b > 0,
                    T.min(prefix_len_b, block_N),
                    0,
                )
                tile_live_len = T.if_then_else(
                    tile_cache_len < block_N,
                    T.min(live_len_b, block_N - tile_cache_len),
                    0,
                )

                if tile_cache_len > 0 and tile_live_len > 0:
                    physical_block = block_table[b_i, 0]
                    T.copy(
                        K_cache[physical_block, 0:tile_cache_len, :],
                        k_ub[0:tile_cache_len, :],
                    )
                    T.copy(
                        K_live[b_i, 0:tile_live_len, :],
                        k_ub[tile_cache_len : tile_cache_len + tile_live_len, :],
                    )

                elif tile_cache_len > 0:
                    physical_block = block_table[b_i, 0]
                    T.copy(
                        K_cache[physical_block, 0:tile_cache_len, :],
                        k_ub[0:tile_cache_len, :],
                    )

                elif tile_live_len > 0:
                    T.copy(
                        K_live[b_i, 0:tile_live_len, :],
                        k_ub[0:tile_live_len, :],
                    )

    return main


def _compile_and_get_source(target):
    prim_func = dev_L1_explicit_splice(dim=64)
    with (
        patch("tilelang.jit.adapter.libgen.LibraryGenerator.compile_lib") as mock_compile,
        patch(
            "tilelang.jit.adapter.libgen.LibraryGenerator.load_lib",
            return_value=None,
        ),
    ):
        mock_compile.return_value = None
        compiled = tilelang.compile(
            prim_func,
            out_idx=[2],
            pass_configs=DEV_CONFIGS,
            target=target,
        )
    return compiled.get_kernel_source()


def _compile_ub_and_get_source(target):
    prim_func = dev_ub_explicit_splice(dim=64)
    with (
        patch("tilelang.jit.adapter.libgen.LibraryGenerator.compile_lib") as mock_compile,
        patch(
            "tilelang.jit.adapter.libgen.LibraryGenerator.load_lib",
            return_value=None,
        ),
    ):
        mock_compile.return_value = None
        compiled = tilelang.compile(
            prim_func,
            out_idx=[2],
            pass_configs=DEV_CONFIGS,
            target=target,
        )
    return compiled.get_kernel_source()


@pytest.mark.parametrize("target", ["ascendc", "pto"])
def test_l1_splice_codegen(target):
    code = _compile_and_get_source(target)

    if target == "ascendc":
        assert "ascend_l1.GetWithOffset" in code, (
            f"k_l1 should be allocated as ascend_l1 (L1 buffer), but 'ascend_l1.GetWithOffset' not found in generated AscendC code:\n{code}"
        )
        k_l1_alloc_line = [line for line in code.splitlines() if "k_l1" in line and "GetWithOffset" in line]
        assert len(k_l1_alloc_line) > 0, "k_l1 GetWithOffset line not found"
        assert "ascend_ub" not in k_l1_alloc_line[0], (
            f"k_l1 was incorrectly allocated as ascend_ub instead of ascend_l1:\n{k_l1_alloc_line[0]}"
        )

    elif target == "pto":
        assert "TileMatL1" in code, (
            f"k_l1 should be allocated as TileMatL1 (L1 buffer), but 'TileMatL1' not found in generated PTO code:\n{code}"
        )
        k_l1_alloc_line = [line for line in code.splitlines() if "k_l1" in line and "TileMat" in line]
        assert len(k_l1_alloc_line) > 0, "k_l1 TileMat declaration line not found"
        assert "TileUbDataND" not in k_l1_alloc_line[0], (
            f"k_l1 was incorrectly allocated as TileUbDataND instead of TileMatL1:\n{k_l1_alloc_line[0]}"
        )


@pytest.mark.parametrize("target", ["ascendc", "pto"])
def test_ub_splice_codegen(target):
    code = _compile_ub_and_get_source(target)

    if target == "ascendc":
        assert "ascend_ub.GetWithOffset" in code, (
            f"k_ub should be allocated as ascend_ub (UB buffer), but 'ascend_ub.GetWithOffset' not found in generated AscendC code:\n{code}"
        )
        k_ub_alloc_line = [line for line in code.splitlines() if "k_ub" in line and "GetWithOffset" in line]
        assert len(k_ub_alloc_line) > 0, "k_ub GetWithOffset line not found"
        assert "ascend_l1" not in k_ub_alloc_line[0], (
            f"k_ub was incorrectly allocated as ascend_l1 instead of ascend_ub:\n{k_ub_alloc_line[0]}"
        )

    elif target == "pto":
        assert "TileUbDataND" in code, (
            f"k_ub should be allocated as TileUbDataND (UB buffer), but 'TileUbDataND' not found in generated PTO code:\n{code}"
        )
        k_ub_alloc_line = [line for line in code.splitlines() if "k_ub" in line and "Tile" in line]
        assert len(k_ub_alloc_line) > 0, "k_ub Tile declaration line not found"
        assert "TileMatL1" not in k_ub_alloc_line[0], (
            f"k_ub was incorrectly allocated as TileMatL1 instead of TileUbDataND:\n{k_ub_alloc_line[0]}"
        )


# ---------------------------------------------------------------------------
# Regression tests for issue #1301: PTO codegen "Find undefined Variable batch"
# when a buffer's shape is a composite symbolic expression (e.g. batch + 1)
# that appears before a buffer whose shape is the bare symbolic variable.
# ---------------------------------------------------------------------------


def _composite_shape_first():
    """``[batch + 1]`` buffer precedes the ``[batch]`` buffer.

    This is the exact ordering that triggered the original crash: ``batch``
    is never registered by a preceding bare-variable buffer, so PTO codegen
    cannot resolve it when printing the ``batch + 1`` shape."""
    batch = T.symbolic("batch")

    @T.prim_func
    def main(
        B: T.Tensor([batch + 1], "int32"),
        A: T.Tensor([batch, 16], "float32"),
    ):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            s = T.alloc_var("int32", init=0)
            for _i in T.serial(batch):
                s = s + 1

    return main


def _multi_var_composite_shape():
    """Shape expression contains two distinct symbolic vars (``batch + seq``).

    Exercises recursive collection of more than one VarNode from a single
    composite shape expression."""
    batch = T.symbolic("batch")
    seq = T.symbolic("seq")

    @T.prim_func
    def main(
        B: T.Tensor([batch + seq], "int32"),
        A: T.Tensor([batch, seq], "float32"),
    ):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            s = T.alloc_var("int32", init=0)
            for _i in T.serial(batch):
                s = s + 1

    return main


def _nested_composite_shape():
    """Shape expression is a nested arithmetic (``batch * 2 + 1``).

    Verifies the visitor descends through MulNode -> AddNode -> VarNode."""
    batch = T.symbolic("batch")

    @T.prim_func
    def main(
        B: T.Tensor([batch * 2 + 1], "int32"),
        A: T.Tensor([batch, 16], "float32"),
    ):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            s = T.alloc_var("int32", init=0)
            for _i in T.serial(batch):
                s = s + 1

    return main


def _compile_symbolic_and_get_source(prim_func, target):
    with (
        patch("tilelang.jit.adapter.libgen.LibraryGenerator.compile_lib") as mock_compile,
        patch(
            "tilelang.jit.adapter.libgen.LibraryGenerator.load_lib",
            return_value=None,
        ),
    ):
        mock_compile.return_value = None
        compiled = tilelang.compile(prim_func, target=target)
    return compiled.get_kernel_source()


def _assert_shape_var_in_signature(code, *var_names):
    """Assert that each symbolic var is emitted as a kernel parameter.

    Both backends emit shape variables as ``int64_t <name>`` in the kernel
    signature, so we check the first line containing ``main_kernel``."""
    sig_lines = [line for line in code.splitlines() if "main_kernel" in line and "(" in line]
    assert sig_lines, f"kernel signature not found in generated code:\n{code}"
    sig = sig_lines[0]
    for name in var_names:
        assert f"int64_t {name}" in sig, f"symbolic var '{name}' should be a kernel parameter, signature:\n{sig}"


@pytest.mark.parametrize("target", ["ascendc", "pto"])
def test_composite_shape_first_codegen(target):
    """Issue #1301 regression: ``[batch + 1]`` before ``[batch]`` must not
    crash codegen, and ``batch`` must be emitted as a kernel parameter."""
    code = _compile_symbolic_and_get_source(_composite_shape_first(), target)
    _assert_shape_var_in_signature(code, "batch")


@pytest.mark.parametrize("target", ["ascendc", "pto"])
def test_multi_var_composite_shape_codegen(target):
    """A composite shape (``batch + seq``) must register both vars."""
    code = _compile_symbolic_and_get_source(_multi_var_composite_shape(), target)
    _assert_shape_var_in_signature(code, "batch", "seq")


@pytest.mark.parametrize("target", ["ascendc", "pto"])
def test_nested_composite_shape_codegen(target):
    """A nested arithmetic shape (``batch * 2 + 1``) must still register
    ``batch``."""
    code = _compile_symbolic_and_get_source(_nested_composite_shape(), target)
    _assert_shape_var_in_signature(code, "batch")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
