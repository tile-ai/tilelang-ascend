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
    with patch(
        "tilelang.jit.adapter.libgen.LibraryGenerator.compile_lib"
    ) as mock_compile, patch(
        "tilelang.jit.adapter.libgen.LibraryGenerator.load_lib",
        return_value=None,
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
    with patch(
        "tilelang.jit.adapter.libgen.LibraryGenerator.compile_lib"
    ) as mock_compile, patch(
        "tilelang.jit.adapter.libgen.LibraryGenerator.load_lib",
        return_value=None,
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
            f"k_l1 should be allocated as ascend_l1 (L1 buffer), "
            f"but 'ascend_l1.GetWithOffset' not found in generated AscendC code:\n{code}"
        )
        k_l1_alloc_line = [
            line for line in code.splitlines() if "k_l1" in line and "GetWithOffset" in line
        ]
        assert len(k_l1_alloc_line) > 0, "k_l1 GetWithOffset line not found"
        assert "ascend_ub" not in k_l1_alloc_line[0], (
            f"k_l1 was incorrectly allocated as ascend_ub instead of ascend_l1:\n"
            f"{k_l1_alloc_line[0]}"
        )

    elif target == "pto":
        assert "TileMatL1" in code, (
            f"k_l1 should be allocated as TileMatL1 (L1 buffer), "
            f"but 'TileMatL1' not found in generated PTO code:\n{code}"
        )
        k_l1_alloc_line = [
            line for line in code.splitlines() if "k_l1" in line and "TileMat" in line
        ]
        assert len(k_l1_alloc_line) > 0, "k_l1 TileMat declaration line not found"
        assert "TileUbDataND" not in k_l1_alloc_line[0], (
            f"k_l1 was incorrectly allocated as TileUbDataND instead of TileMatL1:\n"
            f"{k_l1_alloc_line[0]}"
        )


@pytest.mark.parametrize("target", ["ascendc", "pto"])
def test_ub_splice_codegen(target):
    code = _compile_ub_and_get_source(target)

    if target == "ascendc":
        assert "ascend_ub.GetWithOffset" in code, (
            f"k_ub should be allocated as ascend_ub (UB buffer), "
            f"but 'ascend_ub.GetWithOffset' not found in generated AscendC code:\n{code}"
        )
        k_ub_alloc_line = [
            line for line in code.splitlines() if "k_ub" in line and "GetWithOffset" in line
        ]
        assert len(k_ub_alloc_line) > 0, "k_ub GetWithOffset line not found"
        assert "ascend_l1" not in k_ub_alloc_line[0], (
            f"k_ub was incorrectly allocated as ascend_l1 instead of ascend_ub:\n"
            f"{k_ub_alloc_line[0]}"
        )

    elif target == "pto":
        assert "TileUbDataND" in code, (
            f"k_ub should be allocated as TileUbDataND (UB buffer), "
            f"but 'TileUbDataND' not found in generated PTO code:\n{code}"
        )
        k_ub_alloc_line = [
            line for line in code.splitlines() if "k_ub" in line and "Tile" in line
        ]
        assert len(k_ub_alloc_line) > 0, "k_ub Tile declaration line not found"
        assert "TileMatL1" not in k_ub_alloc_line[0], (
            f"k_ub was incorrectly allocated as TileMatL1 instead of TileUbDataND:\n"
            f"{k_ub_alloc_line[0]}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
