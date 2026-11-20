"""Computed compact UB tiles copied to strided GM columns (issue #2017)."""

import pytest
import torch

import tilelang
import tilelang.language as T


@tilelang.jit(
    target="ascendc",
    pass_configs={
        tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
        tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
    },
)
def copy_computed_column(
    rows: int,
    block: int,
    heads: int,
    dtype: str,
    threads: int,
    guard: int,
):
    """Compute in compact 1D UB, then scatter each head along the row axis.

    The loop and partial final block also exercise reuse of the staging storage.
    Guard rows and columns detect writes outside the requested GM slice.
    """
    iterations = (rows + block - 1) // block

    @T.prim_func
    def main(
        source: T.Tensor((heads, rows), dtype),
        output: T.Tensor((1, rows + 2 * guard, heads + 2 * guard), dtype),
    ):
        with T.Kernel(heads, threads=threads, is_npu=True) as cid:
            loaded = T.alloc_shared([block], dtype)
            computed = T.alloc_shared([block], dtype)
            loaded_fp32 = T.alloc_shared([block], "float32")
            computed_fp32 = T.alloc_shared([block], "float32")
            for tile in T.serial(iterations):
                valid = T.min(block, rows - tile * block)
                T.copy(
                    source[cid, tile * block : tile * block + valid],
                    loaded[0:valid],
                )
                if dtype == "bfloat16":
                    # BF16 vector Add is unsupported; cast the computed FP32
                    # result to compact BF16 before exercising the same store.
                    T.copy(loaded, loaded_fp32)
                    T.tile.add(computed_fp32, loaded_fp32, loaded_fp32)
                    T.copy(computed_fp32, computed)
                else:
                    T.tile.add(computed, loaded, loaded)
                T.copy(
                    computed[0:valid],
                    output[
                        0,
                        guard + tile * block : guard + tile * block + valid,
                        guard + cid,
                    ],
                )

    return main


@pytest.mark.parametrize(
    "rows,block,heads,dtype,threads,guard",
    [
        pytest.param(64, 64, 8, "float32", 1, 0, id="issue2017"),
        pytest.param(5, 5, 17, "float32", 1, 1, id="short_unaligned_pitch"),
        pytest.param(63, 63, 3, "float32", 1, 1, id="odd_compact_length"),
        pytest.param(192, 64, 8, "float32", 1, 1, id="reuse_three_iterations"),
        pytest.param(133, 64, 3, "float32", 1, 1, id="dynamic_tail"),
        pytest.param(64, 64, 8, "float16", 1, 1, id="float16"),
        pytest.param(64, 64, 8, "bfloat16", 1, 1, id="bfloat16"),
        pytest.param(32, 32, 3, "int32", 1, 1, id="int32"),
    ],
)
def test_copy_computed_1d_to_strided_gm(rows, block, heads, dtype, threads, guard):
    torch_dtype = getattr(torch, dtype)
    source_cpu = ((torch.arange(heads * rows) % 251) - 113).reshape(heads, rows)
    source_cpu = source_cpu.to(torch_dtype)
    expected = torch.full((1, rows + 2 * guard, heads + 2 * guard), -123, dtype=torch_dtype)
    output = expected.npu()
    expected[0, guard : guard + rows, guard : guard + heads] = (source_cpu * 2).T

    kernel = copy_computed_column(rows, block, heads, dtype, threads, guard)
    kernel(source_cpu.npu(), output)
    torch.npu.synchronize()
    torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)


@tilelang.jit(
    target="ascendc",
    pass_configs={
        tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
        tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
    },
)
def copy_computed_column_twice():
    @T.prim_func
    def main(
        source: T.Tensor((8, 64), "float32"),
        output: T.Tensor((2, 64, 8), "float32"),
    ):
        with T.Kernel(8, threads=1, is_npu=True) as cid:
            copy_column_ub = T.alloc_shared([64], "float32")
            computed = T.alloc_shared([64], "float32")
            T.copy(source[cid, 0:64], copy_column_ub)
            T.tile.add(computed, copy_column_ub, copy_column_ub)
            T.copy(computed, output[0, 0:64, cid])
            T.tile.add(computed, computed, copy_column_ub)
            T.copy(computed[3:35], output[1, 0:32, cid])

    return main


def test_multiple_computed_column_copies():
    """Independent copy nodes need distinct scratch buffers and source offsets."""
    source = torch.arange(8 * 64, dtype=torch.float32).reshape(8, 64)
    expected = torch.full((2, 64, 8), -123.0, dtype=torch.float32)
    output = expected.npu()
    expected[0] = (source * 2).T
    expected[1, :32] = (source[:, 3:35] * 3).T
    copy_computed_column_twice()(source.npu(), output)
    torch.npu.synchronize()
    torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)


def test_computed_column_rejects_incompatible_two_core_mapping():
    # Keep this region static: dynamic tails are rejected earlier by the VID
    # pass. Here the rank-1 UB is halved while the GM row axis remains whole.
    @T.prim_func
    def main(
        source: T.Tensor((8, 64), "float32"),
        output: T.Tensor((1, 64, 8), "float32"),
    ):
        with T.Kernel(8, threads=2, is_npu=True) as cid:
            loaded = T.alloc_shared([64], "float32")
            computed = T.alloc_shared([64], "float32")
            T.copy(source[cid, 0:64], loaded)
            T.tile.add(computed, loaded, loaded)
            T.copy(computed, output[0, 0:64, cid])

    with pytest.raises(Exception, match="same logical extent"):
        tilelang.compile(main, target="ascendc")


def test_copy_computed_column_manual_sync():
    """Hidden staging must synchronize a computed source without auto-sync."""
    heads, rows, repeats = 8, 64, 3

    @T.prim_func
    def main(
        source: T.Tensor((heads, rows), "float32"),
        output: T.Tensor((repeats, rows, heads), "float32"),
    ):
        with T.Kernel(heads, threads=1, is_npu=True) as cid:
            loaded = T.alloc_shared([rows], "float32")
            computed = T.alloc_shared([rows], "float32")
            T.copy(source[cid, :], loaded)
            T.set_flag("MTE2", "V", 0)
            T.wait_flag("MTE2", "V", 0)
            for repeat in T.serial(repeats):
                T.tile.add(computed, loaded, loaded)
                T.copy(computed, output[repeat, 0:rows, cid])
                T.tile.add(loaded, computed, loaded)

    kernel = tilelang.compile(
        main,
        target="ascendc",
        pass_configs={
            tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: False,
            tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
        },
    )
    source = torch.arange(heads * rows, dtype=torch.float32).reshape(heads, rows) - 113
    expected = torch.stack([(source * (2 * 3**repeat)).T for repeat in range(repeats)])
    output = torch.full_like(expected, -123, device="npu")
    kernel(source.npu(), output)
    torch.npu.synchronize()
    torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)


@pytest.mark.parametrize("dtype,auto_plan", [("float32", False), ("float16", True)])
def test_narrow_slot_roundtrip_preserves_live_buffer(dtype, auto_plan):
    """Preserve MTE-filled row slots and the live buffer after their footprint."""
    heads, rows, guard_size = 8, 64, 64

    @T.prim_func
    def main(
        source: T.Tensor((heads, rows, 1), dtype),
        guard_input: T.Tensor((heads, guard_size), "float32"),
        output: T.Tensor((1, rows + 2, heads + 2), dtype),
        guard_output: T.Tensor((heads, guard_size), "float32"),
    ):
        with T.Kernel(heads, threads=1, is_npu=True) as cid:
            narrow_ub = T.alloc_ub((rows, 1), dtype)
            guard_ub = T.alloc_ub((guard_size,), "float32")
            T.copy(source[cid, 0:32, 0:1], narrow_ub[0:32, 0:1])
            T.copy(guard_input[cid, :], guard_ub)
            T.barrier_all()
            T.copy(source[cid, 0:rows, 0:1], narrow_ub)
            T.barrier_all()
            T.copy(narrow_ub, output[0, 1 : rows + 1, cid + 1])
            T.copy(guard_ub, guard_output[cid, :])

    kernel = tilelang.compile(
        main,
        target="ascendc",
        pass_configs={
            tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
            tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: auto_plan,
        },
    )
    source = ((torch.arange(heads * rows) % 251) - 113).reshape(heads, rows, 1)
    source = source.to(getattr(torch, dtype))
    expected = torch.full((1, rows + 2, heads + 2), -123, dtype=source.dtype)
    output = expected.npu()
    expected[0, 1 : rows + 1, 1 : heads + 1] = source[:, :, 0].T
    guard_input = torch.full((heads, guard_size), -321.0, device="npu")
    guard_output = torch.full_like(guard_input, -42.0)
    kernel(source.npu(), guard_input, output, guard_output)
    torch.npu.synchronize()
    torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)
    torch.testing.assert_close(guard_output, guard_input, rtol=0, atol=0)
