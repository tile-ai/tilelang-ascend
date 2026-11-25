"""Compact UB tiles copied to and from strided GM columns (issue #2017)."""

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


@pytest.mark.parametrize("dtype", ["float32", "float16", "bfloat16", "int32"])
@pytest.mark.parametrize("auto_sync,auto_plan", [(True, True), (False, False)])
def test_dma_column_1d_layout_and_offset_guards(dtype, auto_sync, auto_plan):
    """Column DMA produces compact 1D elements and preserves partial-block guards."""
    rows, heads, capacity, offset = 5, 3, 32, 3

    @T.prim_func
    def main(
        source: T.Tensor((rows, heads), dtype),
        initial: T.Tensor((heads + 1, capacity), dtype),
        output: T.Tensor((rows + 2, heads + 3), dtype),
        compact_output: T.Tensor((heads + 1, capacity), dtype),
        valid_length: T.int32,
    ):
        # The extra core sees an empty source column. The requested UB slice
        # receives the default pad value; lanes outside that slice stay intact.
        with T.Kernel(heads + 1, threads=1, is_npu=True) as cid:
            tmp = T.alloc_ub((capacity,), dtype)
            n = T.max(0, T.min(valid_length, rows))
            T.copy(initial[cid, :], tmp)
            T.barrier_all()
            T.copy(source[:n, cid], tmp[offset : offset + n])
            T.copy(tmp, compact_output[cid, :])
            T.barrier_all()
            T.copy(tmp[offset : offset + n], output[1 : 1 + n, cid + 1])

    kernel = tilelang.compile(
        main,
        target="ascendc",
        pass_configs={
            tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: auto_sync,
            tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: auto_plan,
        },
    )
    torch_dtype = getattr(torch, dtype)
    source = (torch.arange(rows * heads).reshape(rows, heads) + 7).to(torch_dtype)
    initial = torch.full((heads + 1, capacity), -123, dtype=torch_dtype)
    for length in [-1, 0, 1, 4, 5, 8]:
        n = max(0, min(length, rows))
        expected = torch.full((rows + 2, heads + 3), -123, dtype=torch_dtype)
        expected[1 : n + 1, 1 : heads + 1] = source[:n]
        expected[1 : n + 1, heads + 1] = 0
        compact_expected = initial.clone()
        compact_expected[:heads, offset : offset + n] = source[:n].T
        compact_expected[heads, offset : offset + n] = 0
        output = torch.full_like(expected, -123, device="npu")
        compact_output = torch.full_like(compact_expected, -42, device="npu")
        kernel(source.npu(), initial.npu(), output, compact_output, length)
        torch.npu.synchronize()
        torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)
        torch.testing.assert_close(compact_output.cpu(), compact_expected, rtol=0, atol=0)


def test_dma_column_1d_tail_padding_preserves_slice_guards():
    rows, heads, capacity, offset = 5, 3, 80, 3

    @T.prim_func
    def main(
        source: T.Tensor((rows, heads), "float32"),
        initial: T.Tensor((heads + 1, capacity), "float32"),
        output: T.Tensor((heads + 1, capacity), "float32"),
        valid_length: T.int32,
    ):
        with T.Kernel(heads + 1, threads=1, is_npu=True) as cid:
            tmp = T.alloc_ub((capacity,), "float32")
            n = T.max(0, T.min(valid_length, 64))
            T.copy(initial[cid, :], tmp)
            T.copy(source[:n, cid], tmp[offset : offset + n], pad_value=7)
            T.copy(tmp, output[cid, :])

    kernel = tilelang.compile(
        main,
        target="ascendc",
        pass_configs={
            tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
            tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
        },
    )
    source = torch.arange(rows * heads, dtype=torch.float32).reshape(rows, heads) + 13
    initial = torch.full((heads + 1, capacity), -123.0, device="npu")
    for length in [0, 1, 5, 13, 64, 80]:
        n = min(length, 64)
        copied_rows = min(rows, n)
        expected = torch.full((heads + 1, capacity), -123.0)
        expected[:, offset : offset + n] = 7
        expected[:heads, offset : offset + copied_rows] = source[:copied_rows].T
        output = torch.full_like(expected, -42, device="npu")
        kernel(source.npu(), initial, output, length)
        torch.npu.synchronize()
        torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)


@pytest.mark.parametrize("auto_sync", [True, False])
def test_dma_column_1d_computation_and_reuse(auto_sync):
    """The same UB can alternate DMA and computed producers inside a loop."""
    rows, heads, capacity, offset = 4, 3, 32, 3

    @T.prim_func
    def main(
        source: T.Tensor((2, rows, heads), "float32"),
        initial: T.Tensor((heads, capacity), "float32"),
        output: T.Tensor((2, 2, rows, heads), "float32"),
    ):
        with T.Kernel(heads, threads=1, is_npu=True) as cid:
            tmp = T.alloc_ub((capacity,), "float32")
            copied = T.alloc_ub((capacity,), "float32")
            T.copy(initial[cid, :], tmp)
            for repeat in T.serial(2):
                # Seed an unrelated old tail mask. The following column load
                # must invalidate it before the full-tile add or UB copy.
                T.copy(initial[cid, :2], tmp[:2])
                T.barrier_all()
                T.copy(source[repeat, :rows, cid], tmp[offset : offset + rows])
                T.copy(tmp, copied)
                T.barrier_all()
                T.copy(copied[offset : offset + rows], output[repeat, 0, :rows, cid])
                T.tile.add(tmp, tmp, tmp)
                T.copy(tmp[offset : offset + rows], output[repeat, 1, :rows, cid])

    kernel = tilelang.compile(
        main,
        target="ascendc",
        pass_configs={
            tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: auto_sync,
            tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
            tilelang.PassConfigKey.TL_ASCEND_TAIL_MASK: True,
        },
    )
    source = torch.arange(2 * rows * heads, dtype=torch.float32).reshape(2, rows, heads)
    initial = torch.full((heads, capacity), -123.0, device="npu")
    expected = torch.stack((source, source * 2), dim=1)
    output = torch.full_like(expected, -123, device="npu")
    kernel(source.npu(), initial, output)
    torch.npu.synchronize()
    torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)


@pytest.mark.parametrize("tail_mask", [False, True])
@pytest.mark.parametrize("rows", [1, 5])
def test_dma_column_1d_full_tail_keeps_padding_through_exp(tail_mask, rows):
    @T.prim_func
    def main(
        source: T.Tensor((rows, 3), "float32"),
        output: T.Tensor((4, 64), "float32"),
    ):
        with T.Kernel(4, threads=1, is_npu=True) as cid:
            tmp = T.alloc_ub((64,), "float32")
            copied = T.alloc_ub((64,), "float32")
            T.copy(source[:64, cid], tmp, pad_value=0)
            T.copy(tmp, copied)
            T.tile.add(tmp, copied, copied)
            T.tile.exp(tmp, tmp)
            T.copy(tmp, output[cid, :])

    kernel = tilelang.compile(
        main,
        target="ascendc",
        pass_configs={
            tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
            tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
            tilelang.PassConfigKey.TL_ASCEND_TAIL_MASK: tail_mask,
        },
    )
    source = torch.arange(rows * 3, dtype=torch.float32).reshape(rows, 3) / 16
    expected = torch.full((4, 64), 0.0 if tail_mask else 1.0)
    expected[:3, :rows] = (source * 2).exp().T
    output = torch.full_like(expected, -123, device="npu")
    kernel(source.npu(), output)
    torch.npu.synchronize()
    torch.testing.assert_close(output.cpu(), expected, rtol=1e-5, atol=1e-5)


def test_dynamic_column_slice_of_large_1d_ub():
    """A proven 64-row slice needs 64 slots even when the UB capacity is 8192."""

    @T.prim_func
    def main(
        source: T.Tensor((64,), "float32"),
        output: T.Tensor((64, 2), "float32"),
        valid_length: T.int32,
    ):
        with T.Kernel(1, threads=1, is_npu=True):
            tmp = T.alloc_ub((8192,), "float32")
            n = T.min(valid_length, 64)
            T.copy(source[:64], tmp[:64])
            T.copy(tmp[:n], output[:n, 1])

    kernel = tilelang.compile(
        main,
        target="ascendc",
        pass_configs={
            tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
            tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
        },
    )
    source = torch.arange(64, dtype=torch.float32) + 3
    for length in [-1, 0, 1, 63, 64, 80]:
        n = max(0, min(length, 64))
        expected = torch.full((64, 2), -123.0)
        expected[:n, 1] = source[:n]
        output = torch.full_like(expected, -123, device="npu")
        kernel(source.npu(), output, length)
        torch.npu.synchronize()
        torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)


def test_unbounded_large_column_slice_is_rejected():
    @T.prim_func
    def main(output: T.Tensor((8192, 2), "float32"), valid_length: T.int32):
        with T.Kernel(1, threads=1, is_npu=True):
            tmp = T.alloc_ub((8192,), "float32")
            T.tile.fill(tmp, 1.0)
            T.copy(tmp[:valid_length], output[:valid_length, 1])

    with pytest.raises(Exception, match="at most 4095 rows"):
        tilelang.compile(main, target="ascendc")
