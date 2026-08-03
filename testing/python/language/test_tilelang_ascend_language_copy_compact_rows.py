"""Regression tests for compact multi-row GM/UB copies on AscendC.

DataCopyPad rounds each local block to 32 bytes.  Representing a compact UB tile
with a non-32-byte-aligned row pitch as one DMA block per logical row therefore
reads or writes a padded layout instead.  These tests isolate GM->UB and UB->GM
so that two symmetric layout errors cannot cancel in a round trip.
"""

import pytest
import tilelang
import tilelang.language as T
import torch
import tvm


PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}


@pytest.fixture(scope="session", autouse=True)
def clear_cache():
    tilelang.cache.clear_cache()
    yield


def _dtype_bytes(dtype):
    return {"int8": 1, "float16": 2, "float": 4}[dtype]


def _torch_dtype(dtype):
    return {"int8": torch.int8, "float16": torch.float16, "float": torch.float32}[dtype]


def _ordered_input(rows, cols, torch_dtype):
    # Keep integer inputs representable so expected values are independent of
    # host-side overflow behaviour.
    values = torch.arange(rows * cols, dtype=torch.int32, device="npu") % 127
    return values.to(torch_dtype).reshape(rows, cols)


def compact_row_copy(rows, cols, dtype):
    dtype_bytes = _dtype_bytes(dtype)
    aligned_cols = ((cols * dtype_bytes + 31) // 32 * 32) // dtype_bytes

    @T.prim_func
    def main(
        A: T.Tensor((rows, cols), dtype),
        B: T.Tensor((rows, aligned_cols), dtype),
        C: T.Tensor((rows, cols), dtype),
    ):
        with T.Kernel(1, is_npu=True) as (_cid, _vid):
            loaded = T.alloc_ub((rows, cols), dtype)
            loaded_out = T.alloc_ub((rows, aligned_cols), dtype)
            produced = T.alloc_ub((rows, cols), dtype)

            # Isolate GM->UB: a scalar consumer reads compact logical offsets,
            # then the aligned output tile takes the safe MTE3 path.
            T.copy(A, loaded)
            for row in T.serial(rows):
                for col in T.serial(aligned_cols):
                    loaded_out[row, col] = T.cast(0, dtype)
            for row in T.serial(rows):
                for col in T.serial(cols):
                    loaded_out[row, col] = loaded[row, col]
            T.copy(loaded_out, B)

            # Isolate UB->GM: scalar stores produce a genuinely compact tile.
            for row in T.serial(rows):
                for col in T.serial(cols):
                    produced[row, col] = T.cast((row * cols + col) % 127, dtype)
            T.copy(produced, C)

    return main


@pytest.mark.parametrize(
    "rows, cols, dtype",
    [
        (256, 4, "float"),  # 16 B: original SVD transpose failure
        (16, 9, "float"),  # 36 B: above 32 B but still unaligned
        (16, 15, "int8"),  # 15 B
        (16, 33, "int8"),  # 33 B
        (16, 15, "float16"),  # 30 B
        (16, 17, "float16"),  # 34 B
        (16, 8, "float"),  # 32 B aligned control
        (16, 16, "float16"),  # 32 B aligned control
    ],
)
def test_compact_row_copy(rows, cols, dtype):
    kernel = tilelang.compile(
        compact_row_copy(rows, cols, dtype),
        out_idx=[],
        pass_configs=PASS_CONFIGS,
        target="ascendc",
    )
    torch_dtype = _torch_dtype(dtype)
    a = _ordered_input(rows, cols, torch_dtype)
    aligned_cols = ((cols * _dtype_bytes(dtype) + 31) // 32 * 32) // _dtype_bytes(dtype)
    b = torch.empty((rows, aligned_cols), dtype=torch_dtype, device="npu")
    c = torch.empty((rows, cols), dtype=torch_dtype, device="npu")

    kernel(a, b, c)
    torch.npu.synchronize()

    expected_c = _ordered_input(rows, cols, torch_dtype)
    torch.testing.assert_close(b[:, :cols], a, rtol=0, atol=0)
    torch.testing.assert_close(b[:, cols:], torch.zeros_like(b[:, cols:]), rtol=0, atol=0)
    torch.testing.assert_close(c, expected_c, rtol=0, atol=0)


def compact_row_m_tail_copy():
    rows, cols, block_rows = 10, 9, 4

    @T.prim_func
    def main(A: T.Tensor((rows, cols), "float"), C: T.Tensor((rows, cols), "float")):
        with T.Kernel(T.ceildiv(rows, block_rows), is_npu=True) as (cid, _vid):
            data = T.alloc_ub((block_rows, cols), "float")
            T.copy(A[cid * block_rows, 0], data)
            for row in T.serial(block_rows):
                for col in T.serial(cols):
                    data[row, col] = data[row, col] + 1
            T.copy(data, C[cid * block_rows, 0])

    return main


def test_compact_row_m_tail_copy():
    kernel = tilelang.compile(
        compact_row_m_tail_copy(),
        out_idx=[-1],
        pass_configs=PASS_CONFIGS,
        target="ascendc",
    )
    a = torch.arange(10 * 9, dtype=torch.float32, device="npu").reshape(10, 9)
    c = kernel(a)
    torch.npu.synchronize()
    torch.testing.assert_close(c, a + 1, rtol=0, atol=0)


def aligned_pitch_slice_copy():
    rows, gm_cols, ub_cols, copy_cols = 8, 32, 16, 8

    @T.prim_func
    def main(
        A: T.Tensor((rows, gm_cols), "float"),
        B: T.Tensor((rows, gm_cols), "float"),
    ):
        with T.Kernel(1, is_npu=True) as (_cid, _vid):
            data = T.alloc_ub((rows, ub_cols), "float")
            T.copy(A[:, :copy_cols], data[:, :copy_cols])
            for row in T.serial(rows):
                for col in T.serial(copy_cols):
                    data[row, col] = data[row, col] + 1
            T.copy(data[:, :copy_cols], B[:, :copy_cols])

    return main


def test_aligned_physical_ub_pitch_for_slices():
    with tvm.transform.PassContext(opt_level=3, config=PASS_CONFIGS):
        artifact = tilelang.lower(aligned_pitch_slice_copy(), target="ascendc")
    assert "copy_gm_to_ub<float, 16, 8>" in artifact.kernel_source
    assert "copy_ub_to_gm<float, 16, 8>" in artifact.kernel_source

    kernel = tilelang.compile(
        aligned_pitch_slice_copy(),
        out_idx=[],
        pass_configs=PASS_CONFIGS,
        target="ascendc",
    )
    a = torch.arange(8 * 32, dtype=torch.float32, device="npu").reshape(8, 32)
    b = torch.zeros_like(a)
    kernel(a, b)
    torch.npu.synchronize()
    torch.testing.assert_close(b[:, :8], a[:, :8] + 1, rtol=0, atol=0)


def unsupported_unaligned_strided_copy():
    @T.prim_func
    def main(A: T.Tensor((8, 8), "float"), B: T.Tensor((8, 4), "float")):
        with T.Kernel(1, is_npu=True) as (_cid, _vid):
            data = T.alloc_ub((8, 4), "float")
            T.copy(A[:, :4], data)
            T.copy(data, B)

    return main


def test_unaligned_compact_rows_with_gm_stride_fail_loudly():
    with pytest.raises(
        tvm.error.InternalError,
        match="non-32-byte-aligned UB row pitch require compact full rows",
    ):
        tilelang.lower(unsupported_unaligned_strided_copy(), target="ascendc")


def compact_row_atomic_add():
    @T.prim_func
    def main(C: T.Tensor((16, 9), "float")):
        with T.Kernel(1, is_npu=True) as (_cid, _vid):
            produced = T.alloc_ub((16, 9), "float")
            for row in T.serial(16):
                for col in T.serial(9):
                    produced[row, col] = T.cast(row * 9 + col, "float")
            T.tile.atomic_add(C, produced)

    return main


def test_compact_row_atomic_add():
    kernel = tilelang.compile(
        compact_row_atomic_add(),
        out_idx=[],
        pass_configs=PASS_CONFIGS,
        target="ascendc",
    )
    c = torch.zeros((16, 9), dtype=torch.float32, device="npu")
    kernel(c)
    torch.npu.synchronize()
    # Both vector cores execute the body, so every value is atomically added
    # twice on 910B.
    expected = 2 * torch.arange(16 * 9, dtype=torch.float32, device="npu").reshape(16, 9)
    torch.testing.assert_close(c, expected, rtol=0, atol=0)


def unsupported_unaligned_strided_atomic_add():
    @T.prim_func
    def main(C: T.Tensor((8, 8), "float")):
        with T.Kernel(1, is_npu=True) as (_cid, _vid):
            produced = T.alloc_ub((8, 4), "float")
            T.tile.fill(produced, 1)
            T.tile.atomic_add(C[:, :4], produced)

    return main


def test_unaligned_atomic_add_with_gm_stride_fails_loudly():
    with pytest.raises(
        tvm.error.InternalError,
        match="non-32-byte-aligned UB row pitch require compact full rows",
    ):
        tilelang.lower(unsupported_unaligned_strided_atomic_add(), target="ascendc")


def test_scalar_to_mte3_sync_is_preserved():
    with tvm.transform.PassContext(opt_level=3, config=PASS_CONFIGS):
        artifact = tilelang.lower(compact_row_copy(16, 9, "float"), target="ascendc")
    assert "SetFlag<AscendC::HardEvent::S_MTE3>" in artifact.kernel_source
    assert "WaitFlag<AscendC::HardEvent::S_MTE3>" in artifact.kernel_source


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-n", "0"])
