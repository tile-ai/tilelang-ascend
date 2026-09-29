import pytest
import torch

import tilelang
import tilelang.language as T
from tilelang.intrinsics import make_zn_layout
from tilelang.language import proxy


PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: False,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: False,
}


def _gm_roundtrip_view_program():
    @T.prim_func
    def main(
        A: T.Tensor((64,), "int32"),
        C: T.Tensor((64,), "int32"),
    ):
        with T.Kernel(1, threads=1, is_npu=True) as _:  # noqa: SIM117
            with T.Scope("V"):
                values = T.alloc_ub((4, 16), "float32")
                T.copy(T.view(A, (4, 16), "float32"), values)
                T.copy(values, T.view(C, (4, 16), "float32"))

    return main


def _ub_view_scalar_add_program():
    @T.prim_func
    def main(
        A: T.Tensor((64,), "int32"),
        C: T.Tensor((4, 16), "float32"),
    ):
        with T.Kernel(1, threads=1, is_npu=True) as _:  # noqa: SIM117
            with T.Scope("V"):
                storage = T.alloc_ub((64,), "int32")
                T.copy(A, storage)
                values = T.view(storage, (4, 16), "float32")
                T.tile.add(values, values, 1.0)
                T.copy(values, C)

    return main


def _ub_view_scalar_load_store_program():
    @T.prim_func
    def main(
        A: T.Tensor((64,), "float32"),
        C: T.Tensor((64,), "int32"),
        D: T.Tensor((64,), "float32"),
    ):
        with T.Kernel(1, threads=1, is_npu=True) as _:  # noqa: SIM117
            with T.Scope("V"):
                storage = T.alloc_ub((64,), "int32")
                values = T.view(storage, dtype="float32")
                for i in T.serial(64):
                    values[i] = A[i]
                for i in T.serial(64):
                    D[i] = values[i]
                T.copy(storage, C)

    return main


def _ub_view_scalar_lifetime_program():
    @T.prim_func
    def main(
        A: T.Tensor((1, 128), "uint8"),
        B: T.Tensor((64,), "int32"),
        C: T.Tensor((32,), "int32"),
        D: T.Tensor((64,), "int32"),
    ):
        with T.Kernel(1, threads=1, is_npu=True) as _:  # noqa: SIM117
            with T.Scope("V"):
                storage = T.alloc_ub((1, 128), "uint8")
                other = T.alloc_ub((64,), "int32")
                T.copy(A, storage)
                T.copy(B, other)
                words = T.view(storage, (32,), "int32")
                for i in T.serial(32):
                    C[i] = words[i]
                T.copy(other, D)

    return main


def _ub_reshape_program():
    @T.prim_func
    def main(
        A: T.Tensor((64,), "int32"),
        C: T.Tensor((4, 16), "int32"),
    ):
        with T.Kernel(1, threads=1, is_npu=True) as _:  # noqa: SIM117
            with T.Scope("V"):
                storage = T.alloc_ub((64,), "int32")
                T.copy(A, storage)
                T.copy(T.reshape(storage, (4, 16)), C)

    return main


def _ub_retyped_row_reduce_program():
    @T.prim_func
    def main(
        A: T.Tensor((4, 64), "int16"),
        C: T.Tensor((4,), "float32"),
    ):
        with T.Kernel(1, threads=1, is_npu=True) as _:  # noqa: SIM117
            with T.Scope("V"):
                storage = T.alloc_ub((4, 64), "int16")
                reduced = T.alloc_ub((4,), "float32")
                T.copy(A, storage)
                values = T.view(storage, (4, 32), "float32")
                T.reduce_sum(values, reduced, dim=-1)
                T.copy(reduced, C)

    return main


def _decl_buffer_alias_program():
    @T.prim_func
    def main(
        A: T.Tensor((64,), "int32"),
        C: T.Tensor((4, 16), "float32"),
    ):
        with T.Kernel(1, threads=1, is_npu=True) as _:  # noqa: SIM117
            with T.Scope("V"):
                storage = T.alloc_ub((64,), "int32")
                T.copy(A, storage)
                with T.decl_buffer(
                    (4, 16),
                    "float32",
                    data=storage.data,
                    elem_offset=0,
                    scope="shared.ub",
                ) as values:
                    T.copy(values, C)

    return main


def _incompatible_ub_view_program():
    @T.prim_func
    def main(
        A: T.Tensor((2, 17), "uint8"),
        C: T.Tensor((1, 17), "int16"),
    ):
        with T.Kernel(1, threads=1, is_npu=True) as _:  # noqa: SIM117
            with T.Scope("V"):
                storage = T.alloc_ub((2, 17), "uint8")
                T.copy(A, storage)
                T.copy(T.view(storage, (1, 17), "int16"), C)

    return main


def _local_int4_view_program():
    @T.prim_func
    def main(
        A: T.Tensor((128,), "int4"),
        C: T.Tensor((64,), "uint8"),
    ):
        with T.Kernel(1, threads=1, is_npu=True) as _:  # noqa: SIM117
            with T.Scope("V"):
                storage = T.alloc_ub((128,), "int4")
                T.copy(A, storage)
                T.copy(T.view(storage, (64,), "uint8"), C)

    return main


def _multiple_global_alias_program():
    @T.prim_func
    def main(
        A: T.Tensor((64,), "int32"),
        C: T.Tensor((64,), "float32"),
        D: T.Tensor((64,), "uint32"),
    ):
        with T.Kernel(1, threads=1, is_npu=True) as _:  # noqa: SIM117
            with T.Scope("V"):
                float_ub = T.alloc_ub((64,), "float32")
                uint_ub = T.alloc_ub((64,), "uint32")
                T.copy(T.view(A, dtype="float32"), float_ub)
                T.copy(T.view(A, dtype="uint32"), uint_ub)
                T.copy(float_ub, C)
                T.copy(uint_ub, D)

    return main


def _fractal_view_mma_program():
    @T.prim_func
    def main(
        A: T.Tensor((16, 16), "float16"),
        B: T.Tensor((16, 16), "float16"),
        C: T.Tensor((16, 16), "int32"),
    ):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            a_l1 = T.alloc_L1((16, 16), "float16")
            b_l1 = T.alloc_L1((16, 16), "float16")
            a_l0 = T.alloc_L0A((16, 32), "int8")
            b_l0 = T.alloc_L0B((16, 16), "float16")
            c_l0 = T.alloc_L0C((16, 16), "int32")
            T.annotate_layout(
                {
                    a_l1: make_zn_layout(a_l1),
                    b_l1: make_zn_layout(b_l1),
                }
            )
            with T.Scope("C"):
                T.copy(A, a_l1)
                T.copy(B, b_l1)
                T.barrier_all()
                T.copy(T.view(a_l1, (16, 32), "int8"), a_l0)
                T.copy(b_l1, b_l0)
                T.barrier_all()
                T.mma(a_l0, T.view(b_l0, (32, 16), "int8"), c_l0, init=True)
                T.barrier_all()
                T.copy(c_l0, C)
                T.barrier_all()

    return main


def _buffer_proxy_program():
    @T.prim_func
    def main(A: T.Buffer((16,), "float16")):
        A[0] = A[0]

    return main


def _lower(program, target):
    with tilelang.tvm.transform.PassContext(opt_level=3, config=PASS_CONFIGS):
        return tilelang.lower(program, target=target).kernel_source


def _compile(program, target, out_idx=None):
    tilelang.disable_cache()
    return tilelang.compile(
        program,
        out_idx=[1] if out_idx is None else out_idx,
        pass_configs=PASS_CONFIGS,
        target=target,
    )


def test_buffer_proxy_is_not_shadowed_by_ascend_imports():
    assert T.Buffer is proxy.Buffer
    program = _buffer_proxy_program()
    assert str(next(iter(program.buffer_map.values())).dtype) == "float16"


def test_view_frontend_preserves_storage_metadata_and_total_bits():
    src = tilelang.tvm.tir.decl_buffer(
        (64,),
        "float32",
        elem_offset=0,
        offset_factor=8,
        scope="shared.ub",
    )
    viewed = T.view(src, (4, 32), dtype="float16")
    reshaped = T.reshape(src, (4, 16))

    assert viewed.data.same_as(src.data)
    assert reshaped.data.same_as(src.data)
    assert viewed.elem_offset.same_as(src.elem_offset)
    assert viewed.offset_factor == src.offset_factor
    assert str(viewed.dtype) == "float16"
    assert [int(dim) for dim in viewed.shape] == [4, 32]

    packed = tilelang.tvm.tir.decl_buffer((128,), "int4", scope="shared.ub")
    assert T.view(packed, (64,), "uint8").data.same_as(packed.data)

    with pytest.raises(ValueError, match="same total bit size"):
        T.view(src, (64,), dtype="float16")
    with pytest.raises(ValueError, match="same total bit size"):
        T.reshape(src, (32,))


def test_view_frontend_rejects_non_whole_storage_buffers():
    tir = tilelang.tvm.tir
    strided = tir.decl_buffer((4, 16), "float32", strides=(32, 1))
    offset = tir.decl_buffer((64,), "float32", elem_offset=1)
    separated = tir.decl_buffer((4, 16), "float32", axis_separators=[1])
    complete = tir.decl_buffer((4, 16), "float32")

    with pytest.raises(ValueError, match="explicit strides"):
        T.view(strided)
    with pytest.raises(ValueError, match="elem_offset"):
        T.view(offset)
    with pytest.raises(ValueError, match="axis separators"):
        T.view(separated)
    with pytest.raises(TypeError, match="complete tir.Buffer"):
        T.view(complete[0:4, 0:16])


def test_view_frontend_checks_symbolic_sizes_and_fractal_scopes():
    tir = tilelang.tvm.tir
    rows = tir.Var("rows", "int32")
    other_rows = tir.Var("other_rows", "int32")
    symbolic = tir.decl_buffer((rows, 32), "float16")

    assert T.view(symbolic, (rows, 16), "float32").data.same_as(symbolic.data)
    with pytest.raises(ValueError, match="same total bit size"):
        T.view(symbolic, (other_rows, 16), "float32")

    l1 = tir.decl_buffer((16, 16), "float16", scope="shared.l1")
    assert T.view(l1).data.same_as(l1.data)
    assert T.view(l1, (16, 32), "int8").data.same_as(l1.data)
    assert T.view(l1, (32, 16), "int8").data.same_as(l1.data)
    with pytest.raises(ValueError, match="fractal-block grid"):
        T.view(l1, (8, 64), "int8")

    l0b = tir.decl_buffer((16, 16), "float16", scope="wmma.matrix_b")
    assert T.view(l0b, (32, 16), "int8").data.same_as(l0b.data)
    with pytest.raises(ValueError, match="fractal-block grid"):
        T.view(l0b, (16, 32), "int8")

    tiled = tir.decl_buffer((2, 3, 16, 16), "float16", scope="shared.l1")
    assert T.view(tiled, (2, 3, 16, 32), "int8").data.same_as(tiled.data)
    with pytest.raises(ValueError, match="leading dimensions"):
        T.view(tiled, (3, 2, 16, 32), "int8")


def test_reinterpretcast_is_removed_from_the_public_language():
    assert not hasattr(T, "reinterpretcast")
    with pytest.raises(tilelang.tvm.error.InternalError, match="is not registered"):
        tilelang.tvm.ir.Op.get("tl.ascend_reinterpretcast")


def test_pto_local_view_rejects_unsupported_physical_layouts():
    with pytest.raises(
        tilelang.tvm.error.InternalError,
        match="PTO local buffer alias must preserve row boundaries",
    ):
        _lower(_incompatible_ub_view_program(), "pto")

    with pytest.raises(
        tilelang.tvm.error.InternalError,
        match="PTO local buffer alias requires byte-addressable storage",
    ):
        _lower(_local_int4_view_program(), "pto")


def test_ascendc_multiple_global_aliases_final_compile():
    source = _lower(_multiple_global_alias_program(), "ascendc")
    assert "AscendC::GlobalTensor<float>" in source
    assert "AscendC::GlobalTensor<uint32_t>" in source

    tilelang.disable_cache()
    tilelang.compile(
        _multiple_global_alias_program(),
        out_idx=[1, 2],
        pass_configs=PASS_CONFIGS,
        target="ascendc",
    )


@pytest.mark.parametrize("target", ["ascendc", "pto"])
def test_global_view_runtime_is_bitwise(target):
    generated = _lower(_gm_roundtrip_view_program(), target)
    assert "tl.ascend_reinterpretcast" not in generated
    if target == "ascendc":
        assert "AscendC::GlobalTensor<float> A_view" in generated
        assert "AscendC::GlobalTensor<float> C_view" in generated
    else:
        assert "reinterpret_cast<__gm__ float *>(A_handle)" in generated
        assert "reinterpret_cast<__gm__ float *>(C_handle)" in generated

    source = (torch.arange(64, dtype=torch.int32) * 2654435761).contiguous()
    actual = _compile(_gm_roundtrip_view_program(), target)(source.npu()).cpu()
    assert torch.equal(actual, source)


@pytest.mark.parametrize("target", ["ascendc", "pto"])
def test_local_access_ptr_view_runtime(target):
    generated = _lower(_ub_view_scalar_add_program(), target)
    if target == "ascendc":
        assert ".ReinterpretCast<float>()" in generated
    else:
        assert "TRESHAPE(" in generated
        assert "TADDS(" in generated

    expected = torch.linspace(-1.0, 1.0, 64, dtype=torch.float32).reshape(4, 16)
    source = expected.contiguous().view(torch.int32)
    actual = _compile(_ub_view_scalar_add_program(), target)(source.npu()).cpu()
    torch.testing.assert_close(actual, expected + 1.0, rtol=0, atol=0)


@pytest.mark.parametrize("target", ["ascendc", "pto"])
def test_local_scalar_view_runtime_is_bitwise(target):
    values = torch.arange(64, dtype=torch.float32)
    tilelang.disable_cache()
    kernel = tilelang.compile(
        _ub_view_scalar_load_store_program(),
        out_idx=[1, 2],
        pass_configs=PASS_CONFIGS,
        target=target,
    )
    words, roundtrip = kernel(values.npu())
    assert torch.equal(words.cpu(), values.view(torch.int32))
    assert torch.equal(roundtrip.cpu(), values)

    source = torch.arange(128, dtype=torch.uint8).reshape(1, 128)
    other = torch.arange(64, dtype=torch.int32)
    tilelang.disable_cache()
    lifetime_kernel = tilelang.compile(
        _ub_view_scalar_lifetime_program(),
        out_idx=[2, 3],
        pass_configs=PASS_CONFIGS,
        target=target,
    )
    viewed, copied = lifetime_kernel(source.npu(), other.npu())
    assert torch.equal(viewed.cpu(), source.view(torch.int32).reshape(32))
    assert torch.equal(copied.cpu(), other)


def test_pto_same_dtype_reshape_runtime():
    generated = _lower(_ub_reshape_program(), "pto")
    assert "copy_ub_to_gm_dynamic<int, int, 1, 1, 1, 4, 16" in generated

    source = torch.arange(64, dtype=torch.int32)
    actual = _compile(_ub_reshape_program(), "pto")(source.npu()).cpu()
    assert torch.equal(actual, source.reshape(4, 16))


def test_pto_retyped_shape_sensitive_reduce_runtime():
    generated = _lower(_ub_retyped_row_reduce_program(), "pto")
    assert "TileUbDataND<float, 4, 32, 4, 32>" in generated
    assert "TROWSUM(" in generated

    values = torch.arange(128, dtype=torch.float32).reshape(4, 32) / 128
    source = values.contiguous().view(torch.int16).reshape(4, 64)
    actual = _compile(_ub_retyped_row_reduce_program(), "pto")(source.npu()).cpu()
    torch.testing.assert_close(actual, values.sum(dim=-1), rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("target", ["ascendc", "pto"])
def test_fractal_view_runtime_is_bitwise(target):
    torch.manual_seed(0)
    a = torch.randn(16, 16, dtype=torch.float16)
    b = torch.randn(16, 16, dtype=torch.float16)
    a_view = a.contiguous().view(torch.int8).reshape(16, 32)
    b_view = b.contiguous().view(torch.int8).reshape(16, 16, 2)
    b_view = b_view.permute(0, 2, 1).reshape(32, 16)
    expected = a_view.to(torch.int32) @ b_view.to(torch.int32)

    actual = _compile(_fractal_view_mma_program(), target, out_idx=[2])(a.npu(), b.npu()).cpu()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("target", ["ascendc", "pto"])
def test_decl_buffer_dtype_alias_runtime_is_bitwise(target):
    source = (torch.arange(64, dtype=torch.int32) * 2654435761).contiguous()
    expected = source.view(torch.float32).reshape(4, 16)
    actual = _compile(_decl_buffer_alias_program(), target)(source.npu()).cpu()
    assert torch.equal(actual.contiguous().view(torch.uint8), expected.view(torch.uint8))
